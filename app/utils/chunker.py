"""
app/utils/chunker.py

Chunking strategy for feeding long papers to the LLM without truncation.

Improvements over original:
- merge_section_dicts now merges key_figures lists across all chunks (deduplicated)
- Text field selection uses a smarter scoring function (prefers content with numbers
  over generic verbose summaries)
- Deduplication of key_figures by (label, value) pair
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DIRECT_THRESHOLD = 8_000    # chars — send directly if shorter than this
CHUNK_SIZE       = 8_000    # chars per chunk
OVERLAP          = 500      # chars overlap between consecutive chunks
MAX_CHUNKS       = 6       # hard cap — never send more than 12 LLM calls


@dataclass
class Chunk:
    index: int
    text:  str
    start: int
    end:   int


def chunk_text(text: str) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    If text fits in one chunk, returns a single-element list.
    Tries to split on paragraph boundaries to avoid cutting mid-sentence.
    """
    if len(text) <= DIRECT_THRESHOLD:
        return [Chunk(index=0, text=text, start=0, end=len(text))]

    chunks: list[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text) and idx < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            para_break = text.rfind("\n\n", start + CHUNK_SIZE // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                sent_break = text.rfind(". ", start + CHUNK_SIZE // 2, end)
                if sent_break != -1:
                    end = sent_break + 2

        chunks.append(Chunk(index=idx, text=text[start:end], start=start, end=end))
        start = max(end - OVERLAP, end - CHUNK_SIZE // 4)
        idx  += 1

    logger.info(
        "Chunked %d chars into %d chunks (max %d chars each).",
        len(text), len(chunks), CHUNK_SIZE,
    )
    return chunks


def is_single_chunk(text: str) -> bool:
    return len(text) <= DIRECT_THRESHOLD


# ── Section merging ──────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r'\d')


def _score_field(text: str) -> float:
    """
    Score a candidate text field value.
    Prefers values that:
    - Are longer (more detail)
    - Contain numbers (more data-rich)
    - Don't start with generic phrases
    """
    if not text or text.strip().lower() in ("not found", ""):
        return 0.0
    score = len(text) * 0.5
    # Bonus for every numeric character — numbers indicate concrete data
    score += len(_NUMBER_RE.findall(text)) * 8
    # Bonus for percentage signs, units
    score += text.count('%') * 15
    score += text.count('±') * 10
    # Penalty for obviously generic openers
    generic_openers = ("the paper", "this paper", "the study", "the research",
                       "the authors", "in this", "this work")
    lower = text.strip().lower()
    for g in generic_openers:
        if lower.startswith(g):
            score -= 20
            break
    return score


def _merge_key_figures(dicts: list[dict]) -> list[dict]:
    """
    Collect all key_figures from all chunks, deduplicate by (label, value).
    Returns a unified list sorted by section priority (results > abstract > others).
    """
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    section_priority = {"results": 0, "abstract": 1, "conclusion": 2,
                        "methodology": 3, "introduction": 4}

    all_figures: list[dict] = []
    for d in dicts:
        raw = d.get("key_figures", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("label") and item.get("value"):
                    all_figures.append(item)

    # Sort by section priority so high-quality figures win deduplication
    all_figures.sort(
        key=lambda x: section_priority.get(str(x.get("section", "")).lower(), 99)
    )

    for item in all_figures:
        key = (str(item.get("label", "")).strip().lower(),
               str(item.get("value", "")).strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(item)

    return merged


def merge_section_dicts(dicts: list[dict]) -> dict:
    """
    Merge multiple JSON section extractions into one coherent result.

    For each text field: pick the value with the highest score (not just longest).
    For key_figures: collect all unique figures from all chunks.
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]

    TEXT_FIELDS = [
        "title", "authors", "abstract", "introduction",
        "methodology", "results", "conclusion", "limitations", "future_work",
    ]

    merged: dict = {}
    for f in TEXT_FIELDS:
        candidates = [
            d.get(f, "")
            for d in dicts
            if isinstance(d.get(f), str)
            and d.get(f, "").strip()
            and d.get(f, "").strip().lower() != "not found"
        ]
        if not candidates:
            merged[f] = "Not found"
        else:
            # Pick the highest-scoring (most data-rich) value
            merged[f] = max(candidates, key=_score_field)

    # Merge key_figures from all chunks
    merged["key_figures"] = _merge_key_figures(dicts)

    return merged
