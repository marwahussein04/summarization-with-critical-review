"""
app/services/figures_extractor.py

Dedicated "Key Numeric Figures" extraction pass.

This is a SEPARATE pipeline step — not part of general section extraction.
Its ONLY job is to hunt for every concrete number, metric, percentage,
statistic, and measurement in the paper and return them as a structured list.

Why separate?
- General section extraction prompts the LLM to summarise text.
  Numbers get paraphrased or dropped in the process.
- This pass tells the LLM to IGNORE prose and ONLY extract numbers.
- Running it on the full paper text (chunked) with a laser-focused prompt
  yields far more numeric findings than any general extraction.

Token cost: ~400 output tokens per chunk (numbers only, no prose).
"""
from __future__ import annotations

import json
import logging
import re

from groq import Groq

from app.core.config import Settings
from app.domain.models import KeyFigure
from app.services.llm_service import chat_completion
from app.utils.chunker import chunk_text
from app.utils.text import strip_code_fences

logger = logging.getLogger(__name__)

# ── Scoring weights (mirrors chunker.py logic) ───────────────────────────────
_SECTION_PRIORITY = {
    "results": 0, "experiments": 1, "evaluation": 1,
    "abstract": 2, "conclusion": 3,
    "methodology": 4, "introduction": 5,
}

# Patterns that indicate a value is NOT a real finding
# (citation numbers, page numbers, year references, section numbers)
_NOISE_PATTERNS = [
    re.compile(r'^\d{4}$'),                    # bare year: 2023
    re.compile(r'^\[\d+\]$'),                  # citation: [12]
    re.compile(r'^(section|figure|table|eq\.?)\s*\d+$', re.I),  # ref: Figure 3
    re.compile(r'^page\s*\d+$', re.I),         # page number
]


# ════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a scientific data extraction engine. Your ONLY task is to find and \
extract every concrete numeric value from research paper text.

CRITICAL RULES:
1. Return ONLY a JSON array. No prose, no markdown fences, no explanation.
2. Extract EVERY number that represents a scientific finding or experimental result.
3. Copy values EXACTLY as they appear — never round, paraphrase, or estimate.
4. Each item must have: label, value, context, section.
5. Skip: citation numbers like [12], years like 2023, page numbers, section numbers."""


def _build_user_prompt(text: str) -> str:
    return f"""Extract every numeric finding from this research paper text.

TARGET TYPES (extract all you find):
- Performance metrics: accuracy, F1, BLEU, ROUGE, AUC, mAP, perplexity, RMSE, MAE...
- Comparison improvements: "+2.3% over baseline", "3× faster", "50% reduction"...
- Dataset stats: number of samples, classes, train/val/test split sizes, vocabulary size...
- Model specs: parameter count, layer count, hidden dimensions, attention heads, FLOPs...
- Training details: batch size, learning rate, epochs, training time, GPU hours...
- Statistical results: p-values, confidence intervals, standard deviations, effect sizes...
- Thresholds and cutoffs: any numeric threshold used in the method...
- Any other number that supports a scientific claim...

Return a JSON ARRAY (not object) of items. Each item:
{{
  "label": "short metric name (e.g. Top-1 Accuracy, Dataset Size, Learning Rate)",
  "value": "exact value as written (e.g. 94.3%, 1.2M, 3e-4, p < 0.001)",
  "context": "one sentence: what does this number mean in the paper",
  "section": "which section: abstract/introduction/methodology/results/conclusion/other"
}}

Return [] if no numeric findings exist in this chunk.
Return ONLY the JSON array — nothing else.

Paper text:
\"\"\"
{text}
\"\"\"
"""


# ════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION & FILTERING
# ════════════════════════════════════════════════════════════════════════════

def _is_noise(item: dict) -> bool:
    """Return True if this item is a noise value (citation, year, page number, etc.)."""
    value = str(item.get("value", "")).strip()
    label = str(item.get("label", "")).strip().lower()

    for pat in _NOISE_PATTERNS:
        if pat.match(value):
            return True

    # Skip items with empty or trivially short values
    if len(value) < 1:
        return True

    # Skip if label suggests it's a reference
    noise_labels = {"figure", "table", "section", "equation", "page", "reference", "citation"}
    if any(nl in label for nl in noise_labels):
        return True

    return False


def _deduplicate(all_figures: list[dict]) -> list[KeyFigure]:
    """
    Deduplicate by (label, value), keeping the entry from the highest-priority section.
    Then convert to KeyFigure objects.
    """
    # Sort by section priority so the best source wins deduplication
    all_figures.sort(
        key=lambda x: _SECTION_PRIORITY.get(
            str(x.get("section", "")).lower().strip(), 99
        )
    )

    seen: set[tuple[str, str]] = set()
    result: list[KeyFigure] = []

    for item in all_figures:
        if _is_noise(item):
            continue

        key = (
            str(item.get("label", "")).strip().lower(),
            str(item.get("value", "")).strip().lower(),
        )
        if key in seen or key == ("", ""):
            continue
        seen.add(key)

        result.append(KeyFigure(
            label=str(item.get("label", "")).strip(),
            value=str(item.get("value", "")).strip(),
            context=str(item.get("context", "")).strip(),
            section=str(item.get("section", "")).strip(),
        ))

    # Final sort: results first, then abstract, then others
    result.sort(key=lambda kf: _SECTION_PRIORITY.get(kf.section.lower(), 99))
    return result


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def extract_key_figures(
    full_text: str,
    client: Groq,
    settings: Settings,
) -> list[KeyFigure]:
    """
    Dedicated numeric figures extraction pass over the full paper text.

    Runs independently of section extraction. Chunks the full text and
    extracts numeric findings from each chunk, then deduplicates and
    returns a clean, prioritised list.

    Args:
        full_text: Complete paper text (from marker or raw PDF).
        client:    Groq client.
        settings:  App settings.

    Returns:
        Deduplicated list of KeyFigure objects, sorted by section priority.
    """
    chunks = chunk_text(full_text)
    logger.info(
        "Key figures extraction: %d chunks, %d total chars.",
        len(chunks), len(full_text),
    )

    raw_items: list[dict] = []

    for chunk in chunks:
        user_prompt = _build_user_prompt(chunk.text)
        try:
            raw = chat_completion(
                client=client,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                settings=settings,
                max_tokens=600,   # numbers only — short output, cheap on tokens
            )
            cleaned = strip_code_fences(raw).strip()

            # Handle both array and object responses defensively
            if cleaned.startswith("{"):
                # LLM returned an object — try to extract array from it
                obj = json.loads(cleaned)
                arr = next(
                    (v for v in obj.values() if isinstance(v, list)), []
                )
            else:
                arr = json.loads(cleaned)

            if isinstance(arr, list):
                valid = [
                    item for item in arr
                    if isinstance(item, dict)
                    and item.get("label")
                    and item.get("value")
                ]
                raw_items.extend(valid)
                logger.debug(
                    "  Chunk %d/%d: %d figures found.",
                    chunk.index + 1, len(chunks), len(valid),
                )
            else:
                logger.warning("  Chunk %d: unexpected response type.", chunk.index + 1)

        except json.JSONDecodeError as exc:
            logger.warning("  Chunk %d JSON parse failed: %s", chunk.index + 1, exc)
        except Exception as exc:
            logger.warning("  Chunk %d extraction failed: %s", chunk.index + 1, exc)

    figures = _deduplicate(raw_items)
    logger.info(
        "Key figures extraction complete: %d unique figures from %d raw items.",
        len(figures), len(raw_items),
    )
    return figures
