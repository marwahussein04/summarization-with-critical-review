"""
app/services/critical_review_service.py
Conservative critical-review profiling and pairwise comparison workflow.

Changes in this version
────────────────────────
1. resolve_stable_metadata — smarter title selection
   - Scores all candidate titles with _title_score() and picks the best one.
   - Strings like "Pushing the Chatbot State-of-the-art with QLoRA" score
     negatively (prose/blog-post language) while "QLORA: Efficient Finetuning
     of Quantized LLMs" scores positively (colon + acronym).
   - _sanitize_title rejects strings whose _title_score < -3.

2. _normalize_assessment — calibrated conservatism
   - weaknesses / threats_to_validity: extended implicit-token scan to also
     include abstract + results (not only intro + methodology).
   - threats_to_validity: LLM verdict with evidence is kept even when
     implicit tokens are absent, so real findings aren't silently dropped.
   - assumptions: added more trigger tokens (pretrain, rank, low-rank,
     frozen, quantiz, intrinsic, subspace) so QLoRA / LoRA evidence registers.

3. Reproducibility confidence — corrected over-optimism
   - "Reasonably reproducible" now requires ≥ 3 signals AND at least one of
     (hyperparameters, data).
   - Confidence "High" requires release signal OR ≥ 5 signals; otherwise
     capped at Medium/Low.

4. Fairness confidence — corrected over-optimism
   - "Comparison setup appears reasonably supported" now requires BOTH
     baselines AND metrics signals present.
   - Confidence "High" requires controls signal explicitly present.

5. compare_paper_profiles — retry logic with up to MAX_PAIRWISE_RETRIES.

6. _normalize_pairwise — relaxed guard (only falls back when evidence is
   empty AND banned phrases are present, not on either condition alone).

7. _repair_truncated_json — recovers from mid-stream truncation.
"""
from __future__ import annotations

import json
import logging
import re
import time

from groq import Groq
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import CriticalReviewError
from app.domain.models import (
    NOT_FOUND,
    CriticalComparisonResult,
    PairwiseDimensionComparison,
    PaperCriticalProfile,
    PaperSections,
    ReviewDimensionAssessment,
)
from app.prompts.critical_review_generation import (
    CRITICAL_REVIEW_DIMENSIONS,
    build_critical_profile_system_prompt,
    build_critical_profile_user_prompt,
    build_pairwise_comparison_system_prompt,
    build_pairwise_comparison_user_prompt,
)
from app.services.llm_service import chat_completion
from app.utils.text import strip_code_fences

logger = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────
MAX_PAIRWISE_RETRIES = 3
PAIRWISE_RETRY_DELAY = 2.0
MAX_PAIRWISE_TOKENS  = 3800   # compressed profiles ~900 tokens → plenty of room for 8 dims
MAX_PROFILE_TOKENS   = 3200

# ── Dimension metadata ────────────────────────────────────────────────────────
DIMENSION_LABELS = {
    "strengths":              "Strengths",
    "weaknesses":             "Weaknesses",
    "novelty":                "Novelty",
    "assumptions":            "Assumptions",
    "threats_to_validity":    "Threats to Validity",
    "reproducibility":        "Reproducibility",
    "fairness_of_comparison": "Fairness of Comparison",
    "applicability":          "Applicability",
}

NOT_ENOUGH_EVIDENCE = "Not enough evidence"
GENERIC_PHRASES = (
    "may not be suitable for all tasks",
    "may not capture complex patterns",
    "high-quality methodology",
    "greater impact on the field",
    "fair evaluation metrics used",
    "thorough analysis",
)
ASSERTIVE_PHRASES = (
    "more innovative",
    "greater impact",
    "superior",
    "best overall",
    "wins on",
    "clearly stronger",
)
PAIRWISE_FALLBACK_JUDGEMENT = (
    "Insufficient structured pairwise evidence to rank overall; showing per-paper judgements instead."
)

# Controlled verdict vocabulary per dimension — used for post-processing enforcement
# If the LLM verdict doesn't match any of these, it gets remapped to the best fit or NEE.
VERDICT_VOCAB: dict[str, tuple[str, ...]] = {
    "strengths": (
        "Multiple concrete strengths supported by evidence",
        "Some strengths are supported, others are partial",
        "Limited concrete strengths identifiable from the text",
        NOT_ENOUGH_EVIDENCE,
    ),
    "weaknesses": (
        "Explicit limitations acknowledged by the authors",
        "Weaknesses identifiable from omissions or partial evidence",
        "No significant weaknesses are explicit in the text",
        NOT_ENOUGH_EVIDENCE,
    ),
    "novelty": (
        "Foundational contribution: introduces a new paradigm or primitive",
        "Efficiency innovation built on prior named methods",
        "Incremental extension of prior work",
        "Practical or engineering innovation",
        "Mixed novelty: combines foundational and efficiency elements",
        NOT_ENOUGH_EVIDENCE,
    ),
    "assumptions": (
        "Explicit assumptions stated in the paper",
        "Implicit assumptions identifiable from the methodology",
        "No clear assumptions identifiable from the text",
        NOT_ENOUGH_EVIDENCE,
    ),
    "threats_to_validity": (
        "Multiple threats to validity are explicitly stated",
        "Limited threats acknowledged; evaluation scope may be narrow",
        "No threats to validity explicitly discussed",
        NOT_ENOUGH_EVIDENCE,
    ),
    "reproducibility": (
        "Reasonably reproducible: ≥3 signals present",
        "Partially reproducible: some details present but incomplete",
        "Reproducibility is limited: key details are absent",
        NOT_ENOUGH_EVIDENCE,
    ),
    "fairness_of_comparison": (
        "Comparison setup appears fair: named baselines and matched metrics present",
        "Partially fair: baselines named but conditions not fully matched",
        "Fairness cannot be confirmed: baselines or metrics insufficiently specified",
        NOT_ENOUGH_EVIDENCE,
    ),
    "applicability": (
        "Broad applicability suggested by evidence",
        "Applicability is domain- or resource-constrained",
        "Applicability is unclear from the text",
        NOT_ENOUGH_EVIDENCE,
    ),
}

PAIRWISE_VOCAB: dict[str, tuple[str, ...]] = {
    "strengths": (
        "Paper A appears stronger on strengths from the available evidence",
        "Paper B appears stronger on strengths from the available evidence",
        "Both papers show comparable strengths; no clear ordering is supported",
        "Insufficient evidence to compare strengths",
    ),
    "weaknesses": (
        "Paper A has more explicitly acknowledged weaknesses",
        "Paper B has more explicitly acknowledged weaknesses",
        "Both papers show different weakness profiles; no ranking is appropriate",
        "Insufficient evidence to compare weaknesses",
    ),
    "novelty": (
        "Paper A appears more foundational; Paper B appears to extend or build on prior work",
        "Paper B appears more foundational; Paper A appears to extend or build on prior work",
        "Both papers appear to offer efficiency or incremental innovations of comparable scope",
        "Insufficient evidence to rank novelty from the paper text alone",
    ),
    "assumptions": (
        "Paper A carries more explicit or restrictive assumptions",
        "Paper B carries more explicit or restrictive assumptions",
        "Both papers have comparable assumption profiles",
        "Insufficient evidence to compare assumptions",
    ),
    "threats_to_validity": (
        "Paper A has more explicit threats to validity or evaluation gaps",
        "Paper B has more explicit threats to validity or evaluation gaps",
        "Both papers have similar threat profiles",
        "Insufficient evidence to compare threats to validity",
    ),
    "reproducibility": (
        "Paper A appears more reproducible based on available signals",
        "Paper B appears more reproducible based on available signals",
        "Both papers have comparable reproducibility levels",
        "Insufficient evidence to compare reproducibility",
    ),
    "fairness_of_comparison": (
        "Paper A's comparison setup appears better supported",
        "Paper B's comparison setup appears better supported",
        "Both papers have comparable comparison setups",
        "Insufficient evidence to compare fairness of comparison",
    ),
    "applicability": (
        "Paper A appears more broadly applicable based on the evidence",
        "Paper B appears more broadly applicable based on the evidence",
        "Both papers have comparable applicability scope",
        "Insufficient evidence to compare applicability",
    ),
}
EVIDENCE_KEYWORDS = (
    "dataset", "benchmark", "baseline", "ablation", "metric", "accuracy", "f1",
    "bleu", "rouge", "auc", "map", "latency", "memory", "compute", "parameter",
    "batch size", "learning rate", "epoch", "optimizer", "split", "train",
    "validation", "test", "code", "checkpoint", "release", "quantization",
    "gpu", "task", "deployment", "compare", "evaluation", "limitation",
)
REPRO_SIGNALS = {
    "hyperparameters": ("learning rate", "batch size", "epoch", "optimizer", "seed"),
    "setup":           ("architecture", "layer", "parameter", "implementation", "model", "quantization"),
    "data":            ("dataset", "benchmark", "split", "train", "validation", "test"),
    "protocol":        ("metric", "evaluation", "ablation", "baseline", "protocol"),
    "release":         ("code", "checkpoint", "github", "repository", "release"),
}
FAIRNESS_SIGNALS = {
    "baselines":          ("baseline", "compared", "comparison"),
    "tasks_or_datasets":  ("dataset", "benchmark", "task", "split"),
    "metrics":            ("metric", "accuracy", "f1", "bleu", "rouge", "auc", "map"),
    "controls":           ("same setting", "same budget", "matched", "controlled", "protocol", "tuned"),
    "caveats":            ("limitation", "caveat", "unclear", "not directly comparable", "different setting"),
}
NOVELTY_TYPES = {
    "foundational": ("foundational", "introduces", "new formulation", "new framework", "new paradigm", "first"),
    "extension":    ("extends", "extension", "builds on", "variant", "adaptation", "incremental"),
    "efficiency":   ("efficient", "efficiency", "parameter-efficient", "memory", "compute", "quantization", "faster"),
    "practical":    ("practical", "engineering", "deployment", "scalable", "real-world"),
}

# Prose / section-heading patterns that make a title candidate look like a
# blog post or chapter heading rather than a paper title.
# Note: only reject clear prose patterns, not academic phrasing.
_TITLE_PROSE_REJECTS = re.compile(
    r"\b(pushing the|state[- ]of[- ]the[- ]art with\s+\w+|chatbot state|"
    r"our approach in|in this paper we|recent advances in a survey|an overview of|"
    r"^(pushing|achieving|improving|exploring|leveraging|harnessing)\b)\b",
    re.I,
)
_TITLE_REJECTION_PATTERNS = (
    re.compile(r"^(table|figure|fig\.?|appendix)\s*\d+[:.\s-]", re.I),
    re.compile(r"^(results|ablation|discussion|conclusion|references)\b", re.I),
)
_AUTHORS_REJECT_PATTERNS = re.compile(
    r"\b(permission|license|grant|copyright|rights reserved|attribution|"
    r"hereby|reproduce|commercial|redistribution|provided that|licen[cs]e|"
    r"all rights|published by|proceedings of|conference on)\b",
    re.I,
)
_ARCH_NOISE = re.compile(
    r"\b(composed of|stack of|layer|sub-layer|identical layer|"
    r"encoder|decoder|dot-product|scaling factor|embedding|dimension(?:ality)*)\b",
    re.I,
)
_FOREIGN_MODEL_NOISE = re.compile(
    r"\b(gpt-?\d|llama|mistral|palm|t5|xlnet|roberta|albert|electra|gptq|lora)\b",
    re.I,
)
_NEGATIVE_VERDICT_DIMS: dict[str, tuple[str, ...]] = {
    "threats_to_validity": ("No threats", "Not enough evidence"),
    "weaknesses": ("No significant weaknesses", "Not enough evidence"),
    "fairness_of_comparison": ("Fairness cannot be confirmed", "Not enough evidence"),
}

DIMENSION_EVIDENCE_SOURCES = {
    "strengths":              ("methodology", "results", "conclusion", "key_figures"),
    "weaknesses":             ("limitations", "future_work", "results"),
    "novelty":                ("introduction", "methodology", "conclusion", "key_figures"),
    "assumptions":            ("methodology", "limitations", "introduction"),
    "threats_to_validity":    ("limitations", "results", "future_work"),
    "reproducibility":        ("methodology", "results", "key_figures"),
    "fairness_of_comparison": ("results", "methodology", "key_figures"),
    "applicability":          ("methodology", "results", "conclusion", "future_work", "key_figures"),
}

DIMENSION_KEYWORDS = {
    "strengths":              ("improve", "outperform", "achieves", "reduce", "efficient", "robust", "benchmark", "dataset", "baseline"),
    "weaknesses":             ("limitation", "limited", "only", "missing", "unclear", "future work", "not report"),
    "novelty":                ("novel", "new", "introduces", "first", "efficient", "parameter-efficient", "quantization", "extension"),
    "assumptions":            ("assume", "requires", "depends", "availability", "fixed", "pretrained", "rank", "low-rank", "frozen", "quantiz"),
    "threats_to_validity":    ("bias", "limited", "only", "ablation", "generalization", "scope", "benchmark"),
    "reproducibility":        ("learning rate", "batch", "epoch", "optimizer", "dataset", "split", "metric", "code", "checkpoint"),
    "fairness_of_comparison": ("baseline", "compare", "comparison", "benchmark", "metric", "same", "matched"),
    "applicability":          ("deployment", "efficient", "latency", "memory", "compute", "practical", "real-world", "scalable"),
}


# ── String helpers ────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_first_line(value: object, field_name: str = "") -> str:
    raw = str(value or "")
    lines = raw.splitlines()
    return _norm(lines[0]) if lines else ""


def _clean_bullet(text: str) -> str:
    return re.sub(r"^[\-\*\u2022]+\s*", "", _norm(text)).strip(" .")


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lower = _norm(text).lower()
    return any(phrase in lower for phrase in phrases)


def _is_raw_number_dump(text: str) -> bool:
    tokens = [t.strip() for t in text.split(",")]
    numeric = [t for t in tokens if re.match(r'^\d[\d.±]*$', t)]
    return len(tokens) > 4 and len(numeric) / len(tokens) > 0.5


def _clean_evidence_items(evidence: list[str], limit: int = 5) -> list[str]:
    seen: set[str] = set()
    seen_nums: set[str] = set()
    items: list[str] = []
    for item in evidence:
        cleaned = _clean_bullet(item)
        if "|" in cleaned:
            segments = [s.strip() for s in cleaned.split("|")]
            first = segments[0].strip()
            cleaned = first if len(first) > 15 else " — ".join(s for s in segments[:2] if s)
        lowered = cleaned.lower()
        if not cleaned or len(cleaned) < 10 or lowered in seen or _contains_phrase(cleaned, GENERIC_PHRASES):
            continue
        if len(cleaned) > 180:
            cleaned = cleaned[:177].rsplit(" ", 1)[0] + "…"
            lowered = cleaned.lower()
        if _is_raw_number_dump(cleaned):
            continue
        for kw in ("elo", "latency", "perplexity"):
            if kw in lowered:
                m = re.search(r'\b(\d[\d,.]*)\b', lowered)
                fp = m.group(1) if m else ""
                if fp and fp in seen_nums:
                    break
                if fp:
                    seen_nums.add(fp)
        else:
            seen.add(lowered)
            items.append(cleaned)
            continue
        # inner break hit — skip item
    specific = [i for i in items if _is_specific_evidence(i)]
    return (specific or items)[:limit]


def _is_specific_evidence(item: str) -> bool:
    cleaned = _clean_bullet(item)
    lower = cleaned.lower()
    if not cleaned or _contains_phrase(cleaned, GENERIC_PHRASES):
        return False
    if re.search(r"\d", cleaned):
        return True
    if any(k in lower for k in EVIDENCE_KEYWORDS):
        return True
    return any(p in lower for p in (
        "not report", "not mention", "release not mentioned",
        "does not report", "does not mention", "only compares",
        "only evaluates", "limited to", "no code",
    ))


def _specific_count(evidence: list[str]) -> int:
    return sum(1 for i in evidence if _is_specific_evidence(i))


def _collect_signal_groups(text: str, signals: dict[str, tuple[str, ...]]) -> set[str]:
    """
    Collect signal groups from text — negation-aware.
    Sentences containing "does not", "not mention", "not provide", "not release",
    "no code", "no checkpoint", "omit" are excluded from positive signal detection.
    """
    lower = text.lower()
    found: set[str] = set()

    # Split into sentences for negation-scoped detection
    sentences = re.split(r"(?<=[.!?])\s+|\n", lower)

    # Build per-sentence negation flag
    _NEGATION = re.compile(
        r"\b(does not|do not|not mention|not provide|not release|not report|"
        r"not specify|not explicit|no code|no checkpoint|omit|without|absent|missing|"
        r"lack|not available|not included|unreported)\b"
    )

    for lbl, kws in signals.items():
        for sentence in sentences:
            is_negated = bool(_NEGATION.search(sentence))
            if any(k in sentence for k in kws):
                if not is_negated:
                    found.add(lbl)
                    break
    return found


def _split_sentences(text: str) -> list[str]:
    if not text or text == NOT_FOUND:
        return []
    sentences: list[str] = []
    for block in text.replace("\r", "\n").split("\n"):
        block = _norm(block)
        if block:
            sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip())
    return sentences


def _score_candidate(sentence: str, keywords: tuple[str, ...], dimension: str = "") -> int:
    lower = sentence.lower()
    score  = len(re.findall(r"\d", sentence)) * 4
    score += sum(3 for k in keywords if k in lower)
    score += 2 if "%" in sentence else 0
    score += 2 if any(m in lower for m in ("not ", "only ", "missing", "limited", "unclear")) else 0
    score += 1 if len(sentence) > 40 else 0
    if dimension == "threats_to_validity":
        threat_tokens = ("limitation", "limited", "only", "bias", "generalization", "scope", "not evaluat", "narrow", "caveat", "concern")
        if not any(t in lower for t in threat_tokens):
            score -= 5
        if _ARCH_NOISE.search(sentence):
            score -= 6
    return score


def _fallback_dimension_evidence(dimension: str, sections: PaperSections, limit: int = 4) -> list[str]:
    keywords = DIMENSION_KEYWORDS.get(dimension, ())
    candidates: list[str] = []
    title_tokens = _paper_identity_tokens(sections)
    _KF_SCOPE = {
        "strengths": {"results", "abstract"},
        "weaknesses": {"limitations"},
        "novelty": {"abstract", "introduction"},
        "assumptions": {"methodology"},
        "threats_to_validity": {"limitations", "results"},
        "reproducibility": {"methodology", "results"},
        "fairness_of_comparison": {"results"},
        "applicability": {"results", "conclusion"},
    }
    for source in DIMENSION_EVIDENCE_SOURCES.get(dimension, ()):  # noqa: PLC0206
        if source == "key_figures":
            allowed = _KF_SCOPE.get(dimension, set())
            for fig in sections.key_figures[:15]:
                fig_section = _norm(fig.section).lower()
                if allowed and not any(scope_word in fig_section for scope_word in allowed):
                    continue
                text = " | ".join(p for p in (f"{fig.label}: {fig.value}", fig.context, fig.section) if p)
                lower = text.lower()
                if not text:
                    continue
                if dimension == "threats_to_validity" and _ARCH_NOISE.search(text):
                    continue
                if _FOREIGN_MODEL_NOISE.search(lower) and title_tokens and not any(tok in lower for tok in title_tokens):
                    continue
                if keywords and dimension in {"novelty", "reproducibility", "fairness_of_comparison", "applicability"}:
                    if not any(kw in lower for kw in keywords):
                        continue
                candidates.append(text)
            continue
        candidates.extend(_split_sentences(getattr(sections, source, NOT_FOUND)))
    ranked = sorted(candidates, key=lambda s: _score_candidate(s, keywords, dimension), reverse=True)
    if dimension == "threats_to_validity":
        ranked = [s for s in ranked if not _ARCH_NOISE.search(s)]
    return _clean_evidence_items(ranked, limit=limit)


def _source_text(sections: PaperSections) -> str:
    parts = [
        sections.title, sections.authors, sections.abstract,
        sections.introduction, sections.methodology, sections.results,
        sections.conclusion, sections.limitations, sections.future_work,
    ]
    parts.extend(f"{f.label} {f.value} {f.context} {f.section}" for f in sections.key_figures)
    return _norm(" ".join(p for p in parts if p and p != NOT_FOUND))


def _paper_identity_tokens(sections: PaperSections) -> set[str]:
    title = _norm(sections.title)
    raw_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", title)
    stop = {"the", "and", "for", "with", "from", "into", "using", "novel", "architecture", "model", "language", "understanding", "sequence", "paper"}
    acronyms = {tok.lower() for tok in raw_tokens if tok.isupper() and len(tok) >= 3}
    long_words = {tok.lower() for tok in raw_tokens if len(tok) >= 5 and tok.lower() not in stop}
    return acronyms | set(sorted(long_words)[:6])


def _check_verdict_evidence_consistency(
    dimension: str,
    verdict: str,
    evidence: list[str],
) -> list[str]:
    targets = _NEGATIVE_VERDICT_DIMS.get(dimension, ())
    if not targets or not any(verdict.startswith(t) for t in targets):
        return evidence
    positive_noise = re.compile(
        r"\b(composed of|is a stack|consists of|performs|achieves|introduces|"
        r"uses|employs|applies|demonstrates|shows|reports)\b",
        re.I,
    )
    filtered = [e for e in evidence if not positive_noise.search(e)]
    return filtered if filtered else evidence


def _remove_contradicted_evidence(evidence: list[str]) -> list[str]:
    dataset_tokens = (
        "glue", "squad", "mnli", "qqp", "sst", "wikitext", "wmt",
        "iwslt", "bleu", "rouge", "accuracy", "f1", "auc", "map",
    )
    negation_items: list[int] = []
    metric_items: list[str] = []
    for idx, item in enumerate(evidence):
        lower = item.lower()
        if any(p in lower for p in ("not evaluated", "not tested", "has not been evaluated", "has not been tested")):
            negation_items.append(idx)
        if re.search(r"\b\d[\d.]*\s*(%|f1|accuracy|bleu|rouge|auc|map)\b", lower):
            metric_items.append(lower)
        elif any(tok in lower for tok in dataset_tokens) and re.search(r"\d", lower):
            metric_items.append(lower)
    if not negation_items or not metric_items:
        return evidence
    drop: set[int] = set()
    for idx in negation_items:
        lower = evidence[idx].lower()
        if any(tok in lower and any(tok in metric for metric in metric_items) for tok in dataset_tokens):
            drop.add(idx)
            continue
        if "text classification" in lower and any(tok in metric for metric in metric_items for tok in ("glue", "mnli", "qqp", "sst")):
            drop.add(idx)
    return [item for j, item in enumerate(evidence) if j not in drop]


def _reproducibility_rationale(found: set[str], evidence: list[str]) -> str:
    all_groups = set(REPRO_SIGNALS.keys())
    present = sorted(found)
    absent = sorted(all_groups - found)
    parts: list[str] = []
    if present:
        parts.append(f"Reproducibility signals present: {', '.join(present)}.")
    if absent:
        parts.append(f"Missing signals: {', '.join(absent)}.")
    if not any(
        any(tok in e.lower() for tok in ("code", "checkpoint", "github", "repository"))
        and not any(neg in e.lower() for neg in ("not", "no ", "without", "missing", "absent"))
        for e in evidence
    ):
        parts.append("No code or checkpoint release is mentioned in the extracted text.")
    return " ".join(parts) if parts else "Partial reproducibility evidence found in the extracted text."


def _cap_confidence(confidence: str, evidence: list[str], specific: int) -> str:
    capped = _normalize_confidence(confidence)
    joined = " ".join(evidence)
    if capped == "High" and (_contains_phrase(joined, GENERIC_PHRASES) or specific < 2):
        capped = "Medium"
    if capped == "Medium" and specific < 1:
        capped = "Low"
    return capped


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _normalize_confidence(value: str) -> str:
    v = _norm(value).lower()
    if v == "high":   return "High"
    if v == "medium": return "Medium"
    return "Low"


def _enforce_verdict(dimension: str, raw_verdict: str) -> str:
    """
    Map a free-form LLM verdict to the nearest controlled vocabulary string.
    Strategy:
      1. If exact match (case-insensitive) → use it.
      2. If partial keyword match → pick best candidate.
      3. Otherwise → NOT_ENOUGH_EVIDENCE.
    """
    vocab = VERDICT_VOCAB.get(dimension)
    if not vocab:
        return _norm(raw_verdict) or NOT_ENOUGH_EVIDENCE

    v_lower = _norm(raw_verdict).lower()

    # Exact match (normalized)
    for candidate in vocab:
        if candidate.lower() == v_lower:
            return candidate

    # Keyword overlap scoring
    best_candidate = NOT_ENOUGH_EVIDENCE
    best_score = 0
    for candidate in vocab:
        if candidate == NOT_ENOUGH_EVIDENCE:
            continue
        # Count shared meaningful words (3+ chars)
        c_words = set(re.findall(r'\b\w{3,}\b', candidate.lower()))
        v_words = set(re.findall(r'\b\w{3,}\b', v_lower))
        overlap = len(c_words & v_words)
        if overlap > best_score:
            best_score = overlap
            best_candidate = candidate

    # Require at least 2 matching words to use a candidate
    return best_candidate if best_score >= 2 else NOT_ENOUGH_EVIDENCE


def _enforce_pairwise_verdict(dimension: str, raw_judgement: str) -> str:
    """Map a free-form pairwise comparative judgement to the controlled vocabulary."""
    vocab = PAIRWISE_VOCAB.get(dimension)
    if not vocab:
        return raw_judgement or PAIRWISE_FALLBACK_JUDGEMENT

    v_lower = _norm(raw_judgement).lower()

    # Exact match
    for candidate in vocab:
        if candidate.lower() == v_lower:
            return candidate

    # Detect direction: which paper is mentioned positively?
    a_positive = any(p in v_lower for p in ("paper a appears stronger", "paper a is stronger",
                                              "paper a appears more", "paper a has better",
                                              "paper a seems stronger"))
    b_positive = any(p in v_lower for p in ("paper b appears stronger", "paper b is stronger",
                                              "paper b appears more", "paper b has better",
                                              "paper b seems stronger"))
    insufficient = any(p in v_lower for p in ("insufficient", "not enough", "cannot", "unclear",
                                               "no clear", "comparable", "similar"))

    # Pick best match based on direction
    options = list(vocab)
    a_opts = [o for o in options if "Paper A" in o and "Paper B" not in o.split("Paper B")[0] if "Paper B" not in o]
    b_opts = [o for o in options if "Paper B appears" in o or "Paper B has" in o or "Paper B's" in o]
    both_opts = [o for o in options if "Both" in o]
    insuf_opts = [o for o in options if "Insufficient" in o]

    if a_positive and not b_positive and a_opts:
        return a_opts[0]
    if b_positive and not a_positive and b_opts:
        return b_opts[0]
    if insufficient and insuf_opts:
        return insuf_opts[0]
    if both_opts:
        return both_opts[0]
    return insuf_opts[0] if insuf_opts else PAIRWISE_FALLBACK_JUDGEMENT


def _title_score(title: str) -> int:
    """
    Heuristic quality score for a paper-title candidate.
    Positive = looks like a real academic title.
    Negative = looks like a section heading or blog-post phrase.
    """
    if not title or title == NOT_FOUND:
        return -99
    score = 0
    if ":" in title:                                         score += 5   # "LoRA: Low-Rank..."
    if re.search(r'\b[A-Z]{2,}\b', title):                  score += 4   # acronym like BERT, QLoRA
    if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', title):     score += 3   # mixed proper-noun
    if _TITLE_PROSE_REJECTS.search(title):                   score -= 8   # prose / blog-post
    if len(title) < 15:                                      score -= 3
    if len(title) > 250:                                     score -= 2
    # Heavily penalize section headings masquerading as titles
    if re.match(r'^(Introduction|Abstract|Methodology|Results|Conclusion)\b', title, re.I):
        score -= 10
    return score


def _sanitize_title(value: str) -> str:
    title = _safe_first_line(value)
    if not title or title.lower() == NOT_FOUND.lower():
        return NOT_FOUND
    if len(title) < 8 or len(title) > 300:
        return NOT_FOUND
    if any(p.match(title) for p in _TITLE_REJECTION_PATTERNS):
        return NOT_FOUND
    if any(t in title.lower() for t in ("doi.org", "http://", "https://", "arxiv.org", "@")):
        return NOT_FOUND
    if any(t in title.lower() for t in ("abstract", "introduction")):
        return NOT_FOUND
    if _title_score(title) < -3:
        return NOT_FOUND
    return title


def _sanitize_authors(value: str) -> str:
    authors = _safe_first_line(value)
    if not authors or authors.lower() == NOT_FOUND.lower():
        return NOT_FOUND
    if len(authors) < 3 or len(authors) > 300:
        return NOT_FOUND
    lower = authors.lower()
    if any(t in lower for t in ("abstract", "introduction", "doi", "http", "arxiv", "@")):
        return NOT_FOUND
    if _AUTHORS_REJECT_PATTERNS.search(authors):
        return NOT_FOUND
    if authors.count(" ") > 10 and re.search(r"\b(is|are|has|have|was|were|the|a|an)\b", lower):
        return NOT_FOUND
    return authors


def resolve_stable_metadata(
    primary_title:    str,
    primary_authors:  str,
    fallback_title:   str = "",
    fallback_authors: str = "",
) -> tuple[str, str]:
    """
    Pick the best title from all candidates using _title_score.
    If all sanitized candidates return NOT_FOUND, fall back to keeping the
    longest raw candidate that passes basic sanity checks (not a section
    heading, not a URL, not too short/long).
    """
    raw_candidates = [primary_title, fallback_title]
    sanitized = [_sanitize_title(r) for r in raw_candidates]
    valid_titles = [t for t in sanitized if t != NOT_FOUND]

    if valid_titles:
        title = max(valid_titles, key=_title_score)
    else:
        # Last resort: take the longest raw candidate that looks like a title
        survivors = []
        for r in raw_candidates:
            t = _safe_first_line(r)
            if not t or t.lower() == NOT_FOUND.lower():
                continue
            if len(t) < 8 or len(t) > 300:
                continue
            if any(p.match(t) for p in _TITLE_REJECTION_PATTERNS):
                continue
            if any(tok in t.lower() for tok in ("doi.org", "http://", "https://", "arxiv.org", "@")):
                continue
            survivors.append(t)
        title = max(survivors, key=len) if survivors else NOT_FOUND

    authors = _sanitize_authors(primary_authors)
    if authors == NOT_FOUND and fallback_authors:
        authors = _sanitize_authors(fallback_authors)

    return title, authors


# ── Assessment helpers ────────────────────────────────────────────────────────

def _make_insufficient(dimension: str, reason: str | None = None) -> ReviewDimensionAssessment:
    return ReviewDimensionAssessment(
        verdict=NOT_ENOUGH_EVIDENCE,
        rationale=reason or f"Insufficient explicit evidence in the paper text to assess {DIMENSION_LABELS[dimension].lower()} confidently.",
        evidence=[],
        confidence="Low",
    )


def _template_rationale(dimension: str, evidence: list[str]) -> str:
    if evidence:
        return f"The judgement is grounded in explicit evidence from the extracted paper text, including {'; '.join(evidence[:2])}."
    return f"Insufficient explicit evidence in the paper text to assess {DIMENSION_LABELS[dimension].lower()} confidently."


def _normalize_assessment(
    dimension:  str,
    assessment: ReviewDimensionAssessment,
    sections:   PaperSections,
) -> ReviewDimensionAssessment:
    fallback_ev = _fallback_dimension_evidence(dimension, sections)
    evidence = _clean_evidence_items([*assessment.evidence, *fallback_ev], limit=6)
    confidence = _normalize_confidence(assessment.confidence)

    # ── Enforce controlled verdict vocabulary ─────────────────────────────────
    enforced_verdict = _enforce_verdict(dimension, assessment.verdict)
    evidence = _check_verdict_evidence_consistency(dimension, enforced_verdict, evidence)
    evidence = _remove_contradicted_evidence(evidence)
    specific = _specific_count(evidence)
    text = _norm(" ".join([assessment.verdict, assessment.rationale, *evidence]))

    # ── Generic / empty guard ─────────────────────────────────────────────────
    if not evidence or (_contains_phrase(text, GENERIC_PHRASES) and specific == 0):
        if dimension == "reproducibility":
            return _make_insufficient(dimension,
                "Insufficient explicit evidence on hyperparameters, setup details, data splits, "
                "evaluation protocol, or release artifacts to judge reproducibility confidently.")
        if dimension == "fairness_of_comparison":
            return _make_insufficient(dimension,
                "Insufficient explicit evidence on matched datasets, metrics, baselines, "
                "or experimental controls to judge fairness confidently.")
        return _make_insufficient(dimension)

    # ── Novelty ───────────────────────────────────────────────────────────────
    if dimension == "novelty":
        nt = _norm(" ".join([text, _source_text(sections)])).lower()
        found_types = [n for n, kws in NOVELTY_TYPES.items() if any(k in nt for k in kws)]
        # "foundational" requires stronger signals than just "introduces" —
        # require at least one of: "new framework", "new formulation", "new paradigm", "first"
        strong_foundational = any(p in nt for p in (
            "new framework", "new formulation", "new paradigm", "introduces a new",
            "we propose a new", "first to", "first approach",
        ))
        is_foundational = "foundational" in found_types and strong_foundational
        if is_foundational and "efficiency" in found_types:
            verdict = "Mixed novelty: combines foundational and efficiency elements"
        elif is_foundational:
            verdict = "Foundational contribution: introduces a new paradigm or primitive"
        elif "efficiency" in found_types and "extension" in found_types:
            verdict = "Efficiency innovation built on prior named methods"
        elif "efficiency" in found_types:
            verdict = "Efficiency innovation built on prior named methods"
        elif "practical" in found_types:
            verdict = "Practical or engineering innovation"
        elif "extension" in found_types:
            verdict = "Incremental extension of prior work"
        else:
            verdict = NOT_ENOUGH_EVIDENCE
        rationale = assessment.rationale
        if not rationale or _contains_phrase(rationale, GENERIC_PHRASES) or _contains_phrase(rationale, ASSERTIVE_PHRASES):
            rationale = "The paper supports a contribution claim, but the paper text alone does not justify a strong ranking of overall field impact."
        raw_conf = "High" if specific >= 3 and is_foundational else "Medium"
        return ReviewDimensionAssessment(
            verdict=verdict, rationale=rationale, evidence=evidence,
            confidence=_cap_confidence(raw_conf, evidence, specific),
        )

    # ── Reproducibility ───────────────────────────────────────────────────────
    if dimension == "reproducibility":
        source = _source_text(sections)
        if not any("release" in i.lower() or "code" in i.lower() for i in evidence):
            evidence = _clean_evidence_items([*evidence, "Code or checkpoint release is not mentioned in the extracted text."])
        if not any("split" in i.lower() for i in evidence):
            evidence = _clean_evidence_items([*evidence, "Train/validation/test split details are not explicit in the extracted text."])
        combined_text = _norm(" ".join([source, *evidence]))
        found          = _collect_signal_groups(combined_text, REPRO_SIGNALS)
        has_hyper      = "hyperparameters" in found
        has_data       = "data" in found
        # Negation-aware: evidence items saying "does not release" don't count as release signal
        _RELEASE_NEGATION = re.compile(
            r"\b(does not|do not|not mention|not release|no code|no checkpoint|"
            r"omit|not available|not included|unreported)\b.*\b(code|checkpoint|release|github|repository)\b",
            re.I,
        )
        has_release = (
            "release" in found
            and not any(_RELEASE_NEGATION.search(i) for i in evidence)
        )
        enough_signals = len(found) >= 3 and (has_hyper or has_data)
        strong_signals = len(found) >= 4
        if not enough_signals:
            return _make_insufficient(dimension,
                "Insufficient explicit evidence on hyperparameters, setup details, data splits, "
                "evaluation protocol, or release artifacts to judge reproducibility confidently.")
        # Map to controlled vocabulary
        if strong_signals:
            verdict = "Reasonably reproducible: ≥3 signals present"
        else:
            verdict = "Partially reproducible: some details present but incomplete"
        rationale = _reproducibility_rationale(found, evidence)
        if not strong_signals:
            rationale += " Important implementation details remain incomplete in the extracted text."
        # Calibrated confidence — no release → cap at Medium
        if has_release and strong_signals:    raw_conf = "High"
        elif strong_signals:                  raw_conf = "Medium"
        elif has_hyper or has_data:           raw_conf = "Medium"
        else:                                 raw_conf = "Low"
        return ReviewDimensionAssessment(verdict=verdict, rationale=rationale,
                                         evidence=evidence, confidence=_cap_confidence(raw_conf, evidence, specific))

    # ── Fairness of Comparison ────────────────────────────────────────────────
    if dimension == "fairness_of_comparison":
        source = _source_text(sections)
        if not any("matched" in i.lower() or "controlled" in i.lower() for i in evidence):
            evidence = _clean_evidence_items([*evidence, "Matched training or evaluation controls are not explicit in the extracted text."])
        combined_text = _norm(" ".join([source, *evidence]))
        found         = _collect_signal_groups(combined_text, FAIRNESS_SIGNALS)
        has_baselines = "baselines" in found
        has_metrics   = "metrics" in found
        has_controls  = "controls" in found
        if not (has_baselines and has_metrics):
            return _make_insufficient(dimension,
                "Insufficient explicit evidence on matched datasets, metrics, baselines, "
                "or experimental controls to judge fairness confidently.")
        # Map to controlled vocabulary
        if has_controls:
            verdict = "Comparison setup appears fair: named baselines and matched metrics present"
        else:
            verdict = "Partially fair: baselines named but conditions not fully matched"
        rationale = _template_rationale(dimension, evidence)
        if not has_controls:
            rationale += " The extracted text does not make matched experimental controls fully explicit."
        if has_controls and len(found) >= 4:  raw_conf = "High"
        elif has_baselines and has_metrics:   raw_conf = "Medium"
        else:                                 raw_conf = "Low"
        return ReviewDimensionAssessment(verdict=verdict, rationale=rationale,
                                         evidence=evidence, confidence=_cap_confidence(raw_conf, evidence, specific))

    if dimension == "applicability":
        source = _source_text(sections)
        deploy_sigs = (
            "deployment", "efficient", "latency", "memory", "compute", "practical",
            "real-world", "scalable", "gpu", "parameter", "hardware", "inference",
            "resource", "overhead", "cost", "throughput", "vram", "quantization",
        )
        hits = [s for s in deploy_sigs if s in source.lower() or s in text.lower()]
        if len(hits) >= 2 and enforced_verdict in (NOT_ENOUGH_EVIDENCE, ""):
            # We have deploy signals — override to constrained applicability at minimum
            enforced_verdict = "Applicability is domain- or resource-constrained"
            raw_conf = "Medium" if specific >= 2 else "Low"
            return ReviewDimensionAssessment(
                verdict=enforced_verdict,
                rationale=(f"The paper contains explicit deployment-relevant signals ({', '.join(hits[:4])}), "
                           "which allow an applicability assessment grounded in the extracted text."),
                evidence=evidence,
                confidence=_cap_confidence(raw_conf, evidence, specific),
            )

    # ── Weaknesses / Threats to Validity ─────────────────────────────────────
    if dimension in {"weaknesses", "threats_to_validity"}:
        implicit_tokens = (
            "limitation", "limited", "lack", "missing", "not report", "only", "unclear",
            "bias", "omits", "cannot", "does not", "may not", "however", "constraint",
            "requires", "depend", "overhead", "trade-off", "tradeoff", "caveat",
            "suboptimal", "challenging", "expensive", "restrictive",
        )
        extended = " ".join([
            text.lower(),
            sections.introduction.lower(), sections.methodology.lower(),
            sections.abstract.lower(),     sections.results.lower(),
        ])
        has_implicit = any(t in extended for t in implicit_tokens)
        llm_ok = (_norm(assessment.verdict).lower() not in ("not enough evidence", "") and specific > 0)
        if not has_implicit and not llm_ok:
            return _make_insufficient(dimension)

    # ── Assumptions ───────────────────────────────────────────────────────────
    if dimension == "assumptions":
        assumption_tokens = (
            "assume", "requires", "depends on", "availability", "under",
            "pretrain", "rank", "low-rank", "full rank", "frozen", "quantiz",
            "intrinsic", "subspace", "low intrinsic",
        )
        extended = " ".join([
            text.lower(), sections.methodology.lower(),
            sections.introduction.lower(), sections.abstract.lower(),
        ])
        if not any(t in extended for t in assumption_tokens):
            return _make_insufficient(dimension)

    # ── Generic confidence calibration ────────────────────────────────────────
    if confidence == "High" and specific < 3:
        confidence = "Medium"
    if confidence == "Medium" and specific < 2:
        confidence = "Low"
    confidence = _cap_confidence(confidence, evidence, specific)

    rationale = (
        assessment.rationale
        if assessment.rationale and not _contains_phrase(assessment.rationale, GENERIC_PHRASES)
        else _template_rationale(dimension, evidence)
    )
    return ReviewDimensionAssessment(
        verdict=enforced_verdict or NOT_ENOUGH_EVIDENCE,
        rationale=rationale, evidence=evidence, confidence=confidence,
    )


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> str:
    depth_brace = depth_square = 0
    last_safe   = 0
    in_string   = False
    escape_next = False
    for i, ch in enumerate(raw):
        if escape_next:             escape_next = False; continue
        if ch == "\\" and in_string: escape_next = True; continue
        if ch == '"':               in_string = not in_string; continue
        if in_string:               continue
        if   ch == "{":  depth_brace  += 1
        elif ch == "}":
            depth_brace -= 1
            if depth_brace == 0: last_safe = i
        elif ch == "[":  depth_square += 1
        elif ch == "]":  depth_square -= 1
    if depth_brace == 0 and depth_square == 0:
        return raw
    repaired = raw[: last_safe + 1] if last_safe > 0 else raw
    return repaired + "]" * max(depth_square, 0) + "}" * max(depth_brace - 1, 0)


def _parse_json_response(raw: str, context: str) -> dict:
    cleaned = strip_code_fences(raw).strip()
    logger.debug("%s raw response (first 500): %s", context, cleaned[:500])
    for attempt in (cleaned, _repair_truncated_json(cleaned)):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):  return {"pairwise_comparisons": parsed}
            if isinstance(parsed, dict):  return parsed
            raise TypeError(f"Unexpected type: {type(parsed).__name__}")
        except json.JSONDecodeError:
            continue
    logger.error("%s JSON parse failed. Raw (first 800): %s", context, cleaned[:800])
    raise CriticalReviewError("The AI returned invalid critical-review data. Please try again.")


def _validate_profile(data: dict) -> PaperCriticalProfile:
    try:
        return PaperCriticalProfile.model_validate(data)
    except ValidationError as exc:
        logger.error("Profile validation failed: %s", exc)
        raise CriticalReviewError(
            "The AI returned an incomplete critical-review profile. Please try again.", original=exc
        ) from exc


# ── Pairwise helpers ──────────────────────────────────────────────────────────

def _ordered_pairwise(items: list[dict]) -> dict[str, PairwiseDimensionComparison]:
    result: dict[str, PairwiseDimensionComparison] = {}
    for item in items:
        norm = _normalize_pairwise_item(item)
        if norm is None:
            logger.warning("Skipping pairwise item with missing dimension: %s", item)
            continue
        try:
            cmp = PairwiseDimensionComparison.model_validate(norm)
        except ValidationError as exc:
            logger.error("Pairwise validation failed: %s | %s", exc, norm)
            continue
        key = cmp.dimension.strip().lower()
        if key in CRITICAL_REVIEW_DIMENSIONS and key not in result:
            result[key] = cmp
    return result


def _profile_score(a: ReviewDimensionAssessment) -> int:
    return _specific_count(a.evidence) + {"High": 3, "Medium": 2, "Low": 1}.get(
        _normalize_confidence(a.confidence), 1
    )


def _novelty_types(a: ReviewDimensionAssessment) -> set[str]:
    text = _norm(" ".join([a.verdict, a.rationale, *a.evidence])).lower()
    return {lbl for lbl, kws in NOVELTY_TYPES.items() if any(k in text for k in kws)}


def _pairwise_evidence(
    a: ReviewDimensionAssessment,
    b: ReviewDimensionAssessment,
    raw: PairwiseDimensionComparison | None,
) -> list[str]:
    evidence: list[str] = []
    a_set = {_clean_bullet(i).lower() for i in a.evidence}
    b_set = {_clean_bullet(i).lower() for i in b.evidence}
    if raw:
        for item in raw.evidence[:4]:
            if _clean_bullet(item).lower() not in a_set | b_set:
                evidence.append(item)
    evidence.extend(f"Paper A: {i}" for i in a.evidence[:2])
    evidence.extend(f"Paper B: {i}" for i in b.evidence[:2])
    return _clean_evidence_items(evidence, limit=6)


def _safe_pairwise_fallback(
    dimension: str,
    a: ReviewDimensionAssessment,
    b: ReviewDimensionAssessment,
    reason: str,
) -> PairwiseDimensionComparison:
    label    = DIMENSION_LABELS[dimension].lower()
    evidence = _clean_evidence_items(
        [f"Paper A: {i}" for i in a.evidence[:2]] + [f"Paper B: {i}" for i in b.evidence[:2]],
        limit=4,
    )
    # Sanitize reason: hide raw provider/API/rate-limit error text from users
    _ERROR_PATTERNS = re.compile(
        r"(rate.?limit|groq|api.?key|http[s]?|error code|status.?code|provider|"
        r"traceback|exception|stacktrace|internal server|token|request.?id)",
        re.I,
    )
    if _ERROR_PATTERNS.search(reason) or len(reason) > 200:
        safe_reason = "Pairwise synthesis was incomplete due to a processing issue."
    else:
        safe_reason = reason or "Pairwise synthesis was incomplete."
    return PairwiseDimensionComparison(
        dimension=dimension,
        paper_a=a.verdict or NOT_ENOUGH_EVIDENCE,
        paper_b=b.verdict or NOT_ENOUGH_EVIDENCE,
        comparative_judgement=PAIRWISE_FALLBACK_JUDGEMENT,
        rationale=(
            f"For {label}: Paper A is assessed as '{a.verdict}' (confidence: {a.confidence or 'Low'}), "
            f"while Paper B is assessed as '{b.verdict}' (confidence: {b.confidence or 'Low'}). "
            f"Direct pairwise synthesis was unavailable: {safe_reason}"
        ).strip(),
        evidence=evidence,
    )


def _fallback_pairwise(
    dimension: str,
    a: ReviewDimensionAssessment,
    b: ReviewDimensionAssessment,
    raw: PairwiseDimensionComparison | None = None,
) -> PairwiseDimensionComparison:
    evidence = _pairwise_evidence(a, b, raw)
    label    = DIMENSION_LABELS[dimension].lower()
    sa, sb   = _profile_score(a), _profile_score(b)

    if dimension == "novelty":
        ta, tb   = _novelty_types(a), _novelty_types(b)
        ok_a, ok_b = _specific_count(a.evidence) >= 2, _specific_count(b.evidence) >= 2
        # Require strong foundational signal — not just keyword presence
        text_a = _norm(" ".join([a.verdict, a.rationale, *a.evidence])).lower()
        text_b = _norm(" ".join([b.verdict, b.rationale, *b.evidence])).lower()
        _strong_found = lambda t: any(p in t for p in (
            "new framework", "new formulation", "new paradigm", "introduces a new",
            "we propose a new", "first to", "first approach",
        ))
        found_a = "foundational" in ta and _strong_found(text_a)
        found_b = "foundational" in tb and _strong_found(text_b)
        if found_a and {"extension","efficiency","practical"} & tb and ok_a and ok_b:
            judgement = "Paper A appears more foundational, while Paper B appears more like an extension or efficiency innovation."
        elif found_b and {"extension","efficiency","practical"} & ta and ok_a and ok_b:
            judgement = "Paper B appears more foundational, while Paper A appears more like an extension or efficiency innovation."
        elif "efficiency" in ta and "efficiency" not in tb:
            judgement = "Paper A appears more explicitly focused on efficiency innovation from the available evidence."
        elif "efficiency" in tb and "efficiency" not in ta:
            judgement = "Paper B appears more explicitly focused on efficiency innovation from the available evidence."
        else:
            judgement = "Insufficient evidence to rank overall novelty from the paper text alone."
        rationale = (
            f"Paper A: '{a.verdict}' (confidence: {a.confidence or 'Low'}). "
            f"Paper B: '{b.verdict}' (confidence: {b.confidence or 'Low'}). "
            "Novelty comparison stays conservative unless papers clearly signal different contribution types."
        )
    elif dimension in {"weaknesses", "assumptions", "threats_to_validity"}:
        judgement = f"The papers show different {label} profiles; the evidence does not justify a simple ranking."
        rationale = (
            f"Paper A: '{a.verdict}' (confidence: {a.confidence or 'Low'}). "
            f"Paper B: '{b.verdict}' (confidence: {b.confidence or 'Low'}). "
            "Reporting the differences is more informative than an overall ranking here."
        )
    elif a.confidence == "Low" and b.confidence == "Low":
        judgement = f"Insufficient evidence to rank {label} from the paper text alone."
        rationale = (
            f"Both papers have limited evidence for {label} "
            f"(Paper A: '{a.verdict}', Paper B: '{b.verdict}')."
        )
    elif sa >= sb + 2 and a.confidence != "Low":
        judgement = f"Paper A appears stronger on {label} from the available evidence."
        rationale = (
            f"Paper A ('{a.verdict}', {a.confidence or 'Low'}) provides more specific support "
            f"than Paper B ('{b.verdict}', {b.confidence or 'Low'})."
        )
    elif sb >= sa + 2 and b.confidence != "Low":
        judgement = f"Paper B appears stronger on {label} from the available evidence."
        rationale = (
            f"Paper B ('{b.verdict}', {b.confidence or 'Low'}) provides more specific support "
            f"than Paper A ('{a.verdict}', {a.confidence or 'Low'})."
        )
    else:
        judgement = f"The evidence does not justify a strong ordering on {label}."
        rationale = (
            f"Both papers have some support for {label} "
            f"(Paper A: '{a.verdict}', Paper B: '{b.verdict}'), "
            "but the balance is too close for a stronger claim."
        )

    return PairwiseDimensionComparison(
        dimension=dimension, paper_a=a.verdict, paper_b=b.verdict,
        comparative_judgement=_enforce_pairwise_verdict(dimension, judgement),
        rationale=rationale, evidence=evidence,
    )


def _normalize_pairwise(
    dimension: str,
    raw: PairwiseDimensionComparison | None,
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
) -> PairwiseDimensionComparison:
    a_assessment = getattr(profile_a, dimension)
    b_assessment = getattr(profile_b, dimension)
    fallback = _fallback_pairwise(dimension, a_assessment, b_assessment, raw)
    if a_assessment.verdict == NOT_ENOUGH_EVIDENCE and b_assessment.verdict == NOT_ENOUGH_EVIDENCE:
        vocab = PAIRWISE_VOCAB.get(dimension, ())
        insuf = next((v for v in vocab if "Insufficient" in v), PAIRWISE_FALLBACK_JUDGEMENT)
        return PairwiseDimensionComparison(
            dimension=dimension,
            paper_a=a_assessment.verdict,
            paper_b=b_assessment.verdict,
            comparative_judgement=insuf,
            rationale=f"Both papers returned '{NOT_ENOUGH_EVIDENCE}' for {dimension.replace('_', ' ')}. No comparison is possible.",
            evidence=[],
        )
    if raw is None:
        return fallback
    raw_ev = _clean_evidence_items(raw.evidence, limit=4)
    raw_text = _norm(" ".join([raw.paper_a, raw.paper_b, raw.comparative_judgement, raw.rationale, *raw_ev]))
    specific = _specific_count(raw_ev)
    is_fb_j = PAIRWISE_FALLBACK_JUDGEMENT.lower() in raw_text.lower()
    # Only discard when BOTH generic/assertive AND no specific evidence
    if is_fb_j or ((_contains_phrase(raw_text, ASSERTIVE_PHRASES) or _contains_phrase(raw_text, GENERIC_PHRASES)) and specific == 0):
        return fallback

    # Enforce controlled pairwise vocabulary
    enforced_judgement = _enforce_pairwise_verdict(dimension, raw.comparative_judgement)

    if dimension == "novelty" and enforced_judgement == "Both papers appear to offer efficiency or incremental innovations of comparable scope":
        a_verdict = a_assessment.verdict
        b_verdict = b_assessment.verdict
        if "Foundational contribution" in a_verdict and any(tok in b_verdict for tok in ("Efficiency innovation", "Incremental extension", "Practical or engineering innovation")):
            enforced_judgement = "Paper A appears more foundational; Paper B appears to extend or build on prior work"
        elif "Foundational contribution" in b_verdict and any(tok in a_verdict for tok in ("Efficiency innovation", "Incremental extension", "Practical or engineering innovation")):
            enforced_judgement = "Paper B appears more foundational; Paper A appears to extend or build on prior work"

    return PairwiseDimensionComparison(
        dimension=dimension,
        paper_a=fallback.paper_a,
        paper_b=fallback.paper_b,
        comparative_judgement=enforced_judgement,
        rationale=raw.rationale or fallback.rationale,
        evidence=_clean_evidence_items([*raw_ev, *fallback.evidence], limit=6),
    )


def _coerce_string(value: object, fallback: str = "") -> str:
    return _norm(str(value or "")) or fallback


def _coerce_evidence(value: object) -> list[str]:
    if isinstance(value, list):
        raw = [str(i) for i in value if str(i).strip()]
    elif isinstance(value, str):
        raw = [p.strip() for p in re.split(r"[\n;]+", value) if p.strip()]
    else:
        raw = []
    return _clean_evidence_items(raw, limit=6)


def _normalize_pairwise_item(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    dim = _coerce_string(item.get("dimension")).lower()
    if dim not in CRITICAL_REVIEW_DIMENSIONS:
        return None
    return {
        "dimension":             dim,
        "paper_a":               _coerce_string(item.get("paper_a"),               NOT_ENOUGH_EVIDENCE),
        "paper_b":               _coerce_string(item.get("paper_b"),               NOT_ENOUGH_EVIDENCE),
        "comparative_judgement": _coerce_string(item.get("comparative_judgement"), PAIRWISE_FALLBACK_JUDGEMENT),
        "rationale":             _coerce_string(item.get("rationale"), "Pairwise synthesis was incomplete."),
        "evidence":              _coerce_evidence(item.get("evidence")),
    }


def _extract_pairwise_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [i for i in payload if isinstance(i, dict)]
    if isinstance(payload, dict):
        direct = payload.get("pairwise_comparisons")
        if isinstance(direct, list):
            return [i for i in direct if isinstance(i, dict)]
        for v in payload.values():
            if isinstance(v, list) and any(isinstance(i, dict) for i in v):
                return [i for i in v if isinstance(i, dict)]
    return []


def _compress_profile(p: PaperCriticalProfile) -> dict:
    """
    Build a compact profile dict for the pairwise prompt.
    Keeps only verdict + confidence + first 2 evidence items per dimension.
    Drops long rationales to save ~40% tokens vs full model_dump_json.
    """
    dims = {}
    for dim in CRITICAL_REVIEW_DIMENSIONS:
        a: ReviewDimensionAssessment = getattr(p, dim)
        dims[dim] = {
            "verdict":    a.verdict,
            "confidence": a.confidence,
            "evidence":   a.evidence[:2],
        }
    return {
        "title":   p.title,
        "authors": p.authors,
        **dims,
    }



def _format_evidence_lines(evidence: list[str]) -> list[str]:
    return [f"- {i}" for i in evidence] if evidence else ["- Not enough evidence cited."]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_paper_critical_profile(
    sections: PaperSections,
    client:   Groq,
    settings: Settings,
) -> PaperCriticalProfile:
    # resolve_stable_metadata already called upstream in pipeline_service,
    # so sections.title/authors are already the best available.
    # We call it again here as a safety net with both fields as primary+fallback.
    title, authors = resolve_stable_metadata(
        sections.title, sections.authors,
        sections.title, sections.authors,
    )
    prompt_sections = sections.model_copy(update={"title": title, "authors": authors})
    logger.info("Generating critical profile for: %s", title)
    raw = chat_completion(
        client=client,
        system_prompt=build_critical_profile_system_prompt(),
        user_prompt=build_critical_profile_user_prompt(prompt_sections),
        settings=settings,
        max_tokens=MAX_PROFILE_TOKENS,
    )
    profile = _validate_profile(_parse_json_response(raw, "Critical profile"))
    updates = {
        dim: _normalize_assessment(dim, getattr(profile, dim), prompt_sections)
        for dim in CRITICAL_REVIEW_DIMENSIONS
    }
    return profile.model_copy(update={"title": title, "authors": authors, **updates})


def compare_paper_profiles(
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
    client:    Groq,
    settings:  Settings,
) -> list[PairwiseDimensionComparison]:
    logger.info("Pairwise comparison (up to %d attempts).", MAX_PAIRWISE_RETRIES)
    raw_items: dict[str, PairwiseDimensionComparison] = {}
    last_fail = ""
    sys_prompt = build_pairwise_comparison_system_prompt()
    compressed_a = _compress_profile(profile_a)
    compressed_b = _compress_profile(profile_b)

    for attempt in range(1, MAX_PAIRWISE_RETRIES + 1):
        try:
            logger.info("Pairwise attempt %d/%d.", attempt, MAX_PAIRWISE_RETRIES)
            user_prompt = build_pairwise_comparison_user_prompt(compressed_a, compressed_b)
            if attempt > 1:
                user_prompt = (
                    f"[Attempt {attempt}: previous response was incomplete — "
                    f"include all 8 dimensions and ensure the JSON is valid.]\n\n" + user_prompt
                )
            raw      = chat_completion(client=client, system_prompt=sys_prompt,
                                       user_prompt=user_prompt, settings=settings,
                                       max_tokens=MAX_PAIRWISE_TOKENS)
            payload  = _parse_json_response(raw, f"Pairwise attempt {attempt}")
            items    = _extract_pairwise_items(payload)
            cands    = _ordered_pairwise(items)
            missing  = [d for d in CRITICAL_REVIEW_DIMENSIONS if d not in cands]
            if not missing:
                logger.info("All 8 dimensions on attempt %d.", attempt)
                raw_items = cands; last_fail = ""; break
            logger.warning("Attempt %d missing: %s.", attempt, ", ".join(missing))
            if len(cands) > len(raw_items):
                raw_items = cands
            last_fail = f"Missing after {attempt} attempt(s): {', '.join(missing)}."
            if attempt < MAX_PAIRWISE_RETRIES:
                time.sleep(PAIRWISE_RETRY_DELAY)
        except CriticalReviewError as exc:
            last_fail = str(exc)
            logger.warning("Attempt %d CriticalReviewError: %s.", attempt, exc)
            if attempt < MAX_PAIRWISE_RETRIES: time.sleep(PAIRWISE_RETRY_DELAY)
        except Exception as exc:
            last_fail = f"{type(exc).__name__}: {exc}"
            logger.error("Attempt %d unexpected: %s.", attempt, exc, exc_info=True)
            if attempt < MAX_PAIRWISE_RETRIES: time.sleep(PAIRWISE_RETRY_DELAY)

    comparisons: list[PairwiseDimensionComparison] = []
    for dim in CRITICAL_REVIEW_DIMENSIONS:
        try:
            raw_item = raw_items.get(dim)
            if raw_item is None:
                fn = _safe_pairwise_fallback if last_fail else _fallback_pairwise
                kw = {"reason": last_fail} if last_fail else {}
                comparisons.append(fn(dim, getattr(profile_a, dim), getattr(profile_b, dim), **kw))
            else:
                comparisons.append(_normalize_pairwise(dim, raw_item, profile_a, profile_b))
        except Exception as exc:
            logger.error("Assembly failed for '%s': %s.", dim, exc, exc_info=True)
            comparisons.append(_safe_pairwise_fallback(
                dim, getattr(profile_a, dim), getattr(profile_b, dim),
                f"Assembly error: {type(exc).__name__}: {exc}.",
            ))
    return comparisons


def validate_critical_comparison_result(result: CriticalComparisonResult) -> None:
    if result.paper_a_profile is None or result.paper_b_profile is None:
        raise CriticalReviewError("Missing one or both paper profiles.")
    if len(result.pairwise_comparisons) != len(CRITICAL_REVIEW_DIMENSIONS):
        raise CriticalReviewError(
            f"Expected {len(CRITICAL_REVIEW_DIMENSIONS)} comparisons, got {len(result.pairwise_comparisons)}."
        )
    present = {i.dimension for i in result.pairwise_comparisons}
    missing = [d for d in CRITICAL_REVIEW_DIMENSIONS if d not in present]
    if missing:
        raise CriticalReviewError(f"Missing dimensions: {', '.join(missing)}.")
    for item in result.pairwise_comparisons:
        if not all([item.dimension, item.paper_a, item.paper_b, item.comparative_judgement, item.rationale]):
            raise CriticalReviewError(f"Incomplete entry for '{item.dimension or 'unknown'}'.")
        if not isinstance(item.evidence, list):
            raise CriticalReviewError(f"Malformed evidence for '{item.dimension}'.")


def build_critical_comparison_markdown(result: CriticalComparisonResult) -> str:
    try:
        validate_critical_comparison_result(result)
        lines = ["# Critical Comparison", ""]
        for lbl, prof in (("Paper A", result.paper_a_profile), ("Paper B", result.paper_b_profile)):
            lines += [f"## {lbl}", f"**Title:** {prof.title}", f"**Authors:** {prof.authors}", ""]
            for dim in CRITICAL_REVIEW_DIMENSIONS:
                a = getattr(prof, dim)
                lines += [
                    f"### {DIMENSION_LABELS[dim]}",
                    f"**Verdict:** {a.verdict}", f"**Confidence:** {a.confidence or 'Low'}",
                    a.rationale, "**Evidence:**", *_format_evidence_lines(a.evidence), "",
                ]
        lines += ["## Direct Comparison", ""]
        for cmp in result.pairwise_comparisons:
            lines += [
                f"### {DIMENSION_LABELS.get(cmp.dimension, cmp.dimension.replace('_',' ').title())}",
                f"**Paper A:** {cmp.paper_a}", f"**Paper B:** {cmp.paper_b}",
                f"**Comparative Judgement:** {cmp.comparative_judgement}",
                cmp.rationale, "**Evidence:**", *_format_evidence_lines(cmp.evidence), "",
            ]
        return "\n".join(lines).strip()
    except Exception as exc:
        logger.error("Markdown assembly failed: %s", exc, exc_info=True)
        if isinstance(exc, CriticalReviewError): raise
        raise CriticalReviewError("Failed to assemble comparison result.", original=exc) from exc


def generate_critical_comparison(
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
    client:    Groq,
    settings:  Settings,
) -> CriticalComparisonResult:
    if profile_a is None or profile_b is None:
        raise CriticalReviewError("Cannot compare without both profiles.")
    comparisons = compare_paper_profiles(profile_a, profile_b, client, settings)
    result = CriticalComparisonResult(
        paper_a_profile=profile_a, paper_b_profile=profile_b,
        pairwise_comparisons=comparisons, comparison_markdown="",
    )
    validate_critical_comparison_result(result)
    return result.model_copy(update={"comparison_markdown": build_critical_comparison_markdown(result)})