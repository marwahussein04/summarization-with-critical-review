"""
app/services/critical_review_service.py
Conservative critical-review profiling and pairwise comparison workflow.

Changes in this version
────────────────────────
1. resolve_stable_metadata — smarter title selection + abstract fallback
   - Scores all candidate titles with _title_score() and picks the best one.
   - New _extract_title_from_abstract() tries to recover a real title when the
     primary title field is corrupted (body-text line instead of actual title).
   - Accepts optional `abstract` kwarg at both call sites.

2. _local_verdict_for_dimension — richer fallback verdicts
   - novelty: keyword-based classification into foundational / efficiency /
     extension / incremental instead of always returning NEE.
   - weaknesses / threats_to_validity: any explicit limitation token now
     produces a positive verdict instead of NEE.
   - strengths: upgrades to "Some strengths supported" when numeric results
     or multiple benchmark signals are present.

3. Reproducibility confidence — corrected under-reporting
   - has_release + strong_signals now correctly yields "High" confidence
     (was incorrectly capped at "Medium").

4. _profile_score — verdict-level bonus
   - Adds +2 for any non-NEE verdict, ensuring positive-verdict Low-confidence
     profiles always outrank NEE profiles in pairwise score comparisons.
     Fixes the over-conservative "Insufficient evidence" in pairwise fallback.

5. _fallback_pairwise — asymmetry-aware Low/Low handling
   - When both confidences are Low, checks for verdict asymmetry (one positive,
     one NEE) and emits a cautious directional judgement instead of blanket
     "Insufficient evidence to rank".
   - weaknesses / assumptions / threats_to_validity: same NEE-vs-positive
     asymmetry logic applied before the "different profiles" fallback.

6. _normalize_assessment — calibrated conservatism (unchanged from prior)
7. compare_paper_profiles — retry logic with up to MAX_PAIRWISE_RETRIES.
8. _normalize_pairwise — relaxed guard (unchanged from prior).
9. _repair_truncated_json — recovers from mid-stream truncation.
"""
from __future__ import annotations

import json
import logging
import os
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
MAX_PAIRWISE_RETRIES = 2
PAIRWISE_RETRY_DELAY = 2.0
MAX_PAIRWISE_TOKENS  = 3800   # compressed profiles → plenty of room for 8 dims
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
        "Reasonably reproducible: ≥3 signals present (specify which)",
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
        "Paper A has more identifiable novelty evidence than Paper B",
        "Paper B has more identifiable novelty evidence than Paper A",
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
# Evidence keywords used for 'specific evidence' classification.
# Deliberately excludes generic terms (train, task, compare) that appear in any paper.
EVIDENCE_KEYWORDS = (
    "dataset", "benchmark", "baseline", "ablation", "metric", "accuracy", "f1",
    "bleu", "rouge", "auc", "map", "latency", "memory", "compute", "parameter",
    "batch size", "learning rate", "epoch", "optimizer", "split",
    "validation", "test set", "code", "checkpoint", "release", "quantization",
    "gpu", "deployment", "evaluation protocol", "limitation",
)
# Reproducibility signals — tuned so generic mentions ("the model", "the architecture")
# do NOT trigger a positive signal. Each group needs a concrete, specific keyword.
REPRO_SIGNALS = {
    # Group fires only on explicit hyperparameter values or named optimizers
    "hyperparameters": (
        "learning rate", "batch size", "epoch", "optimizer", "seed",
        "weight decay", "dropout", "warmup",
    ),
    # Group fires only on explicit implementation choices, NOT generic "model" mentions
    "setup": (
        "implementation details", "training setup", "training procedure",
        "quantization scheme", "hardware setup", "training configuration",
        "number of layers", "hidden size", "attention heads",
    ),
    # Group fires on named dataset/split references
    "data": (
        "dataset", "benchmark", "train split", "validation split", "test split",
        "training set", "dev set", "held-out",
    ),
    # Group fires on evaluation protocol mentions
    "protocol": ("metric", "ablation study", "evaluation protocol", "baseline comparison", "protocol"),
    # Group fires only on explicit release mentions
    "release": ("code", "checkpoint", "github", "repository", "release", "open-source"),
}
FAIRNESS_SIGNALS = {
    # Requires a named baseline, not just the word "comparison"
    "baselines":         ("baseline", "baselines", "vs.", "compared to", "against"),
    "tasks_or_datasets": ("dataset", "benchmark", "test set", "evaluation set"),
    # Requires a named metric, not just the word "metric"
    "metrics":           ("accuracy", "f1", "bleu", "rouge", "auc", "map", "perplexity", "exact match"),
    "controls":          ("same setting", "same budget", "matched", "controlled", "same training", "tuned"),
    "caveats":           ("limitation", "caveat", "not directly comparable", "different setting"),
}
# Novelty types — foundational keywords tightened to structural/paradigm signals only.
# "introduces" and "first" removed: they appear in nearly any paper and cause false positives.
NOVELTY_TYPES = {
    "foundational": (
        "new formulation", "new framework", "new paradigm",
        "novel architecture", "novel mechanism", "novel framework",
        "we propose the", "we introduce the", "new model",
    ),
    "extension":    ("extends", "extension", "builds on", "variant", "adaptation", "incremental", "fine-tuning", "finetuning"),
    "efficiency":   ("parameter-efficient", "memory-efficient", "compute-efficient", "quantization", "pruning", "faster inference"),
    "practical":    ("deployment", "scalable", "real-world application", "production", "engineering system"),
}

# Prose / section-heading patterns that make a title candidate look like a
# blog post or chapter heading rather than a paper title.
# Note: only reject clear prose patterns, not academic phrasing.
_TITLE_PROSE_REJECTS = re.compile(
    r"(?:^|\b)(pushing the|state[- ]of[- ]the[- ]art with\s+\w+|chatbot state|"
    r"our approach in|in this paper we|recent advances in a survey|an overview of|"
    r"^(pushing|achieving|improving|exploring|leveraging|harnessing)\b)",
    re.I,
)
_TITLE_REJECTION_PATTERNS = (
    re.compile(r"^(table|figure|fig\.?|appendix)\s*\d+[:.\s-]", re.I),
    re.compile(r"^(results|ablation|discussion|conclusion|references)\b", re.I),
    # Reject pure URL lines
    re.compile(r"^https?://", re.I),
)
# Patterns that indicate a string is NOT a real author list:
# legal/permission text, affiliations-only, emails-only, or section headings.
_AUTHORS_REJECT_PATTERNS = re.compile(
    r"\b(permission|license|grant|copyright|rights reserved|attribution|"
    r"hereby|reproduce|commercial|redistribution|provided that|licen[cs]e|"
    r"all rights|published by|proceedings of|conference on|acm|ieee|"
    r"this work|this paper|we present|we propose|abstract|introduction|doi)\b",
    re.I,
)
# Matches lines that look like pure affiliations: "University of X", "Dept. of Y",
# "Google Research", "MIT CSAIL", etc. — i.e., no personal names.
_AFFILIATION_ONLY = re.compile(
    r"^(?:university|dept|department|institute|school|lab|laboratory|center|centre|"
    r"google|microsoft|meta|openai|deepmind|amazon|apple|facebook|ibm|nvidia|"
    r"research|corporation|inc\.|ltd\.|llc)\b",
    re.I,
)
# Matches lines that are primarily email addresses.
_EMAIL_ONLY = re.compile(r"^[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}$")

_ARCH_NOISE = re.compile(
    r"\b(composed of|stack of|layer|sub-layer|identical layer|"
    r"encoder|decoder|dot-product|scaling factor|embedding|dimension(?:ality)*)\b",
    re.I,
)
_APPLICABILITY_ARCH_NOISE = re.compile(
    r"\b(multi-head|self-attention|feed-forward|residual|layer norm|attention head|"
    r"positional encoding|token embedding|attention weight|query|key|value matrix)\b",
    re.I,
)
_FOREIGN_MODEL_NOISE = re.compile(
    # Matches tokens that look like external model/method names: uppercase acronyms
    # or CamelCase names longer than 3 chars that are NOT the paper under review.
    # We use a heuristic: 2+ uppercase letters adjacent, or CamelCase with digit suffix.
    # The service's _paper_identity_tokens() tells us what belongs to this paper.
    # This regex is intentionally generic — the paper-identity filter in
    # _fallback_dimension_evidence handles the actual exclusion.
    r"\b([A-Z]{2,}[0-9]?[a-z]*|[A-Z][a-z]+[A-Z][a-z0-9]+)\b",
    re.MULTILINE,
)
_NEGATIVE_VERDICT_DIMS: dict[str, tuple[str, ...]] = {
    "threats_to_validity": ("No threats", "Not enough evidence"),
    "weaknesses": ("No significant weaknesses", "Not enough evidence"),
    "fairness_of_comparison": ("Fairness cannot be confirmed", "Not enough evidence"),
}

DIMENSION_EVIDENCE_SOURCES = {
    "strengths":              ("methodology", "results", "conclusion", "key_figures"),
    "weaknesses":             ("limitations", "future_work", "results", "abstract", "conclusion"),
    "novelty":                ("introduction", "methodology", "conclusion", "key_figures", "abstract"),
    "assumptions":            ("methodology", "limitations", "introduction"),
    "threats_to_validity":    ("limitations", "results", "future_work", "abstract", "conclusion"),
    "reproducibility":        ("methodology", "results", "key_figures"),
    "fairness_of_comparison": ("results", "methodology", "key_figures"),
    "applicability":          ("methodology", "results", "conclusion", "future_work", "key_figures"),
}

DIMENSION_KEYWORDS = {
    "strengths":              ("improve", "outperform", "achieves", "reduce", "efficient", "robust", "benchmark", "dataset", "baseline"),
    "weaknesses":             (
        "limitation", "limited", "only", "missing", "unclear", "future work", "not report",
        "does not", "cannot", "lack", "omit", "constraint", "expensive", "narrow",
        "not evaluat", "not tested", "fails", "suboptimal", "trade-off", "tradeoff",
        "restricted", "solely", "without", "no evaluation", "left for future",
    ),
    "novelty":                ("novel", "new", "introduces", "first", "efficient", "parameter-efficient", "quantization", "extension"),
    "assumptions":            ("assume", "requires", "depends", "availability", "fixed", "pretrained", "rank", "low-rank", "frozen", "quantiz"),
    "threats_to_validity":    (
        "limitation", "limited to", "bias", "generalization", "scope",
        "not evaluat", "not tested", "narrow", "caveat", "only evaluates",
        "cannot generalize", "threat", "evaluation gap", "ablation",
        "future work", "does not", "restricted to", "no evaluation",
    ),
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
    """Deduplicate, truncate, and filter evidence items.
    Prefers specific items (those with named entities or numeric results).
    Caps output at `limit`; trims long items at word boundary ≤ 160 chars.
    """
    seen: set[str] = set()
    seen_nums: set[str] = set()
    items: list[str] = []
    for item in evidence:
        cleaned = _clean_bullet(item)
        # Collapse pipe-separated key-figure entries to first meaningful segment
        if "|" in cleaned:
            segments = [s.strip() for s in cleaned.split("|")]
            first = segments[0].strip()
            cleaned = first if len(first) > 15 else " — ".join(s for s in segments[:2] if s)
        lowered = cleaned.lower()
        if not cleaned or len(cleaned) < 12 or lowered in seen:
            continue
        if _contains_phrase(cleaned, GENERIC_PHRASES):
            continue
        # Trim long items (tightened from 180 → 160 for readability)
        if len(cleaned) > 160:
            cleaned = cleaned[:157].rsplit(" ", 1)[0] + "…"
            lowered = cleaned.lower()
        if _is_raw_number_dump(cleaned):
            continue
        # Deduplicate on the first numeric value for metric-like items
        _metric_kws = ("elo", "latency", "perplexity", "accuracy", "bleu", "rouge", "f1")
        if any(kw in lowered for kw in _metric_kws):
            m = re.search(r'\b(\d[\d,.]*)\b', lowered)
            fp = m.group(1) if m else ""
            if fp and fp in seen_nums:
                continue
            if fp:
                seen_nums.add(fp)
        seen.add(lowered)
        items.append(cleaned)
    # Return specific items first; fall back to all items if none qualify
    specific = [i for i in items if _is_specific_evidence(i)]
    return (specific or items)[:limit]


# Named-entity signals that strongly indicate a concrete, grounded evidence item.
# Deliberately narrow: a digit alone is insufficient if the sentence is generic.
_SPECIFIC_EVIDENCE_PATTERNS = re.compile(
    r"\b("
    r"dataset|benchmark|baseline|ablation|accuracy|f1[- ]score|bleu|rouge|"
    r"perplexity|auc|exact match|memory footprint|latency|throughput|"
    r"learning rate|batch size|epoch|optimizer|checkpoint|github|open-source|"
    r"train split|test split|validation split|held-out|evaluation protocol|"
    r"does not report|does not mention|no code|no checkpoint|not release|"
    r"not evaluat|limited to|only evaluates|only compares"
    r")\b",
    re.I,
)

def _is_specific_evidence(item: str) -> bool:
    """Return True only when the item cites a concrete named entity or omission,
    not just any item that contains a digit or a broad keyword.
    """
    cleaned = _clean_bullet(item)
    if not cleaned or _contains_phrase(cleaned, GENERIC_PHRASES):
        return False
    # Require a named-entity pattern match
    if _SPECIFIC_EVIDENCE_PATTERNS.search(cleaned):
        return True
    # A numeric result paired with a unit/metric marker is also specific
    if re.search(r"\d", cleaned) and re.search(
        r"(%|\bpp\b|score|point|param|mb|gb|ms|second|token|sample)", cleaned, re.I
    ):
        return True
    return False


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


# Per-dimension tokens that MUST appear for a sentence to be considered on-dimension.
# Used in _score_candidate to penalize off-dimension evidence leaking in.
_DIMENSION_REQUIRED_TOKENS: dict[str, tuple[str, ...]] = {
    "threats_to_validity": (
        "limitation", "limited", "bias", "generalization", "scope",
        "not evaluat", "narrow", "caveat", "concern", "only", "cannot",
    ),
    "assumptions": (
        "assume", "assumption", "requires", "dependent on", "relies on",
        "presuppose", "condition", "constraint",
    ),
    "reproducibility": (
        "learning rate", "batch size", "epoch", "optimizer", "seed",
        "dataset", "split", "code", "checkpoint", "release",
        "implementation", "github", "repository", "hyperparameter",
        "training setup", "evaluation protocol", "train split", "test split",
    ),
    "fairness_of_comparison": (
        "baseline", "compared", "metric", "accuracy", "f1", "bleu", "rouge",
        "same setting", "matched", "benchmark",
    ),
    "applicability": (
        "deployment", "latency", "memory", "compute", "hardware",
        "real-world", "scalable", "inference", "throughput", "resource",
    ),
}

_RESULT_NOISE = re.compile(
    r"(latency|throughput|tokens/s|tokens per second|ms|millisecond|"
    r"inference time|forward pass|inference speed)",
    re.I,
)
_SETUP_KWS_REPRO = (
    "learning rate", "batch", "epoch", "optimizer", "seed", "split",
    "implementation", "checkpoint", "github", "repository", "hyperparameter",
    "training setup", "evaluation protocol",
)


def _score_candidate(sentence: str, keywords: tuple[str, ...], dimension: str = "") -> int:
    """Score a candidate evidence sentence for a given dimension.

    Rewards:
      - digit density (concrete results)
      - dimension keyword matches
      - negation markers (limitations, omissions)
      - length ≥ 40 chars

    Penalizes:
      - architecture noise in threats/applicability
      - sentences missing required dimension tokens
    """
    lower = sentence.lower()
    score  = len(re.findall(r"\d", sentence)) * 4
    score += sum(3 for k in keywords if k in lower)
    score += 2 if "%" in sentence else 0
    score += 2 if any(m in lower for m in ("not ", "only ", "missing", "limited", "unclear")) else 0
    score += 1 if len(sentence) > 40 else 0
    # Dimension-specific on-topic enforcement
    required = _DIMENSION_REQUIRED_TOKENS.get(dimension)
    if required and not any(t in lower for t in required):
        score -= 6   # heavy penalty for off-dimension content
    if dimension in {"threats_to_validity", "applicability"} and _ARCH_NOISE.search(sentence):
        score -= 6
    if dimension == "applicability" and _APPLICABILITY_ARCH_NOISE.search(sentence):
        score -= 6
    # Reproducibility: penalise sentences that are purely latency/throughput results —
    # they are not setup or protocol details. Sentences with "ms", "tokens/s", "latency"
    # but no setup keywords get a score reduction.
    if dimension == "reproducibility":
        if _RESULT_NOISE.search(sentence) and not any(k in lower for k in _SETUP_KWS_REPRO):
            score -= 12
    return score


def _filter_evidence_by_dimension(dimension: str, items: list[str]) -> list[str]:
    """Drop evidence items that are clearly off-dimension.

    Uses the required-token list for strict dimensions, and the dimension keyword
    list as a soft gate for others. Items that pass neither gate are only dropped
    when the dimension is strict (threats, assumptions, reproducibility, applicability)
    to avoid over-filtering for softer dimensions like strengths/weaknesses.
    """
    strict_dims = {"threats_to_validity", "assumptions", "reproducibility",
                   "fairness_of_comparison", "applicability"}
    required = _DIMENSION_REQUIRED_TOKENS.get(dimension)
    kws      = DIMENSION_KEYWORDS.get(dimension, ())
    result   = []
    for item in items:
        lower = item.lower()
        # Always pass items that explicitly state absence/omission — they are
        # useful as negative evidence across dimensions
        is_absence_claim = bool(re.search(
            r"\b(not report|not mention|not release|not evaluat|missing|absent|"
            r"no code|no checkpoint|does not|lack|omit|unclear)\b", lower
        ))
        if is_absence_claim:
            result.append(item)
            continue
        if required and any(t in lower for t in required):
            result.append(item)
            continue
        if kws and any(k in lower for k in kws):
            result.append(item)
            continue
        # For strict dimensions: drop items that match neither gate
        if dimension in strict_dims:
            continue
        # For non-strict dimensions: keep but score will be low (cleaned later)
        result.append(item)
    return result


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
    absent  = sorted(all_groups - found)
    parts: list[str] = []
    if present:
        parts.append(f"Reproducibility signals present: {', '.join(present)}.")
    if absent:
        parts.append(f"Missing signals: {', '.join(absent)}.")
    # Explicit release check (negation-aware)
    has_positive_release = any(
        any(tok in e.lower() for tok in ("code", "checkpoint", "github", "repository", "open-source"))
        and not any(neg in e.lower() for neg in ("not", "no ", "without", "missing", "absent"))
        for e in evidence
    )
    if not has_positive_release:
        parts.append("No code or checkpoint release is mentioned in the extracted text.")
    return " ".join(parts) if parts else "Partial reproducibility evidence found in the extracted text."


def _cap_confidence(confidence: str, evidence: list[str], specific: int) -> str:
    """Cap confidence based on evidence quality.

    Thresholds (generic, not paper-specific):
      High   → requires ≥ 3 specific items AND no generic phrases.
      Medium → requires ≥ 2 specific items.
      Low    → everything else.
    """
    capped = _normalize_confidence(confidence)
    joined = " ".join(evidence)
    if capped == "High" and (_contains_phrase(joined, GENERIC_PHRASES) or specific < 3):
        capped = "Medium"
    if capped == "Medium" and specific < 2:
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
    # "comparable" and "similar" no longer map to insufficient — they should map
    # to the "Both papers…" option when both papers are on equal footing, not to
    # the generic fallback. Only true inability words trigger insufficient.
    insufficient = any(p in v_lower for p in (
        "insufficient", "not enough", "cannot compare", "unclear", "no clear evidence"
    ))
    both_equal = any(p in v_lower for p in ("comparable", "similar", "both papers", "no clear ordering"))

    # Pick best match based on direction
    options = list(vocab)
    a_opts = [o for o in options if "Paper A" in o and "Paper B" not in o]
    b_opts = [o for o in options if "Paper B appears" in o or "Paper B has" in o or "Paper B's" in o]
    both_opts = [o for o in options if "Both" in o]
    insuf_opts = [o for o in options if "Insufficient" in o]

    if a_positive and not b_positive and a_opts:
        return a_opts[0]
    if b_positive and not a_positive and b_opts:
        return b_opts[0]
    if insufficient and insuf_opts:
        return insuf_opts[0]
    if both_equal and both_opts:
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
    """Conservative author-list sanitation.

    Rejects: legal/permission text, pure affiliation lines, email-only lines,
    section headings, descriptive prose, and strings with no plausible
    personal name (capitalised word pair).
    """
    authors = _safe_first_line(value)
    if not authors or authors.lower() == NOT_FOUND.lower():
        return NOT_FOUND
    if len(authors) < 3 or len(authors) > 350:
        return NOT_FOUND
    lower = authors.lower()
    # Reject if it contains any URL, DOI, or lone email token
    if any(t in lower for t in ("doi", "http", "arxiv", "@", "doi.org")):
        return NOT_FOUND
    # Reject section headings / prose keywords
    if any(t in lower for t in ("abstract", "introduction", "conclusion",
                                "table ", "figure ", "appendix")):
        return NOT_FOUND
    # Reject legal / permission / publication venue text
    if _AUTHORS_REJECT_PATTERNS.search(authors):
        return NOT_FOUND
    # Reject pure affiliation-only lines (no personal name)
    if _AFFILIATION_ONLY.match(authors.strip()):
        return NOT_FOUND
    # Reject pure email addresses
    if _EMAIL_ONLY.match(authors.strip()):
        return NOT_FOUND
    # Reject long sentences that read as prose (contain is/are/the/a)
    if authors.count(" ") > 12 and re.search(
        r"\b(is|are|has|have|was|were|the|a\s|an\s|we\s|our\s|this\s)\b", lower
    ):
        return NOT_FOUND
    # Must contain at least one plausible personal-name token:
    # a capitalised word ≥ 3 chars that is not a known affiliation keyword.
    name_like = re.findall(r"\b[A-Z][a-z]{2,}\b", authors)
    _INST_WORDS = {"University", "Department", "Institute", "School", "Center",
                   "Laboratory", "Research", "Google", "Microsoft", "Meta",
                   "Openai", "Deepmind", "Amazon", "Nvidia", "Facebook"}
    personal_names = [w for w in name_like if w not in _INST_WORDS]
    if not personal_names:
        return NOT_FOUND
    return authors


def _extract_title_from_text(text: str, min_score: int = 2) -> str:
    """Scan multi-line text (abstract, introduction front-matter) for a title candidate.

    Strategy: every line that is short enough to be a title (12–150 chars), passes
    _sanitize_title, and scores >= min_score with _title_score is a candidate.
    We pick the highest-scoring one rather than the first.
    Returns NOT_FOUND when nothing useful is found.
    """
    if not text or text == NOT_FOUND:
        return NOT_FOUND
    best, best_score = NOT_FOUND, -99
    for line in text.splitlines():
        candidate = _norm(line)
        if 12 <= len(candidate) <= 180:
            if _sanitize_title(candidate) != NOT_FOUND:
                s = _title_score(candidate)
                if s >= min_score and s > best_score:
                    best, best_score = candidate, s
    return best


# Keep backward-compat alias used in resolve_stable_metadata
def _extract_title_from_abstract(abstract: str) -> str:
    return _extract_title_from_text(abstract)


def resolve_stable_metadata(
    primary_title:    str,
    primary_authors:  str,
    fallback_title:   str = "",
    fallback_authors: str = "",
    abstract:         str = "",
    introduction:     str = "",
) -> tuple[str, str]:
    """
    Pick the best title from all candidates using _title_score.
    If all sanitized candidates return NOT_FOUND, fall back to keeping the
    longest raw candidate that passes basic sanity checks (not a section
    heading, not a URL, not too short/long).

    If that also fails, try extracting a title-like line from the abstract.
    """
    abstract_candidate = _extract_title_from_text(abstract) if abstract else NOT_FOUND
    # Also try the very first ~400 chars of the introduction — many PDFs embed the
    # real title at the top of the first body section rather than in a dedicated field.
    intro_head = (introduction or "")[:400]
    intro_candidate = _extract_title_from_text(intro_head, min_score=3) if intro_head else NOT_FOUND
    raw_candidates = [primary_title, fallback_title, abstract_candidate, intro_candidate]
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


# ── Consistency checker ────────────────────────────────────────────────────

# Positive-assertion patterns that should never appear in a NEE or negative rationale.
_ASSERTIVE_RATIONALE = re.compile(
    r"\b(demonstrates|achieves|shows|significantly|outperforms|introduces|provides|"
    r"establishes|confirms|supports|enables|allows|offers|delivers|proves)\b",
    re.I,
)
# Absence/negative patterns that should never appear in a positive rationale.
_ABSENCE_RATIONALE = re.compile(
    r"\b(does not|not mention|not report|not provide|not release|no code|no checkpoint|"
    r"insufficient|missing|absent|unclear|not evaluated|limited information)\b",
    re.I,
)

def _assert_nee_consistency(result: ReviewDimensionAssessment) -> ReviewDimensionAssessment:
    """Post-processing consistency check.

    Rule 1: If verdict is NEE, strip any assertive language from rationale
            and ensure evidence list is empty or contains only absence claims.
    Rule 2: If verdict is positive (not NEE), strip any absence-language from rationale.

    This is a safeguard — ideally the LLM and normalisation logic already handle
    consistency, but this catches residual contradictions generically.
    """
    is_nee = result.verdict == NOT_ENOUGH_EVIDENCE
    rationale = result.rationale or ""

    if is_nee:
        # NEE rationale must not contain assertive claims
        if _ASSERTIVE_RATIONALE.search(rationale):
            # Replace with a neutral explanation
            rationale = (
                "The extracted paper text does not provide sufficient evidence "
                "to support a confident verdict for this dimension."
            )
        # NEE evidence must only contain absence claims, not positive metrics
        _POSITIVE_METRIC = re.compile(
            r"\b\d[\d.]*\s*(%|f1|accuracy|bleu|rouge|auc|map|score|point)\b", re.I
        )
        clean_ev = [e for e in result.evidence if not _POSITIVE_METRIC.search(e)]
        return ReviewDimensionAssessment(
            verdict=result.verdict,
            rationale=rationale,
            evidence=clean_ev,
            confidence="Low",   # NEE always gets Low confidence
        )
    else:
        # Positive verdict rationale must not contain absence language
        if _ABSENCE_RATIONALE.search(rationale):
            # Do not modify rationale — just downgrade confidence to signal inconsistency.
            # A rationale saying "does not report X" while verdict is positive is a red flag.
            confidence = "Low" if result.confidence == "High" else result.confidence
            return ReviewDimensionAssessment(
                verdict=result.verdict,
                rationale=rationale,
                evidence=result.evidence,
                confidence=confidence,
            )
    return result



def _normalize_assessment(
    dimension:  str,
    assessment: ReviewDimensionAssessment,
    sections:   PaperSections,
) -> ReviewDimensionAssessment:
    fallback_ev = _fallback_dimension_evidence(dimension, sections)
    # Filter both LLM evidence and fallback evidence through the dimension lens
    # before merging — prevents off-dimension snippets from inflating signal counts.
    merged_raw = _filter_evidence_by_dimension(
        dimension, [*assessment.evidence, *fallback_ev]
    )
    evidence = _clean_evidence_items(merged_raw, limit=6)
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
        # "foundational" requires stronger signals — not just the word "introduces".
        # We look for patterns indicating the paper itself proposes a novel primitive,
        # architecture, or paradigm from scratch (not derived from a prior named method).
        _STRONG_FOUNDATIONAL = (
            "new framework", "new formulation", "new paradigm", "introduces a new",
            "we propose a new", "we propose the", "first to", "first approach",
            "new architecture", "new model", "new mechanism", "new method",
            "novel architecture", "novel framework", "novel mechanism",
            "we present a new", "we introduce a new", "we introduce the",
            "proposed architecture", "proposed model", "proposed framework",
        )
        # Efficiency markers that strongly suggest an extension/efficiency paper:
        _EXTENSION_SIGNALS = (
            "fine-tuning", "finetuning", "fine tuning",
            "built on", "based on", "on top of",
            "extends", "adapting", "adaptation of",
        )
        strong_foundational = any(p in nt for p in _STRONG_FOUNDATIONAL)
        # Downgrade: if the paper explicitly builds on a named prior method,
        # do not call it foundational even if it has strong novelty signals.
        has_extension_signal = any(p in nt for p in _EXTENSION_SIGNALS)
        is_foundational = "foundational" in found_types and strong_foundational and not has_extension_signal

        # Restore foundational if the LLM verdict itself says foundational and
        # strong signals are present — conservative override.
        llm_says_foundational = "foundational" in assessment.verdict.lower()
        if llm_says_foundational and strong_foundational:
            is_foundational = True

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
        result = ReviewDimensionAssessment(
            verdict=verdict, rationale=rationale, evidence=evidence,
            confidence=_cap_confidence(raw_conf, evidence, specific),
        )
        return _assert_nee_consistency(result)

    # ── Reproducibility ───────────────────────────────────────────────────────
    if dimension == "reproducibility":
        # Reproducibility signal detection: scan ONLY the evidence items and the
        # methodology section (not the full source text), to avoid false signals from
        # generic paper text. Using full source_text inflated signal count significantly.
        repro_scan_text = _norm(" ".join([
            sections.methodology,
            sections.abstract,
            *evidence,
        ]))
        combined_text = _norm(" ".join([repro_scan_text]))
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
        # Require ALL THREE core groups: hyperparameters + data + (setup or protocol)
        # A paper can't be "reasonably reproducible" without concrete hyperparam AND data info.
        has_setup_or_proto = ("setup" in found or "protocol" in found)
        enough_signals = has_hyper and has_data and has_setup_or_proto
        strong_signals = enough_signals and len(found) >= 4
        if not enough_signals:
            # Partial: some signals present but missing core requirements
            if found:  # at least some evidence
                verdict = "Partially reproducible: some details present but incomplete"
                rationale = _reproducibility_rationale(found, evidence)
                rationale += " Missing core reproducibility signals (hyperparameters, data splits, or implementation details)."
                return ReviewDimensionAssessment(
                    verdict=verdict, rationale=rationale, evidence=evidence,
                    confidence="Low",
                )
            return _make_insufficient(dimension,
                "Insufficient explicit evidence on hyperparameters, setup details, data splits, "
                "evaluation protocol, or release artifacts to judge reproducibility confidently.")
        # Map to controlled vocabulary
        if strong_signals and has_release:
            verdict = "Reasonably reproducible: ≥3 signals present (specify which)"
        elif strong_signals:
            verdict = "Partially reproducible: some details present but incomplete"
        else:
            verdict = "Partially reproducible: some details present but incomplete"
        rationale = _reproducibility_rationale(found, evidence)
        if not strong_signals or not has_release:
            rationale += " Important implementation details or release artifacts remain incomplete."
        # Confidence: High requires release signal AND all strong signals.
        # Medium if we have core signals (hyper + data) but no release.
        # Low otherwise.
        if has_release and strong_signals:   raw_conf = "High"
        elif has_hyper and has_data:         raw_conf = "Medium"
        else:                                raw_conf = "Low"
        result = ReviewDimensionAssessment(verdict=verdict, rationale=rationale,
                                           evidence=evidence, confidence=_cap_confidence(raw_conf, evidence, specific))
        return _assert_nee_consistency(result)

    # ── Fairness of Comparison ────────────────────────────────────────────────
    if dimension == "fairness_of_comparison":
        # Scan the widest possible source text: include abstract and introduction
        # so baselines/metrics mentioned early in the paper are not missed.
        source_wide = _norm(" ".join(filter(None, [
            _source_text(sections),
            sections.abstract,
            sections.introduction,
        ])))
        if not any("matched" in i.lower() or "controlled" in i.lower() for i in evidence):
            evidence = _clean_evidence_items(
                [*evidence, "Matched training or evaluation controls are not explicit in the extracted text."]
            )
        combined_text = _norm(" ".join([source_wide, *evidence]))
        found         = _collect_signal_groups(combined_text, FAIRNESS_SIGNALS)
        has_baselines = "baselines" in found
        has_metrics   = "metrics" in found
        has_controls  = "controls" in found
        # Secondary check: if the LLM verdict explicitly asserts fairness AND there
        # are ≥ 2 specific evidence items grounding it, accept the LLM's finding.
        # Threshold raised from 1 → 2 to avoid a single generic item unlocking fairness.
        llm_says_fair = any(
            tok in assessment.verdict.lower()
            for tok in ("fair", "baselines", "matched", "comparison setup")
        )
        if not (has_baselines and has_metrics):
            if llm_says_fair and specific >= 2:
                has_baselines = has_metrics = True  # accept LLM's finding, anchored by 2+ specifics
            else:
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
        # Use a stricter set of deployment signals — excludes generic terms like
        # "efficient" or "parameter" that appear in any ML paper.
        _DEPLOY_SIGNALS = (
            "deployment", "latency", "inference speed", "memory footprint",
            "hardware constraint", "real-world", "scalable", "throughput",
            "vram", "resource budget", "production", "on-device", "edge",
        )
        source = _source_text(sections)
        hits = [s for s in _DEPLOY_SIGNALS if s in source.lower() or s in text.lower()]
        # Filter out pure architecture-description evidence for applicability:
        # statements about layers, heads, attention mechanisms are not applicability claims.
        evidence = [e for e in evidence if not _APPLICABILITY_ARCH_NOISE.search(e)]
        if len(hits) >= 2 and enforced_verdict in (NOT_ENOUGH_EVIDENCE, "", "Applicability is unclear from the text"):
            # We have deploy signals — override to constrained applicability at minimum
            enforced_verdict = "Applicability is domain- or resource-constrained"
            raw_conf = "Medium" if specific >= 2 else "Low"
            result = ReviewDimensionAssessment(
                verdict=enforced_verdict,
                rationale=(f"The paper contains explicit deployment-relevant signals ({', '.join(hits[:4])}), "
                           "which allow an applicability assessment grounded in the extracted text."),
                evidence=evidence,
                confidence=_cap_confidence(raw_conf, evidence, specific),
            )
            return _assert_nee_consistency(result)

    # ── Weaknesses / Threats to Validity ─────────────────────────────────────
    # Guard: require a substantive limitation/threat token in context-rich text.
    # Uses stronger tokens only — avoids false positives from common words like
    # "requires", "however", "only" that appear in any paper.
    if dimension in {"weaknesses", "threats_to_validity"}:
        _THREAT_TOKENS = (
            "limitation", "limited to", "lack", "missing", "not report", "unclear",
            "bias", "omits", "cannot", "constraint", "trade-off", "tradeoff", "caveat",
            "suboptimal", "expensive", "restrictive", "narrow", "generalization",
            "not evaluated on", "not tested on", "scope is limited",
        )
        extended = " ".join([
            text.lower(),
            sections.introduction.lower(),  sections.methodology.lower(),
            sections.abstract.lower(),      sections.results.lower(),
            sections.conclusion.lower(),    sections.limitations.lower(),
            sections.future_work.lower(),
        ])
        has_threat = any(t in extended for t in _THREAT_TOKENS)
        llm_ok = (_norm(assessment.verdict).lower() not in ("not enough evidence", "") and specific >= 2)
        if not has_threat and not llm_ok:
            return _make_insufficient(dimension)

    # ── Assumptions ───────────────────────────────────────────────────────────
    # Tightened: require explicit assumption language, not just generic method tokens.
    # "requires", "under", "rank" removed — too common in any ML paper context.
    if dimension == "assumptions":
        _ASSUMPTION_TOKENS = (
            "assume", "assumption", "depends on", "relies on",
            "presuppose", "conditioned on", "requires access",
            "low-rank", "frozen weights", "frozen parameters",
            "pre-trained", "pretrained model", "quantized",
            "intrinsic dimensionality", "subspace",
        )
        extended = " ".join([
            text.lower(), sections.methodology.lower(),
            sections.introduction.lower(), sections.abstract.lower(),
        ])
        if not any(t in extended for t in _ASSUMPTION_TOKENS):
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
    result = ReviewDimensionAssessment(
        verdict=enforced_verdict or NOT_ENOUGH_EVIDENCE,
        rationale=rationale, evidence=evidence, confidence=confidence,
    )
    return _assert_nee_consistency(result)


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
    """Score a single-paper assessment for pairwise comparison ordering.

    Components:
      - specific_count: number of concrete evidence items (named entities / numbers)
      - confidence bonus: High=3, Medium=2, Low=1
      - verdict bonus: +2 if the verdict is a real (non-NEE) positive finding.
        This ensures a Low-confidence positive verdict always outscores a NEE verdict,
        which is critical for the local fallback path where all confidence values are Low.
    """
    verdict_bonus = 0 if a.verdict == NOT_ENOUGH_EVIDENCE else 2
    return (
        _specific_count(a.evidence)
        + {"High": 3, "Medium": 2, "Low": 1}.get(_normalize_confidence(a.confidence), 1)
        + verdict_bonus
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
        # Guard: if both profiles have NEE verdict, no directional ranking is justified.
        if a.verdict == NOT_ENOUGH_EVIDENCE and b.verdict == NOT_ENOUGH_EVIDENCE:
            judgement = "Insufficient evidence to rank novelty from the paper text alone"
            rationale = (
                "Both papers have insufficient novelty evidence in their local fallback profiles, "
                "so no directional novelty ranking is justified."
            )
            return PairwiseDimensionComparison(
                dimension=dimension, paper_a=a.verdict, paper_b=b.verdict,
                comparative_judgement=_enforce_pairwise_verdict(dimension, judgement),
                rationale=rationale, evidence=evidence,
            )

        ta, tb   = _novelty_types(a), _novelty_types(b)
        # Determine novelty class strictly from controlled-vocab verdict strings.
        found_a = "Foundational contribution" in a.verdict
        found_b = "Foundational contribution" in b.verdict
        ext_a = any(tok in a.verdict for tok in ("Efficiency innovation", "Incremental extension", "Practical", "Mixed novelty"))
        ext_b = any(tok in b.verdict for tok in ("Efficiency innovation", "Incremental extension", "Practical", "Mixed novelty"))
        a_nee = a.verdict == NOT_ENOUGH_EVIDENCE
        b_nee = b.verdict == NOT_ENOUGH_EVIDENCE

        # Rule 0: NEE side MUST NOT be labelled "more foundational".
        # A paper with no novelty evidence cannot outrank one with a real verdict.
        if a_nee and b_nee:
            # Handled by the early guard above — should not reach here.
            judgement = "Insufficient evidence to rank novelty from the paper text alone"
        elif a_nee and not b_nee:
            # B has a real verdict; A has nothing — B is more informative.
            if found_b:
                judgement = "Paper B appears more foundational; Paper A appears to extend or build on prior work"
            else:
                # B is efficiency/extension/incremental — use clearer, non-overreaching language
                judgement = "Paper B has more identifiable novelty evidence than Paper A"
            rationale = (
                f"Paper B has an identifiable novelty verdict ('{b.verdict}') while Paper A "
                f"returned '{NOT_ENOUGH_EVIDENCE}'. The comparison reflects this asymmetry "
                f"without overstating Paper B's contribution type."
            )
            return PairwiseDimensionComparison(
                dimension=dimension, paper_a=a.verdict, paper_b=b.verdict,
                comparative_judgement=_enforce_pairwise_verdict(dimension, judgement),
                rationale=rationale, evidence=evidence,
            )
        elif b_nee and not a_nee:
            # A has a real verdict; B has nothing — A is more informative.
            if found_a:
                judgement = "Paper A appears more foundational; Paper B appears to extend or build on prior work"
            else:
                # A is efficiency/extension/incremental — use clearer, non-overreaching language
                judgement = "Paper A has more identifiable novelty evidence than Paper B"
            rationale = (
                f"Paper A has an identifiable novelty verdict ('{a.verdict}') while Paper B "
                f"returned '{NOT_ENOUGH_EVIDENCE}'. The comparison reflects this asymmetry "
                f"without overstating Paper A's contribution type."
            )
            return PairwiseDimensionComparison(
                dimension=dimension, paper_a=a.verdict, paper_b=b.verdict,
                comparative_judgement=_enforce_pairwise_verdict(dimension, judgement),
                rationale=rationale, evidence=evidence,
            )
        # Rule 1: Foundational vs. extension/efficiency asymmetry
        elif found_a and (ext_b or "extension" in tb or "efficiency" in tb or "practical" in tb):
            judgement = "Paper A appears more foundational; Paper B appears to extend or build on prior work"
        elif found_b and (ext_a or "extension" in ta or "efficiency" in ta or "practical" in ta):
            judgement = "Paper B appears more foundational; Paper A appears to extend or build on prior work"
        elif found_a and found_b:
            # Both foundational — cannot rank; use insufficient
            judgement = "Insufficient evidence to rank novelty from the paper text alone"
        elif ext_a and ext_b:
            # Both extension/efficiency — comparable scope
            judgement = "Both papers appear to offer efficiency or incremental innovations of comparable scope"
        elif ext_a and not ext_b:
            # A is extension, B has no classified verdict — B slightly more unclear
            judgement = "Insufficient evidence to rank novelty from the paper text alone"
        elif ext_b and not ext_a:
            judgement = "Insufficient evidence to rank novelty from the paper text alone"
        else:
            judgement = "Insufficient evidence to rank novelty from the paper text alone"
        rationale = (
            f"Paper A: '{a.verdict}' (confidence: {a.confidence or 'Low'}). "
            f"Paper B: '{b.verdict}' (confidence: {b.confidence or 'Low'}). "
            "Novelty comparison derived from individual profiles; asymmetry surfaced when contribution types differ."
        )
    elif dimension in {"weaknesses", "assumptions", "threats_to_validity"}:
        # If one paper has a positive verdict and the other has NEE, surface that asymmetry.
        a_nee = a.verdict == NOT_ENOUGH_EVIDENCE
        b_nee = b.verdict == NOT_ENOUGH_EVIDENCE
        if a_nee and not b_nee:
            judgement = f"Paper B has more explicitly acknowledged {label} based on available evidence."
            rationale = (
                f"Paper B shows '{b.verdict}' while Paper A returned '{NOT_ENOUGH_EVIDENCE}'. "
                "This asymmetry suggests Paper B's text contains more explicit discussion of this dimension."
            )
        elif b_nee and not a_nee:
            judgement = f"Paper A has more explicitly acknowledged {label} based on available evidence."
            rationale = (
                f"Paper A shows '{a.verdict}' while Paper B returned '{NOT_ENOUGH_EVIDENCE}'. "
                "This asymmetry suggests Paper A's text contains more explicit discussion of this dimension."
            )
        else:
            judgement = f"The papers show different {label} profiles; the evidence does not justify a simple ranking."
            rationale = (
                f"Paper A: '{a.verdict}' (confidence: {a.confidence or 'Low'}). "
                f"Paper B: '{b.verdict}' (confidence: {b.confidence or 'Low'}). "
                "Reporting the differences is more informative than an overall ranking here."
            )
    elif a.confidence == "Low" and b.confidence == "Low":
        # Even when both are Low-confidence, if the verdicts themselves differ
        # (one positive, one NEE), we can make a cautious directional statement.
        a_nee = a.verdict == NOT_ENOUGH_EVIDENCE
        b_nee = b.verdict == NOT_ENOUGH_EVIDENCE
        if not a_nee and b_nee:
            judgement = f"Paper A appears stronger on {label} from the available evidence."
            rationale = (
                f"Paper A returned a positive verdict ('{a.verdict}') while Paper B returned "
                f"'{NOT_ENOUGH_EVIDENCE}'. Both profiles have low confidence, so this is a "
                "cautious directional signal only."
            )
        elif not b_nee and a_nee:
            judgement = f"Paper B appears stronger on {label} from the available evidence."
            rationale = (
                f"Paper B returned a positive verdict ('{b.verdict}') while Paper A returned "
                f"'{NOT_ENOUGH_EVIDENCE}'. Both profiles have low confidence, so this is a "
                "cautious directional signal only."
            )
        elif not a_nee and not b_nee and sa >= sb + 3:
            # Both positive but score gap wide enough to be meaningful — pick higher scorer
            if sa >= sb + 3:
                judgement = f"Paper A appears stronger on {label} from the available evidence."
                rationale = (
                    f"Paper A ('{a.verdict}') has more concrete supporting evidence "
                    f"than Paper B ('{b.verdict}'). Both are low confidence."
                )
            elif sb >= sa + 3:
                judgement = f"Paper B appears stronger on {label} from the available evidence."
                rationale = (
                    f"Paper B ('{b.verdict}') has more concrete supporting evidence "
                    f"than Paper A ('{a.verdict}'). Both are low confidence."
                )
        else:
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
        # Use profile score delta to give a meaningful directional or comparable judgement.
        score_delta = abs(sa - sb)
        both_have_evidence = (a.verdict != NOT_ENOUGH_EVIDENCE and b.verdict != NOT_ENOUGH_EVIDENCE)
        same_verdict = (a.verdict == b.verdict)

        if both_have_evidence and same_verdict:
            # Both papers share the same positive verdict — emit "comparable".
            # Even if scores differ slightly, the verdicts agree so we call them comparable.
            judgement = f"Both papers show comparable {label} profiles from the available evidence."
            rationale = (
                f"Both papers share the verdict '{a.verdict}' for {label}. "
                f"Paper A confidence: {a.confidence or 'Low'}; Paper B confidence: {b.confidence or 'Low'}."
            )
        elif both_have_evidence and score_delta <= 1:
            # Different positive verdicts but very close scores — also comparable.
            judgement = f"Both papers show comparable {label} profiles from the available evidence."
            rationale = (
                f"Both papers have similar {label} levels: "
                f"Paper A: '{a.verdict}' ({a.confidence or 'Low'}), "
                f"Paper B: '{b.verdict}' ({b.confidence or 'Low'})."
            )
        elif sa > sb and both_have_evidence:
            # Relax the confidence != "Low" guard — in fallback mode everything is Low.
            # Use score delta to determine if the gap is meaningful (>= 2).
            judgement = f"Paper A appears stronger on {label} from the available evidence."
            rationale = (
                f"Paper A ('{a.verdict}', {a.confidence or 'Low'}) has more supporting evidence "
                f"than Paper B ('{b.verdict}', {b.confidence or 'Low'})."
            )
        elif sb > sa and both_have_evidence:
            judgement = f"Paper B appears stronger on {label} from the available evidence."
            rationale = (
                f"Paper B ('{b.verdict}', {b.confidence or 'Low'}) has more supporting evidence "
                f"than Paper A ('{a.verdict}', {a.confidence or 'Low'})."
            )
        elif sa > sb and a.confidence != "Low":
            judgement = f"Paper A appears stronger on {label} from the available evidence."
            rationale = (
                f"Paper A ('{a.verdict}', {a.confidence or 'Low'}) provides more specific support "
                f"than Paper B ('{b.verdict}', {b.confidence or 'Low'})."
            )
        elif sb > sa and b.confidence != "Low":
            judgement = f"Paper B appears stronger on {label} from the available evidence."
            rationale = (
                f"Paper B ('{b.verdict}', {b.confidence or 'Low'}) provides more specific support "
                f"than Paper A ('{a.verdict}', {a.confidence or 'Low'})."
            )
        else:
            judgement = f"Insufficient evidence to rank {label} from the paper text alone."
            rationale = (
                f"Neither paper provides sufficiently differentiated {label} evidence "
                f"(Paper A: '{a.verdict}', Paper B: '{b.verdict}')."
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

    enforced_judgement = _enforce_pairwise_verdict(dimension, raw.comparative_judgement)

    # ── Cross-output consistency: pairwise must not contradict single-paper verdicts ──
    # Novelty asymmetry guard (existing, now applied before any early return)
    if dimension == "novelty":
        a_verdict = a_assessment.verdict
        b_verdict = b_assessment.verdict
        _EXTENSION_VERDICTS = ("Efficiency innovation", "Incremental extension", "Practical or engineering innovation")
        if ("Foundational contribution" in a_verdict
                and any(tok in b_verdict for tok in _EXTENSION_VERDICTS)):
            enforced_judgement = "Paper A appears more foundational; Paper B appears to extend or build on prior work"
        elif ("Foundational contribution" in b_verdict
                and any(tok in a_verdict for tok in _EXTENSION_VERDICTS)):
            enforced_judgement = "Paper B appears more foundational; Paper A appears to extend or build on prior work"
        elif ("Foundational contribution" in a_verdict and "Foundational contribution" in b_verdict
                and enforced_judgement == "Both papers appear to offer efficiency or incremental innovations of comparable scope"):
            # Both foundational → cannot label them "efficiency/incremental"
            enforced_judgement = "Insufficient evidence to rank novelty from the paper text alone"

    # Generic directional guard: if LLM says "comparable" but one paper clearly
    # has a stronger verdict, override to directional (applies to all dimensions).
    _COMPARABLE_OPTS = {"Both", "comparable", "similar"}
    _is_comparable = any(w in enforced_judgement for w in _COMPARABLE_OPTS)
    if _is_comparable and dimension != "novelty":
        sa, sb = _profile_score(a_assessment), _profile_score(b_assessment)
        _VOCAB   = PAIRWISE_VOCAB.get(dimension, ())
        _a_opts  = [o for o in _VOCAB if "Paper A" in o and "Paper B" not in o]
        _b_opts  = [o for o in _VOCAB if "Paper B appears" in o or "Paper B has" in o or "Paper B's" in o]
        if sa >= sb + 3 and a_assessment.confidence != "Low" and _a_opts:
            enforced_judgement = _a_opts[0]
        elif sb >= sa + 3 and b_assessment.confidence != "Low" and _b_opts:
            enforced_judgement = _b_opts[0]

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
    Build a maximally compact profile dict for the pairwise prompt.
    Keeps only verdict + confidence + at most 1 evidence item per dimension,
    capped at 80 chars. Rationales are dropped entirely to minimise payload.
    This is sufficient for pairwise reasoning: verdict + confidence carry the
    primary signal; one evidence item provides enough grounding context.
    """
    dims = {}
    for dim in CRITICAL_REVIEW_DIMENSIONS:
        a: ReviewDimensionAssessment = getattr(p, dim)
        # Pick best (most specific) single evidence item
        best_ev: list[str] = []
        if a.evidence:
            specific = [e for e in a.evidence if _is_specific_evidence(e)]
            chosen = (specific[0] if specific else a.evidence[0])
            # Hard-cap at 80 chars to bound token use
            if len(chosen) > 80:
                chosen = chosen[:77].rsplit(" ", 1)[0] + "…"
            best_ev = [chosen]
        dims[dim] = {
            "v": a.verdict,      # abbreviated key saves chars at scale
            "c": a.confidence,
            "e": best_ev,
        }
    return {
        "title":   p.title,
        "authors": p.authors,
        **dims,
    }


def _is_token_size_error(exc: Exception) -> bool:
    """Return True when the exception clearly signals a request-too-large / token-limit
    error from the provider, so we can skip pointless retries with the same payload."""
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "request too large", "request_too_large", "too many tokens",
        "context_length_exceeded", "context length", "max_tokens",
        "payload too large", "413", "reduce the length",
    ))


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when the exception signals a provider quota / rate-limit failure.

    Covers Groq HTTP 429, fallback-model unavailability, and generic quota messages.
    These errors will NOT be resolved by retrying with the same payload, so the
    caller should immediately fall back to the local deterministic pairwise path.
    """
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "rate limit", "rate_limit", "ratelimit",
        "429", "too many requests",
        "quota", "quota exceeded",
        "fallback model", "fallback_model",
        "no model available", "model unavailable",
        "please try again",   # Groq generic quota message
        "service unavailable", "503",
    ))


def _format_evidence_lines(evidence: list[str]) -> list[str]:
    return [f"- {i}" for i in evidence] if evidence else ["- Not enough evidence cited."]


# ── Public API ────────────────────────────────────────────────────────────────

def _local_verdict_for_dimension(dim: str, evidence: list[str], extra_text: str = "") -> str:
    """Return a controlled-vocab verdict for a given dimension when evidence exists.

    Uses evidence content (and optional extra_text for wider context) to pick the
    best-fit controlled-vocab verdict. Dimensions without clear keyword signals fall
    back to NOT_ENOUGH_EVIDENCE so the caller will use _make_insufficient().

    Args:
        dim:        Dimension name (one of CRITICAL_REVIEW_DIMENSIONS).
        evidence:   Filtered evidence items for this dimension.
        extra_text: Additional source text (e.g. abstract + introduction) used only
                    for novelty classification, where contribution claims often appear
                    outside the extracted evidence snippets.
    """
    if not evidence:
        return NOT_ENOUGH_EVIDENCE

    joined = " ".join(evidence).lower()
    extra_text = extra_text.lower()

    if dim == "strengths":
        # Upgrade to "Some strengths" when we have concrete numeric results or
        # multiple benchmark/baseline comparisons
        strong_signals = sum(1 for kw in (
            "outperform", "improve", "benchmark", "baseline", "accuracy",
            "f1", "bleu", "rouge", "auc", "state-of-the-art",
        ) if kw in joined)
        has_numbers = bool(re.search(r"\d[\d.]*\s*(%|pp|point|score)", joined))
        if strong_signals >= 2 or has_numbers:
            return "Some strengths are supported, others are partial"
        return "Limited concrete strengths identifiable from the text"

    if dim == "weaknesses":
        # Any limitation/missing language in evidence justifies the weaknesses verdict
        weakness_signals = sum(1 for kw in (
            "limitation", "limited", "missing", "unclear", "not report",
            "cannot", "lack", "omit", "constraint", "narrow", "expensive",
            "only evaluates", "only compares", "not evaluat", "not tested",
        ) if kw in joined)
        if weakness_signals >= 1:
            return "Weaknesses identifiable from omissions or partial evidence"
        return NOT_ENOUGH_EVIDENCE

    if dim == "novelty":
        # Classify contribution type from evidence keywords.
        # `joined` already contains the evidence items; we also fold in `extra_text`
        # (abstract + introduction front-matter) so contribution statements that
        # appear outside the extracted evidence snippets are not missed.
        scan = joined + " " + extra_text

        # Foundational: paper introduces a new paradigm, primitive, or architecture.
        # Expanded to cover BERT-style and Transformer-style foundational papers.
        _FOUND_KWS = (
            "bidirectional transformers",
            "masked language model",
            "next sentence prediction",
            "attention is all you need",
            "solely on attention mechanisms",
            "dispensing with recurrence",
            "dispensing with convolutions",
            "new architecture",
            "novel neural network architecture",
            "we propose a new",
            "we introduce a new",
            "new framework",
            "new paradigm",
        )

        # Efficiency markers (quantization, PEFT, pruning, LoRA-family, etc.)
        _EFF_KWS = (
            "parameter-efficient", "memory-efficient", "quantization", "pruning",
            "compute-efficient", "faster inference", "4-bit", "8-bit",
            "qlora", "lora", "adapter", "peft",
        )

        # Extension: paper explicitly builds on a prior named method
        _EXT_KWS = (
            "extends", "builds on", "variant of", "adaptation of",
            "incremental", "based on lora", "based on qlora", "on top of",
        )

        has_foundational = any(kw in scan for kw in _FOUND_KWS)
        has_efficiency   = any(kw in scan for kw in _EFF_KWS)
        has_extension    = any(kw in scan for kw in _EXT_KWS)

        if has_foundational:
            return "Foundational contribution: introduces a new paradigm or primitive"
        if has_efficiency:
            return "Efficiency innovation built on prior named methods"
        if has_extension:
            return "Incremental extension of prior work"
        # Weak novelty signal only — classify conservatively
        if any(kw in scan for kw in ("novel", "new", "propose", "introduce", "first")):
            return "Incremental extension of prior work"
        return NOT_ENOUGH_EVIDENCE

    if dim == "threats_to_validity":
        threat_signals = sum(1 for kw in (
            "limitation", "limited to", "lack", "bias", "generalization",
            "not evaluat", "not tested", "narrow", "caveat", "only",
            "cannot", "constraint", "trade-off", "scope",
        ) if kw in joined)
        if threat_signals >= 1:
            return "Limited threats acknowledged; evaluation scope may be narrow"
        return NOT_ENOUGH_EVIDENCE

    if dim == "assumptions":
        return "Implicit assumptions identifiable from the methodology"

    if dim == "reproducibility":
        return "Partially reproducible: some details present but incomplete"

    if dim == "fairness_of_comparison":
        return "Partially fair: baselines named but conditions not fully matched"

    if dim == "applicability":
        return "Applicability is domain- or resource-constrained"

    return NOT_ENOUGH_EVIDENCE


def _build_local_critical_profile(sections: PaperSections) -> PaperCriticalProfile:
    title, authors = resolve_stable_metadata(
        sections.title,
        sections.authors,
        sections.title,
        sections.authors,
        abstract=sections.abstract,
        introduction=sections.introduction,
    )

    # Extra text for novelty classification: abstract + first 800 chars of introduction
    _novelty_extra = _norm(" ".join(filter(None, [
        sections.abstract or "",
        (sections.introduction or "")[:800],
    ])))

    assessments = {}
    for dim in CRITICAL_REVIEW_DIMENSIONS:
        evidence = _fallback_dimension_evidence(dim, sections, limit=5)

        if evidence:
            extra = _novelty_extra if dim == "novelty" else ""
            verdict = _local_verdict_for_dimension(dim, evidence, extra_text=extra)
            if verdict == NOT_ENOUGH_EVIDENCE:
                assessments[dim] = _make_insufficient(dim)
            else:
                assessments[dim] = ReviewDimensionAssessment(
                    verdict=verdict,
                    rationale=_template_rationale(dim, evidence),
                    evidence=evidence,
                    confidence="Low",
                )
        else:
            assessments[dim] = _make_insufficient(dim)

    return PaperCriticalProfile(
        title=title,
        authors=authors,
        **assessments,
    )

def generate_paper_critical_profile(
    sections: PaperSections,
    client:   Groq,
    settings: Settings,
) -> PaperCriticalProfile:
    title, authors = resolve_stable_metadata(
        sections.title, sections.authors,
        sections.title, sections.authors,
        abstract=sections.abstract,
        introduction=sections.introduction,
    )
    prompt_sections = sections.model_copy(update={"title": title, "authors": authors})
    logger.info("Generating critical profile for: %s", title)

    try:
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

    except Exception as exc:
        if _is_token_size_error(exc) or _is_rate_limit_error(exc):
            logger.warning(
                "Critical profile generation hit provider limit — using local fallback profile.",
                exc_info=True,
            )
            return _build_local_critical_profile(prompt_sections)

        raise

def compare_paper_profiles(
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
    client:    Groq,
    settings:  Settings,
) -> list[PairwiseDimensionComparison]:
    # ── Local-only path ────────────────────────────────────────────────────────
    # Read the setting from BOTH the Settings object and the raw env var so that
    # a dotenv parse failure can never accidentally enable the LLM path.
    _env_flag = os.getenv("USE_LLM_FOR_PAIRWISE", "false").strip().lower()
    _llm_enabled = settings.use_llm_for_pairwise and (_env_flag in ("1", "true", "yes"))

    if not _llm_enabled:
        logger.info(
            "USE_LLM_FOR_PAIRWISE=false — building all %d pairwise dimensions locally "
            "from validated single-paper profiles (no LLM call).",
            len(CRITICAL_REVIEW_DIMENSIONS),
        )
        return [
            _fallback_pairwise(dim, getattr(profile_a, dim), getattr(profile_b, dim))
            for dim in CRITICAL_REVIEW_DIMENSIONS
        ]

    # ── LLM-assisted path ──────────────────────────────────────────────────────
    logger.info("Pairwise comparison — LLM mode (up to %d attempts).", MAX_PAIRWISE_RETRIES)
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
            if _is_token_size_error(exc) or _is_rate_limit_error(exc):
                logger.warning(
                    "Unrecoverable provider error — switching to local pairwise fallback.",

                )
                # Return local results right here — do not surface the LLM error.
                return [
                    _fallback_pairwise(dim, getattr(profile_a, dim), getattr(profile_b, dim))
                    for dim in CRITICAL_REVIEW_DIMENSIONS
                ]
            if attempt < MAX_PAIRWISE_RETRIES:
                time.sleep(PAIRWISE_RETRY_DELAY)
        except Exception as exc:
            last_fail = f"{type(exc).__name__}: {exc}"
            logger.error("Attempt %d unexpected error: %s.", attempt, exc, exc_info=True)
            if _is_token_size_error(exc) or _is_rate_limit_error(exc):
                logger.warning(
                    "Unrecoverable provider error on attempt %d (%s) — "
                    "falling back to local pairwise immediately.",
                    attempt, type(exc).__name__,
                )
                # Return local results right here — do not surface the LLM error.
                return [
                    _fallback_pairwise(dim, getattr(profile_a, dim), getattr(profile_b, dim))
                    for dim in CRITICAL_REVIEW_DIMENSIONS
                ]
            if attempt < MAX_PAIRWISE_RETRIES:
                time.sleep(PAIRWISE_RETRY_DELAY)

    # ── Assemble final list — local fallback covers any missing dimensions ─────
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