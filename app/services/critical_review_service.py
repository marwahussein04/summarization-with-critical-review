"""
app/services/critical_review_service.py
Conservative critical-review profiling and pairwise comparison workflow.
"""
from __future__ import annotations

import json
import logging
import re

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

DIMENSION_LABELS = {
    "strengths": "Strengths",
    "weaknesses": "Weaknesses",
    "novelty": "Novelty",
    "assumptions": "Assumptions",
    "threats_to_validity": "Threats to Validity",
    "reproducibility": "Reproducibility",
    "fairness_of_comparison": "Fairness of Comparison",
    "applicability": "Applicability",
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
EVIDENCE_KEYWORDS = (
    "dataset", "benchmark", "baseline", "ablation", "metric", "accuracy", "f1",
    "bleu", "rouge", "auc", "map", "latency", "memory", "compute", "parameter",
    "batch size", "learning rate", "epoch", "optimizer", "split", "train",
    "validation", "test", "code", "checkpoint", "release", "quantization",
    "gpu", "task", "deployment", "compare", "evaluation", "limitation",
)
REPRO_SIGNALS = {
    "hyperparameters": ("learning rate", "batch size", "epoch", "optimizer", "seed"),
    "setup": ("architecture", "layer", "parameter", "implementation", "model", "quantization"),
    "data": ("dataset", "benchmark", "split", "train", "validation", "test"),
    "protocol": ("metric", "evaluation", "ablation", "baseline", "protocol"),
    "release": ("code", "checkpoint", "github", "repository", "release"),
}
FAIRNESS_SIGNALS = {
    "baselines": ("baseline", "compared", "comparison"),
    "tasks_or_datasets": ("dataset", "benchmark", "task", "split"),
    "metrics": ("metric", "accuracy", "f1", "bleu", "rouge", "auc", "map"),
    "controls": ("same setting", "same budget", "matched", "controlled", "protocol", "tuned"),
    "caveats": ("limitation", "caveat", "unclear", "not directly comparable", "different setting"),
}
NOVELTY_TYPES = {
    "foundational": ("foundational", "introduces", "new formulation", "new framework", "new paradigm", "first"),
    "extension": ("extends", "extension", "builds on", "variant", "adaptation", "incremental"),
    "efficiency": ("efficient", "efficiency", "parameter-efficient", "memory", "compute", "quantization", "faster"),
    "practical": ("practical", "engineering", "deployment", "scalable", "real-world"),
}
TITLE_REJECTION_PATTERNS = (
    re.compile(r"^(table|figure|fig\.?|appendix)\s*\d+[:.\s-]", re.I),
    re.compile(r"^(results|ablation|discussion|conclusion|references)\b", re.I),
)

DIMENSION_EVIDENCE_SOURCES = {
    "strengths": ("methodology", "results", "conclusion", "key_figures"),
    "weaknesses": ("limitations", "future_work", "results"),
    "novelty": ("introduction", "methodology", "conclusion", "key_figures"),
    "assumptions": ("methodology", "limitations"),
    "threats_to_validity": ("limitations", "results", "future_work"),
    "reproducibility": ("methodology", "results", "key_figures"),
    "fairness_of_comparison": ("results", "methodology", "key_figures"),
    "applicability": ("methodology", "results", "conclusion", "future_work", "key_figures"),
}

DIMENSION_KEYWORDS = {
    "strengths": ("improve", "outperform", "achieves", "reduce", "efficient", "robust", "benchmark", "dataset", "baseline"),
    "weaknesses": ("limitation", "limited", "only", "missing", "unclear", "future work", "not report"),
    "novelty": ("novel", "new", "introduces", "first", "efficient", "parameter-efficient", "quantization", "extension"),
    "assumptions": ("assume", "requires", "depends", "availability", "fixed", "pretrained"),
    "threats_to_validity": ("bias", "limited", "only", "ablation", "generalization", "scope", "benchmark"),
    "reproducibility": ("learning rate", "batch", "epoch", "optimizer", "dataset", "split", "metric", "code", "checkpoint"),
    "fairness_of_comparison": ("baseline", "compare", "comparison", "benchmark", "metric", "same", "matched"),
    "applicability": ("deployment", "efficient", "latency", "memory", "compute", "practical", "real-world", "scalable"),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_first_line(value: object, field_name: str) -> str:
    raw = str(value or "")
    lines = raw.splitlines()
    if not lines:
        logger.debug("Metadata normalization found empty line list for %s.", field_name)
        return ""
    return _norm(lines[0])


def _clean_bullet(text: str) -> str:
    return re.sub(r"^[\-\*\u2022]+\s*", "", _norm(text)).strip(" .")


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lower = _norm(text).lower()
    return any(phrase in lower for phrase in phrases)


def _is_raw_number_dump(text: str) -> bool:
    """Detect items that are just a long raw list of numbers (e.g. GLUE score dumps)."""
    # Count comma-separated numeric tokens
    tokens = [t.strip() for t in text.split(",")]
    numeric_tokens = [t for t in tokens if re.match(r'^\d[\d.±]*$', t)]
    # If more than 4 out of the tokens are bare numbers → it's a dump
    return len(tokens) > 4 and len(numeric_tokens) / len(tokens) > 0.5


def _clean_evidence_items(evidence: list[str], limit: int = 5) -> list[str]:
    seen: set[str] = set()
    seen_numeric_fingerprints: set[str] = set()
    items: list[str] = []
    for item in evidence:
        cleaned = _clean_bullet(item)
        # Strip pipe-formatted key_figure dumps (e.g. "Label: val | context | section")
        # Keep only the meaningful first segment when pipes make it unwieldy
        if "|" in cleaned:
            segments = [s.strip() for s in cleaned.split("|")]
            # Only keep if at least the first segment is meaningful and short enough
            first = segments[0].strip()
            if len(first) > 15:
                cleaned = first
            else:
                cleaned = " — ".join(s for s in segments[:2] if s)
        lowered = cleaned.lower()
        if not cleaned or len(cleaned) < 10 or lowered in seen or _contains_phrase(cleaned, GENERIC_PHRASES):
            continue
        # Cap individual items at 180 chars — longer items are usually sentence dumps
        if len(cleaned) > 180:
            cleaned = cleaned[:177].rsplit(" ", 1)[0] + "…"
            lowered = cleaned.lower()
        # Filter out raw number-list dumps (e.g. "87.6, 94.8, 90.2, 63.6...")
        if _is_raw_number_dump(cleaned):
            continue
        # Deduplicate key-figure style lines by leading numeric value
        if "elo" in lowered or "latency" in lowered or "perplexity" in lowered:
            num_match = re.search(r'\b(\d[\d,.]*)\b', lowered)
            fingerprint = num_match.group(1) if num_match else ""
            if fingerprint and fingerprint in seen_numeric_fingerprints:
                continue
            if fingerprint:
                seen_numeric_fingerprints.add(fingerprint)
        seen.add(lowered)
        items.append(cleaned)
    specific = [item for item in items if _is_specific_evidence(item)]
    return (specific or items)[:limit]


def _is_specific_evidence(item: str) -> bool:
    cleaned = _clean_bullet(item)
    lower = cleaned.lower()
    if not cleaned or _contains_phrase(cleaned, GENERIC_PHRASES):
        return False
    if re.search(r"\d", cleaned):
        return True
    if any(keyword in lower for keyword in EVIDENCE_KEYWORDS):
        return True
    return any(
        phrase in lower
        for phrase in (
            "not report",
            "not mention",
            "release not mentioned",
            "does not report",
            "does not mention",
            "only compares",
            "only evaluates",
            "limited to",
            "no code",
        )
    )


def _specific_count(evidence: list[str]) -> int:
    return sum(1 for item in evidence if _is_specific_evidence(item))


def _collect_signal_groups(text: str, signals: dict[str, tuple[str, ...]]) -> set[str]:
    lower = text.lower()
    return {label for label, keywords in signals.items() if any(keyword in lower for keyword in keywords)}


def _split_sentences(text: str) -> list[str]:
    if not text or text == NOT_FOUND:
        return []
    normalized = text.replace("\r", "\n")
    sentences: list[str] = []
    for block in normalized.split("\n"):
        block = _norm(block)
        if not block:
            continue
        sentences.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", block)
            if sentence.strip()
        )
    return sentences


def _score_candidate(sentence: str, keywords: tuple[str, ...]) -> int:
    lower = sentence.lower()
    score = len(re.findall(r"\d", sentence)) * 4
    score += sum(3 for keyword in keywords if keyword in lower)
    score += 2 if "%" in sentence else 0
    score += 2 if any(marker in lower for marker in ("not ", "only ", "missing", "limited", "unclear")) else 0
    score += 1 if len(sentence) > 40 else 0
    return score


def _fallback_dimension_evidence(dimension: str, sections: PaperSections, limit: int = 4) -> list[str]:
    keywords = DIMENSION_KEYWORDS.get(dimension, ())
    candidates: list[str] = []

    # Which key_figure sections are relevant for each review dimension
    _KF_SECTION_SCOPE: dict[str, set[str]] = {
        "strengths":              {"results", "abstract"},
        "weaknesses":             {"limitations"},
        "novelty":                {"abstract", "introduction"},
        "assumptions":            {"methodology"},
        "threats_to_validity":    {"limitations", "results"},
        "reproducibility":        {"methodology", "results"},
        "fairness_of_comparison": {"results"},
        "applicability":          {"results", "conclusion"},
    }

    for source in DIMENSION_EVIDENCE_SOURCES.get(dimension, ()):
        if source == "key_figures":
            allowed_sections = _KF_SECTION_SCOPE.get(dimension, set())
            for figure in sections.key_figures[:15]:
                fig_sec = figure.section.lower().strip()
                if allowed_sections and fig_sec not in allowed_sections:
                    continue
                text = " | ".join(
                    part for part in (f"{figure.label}: {figure.value}", figure.context, figure.section) if part
                )
                if text:
                    candidates.append(text)
            continue

        raw_text = getattr(sections, source, NOT_FOUND)
        candidates.extend(_split_sentences(raw_text))

    ranked = sorted(candidates, key=lambda sentence: _score_candidate(sentence, keywords), reverse=True)
    return _clean_evidence_items(ranked, limit=limit)


def _source_text(sections: PaperSections) -> str:
    parts = [
        sections.title,
        sections.authors,
        sections.abstract,
        sections.introduction,
        sections.methodology,
        sections.results,
        sections.conclusion,
        sections.limitations,
        sections.future_work,
    ]
    parts.extend(f"{item.label} {item.value} {item.context} {item.section}" for item in sections.key_figures)
    return _norm(" ".join(part for part in parts if part and part != NOT_FOUND))


def _normalize_confidence(value: str) -> str:
    lowered = _norm(value).lower()
    if lowered == "high":
        return "High"
    if lowered == "medium":
        return "Medium"
    return "Low"


def _sanitize_title(value: str) -> str:
    title = _safe_first_line(value, "title")
    if not title or title.lower() == NOT_FOUND.lower():
        return NOT_FOUND
    # Accept titles up to 300 chars (subtitles like "LoRA: Low-Rank Adaptation..." are valid)
    if len(title) < 8 or len(title) > 300:
        return NOT_FOUND
    if any(pattern.match(title) for pattern in TITLE_REJECTION_PATTERNS):
        return NOT_FOUND
    # Only reject on URL/email patterns, not on the word "doi" alone
    if any(token in title.lower() for token in ("doi.org", "http://", "https://", "arxiv.org", "@")):
        return NOT_FOUND
    if any(token in title.lower() for token in ("abstract", "introduction")):
        return NOT_FOUND
    return title


def _sanitize_authors(value: str) -> str:
    authors = _safe_first_line(value, "authors")
    if not authors or authors.lower() == NOT_FOUND.lower():
        return NOT_FOUND
    if len(authors) < 3 or len(authors) > 300:
        return NOT_FOUND
    if any(token in authors.lower() for token in ("abstract", "introduction", "doi", "http", "arxiv", "@")):
        return NOT_FOUND
    return authors


def resolve_stable_metadata(
    primary_title: str,
    primary_authors: str,
    fallback_title: str = "",
    fallback_authors: str = "",
) -> tuple[str, str]:
    title = _sanitize_title(primary_title)
    if title == NOT_FOUND:
        title = _sanitize_title(fallback_title)
    authors = _sanitize_authors(primary_authors)
    if authors == NOT_FOUND:
        authors = _sanitize_authors(fallback_authors)
    return title, authors


def _make_insufficient(dimension: str, reason: str | None = None) -> ReviewDimensionAssessment:
    label = DIMENSION_LABELS[dimension].lower()
    return ReviewDimensionAssessment(
        verdict=NOT_ENOUGH_EVIDENCE,
        rationale=reason or f"Insufficient explicit evidence in the paper text to assess {label} confidently.",
        evidence=[],
        confidence="Low",
    )


def _template_rationale(dimension: str, evidence: list[str]) -> str:
    if evidence:
        joined = "; ".join(evidence[:2])
        return f"The judgement is grounded in explicit evidence from the extracted paper text, including {joined}."
    return f"Insufficient explicit evidence in the paper text to assess {DIMENSION_LABELS[dimension].lower()} confidently."


def _normalize_assessment(
    dimension: str,
    assessment: ReviewDimensionAssessment,
    sections: PaperSections,
) -> ReviewDimensionAssessment:
    fallback_evidence = _fallback_dimension_evidence(dimension, sections)
    evidence = _clean_evidence_items([*assessment.evidence, *fallback_evidence], limit=6)
    specific = _specific_count(evidence)
    text = _norm(" ".join([assessment.verdict, assessment.rationale, *evidence]))
    confidence = _normalize_confidence(assessment.confidence)

    if not evidence or (_contains_phrase(text, GENERIC_PHRASES) and specific == 0):
        if dimension == "reproducibility":
            return _make_insufficient(
                dimension,
                "Insufficient explicit evidence on hyperparameters, setup details, data splits, evaluation protocol, or release artifacts to judge reproducibility confidently.",
            )
        if dimension == "fairness_of_comparison":
            return _make_insufficient(
                dimension,
                "Insufficient explicit evidence on matched datasets, metrics, baselines, or experimental controls to judge fairness confidently.",
            )
        return _make_insufficient(dimension)

    if dimension == "novelty":
        novelty_text = _norm(" ".join([text, _source_text(sections)])).lower()
        found_types = [name for name, keywords in NOVELTY_TYPES.items() if any(keyword in novelty_text for keyword in keywords)]
        if "foundational" in found_types and "efficiency" in found_types:
            verdict = "Appears to combine foundational and efficiency-oriented novelty"
        elif "foundational" in found_types:
            verdict = "Appears more foundational than purely incremental"
        elif "efficiency" in found_types and "extension" in found_types:
            verdict = "Appears to extend prior work mainly through efficiency innovation"
        elif "efficiency" in found_types:
            verdict = "Appears to contribute mainly through efficiency innovation"
        elif "practical" in found_types:
            verdict = "Appears to contribute mainly through practical or engineering innovation"
        elif "extension" in found_types:
            verdict = "Appears to be an incremental extension of prior work"
        else:
            verdict = "Novelty is supported only partially by the paper's own positioning"
        rationale = assessment.rationale
        if not rationale or _contains_phrase(rationale, GENERIC_PHRASES) or _contains_phrase(rationale, ASSERTIVE_PHRASES):
            rationale = "The paper supports a contribution claim, but the paper text alone does not justify a strong ranking of overall field impact."
        return ReviewDimensionAssessment(
            verdict=verdict,
            rationale=rationale,
            evidence=evidence,
            confidence="High" if specific >= 3 and "foundational" in found_types else "Medium",
        )

    if dimension == "reproducibility":
        source = _source_text(sections)
        if "release" not in source.lower() and not any("release" in item.lower() or "code" in item.lower() for item in evidence):
            evidence = _clean_evidence_items([*evidence, "Code or checkpoint release is not mentioned in the extracted text."])
        if "split" not in source.lower() and not any("split" in item.lower() for item in evidence):
            evidence = _clean_evidence_items([*evidence, "Train/validation/test split details are not explicit in the extracted text."])
        found = _collect_signal_groups(_norm(" ".join([source, *evidence])), REPRO_SIGNALS)
        if len(found) < 2:
            return _make_insufficient(
                dimension,
                "Insufficient explicit evidence on hyperparameters, setup details, data splits, evaluation protocol, or release artifacts to judge reproducibility confidently.",
            )
        verdict = "Reasonably reproducible from the paper text" if len(found) >= 4 else "Only partial reproducibility evidence is available"
        # Rebuild rationale from evidence to avoid contradiction with verdict
        rationale = _template_rationale(dimension, evidence)
        if verdict == "Only partial reproducibility evidence is available":
            rationale += " Important implementation details remain incomplete in the extracted text."
        # Tighten confidence: no code release + fewer than 4 signals → cap at Medium
        has_release = "release" in found
        raw_confidence = "High" if len(found) >= 5 else ("Medium" if len(found) >= 3 else "Low")
        if not has_release and raw_confidence == "High":
            raw_confidence = "Medium"
        # No hyperparams signal → cap at Low
        has_hyperparams = "hyperparameters" in found
        if not has_hyperparams and raw_confidence in ("High", "Medium") and len(found) < 4:
            raw_confidence = "Low"
        return ReviewDimensionAssessment(
            verdict=verdict,
            rationale=rationale,
            evidence=evidence,
            confidence=raw_confidence,
        )

    if dimension == "fairness_of_comparison":
        source = _source_text(sections)
        if "matched" not in source.lower() and not any("matched" in item.lower() or "controlled" in item.lower() for item in evidence):
            evidence = _clean_evidence_items([*evidence, "Matched training or evaluation controls are not explicit in the extracted text."])
        found = _collect_signal_groups(_norm(" ".join([source, *evidence])), FAIRNESS_SIGNALS)
        if len(found) < 2:
            return _make_insufficient(
                dimension,
                "Insufficient explicit evidence on matched datasets, metrics, baselines, or experimental controls to judge fairness confidently.",
            )
        verdict = "Comparison setup appears reasonably supported" if len(found) >= 4 and "controls" in found else "Only partial fairness evidence is available"
        # Rebuild rationale from evidence to avoid contradiction with verdict
        rationale = _template_rationale(dimension, evidence)
        if "controls" not in found:
            rationale += " The extracted text does not make matched experimental controls fully explicit."
        # Tighten confidence: no explicit controls → cap at Medium; also missing baselines → Low
        has_controls = "controls" in found
        has_baselines = "baselines" in found
        raw_confidence = "High" if len(found) >= 5 and has_controls else ("Medium" if len(found) >= 3 else "Low")
        if not has_controls and raw_confidence == "High":
            raw_confidence = "Medium"
        if not has_baselines and not has_controls and raw_confidence == "Medium":
            raw_confidence = "Low"
        return ReviewDimensionAssessment(
            verdict=verdict,
            rationale=rationale,
            evidence=evidence,
            confidence=raw_confidence,
        )

    if dimension == "applicability":
        source = _source_text(sections)
        deploy_signals = (
            "deployment", "efficient", "latency", "memory", "compute", "practical",
            "real-world", "scalable", "gpu", "parameter", "hardware", "inference",
            "resource", "overhead", "cost", "throughput", "vram", "quantization",
        )
        deploy_hits = [sig for sig in deploy_signals if sig in source.lower() or sig in text.lower()]
        if len(deploy_hits) >= 2:
            # Enough deployment evidence — rebuild a supported verdict rather than "Not enough evidence"
            if _norm(assessment.verdict).lower() in ("not enough evidence", NOT_ENOUGH_EVIDENCE.lower(), ""):
                deploy_summary = ", ".join(deploy_hits[:4])
                verdict = "Deployment and applicability evidence is present in the paper text"
                rationale = (
                    f"The paper contains explicit deployment-relevant signals ({deploy_summary}), "
                    "which allow an applicability assessment grounded in the extracted text."
                )
                return ReviewDimensionAssessment(
                    verdict=verdict,
                    rationale=rationale,
                    evidence=evidence,
                    confidence="Medium" if specific >= 2 else "Low",
                )

    if dimension in {"weaknesses", "threats_to_validity"}:
        # Broaden search: LoRA-style papers often hide limitations in intro/methodology
        implicit_weakness_tokens = (
            "limitation", "limited", "lack", "missing", "not report", "only", "unclear",
            "bias", "omits", "cannot", "does not", "may not", "however", "constraint",
            "requires", "depend", "overhead", "trade-off", "tradeoff", "caveat",
            "suboptimal", "challenging", "expensive", "restrictive",
        )
        # Check text (evidence + verdict + rationale) AND raw section text
        extended_text = text.lower() + " " + sections.introduction.lower() + " " + sections.methodology.lower()
        if not any(token in extended_text for token in implicit_weakness_tokens):
            return _make_insufficient(dimension)

    if dimension == "assumptions" and not any(token in text.lower() for token in ("assume", "requires", "depends on", "availability", "under")):
        return _make_insufficient(dimension)

    if confidence == "High" and specific < 3:
        confidence = "Medium"
    if confidence == "Medium" and specific < 2:
        confidence = "Low"

    rationale = assessment.rationale if assessment.rationale and not _contains_phrase(assessment.rationale, GENERIC_PHRASES) else _template_rationale(dimension, evidence)
    return ReviewDimensionAssessment(
        verdict=_norm(assessment.verdict) or NOT_ENOUGH_EVIDENCE,
        rationale=rationale,
        evidence=evidence,
        confidence=confidence,
    )


def _parse_json_response(raw: str, context: str) -> dict:
    cleaned = strip_code_fences(raw).strip()
    logger.debug("%s raw LLM response: %s", context, cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"pairwise_comparisons": parsed}
        if isinstance(parsed, dict):
            return parsed
        raise TypeError(f"Unexpected JSON payload type: {type(parsed).__name__}")
    except json.JSONDecodeError as exc:
        logger.error("%s JSON parse failed: %s", context, exc)
        raise CriticalReviewError("The AI returned invalid critical-review data. Please try again.", original=exc) from exc
    except TypeError as exc:
        logger.error("%s JSON payload type error: %s", context, exc)
        raise CriticalReviewError("The AI returned an unexpected critical-review payload type. Please try again.", original=exc) from exc


def _validate_profile(data: dict) -> PaperCriticalProfile:
    try:
        return PaperCriticalProfile.model_validate(data)
    except ValidationError as exc:
        logger.error("Critical profile validation failed: %s", exc)
        raise CriticalReviewError("The AI returned an incomplete critical-review profile. Please try again.", original=exc) from exc


def _ordered_pairwise(items: list[dict]) -> dict[str, PairwiseDimensionComparison]:
    result: dict[str, PairwiseDimensionComparison] = {}
    for item in items:
        normalized = _normalize_pairwise_item(item)
        if normalized is None:
            logger.warning("Skipping pairwise item with invalid or missing dimension: %s", item)
            continue
        try:
            comparison = PairwiseDimensionComparison.model_validate(normalized)
        except ValidationError as exc:
            logger.error("Pairwise comparison validation failed: %s | item=%s", exc, normalized)
            continue
        key = comparison.dimension.strip().lower()
        if key in CRITICAL_REVIEW_DIMENSIONS and key not in result:
            result[key] = comparison
        elif key in result:
            logger.warning("Duplicate pairwise dimension '%s' received; keeping first instance.", key)
    return result


def _profile_score(assessment: ReviewDimensionAssessment) -> int:
    return _specific_count(assessment.evidence) + {"High": 3, "Medium": 2, "Low": 1}.get(_normalize_confidence(assessment.confidence), 1)


def _novelty_types(assessment: ReviewDimensionAssessment) -> set[str]:
    text = _norm(" ".join([assessment.verdict, assessment.rationale, *assessment.evidence])).lower()
    return {label for label, keywords in NOVELTY_TYPES.items() if any(keyword in text for keyword in keywords)}


def _pairwise_evidence(a: ReviewDimensionAssessment, b: ReviewDimensionAssessment, raw: PairwiseDimensionComparison | None) -> list[str]:
    # Always use labeled "Paper A:" / "Paper B:" prefixes — skip unlabeled raw evidence
    # which is typically a duplicate of the per-paper evidence without the label prefix.
    evidence = [f"Paper A: {item}" for item in a.evidence[:2]]
    evidence.extend(f"Paper B: {item}" for item in b.evidence[:2])
    if raw:
        # Only include raw evidence items that are NOT already covered by a or b evidence
        a_texts = {_clean_bullet(item).lower() for item in a.evidence}
        b_texts = {_clean_bullet(item).lower() for item in b.evidence}
        for raw_item in raw.evidence[:2]:
            cleaned = _clean_bullet(raw_item).lower()
            if cleaned not in a_texts and cleaned not in b_texts:
                evidence.append(raw_item)
    return _clean_evidence_items(evidence, limit=6)


def _safe_pairwise_fallback(
    dimension: str,
    a: ReviewDimensionAssessment,
    b: ReviewDimensionAssessment,
    reason: str,
) -> PairwiseDimensionComparison:
    evidence = _clean_evidence_items(
        [f"Paper A: {item}" for item in a.evidence[:2]]
        + [f"Paper B: {item}" for item in b.evidence[:2]],
        limit=4,
    )
    return PairwiseDimensionComparison(
        dimension=dimension,
        paper_a=a.verdict or NOT_ENOUGH_EVIDENCE,
        paper_b=b.verdict or NOT_ENOUGH_EVIDENCE,
        comparative_judgement=PAIRWISE_FALLBACK_JUDGEMENT,
        rationale=(
            f"Pairwise synthesis was unavailable or incomplete for {DIMENSION_LABELS[dimension].lower()}. "
            f"{reason} Per-paper critical profiles were available, so this view surfaces those judgements conservatively."
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
    label = DIMENSION_LABELS[dimension].lower()
    score_a = _profile_score(a)
    score_b = _profile_score(b)

    if dimension == "novelty":
        types_a = _novelty_types(a)
        types_b = _novelty_types(b)
        enough_a = _specific_count(a.evidence) >= 2
        enough_b = _specific_count(b.evidence) >= 2
        if "foundational" in types_a and {"extension", "efficiency", "practical"} & types_b and enough_a and enough_b:
            judgement = "Paper A appears more foundational from the paper text, while Paper B appears more like an extension or practical-efficiency innovation."
        elif "foundational" in types_b and {"extension", "efficiency", "practical"} & types_a and enough_a and enough_b:
            judgement = "Paper B appears more foundational from the paper text, while Paper A appears more like an extension or practical-efficiency innovation."
        elif "efficiency" in types_a and "efficiency" not in types_b:
            judgement = "Paper A appears more explicitly focused on efficiency innovation from the available evidence."
        elif "efficiency" in types_b and "efficiency" not in types_a:
            judgement = "Paper B appears more explicitly focused on efficiency innovation from the available evidence."
        else:
            judgement = "Insufficient evidence to rank overall novelty or impact from the paper text alone."
        rationale = (
            f"Paper A is assessed as '{a.verdict}', while Paper B is assessed as '{b.verdict}'. "
            "Novelty can be multidimensional, so the comparison stays conservative unless each paper clearly signals a different contribution type."
        )
    elif dimension in {"weaknesses", "assumptions", "threats_to_validity"}:
        judgement = f"The papers show different {label} profiles, and the available evidence does not justify a simple overall ranking."
        rationale = f"Paper A is assessed as '{a.verdict}', while Paper B is assessed as '{b.verdict}'. Reporting differences may matter here."
    elif a.confidence == "Low" and b.confidence == "Low":
        judgement = f"Insufficient evidence to rank {label} overall from the paper text alone."
        rationale = f"Both papers have limited explicit evidence for {label}, so a stronger comparative claim would overstate the extracted text."
    elif score_a >= score_b + 2 and a.confidence != "Low":
        judgement = f"Paper A appears stronger on {label} from the available evidence."
        rationale = f"Paper A provides more specific support for {label} in the extracted text than Paper B."
    elif score_b >= score_a + 2 and b.confidence != "Low":
        judgement = f"Paper B appears stronger on {label} from the available evidence."
        rationale = f"Paper B provides more specific support for {label} in the extracted text than Paper A."
    else:
        judgement = f"The available evidence does not justify a strong ordering on {label}."
        rationale = f"Both papers have some support for {label}, but the evidence is too incomplete or balanced for a stronger claim."

    return PairwiseDimensionComparison(
        dimension=dimension,
        paper_a=a.verdict,
        paper_b=b.verdict,
        comparative_judgement=judgement,
        rationale=rationale,
        evidence=evidence,
    )


def _normalize_pairwise(
    dimension: str,
    raw: PairwiseDimensionComparison | None,
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
) -> PairwiseDimensionComparison:
    fallback = _fallback_pairwise(dimension, getattr(profile_a, dimension), getattr(profile_b, dimension), raw)
    if raw is None:
        return fallback
    raw_evidence = _clean_evidence_items(raw.evidence, limit=4)
    raw_text = _norm(" ".join([raw.paper_a, raw.paper_b, raw.comparative_judgement, raw.rationale, *raw_evidence]))
    if _contains_phrase(raw_text, GENERIC_PHRASES) or _contains_phrase(raw_text, ASSERTIVE_PHRASES) or _specific_count(raw_evidence) == 0:
        return fallback
    return PairwiseDimensionComparison(
        dimension=dimension,
        paper_a=fallback.paper_a,
        paper_b=fallback.paper_b,
        comparative_judgement=raw.comparative_judgement,
        rationale=raw.rationale or fallback.rationale,
        evidence=_clean_evidence_items([*fallback.evidence, *raw_evidence], limit=6),
    )


def _format_evidence_lines(evidence: list[str]) -> list[str]:
    return [f"- {item}" for item in evidence] if evidence else ["- Not enough evidence cited."]


def _coerce_string(value: object, fallback: str = "") -> str:
    text = _norm(str(value or ""))
    return text or fallback


def _coerce_evidence(value: object) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value if str(item).strip()]
    elif isinstance(value, str):
        raw_items = [part.strip() for part in re.split(r"[\n;]+", value) if part.strip()]
    else:
        raw_items = []
    return _clean_evidence_items(raw_items, limit=6)


def _normalize_pairwise_item(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None

    dimension = _coerce_string(item.get("dimension")).lower()
    if dimension not in CRITICAL_REVIEW_DIMENSIONS:
        return None

    return {
        "dimension": dimension,
        "paper_a": _coerce_string(item.get("paper_a"), NOT_ENOUGH_EVIDENCE),
        "paper_b": _coerce_string(item.get("paper_b"), NOT_ENOUGH_EVIDENCE),
        "comparative_judgement": _coerce_string(
            item.get("comparative_judgement"),
            PAIRWISE_FALLBACK_JUDGEMENT,
        ),
        "rationale": _coerce_string(
            item.get("rationale"),
            "The pairwise synthesis response was incomplete, so this comparison is kept conservative.",
        ),
        "evidence": _coerce_evidence(item.get("evidence")),
    }


def _extract_pairwise_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        direct = payload.get("pairwise_comparisons")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, dict)]
        for value in payload.values():
            if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                return [item for item in value if isinstance(item, dict)]
    return []


def generate_paper_critical_profile(sections: PaperSections, client: Groq, settings: Settings) -> PaperCriticalProfile:
    title, authors = resolve_stable_metadata(sections.title, sections.authors)
    prompt_sections = sections.model_copy(update={"title": title, "authors": authors})
    logger.info("Generating critical profile for paper: %s", title)
    raw = chat_completion(
        client=client,
        system_prompt=build_critical_profile_system_prompt(),
        user_prompt=build_critical_profile_user_prompt(prompt_sections),
        settings=settings,
        max_tokens=settings.max_tokens_critical_review,
    )
    profile = _validate_profile(_parse_json_response(raw, "Critical profile"))
    updates = {
        dimension: _normalize_assessment(dimension, getattr(profile, dimension), prompt_sections)
        for dimension in CRITICAL_REVIEW_DIMENSIONS
    }
    return profile.model_copy(update={"title": title, "authors": authors, **updates})


def compare_paper_profiles(
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
    client: Groq,
    settings: Settings,
) -> list[PairwiseDimensionComparison]:
    logger.info("Generating pairwise critical comparison.")
    raw_items: dict[str, PairwiseDimensionComparison] = {}
    failure_reason = ""

    try:
        logger.info("Pairwise comparison LLM call started.")
        raw = chat_completion(
            client=client,
            system_prompt=build_pairwise_comparison_system_prompt(),
            user_prompt=build_pairwise_comparison_user_prompt(profile_a, profile_b),
            settings=settings,
            max_tokens=settings.max_tokens_critical_review,
        )
        logger.debug("Pairwise comparison raw LLM response before parsing: %s", strip_code_fences(raw).strip())
        payload = _parse_json_response(raw, "Pairwise comparison")
        extracted_items = _extract_pairwise_items(payload)
        logger.info("Pairwise comparison payload yielded %d raw items.", len(extracted_items))
        raw_items = _ordered_pairwise(extracted_items)
        missing = [dimension for dimension in CRITICAL_REVIEW_DIMENSIONS if dimension not in raw_items]
        if missing:
            logger.warning("Pairwise comparison missing required dimensions: %s", ", ".join(missing))
            failure_reason = f"Missing dimensions: {', '.join(missing)}."
    except CriticalReviewError as exc:
        failure_reason = str(exc)
        logger.warning("Pairwise comparison generation failed; using conservative fallback comparisons.", exc_info=True)
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        logger.error("Unexpected pairwise comparison failure; using conservative fallback comparisons.", exc_info=True)

    comparisons: list[PairwiseDimensionComparison] = []
    for dimension in CRITICAL_REVIEW_DIMENSIONS:
        try:
            raw_item = raw_items.get(dimension)
            if raw_item is None and failure_reason:
                logger.warning(
                    "Pairwise comparison using fallback for missing dimension '%s'. Reason: %s",
                    dimension,
                    failure_reason,
                )
                comparisons.append(
                    _safe_pairwise_fallback(
                        dimension,
                        getattr(profile_a, dimension),
                        getattr(profile_b, dimension),
                        failure_reason,
                    )
                )
                continue
            comparisons.append(_normalize_pairwise(dimension, raw_item, profile_a, profile_b))
        except Exception as exc:
            logger.error(
                "Pairwise comparison assembly failed for dimension '%s': %s: %s",
                dimension,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            comparisons.append(
                _safe_pairwise_fallback(
                    dimension,
                    getattr(profile_a, dimension),
                    getattr(profile_b, dimension),
                    f"Dimension assembly failed with {type(exc).__name__}: {exc}.",
                )
            )

    return comparisons


def validate_critical_comparison_result(result: CriticalComparisonResult) -> None:
    if result.paper_a_profile is None or result.paper_b_profile is None:
        raise CriticalReviewError("Critical comparison result is missing one or both paper profiles.")
    if len(result.pairwise_comparisons) != len(CRITICAL_REVIEW_DIMENSIONS):
        raise CriticalReviewError(
            f"Critical comparison result must contain exactly {len(CRITICAL_REVIEW_DIMENSIONS)} pairwise comparisons."
        )
    present = {item.dimension for item in result.pairwise_comparisons}
    missing = [dimension for dimension in CRITICAL_REVIEW_DIMENSIONS if dimension not in present]
    if missing:
        logger.error("Critical comparison result missing dimensions at validation: %s", ", ".join(missing))
        raise CriticalReviewError(f"Critical comparison result is missing required dimensions: {', '.join(missing)}.")
    for item in result.pairwise_comparisons:
        if not item.dimension or not item.paper_a or not item.paper_b or not item.comparative_judgement or not item.rationale:
            raise CriticalReviewError(f"Critical comparison entry for '{item.dimension or 'unknown'}' is incomplete.")
        if not isinstance(item.evidence, list):
            raise CriticalReviewError(f"Critical comparison evidence for '{item.dimension}' is malformed.")


def build_critical_comparison_markdown(result: CriticalComparisonResult) -> str:
    try:
        validate_critical_comparison_result(result)
        lines = ["# Critical Comparison", ""]

        for label, profile in (("Paper A", result.paper_a_profile), ("Paper B", result.paper_b_profile)):
            lines.extend([f"## {label}", f"**Title:** {profile.title}", f"**Authors:** {profile.authors}", ""])
            for dimension in CRITICAL_REVIEW_DIMENSIONS:
                assessment = getattr(profile, dimension)
                lines.extend(
                    [
                        f"### {DIMENSION_LABELS[dimension]}",
                        f"**Verdict:** {assessment.verdict}",
                        f"**Confidence:** {assessment.confidence or 'Low'}",
                        assessment.rationale,
                        "**Evidence:**",
                        *_format_evidence_lines(assessment.evidence),
                        "",
                    ]
                )

        lines.extend(["## Direct Comparison", ""])
        for comparison in result.pairwise_comparisons:
            lines.extend(
                [
                    f"### {DIMENSION_LABELS.get(comparison.dimension, comparison.dimension.replace('_', ' ').title())}",
                    f"**Comparative Judgement:** {comparison.comparative_judgement}",
                    comparison.rationale,
                    "**Evidence:**",
                    *_format_evidence_lines(comparison.evidence),
                    "",
                ]
            )
        return "\n".join(lines).strip()
    except Exception as exc:
        logger.error("Critical comparison markdown assembly failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        if isinstance(exc, CriticalReviewError):
            raise
        raise CriticalReviewError("Failed to assemble the critical comparison result.", original=exc) from exc


def generate_critical_comparison(
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
    client: Groq,
    settings: Settings,
) -> CriticalComparisonResult:
    if profile_a is None or profile_b is None:
        raise CriticalReviewError("Cannot generate a critical comparison without both paper profiles.")
    comparisons = compare_paper_profiles(profile_a, profile_b, client, settings)
    result = CriticalComparisonResult(
        paper_a_profile=profile_a,
        paper_b_profile=profile_b,
        pairwise_comparisons=comparisons,
        comparison_markdown="",
    )
    validate_critical_comparison_result(result)
    comparison_markdown = build_critical_comparison_markdown(result)
    return result.model_copy(update={"comparison_markdown": comparison_markdown})