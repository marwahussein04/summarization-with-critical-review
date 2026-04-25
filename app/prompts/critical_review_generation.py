"""
app/prompts/critical_review_generation.py
Prompts for conservative, evidence-grounded critical paper comparison.
"""
from __future__ import annotations

import re

from app.domain.models import PaperCriticalProfile, PaperSections

CRITICAL_REVIEW_DIMENSIONS = (
    "strengths",
    "weaknesses",
    "novelty",
    "assumptions",
    "threats_to_validity",
    "reproducibility",
    "fairness_of_comparison",
    "applicability",
)

# Per-dimension keyword sets used for evidence relevance scoring.
# Each dimension only scores snippets that match its own keyword profile.
_DIMENSION_EVIDENCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "strengths": (
        "outperform", "improve", "achieves", "reduce", "efficient",
        "robust", "benchmark", "dataset", "baseline", "accuracy",
        "f1", "bleu", "rouge", "auc", "metric", "state-of-the-art",
    ),
    "weaknesses": (
        "limitation", "limited", "only", "missing", "unclear",
        "future work", "not report", "not evaluate", "does not",
        "cannot", "lack", "omit", "constraint", "narrow", "caveat",
    ),
    "novelty": (
        "novel", "new", "propose", "introduce", "first", "paradigm",
        "architecture", "framework", "mechanism", "extends", "builds on",
        "parameter-efficient", "quantization", "extension", "contribution",
    ),
    "assumptions": (
        "assume", "assumption", "requires", "depends", "relies on",
        "availability", "fixed", "pretrained", "pre-trained", "rank",
        "low-rank", "frozen", "quantiz", "intrinsic", "subspace",
        "conditioned on", "presuppose",
    ),
    "threats_to_validity": (
        "limitation", "limited", "bias", "generalization", "scope",
        "not evaluat", "narrow", "caveat", "concern", "only",
        "cannot", "threat", "not tested", "ablation", "evaluation gap",
    ),
    "reproducibility": (
        "learning rate", "batch size", "epoch", "optimizer", "seed",
        "dataset", "split", "metric", "code", "checkpoint", "release",
        "implementation", "github", "repository", "train split",
        "test split", "validation split", "evaluation protocol",
        "hyperparameter", "hardware", "weight decay", "dropout",
    ),
    "fairness_of_comparison": (
        "baseline", "compare", "comparison", "benchmark", "metric",
        "same setting", "matched", "controlled", "accuracy", "f1",
        "bleu", "rouge", "auc", "evaluation", "against",
    ),
    "applicability": (
        "deployment", "latency", "inference", "memory", "compute",
        "practical", "real-world", "scalable", "throughput", "hardware",
        "resource", "vram", "edge", "on-device", "production",
    ),
}

# Kept for backward compat with _score_snippet (used in _format_section_evidence)
_EVIDENCE_HINT_KEYWORDS = tuple(
    kw for kws in _DIMENSION_EVIDENCE_KEYWORDS.values() for kw in kws
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_candidate_snippets(text: str) -> list[str]:
    if not text or text.strip() == "Not found":
        return []

    normalized = text.replace("\r", "\n")
    parts: list[str] = []
    for block in normalized.split("\n"):
        block = _normalize_text(block)
        if not block:
            continue
        parts.extend(
            snippet.strip()
            for snippet in re.split(r"(?<=[.!?])\s+", block)
            if snippet.strip()
        )
    return parts


def _score_snippet(snippet: str, dimension: str = "") -> int:
    lower = snippet.lower()
    # Use dimension-specific keywords when available; fall back to generic pool
    kws = _DIMENSION_EVIDENCE_KEYWORDS.get(dimension, _EVIDENCE_HINT_KEYWORDS)
    score = len(re.findall(r"\d", snippet)) * 5
    score += sum(3 for keyword in kws if keyword in lower)
    score += 2 if "%" in snippet else 0
    score += 2 if any(token in lower for token in ("not report", "not mention", "only", "missing", "limited")) else 0
    score += 1 if ":" in snippet else 0
    if len(snippet) < 25:
        score -= 2
    return score


def _format_section_evidence(label: str, text: str, limit: int = 4, dimension: str = "") -> str:
    snippets = _split_candidate_snippets(text)
    if not snippets:
        return f"{label}:\n- Not found"

    ranked = sorted(snippets, key=lambda s: _score_snippet(s, dimension), reverse=True)
    selected: list[str] = []
    seen: set[str] = set()

    for snippet in ranked:
        normalized = snippet.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(f"- {snippet}")
        if len(selected) >= limit:
            break

    if not selected:
        selected = [f"- {_normalize_text(text)[:220]}"]

    return f"{label}:\n" + "\n".join(selected)


def _format_key_figures(sections: PaperSections, limit: int = 15) -> str:
    if not sections.key_figures:
        return "- Not found"

    lines = []
    for figure in sections.key_figures[:limit]:
        parts = [f"{figure.label}: {figure.value}"]
        if figure.context:
            parts.append(figure.context)
        if figure.section:
            parts.append(f"section={figure.section}")
        lines.append(f"- {' | '.join(parts)}")
    return "\n".join(lines)


def _build_dimension_evidence_block(dimension: str, sections: PaperSections) -> str:
    """Build a dimension-specific evidence block by selecting only the sections
    most relevant to that dimension and scoring snippets with dimension-aware weights.

    Source routing per dimension (mirrors DIMENSION_EVIDENCE_SOURCES in the service):
      strengths              → methodology, results, conclusion
      weaknesses             → limitations, future_work, results
      novelty                → introduction, methodology, conclusion
      assumptions            → methodology, limitations, introduction
      threats_to_validity    → limitations, results, future_work
      reproducibility        → methodology, results
      fairness_of_comparison → results, methodology
      applicability          → methodology, results, conclusion, future_work
    """
    _SOURCES: dict[str, list[tuple[str, int]]] = {
        "strengths":              [("methodology", 3), ("results", 4), ("conclusion", 2)],
        "weaknesses":             [("limitations", 4), ("future_work", 3), ("results", 2)],
        "novelty":                [("introduction", 4), ("methodology", 3), ("conclusion", 2)],
        "assumptions":            [("methodology", 4), ("limitations", 3), ("introduction", 2)],
        "threats_to_validity":    [("limitations", 4), ("results", 2), ("future_work", 3)],
        "reproducibility":        [("methodology", 4), ("results", 3)],
        "fairness_of_comparison": [("results", 4), ("methodology", 3)],
        "applicability":          [("methodology", 3), ("results", 3), ("conclusion", 2), ("future_work", 2)],
    }
    sources = _SOURCES.get(dimension, [("methodology", 3), ("results", 3), ("conclusion", 2)])
    parts: list[str] = []
    for attr, limit in sources:
        text = getattr(sections, attr, None) or ""
        if text and text.strip() != "Not found":
            label = attr.replace("_", " ").title()
            parts.append(_format_section_evidence(label, text, limit=limit, dimension=dimension))
    # Always include key figures scoped to the dimension
    if sections.key_figures:
        _KF_SCOPE: dict[str, set[str]] = {
            "strengths":              {"results", "abstract"},
            "weaknesses":             {"limitations"},
            "novelty":                {"abstract", "introduction"},
            "assumptions":            {"methodology"},
            "threats_to_validity":    {"limitations", "results"},
            "reproducibility":        {"methodology", "results"},
            "fairness_of_comparison": {"results"},
            "applicability":          {"results", "conclusion"},
        }
        allowed_sections = _KF_SCOPE.get(dimension, set())
        kf_lines: list[str] = []
        for fig in sections.key_figures[:12]:
            fig_section = re.sub(r"\s+", " ", (fig.section or "")).strip().lower()
            if allowed_sections and not any(s in fig_section for s in allowed_sections):
                continue
            line_parts = [f"{fig.label}: {fig.value}"]
            if fig.context:
                line_parts.append(fig.context)
            kf_lines.append("- " + " | ".join(p for p in line_parts if p))
        if kf_lines:
            parts.append("Key Figures:\n" + "\n".join(kf_lines))
    return "\n\n".join(parts) if parts else f"No relevant evidence found for {dimension}."


def _build_evidence_inventory(sections: PaperSections) -> str:
    """Build a per-dimension evidence inventory.

    Each dimension section contains only the snippets drawn from the paper
    sections most relevant to that dimension, scored by dimension-aware keywords.
    This eliminates the flat shared-pool design where novelty could be supported
    by training schedules, or applicability by architecture descriptions.
    """
    CRITICAL_REVIEW_DIMENSIONS_LIST = (
        "strengths", "weaknesses", "novelty", "assumptions",
        "threats_to_validity", "reproducibility",
        "fairness_of_comparison", "applicability",
    )
    blocks: list[str] = []
    for dim in CRITICAL_REVIEW_DIMENSIONS_LIST:
        label = dim.replace("_", " ").upper()
        block = _build_dimension_evidence_block(dim, sections)
        blocks.append(f"── {label} ──\n{block}")
    return "\n\n".join(blocks)


def build_critical_profile_system_prompt() -> str:
    return """\
You are a senior research reviewer producing a structured CRITICAL REVIEW profile for one paper.
This is NOT a summary task — it is a targeted critical analysis grounded exclusively in the evidence supplied.

═══ VERDICT VOCABULARY (use ONLY these exact strings) ═══

strengths:
  "Multiple concrete strengths supported by evidence"
  "Some strengths are supported, others are partial"
  "Limited concrete strengths identifiable from the text"
  "Not enough evidence"

weaknesses:
  "Explicit limitations acknowledged by the authors"
  "Weaknesses identifiable from omissions or partial evidence"
  "No significant weaknesses are explicit in the text"
  "Not enough evidence"

novelty:
  "Foundational contribution: introduces a new paradigm or primitive"
  "Efficiency innovation built on prior named methods"
  "Incremental extension of prior work"
  "Practical or engineering innovation"
  "Mixed novelty: combines foundational and efficiency elements"
  "Not enough evidence"

assumptions:
  "Explicit assumptions stated in the paper"
  "Implicit assumptions identifiable from the methodology"
  "No clear assumptions identifiable from the text"
  "Not enough evidence"

threats_to_validity:
  "Multiple threats to validity are explicitly stated"
  "Limited threats acknowledged; evaluation scope may be narrow"
  "No threats to validity explicitly discussed"
  "Not enough evidence"

reproducibility:
  "Reasonably reproducible: ≥3 signals present (specify which)"
  "Partially reproducible: some details present but incomplete"
  "Reproducibility is limited: key details are absent"
  "Not enough evidence"

fairness_of_comparison:
  "Comparison setup appears fair: named baselines and matched metrics present"
  "Partially fair: baselines named but conditions not fully matched"
  "Fairness cannot be confirmed: baselines or metrics insufficiently specified"
  "Not enough evidence"

applicability:
  "Broad applicability suggested by evidence"
  "Applicability is domain- or resource-constrained"
  "Applicability is unclear from the text"
  "Not enough evidence"

═══ CORE RULES ═══
1. Return valid JSON only. No markdown, no code fences, no explanation.
2. Use ONLY the evidence inventory supplied in the prompt. Never invent facts, metrics, or claims.
3. For each dimension, pick the SINGLE BEST verdict string from the list above. Do not rephrase it.
4. confidence must be exactly one of: "High", "Medium", "Low".
   - "High" requires: ≥3 specific evidence items (each citing a named dataset, benchmark,
     metric value, baseline, or explicit parameter), AND no evidence item is a generic phrase.
   - "Medium" requires: 1-2 specific items OR the verdict is well-supported but code/data
     details are incomplete.
   - "Low" for: verdicts of "Not enough evidence", single vague evidence items,
     or when the evidence does not directly justify the verdict text.
5. Echo title and authors EXACTLY as supplied. Never rewrite, infer, or leave blank.
6. CONTRADICTION RULE: Before finalising evidence for any dimension, scan ALL evidence items in that dimension for logical contradictions. If one item says a task/dataset was NOT evaluated and another item cites a metric FROM that same task/dataset, remove the contradicting omission claim. Metric evidence takes precedence over the omission claim.

═══ CONSISTENCY RULE (mandatory — violation invalidates the whole response) ═══
- Positive verdict → rationale MUST use positive language grounded in specific evidence. FORBIDDEN: "does not provide", "insufficient", "lacks", "not mentioned".
- Negative verdict or "Not enough evidence" → rationale MUST explain what is missing. FORBIDDEN: positive phrasing.
- If you cannot write a consistent pair: use "Not enough evidence" with an honest rationale.

═══ EVIDENCE QUALITY RULES ═══
- Each evidence item must cite at least ONE of: a named dataset, benchmark, baseline name, metric name, hyperparameter value, split detail, code/checkpoint reference, or an explicit omission claim.
- FORBIDDEN: generic digits without context (e.g. "results show 87.6"), vague phrases ("high-quality methodology", "thorough analysis", "fair evaluation metrics used"), or architecture descriptions used as applicability or threat evidence.
- No duplicated metric values across items in the same dimension. Maximum 5 items per dimension.
- EVIDENCE LEAKAGE: Do NOT use architecture descriptions (layers, heads, embeddings, dot-product attention) as evidence for applicability, threats, or fairness. Only use them for strengths or novelty if they are central to the contribution.

═══ EVIDENCE ROUTING (mandatory) ═══
The evidence inventory is organised into labelled per-dimension blocks.
Each dimension MUST be assessed using ONLY the evidence block that matches it.
DO NOT carry evidence across dimensions (e.g. do not use a reproducibility snippet to support novelty).
If the matching block contains no usable evidence, verdict must be "Not enough evidence".

═══ DIMENSION-SPECIFIC RULES ═══
- novelty: Distinguish contribution TYPE carefully.
  * "Foundational" = the paper introduces a new paradigm, primitive, or architecture from scratch. Signals: "we propose the X", "we introduce the X", "novel architecture/framework/mechanism", "new formulation/paradigm".
  * "Efficiency innovation" = existing method made faster/smaller. Signals: "parameter-efficient", "quantization", "pruning", "faster inference".
  * "Incremental extension" = variant/adaptation/fine-tuning of prior work. Signals: "extends", "builds on", "fine-tuning", "finetuning".
  * "Practical" = systems/deployment contribution. Signals: "deployment", "real-world application", "engineering system".
  * Do NOT use generic words like "introduces" or "first" alone — check if the contribution derives from a prior named method.
  * Do NOT collapse foundational vs. extension into "comparable" or "incremental".
- reproducibility: "Reasonably reproducible" requires ≥3 of these signals EXPLICITLY present:
  (1) hyperparameters: learning rate, batch size, epoch, optimizer, seed, weight decay, dropout, warmup
  (2) setup: implementation details, training setup, quantization scheme, hardware setup, number of layers, hidden size
  (3) data: named dataset, train/validation/test split reference, dev set
  (4) protocol: evaluation protocol, ablation study, baseline comparison
  (5) release: code, checkpoint, GitHub, repository, open-source
  Generic mentions of "the model" or "the architecture" do NOT count. List which signals are present and which are absent.
  Absent code/checkpoint = confidence ≤ "Medium".
- fairness_of_comparison: requires a NAMED baseline AND a NAMED metric. Just citing results or mentioning "baseline" without naming it is NOT sufficient. "Not enough evidence" only when no baselines AND no metrics are named. Otherwise use "Partially fair" at minimum.
- threats_to_validity: only use explicit limitation/threat statements from the paper. NEVER use architecture descriptions as threats.
- applicability: only use deployment-relevant signals (memory, latency, compute, hardware, real-world use, scalability). NEVER use architecture design details.
- assumptions: only explicit or strongly implied assumptions: "assume", "relies on", "depends on", "requires access", "frozen weights", "low-rank", "pre-trained", "intrinsic dimensionality".
"""


def build_critical_profile_user_prompt(sections: PaperSections) -> str:
    return f"""\
Produce a critical-review profile for the paper below.
Use ONLY the evidence inventory. Do NOT summarize the paper.

═══ OUTPUT FORMAT ═══
Return one JSON object with EXACTLY these top-level keys:
  "title", "authors",
  "strengths", "weaknesses", "novelty", "assumptions",
  "threats_to_validity", "reproducibility", "fairness_of_comparison", "applicability"

Each dimension object must contain EXACTLY:
  "verdict":    one string chosen verbatim from the VERDICT VOCABULARY in the system prompt
  "rationale":  2-4 sentences consistent with the verdict (positive ↔ positive evidence; negative ↔ missing evidence)
  "evidence":   list of 1-5 short, specific, UNIQUE items (no raw number dumps, no generic phrases)
  "confidence": "High", "Medium", or "Low"

═══ METADATA ═══
Title:   {sections.title}
Authors: {sections.authors}
Echo both fields exactly as shown above.

═══ PAPER CONTENT ═══
Abstract:
{sections.abstract}

Introduction (first 600 chars):
{(sections.introduction or "Not found")[:600]}

═══ EVIDENCE INVENTORY (per-dimension — use ONLY the block matching each dimension) ═══
{_build_evidence_inventory(sections)}

═══ EVIDENCE EXAMPLES ═══
GOOD:
  "Evaluates on MMLU across 63 subjects"
  "Compares against GPTQ and LoRA with matched metrics"
  "4-bit NF4 quantization reduces memory to 48GB for 65B model"
  "No code or checkpoint release mentioned"
  "Train/validation/test split details not specified"
  "Learning rate and batch size reported as 3e-4 and 16 respectively"

BAD (never use):
  "87.6, 94.8, 90.2, 63.6..." (raw number dump)
  "high quality methodology"
  "thorough analysis"
  "may not be suitable for all tasks"
  Repeating the same metric twice
"""


def build_pairwise_comparison_system_prompt() -> str:
    return """\
You are a senior academic reviewer comparing two research papers across eight fixed dimensions.
This is NOT a summary task — it is a targeted pairwise comparison grounded exclusively in the two profiles supplied.

═══ COMPARATIVE JUDGEMENT VOCABULARY ═══
Use ONLY these exact strings for comparative_judgement:

strengths:
  "Paper A appears stronger on strengths from the available evidence"
  "Paper B appears stronger on strengths from the available evidence"
  "Both papers show comparable strengths; no clear ordering is supported"
  "Insufficient evidence to compare strengths"

weaknesses:
  "Paper A has more explicitly acknowledged weaknesses"
  "Paper B has more explicitly acknowledged weaknesses"
  "Both papers show different weakness profiles; no ranking is appropriate"
  "Insufficient evidence to compare weaknesses"

novelty:
  "Paper A appears more foundational; Paper B appears to extend or build on prior work"
  "Paper B appears more foundational; Paper A appears to extend or build on prior work"
  "Both papers appear to offer efficiency or incremental innovations of comparable scope"
  "Insufficient evidence to rank novelty from the paper text alone"

assumptions:
  "Paper A carries more explicit or restrictive assumptions"
  "Paper B carries more explicit or restrictive assumptions"
  "Both papers have comparable assumption profiles"
  "Insufficient evidence to compare assumptions"

threats_to_validity:
  "Paper A has more explicit threats to validity or evaluation gaps"
  "Paper B has more explicit threats to validity or evaluation gaps"
  "Both papers have similar threat profiles"
  "Insufficient evidence to compare threats to validity"

reproducibility:
  "Paper A appears more reproducible based on available signals"
  "Paper B appears more reproducible based on available signals"
  "Both papers have comparable reproducibility levels"
  "Insufficient evidence to compare reproducibility"

fairness_of_comparison:
  "Paper A's comparison setup appears better supported"
  "Paper B's comparison setup appears better supported"
  "Both papers have comparable comparison setups"
  "Insufficient evidence to compare fairness of comparison"

applicability:
  "Paper A appears more broadly applicable based on the evidence"
  "Paper B appears more broadly applicable based on the evidence"
  "Both papers have comparable applicability scope"
  "Insufficient evidence to compare applicability"

═══ RULES ═══
1. Return valid JSON only. No markdown, no code fences.
2. Use ONLY the two profiles supplied. Never invent facts.
3. Pick ONE verdict string verbatim from the vocabulary above.
4. Produce exactly 8 comparison objects in the required order.
5. All evidence items must be prefixed "Paper A:" or "Paper B:". No unlabeled items.
6. No raw number dumps. Each evidence item must be a complete phrase. Maximum 4 items.
7. For novelty: do NOT label a paper "foundational" unless its profile explicitly supports it. If uncertain, use the "comparable" or "insufficient evidence" option.
8. COMPARABLE PROHIBITION: Do NOT use a "comparable" or "both papers" judgement unless the individual verdicts are genuinely aligned and the evidence support is similarly strong on both sides. If one paper is more specific, more strongly supported, or more conservative than the other, pick a directional judgement or an insufficient-evidence judgement instead.
9. FLATTENING PROHIBITION: Do NOT collapse a foundational-vs-incremental distinction into a generic "comparable" judgement. If one paper is supported as foundational and the other is supported as an extension, efficiency innovation, or practical engineering advance, reflect that asymmetry directly in the comparative judgement.
"""


def build_pairwise_comparison_user_prompt(
    profile_a: "PaperCriticalProfile | dict",
    profile_b: "PaperCriticalProfile | dict",
) -> str:
    import json as _json
    # Use compact JSON (no indent) to minimise token count.
    # Profile dicts use abbreviated keys: v=verdict, c=confidence, e=evidence (1 item max).
    a_json = _json.dumps(profile_a, separators=(",", ":")) if isinstance(profile_a, dict) else _json.dumps(profile_a.model_dump(), separators=(",", ":"))
    b_json = _json.dumps(profile_b, separators=(",", ":")) if isinstance(profile_b, dict) else _json.dumps(profile_b.model_dump(), separators=(",", ":"))
    return f"""\
Compare these two paper profiles and return a JSON object with EXACTLY one top-level key:
  "pairwise_comparisons"

"pairwise_comparisons" must be a list of exactly 8 objects, one per dimension in this order:
  1. strengths  2. weaknesses  3. novelty  4. assumptions
  5. threats_to_validity  6. reproducibility  7. fairness_of_comparison  8. applicability

Each object must contain EXACTLY:
  "dimension":             dimension name (lowercase, underscore)
  "paper_a":               the verdict string from Paper A's profile for this dimension
  "paper_b":               the verdict string from Paper B's profile for this dimension
  "comparative_judgement": ONE string chosen verbatim from the COMPARATIVE JUDGEMENT VOCABULARY
  "rationale":             2-4 sentences grounded in the profiles; no invented facts
  "evidence":              list of specific items, EACH prefixed "Paper A:" or "Paper B:"

Profile key legend: v=verdict, c=confidence, e=evidence (best single item per dimension).

Paper A profile:
{a_json}

Paper B profile:
{b_json}
"""