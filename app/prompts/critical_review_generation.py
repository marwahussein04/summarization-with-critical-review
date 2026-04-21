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

_EVIDENCE_HINT_KEYWORDS = (
    "dataset",
    "benchmark",
    "baseline",
    "ablation",
    "metric",
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "auc",
    "map",
    "latency",
    "memory",
    "compute",
    "parameter",
    "hyperparameter",
    "learning rate",
    "batch size",
    "epoch",
    "optimizer",
    "split",
    "train",
    "validation",
    "test",
    "code",
    "checkpoint",
    "release",
    "limitation",
    "compare",
    "evaluation",
    "quantization",
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


def _score_snippet(snippet: str) -> int:
    lower = snippet.lower()
    score = len(re.findall(r"\d", snippet)) * 5
    score += sum(3 for keyword in _EVIDENCE_HINT_KEYWORDS if keyword in lower)
    score += 2 if "%" in snippet else 0
    score += 2 if any(token in lower for token in ("not report", "not mention", "only", "missing", "limited")) else 0
    score += 1 if ":" in snippet else 0
    if len(snippet) < 25:
        score -= 2
    return score


def _format_section_evidence(label: str, text: str, limit: int = 4) -> str:
    snippets = _split_candidate_snippets(text)
    if not snippets:
        return f"{label}:\n- Not found"

    ranked = sorted(snippets, key=_score_snippet, reverse=True)
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


def _build_evidence_inventory(sections: PaperSections) -> str:
    blocks = [
        _format_section_evidence("Methodology Evidence", sections.methodology),
        _format_section_evidence("Results Evidence", sections.results),
        _format_section_evidence("Conclusion Evidence", sections.conclusion, limit=3),
        _format_section_evidence("Limitations Evidence", sections.limitations, limit=3),
        _format_section_evidence("Future Work Evidence", sections.future_work, limit=3),
        "Key Numerical Evidence:\n" + _format_key_figures(sections),
    ]
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
- Each evidence item: a complete phrase citing a specific dataset, benchmark, metric value, baseline name, parameter count, or explicit omission.
- FORBIDDEN: raw number dumps like "87.6, 94.8, 90.2", vague phrases like "high-quality methodology", "thorough analysis", "fair evaluation metrics used".
- No duplicated metric values across items. Maximum 5 items per dimension.

═══ DIMENSION-SPECIFIC RULES ═══
- novelty: "Foundational" requires the paper to introduce a new paradigm/primitive NOT derived from prior named methods. A paper adding techniques on top of an existing method (LoRA, BERT, etc.) is an efficiency or incremental extension, not foundational. When uncertain, default to the more conservative verdict.
- reproducibility: "Reasonably reproducible" requires ≥3 of these signals EXPLICITLY present: (1) hyperparameters, (2) architecture/model detail, (3) dataset/splits, (4) evaluation protocol, (5) code/checkpoint release. List which are present and which are absent. Absent code/checkpoint = confidence ≤ "Medium".
- fairness_of_comparison: requires named baselines AND matched metrics. Results alone are insufficient.
- threats_to_validity: only state threats EXPLICITLY in the text. No speculation.
- assumptions: only explicit or strongly implied assumptions the method depends on.
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

═══ EVIDENCE INVENTORY ═══
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
    a_json = _json.dumps(profile_a, indent=2) if isinstance(profile_a, dict) else profile_a.model_dump_json(indent=2)
    b_json = _json.dumps(profile_b, indent=2) if isinstance(profile_b, dict) else profile_b.model_dump_json(indent=2)
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

Paper A profile:
{a_json}

Paper B profile:
{b_json}
"""