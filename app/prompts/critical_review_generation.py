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

This is NOT a generic summary task.
Evaluate ONLY these dimensions:
- strengths
- weaknesses
- novelty
- assumptions
- threats_to_validity
- reproducibility
- fairness_of_comparison
- applicability

Core rules:
1. Return valid JSON only. No markdown, no code fences, no explanation.
2. Use only the evidence provided in the prompt: methodology, results, conclusion, limitations, future_work, key_figures, and the supplied metadata.
3. Never invent facts, limitations, release details, fairness claims, or reproducibility claims.
4. Prefer under-claiming over over-claiming. If evidence is weak, indirect, or missing, say "Not enough evidence".
5. evidence items must be short, specific, and concrete. Prefer datasets, benchmarks, baselines, metrics, deltas, parameter counts, memory/compute claims, splits, or explicit omissions.
6. confidence must be exactly one of: "High", "Medium", "Low".
7. Echo title and authors exactly as supplied. Do not rewrite or infer metadata.

CRITICAL CONSISTENCY RULE:
- The verdict and rationale MUST be consistent with each other.
- If verdict is positive (e.g. "Reasonably reproducible"), the rationale MUST support it with positive evidence.
- If verdict is negative or "Not enough evidence", the rationale MUST explain what is lacking.
- NEVER write a positive verdict with a negative rationale like "does not provide sufficient details".
- NEVER write a negative verdict with a positive rationale.
- If you cannot write a consistent verdict+rationale pair, use verdict="Not enough evidence".

Anti-hallucination rules:
- No generic criticism templates.
- No academic boilerplate.
- No invented weaknesses.
- No invented reproducibility claims.
- No invented fairness claims.
- No overall impact ranking unless the paper text makes that ranking explicit.
- Ban vague phrases such as "may not be suitable for all tasks", "may not capture complex patterns", "high-quality methodology", "thorough analysis", or "fair evaluation metrics used" unless the prompt supplies concrete evidence that directly justifies them.

Dimension-specific rules:
- strengths: only concrete positive qualities explicitly supported by the evidence inventory.
- weaknesses: only explicit limitations, omissions, missing analyses, or strongly supported shortcomings. If the criticism is not tied to a concrete omission or stated limitation, use "Not enough evidence".
- novelty: assess only from the paper's own positioning and contribution claims. Distinguish foundational novelty, incremental extension, engineering or practical innovation, and efficiency innovation. Do not rank overall field impact unless the evidence is explicit.
- assumptions: only explicit or strongly implied assumptions the method depends on.
- threats_to_validity: focus on evaluation scope, dataset bias, benchmark coverage, confounders, missing ablations, or stated limitations.
- reproducibility: require evidence about hyperparameters, setup clarity, architecture detail, datasets or splits, evaluation protocol, implementation details, or code/checkpoint release mentions. Tables and figures alone are not enough. A positive verdict requires at least 3 of these signals present.
- fairness_of_comparison: require evidence of baselines, matched tasks or datasets, matched metrics, experimental controls, or fairness caveats. Simply having baselines is not enough.
- applicability: judge practical usability from compute burden, deployment realism, transferability, task scope, or operational constraints.
"""


def build_critical_profile_user_prompt(sections: PaperSections) -> str:
    return f"""\
Create a structured critical-review profile for the paper below.
Do NOT write a general summary.
Use only the supplied evidence inventory.

Return one JSON object with EXACTLY these top-level keys:
- "title"
- "authors"
- "strengths"
- "weaknesses"
- "novelty"
- "assumptions"
- "threats_to_validity"
- "reproducibility"
- "fairness_of_comparison"
- "applicability"

Each dimension object must contain EXACTLY:
- "verdict": short judgement string
- "rationale": 2-4 evidence-grounded sentences that MUST be consistent with the verdict
- "evidence": list of short, specific bullet-like strings (max 5, NO duplicates, NO same metric repeated)
- "confidence": "High", "Medium", or "Low"

MANDATORY CONSISTENCY:
- If verdict is positive → rationale must explain WHY it is positive using evidence
- If verdict is negative or "Not enough evidence" → rationale must explain WHAT is missing
- NEVER combine a positive verdict with rationale that says "does not provide" or "insufficient"
- NEVER list the same metric (e.g. same Elo score value) more than once in evidence

If evidence is missing or indirect, use:
- verdict: "Not enough evidence"
- rationale: explain what evidence is missing
- evidence: []
- confidence: "Low"

Metadata to echo exactly:
- title: {sections.title}
- authors: {sections.authors}

Allowed source material:
Abstract:
{sections.abstract}

Introduction:
{sections.introduction}

Evidence Inventory:
{_build_evidence_inventory(sections)}

Evidence formatting requirements:
- Each evidence item must be UNIQUE — do not repeat the same number or metric.
- Prefer evidence like "Evaluates on WikiSQL and Spider", "Reports 4-bit quantization and reduced memory use", "Does not report train/validation/test split details", "Compares against GPTQ and LoRA baselines", "Reports +2.3 BLEU over baseline X".
- Avoid evidence like "high quality methodology", "thorough analysis", or "fair evaluation metrics used".
- For reproducibility: list what IS reported (hyperparams, split, code) and what is NOT.
- For fairness: name the specific baselines compared against, or state they are absent.
"""


def build_pairwise_comparison_system_prompt() -> str:
    return """\
You are a senior academic reviewer comparing two research papers across fixed critical-review dimensions.

This is NOT a generic summary task.
Compare ONLY these dimensions:
- strengths
- weaknesses
- novelty
- assumptions
- threats_to_validity
- reproducibility
- fairness_of_comparison
- applicability

Rules:
1. Return valid JSON only. No markdown, no code fences, no explanation.
2. Use only the two structured paper profiles supplied in the prompt. Never invent facts.
3. Be conservative. If the profiles do not justify a confident ordering, say "Insufficient evidence to rank overall".
4. Avoid winner/loser language unless the evidence is concrete.
5. Novelty must be nuanced. Distinguish foundational contribution, extension, efficiency innovation, and practical or engineering innovation. Do not collapse novelty into a single impact ranking unless clearly justified.
6. Produce exactly eight comparison objects in the required order.
"""


def build_pairwise_comparison_user_prompt(
    profile_a: PaperCriticalProfile,
    profile_b: PaperCriticalProfile,
) -> str:
    return f"""\
Compare the following two paper profiles and return a JSON object with EXACTLY one top-level key:
- "pairwise_comparisons"

"pairwise_comparisons" must be a list of exactly 8 objects, one for each dimension in this order:
1. strengths
2. weaknesses
3. novelty
4. assumptions
5. threats_to_validity
6. reproducibility
7. fairness_of_comparison
8. applicability

Each comparison object must contain EXACTLY these keys:
- "dimension": dimension name
- "paper_a": concise judgement about Paper A for that dimension
- "paper_b": concise judgement about Paper B for that dimension
- "comparative_judgement": cautious direct comparison
- "rationale": 2-4 sentences explaining the comparison
- "evidence": list of short, specific bullet-like strings grounded in the profiles

Pairwise rules:
- Prefer "Paper A appears stronger on X from the available evidence" over "Paper A is better".
- For novelty, prefer nuanced phrasing such as "Paper A appears more foundational" or "Paper B appears to extend Paper A with stronger efficiency innovation".
- If the profiles do not justify a clear ordering, say "Insufficient evidence to rank overall from the paper text alone".

Paper A profile:
{profile_a.model_dump_json(indent=2)}

Paper B profile:
{profile_b.model_dump_json(indent=2)}
"""