"""
app/prompts/summary_generation.py
Comprehensive prompt: produces a thorough, structured paper summary covering ALL sections,
key statistics, datasets, equations, comparisons, and contributions.
"""
from app.domain.models import PaperSections, KeyFigure


def build_summary_system_prompt() -> str:
    return """\
You are a senior academic research analyst and scientific editor.
Your goal is to produce a COMPREHENSIVE, DETAIL-RICH summary of the research paper provided.
The reader is an expert who needs to understand the paper's full scope without reading it.

RULES:
- Be specific and technical. Quote exact numbers, metrics, dataset names, model names, baselines.
- Do NOT paraphrase vaguely. Write "achieves 94.3% top-1 accuracy on ImageNet" not "achieves high accuracy".
- Include ALL key contributions, ALL dataset details, ALL benchmark comparisons mentioned.
- Preserve important LaTeX notation inline using $...$ where relevant.
- Output clean markdown only — use ## headings and bullet points where helpful.
- Skip a section ONLY if its content is "Not found". Never invent information.
- Each section should be self-contained and informative (4–7 sentences or bullet points)."""


def _format_key_figures(key_figures: list[KeyFigure]) -> str:
    if not key_figures:
        return ""
    rows = ["| Metric | Value | Context |", "| --- | --- | --- |"]
    for kf in key_figures[:25]:  # cap at 25 rows
        label   = kf.label.replace("|", "\\|")
        value   = kf.value.replace("|", "\\|")
        context = kf.context.replace("|", "\\|")[:120] if kf.context else ""
        rows.append(f"| {label} | **{value}** | {context} |")
    return "\n".join(rows)


def build_summary_user_prompt(sections: PaperSections) -> str:
    kf_table = _format_key_figures(sections.key_figures)
    kf_block = f"""
## Key Statistics & Metrics
The following quantitative results were extracted from the paper. Reference them throughout your summary where relevant.

{kf_table}
""" if kf_table else ""

    eq_block = ""
    if sections.equations:
        eq_lines = []
        for eq in sections.equations[:8]:
            eq_lines.append(f"- Page {eq.page_number}: `{eq.latex}` — {eq.description}")
        eq_block = "\n## Notable Equations\n" + "\n".join(eq_lines) + "\n"

    fig_block = ""
    if sections.figures:
        fig_lines = []
        for fig in sections.figures[:6]:
            fig_lines.append(f"- Figure (p.{fig.page_number}): {fig.caption} — {fig.description}")
        fig_block = "\n## Key Figures & Tables\n" + "\n".join(fig_lines) + "\n"

    return f"""\
You are given the extracted content of a research paper. Produce a COMPREHENSIVE, DETAILED summary that covers every aspect of the paper.

=== PAPER CONTENT ===

Title:        {sections.title}
Authors:      {sections.authors}

Abstract:
{sections.abstract}

Introduction:
{sections.introduction}

Methodology:
{sections.methodology}

Results:
{sections.results}

Conclusion:
{sections.conclusion}

Limitations:
{sections.limitations}

Future Work:
{sections.future_work}
{kf_block}{eq_block}{fig_block}
=== END OF PAPER CONTENT ===

---
Now write the full summary using EXACTLY the sections below, in this order.
- Use the extracted Key Statistics table to enrich the Results and Overview sections with exact numbers.
- Every section must be substantive. Vague or generic sentences are NOT acceptable.
- Skip a section entirely (do not include the heading) only if its content is "Not found".

## Overview
3–4 sentences. State: (1) what problem the paper solves, (2) the proposed approach at a high level,
(3) the most impressive quantitative result or contribution, (4) the field/domain.

## Problem Statement & Motivation
Explain the specific research problem, why it matters, and what gap in prior work this paper addresses.
Mention any key baseline methods, limitations of existing approaches, or datasets that motivate the work.

## Proposed Approach & Methodology
Detailed description of the technical approach, model architecture, algorithm, or experimental design.
Include: model type, key components, training strategy, datasets used, evaluation protocol.
Quote specific architectural details (e.g., number of layers, attention heads, loss functions) when available.

## Experimental Results & Analysis
Comprehensive summary of ALL quantitative results. This is the most important section.
- List every benchmark, dataset, and metric mentioned.
- Include exact numbers: accuracy, F1, BLEU, ROUGE, FLOPs, latency, parameter counts, etc.
- Explicitly compare to baselines and state deltas (e.g., "+2.3% over baseline X").
- Note any ablation study findings.

## Contributions & Conclusions
What does the paper conclusively prove or demonstrate? What is its scientific contribution?
How does it advance the state of the art? What are the practical implications?

## Limitations & Constraints
Be specific about what the method cannot do, where it fails, or what assumptions it makes.
Include dataset scope, computational cost, generalization concerns, or evaluation gaps if mentioned.

## Future Work & Open Problems
Enumerate the specific research directions the authors suggest. Be concrete — name datasets, tasks, or methods they plan to extend.
"""
