"""
app/prompts/section_extraction.py

Upgraded section extraction prompt:
- Extracts key_figures: all concrete numeric results (accuracy, dataset size,
  parameters, p-values, BLEU scores, FLOPs, latency, etc.)
- Accurate section content with technical precision
- Structured JSON with strict schema enforcement
"""


def build_section_extraction_system_prompt() -> str:
    return """\
You are an expert academic paper parser and data extractor.
Your job is to extract structured information from research paper text chunks with MAXIMUM ACCURACY.

CRITICAL RULES:
1. Always respond with valid JSON only. No markdown, no code fences, no explanation.
2. If a section is not in this chunk, use "Not found" — never invent content.
3. For key_figures: extract EVERY concrete number, metric, score, percentage, count,
   p-value, or measurement that appears. Numbers are the most important output.
4. Be precise — copy exact values (e.g. "94.3%" not "about 94%", "p < 0.001" not "significant").
5. Preserve LaTeX equations exactly as written."""


def build_section_extraction_user_prompt(text: str, hint: str = "") -> str:
    hint_block = f"\n{hint}\n" if hint else ""

    return f"""Extract structured information from the research paper chunk below.
{hint_block}
Return a JSON object with EXACTLY these keys:

  "title"        — Paper title (exact string, or "Not found")
  "authors"      — Author names (comma-separated, or "Not found")
  "abstract"     — 4-6 sentences summarising the paper's purpose, method, and key result
  "introduction" — 3-5 sentences on the problem, motivation, and gap in prior work
  "methodology"  — 4-6 sentences on the technical approach, architecture, or experimental design
  "results"      — 4-6 sentences on quantitative findings. MUST include all numbers found.
  "conclusion"   — 3-4 sentences on the paper's contribution and significance
  "limitations"  — 2-3 sentences on what the method cannot do or has not tested
  "future_work"  — 2-3 sentences on directions explicitly stated by the authors
  "key_figures"  — LIST of objects with keys: label, value, context, section
                   Extract EVERY concrete numeric finding:
                   - Performance metrics (accuracy, F1, BLEU, ROUGE, AUC, mAP, perplexity...)
                   - Dataset statistics (# samples, # classes, train/val/test splits...)
                   - Model specs (# parameters, # layers, hidden dim, FLOPs, latency...)
                   - Statistical results (p-values, confidence intervals, effect sizes...)
                   - Comparison deltas ("+2.3% over baseline", "3x faster than X"...)
                   - Any other quantitative claim that supports the paper's contribution
                   Each object: {{"label": "metric name", "value": "exact value", "context": "one sentence what it means", "section": "results/abstract/methodology/etc"}}

RULES for key_figures:
- Extract 5–20 figures depending on how many numbers appear in the text
- Exact values only — copy numbers verbatim from the text
- If a number appears multiple times, include it only once with the most descriptive label
- Do NOT include page numbers, citation numbers, or year numbers as key figures

RULES for text sections:
- Each value: a coherent paragraph (not a list). Include key technical terms.
- Preserve LaTeX equations: $$\\frac{{QK^T}}{{\\sqrt{{d_k}}}}$$
- "Not found" if the section is not present in this chunk.
- Return ONLY the JSON object — no preamble, no markdown fences.

Paper text:
\"\"\"
{text}
\"\"\"
"""
