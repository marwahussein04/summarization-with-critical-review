"""
app/services/summarizer_service.py

AI business logic — now with:
  1. Full-paper chunked extraction (no truncation)
  2. key_figures extraction: all numeric findings per chunk, merged and deduplicated
  3. marker-derived section pre-seeding
  4. Vision enrichment (Gemini or Groq)
"""
from __future__ import annotations

import json
import logging

from groq import Groq

from app.core.config import Settings
from app.domain.models import PaperSections, ExtractedEquation, ExtractedFigure, KeyFigure
from app.prompts.section_extraction import (
    build_section_extraction_system_prompt,
    build_section_extraction_user_prompt,
)
from app.prompts.summary_generation import (
    build_summary_system_prompt,
    build_summary_user_prompt,
)
from app.services.llm_service import chat_completion
from app.utils.chunker import chunk_text, merge_section_dicts, is_single_chunk
from app.utils.text import strip_code_fences

logger = logging.getLogger(__name__)

try:
    from app.services.gemini_vision_service import VisionAnalysis
except ImportError:
    from app.services.vision_service import VisionAnalysis  # type: ignore


def extract_sections(
    full_text: str,
    client: Groq,
    settings: Settings,
    marker_sections: dict[str, str] | None = None,
) -> PaperSections:
    """
    Extract structured sections + key numeric figures from the full paper text.

    If the text fits in one LLM call: send directly.
    If not: chunk it, extract from each chunk, merge results.
    key_figures are collected from ALL chunks and deduplicated.
    """
    system_prompt = build_section_extraction_system_prompt()
    chunks        = chunk_text(full_text)

    logger.info("Extracting sections from %d chunks (%d total chars).",
                len(chunks), len(full_text))

    raw_dicts: list[dict] = []

    for chunk in chunks:
        hint = ""
        if marker_sections and chunk.index == 0:
            hint = _build_marker_hint(marker_sections)

        user_prompt = build_section_extraction_user_prompt(chunk.text, hint=hint)

        raw = chat_completion(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            settings=settings,
            max_tokens=settings.max_tokens_section_extraction,
        )

        cleaned = strip_code_fences(raw)
        try:
            data = json.loads(cleaned)
            raw_dicts.append(data)
            n_kf = len(data.get("key_figures", []))
            logger.debug("  Chunk %d/%d extracted OK. key_figures: %d",
                         chunk.index + 1, len(chunks), n_kf)
        except json.JSONDecodeError as exc:
            logger.warning("  Chunk %d JSON parse failed: %s", chunk.index + 1, exc)

    if not raw_dicts:
        logger.error("All chunks failed — returning empty sections.")
        return PaperSections.empty()

    merged = merge_section_dicts(raw_dicts)
    sections = PaperSections.from_dict(merged)
    logger.info("Sections merged. Title: %s | Key figures: %d",
                sections.title, len(sections.key_figures))
    return sections


def _build_marker_hint(marker_sections: dict[str, str]) -> str:
    non_empty = {k: v[:400] for k, v in marker_sections.items() if v.strip()}
    if not non_empty:
        return ""
    lines = ["The paper's structure has been pre-detected. Use this as a guide:"]
    for k, v in non_empty.items():
        lines.append(f"  [{k.upper()}]: {v[:200]}...")
    return "\n".join(lines)


def enrich_sections_with_vision(
    sections: PaperSections,
    vision: VisionAnalysis,
) -> PaperSections:
    """Merge vision analysis into sections model."""
    equations = [
        ExtractedEquation(
            page_number=eq.page_number,
            latex=eq.latex,
            description=eq.description,
        )
        for eq in vision.equations
    ]
    figures = [
        ExtractedFigure(
            page_number=fig.page_number,
            caption=fig.caption,
            description=fig.description,
            png_b64=fig.png_b64,
        )
        for fig in vision.figures
    ]
    return sections.model_copy(update={
        "equations":      equations,
        "figures":        figures,
        "page_summaries": vision.page_summaries,
    })


def enrich_sections_with_marker(
    sections: PaperSections,
    marker_equations: list,
) -> PaperSections:
    """
    Add equations that marker found in the PDF text (LaTeX blocks).
    Only adds marker equations if vision found fewer than marker did.
    """
    if not marker_equations:
        return sections

    existing_latex = {eq.latex.strip() for eq in sections.equations}

    extras: list[ExtractedEquation] = []
    for meq in marker_equations:
        if meq.latex.strip() not in existing_latex and meq.is_block:
            extras.append(ExtractedEquation(
                page_number=0,
                latex=meq.latex,
                description=meq.context or "Equation from paper body",
            ))
            existing_latex.add(meq.latex.strip())

    if extras:
        logger.info("Adding %d extra equations from marker.", len(extras))
        return sections.model_copy(update={
            "equations": sections.equations + extras
        })

    return sections


def generate_summary(
    sections: PaperSections,
    client: Groq,
    settings: Settings,
) -> str:
    """Generate the final markdown overview summary."""
    summary = chat_completion(
        client=client,
        system_prompt=build_summary_system_prompt(),
        user_prompt=build_summary_user_prompt(sections),
        settings=settings,
        max_tokens=settings.max_tokens_summary,
    )
    logger.info("Summary generated (%d chars).", len(summary))
    return summary
