"""
app/services/pipeline_service.py
Simple 3-step pipeline: parse → extract sections → summarize → report.

Changes vs previous version
────────────────────────────
• resolve_stable_metadata now called with 4 args (marker + LLM) in both
  the single-paper and comparison flows, so the best available title/authors
  from either source is always preserved.
• Best-of-two title selection is now handled inside resolve_stable_metadata
  via _title_score, so "QLORA: Efficient Finetuning…" consistently wins over
  any section-heading string regardless of which source produced it.
"""
from __future__ import annotations
import logging

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.domain.models import CriticalComparisonPipelineResult, PaperSections, PipelineResult
from app.services import critical_review_service, llm_service, report_service, summarizer_service
from app.services.marker_service import convert_pdf as marker_convert

logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_bytes:  bytes,
    api_key:    str,
    model:      str | None = None,
    use_vision: bool = False,
    gemini_key: str = "",
) -> PipelineResult:
    try:
        logger.info("Pipeline started (%d bytes)", len(pdf_bytes))
        settings = get_settings()
        if model:
            settings.groq_model = model
        client = llm_service.create_groq_client(api_key)

        logger.info("Step 1/3: Parsing PDF...")
        marker_result = marker_convert(pdf_bytes)
        logger.info("  %d chars extracted", len(marker_result.full_markdown))

        logger.info("Step 2/3: Extracting sections...")
        sections = summarizer_service.extract_sections(
            full_text=marker_result.full_text_for_llm,
            client=client,
            settings=settings,
            marker_sections=marker_result.sections,
        )

        # Resolve metadata using both sources; best title wins via _title_score
        stable_title, stable_authors = critical_review_service.resolve_stable_metadata(
            marker_result.title,   # primary
            marker_result.authors,
            sections.title,        # fallback
            sections.authors,
        )
        if stable_title != sections.title or stable_authors != sections.authors:
            sections = sections.model_copy(update={"title": stable_title, "authors": stable_authors})

        logger.info("Step 3/3: Generating summary...")
        summary_markdown = summarizer_service.generate_summary(sections, client, settings)

        logger.info("Building report...")
        report_pdf_bytes = report_service.build_pdf(summary_markdown, sections)

        logger.info("Pipeline complete.")
        return PipelineResult(
            sections=sections,
            summary_markdown=summary_markdown,
            report_pdf_bytes=report_pdf_bytes,
        )
    except AppError:
        raise
    except Exception as exc:
        logger.error("Unhandled pipeline failure: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise AppError("Unexpected server error. Please try again.", original=exc) from exc


def _extract_sections_from_pdf(
    pdf_bytes:   bytes,
    client,
    settings,
    paper_label: str,
) -> PaperSections:
    logger.info("%s: parsing PDF...", paper_label)
    marker_result = marker_convert(pdf_bytes)
    logger.info("%s: %d chars extracted", paper_label, len(marker_result.full_markdown))

    logger.info("%s: extracting sections...", paper_label)
    sections = summarizer_service.extract_sections(
        full_text=marker_result.full_text_for_llm,
        client=client,
        settings=settings,
        marker_sections=marker_result.sections,
    )

    # Best-of-two metadata: marker is primary, LLM is fallback.
    # _title_score inside resolve_stable_metadata picks the academically
    # correct string even when the order happens to be reversed.
    stable_title, stable_authors = critical_review_service.resolve_stable_metadata(
        marker_result.title,
        marker_result.authors,
        sections.title,
        sections.authors,
    )
    sections = sections.model_copy(update={"title": stable_title, "authors": stable_authors})
    logger.info("%s: metadata resolved — title='%s'", paper_label, stable_title)
    return sections


def run_critical_comparison_pipeline(
    pdf_a_bytes: bytes,
    pdf_b_bytes: bytes,
    api_key:     str,
    model:       str | None = None,
    use_vision:  bool = False,
    gemini_key:  str = "",
) -> CriticalComparisonPipelineResult:
    del use_vision, gemini_key
    try:
        logger.info(
            "Critical comparison pipeline started (A=%d bytes, B=%d bytes)",
            len(pdf_a_bytes), len(pdf_b_bytes),
        )
        settings = get_settings()
        if model:
            settings.groq_model = model
        client = llm_service.create_groq_client(api_key)

        paper_a_sections = _extract_sections_from_pdf(pdf_a_bytes, client, settings, "Paper A")
        paper_b_sections = _extract_sections_from_pdf(pdf_b_bytes, client, settings, "Paper B")

        logger.info("Generating critical profiles...")
        paper_a_profile = critical_review_service.generate_paper_critical_profile(
            paper_a_sections, client, settings
        )
        paper_b_profile = critical_review_service.generate_paper_critical_profile(
            paper_b_sections, client, settings
        )

        logger.info("Generating pairwise comparison...")
        result = critical_review_service.generate_critical_comparison(
            paper_a_profile, paper_b_profile, client, settings
        )

        logger.info("Critical comparison pipeline complete.")
        return CriticalComparisonPipelineResult(
            paper_a_sections=paper_a_sections,
            paper_b_sections=paper_b_sections,
            result=result,
        )
    except AppError:
        raise
    except Exception as exc:
        logger.error("Unhandled comparison failure: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise AppError("Unexpected server error while comparing documents. Please try again.", original=exc) from exc