"""
app/ui/streamlit_ui.py
Streamlit UI for summary and critical comparison workflows.
"""
from __future__ import annotations

import logging

import streamlit as st

from app.core.config import get_settings
from app.core.exceptions import AppError, LLMServiceError, PDFExtractionError
from app.services.pipeline_service import run_critical_comparison_pipeline, run_pipeline

logger = logging.getLogger(__name__)

CRITICAL_DIMENSIONS = (
    ("strengths", "Strengths"),
    ("weaknesses", "Weaknesses"),
    ("novelty", "Novelty"),
    ("assumptions", "Assumptions"),
    ("threats_to_validity", "Threats to Validity"),
    ("reproducibility", "Reproducibility"),
    ("fairness_of_comparison", "Fairness of Comparison"),
    ("applicability", "Applicability"),
)


def _render_evidence(evidence: list[str]) -> None:
    if evidence:
        st.markdown("\n".join(f"- {item}" for item in evidence))
        return
    st.markdown("- Not enough evidence cited.")


def _render_assessment(label: str, assessment) -> None:
    st.subheader(label)
    st.markdown(f"**Verdict:** {assessment.verdict}")
    st.caption(f"Confidence: {assessment.confidence or 'Low'}")
    st.write(assessment.rationale)
    st.markdown("**Evidence**")
    _render_evidence(assessment.evidence)


def _render_profile(heading: str, profile) -> None:
    st.header(heading)
    st.markdown(f"**Title:** {profile.title}")
    st.markdown(f"**Authors:** {profile.authors}")

    for dimension, label in CRITICAL_DIMENSIONS:
        _render_assessment(label, getattr(profile, dimension))


def _render_direct_comparison(comparisons) -> None:
    st.header("Direct Comparison")

    for comparison in comparisons:
        label = dict(CRITICAL_DIMENSIONS).get(
            comparison.dimension,
            comparison.dimension.replace("_", " ").title(),
        )
        st.subheader(label)
        st.markdown(f"**Paper A:** {comparison.paper_a}")
        st.markdown(f"**Paper B:** {comparison.paper_b}")
        st.markdown(f"**Comparative Judgement:** {comparison.comparative_judgement}")
        st.write(comparison.rationale)
        st.markdown("**Evidence**")
        _render_evidence(comparison.evidence)


def _handle_pipeline_error(exc: Exception, status) -> None:
    if isinstance(exc, LLMServiceError):
        status.update(label="Error", state="error")
        st.error(f"AI Error: {exc.message}")
        if exc.original:
            with st.expander("Details"):
                st.code(str(exc.original))
        return

    if isinstance(exc, PDFExtractionError):
        status.update(label="Error", state="error")
        st.error(f"PDF Error: {exc.message}")
        return

    if isinstance(exc, AppError):
        status.update(label="Error", state="error")
        st.error(f"Error: {exc.message}")
        if exc.original:
            with st.expander("Details"):
                st.code(str(exc.original))
        return

    logger.error("Unhandled UI pipeline error: %s: %s", type(exc).__name__, exc, exc_info=True)
    status.update(label="Error", state="error")
    st.error("Unexpected server error while processing the document. Please try again.")
    with st.expander("Details"):
        st.code(str(exc))


def _render_summary_mode(api_key: str, model: str) -> None:
    uploaded = st.file_uploader("Upload research paper (PDF)", type=["pdf"], key="summary_pdf")

    if not uploaded:
        st.info("Upload one PDF to generate the standard summary report.")
        return

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar.")
        return

    if not st.button("Summarize Paper", type="primary", use_container_width=True, key="summary_button"):
        return

    with st.status("Analyzing paper...", expanded=True) as status:
        try:
            st.write("Parsing PDF...")
            st.write("Extracting sections...")
            st.write("Generating summary...")

            result = run_pipeline(
                pdf_bytes=uploaded.getvalue(),
                api_key=api_key,
                model=model,
            )

            st.write("Building report...")
            status.update(label="Done", state="complete")
        except Exception as exc:
            _handle_pipeline_error(exc, status)
            return

    st.success(result.sections.title)

    tab_summary, tab_download = st.tabs(["Summary", "Download Report"])

    with tab_summary:
        st.markdown(result.summary_markdown)

    with tab_download:
        st.download_button(
            label="Download PDF Report",
            data=result.report_pdf_bytes,
            file_name=f"summary_{uploaded.name}",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(f"{result.sections.title} | {result.sections.authors}")


def _render_comparison_mode(api_key: str, model: str) -> None:
    uploaded_a = st.file_uploader("Paper A (PDF)", type=["pdf"], key="comparison_pdf_a")
    uploaded_b = st.file_uploader("Paper B (PDF)", type=["pdf"], key="comparison_pdf_b")

    if not uploaded_a or not uploaded_b:
        st.info("Upload two PDFs to generate the critical comparison.")
        return

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar.")
        return

    if not st.button(
        "Compare Papers Critically",
        type="primary",
        use_container_width=True,
        key="comparison_button",
    ):
        return

    with st.status("Comparing papers critically...", expanded=True) as status:
        try:
            st.write("Parsing Paper A...")
            st.write("Parsing Paper B...")
            st.write("Extracting sections for both papers...")
            st.write("Generating critical profiles...")
            st.write("Building pairwise comparison...")

            pipeline_result = run_critical_comparison_pipeline(
                pdf_a_bytes=uploaded_a.getvalue(),
                pdf_b_bytes=uploaded_b.getvalue(),
                api_key=api_key,
                model=model,
            )

            status.update(label="Done", state="complete")
        except Exception as exc:
            _handle_pipeline_error(exc, status)
            return

    st.success("Critical comparison ready.")
    _render_profile("Paper A", pipeline_result.result.paper_a_profile)
    _render_profile("Paper B", pipeline_result.result.paper_b_profile)
    _render_direct_comparison(pipeline_result.result.pairwise_comparisons)


def render_app() -> None:
    settings = get_settings()
    st.set_page_config(page_title=settings.app_title, page_icon=settings.app_icon, layout="centered")

    st.title(settings.app_title)
    st.caption("Choose a mode: summarize one paper or compare two papers critically.")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a Groq API key from https://console.groq.com",
        )
        model = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
            ],
            help="Choose a Groq model for the extraction and analysis steps.",
        )
        st.divider()
        st.markdown("**Workflows**")
        st.markdown("1. Summarize one paper with the existing report flow.")
        st.markdown("2. Compare two papers across fixed critical-review dimensions only.")

    mode = st.radio(
        "Mode",
        ("Summarize Paper", "Critical Comparison"),
        horizontal=True,
    )

    if mode == "Summarize Paper":
        st.caption("Upload one PDF for the standard section-by-section summary and report.")
        _render_summary_mode(api_key, model)
        return

    st.caption("Upload two PDFs to compare them only across the requested critical-review dimensions.")
    _render_comparison_mode(api_key, model)
