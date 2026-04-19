"""
app/domain/models.py
Pydantic models representing core domain objects.
"""
from __future__ import annotations
from pydantic import BaseModel, Field

NOT_FOUND = "Not found"


class ExtractedEquation(BaseModel):
    page_number: int
    latex: str
    description: str


class ExtractedFigure(BaseModel):
    """A figure extracted from the PDF — image is a CROPPED region, not a full page."""
    page_number: int
    caption: str
    description: str
    png_b64: str = ""


class KeyFigure(BaseModel):
    """
    A concrete numeric finding extracted from the paper.
    Examples: accuracy scores, dataset sizes, parameter counts, p-values, etc.
    """
    label: str       # e.g. "Top-1 Accuracy", "Dataset Size", "Parameters"
    value: str       # e.g. "94.3%", "1.2M samples", "p < 0.001"
    context: str = ""  # one-sentence explanation of what this number means
    section: str = ""  # which section it came from (results, abstract, etc.)


class PaperSections(BaseModel):
    title:        str = Field(default=NOT_FOUND)
    authors:      str = Field(default=NOT_FOUND)
    abstract:     str = Field(default=NOT_FOUND)
    introduction: str = Field(default=NOT_FOUND)
    methodology:  str = Field(default=NOT_FOUND)
    results:      str = Field(default=NOT_FOUND)
    conclusion:   str = Field(default=NOT_FOUND)
    limitations:  str = Field(default=NOT_FOUND)
    future_work:  str = Field(default=NOT_FOUND)

    key_figures:    list[KeyFigure]         = Field(default_factory=list)
    equations:      list[ExtractedEquation] = Field(default_factory=list)
    figures:        list[ExtractedFigure]   = Field(default_factory=list)
    page_summaries: dict[int, str]          = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PaperSections":
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "PaperSections":
        text_fields = {
            "title", "authors", "abstract", "introduction",
            "methodology", "results", "conclusion", "limitations", "future_work",
        }
        cleaned = {
            f: (data.get(f, "") if isinstance(data.get(f), str) and data.get(f, "").strip()
                else NOT_FOUND)
            for f in text_fields
        }

        # Parse key_figures list from extraction dict
        key_figures: list[KeyFigure] = []
        raw_kf = data.get("key_figures", [])
        if isinstance(raw_kf, list):
            for item in raw_kf:
                if isinstance(item, dict) and item.get("label") and item.get("value"):
                    key_figures.append(KeyFigure(
                        label=str(item.get("label", "")),
                        value=str(item.get("value", "")),
                        context=str(item.get("context", "")),
                        section=str(item.get("section", "")),
                    ))

        return cls(**cleaned, key_figures=key_figures)


class ReviewDimensionAssessment(BaseModel):
    verdict: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    confidence: str = ""


class PaperCriticalProfile(BaseModel):
    title: str
    authors: str
    strengths: ReviewDimensionAssessment
    weaknesses: ReviewDimensionAssessment
    novelty: ReviewDimensionAssessment
    assumptions: ReviewDimensionAssessment
    threats_to_validity: ReviewDimensionAssessment
    reproducibility: ReviewDimensionAssessment
    fairness_of_comparison: ReviewDimensionAssessment
    applicability: ReviewDimensionAssessment


class PairwiseDimensionComparison(BaseModel):
    dimension: str
    paper_a: str
    paper_b: str
    comparative_judgement: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class CriticalComparisonResult(BaseModel):
    paper_a_profile: PaperCriticalProfile
    paper_b_profile: PaperCriticalProfile
    pairwise_comparisons: list[PairwiseDimensionComparison] = Field(default_factory=list)
    comparison_markdown: str


class PipelineResult(BaseModel):
    sections:         PaperSections
    summary_markdown: str
    report_pdf_bytes: bytes

    model_config = {"arbitrary_types_allowed": True}


class CriticalComparisonPipelineResult(BaseModel):
    paper_a_sections: PaperSections
    paper_b_sections: PaperSections
    result: CriticalComparisonResult
