"""
app/services/report_service.py

Clean, well-structured PDF report.
Cover → Abstract → Section summaries (one per page section, clear headings).
Nothing else.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, HRFlowable, Table, TableStyle,
)

from app.core.config import get_settings
from app.core.exceptions import ReportGenerationError
from app.domain.models import PaperSections
from app.utils.text import markdown_bold_to_html, safe_html

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = LETTER
MARGIN    = 1.0 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# Colours
C_NAVY  = colors.HexColor("#0f2557")
C_BLUE  = colors.HexColor("#2563eb")
C_LBLUE = colors.HexColor("#dbeafe")
C_STEEL = colors.HexColor("#64748b")
C_RULE  = colors.HexColor("#e2e8f0")
C_TEXT  = colors.HexColor("#1e293b")
C_WHITE = colors.white

# One accent colour per section heading
SECTION_COLORS = {
    # original sections
    "Overview":                      colors.HexColor("#0f2557"),
    "Introduction":                  colors.HexColor("#2563eb"),
    "Methodology":                   colors.HexColor("#0d9488"),
    "Results":                       colors.HexColor("#16a34a"),
    "Conclusion":                    colors.HexColor("#7c3aed"),
    "Limitations":                   colors.HexColor("#dc2626"),
    "Future Work":                   colors.HexColor("#d97706"),
    # new comprehensive sections
    "Problem Statement & Motivation": colors.HexColor("#2563eb"),
    "Proposed Approach & Methodology": colors.HexColor("#0d9488"),
    "Experimental Results & Analysis": colors.HexColor("#16a34a"),
    "Contributions & Conclusions":    colors.HexColor("#7c3aed"),
    "Limitations & Constraints":      colors.HexColor("#dc2626"),
    "Future Work & Open Problems":    colors.HexColor("#d97706"),
    "Key Statistics & Metrics":       colors.HexColor("#0f766e"),
    "Notable Equations":              colors.HexColor("#7e22ce"),
}


def _styles() -> dict:
    base = getSampleStyleSheet()

    def _p(name, **kw) -> ParagraphStyle:
        parent = kw.pop("parent", base["Normal"])
        return ParagraphStyle(name, parent=parent, **kw)

    return {
        "cover_title":   _p("CT", fontSize=24, textColor=C_NAVY,
                              fontName="Helvetica-Bold", leading=30,
                              alignment=TA_CENTER, spaceAfter=12),
        "cover_authors": _p("CA", fontSize=11, textColor=C_STEEL,
                              alignment=TA_CENTER, spaceAfter=6),
        "cover_meta":    _p("CM", fontSize=9,  textColor=C_STEEL,
                              alignment=TA_CENTER),
        "section_head":  _p("SH", fontSize=14, textColor=C_NAVY,
                              fontName="Helvetica-Bold",
                              spaceBefore=6, spaceAfter=4, leading=18),
        "body":          _p("BD", fontSize=10, leading=17, spaceAfter=8,
                              textColor=C_TEXT, alignment=TA_JUSTIFY),
        "bullet":        _p("BL", fontSize=10, leading=16, spaceAfter=5,
                              textColor=C_TEXT, leftIndent=16),
        "abstract":      _p("AB", fontSize=10, leading=17, spaceAfter=0,
                              textColor=C_TEXT, alignment=TA_JUSTIFY),
        "footer":        _p("FT", fontSize=8,  textColor=C_STEEL,
                              alignment=TA_CENTER),
        "table_header":  _p("TH", fontSize=9,  textColor=C_WHITE,
                              fontName="Helvetica-Bold", leading=13),
        "table_cell":    _p("TC", fontSize=9,  textColor=C_TEXT, leading=13),
        "table_value":   _p("TV", fontSize=9,  textColor=colors.HexColor("#0f766e"),
                              fontName="Helvetica-Bold", leading=13),
        "eq":            _p("EQ", fontSize=9,  textColor=colors.HexColor("#7e22ce"),
                              fontName="Courier", leading=14, leftIndent=12),
    }


# ── Canvas callbacks ──────────────────────────────────────────────────────────

def _on_cover(canvas, doc, author):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_STEEL)
    canvas.drawCentredString(PAGE_W / 2, 0.4 * inch,
                             f"AI-generated summary  ·  {author}  ·  "
                             f"{datetime.now().strftime('%B %d, %Y')}")
    canvas.restoreState()


def _on_page(canvas, doc, title, author):
    canvas.saveState()
    # Top navy bar
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, PAGE_H - 0.22 * inch, PAGE_W, 0.22 * inch, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_WHITE)
    canvas.drawString(MARGIN, PAGE_H - 0.15 * inch, title[:90])
    # Footer
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_STEEL)
    canvas.drawString(MARGIN, 0.4 * inch,
                      f"AI-generated summary  ·  {author}  ·  "
                      f"{datetime.now().strftime('%Y-%m-%d')}")
    canvas.drawRightString(PAGE_W - MARGIN, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ── Cover page ────────────────────────────────────────────────────────────────

def _cover(sections: PaperSections, st: dict) -> list:
    story = [Spacer(1, 1.8 * inch)]
    story.append(HRFlowable(width="100%", thickness=4, color=C_NAVY, spaceAfter=20))
    story.append(Paragraph(safe_html(sections.title), st["cover_title"]))
    story.append(Paragraph(safe_html(sections.authors), st["cover_authors"]))
    story.append(Paragraph(
        f"Research Paper Summary  ·  {datetime.now().strftime('%B %d, %Y')}",
        st["cover_meta"],
    ))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(PageBreak())
    return story


# ── Abstract box ──────────────────────────────────────────────────────────────

def _abstract(sections: PaperSections, st: dict) -> list:
    if not sections.abstract or sections.abstract == "Not found":
        return []
    story = []
    story.extend(_section_header("Abstract", C_NAVY, st))
    cell = Paragraph(safe_html(sections.abstract), st["abstract"])
    box  = Table([[cell]], colWidths=[CONTENT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LBLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("LINEABOVE",     (0, 0), (-1,  0),  3, C_BLUE),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_RULE),
    ]))
    story.append(box)
    story.append(Spacer(1, 20))
    return story


# ── Section header helper ─────────────────────────────────────────────────────

def _section_header(title: str, colour, st: dict) -> list:
    return [
        Spacer(1, 8),
        Paragraph(safe_html(title), st["section_head"]),
        HRFlowable(width="100%", thickness=2, color=colour, spaceAfter=8),
    ]


# ── Key statistics table ──────────────────────────────────────────────────────

def _key_stats(sections: PaperSections, st: dict) -> list:
    """Render an extracted key_figures table in the PDF report."""
    from app.domain.models import KeyFigure  # local import to avoid circularity
    if not sections.key_figures:
        return []

    story: list = []
    story.extend(_section_header("Key Statistics & Metrics",
                                 colors.HexColor("#0f766e"), st))

    col_w = [CONTENT_W * 0.28, CONTENT_W * 0.18, CONTENT_W * 0.54]
    rows = [
        [
            Paragraph("Metric",  st["table_header"]),
            Paragraph("Value",   st["table_header"]),
            Paragraph("Context", st["table_header"]),
        ]
    ]
    for kf in sections.key_figures[:30]:
        rows.append([
            Paragraph(safe_html(kf.label),   st["table_cell"]),
            Paragraph(safe_html(kf.value),   st["table_value"]),
            Paragraph(safe_html(kf.context[:160] if kf.context else ""), st["table_cell"]),
        ])

    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1,  0), colors.HexColor("#0f766e")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0fdf4"), colors.HexColor("#dcfce7")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#bbf7d0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 18))
    return story


# ── Notable equations section ─────────────────────────────────────────────────

def _equations_section(sections: PaperSections, st: dict) -> list:
    if not sections.equations:
        return []
    story: list = []
    story.extend(_section_header("Notable Equations",
                                 colors.HexColor("#7e22ce"), st))
    for eq in sections.equations[:8]:
        label = f"p.{eq.page_number} — {safe_html(eq.description)}"
        story.append(Paragraph(label, st["table_cell"]))
        story.append(Paragraph(safe_html(eq.latex), st["eq"]))
        story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))
    return story


# ── Markdown → flowables ──────────────────────────────────────────────────────

_INLINE_EQ = re.compile(r'\$([^$\n]+)\$')


def _inline_eq(text: str) -> str:
    return _INLINE_EQ.sub(
        lambda m: f'<font name="Courier" color="#0f2557">{safe_html(m.group(1))}</font>',
        text,
    )


def _parse(summary: str, st: dict) -> list:
    story = []
    for raw in summary.splitlines():
        line = raw.strip()

        if not line:
            story.append(Spacer(1, 4))
            continue

        # ## Section heading
        if line.startswith("## "):
            title = line[3:].strip()
            # Strip leading emoji for colour lookup
            clean = re.sub(r'^[\U00010000-\U0010ffff\u2600-\u27BF\uFE00-\uFE0F]+\s*', '', title)
            colour = SECTION_COLORS.get(clean.strip(), C_NAVY)
            story.append(Spacer(1, 10))
            items = _section_header(title, colour, st)
            story.extend(items)
            continue

        # Bullet
        if line.startswith(("- ", "* ")):
            content = markdown_bold_to_html(_inline_eq(safe_html(line[2:])))
            story.append(Paragraph(f"• {content}", st["bullet"]))
            continue

        # Body paragraph
        content = markdown_bold_to_html(_inline_eq(safe_html(line)))
        story.append(Paragraph(content, st["body"]))

    return story


# ── Public API ────────────────────────────────────────────────────────────────

def build_pdf(summary: str, sections: PaperSections) -> bytes:
    settings = get_settings()
    st  = _styles()
    buf = io.BytesIO()

    short = (sections.title[:80] + "…") if len(sections.title) > 80 else sections.title

    try:
        doc = SimpleDocTemplate(
            buf, pagesize=LETTER,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN + 0.25 * inch,
            bottomMargin=MARGIN,
        )

        story: list = []
        story.extend(_cover(sections, st))
        story.extend(_abstract(sections, st))
        story.extend(_key_stats(sections, st))
        story.extend(_equations_section(sections, st))
        story.extend(_parse(summary, st))

        story.append(Spacer(1, 24))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=6))
        story.append(Paragraph(
            "AI-generated summary — always verify against the original publication.",
            st["footer"],
        ))

        doc.build(
            story,
            onFirstPage=lambda c, d: _on_cover(c, d, settings.report_author),
            onLaterPages=lambda c, d: _on_page(c, d, short, settings.report_author),
        )

    except Exception as exc:
        logger.error("PDF build failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise ReportGenerationError("Failed to build PDF.", original=exc) from exc

    result = buf.getvalue()
    logger.info("PDF built: %d bytes", len(result))
    return result
