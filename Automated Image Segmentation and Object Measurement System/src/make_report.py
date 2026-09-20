"""
PDF renderer of the report.

The content itself lives in report_content.py as a sequence of neutral blocks.
This module turns those blocks into the report PDF. Run the
experiments first, then this script.

    python run_experiments.py
    python make_report.py
"""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)

import report_content as C
from config import FIG_DIR, REPORT_DIR

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

AUTHOR = C.AUTHOR
PROJECT = C.PROJECT
TITLE = C.TITLE

# ----------------------------------------------------------------------------
# Paragraph styles
# ----------------------------------------------------------------------------
_sheet = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_sheet["BodyText"], fontSize=9.6,
                      leading=13.4, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=_sheet["Heading1"], fontSize=15, leading=18,
                    spaceBefore=14, spaceAfter=8,
                    textColor=colors.HexColor("#12325c"))
H2 = ParagraphStyle("h2", parent=_sheet["Heading2"], fontSize=11.8, leading=15,
                    spaceBefore=10, spaceAfter=5,
                    textColor=colors.HexColor("#1d4d80"))
H3 = ParagraphStyle("h3", parent=_sheet["Heading3"], fontSize=10.4, leading=13,
                    spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#28527a"))
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.2, leading=10.5,
                         alignment=TA_CENTER,
                         textColor=colors.HexColor("#44484f"),
                         spaceBefore=2, spaceAfter=10)
CODE = ParagraphStyle("code", parent=_sheet["Code"], fontSize=5.6, leading=6.6)
CODE_HEAD = ParagraphStyle("codehead", parent=H3, fontSize=9.4, spaceBefore=10)
PRE = ParagraphStyle("pre", parent=_sheet["Code"], fontSize=8, leading=10.5)


def paragraph_style(name):
    """Build a reportlab style from one of the shared style descriptions."""
    spec = C.STYLES[name]
    font = "Helvetica"
    if spec.get("mono"):
        font = "Courier"
    elif spec["bold"]:
        font = "Helvetica-Bold"
    elif spec["italic"]:
        font = "Helvetica-Oblique"
    indent = spec.get("indent_cm", 0) * cm
    return ParagraphStyle(
        "shared_" + name, parent=BODY, fontName=font, fontSize=spec["size"],
        leading=spec["size"] * 1.38,
        alignment=TA_CENTER if spec["centre"] else TA_JUSTIFY,
        textColor=colors.HexColor(spec["colour"]),
        leftIndent=indent, rightIndent=indent,
        spaceBefore=4 if spec["centre"] else 0, spaceAfter=6)


STYLE_CACHE = {name: paragraph_style(name) for name in C.STYLES}


# ----------------------------------------------------------------------------
# Block renderers
# ----------------------------------------------------------------------------
def render_table(header, rows, col_widths_cm, font_size, highlight_rows):
    widths = [w * cm for w in col_widths_cm] if col_widths_cm else None
    data = [[Paragraph("<b>%s</b>" % c, ParagraphStyle(
        "th", parent=BODY, fontSize=font_size, leading=font_size + 2,
        textColor=colors.white)) for c in header]]
    for r in rows:
        data.append([Paragraph(str(c), ParagraphStyle(
            "td", parent=BODY, fontSize=font_size, leading=font_size + 2.2))
            for c in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4d80")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f7")]),
    ]
    for i in highlight_rows:
        style.append(("BACKGROUND", (0, i + 1), (-1, i + 1),
                      colors.HexColor("#d8ecd8")))
    t.setStyle(TableStyle(style))
    return t


def render_figure(name, caption, max_height_cm):
    path = FIG_DIR / name
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    w = CONTENT_W
    h = w * ih / float(iw)
    if h > max_height_cm * cm:
        h = max_height_cm * cm
        w = h * iw / float(ih)
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    return KeepTogether([img, Paragraph(caption, CAPTION)])


def render_rule(width_cm, colour, thickness):
    width = width_cm * cm if width_cm else CONTENT_W
    t = Table([[""]], colWidths=[width], rowHeights=[0.01])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), thickness, colors.HexColor(colour)),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    t.hAlign = "CENTER"
    return t


def build_story():
    story = []
    for block in C.BLOCKS:
        kind = block[0]
        if kind == "p":
            story.append(Paragraph(block[1], STYLE_CACHE[block[2]]))
        elif kind == "h1":
            story.append(Paragraph(block[1], H1))
        elif kind == "h2":
            story.append(Paragraph(block[1], H2))
        elif kind == "h3":
            story.append(Paragraph(block[1], H3))
        elif kind == "bullets":
            for item in block[1]:
                story.append(Paragraph("&bull;&nbsp;&nbsp;" + item,
                                       ParagraphStyle("b", parent=BODY,
                                                      leftIndent=12,
                                                      spaceAfter=3)))
            story.append(Spacer(1, 4))
        elif kind == "table":
            story.append(render_table(block[1], block[2], block[3], block[4],
                                      block[5]))
            story.append(Spacer(1, 8))
        elif kind == "caption":
            story.append(Paragraph(block[1], CAPTION))
        elif kind == "figure":
            story.append(render_figure(block[1], block[2], block[3]))
        elif kind == "pre":
            story.append(Preformatted(block[1], PRE))
        elif kind == "code_heading":
            story.append(Paragraph(block[1], CODE_HEAD))
        elif kind == "code":
            story.append(Preformatted(block[1], CODE))
        elif kind == "rule":
            _, width_cm, colour, thickness, before, after = block
            story.append(Spacer(1, before))
            story.append(render_rule(width_cm, colour, thickness))
            story.append(Spacer(1, after))
        elif kind == "spacer":
            story.append(Spacer(1, block[1]))
        elif kind == "page_break":
            story.append(PageBreak())
        else:
            raise ValueError("Unknown block: %s" % kind)
    return story


# ----------------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------------
def draw_furniture(canvas, doc):
    # The cover page carries no running header and no page number.
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(colors.HexColor("#6a707a"))
    canvas.drawString(MARGIN, 1.15 * cm, AUTHOR)
    canvas.drawRightString(PAGE_W - MARGIN, 1.15 * cm, "Page %d" % doc.page)
    canvas.setStrokeColor(colors.HexColor("#c6ccd4"))
    canvas.line(MARGIN, PAGE_H - MARGIN + 0.35 * cm,
                PAGE_W - MARGIN, PAGE_H - MARGIN + 0.35 * cm)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 0.55 * cm,
                      PROJECT + "  |  " + TITLE)
    canvas.restoreState()


def build():
    out = REPORT_DIR / "Automated_Image_Segmentation_and_Object_Measurement.pdf"
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN,
                          rightMargin=MARGIN, topMargin=MARGIN,
                          bottomMargin=MARGIN + 0.4 * cm,
                          title=TITLE, author=AUTHOR)
    frame = Frame(MARGIN, MARGIN + 0.4 * cm, CONTENT_W,
                  PAGE_H - 2 * MARGIN - 0.4 * cm, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=draw_furniture)])
    doc.build(build_story())
    print("written:", out)
    return out


if __name__ == "__main__":
    build()
