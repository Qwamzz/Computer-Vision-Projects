"""
Word renderer of the report.

The content lives in report_content.py as a sequence of neutral blocks, exactly
the same blocks that make_report.py renders into the PDF. This module turns
them into the report DOCX, so the Word file and the PDF always
carry the same text, the same tables and the same figures.

    python run_experiments.py
    python make_report_docx.py
"""

import os
import re
from html import unescape

# Selects the Word page numbers in the contents table of report_content.
os.environ.setdefault("REPORT_TARGET", "docx")

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import report_content as C
from config import FIG_DIR, REPORT_DIR

PAGE_MARGIN_CM = 2.0
CONTENT_W_CM = C.CONTENT_W_CM

NAVY = RGBColor(0x12, 0x32, 0x5C)
BLUE = RGBColor(0x1D, 0x4D, 0x80)
STEEL = RGBColor(0x28, 0x52, 0x7A)
GREY = RGBColor(0x44, 0x48, 0x4F)

HEADER_FILL = "1D4D80"
BAND_FILL = "EEF2F7"
HIGHLIGHT_FILL = "D8ECD8"

BODY_FONT = "Calibri"
MONO_FONT = "Consolas"


# ----------------------------------------------------------------------------
# Low level Word helpers
# ----------------------------------------------------------------------------
def shade_cell(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def paragraph_border_below(paragraph, hex_colour, size_eighths):
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size_eighths))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), hex_colour.lstrip("#"))
    borders.append(bottom)
    p_pr.append(borders)


def add_field(paragraph, instruction):
    """Insert a Word field, used for the page number in the footer."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)
    return run


def keep_with_next(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    keep = OxmlElement("w:keepNext")
    p_pr.append(keep)


# ----------------------------------------------------------------------------
# Inline markup
#
# The content uses a very small subset of markup: <b>bold</b>, a fixed pitch
# span written as <font face='Courier'>...</font>, and HTML entities. This
# parser turns one marked up string into a list of (text, bold, mono) runs.
# ----------------------------------------------------------------------------
TAG = re.compile(r"</?(b|i|font)(?:\s[^>]*)?>", re.IGNORECASE)


def parse_runs(text):
    runs = []
    bold = italic = mono = False
    position = 0
    for match in TAG.finditer(text):
        chunk = text[position:match.start()]
        if chunk:
            runs.append((unescape(chunk), bold, italic, mono))
        tag = match.group(0).lower()
        name = match.group(1).lower()
        closing = tag.startswith("</")
        if name == "b":
            bold = not closing
        elif name == "i":
            italic = not closing
        elif name == "font":
            mono = (not closing) and "courier" in tag
        position = match.end()
    tail = text[position:]
    if tail:
        runs.append((unescape(tail), bold, italic, mono))
    return runs


def write_runs(paragraph, text, size, bold=False, italic=False, colour=None,
               mono=False):
    for chunk, run_bold, run_italic, run_mono in parse_runs(text):
        run = paragraph.add_run(chunk.replace(" ", " "))
        run.font.size = Pt(size)
        run.font.name = MONO_FONT if (mono or run_mono) else BODY_FONT
        run.font.bold = bold or run_bold
        run.font.italic = italic or run_italic
        if colour is not None:
            run.font.color.rgb = colour
    return paragraph


# ----------------------------------------------------------------------------
# Block renderers
# ----------------------------------------------------------------------------
def add_paragraph(doc, text, style_name):
    spec = C.STYLES[style_name]
    par = doc.add_paragraph()
    par.alignment = (WD_ALIGN_PARAGRAPH.CENTER if spec["centre"]
                     else WD_ALIGN_PARAGRAPH.JUSTIFY)
    par.paragraph_format.space_after = Pt(6)
    par.paragraph_format.space_before = Pt(4 if spec["centre"] else 0)
    if spec.get("indent_cm"):
        par.paragraph_format.left_indent = Cm(spec["indent_cm"])
        par.paragraph_format.right_indent = Cm(spec["indent_cm"])
    colour = RGBColor.from_string(spec["colour"].lstrip("#").upper())
    write_runs(par, text, spec["size"], spec["bold"], spec["italic"], colour,
               spec.get("mono", False))
    return par


def add_heading(doc, text, level):
    sizes = {1: 15, 2: 11.8, 3: 10.4}
    colours = {1: NAVY, 2: BLUE, 3: STEEL}
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt({1: 14, 2: 10, 3: 8}[level])
    par.paragraph_format.space_after = Pt({1: 8, 2: 5, 3: 4}[level])
    write_runs(par, text, sizes[level], bold=True, colour=colours[level])
    keep_with_next(par)
    # Marking the outline level makes the headings appear in the Word
    # navigation pane and lets Word build a table of contents from them.
    p_pr = par._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    p_pr.append(outline)
    return par


def add_bullets(doc, items):
    for item in items:
        par = doc.add_paragraph(style="List Bullet")
        par.paragraph_format.space_after = Pt(3)
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        write_runs(par, item, 9.6)


def add_caption(doc, text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(2)
    par.paragraph_format.space_after = Pt(10)
    write_runs(par, text, 8.2, italic=True, colour=GREY)


def add_table(doc, header, rows, col_widths_cm, font_size, highlight_rows):
    # A header of empty strings is a decorative rule in the PDF. In Word an
    # empty shaded row reads as a mistake, so it is dropped.
    show_header = any(str(c).strip() for c in header)
    table = doc.add_table(rows=1 if show_header else 0, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    widths = col_widths_cm
    if not widths:
        widths = [CONTENT_W_CM / float(len(header))] * len(header)

    if show_header:
        for i, text in enumerate(header):
            cell = table.rows[0].cells[i]
            cell.text = ""
            par = cell.paragraphs[0]
            write_runs(par, str(text), font_size, bold=True,
                       colour=RGBColor(0xFF, 0xFF, 0xFF))
            shade_cell(cell, HEADER_FILL)

    for r, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            write_runs(cells[i].paragraphs[0], str(value), font_size)
        if r in highlight_rows:
            fill = HIGHLIGHT_FILL
        elif r % 2 == 1:
            fill = BAND_FILL
        else:
            fill = None
        if fill:
            for cell in cells:
                shade_cell(cell, fill)

    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Cm(widths[i])

    if show_header:
        # Repeat the header row when a table runs over a page boundary.
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        header_flag = OxmlElement("w:tblHeader")
        header_flag.set(qn("w:val"), "true")
        tr_pr.append(header_flag)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def add_figure(doc, name, caption, max_height_cm):
    from PIL import Image as PILImage  # Pillow ships with python-docx

    path = FIG_DIR / name
    with PILImage.open(path) as im:
        iw, ih = im.size
    width = CONTENT_W_CM
    height = width * ih / float(iw)
    if height > max_height_cm:
        height = max_height_cm
        width = height * iw / float(ih)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(2)
    par.add_run().add_picture(str(path), width=Cm(width), height=Cm(height))
    keep_with_next(par)
    add_caption(doc, caption)


def add_pre(doc, text, size=8.0):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.space_before = Pt(2)
    par.paragraph_format.left_indent = Cm(0.6)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        run = par.add_run(line)
        run.font.name = MONO_FONT
        run.font.size = Pt(size)
        if i < len(lines) - 1:
            run.add_break()
    return par


def add_rule(doc, width_cm, colour, thickness, space_before, space_after):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(space_before)
    par.paragraph_format.space_after = Pt(space_after)
    if width_cm and width_cm < CONTENT_W_CM:
        indent = (CONTENT_W_CM - width_cm) / 2.0
        par.paragraph_format.left_indent = Cm(indent)
        par.paragraph_format.right_indent = Cm(indent)
    # Word border widths are given in eighths of a point.
    paragraph_border_below(par, colour, max(4, int(round(thickness * 8))))


def add_spacer(doc, points):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(0)
    par.paragraph_format.space_before = Pt(0)
    run = par.add_run("")
    run.font.size = Pt(points)


# ----------------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------------
def setup_section(doc):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(PAGE_MARGIN_CM)
    section.right_margin = Cm(PAGE_MARGIN_CM)
    section.top_margin = Cm(PAGE_MARGIN_CM)
    section.bottom_margin = Cm(PAGE_MARGIN_CM)
    # The cover page carries no running header and no page number.
    section.different_first_page_header_footer = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    write_runs(header, C.COURSE + "  |  " + C.TITLE, 7.4,
               colour=RGBColor(0x6A, 0x70, 0x7A))
    paragraph_border_below(header, "#C6CCD4", 4)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.LEFT
    write_runs(footer, "%s  " % C.AUTHOR,
               7.4, colour=RGBColor(0x6A, 0x70, 0x7A))
    footer.paragraph_format.tab_stops.add_tab_stop(
        Cm(CONTENT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
    run = footer.add_run("\t")
    run.font.size = Pt(7.4)
    page_run = footer.add_run("Page ")
    page_run.font.size = Pt(7.4)
    page_run.font.color.rgb = RGBColor(0x6A, 0x70, 0x7A)
    field = add_field(footer, "PAGE")
    field.font.size = Pt(7.4)
    field.font.color.rgb = RGBColor(0x6A, 0x70, 0x7A)


def set_default_font(doc):
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(9.6)
    style.paragraph_format.space_after = Pt(6)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), BODY_FONT)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build():
    doc = Document()
    set_default_font(doc)
    setup_section(doc)

    for block in C.BLOCKS:
        kind = block[0]
        if kind == "p":
            add_paragraph(doc, block[1], block[2])
        elif kind == "h1":
            add_heading(doc, block[1], 1)
        elif kind == "h2":
            add_heading(doc, block[1], 2)
        elif kind == "h3":
            add_heading(doc, block[1], 3)
        elif kind == "bullets":
            add_bullets(doc, block[1])
        elif kind == "table":
            add_table(doc, block[1], block[2], block[3], block[4], block[5])
        elif kind == "caption":
            add_caption(doc, block[1])
        elif kind == "figure":
            add_figure(doc, block[1], block[2], block[3])
        elif kind == "pre":
            add_pre(doc, block[1])
        elif kind == "code_heading":
            par = doc.add_paragraph()
            par.paragraph_format.space_before = Pt(10)
            par.paragraph_format.space_after = Pt(3)
            write_runs(par, block[1], 9.4, bold=True, colour=STEEL)
            keep_with_next(par)
        elif kind == "code":
            add_pre(doc, block[1], size=6.0)
        elif kind == "rule":
            add_rule(doc, block[1], block[2], block[3], block[4], block[5])
        elif kind == "spacer":
            add_spacer(doc, block[1])
        elif kind == "page_break":
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        else:
            raise ValueError("Unknown block: %s" % kind)

    doc.core_properties.title = C.TITLE
    doc.core_properties.author = C.AUTHOR
    doc.core_properties.subject = C.PROJECT

    out = REPORT_DIR / "Automated_Image_Segmentation_and_Object_Measurement.docx"
    doc.save(str(out))
    print("written:", out)
    return out


if __name__ == "__main__":
    build()
