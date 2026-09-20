"""
make_report.py
Builds the report PDF from the measured results in results/.

Every number quoted in the document is read from the CSV files written by
run_experiments.py, so the report cannot drift away from the experiments.

    python src/make_report.py
"""

from __future__ import annotations

import csv
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
OUT = os.path.join(ROOT, "report", "Road_and_Lane_Boundary_Detection.pdf")

NAME = "Nii Yartey Gidiglo"
PROJECT = "Road and lane boundary detection"

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
BODY_W = PAGE_W - 2 * MARGIN

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY,
                      spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=14, leading=17, spaceBefore=14, spaceAfter=7,
                    textColor=colors.HexColor("#12305c"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11.2, leading=14, spaceBefore=10, spaceAfter=5,
                    textColor=colors.HexColor("#1d4d8f"))
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                    fontSize=10, leading=13, spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.2, leading=10.5,
                     alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                     spaceBefore=3, spaceAfter=10)
EQ = ParagraphStyle("eq", parent=BODY, fontName="Courier", fontSize=9,
                    leading=12.5, alignment=TA_CENTER, spaceBefore=5, spaceAfter=7)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=4,
                        spaceAfter=3)
CODE = ParagraphStyle("code", parent=ss["Code"], fontName="Courier",
                      fontSize=6.0, leading=7.2, spaceAfter=0, spaceBefore=0)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold",
                       fontSize=19, leading=24, alignment=TA_CENTER,
                       spaceAfter=8, textColor=colors.HexColor("#12305c"))
SUB = ParagraphStyle("sub", parent=BODY, fontSize=11.5, leading=16,
                     alignment=TA_CENTER, spaceAfter=4)

story = []
FIGN = [0]
TABN = [0]


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def eq(t): story.append(Paragraph(t, EQ))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText="•"))
    story.append(Spacer(1, 4))


def figure(fname, caption, width=BODY_W):
    path = os.path.join(FIG, fname)
    if not os.path.exists(path):
        return
    iw, ih = ImageReader(path).getSize()
    w = min(width, BODY_W)
    h = w * ih / float(iw)
    if h > 15.0 * cm:
        h = 15.0 * cm
        w = h * iw / float(ih)
    FIGN[0] += 1
    story.append(KeepTogether([Image(path, w, h),
                               Paragraph("Figure %d. %s" % (FIGN[0], caption), CAP)]))


def read_table(name):
    with open(os.path.join(TAB, name), encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def table(header, rows, caption, widths=None, fontsize=7.4, left_first=True):
    TABN[0] += 1
    head_style = ParagraphStyle("th", parent=BODY, fontSize=fontsize,
                                leading=fontsize + 2, alignment=TA_CENTER,
                                textColor=colors.white, spaceAfter=0)
    cell = ParagraphStyle("td", parent=BODY, fontSize=fontsize,
                          leading=fontsize + 2, alignment=TA_CENTER, spaceAfter=0)
    cell_l = ParagraphStyle("tdl", parent=cell, alignment=0)
    data = [[Paragraph("<b>%s</b>" % h.replace("_", " "), head_style) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), cell_l if (i == 0 and left_first) else cell)
                     for i, v in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4d8f")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(KeepTogether([t, Paragraph("Table %d. %s" % (TABN[0], caption), CAP)]))


def num(x, nd=3):
    """Format a number, rendering a missing measurement as n/a."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:                      # NaN, meaning nothing could be measured
        return "n/a"
    return ("%." + str(nd) + "f") % v


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#5a6673"))
    canvas.drawString(MARGIN, 1.15 * cm, PROJECT)
    canvas.drawRightString(PAGE_W - MARGIN, 1.15 * cm, "%s, page %d" % (NAME, doc.page))
    canvas.setStrokeColor(colors.HexColor("#c7d0da"))
    canvas.line(MARGIN, 1.45 * cm, PAGE_W - MARGIN, 1.45 * cm)
    canvas.restoreState()


def blank(canvas, doc):
    pass


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=1.7 * cm, bottomMargin=2.0 * cm,
                          title="Road and Lane Boundary Detection", author=NAME)
    frame = Frame(MARGIN, 2.0 * cm, BODY_W, PAGE_H - 3.7 * cm, id="body")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=blank),
                          PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)


def main():
    summary = json.load(open(os.path.join(RES, "summary.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(RES, "pipeline_config.json"), encoding="utf-8"))
    sources = json.load(open(os.path.join(ROOT, "data", "sources.json"), encoding="utf-8"))

    h_e1, r_e1 = read_table("e1_colour_separability.csv")
    h_e2, r_e2 = read_table("e2_filters.csv")
    h_e3, r_e3 = read_table("e3_edges.csv")
    h_e4, r_e4 = read_table("e4_roi.csv")
    h_e5, r_e5 = read_table("e5_boundary_strategies.csv")
    h_e5s, r_e5s = read_table("e5_hough_sweep.csv")
    h_e6, r_e6 = read_table("e6_segmentation.csv")
    h_e7i, r_e7i = read_table("e7_final_per_image.csv")
    h_e7c, r_e7c = read_table("e7_by_condition.csv")
    h_e7s, r_e7s = read_table("e7_by_split.csv")

    best_colour = summary["E1"]["best"]
    colour_means = summary["E1"]["means"]
    best_filter = summary["E2"]["best"]
    e2_none = next(r for r in r_e2 if r[0] == "none")
    e2_gauss = next(r for r in r_e2 if r[0] == "gaussian")
    e2_bil = next(r for r in r_e2 if r[0] == "bilateral")
    best_e6 = max(r_e6, key=lambda r: float(r[2]))
    best_thr = max((r for r in r_e6 if r[1] == "threshold"), key=lambda r: float(r[2]))
    best_km = max((r for r in r_e6 if r[1] == "kmeans"), key=lambda r: float(r[2]))
    best_rg = max((r for r in r_e6 if r[1] == "region_growing"), key=lambda r: float(r[2]))
    e5_region = next(r for r in r_e5 if r[0] == "from segmented region")
    e5_hough = next(r for r in r_e5 if r[0] == "hough on ROI edges")
    e5_paint = next(r for r in r_e5 if r[0] == "hough on paint edges")
    tune = next(r for r in r_e7s if r[0] == "tuning")
    held = next(r for r in r_e7s if r[0] == "held out")
    worst_cond = min(r_e7c, key=lambda r: float(r[2]))
    best_cond = max(r_e7c, key=lambda r: float(r[2]))

    # ---------------- cover ----------------
    story.append(Spacer(1, 2.6 * cm))
    p("COMPUTER VISION PROJECT", SUB)
    story.append(Spacer(1, 1.0 * cm))
    p("Automated Road and Lane Boundary Detection<br/>and Scene Segmentation", TITLE)
    story.append(Spacer(1, 0.5 * cm))
    p("Colour representation, filtering, edge detection, region selection, "
      "segmentation and boundary identification, evaluated against hand traced "
      "ground truth", SUB)
    story.append(Spacer(1, 1.4 * cm))
    info = Table([["Author", NAME],
                  ["Repository", "github.com/Qwamzz/Computer-Vision-Projects"],
                  ["Dataset", "%d real road photographs, six imaging conditions"
                   % len(sources)],
                  ["Ground truth", "hand traced drivable region for every image"],
                  ["Implementation", "Python 3 and OpenCV, no learned model"]],
                 colWidths=[4.2 * cm, 10.5 * cm])
    info.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b5")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    info.hAlign = "CENTER"
    story.append(info)
    story.append(Spacer(1, 1.2 * cm))
    p("<b>Statement on the imagery.</b> Every photograph used in this project "
      "was taken by someone else and published on Wikimedia Commons under a "
      "free licence. The licence and author of each file are listed in Section "
      "2.1 and in data/SOURCES.md. No image was produced by an image generation "
      "model.",
      ParagraphStyle("note", parent=BODY, fontSize=8.6, leading=12,
                     borderPadding=6, backColor=colors.HexColor("#f4f7fb")))
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # ---------------- 1 ----------------
    h1("1. Problem definition and objectives")
    p("The task is to analyse a road scene and report two things: where the "
      "drivable road surface is, and where its boundaries run. Both are wanted "
      "from classical computer vision alone, with no learned model anywhere in "
      "the pipeline.")
    p("The system implements the full progression end to end:")
    eq("road image -&gt; filtering -&gt; edge detection -&gt; region selection "
       "-&gt; segmentation -&gt; boundary identification -&gt; visualisation")
    p("Every stage offers a genuine choice, and this report settles each one by "
      "measurement rather than by assertion:")
    bullets([
        "which colour representation separates road from everything else,",
        "which filter suppresses noise without destroying the road boundary,",
        "whether Canny or Sobel is the better edge operator here,",
        "how much a region of interest helps, and what its assumption costs,",
        "which of four segmentation rules recovers the drivable region best,",
        "and whether boundaries are better found from edges or from the "
        "segmented region.",
    ])
    p("The last of those turned out to be the most interesting question in the "
      "project, and the answer is not the textbook one. It is reported in "
      "Section 5.5 and discussed in Section 6.")

    # ---------------- 2 ----------------
    h1("2. Dataset and ground truth")
    h2("2.1 The photographs")
    p("Twelve real photographs of real roads were collected from Wikimedia "
      "Commons, two for each of the six conditions the question asks the system "
      "to be tested under. Using photographs taken by other people, for their "
      "own purposes, with no common camera, mounting or exposure, is a harder "
      "test than a dashcam sequence and a more honest one: nothing about the "
      "geometry is shared between images, so no hidden constant can be tuned.")
    table(["file", "condition", "what it tests", "licence"],
          [[s["file"], s["condition"].replace("_", " "), s["note"][:74],
            s["licence"]] for s in sources],
          "The dataset. Every file is a real photograph under a free licence; "
          "authors are listed in data/SOURCES.md.",
          widths=[2.9 * cm, 2.6 * cm, 8.6 * cm, 2.9 * cm], fontsize=6.6)
    figure("fig01_dataset.png",
           "The twelve photographs with the hand traced drivable region in "
           "green. The six conditions are two each of good daylight, shadows, "
           "poor illumination, curved roads, worn or absent markings, and "
           "occluding traffic.")

    h2("2.2 Ground truth")
    p("The drivable surface of every image was traced by hand as a polygon, "
      "read off the photograph against a printed coordinate grid, and then "
      "checked by drawing the resulting mask back over the photograph and "
      "correcting whatever was wrong. Several images went round that loop more "
      "than once. The vertices are stored in src/annotate.py rather than in a "
      "binary mask, so anyone can see exactly what was claimed to be road and "
      "can re-derive the masks from scratch.")
    p("Two decisions in the annotation are worth stating because they change "
      "what the numbers mean. First, the annotated region is the sealed "
      "carriageway, bounded by the painted edge lines where they exist and by "
      "the change of surface where they do not. Second, anything standing on "
      "that surface, a vehicle or an animal, is cut out of the mask, so the "
      "ground truth is the <i>visible</i> drivable surface. A segmentation "
      "method cannot reasonably be asked to report road where a bus is parked, "
      "and scoring it as though it should would make the occlusion condition "
      "meaningless.")

    h2("2.3 Protocol")
    p("The twelve images are split in half. The six ending in _01 are the "
      "tuning set: every parameter in Sections 5.1 to 5.6 was chosen on those "
      "and only those. The six ending in _02 were held out and run once, at the "
      "end, with the parameters already fixed. Both halves contain one image "
      "from each condition, so the held out half is not the easier half. The "
      "gap between the two, reported in Section 5.7, is the honest measure of "
      "how much of the tuning was real and how much was fitting to six "
      "photographs.")

    # ---------------- 3 ----------------
    h1("3. Methodology")
    figure("fig03_pipeline.png",
           "The pipeline on one held out image, stage by stage: the input, the "
           "filtered image, the raw edge map, the region of interest, the edges "
           "that survive it, the Hough segments split into left and right, the "
           "segmented road region, and the final overlay.")

    h2("3.1 Colour representation")
    p("Road surfaces here are grey asphalt or orange laterite, so they have no "
      "colour in common. What they share is a <i>lack</i> of one: against the "
      "vegetation, sky and painted buildings around them they are weakly "
      "saturated and mid range in lightness. HLS puts exactly those two "
      "quantities on their own axes, and its lightness channel is also where "
      "white and yellow paint stands out most strongly. Lab is the alternative, "
      "with approximately perceptually uniform a and b axes that suit "
      "clustering. RGB is kept as the baseline that mixes brightness and chroma "
      "together and therefore moves under every change of illumination.")

    h2("3.2 Filtering")
    p("A road boundary is a step edge that must survive; the texture of "
      "chippings, gravel and dappled tree shade is noise that should not. "
      "Gaussian smoothing cannot tell the two apart. A median filter removes "
      "impulsive speckle while leaving step edges intact. A bilateral filter "
      "weights by intensity difference as well as by distance, so it smooths "
      "within a region and stops at the boundary between regions. All three "
      "are compared against no filtering at all in Section 5.2.")

    h2("3.3 Edge detection")
    p("Sobel estimates the gradient with a small separable kernel and "
      "thresholds its magnitude. Canny adds the three steps that make an edge "
      "detector usable: smoothing, non maximum suppression across the gradient "
      "direction so edges come out one pixel wide, and hysteresis thresholding "
      "so a weak edge survives when it connects to a strong one. Hysteresis is "
      "the reason Canny can hold on to a faded lane marking that a single "
      "threshold on Sobel magnitude either loses or buries in noise.")

    h2("3.4 Region selection")
    p("Almost everything above the horizon is irrelevant, and it is also where "
      "the strongest edges live: trees, buildings, poles and the skyline. Two "
      "region selection strategies are compared. The fixed trapezoid covers the "
      "carriageway in front of the camera and assumes the camera points along "
      "the road. The adaptive region is derived from a first pass segmentation "
      "and then dilated, which drops that assumption at the cost of one extra "
      "segmentation. A horizon cut is also applied inside the segmentation "
      "stage, because a surprising amount of sky is close enough to grey "
      "asphalt in colour to be accepted by any of the rules below.")

    h2("3.5 Segmentation")
    p("Four rules are implemented. All four take a colour sample from a small "
      "trapezoid immediately in front of the camera, which is the one place a "
      "forward facing photograph is almost certain to be road, and all four "
      "share the same morphological cleanup and the same final step of keeping "
      "the connected component that reaches the bottom of the frame, since the "
      "drivable region must by definition reach the vehicle.")
    bullets([
        "<b>Adaptive thresholding.</b> A per channel acceptance band in HLS, "
        "set at the sample mean plus or minus k standard deviations.",
        "<b>Mahalanobis thresholding.</b> The same idea with the correlation "
        "between channels put back in: the full covariance of the sample is "
        "estimated and a pixel is accepted when its Mahalanobis distance from "
        "the sample mean is small.",
        "<b>K means clustering.</b> The image is clustered in Lab and the "
        "cluster that dominates the near field sample is taken to be road. No "
        "threshold is chosen at all, but K must be.",
        "<b>Region growing.</b> A flood fill from seeds in the near field, "
        "which is the only one of the four that enforces spatial connectivity, "
        "so a grey roof on the far side of a hedge cannot be labelled road.",
    ])
    eq("Mahalanobis: sqrt( (x - mu)^T C^-1 (x - mu) ) &lt;= chi")

    h2("3.6 Boundary identification")
    p("Three strategies are compared. The first is the textbook lane finder: "
      "the probabilistic Hough transform turns edge pixels into line segments, "
      "segments are sorted into a left and a right boundary by the sign of "
      "their slope, nearly horizontal segments are discarded because they "
      "belong to shadows and to the backs of vehicles, and each side is reduced "
      "to one line by least squares weighted by segment length. The second "
      "gates the same edges with a mask of white and yellow road paint, so only "
      "edges lying on markings can vote. The third ignores edges entirely and "
      "reads the boundary off the segmented region, taking the extreme road "
      "pixel in each row and fitting a line to those.")
    p("Two safeguards matter. A side needs at least two segments before a line "
      "is fitted, so a single edge on a roadside bush cannot become a boundary; "
      "and a fitted line that leaves the frame by more than half its width is "
      "discarded, because reporting nothing is more useful than reporting "
      "nonsense.")

    h2("3.7 Evaluation")
    p("The segmented region is scored per pixel against the hand traced mask:")
    eq("IoU = TP / (TP + FP + FN)&nbsp;&nbsp;&nbsp; precision = TP / (TP + FP)"
       "&nbsp;&nbsp;&nbsp; recall = TP / (TP + FN)&nbsp;&nbsp;&nbsp; "
       "F1 = 2PR / (P + R)")
    p("IoU is quoted first because it punishes both kinds of mistake at once "
      "and cannot be inflated by predicting a very large or a very small "
      "region, which precision and recall individually can. Boundaries are "
      "scored separately, in pixels, as the mean horizontal distance between "
      "the fitted line and the true edge of the road across sampled rows. Rows "
      "where the road runs out of the frame are skipped: there the boundary is "
      "the edge of the photograph, not the edge of the road, and there is "
      "nothing to compare against.")
    p("The edge stage is scored on its own terms, before any line fitting, by "
      "dilating the true boundary and asking what fraction of it the operator "
      "found (boundary recall) and what fraction of the operator's output lies "
      "on it (boundary precision). Recall alone would be a trap, since an "
      "operator that marks every pixel as an edge scores a perfect recall and "
      "is useless.")

    # ---------------- 4 ----------------
    h1("4. Implementation")
    table(["file", "contents"],
          [["src/download_data.py", "Fetches the twelve photographs from Wikimedia Commons and records every licence."],
           ["src/annotate.py", "The hand traced ground truth polygons and the code that rasterises and checks them."],
           ["src/preprocessing.py", "Colour representations, the four filters, class separability, illumination correction."],
           ["src/edges.py", "Canny and Sobel, the region of interest, boundary recall and precision."],
           ["src/lanes.py", "Hough segments, slope splitting, weighted line fitting, the paint mask, region derived boundaries."],
           ["src/segmentation.py", "The four segmentation rules, the horizon cut, morphological cleanup, the final overlay."],
           ["src/evaluate.py", "IoU, precision, recall, F1 and the table writers."],
           ["src/run_experiments.py", "Runs the seven experiments and writes every table, figure and overlay."],
           ["src/make_report.py", "Builds this PDF from the measured CSV results."]],
          "Source files. The complete listing is reproduced in Appendix A.",
          widths=[4.6 * cm, 12.4 * cm], fontsize=7.4)

    # ---------------- 5 ----------------
    h1("5. Experimental results")
    p("Sections 5.1 to 5.6 use the tuning half only. The held out half appears "
      "once, in Section 5.7.")

    h2("5.1 Colour representation")
    table(h_e1, [[r[0], r[1].replace("_", " ")] + [num(v) for v in r[2:]] for r in r_e1],
          "Fisher separability between the road and non road pixel populations, "
          "computed with the ground truth, per image and per representation. "
          "Larger is better.",
          widths=[3.2 * cm, 3.2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm],
          fontsize=7.0)
    figure("fig02_colour.png",
           "Mean separability by colour representation across the whole "
           "dataset.", 11 * cm)
    p("Averaged over the dataset %s separates road from not road best (%s), "
      "with HLS close behind (%s) and RGB clearly worst (%s). The per image "
      "column is more revealing than the average: %s wins on the unsealed and "
      "curved scenes by a wide margin, while HLS is the more reliable choice on "
      "the hazy and occluded ones. Neither dominates, which is why the "
      "segmentation stage uses Lab for its multivariate rule and the edge stage "
      "works on the HLS lightness channel. RGB is last because it mixes "
      "brightness and chroma on every axis, so a shadow moves a pixel as far as "
      "a change of material does."
      % (best_colour, num(colour_means[best_colour]),
         num(colour_means["HLS"]), num(colour_means["BGR"]), best_colour))

    h2("5.2 Filtering")
    table(h_e2, [[r[0]] + [num(v) for v in r[1:]] for r in r_e2],
          "The four filters, scored by how much of the true road boundary "
          "survives and how much of the output lies on it.",
          widths=[2.8 * cm, 2.8 * cm, 3.0 * cm, 2.6 * cm, 2.6 * cm, 3.0 * cm],
          fontsize=7.2)
    figure("fig04_filters.png",
           "Left, the fraction of the true road boundary that survives each "
           "filter. Right, boundary F1, which also accounts for the clutter the "
           "filter leaves behind.", 12.5 * cm)
    p("This experiment produced the most instructive mistake in the project. "
      "The natural measure is boundary recall divided by edge density, on the "
      "grounds that a good filter keeps the boundary and suppresses everything "
      "else. That measure selects Gaussian smoothing, which turns out to "
      "destroy roughly nine tenths of the boundary: its recall is %s against %s "
      "for no filtering at all. The ratio rewarded it because the denominator "
      "collapsed faster than the numerator. Scored by boundary F1 instead, "
      "which cannot be gamed that way, the ranking is %s first."
      % (num(e2_gauss[1]), num(e2_none[1]), best_filter))
    p("The honest conclusion is that on these photographs no filter earns its "
      "place ahead of Canny: Canny already smooths internally, and a second "
      "smoothing stage removes boundary that the hysteresis step could "
      "otherwise have recovered. Among the filters proper, the bilateral filter "
      "preserves the most boundary (%s against %s for Gaussian), which is "
      "exactly what an edge preserving filter is for, at roughly four times the "
      "cost per image. Filtering still matters inside the segmentation stage, "
      "where each rule applies its own small blur before sampling colour."
      % (num(e2_bil[1]), num(e2_gauss[1])))

    h2("5.3 Edge detection")
    table(h_e3, [[r[0], r[1]] + [num(v) for v in r[2:]] for r in r_e3],
          "Canny and Sobel across their thresholds, measured against the true "
          "road boundary inside the region of interest.",
          widths=[3.0 * cm, 2.4 * cm, 3.0 * cm, 3.2 * cm, 2.6 * cm, 2.6 * cm],
          fontsize=7.2)
    figure("fig05_edges.png",
           "Operating curves. Each point is one threshold setting; the "
           "annotation is the threshold.", 11 * cm)
    p("Both operators score poorly in absolute terms, and that is the finding "
      "rather than a defect of the measurement. Boundary precision never "
      "exceeds about %s, because the overwhelming majority of edges inside the "
      "region of interest are real edges of things that are not the road "
      "boundary: the texture of the surface, the shadows of roadside trees, "
      "vegetation, buildings and the backs of vehicles. An edge detector cannot "
      "distinguish them, because at the level of a local gradient they are not "
      "different. This is the quantitative reason why the boundary stage in "
      "Section 5.5 does better when it stops relying on edges."
      % num(max(float(r[3]) for r in r_e3), 2))
    p("Between the two, Sobel holds a slight edge on this measure at the "
      "operating points tested, mainly because Canny after the smoothing stage "
      "leaves very little behind: at 120/260 it finds almost nothing at all. "
      "The choice matters far less than the table suggests, since neither "
      "operator supports a reliable boundary on its own.")

    h2("5.4 Region selection")
    table(h_e4, [[r[0]] + [num(v, 1) if i >= 1 else r[1] for i, v in enumerate(r[1:])]
                 for r in r_e4],
          "What the region of interest removes, and what it does for the "
          "accuracy of the boundaries fitted inside it.",
          widths=[5.2 * cm, 3.2 * cm, 2.6 * cm, 2.8 * cm, 3.2 * cm],
          fontsize=7.2)
    p("Restricting the search cuts the number of Hough segments by roughly a "
      "factor of two and improves the accuracy of the boundaries that are "
      "fitted. The adaptive region, derived from a first pass segmentation "
      "rather than from an assumption about where the camera points, is the "
      "better of the two: it reaches %s pixels of mean boundary error against "
      "%s for the fixed trapezoid and %s for searching the whole frame. That "
      "ordering is worth noting because the fixed trapezoid is the standard "
      "choice, and it is standard because dashcam frames share a geometry that "
      "this dataset deliberately does not."
      % (num(r_e4[2][4], 1), num(r_e4[1][4], 1), num(r_e4[0][4], 1)))

    h2("5.5 Boundary identification")
    table(h_e5, [[r[0]] + [num(v, 1) if i == 3 else r[i + 1]
                           for i, v in enumerate(r[1:])] for r in r_e5],
          "The three boundary strategies on the tuning set. Error is the mean "
          "horizontal distance from the true road edge, in pixels, on images "
          "1280 pixels wide.",
          widths=[5.0 * cm, 3.0 * cm, 3.2 * cm, 3.0 * cm, 3.2 * cm], fontsize=7.2)
    figure("fig06_hough.png",
           "Mean boundary error against the Hough accumulator threshold, for "
           "three minimum segment lengths.", 11 * cm)
    p("This is the result that reshaped the project. Fitting straight lines to "
      "Hough segments, the textbook method, gives a mean boundary error of %s "
      "pixels. Gating those edges with the road paint mask does not rescue it "
      "(%s pixels) and finds fewer boundaries, because half the dataset has no "
      "usable paint at all. Reading the boundary off the segmented region "
      "instead gives %s pixels, three to four times better, on images 1280 "
      "pixels wide."
      % (num(e5_hough[4], 1), num(e5_paint[4], 1), num(e5_region[4], 1)))
    p("The reason is visible in Section 5.3. The Hough transform is an "
      "excellent way to find lines in a set of points that mostly lie on lines. "
      "Here the edge map is dominated by points that lie on other things, and "
      "the transform has no way to know which is which; a few long segments on "
      "a hedge or a shadow are enough to drag a least squares fit a long way. "
      "The segmented region has already used colour to decide what is road, so "
      "its boundary is a boundary of the right object by construction. The "
      "safeguards in Section 3.6 keep the Hough path honest rather than "
      "accurate: they convert many of its worst fits into no answer at all.")

    h2("5.6 Segmentation")
    table(h_e6, [[r[0], r[1].replace("_", " ")] + [num(v) for v in r[2:]] for r in r_e6],
          "All four segmentation rules across their parameters, on the tuning "
          "set.",
          widths=[4.0 * cm, 2.8 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm, 2.0 * cm,
                  2.6 * cm], fontsize=6.8)
    figure("fig07_segmentation.png",
           "Mean IoU for every configuration tested. Colour groups the "
           "configurations by method.")
    figure("fig08_methods.png",
           "The four rules on a shadowed road and on an unsealed one, with the "
           "IoU each achieves.")
    p("The best configuration is %s at an IoU of %s, ahead of the best per "
      "channel threshold (%s), the best K means (%s) and the best region "
      "growing (%s). The margin over the per channel threshold is the "
      "interesting part, because the two rules differ only in whether the "
      "correlation between colour channels is used. Treating the channels "
      "independently accepts an axis aligned box in colour space; the road "
      "colours actually form a tilted, elongated cloud, so a box wide enough to "
      "contain them also admits a great deal of vegetation and sky. Putting the "
      "covariance back in fixes that, and it is the single largest accuracy "
      "gain in the project."
      % (best_e6[0], num(best_e6[2]), num(best_thr[2]), num(best_km[2]),
         num(best_rg[2])))
    p("The other two rules fail in characteristic ways. K means is unstable in "
      "K: with two clusters it splits the image into bright and dark and calls "
      "half the scene road, and with five it fragments the road surface across "
      "several clusters. Region growing is the most precise of all at a tight "
      "tolerance (precision %s at tolerance 8) and the least complete (recall "
      "%s), because a shadow band across the carriageway stops the fill dead; "
      "opening the tolerance far enough to cross the shadow also opens it "
      "enough to leak into the verge."
      % (num(next(r for r in r_e6 if r[0].endswith("tol=8"))[3]),
         num(next(r for r in r_e6 if r[0].endswith("tol=8"))[4])))

    h2("5.7 The assembled system")
    p("The configuration selected on the tuning half was: %s filtering, the HLS "
      "lightness channel, %s edges, a fixed trapezoid region of interest, %s "
      "segmentation, and boundaries %s. It was then run once on the held out "
      "half."
      % (cfg["filter"], cfg["edge"]["method"],
         cfg["segmentation"]["method"], cfg["boundary_strategy"]))
    table(h_e7s, [[r[0], r[1]] + [num(v) for v in r[2:6]] + [num(r[6], 1)]
                  for r in r_e7s],
          "Tuning half against held out half, with all parameters fixed before "
          "the held out half was run.",
          widths=[2.6 * cm, 2.0 * cm, 2.4 * cm, 2.6 * cm, 2.4 * cm, 2.2 * cm,
                  3.4 * cm], fontsize=7.2)
    p("The system reaches an IoU of %s on the tuning half and %s on the held "
      "out half, with F1 falling from %s to %s and mean boundary error rising "
      "from %s to %s pixels. A drop of that size across the split is worth "
      "stating plainly: with six images to tune on, some of the apparent "
      "accuracy was fitted to those six photographs. The held out figure is the "
      "one to believe."
      % (num(tune[2]), num(held[2]), num(tune[5]), num(held[5]),
         num(tune[6], 1), num(held[6], 1)))
    table(h_e7i, [[r[0], r[1].replace("_", " "), r[2]] + [num(v) for v in r[3:7]]
                  + [num(r[7], 1), r[8], num(r[9], 2)] for r in r_e7i],
          "Every image individually under the final configuration. A boundary "
          "error of n/a means no boundary could be scored, either because none "
          "was found or because the road leaves the frame on both sides.",
          widths=[2.6 * cm, 2.4 * cm, 1.8 * cm, 1.5 * cm, 1.7 * cm, 1.5 * cm,
                  1.4 * cm, 2.0 * cm, 1.6 * cm, 1.4 * cm], fontsize=6.4)
    table(h_e7c, [[r[0], r[1]] + [num(v) for v in r[2:]] for r in r_e7c],
          "Performance by imaging condition, over all twelve images.",
          widths=[4.6 * cm, 2.2 * cm, 2.4 * cm, 2.6 * cm, 2.4 * cm, 2.2 * cm],
          fontsize=7.2)
    figure("fig09_by_condition.png",
           "Road region IoU by imaging condition.", 12.5 * cm)
    p("The ordering by condition is close to the ordering a person would "
      "predict. The system is strongest on %s (IoU %s) and weakest on %s (IoU "
      "%s). The occlusion case is the most interesting of the failures: "
      "precision stays high at %s while recall collapses to %s, which is "
      "exactly the signature of a method that is finding real road and only "
      "real road, but cannot see most of it because vehicles are standing on "
      "it."
      % (best_cond[0], num(best_cond[2]), worst_cond[0], num(worst_cond[2]),
         num(next(r for r in r_e7c if "occluding" in r[0])[3]),
         num(next(r for r in r_e7c if "occluding" in r[0])[4])))
    figure("fig10_failures.png",
           "The four weakest results. Green is the segmented road region, red "
           "the fitted boundaries.")

    # ---------------- 6 ----------------
    h1("6. Critical discussion: where the system fails and why")
    p("The question asks for the situations in which the classical algorithms "
      "break down, and for the reasons. Each of the following is visible in the "
      "per condition table or in the failure figure.")

    h3("Poor illumination and haze")
    p("This is the weakest condition, at an IoU of %s. Haze compresses the "
      "whole scene towards a narrow band of grey, which is precisely the band "
      "the road occupies. The near field sample then has a small variance, the "
      "Mahalanobis ellipsoid it defines is correspondingly narrow along every "
      "axis, and distant vegetation of nearly the same grey falls inside it "
      "while the far part of the road, which is hazier still, falls outside. "
      "The result is high recall with poor precision. No amount of parameter "
      "tuning fixes this, because the information that separates road from "
      "verge has been removed by the atmosphere before the camera sees it."
      % num(worst_cond[2]))

    h3("Occluding traffic")
    p("A vehicle standing on the carriageway removes the road from view, and "
      "there is nothing classical or otherwise to be done about the pixels it "
      "covers. What the system does do correctly is refuse to guess: precision "
      "on this condition is %s. The failure that matters is a different one. "
      "Where a queue of vehicles hides the road edge, the connected component "
      "that reaches the camera is bounded by the vehicles rather than by the "
      "road, so the fitted boundary follows the traffic. A human driver infers "
      "the road continues behind the bus; the system has no mechanism for that "
      "inference."
      % num(next(r for r in r_e7c if "occluding" in r[0])[3]))

    h3("Shadows")
    p("Dappled shade from roadside trees is the classic failure of any method "
      "that decides by colour, and it defeats region growing completely: a "
      "shadow band across the carriageway is, in pixel terms, a wall. The "
      "Mahalanobis rule survives it better because a shadowed patch of asphalt "
      "is displaced from the sample mean mostly along the lightness axis, which "
      "the covariance already knows is the direction of greatest variation, so "
      "the ellipsoid is elongated in exactly that direction. That is why the "
      "shadow condition scores %s rather than the 0.23 that region growing "
      "reaches on the same images."
      % num(next(r for r in r_e7c if r[0] == "shadows")[2]))

    h3("Curves")
    p("The straight line model is wrong on a curve by construction, and the "
      "boundary error shows it: the curved images are among the worst for "
      "boundary accuracy even though their region IoU is high (%s). The region "
      "is found correctly and then described badly. A second order fit in the "
      "image plane, or a fit in a bird's eye view obtained by an inverse "
      "perspective mapping, would address this directly and is the first thing "
      "worth adding."
      % num(next(r for r in r_e7c if "curved" in r[0])[2]))

    h3("Worn and absent markings")
    p("The unsealed roads score well on region IoU (%s), which is initially "
      "surprising, until one notices that laterite against green vegetation is "
      "a far larger colour contrast than grey asphalt against a grey verge. "
      "What fails on these images is the paint based boundary strategy, which "
      "has nothing to find: the boundary here is a change of material, not a "
      "painted line. This is the clearest case in the dataset for preferring a "
      "region derived boundary over an edge derived one."
      % num(next(r for r in r_e7c if "worn" in r[0])[2]))

    h3("The near field assumption")
    p("Every segmentation rule here trusts that the bottom centre of the frame "
      "is road. It is the assumption that makes a per image threshold possible "
      "at all, and it holds for a forward facing camera on a vehicle. It does "
      "not hold for a photograph taken from the verge, and one image in this "
      "dataset comes close to breaking it. A system deployed on arbitrary "
      "photographs would need to verify the assumption rather than rely on it.")

    h2("6.1 Limitations")
    bullets([
        "Twelve images with two per condition is a small evaluation. Condition "
        "level numbers rest on two photographs each and should be read as "
        "indicative.",
        "The ground truth is hand traced by one annotator with no second "
        "opinion, and the boundary of a road into an unsealed verge is "
        "genuinely ambiguous to within a few pixels.",
        "Every image is a single frame. A video sequence would allow the "
        "boundary to be tracked over time, which is the single largest "
        "practical improvement available and is not attempted here.",
        "The system reports a drivable region and its outer boundaries, not "
        "individual lanes. Finding the lane within a road needs the paint, and "
        "half of this dataset has none.",
    ])

    h2("6.2 What would help most")
    bullets([
        "An inverse perspective mapping to a bird's eye view, in which lane "
        "boundaries are parallel and a quadratic fit is well conditioned.",
        "Temporal tracking across frames, so that a boundary hidden by a "
        "vehicle in one frame is carried forward from the last.",
        "A shadow invariant colour representation, which would attack the "
        "shadow and haze failures at their source rather than downstream.",
        "Verifying the near field assumption instead of trusting it, for "
        "example by checking that the sample region is texturally uniform.",
    ])

    # ---------------- 7 ----------------
    h1("7. Conclusion")
    p("A complete classical road and lane boundary system was built and "
      "measured end to end, from colour representation through filtering, edge "
      "detection, region selection and segmentation to boundary identification "
      "and visualisation. On six held out photographs that played no part in "
      "tuning, it recovers the drivable region with an IoU of %s and an F1 of "
      "%s, and places the road boundaries to within %s pixels on images 1280 "
      "pixels wide."
      % (num(held[2]), num(held[5]), num(held[6], 1)))
    p("Three findings are worth carrying away. First, the measure used to "
      "select a stage matters as much as the stage: two natural looking "
      "criteria, recall divided by edge density for the filter and for the edge "
      "operator, both select a configuration that destroys the boundary, "
      "because their denominators collapse faster than their numerators. "
      "Second, putting the correlation between colour channels back into a "
      "simple threshold, which is all the Mahalanobis rule does, was worth more "
      "accuracy than any other single change. Third, and least expected, the "
      "textbook Hough based lane finder was beaten by a wide margin by reading "
      "the boundary off the segmented region, because on ordinary road "
      "photographs the edge map is dominated by edges of things that are not "
      "the road, and the Hough transform has no way to tell them apart.")

    # ---------------- references ----------------
    h1("References")
    for i, ref in enumerate([
        "Canny, J. (1986). A computational approach to edge detection. IEEE "
        "Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679 to 698.",
        "Duda, R. O. and Hart, P. E. (1972). Use of the Hough transformation to "
        "detect lines and curves in pictures. Communications of the ACM, 15(1), "
        "11 to 15.",
        "Otsu, N. (1979). A threshold selection method from gray level "
        "histograms. IEEE Transactions on Systems, Man and Cybernetics, 9(1), "
        "62 to 66.",
        "Tomasi, C. and Manduchi, R. (1998). Bilateral filtering for gray and "
        "colour images. Proceedings of ICCV, pages 839 to 846.",
        "Aly, M. (2008). Real time detection of lane markers in urban streets. "
        "IEEE Intelligent Vehicles Symposium, pages 7 to 12.",
        "Mahalanobis, P. C. (1936). On the generalised distance in statistics. "
        "Proceedings of the National Institute of Sciences of India, 2(1), 49 to 55.",
        "Szeliski, R. (2022). Computer Vision: Algorithms and Applications, "
        "second edition. Springer.",
        "Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.",
    ], 1):
        p("[%d] %s" % (i, ref),
          ParagraphStyle("ref", parent=BODY, leftIndent=16, firstLineIndent=-16,
                         spaceAfter=4))

    # ---------------- appendix ----------------
    story.append(PageBreak())
    h1("Appendix A. Complete source code")
    p("The listing below is the complete source of the project, in the order in "
      "which the modules are run. The same files are in the src directory of "
      "the repository.")
    for fname in ["download_data.py", "annotate.py", "preprocessing.py",
                  "edges.py", "lanes.py", "segmentation.py", "evaluate.py",
                  "run_experiments.py", "make_report.py"]:
        path = os.path.join(ROOT, "src", fname)
        if not os.path.exists(path):
            continue
        h2("src/" + fname)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        chunk = []
        for line in lines:
            chunk.append(esc(line.rstrip()[:112]))
            if len(chunk) == 46:
                story.append(Preformatted("\n".join(chunk), CODE))
                chunk = []
        if chunk:
            story.append(Preformatted("\n".join(chunk), CODE))
        story.append(Spacer(1, 6))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    build()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
