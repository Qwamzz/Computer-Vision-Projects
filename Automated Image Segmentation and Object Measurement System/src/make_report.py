"""
Report generator.

Builds report/CSCD608_Q4_Report.pdf from the tables and figures that
run_experiments.py produced. Run the experiments first, then this script.

    python run_experiments.py
    python make_report.py
"""

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               NextPageTemplate, PageBreak, PageTemplate,
                               Paragraph, Preformatted, Spacer, Table,
                               TableStyle)
from reportlab.lib.utils import ImageReader

from config import (BASE_IMAGES, FIG_DIR, REPORT_DIR, RESULTS_DIR, SRC_DIR,
                    TABLE_DIR, VARIANTS)

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

AUTHOR = "Nii Yartey Gidiglo"
STUDENT_ID = "22424650"
COURSE = "CSCD608 Advanced Computer Vision"
TITLE = "Automated Image Segmentation and Object Measurement System"

styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.6,
                      leading=13.4, alignment=TA_JUSTIFY, spaceAfter=6)
BODY_SMALL = ParagraphStyle("bodysmall", parent=BODY, fontSize=8.6, leading=11.6)
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, leading=18,
                    spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#12325c"))
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.8, leading=15,
                    spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1d4d80"))
H3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=10.4, leading=13,
                    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#28527a"))
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.2, leading=10.5,
                         alignment=TA_CENTER, textColor=colors.HexColor("#44484f"),
                         spaceBefore=2, spaceAfter=10)
CODE = ParagraphStyle("code", parent=styles["Code"], fontSize=5.6, leading=6.6)
TITLE_STYLE = ParagraphStyle("title", parent=styles["Title"], fontSize=21,
                             leading=25, textColor=colors.HexColor("#12325c"))
SUB = ParagraphStyle("sub", parent=styles["Normal"], fontSize=11.5, leading=16,
                     alignment=TA_CENTER)

story = []


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def p(text, style=BODY):
    story.append(Paragraph(text, style))


def h1(text):
    story.append(Paragraph(text, H1))


def h2(text):
    story.append(Paragraph(text, H2))


def h3(text):
    story.append(Paragraph(text, H3))


def spacer(h=6):
    story.append(Spacer(1, h))


def bullets(items, style=BODY):
    for item in items:
        story.append(Paragraph("&bull;&nbsp;&nbsp;" + item, ParagraphStyle(
            "b", parent=style, leftIndent=12, spaceAfter=3)))
    spacer(4)


def figure(name, caption, width=CONTENT_W, max_height=13.5 * cm):
    path = FIG_DIR / name
    if not path.exists():
        return
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    w = width
    h = w * ih / float(iw)
    if h > max_height:
        h = max_height
        w = h * iw / float(ih)
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    story.append(KeepTogether([img, Paragraph(caption, CAPTION)]))


def read_table(name):
    path = TABLE_DIR / name
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def make_table(header, rows, col_widths=None, font_size=7.6, align_right=None,
               highlight_rows=()):
    data = [[Paragraph("<b>%s</b>" % c, ParagraphStyle(
        "th", parent=BODY, fontSize=font_size, leading=font_size + 2,
        textColor=colors.white)) for c in header]]
    for r in rows:
        data.append([Paragraph(str(c), ParagraphStyle(
            "td", parent=BODY, fontSize=font_size, leading=font_size + 2.2))
            for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
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
    story.append(t)
    spacer(8)


def table_caption(text):
    story.append(Paragraph(text, CAPTION))


# ----------------------------------------------------------------------------
# Load results
# ----------------------------------------------------------------------------
with open(RESULTS_DIR / "summary.json", encoding="utf-8") as fh:
    SUMMARY = json.load(fh)

MAIN = read_table("main_comparison.csv")
MAIN_MEAN = read_table("main_comparison_mean.csv")
THRESH = read_table("threshold_study.csv")
RG = read_table("region_growing_study.csv")
KM = read_table("kmeans_study.csv")
FILT = read_table("filter_study.csv")
COND = read_table("conditions.csv")
SPLIT = read_table("split_study.csv")
COUNTS = read_table("measurement_counts.csv")
ACC = read_table("measurement_accuracy.csv")

METHODS = [m["method"] for m in MAIN_MEAN]
BEST = max(MAIN_MEAN, key=lambda r: float(r["mean_iou"]))
FASTEST = min(MAIN_MEAN, key=lambda r: float(r["mean_time_s"]))


def nice(image_id):
    return image_id.replace("_", " ")


# ============================================================================
# Title page
# ============================================================================
story.append(Spacer(1, 2.2 * cm))
p("UNIVERSITY OF GHANA", ParagraphStyle("u", parent=SUB, fontSize=13,
                                        textColor=colors.HexColor("#12325c")))
p("MPhil / MSc Computer Science, Second Semester Examinations 2025 / 2026", SUB)
p(COURSE + " (3 Credits)", SUB)
spacer(26)
p(TITLE, TITLE_STYLE)
p("Final Examination, Question 4", ParagraphStyle("q", parent=SUB, fontSize=12.5))
spacer(30)
p("Segmentation of objects of interest from an image, followed by the "
  "computation of meaningful measurements about the detected objects, using "
  "thresholding, region growing and K means clustering.",
  ParagraphStyle("abs", parent=BODY, alignment=TA_CENTER, fontSize=10.5,
                 leading=15, leftIndent=1.6 * cm, rightIndent=1.6 * cm))
spacer(34)
make_table(["", ""], [
    ["Name", AUTHOR],
    ["Student identification number", STUDENT_ID],
    ["Course", COURSE],
    ["Question attempted", "Question 4, Automated Image Segmentation and Object Measurement System"],
    ["Application domain", "Counting and measuring seeds, beans and coins"],
    ["Implementation", "Python 3 and OpenCV, no learned or pre-trained model"],
    ["Dataset", "%d real photographs and %d controlled degradations of them"
     % (len(BASE_IMAGES), len(VARIANTS))],
], col_widths=[6.0 * cm, CONTENT_W - 6.0 * cm], font_size=9)
story.append(PageBreak())

# ============================================================================
# Contents
# ============================================================================
h1("Contents")
make_table(["Section", "Title", "Page"], [
    ["1", "Problem definition and objectives", "3"],
    ["2", "Dataset", "5"],
    ["3", "Methodology", "7"],
    ["4", "Implementation", "11"],
    ["5", "Experimental results", "12"],
    ["6", "Critical analysis", "24"],
    ["7", "Limitations and failure cases", "27"],
    ["8", "Conclusion", "29"],
    ["9", "References", "29"],
    ["A", "Complete source code", "31"],
], col_widths=[2.0 * cm, CONTENT_W - 4.4 * cm, 2.4 * cm], font_size=9)
p("Page numbers are approximate because the figures are placed automatically.",
  BODY_SMALL)
story.append(PageBreak())

# ============================================================================
# 1. Problem definition
# ============================================================================
h1("1. Problem definition and objectives")
h2("1.1 The problem")
p("Many practical inspection tasks reduce to the same question: how many "
  "objects are present in this picture, and how large is each one. A seed "
  "laboratory grading a sample of cowpea needs the length and the width of "
  "every seed. A quality inspector counting roasted coffee beans needs the "
  "count and the size distribution. A numismatic archive photographing a tray "
  "of coins needs one region per coin so that each coin can be catalogued. In "
  "all of these cases the answer is produced by two steps, a segmentation step "
  "that decides which pixels belong to an object, and a measurement step that "
  "turns each region into numbers.")
p("This project builds that system with classical computer vision only. No "
  "learned detector and no pre-trained recognition model is used anywhere. "
  "Every decision is made by an algorithm whose behaviour can be written down "
  "in closed form, which is what makes the failure analysis in section 7 "
  "possible.")

h2("1.2 Objectives")
bullets([
    "Assemble a dataset of real photographs of countable objects under varying "
    "imaging conditions, and produce manually verified ground truth masks for "
    "a representative subset of it.",
    "Implement threshold based segmentation, region growing and K means "
    "clustering from first principles rather than treating them as black boxes.",
    "Investigate the effect of the parameters that each method exposes: the "
    "threshold strategy, the seed point selection and the similarity criterion "
    "of the region growing, and the number of clusters and the feature space "
    "of the clustering.",
    "Identify object boundaries and extract each object as a separate region, "
    "including a step that separates objects that touch each other.",
    "Measure the count, area, perimeter, centroid, bounding box and size of "
    "every object, and convert those numbers into millimetres where a physical "
    "scale is available.",
    "Evaluate the three approaches quantitatively with Intersection over "
    "Union, precision, recall and F1 score against the ground truth, and "
    "measure the processing time, the sensitivity to noise and the sensitivity "
    "to illumination.",
    "Discuss where each method fails and why, and show which earlier "
    "techniques of the course repair those failures.",
])

h2("1.3 The pipeline that was built")
p("The finished system implements the workflow required by the question:")
p("<b>Input image &gt; image preparation &gt; segmentation &gt; boundary "
  "identification &gt; object extraction &gt; object measurement &gt; "
  "performance evaluation</b>",
  ParagraphStyle("flow", parent=BODY, alignment=TA_CENTER, fontSize=10,
                 spaceBefore=6, spaceAfter=8))
figure("pipeline_stages.png",
       "Figure 1. The eight stages of the pipeline on one photograph of the "
       "dataset. Stage 4 removes the lighting gradient, stage 5 applies the "
       "Otsu threshold, stage 6 cleans the mask morphologically, stage 7 "
       "extracts the boundaries and stage 8 reports the measurements.")

story.append(PageBreak())

# ============================================================================
# 2. Dataset
# ============================================================================
h1("2. Dataset")
h2("2.1 Source of the images")
p("The dataset contains %d base photographs. Every one of them is a real "
  "photograph taken by a person with a camera. None of them was produced by "
  "an image generation model. Seven of them come from Wikimedia Commons under "
  "a Creative Commons licence, and one is the coins photograph that is "
  "distributed as sample data with scikit-image and originates from a museum "
  "collection. The full provenance is recorded in src/config.py and repeated "
  "in Table 1, so that any reader can retrieve the exact files."
  % len(BASE_IMAGES))

rows = []
for rec in BASE_IMAGES:
    rows.append([rec["id"], rec["title"], rec["domain"],
                 rec.get("expected_count") or "not counted",
                 rec.get("licence", "")])
make_table(["identifier", "content", "application domain", "objects", "licence"],
           rows, col_widths=[3.5 * cm, 3.9 * cm, 3.6 * cm, 1.7 * cm, 2.3 * cm],
           font_size=7.2)
table_caption("Table 1. The base photographs of the dataset. The object count "
              "is given for the images that carry an instance level ground "
              "truth.")

figure("dataset_overview.png",
       "Figure 2. The eight base photographs. They cover four of the "
       "application domains suggested by the question and they differ in "
       "contrast, in background, in object density and in the amount of "
       "shadow present.")

h2("2.2 Imaging conditions")
p("The question asks for the system to be tested under illumination changes, "
  "background variation, noise, partially touching objects and objects of "
  "different sizes. Part of this variation is already present in the "
  "photographs themselves: the coins sit on a dark background with a strong "
  "brightness falloff, the coffee beans cast soft shadows and carry specular "
  "highlights, the pale seeds have almost no contrast against their "
  "background, the third sunflower photograph is partly out of focus, and the "
  "pinto bean photograph is a dense pile in which nearly every object touches "
  "a neighbour.")
p("To obtain a controlled comparison rather than an anecdotal one, four "
  "further conditions are generated from two of the photographs by applying a "
  "known degradation to the real image: a diagonal lighting ramp that scales "
  "brightness from 0.30 to 1.20, an under exposure that multiplies the "
  "brightness by 0.45 and lifts the black point, additive zero mean Gaussian "
  "noise with a standard deviation of 18 grey levels, and a defocus blur "
  "modelled by a Gaussian point spread function with a fifteen pixel kernel. "
  "The geometry of the objects is untouched by all four operations, so the "
  "ground truth mask of the original photograph remains valid and the change "
  "in the score can be attributed to the condition alone.")
figure("conditions_strip.png",
       "Figure 3. The original photograph and the four controlled conditions "
       "derived from it.", max_height=6 * cm)

h2("2.3 Ground truth masks")
p("Ground truth masks were produced for five of the photographs. The "
  "procedure follows the way modern annotation tools work rather than painting "
  "every pixel by hand. A per image draft was produced first, using "
  "information that the system under test is not allowed to use, for example a "
  "threshold offset hand picked for that one photograph or an edge based "
  "recipe chosen for the coins. Every draft was then inspected as an overlay "
  "on the photograph and corrected by hand: false regions were deleted, "
  "touching objects were separated by clicking one point inside each object "
  "and letting a marker driven watershed cut along the crevice, and clipped "
  "parts of an outline were painted back in. For the low contrast photograph "
  "the draft failed completely, so the three seeds were outlined by hand from "
  "a coordinate grid and the traced polygons were then tightened onto the "
  "object boundary by a colour model refinement. Every mask was inspected a "
  "second time before it was accepted. The complete edit list is in "
  "src/ground_truth.py, so the annotation is fully reproducible.")
p("Four of the five masks carry one region per physical object, so object "
  "level counting can be scored against them. The mask of the first sunflower "
  "photograph is a foreground mask only, because several of its seeds touch "
  "along a long contact line where no crevice is visible, and an honest "
  "instance level annotation of that photograph is not possible. It is "
  "therefore used for pixel level scoring only. The dense pinto bean "
  "photograph carries no ground truth at all and is reported as a qualitative "
  "failure case.")
figure("ground_truth_overview.png",
       "Figure 4. The manually verified ground truth outlines. Each closed "
       "green contour is one annotated region.", max_height=6 * cm)

story.append(PageBreak())

# ============================================================================
# 3. Methodology
# ============================================================================
h1("3. Methodology")
h2("3.1 Image preparation")
p("Every image is first scaled so that its longest side is 900 pixels. Images "
  "are only ever shrunk, never enlarged, so no detail is invented, and the "
  "common working size makes the timing comparison fair and keeps the "
  "morphological kernel sizes meaningful across the dataset.")
h3("Colour representation")
p("The candidate representations that were examined are the grey level image, "
  "the saturation and value channels of HSV and the lightness, a and b "
  "channels of CIE Lab. For seven of the eight photographs the objects differ "
  "from the background mainly in brightness, so the grey level image carries "
  "the separation and is used by the threshold and the region growing stages. "
  "The exception is the cowpea photograph, where the printed ruler is as dark "
  "as the seeds in the grey image but neutral in colour, while the seeds are "
  "strongly red. The a channel of CIE Lab separates the seeds from the ruler "
  "and is therefore the correct representation for that image. The clustering "
  "stage is able to use the full Lab colour vector, which is one of the "
  "reasons it outperforms the other two methods in section 5.")
h3("Noise suppression")
p("Three filters are provided. The Gaussian filter is a linear low pass "
  "filter, it is the fastest and it blurs edges together with the noise. The "
  "median filter is a rank filter, it removes impulse noise while preserving "
  "step edges. The bilateral filter weights neighbours by both spatial and "
  "photometric distance, so it preserves edges best but costs the most time. "
  "Section 5.5 compares them on the noisy images.")
h3("Illumination correction")
p("An uneven illumination field is the single most damaging condition for a "
  "global threshold, because it shifts the object and the background "
  "distributions by different amounts in different parts of the image. The "
  "system removes it with a classical flat field correction. A morphological "
  "closing with a structuring element far larger than any object removes the "
  "objects and leaves an estimate of the background field, which is then "
  "divided out and rescaled to the full eight bit range. When the objects are "
  "brighter than the background an opening is used instead. The polarity is "
  "decided automatically by comparing the mean of a border ring of the image "
  "with the mean of its interior.")

h2("3.2 Method 1, threshold based segmentation")
p("Otsu's method is implemented from the definition rather than called as a "
  "library flag. For a candidate grey level t the pixels are split into a low "
  "class and a high class, and the between class variance is")
p("<font face='Courier'>sigma_b^2(t) = w0(t) &middot; w1(t) &middot; "
  "(mu0(t) - mu1(t))^2</font>",
  ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER, spaceBefore=4,
                 spaceAfter=4))
p("where w0 and w1 are the probabilities of the two classes and mu0 and mu1 "
  "their means. The implementation evaluates this expression for all 256 "
  "candidate thresholds using cumulative sums, so it runs in linear time in "
  "the number of grey levels, and it returns the whole criterion curve so that "
  "the choice can be plotted. Two further strategies are provided for "
  "comparison, a fixed threshold supplied by the user, and a locally adaptive "
  "Gaussian threshold which computes a separate threshold for every "
  "neighbourhood and therefore tolerates a lighting gradient without an "
  "explicit correction step.")
figure("otsu_histogram.png",
       "Figure 5. The grey level histogram after illumination correction, with "
       "the threshold chosen by the from scratch Otsu implementation. The "
       "histogram is clearly bimodal, which is exactly the situation in which "
       "Otsu is the right tool.", max_height=6.4 * cm)

h2("3.3 Method 2, region growing")
p("Region growing starts from a set of seed pixels and repeatedly adds "
  "neighbouring pixels that satisfy a similarity predicate, so it is a region "
  "based method that uses connectivity explicitly, unlike thresholding which "
  "classifies every pixel independently. The implementation uses a first in "
  "first out queue, so the traversal is a breadth first flood of the "
  "similarity region, and every pixel is visited at most once, which makes the "
  "routine linear in the number of pixels.")
h3("Similarity predicate")
p("A candidate pixel is accepted when the absolute difference between its "
  "intensity and a reference intensity of the region is at most the tolerance. "
  "Two references are implemented. The running mean reference uses the mean "
  "intensity of the pixels accepted so far, which allows a region to follow a "
  "slow intensity drift across a large object. The seed value reference keeps "
  "the intensity of the seed fixed, which is stricter and prevents a region "
  "from drifting through a shading gradient into the background.")
h3("Seed selection")
p("Three strategies are implemented. The distance transform strategy "
  "thresholds the image coarsely, computes the distance transform of the "
  "resulting mask and places one seed at the deepest interior point of every "
  "connected component, so it is an informed strategy that places a seed near "
  "the centre of each object. The grid strategy places seeds on a regular "
  "lattice and keeps those that fall on the object side of the coarse "
  "threshold. The random strategy samples seed positions uniformly from the "
  "object side. Section 5.3 shows how much the result depends on this choice.")

h2("3.4 Method 3, K means clustering")
p("The clustering is implemented as Lloyd's algorithm with a k means plus plus "
  "initialisation. The initialisation samples the first centre uniformly and "
  "then samples each further centre with a probability proportional to the "
  "squared distance to the nearest centre already chosen, which spreads the "
  "initial centres apart and avoids the degenerate solutions that a purely "
  "random initialisation produces. The main loop alternates an assignment step "
  "that attaches every pixel to its nearest centre with an update step that "
  "moves every centre to the mean of its members, and it stops when no centre "
  "moves further than a small tolerance.")
h3("Feature space")
p("Three feature spaces are provided: the intensity of the prepared grey "
  "image, the three dimensional Lab colour vector, and a five dimensional "
  "vector that appends the normalised pixel coordinates to the colour vector "
  "so that spatially compact clusters are preferred.")
h3("Deciding which clusters are objects")
p("Clustering alone does not say which cluster is the object. The system "
  "decides this from the border of the image: a cluster that occupies more "
  "than a quarter of the border ring is declared background, and everything "
  "else is foreground. If that rule leaves no cluster or all clusters as "
  "background, the system falls back to the ordering of the cluster means and "
  "picks the darkest or the brightest cluster according to the polarity of the "
  "scene. This rule is what allows K means with more than two clusters to be "
  "used as a segmentation rather than only as a quantisation.")

h2("3.5 Post processing, boundaries and object extraction")
p("Every raw mask passes through the same clean up: a morphological opening "
  "removes isolated noise pixels, a closing repairs small gaps, a hole filling "
  "step closes the specular highlights that appear inside glossy objects, and "
  "an area filter removes fragments that are far too small to be an object. "
  "The hole filling is implemented as a flood fill from a padded background "
  "border, so that whatever background is connected to the border stays "
  "background and every enclosed pocket becomes object.")
p("Boundaries are obtained with the morphological gradient of the mask, that "
  "is the dilation minus the erosion, which gives a one pixel outline of every "
  "region. Individual objects are extracted with eight connected component "
  "labelling.")
h3("Separating objects that touch")
p("Connected component labelling reports one region for a group of touching "
  "objects. The system therefore offers a separation step built on the "
  "distance transform and the watershed transform. The distance transform "
  "peaks at the centre of every object, thresholding it at a fraction of the "
  "local maximum produces one marker per object even when two objects share a "
  "boundary, and the watershed then grows those markers back to the object "
  "boundaries. The threshold is applied inside each connected component "
  "separately, because a single global threshold silently deletes every object "
  "that is much smaller than the largest object in the image.")

h2("3.6 Object measurement")
p("For every extracted region the system reports the measurements listed in "
  "the question and several standard shape descriptors:")
bullets([
    "<b>Object count</b>, the number of accepted connected components.",
    "<b>Area</b>, the number of pixels in the region.",
    "<b>Perimeter</b>, the arc length of the outer contour.",
    "<b>Centroid</b>, the first order moment of the region.",
    "<b>Bounding box</b>, the axis aligned rectangle around the region.",
    "<b>Object size</b>, reported as the equivalent circular diameter and as "
    "the major and minor axes of the ellipse fitted to the contour, which is "
    "the length and the width of an elongated object such as a seed.",
    "<b>Shape descriptors</b>: circularity 4 pi A / P squared, which is one for a "
    "perfect disc, the aspect ratio, the extent and the solidity.",
])
p("The pixel measurements are converted into millimetres for the cowpea "
  "photograph, which was taken next to a printed ruler. The calibration "
  "routine projects the ruler band onto the horizontal axis, detects the tick "
  "marks and takes the median distance between neighbouring ticks, which gives "
  "13.0 pixels per millimetre at the working resolution and a field of view of "
  "69.2 millimetres. The median is used so that the wider marks at the five "
  "millimetre positions cannot bias the estimate.")

h2("3.7 Evaluation protocol")
p("Predicted masks are compared with the ground truth masks at two levels. At "
  "the pixel level the system reports Intersection over Union, precision, "
  "recall, F1 score, the Dice coefficient and pixel accuracy. At the object "
  "level, predicted regions are matched greedily to ground truth regions in "
  "decreasing order of overlap, a match is accepted at an IoU of 0.5 or more, "
  "and object precision, object recall, object F1 and the count error follow "
  "from that matching. A third measure evaluates the measurements themselves: "
  "for every matched pair the relative error of the area and of the major axis "
  "length is computed, which answers the question that matters for a "
  "measurement system, namely how accurate the reported number is for an "
  "object that was found.")
p("All three methods are run with one common parameter set across the whole "
  "dataset, so that no method is given a per image advantage. The tolerance of "
  "the region growing and the configuration of the clustering are the values "
  "that performed best on average in the parameter studies of sections 5.3 and "
  "5.4, so each method is compared at its own best common setting.")

story.append(PageBreak())

# ============================================================================
# 4. Implementation
# ============================================================================
h1("4. Implementation")
p("The system is written in Python 3 and uses OpenCV, NumPy and Matplotlib. "
  "Otsu's threshold, the region growing and the K means clustering are "
  "implemented in this project rather than called as library primitives, so "
  "that the criterion, the queue and the iteration are all visible in the "
  "source. OpenCV is used for the standard supporting operations: colour "
  "conversion, filtering, morphology, connected component labelling, contour "
  "extraction, the distance transform and the watershed.")
make_table(["module", "responsibility"], [
    ["config.py", "Paths, working resolution and the dataset manifest with the "
     "provenance of every photograph."],
    ["dataset.py", "Retrieval of the base photographs and generation of the "
     "four controlled degradations."],
    ["preprocessing.py", "Colour representation, smoothing filters, contrast "
     "handling, flat field illumination correction and polarity detection."],
    ["segmentation.py", "Otsu from first principles, region growing with an "
     "explicit queue, K means with k means plus plus, morphological clean up "
     "and the watershed separation of touching objects."],
    ["measurement.py", "Boundary identification, connected component "
     "extraction and all object measurements."],
    ["evaluation.py", "Pixel level scores, object level matching and the "
     "measurement error analysis."],
    ["calibration.py", "Pixels per millimetre from the printed ruler."],
    ["ground_truth.py", "Assisted annotation, the manual edit list and the "
     "ground truth masks."],
    ["visualisation.py", "All figures written to results/figures."],
    ["run_experiments.py", "The experiment driver that reproduces every number "
     "and figure in this report."],
    ["make_report.py", "This report."],
], col_widths=[4.0 * cm, CONTENT_W - 4.0 * cm], font_size=8)
table_caption("Table 2. Module map of the submission.")

h3("How to reproduce")
story.append(Preformatted(
    "cd src\n"
    "python dataset.py            # fetch photographs and build the variants\n"
    "python ground_truth.py       # rebuild the annotation masks\n"
    "python run_experiments.py    # all experiments, tables and figures\n"
    "python make_report.py        # this PDF",
    ParagraphStyle("pre", parent=styles["Code"], fontSize=8, leading=10.5)))
spacer(6)
p("The complete run of the experiment driver takes about %s seconds on the "
  "machine used for this submission." % SUMMARY.get("runtime_s", "400"))

story.append(PageBreak())

# ============================================================================
# 5. Results
# ============================================================================
h1("5. Experimental results")
h2("5.1 Comparison of the three segmentation techniques")
rows = []
for r in MAIN:
    rows.append([nice(r["image"]), r["method"], r["iou"], r["precision"],
                 r["recall"], r["f1"], r["time_s"],
                 r["pred_count"], r["gt_count"] or "n/a"])
make_table(["image", "method", "IoU", "precision", "recall", "F1",
            "time (s)", "regions found", "objects in truth"], rows,
           col_widths=[3.5 * cm, 3.4 * cm, 1.4 * cm, 1.6 * cm, 1.4 * cm,
                       1.4 * cm, 1.5 * cm, 1.7 * cm, 1.6 * cm],
           font_size=6.8)
table_caption("Table 3. The three segmentation techniques on every annotated "
              "image, using one common parameter set.")

rows = [[r["method"], r["mean_iou"], r["mean_precision"], r["mean_recall"],
         r["mean_f1"], r["mean_time_s"]] for r in MAIN_MEAN]
best_index = [i for i, r in enumerate(MAIN_MEAN) if r is BEST]
make_table(["method", "mean IoU", "mean precision", "mean recall", "mean F1",
            "mean time (s)"], rows,
           col_widths=[5.2 * cm, 2.2 * cm, 2.6 * cm, 2.2 * cm, 2.0 * cm,
                       2.4 * cm], font_size=8, highlight_rows=best_index)
table_caption("Table 4. Mean scores over the five annotated photographs. The "
              "best method is highlighted.")

p("The summary table requested by the question is therefore the following, "
  "where the sensitivity entries are taken from the condition study in "
  "section 5.6 and the noise study in section 5.5.")
cond_by = {}
for r in COND:
    cond_by.setdefault((r["method"], r["condition"]), []).append(float(r["iou"]))


def cond_delta(method, condition):
    base = cond_by.get((method, "original photograph"), [])
    other = cond_by.get((method, condition), [])
    if not base or not other:
        return "n/a"
    drop = (sum(other) / len(other)) - (sum(base) / len(base))
    return "%+.3f" % drop


rows = []
for m in MAIN_MEAN:
    rows.append([
        m["method"],
        "%.3f" % float(m["mean_iou"]),
        "%.3f" % float(m["mean_time_s"]),
        cond_delta(m["method"], "Additive Gaussian sensor noise"),
        cond_delta(m["method"], "Strong lighting gradient"),
    ])
make_table(["method", "segmentation accuracy (mean IoU)",
            "processing time (s)", "sensitivity to noise (change in IoU)",
            "sensitivity to illumination (change in IoU)"], rows,
           col_widths=[4.2 * cm, 3.6 * cm, 2.7 * cm, 3.2 * cm, 3.0 * cm],
           font_size=7.4, highlight_rows=best_index)
table_caption("Table 5. The comparison table required by the question. A "
              "negative change means the score fell when the condition was "
              "applied.")

figure("iou_by_image.png",
       "Figure 6. Segmentation accuracy of the three methods on each annotated "
       "photograph.", max_height=7 * cm)
figure("time_by_method.png",
       "Figure 7. Mean processing time per image at the common working "
       "resolution.", max_height=6.4 * cm)
for rec in BASE_IMAGES:
    name = "main_%s.png" % rec["id"]
    if (FIG_DIR / name).exists():
        figure(name, "Figure. %s. Green marks pixels that were correctly "
                     "assigned to an object, red marks background accepted as "
                     "object and blue marks object pixels that were missed."
               % rec["title"], max_height=5.4 * cm)

h2("5.2 Threshold strategy and illumination")
rows = [[nice(r["image"]), r["setting"], r["iou"], r["precision"], r["recall"],
         r["f1"]] for r in THRESH]
make_table(["image", "threshold strategy", "IoU", "precision", "recall", "F1"],
           rows, col_widths=[4.2 * cm, 5.4 * cm, 1.7 * cm, 1.9 * cm, 1.7 * cm,
                             1.7 * cm], font_size=6.6)
table_caption("Table 6. Threshold study on the original photographs and on the "
              "same photographs with a strong lighting gradient.")
figure("threshold_study.png",
       "Figure 8. Effect of the threshold strategy. The fixed thresholds are "
       "only competitive when the value happens to suit the image, while Otsu "
       "combined with the flat field correction is stable across all four "
       "cases.", max_height=7.5 * cm)
figure("illumination_correction.png",
       "Figure 9. The lighting gradient case in detail. Without the flat field "
       "correction the single global threshold cannot satisfy both sides of "
       "the image at once.", max_height=5.2 * cm)

h2("5.3 Region growing, seeds and similarity")
rows = [[nice(r["image"]), r["study"], r["setting"], r["n_seeds"], r["iou"],
         r["recall"], r["pred_count"], r["time_s"]] for r in RG]
make_table(["image", "study", "setting", "seeds", "IoU", "recall",
            "regions", "time (s)"], rows,
           col_widths=[3.7 * cm, 2.5 * cm, 2.6 * cm, 1.5 * cm, 1.5 * cm,
                       1.5 * cm, 1.5 * cm, 1.7 * cm], font_size=6.6)
table_caption("Table 7. Region growing study.")
figure("region_growing_tolerance.png",
       "Figure 10. Effect of the similarity tolerance. Below the correct value "
       "the regions stop inside the objects, and above it the regions leak "
       "through the object boundary into the background.", max_height=7 * cm)
figure("region_growing_seeds.png",
       "Figure 11. Effect of the seed selection strategy. The informed "
       "distance transform strategy is the best choice on the coins, while on "
       "the seeds the extra seeds of the random strategy compensate for the "
       "wide intensity range inside each object.", max_height=7 * cm)
figure("region_growing_visual.png",
       "Figure 12. Region growing at three tolerances on the same photograph.",
       max_height=5.2 * cm)

h2("5.4 K means, number of clusters and features")
rows = [[nice(r["image"]), r["study"], r["setting"], r["iou"], r["precision"],
         r["recall"], r["pred_count"], r["time_s"]] for r in KM]
make_table(["image", "study", "setting", "IoU", "precision", "recall",
            "regions", "time (s)"], rows,
           col_widths=[3.7 * cm, 2.2 * cm, 2.4 * cm, 1.6 * cm, 1.8 * cm,
                       1.6 * cm, 1.5 * cm, 1.7 * cm], font_size=6.6)
table_caption("Table 8. K means study.")
figure("kmeans_k.png",
       "Figure 13. Effect of the number of clusters. On an image with two "
       "clearly separated populations, raising K splits the object itself and "
       "the score falls, while on the coins, where the background has its own "
       "internal structure, a larger K is what allows the dark corner of the "
       "background to be separated from the coins.", max_height=7 * cm)
figure("kmeans_clusters.png",
       "Figure 14. Cluster label maps for three values of K on the coffee bean "
       "photograph. At K equal to three the shadow becomes its own cluster.",
       max_height=5.2 * cm)

h2("5.5 Filtering study")
rows = [[nice(r["image"]), r["filter"], r["iou"], r["precision"], r["recall"],
         r["f1"], r["pred_count"], r["time_s"]] for r in FILT]
make_table(["image", "filter", "IoU", "precision", "recall", "F1", "regions",
            "time (s)"], rows,
           col_widths=[4.4 * cm, 2.4 * cm, 1.7 * cm, 1.9 * cm, 1.7 * cm,
                       1.7 * cm, 1.5 * cm, 1.7 * cm], font_size=7)
table_caption("Table 9. Effect of the smoothing filter on the two noisy "
              "images, with Otsu thresholding held fixed.")
figure("filter_study.png",
       "Figure 15. Without a smoothing filter the noise fragments the mask "
       "into thousands of small regions, and any of the three filters repairs "
       "most of the damage.", max_height=7 * cm)

h2("5.6 Robustness to imaging conditions")
rows = [[nice(r["image"]), r["condition"], r["method"], r["iou"], r["f1"],
         r["pred_count"]] for r in COND]
make_table(["image", "condition", "method", "IoU", "F1", "regions"], rows,
           col_widths=[4.0 * cm, 3.6 * cm, 3.6 * cm, 1.6 * cm, 1.6 * cm,
                       1.6 * cm], font_size=6.6)
table_caption("Table 10. Every method under every imaging condition.")
figure("conditions_seeds_sunflower_01.png",
       "Figure 16. Robustness on the sunflower seed photograph.",
       max_height=7 * cm)
figure("conditions_coins_greek_07.png",
       "Figure 17. Robustness on the coins photograph.", max_height=7 * cm)
p("Three observations follow from Table 10. First, the defocus blur is the "
  "mildest of the four conditions for every method, because blurring moves the "
  "boundary of an object by a few pixels but does not change which side of the "
  "threshold the interior of the object falls on. Second, the under exposure "
  "is almost harmless for the threshold and the clustering, because both are "
  "invariant to a monotone rescaling of the grey levels as long as the two "
  "populations stay separable, while it does hurt the region growing, whose "
  "tolerance is an absolute number of grey levels and is therefore far too "
  "wide once the contrast has been compressed. Third, the small positive "
  "changes in the region growing column of Table 5 are not evidence that noise "
  "helps. They arise because the region growing under grows on the clean "
  "photographs, so its baseline is already low, and a condition that changes "
  "the local statistics can accidentally let a region cover more of an object "
  "than it did before. A score that moves in the right direction for the wrong "
  "reason is still a weakness of the method, not a strength.")

h2("5.7 Separating touching objects")
rows = [[nice(r["image"]), r["expected"] or "not counted", r["marker_ratio"],
         r["count_without_split"], r["count_with_split"]] for r in SPLIT]
make_table(["image", "objects present", "marker distance ratio",
            "regions without separation", "regions after separation"], rows,
           col_widths=[4.4 * cm, 2.6 * cm, 3.4 * cm, 3.2 * cm, 3.0 * cm],
           font_size=7)
table_caption("Table 11. Effect of the marker threshold used to separate "
              "touching objects.")
figure("split_study.png",
       "Figure 18. Number of regions reported as a function of the marker "
       "threshold.", max_height=7 * cm)

h2("5.8 Object measurement")
rows = [[nice(r["image"]), r["expected"] or "not counted",
         r["count_before_split"], r["count_after_split"]] for r in COUNTS]
make_table(["image", "objects present", "count before separation",
            "count after separation"], rows,
           col_widths=[5.4 * cm, 3.4 * cm, 3.9 * cm, 3.9 * cm], font_size=8)
table_caption("Table 12. Object counts produced by the full pipeline.")

rows = [[nice(r["image"]), r["method"], r["matched_objects"],
         r["mean_abs_area_error_pct"], r["max_abs_area_error_pct"],
         r["mean_abs_length_error_pct"], r["max_abs_length_error_pct"]]
        for r in ACC]
make_table(["image", "method", "matched objects", "mean area error (%)",
            "max area error (%)", "mean length error (%)",
            "max length error (%)"], rows,
           col_widths=[3.6 * cm, 3.3 * cm, 2.0 * cm, 2.2 * cm, 2.0 * cm,
                       2.2 * cm, 2.0 * cm], font_size=6.8)
table_caption("Table 13. Accuracy of the measurements of the objects that "
              "were correctly found, relative to the ground truth regions.")

figure("measured_full_beans_cowpea_scale_05.png",
       "Figure 19. The measurement output on the calibrated photograph. Every "
       "seed is labelled with its length and width in millimetres, computed "
       "from the ellipse fitted to its contour and the calibration of 13.0 "
       "pixels per millimetre. The measured lengths of 10.5 to 13.0 "
       "millimetres and widths of 4.4 to 5.9 millimetres are consistent with "
       "the published dimensions of cowpea seed, which is an independent check "
       "on the calibration. Region 9 is the letter m of the printed unit "
       "label, a false detection that is discussed in section 7.",
       max_height=10 * cm)
figure("cowpea_length_histogram.png",
       "Figure 20. Length distribution of the eight measured seeds.",
       max_height=6.4 * cm)
for image_id in ("beans_coffee_06", "coins_greek_07"):
    figure("measure_%s.png" % image_id,
           "Figure. Segmentation, separation of touching objects and "
           "measurement on %s." % nice(image_id), max_height=5.2 * cm)
figure("seed_area_histogram.png",
       "Figure 21. Area distribution of the regions found in the sunflower "
       "photograph. The long tail on the right is produced by the clusters of "
       "touching seeds that were not separated.", max_height=6.4 * cm)

story.append(PageBreak())

# ============================================================================
# 6. Critical analysis
# ============================================================================
h1("6. Critical analysis")
h2("6.1 Which segmentation method produced the best results and why")
p("K means clustering in the Lab colour space with three clusters produced the "
  "highest mean Intersection over Union, %s, ahead of Otsu thresholding at %s "
  "and region growing at %s. Two properties explain the difference. First, the "
  "clustering is the only method of the three that can use colour: on the "
  "coins photograph it separates the dark background from the coins where a "
  "grey level threshold cannot, and on the low contrast seed photograph it "
  "uses the slight colour difference between the cream seeds and the bluish "
  "white paper, which raises the IoU from %s to %s. Second, a third cluster "
  "gives the method somewhere to put the shadow, so shadow pixels are not "
  "forced into either the object class or the background class."
  % (BEST["mean_iou"],
     [m for m in MAIN_MEAN if m["method"].startswith("Threshold")][0]["mean_iou"],
     [m for m in MAIN_MEAN if m["method"].startswith("Region")][0]["mean_iou"],
     [r for r in MAIN if r["image"] == "seeds_pale_04"
      and r["method"].startswith("Threshold")][0]["iou"],
     [r for r in MAIN if r["image"] == "seeds_pale_04"
      and r["method"].startswith("K means")][0]["iou"]))
p("The advantage comes at a price. The clustering is roughly two orders of "
  "magnitude slower than the threshold, because it iterates over every pixel "
  "many times, while Otsu touches the image once to build a histogram. It is "
  "also worth stating clearly that K means with two clusters on the intensity "
  "channel is not a genuinely different method: it minimises the within class "
  "variance of a two class partition of the grey levels, which is the same "
  "objective that Otsu maximises in its complementary form. In the study of "
  "section 5.4, K means with two clusters on the intensity channel reproduced "
  "the Otsu result exactly on two of the three photographs and differed by "
  "less than one hundredth of an IoU point on the third, which is a useful "
  "confirmation that both implementations are correct.")

h2("6.2 How illumination changes affected thresholding")
th_rows = [r for r in THRESH if r["image"] == "seeds_sunflower_01_illum"]
otsu_off = [r for r in th_rows if r["setting"].startswith("Otsu, no")]
otsu_on = [r for r in th_rows if r["setting"].startswith("Otsu, illumination")]
p("A global threshold assumes that one grey level separates object from "
  "background everywhere in the image. A lighting gradient breaks that "
  "assumption directly. On the photograph with the diagonal ramp, Otsu without "
  "the flat field correction scored an IoU of %s, and the same threshold after "
  "the correction scored %s. The error is systematic rather than random: in "
  "the brightly lit part of the image the background is pushed above the "
  "threshold, so the mask is correct there, while in the dim part the "
  "background falls below the threshold and is accepted as object, which is "
  "why the precision collapses while the recall stays high. The locally "
  "adaptive threshold reaches a similar accuracy to the corrected global "
  "threshold on the gradient images, because it recomputes the decision in "
  "every neighbourhood, but it is noticeably worse on the clean images because "
  "it also responds to the texture inside the objects."
  % (otsu_off[0]["iou"] if otsu_off else "n/a",
     otsu_on[0]["iou"] if otsu_on else "n/a"))
coins_off = [r for r in THRESH if r["image"] == "coins_greek_07"
             and r["setting"].startswith("Otsu, no")]
coins_on = [r for r in THRESH if r["image"] == "coins_greek_07"
            and r["setting"].startswith("Otsu, illumination")]
p("The correction is not free, and the coins photograph shows the cost "
  "clearly. There the flat field step lowered the score from %s to %s. The "
  "reason is that this image has a dark background, so the estimated "
  "background field is small, and dividing by a small number amplifies both "
  "the noise and the residual texture of the background until part of it "
  "crosses the threshold. The lesson is that a flat field correction should be "
  "applied when the illumination varies across the field of view, and not as a "
  "reflex on every image: on a dark background it is safer to work on the "
  "boundaries of the objects, as the edge based recipe in section 6.7 does."
  % (coins_off[0]["iou"] if coins_off else "n/a",
     coins_on[0]["iou"] if coins_on else "n/a"))

h2("6.3 How seed point selection influenced region growing")
p("Seed selection changed the result more than any other single parameter. On "
  "the coins photograph the informed distance transform strategy placed one "
  "seed near the centre of every detected coin and reached an IoU of %s, while "
  "the regular grid placed only five usable seeds and reached %s, a loss of "
  "more than a factor of five. The reason is structural: region growing can "
  "never recover an object that contains no seed, so a missing seed is a "
  "guaranteed false negative rather than a small error. On the sunflower "
  "photograph the ordering reverses, because the coarse mask merges touching "
  "seeds into seven components and therefore yields only seven seeds, while "
  "the random strategy scatters forty seeds and covers far more of the seed "
  "area. The practical conclusion is that region growing is only as good as "
  "the procedure that produces its seeds, and that procedure is itself usually "
  "a segmentation, which makes the method circular unless the seeds come from "
  "a human or from a physically motivated prior."
  % ([r for r in RG if r["image"] == "coins_greek_07"
      and r["study"] == "seed_strategy" and r["setting"] == "distance"][0]["iou"],
     [r for r in RG if r["image"] == "coins_greek_07"
      and r["study"] == "seed_strategy" and r["setting"] == "grid"][0]["iou"]))
p("The similarity criterion mattered less. The running mean reference was "
  "slightly better than the fixed seed value reference on both images, because "
  "the objects in this dataset have a wide internal intensity range and a "
  "region that can follow a drift covers more of the object before it stops.")

h2("6.4 What effect changing K had on K means segmentation")
p("The sweep over K in Table 8 is run on the intensity feature space, so that "
  "the effect of the number of clusters is not mixed with the effect of the "
  "features. The effect of K depends entirely on how many populations the "
  "image really contains. On the sunflower photograph, which has one object population and "
  "one background population, the best value was K equal to two and every "
  "further cluster split the object itself, so the score fell monotonically "
  "from %s at K equal to two to %s at K equal to six. On the coins "
  "photograph the opposite happened: the background is not uniform, it has a "
  "bright region and a dark corner, so K equal to two forced the dark corner "
  "and the dim coins into the same cluster and the score was only %s, while K "
  "equal to four separated them and reached %s. In short, K must match the "
  "number of distinguishable populations in the image, not the number of "
  "object classes the user is interested in."
  % ([r for r in KM if r["image"] == "seeds_sunflower_01" and r["study"] == "K"
      and r["setting"] == "2"][0]["iou"],
     [r for r in KM if r["image"] == "seeds_sunflower_01" and r["study"] == "K"
      and r["setting"] == "6"][0]["iou"],
     [r for r in KM if r["image"] == "coins_greek_07" and r["study"] == "K"
      and r["setting"] == "2"][0]["iou"],
     [r for r in KM if r["image"] == "coins_greek_07" and r["study"] == "K"
      and r["setting"] == "4"][0]["iou"]))

h2("6.5 Which method was most computationally efficient")
p("Thresholding, by a wide margin. Its mean processing time was %s seconds per "
  "image against %s seconds for the region growing and %s seconds for the "
  "clustering at the common working resolution of 900 pixels. The ordering "
  "follows directly from the algorithms. Otsu makes one pass over the image to "
  "build a histogram and then searches 256 candidate thresholds, so its cost "
  "is one pass over the pixels. Lloyd's algorithm makes one distance "
  "computation per pixel per cluster per iteration, so its cost is "
  "approximately K times the number of iterations passes over the pixels, and "
  "the five dimensional feature space multiplies that again. The region "
  "growing visits every pixel once, but it does so in an interpreted loop with "
  "a queue and a per pixel predicate, so its constant factor is far larger "
  "than the vectorised operations used by the other two. A compiled "
  "implementation of the same algorithm would close most of that gap, so the "
  "timing of the region growing should be read as a property of this "
  "implementation and not of the algorithm."
  % (FASTEST["mean_time_s"],
     [m for m in MAIN_MEAN if m["method"].startswith("Region")][0]["mean_time_s"],
     [m for m in MAIN_MEAN if m["method"].startswith("K means")][0]["mean_time_s"]))

h2("6.6 Under what conditions did each method fail")
make_table(["condition", "thresholding", "region growing", "K means"], [
    ["Lighting gradient",
     "Fails without a flat field correction, the background of the dim side is "
     "accepted as object.",
     "Degrades gracefully, because the predicate is local, but regions leak "
     "across the gradient when the tolerance is high.",
     "Degrades, because a brightness gradient creates spurious clusters that "
     "cut across the object and background populations."],
    ["Low contrast",
     "Fails completely, the histogram has a single mode so the threshold is "
     "meaningless.",
     "Leaks into the background, since object and background differ by less "
     "than the tolerance.",
     "Best of the three, because the residual colour difference survives even "
     "when the brightness difference does not."],
    ["Noise",
     "Fragments the mask into many small regions unless a smoothing filter is "
     "applied first.",
     "Stops early, because a noisy neighbour fails the predicate.",
     "Fairly robust, because averaging over a cluster suppresses zero mean "
     "noise."],
    ["Touching objects",
     "Reports one region for the whole group.",
     "Same, unless a separate seed is placed in each object and the tolerance "
     "is tight.",
     "Same, since clustering has no notion of connectivity."],
    ["Objects of different sizes",
     "Handled, provided the area filter is not set too high.",
     "Handled, provided each object receives a seed.",
     "Handled, the cluster assignment does not depend on region size."],
    ["Shadows",
     "Shadow pixels are darker than the background and are accepted as object.",
     "Regions grow into the penumbra when the tolerance exceeds the shadow "
     "step.",
     "A third cluster absorbs the shadow, which is why K equal to three helps "
     "on the coffee photograph."],
], col_widths=[3.0 * cm, 4.6 * cm, 4.6 * cm, 4.6 * cm], font_size=6.8)
table_caption("Table 14. Failure conditions of the three techniques.")

h2("6.7 How techniques studied in earlier weeks improve the results")
bullets([
    "<b>Morphological filtering.</b> Opening, closing and hole filling turn a "
    "ragged raw threshold into a usable mask. On the coffee beans the "
    "specular highlights inside the objects appear as holes in the raw mask, "
    "and the hole filling step alone repairs them.",
    "<b>Flat field correction by large scale morphology.</b> Estimating the "
    "background with a structuring element larger than any object and dividing "
    "it out restores a bimodal histogram, and it is what makes the global "
    "threshold usable on the gradient images.",
    "<b>Edge detection.</b> The boundaries of the coins are far more reliable "
    "than their absolute grey levels, and an edge based recipe followed by a "
    "morphological closing and a hole fill segments that photograph much "
    "better than a global threshold. The same recipe was used to draft the "
    "ground truth for that image.",
    "<b>The distance transform and the watershed.</b> These separate touching "
    "objects that connected component labelling merges, which is what converts "
    "a foreground mask into an object count.",
    "<b>Colour space transformation.</b> Working in CIE Lab rather than in "
    "grey levels is what allows the ruler to be rejected in the cowpea "
    "photograph and the pale seeds to be found at all.",
    "<b>Shape descriptors.</b> Circularity, solidity and the aspect ratio "
    "provide a cheap post filter: on the cowpea photograph the false detection "
    "produced by the printed unit label has a much lower area and a different "
    "aspect ratio from any seed, so a simple shape rule removes it.",
])

story.append(PageBreak())

# ============================================================================
# 7. Limitations
# ============================================================================
h1("7. Limitations and failure cases")
h2("7.1 Dense piles of objects")
p("The clearest limitation of the whole approach is that it assumes one "
  "connected region corresponds to one object. The pinto bean photograph "
  "breaks that assumption: about two hundred beans are piled so that almost "
  "every bean touches several neighbours, and the segmentation returns a "
  "single region covering the entire pile. The distance transform and "
  "watershed step improves this but does not solve it, because the ridges "
  "between overlapping beans are not deep enough for the distance transform to "
  "produce one marker per bean. Solving this case properly requires either a "
  "different acquisition protocol, spreading the beans out so that they do not "
  "touch, or a method that models object appearance rather than object "
  "connectivity.")
figure("failure_dense.png",
       "Figure 22. The dense pile failure case. The pile is segmented "
       "perfectly as foreground, and the object count is still wrong by two "
       "orders of magnitude.", max_height=5.6 * cm)

h2("7.2 Low contrast objects")
p("When the object and the background have nearly the same brightness the grey "
  "level histogram has a single mode and the Otsu criterion has no meaningful "
  "maximum to find. It then selects a threshold inside the object population "
  "and cuts the objects into pieces. Only the colour based clustering recovers "
  "the objects on this photograph, and even it merges the darker striped "
  "regions of the seed with its bright out of focus lobe imperfectly.")
figure("failure_low_contrast.png",
       "Figure 23. The low contrast failure case for each method.",
       max_height=5.4 * cm)

h2("7.3 Shadows and specular highlights")
p("A cast shadow is darker than the background, so any method that defines "
  "objects as dark pixels will accept part of the shadow as object. This is "
  "the main reason the measured areas on the coffee and cowpea photographs are "
  "larger than the ground truth areas, by about %s per cent on average in "
  "Table 13, while the measured lengths are accurate to about %s per cent. A "
  "shadow adds a thin halo around the whole object, which changes the area "
  "much more than it changes the fitted major axis."
  % (ACC[0]["mean_abs_area_error_pct"], ACC[0]["mean_abs_length_error_pct"]))
figure("failure_shadow.png",
       "Figure 24. Shadow pixels accepted as object, shown in red.",
       max_height=6.4 * cm)

h2("7.4 Other limitations")
bullets([
    "The system has no notion of what an object is. Any dark connected region "
    "large enough to survive the area filter is reported, which is why the "
    "letter m of the printed unit label in the cowpea photograph is counted as "
    "a ninth object.",
    "Objects cut by the image border are measured as if they were whole. The "
    "measurement module flags them with a touches_border column so that they "
    "can be excluded, but the count still includes them.",
    "Measurements are reported in millimetres only when the photograph "
    "contains a scale. Without a scale, only relative sizes are meaningful.",
    "The ground truth is not perfect. It was produced with an assisted "
    "procedure and inspected twice, and the remaining boundary error of a few "
    "pixels per object places an upper bound of roughly one to two per cent on "
    "the IoU figures that can be trusted.",
    "The region growing implementation is interpreted Python, so its timing is "
    "not comparable with the vectorised routines used by the other two "
    "methods.",
    "All experiments are run at a common working resolution of 900 pixels. "
    "Absolute pixel measurements would change at another resolution, although "
    "the relative comparisons would not.",
])

story.append(PageBreak())

# ============================================================================
# 8. Conclusion
# ============================================================================
h1("8. Conclusion")
p("A complete classical pipeline was built and evaluated: image preparation, "
  "three segmentation techniques implemented from first principles, "
  "morphological clean up, boundary identification, separation of touching "
  "objects, object extraction, object measurement in pixels and in "
  "millimetres, and a quantitative evaluation against manually verified ground "
  "truth.")
p("On the five annotated photographs, K means clustering in the Lab colour "
  "space reached a mean Intersection over Union of %s, Otsu thresholding "
  "reached %s and region growing reached %s. Thresholding was about %d times "
  "faster than the clustering and is the right default whenever the histogram "
  "is bimodal and the illumination has been corrected. Clustering earns its "
  "extra cost only when colour carries information that brightness does not, "
  "which was the case on the coins and on the low contrast seeds. Region "
  "growing was the weakest of the three in an automatic setting, not because "
  "the algorithm is poor but because its result is decided by the seed set, "
  "and producing a good seed set automatically already requires a "
  "segmentation."
  % (BEST["mean_iou"],
     [m for m in MAIN_MEAN if m["method"].startswith("Threshold")][0]["mean_iou"],
     [m for m in MAIN_MEAN if m["method"].startswith("Region")][0]["mean_iou"],
     int(round(float(BEST["mean_time_s"]) / max(1e-6, float(FASTEST["mean_time_s"]))))))
p("For the objects that were correctly found, the measurements are usable: on "
  "the calibrated photograph all eight cowpea seeds were matched to the ground "
  "truth and their lengths were reported to within about one and a half per "
  "cent, which is adequate for seed grading, while their areas carried a "
  "larger error because the segmentation includes part of the cast shadow.")
p("The clearest lesson of the experiments is that the acquisition matters more "
  "than the choice among the three segmentation techniques. Diffuse and even "
  "lighting, a background whose colour contrasts with the objects, and objects "
  "that are spread out rather than piled together move every method into the "
  "range above 0.95 IoU. When those conditions are absent, no amount of "
  "parameter tuning on a classical method recovers the lost information.")

# ============================================================================
# 9. References
# ============================================================================
h1("9. References")
refs = [
    "Otsu, N. (1979). A threshold selection method from gray level histograms. "
    "IEEE Transactions on Systems, Man and Cybernetics, 9(1), 62 to 66.",
    "Adams, R. and Bischof, L. (1994). Seeded region growing. IEEE "
    "Transactions on Pattern Analysis and Machine Intelligence, 16(6), 641 to "
    "647.",
    "Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Transactions "
    "on Information Theory, 28(2), 129 to 137.",
    "Arthur, D. and Vassilvitskii, S. (2007). k-means++: the advantages of "
    "careful seeding. Proceedings of the ACM SIAM Symposium on Discrete "
    "Algorithms, 1027 to 1035.",
    "Beucher, S. and Meyer, F. (1993). The morphological approach to "
    "segmentation: the watershed transformation. Mathematical Morphology in "
    "Image Processing, 433 to 481.",
    "Canny, J. (1986). A computational approach to edge detection. IEEE "
    "Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679 to "
    "698.",
    "Gonzalez, R. C. and Woods, R. E. (2018). Digital Image Processing, fourth "
    "edition. Pearson.",
    "Bradski, G. (2000). The OpenCV library. Dr Dobb's Journal of Software "
    "Tools.",
    "Rother, C., Kolmogorov, V. and Blake, A. (2004). GrabCut, interactive "
    "foreground extraction using iterated graph cuts. ACM Transactions on "
    "Graphics, 23(3), 309 to 314. Used only as an annotation aid when the "
    "hand traced outlines were tightened.",
]
for i, r in enumerate(refs, start=1):
    p("[%d] %s" % (i, r), BODY_SMALL)

h2("Image sources")
rows = [[rec["id"], rec["source"], rec.get("licence", "")] for rec in BASE_IMAGES]
make_table(["identifier", "source", "licence"], rows,
           col_widths=[3.8 * cm, 9.4 * cm, 3.4 * cm], font_size=7)
table_caption("Table 15. Provenance of every photograph in the dataset. All of "
              "them are real photographs.")

# ============================================================================
# Appendix: source code
# ============================================================================
story.append(NextPageTemplate("code"))
story.append(PageBreak())
h1("Appendix A. Complete source code")
p("The listing below is the complete source of the submission, in the order in "
  "which the modules are described in Table 2. The same files are included in "
  "the src directory of the accompanying archive.", BODY_SMALL)

CODE_FILES = ["config.py", "dataset.py", "preprocessing.py", "segmentation.py",
              "measurement.py", "evaluation.py", "calibration.py",
              "ground_truth.py", "visualisation.py", "run_experiments.py",
              "make_report.py"]
for name in CODE_FILES:
    path = SRC_DIR / name
    if not path.exists():
        continue
    story.append(Paragraph("src/" + name, ParagraphStyle(
        "codehead", parent=H3, fontSize=9.4, spaceBefore=10)))
    text = path.read_text(encoding="utf-8")
    text = text.replace("\t", "    ")
    lines = text.splitlines()
    chunk = 62
    for i in range(0, len(lines), chunk):
        story.append(Preformatted("\n".join(lines[i:i + chunk]), CODE))


# ============================================================================
# Page furniture
# ============================================================================
def draw_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(colors.HexColor("#6a707a"))
    canvas.drawString(MARGIN, 1.15 * cm,
                      "%s  |  %s  |  Question 4" % (AUTHOR, STUDENT_ID))
    canvas.drawRightString(PAGE_W - MARGIN, 1.15 * cm, "Page %d" % doc.page)
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#c6ccd4"))
        canvas.line(MARGIN, PAGE_H - MARGIN + 0.35 * cm,
                    PAGE_W - MARGIN, PAGE_H - MARGIN + 0.35 * cm)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 0.55 * cm,
                          COURSE + "  |  " + TITLE)
    canvas.restoreState()


def build():
    out = REPORT_DIR / "CSCD608_Q4_Report.pdf"
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN,
                          rightMargin=MARGIN, topMargin=MARGIN,
                          bottomMargin=MARGIN + 0.4 * cm,
                          title=TITLE, author=AUTHOR)
    frame = Frame(MARGIN, MARGIN + 0.4 * cm, CONTENT_W,
                  PAGE_H - 2 * MARGIN - 0.4 * cm, id="normal")
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=draw_furniture),
        PageTemplate(id="code", frames=[frame], onPage=draw_furniture),
    ])
    doc.build(story)
    print("written:", out)
    return out


if __name__ == "__main__":
    build()
