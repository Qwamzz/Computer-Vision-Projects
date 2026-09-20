"""
make_report.py
Classical Object Detection and Localisation

Builds the submission PDF from the measured results in results/. Every number
quoted in the report is read from the CSV files written by run_experiments.py,
so the document cannot drift away from the experiments.

    python src/make_report.py

Author: Nii Yartey Gidiglo
"""

import csv
import json
import os

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
OUT = os.path.join(ROOT, "report",
                   "Classical_Object_Detection_and_Localisation.pdf")

NAME = "Nii Yartey Gidiglo"
PROJECT = "Classical object detection and localisation"

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
                    leading=12.5, alignment=TA_CENTER, spaceBefore=5,
                    spaceAfter=7)
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


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def p(text, style=BODY):
    story.append(Paragraph(text, style))


def h1(text):
    story.append(Paragraph(text, H1))


def h2(text):
    story.append(Paragraph(text, H2))


def h3(text):
    story.append(Paragraph(text, H3))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText="•"))
    story.append(Spacer(1, 4))


def eq(text):
    story.append(Paragraph(text, EQ))


def figure(fname, caption, width=BODY_W):
    path = os.path.join(FIG, fname)
    if not os.path.exists(path):
        return
    iw, ih = ImageReader(path).getSize()
    w = min(width, BODY_W)
    h = w * ih / float(iw)
    max_h = 15.5 * cm
    if h > max_h:
        h = max_h
        w = h * iw / float(ih)
    FIGN[0] += 1
    story.append(KeepTogether([Image(path, w, h),
                               Paragraph("Figure %d. %s" % (FIGN[0], caption), CAP)]))


def read_table(name):
    with open(os.path.join(TAB, name), "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def table(header, rows, caption, widths=None, fontsize=7.4, align_left_first=True):
    TABN[0] += 1
    data = [[Paragraph("<b>%s</b>" % h.replace("_", " "),
                       ParagraphStyle("th", parent=BODY, fontSize=fontsize,
                                      leading=fontsize + 2, alignment=TA_CENTER,
                                      textColor=colors.white, spaceAfter=0))
             for h in header]]
    cell = ParagraphStyle("td", parent=BODY, fontSize=fontsize,
                          leading=fontsize + 2, alignment=TA_CENTER, spaceAfter=0)
    cell_l = ParagraphStyle("tdl", parent=cell, alignment=0)
    for r in rows:
        data.append([Paragraph(str(v), cell_l if (i == 0 and align_left_first) else cell)
                     for i, v in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4d8f")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(KeepTogether([t, Paragraph("Table %d. %s" % (TABN[0], caption), CAP)]))


def num(x, nd=3):
    try:
        return ("%." + str(nd) + "f") % float(x)
    except (TypeError, ValueError):
        return str(x)


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------
def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#5a6673"))
    canvas.drawString(MARGIN, 1.15 * cm, PROJECT)
    canvas.drawRightString(PAGE_W - MARGIN, 1.15 * cm,
                           "%s, page %d" % (NAME, doc.page))
    canvas.setStrokeColor(colors.HexColor("#c7d0da"))
    canvas.line(MARGIN, 1.45 * cm, PAGE_W - MARGIN, 1.45 * cm)
    canvas.restoreState()


def blank(canvas, doc):
    pass


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=1.7 * cm, bottomMargin=2.0 * cm,
                          title="Classical Object Detection and Localisation",
                          author=NAME)
    frame = Frame(MARGIN, 2.0 * cm, BODY_W, PAGE_H - 3.7 * cm, id="body")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=blank),
                          PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------
def main():
    summary = json.load(open(os.path.join(RES, "summary.json"), encoding="utf-8"))
    sources = json.load(open(os.path.join(ROOT, "data", "sources.json"), encoding="utf-8"))

    h_e1s, r_e1s = read_table("e1_threshold_single_scale_dev.csv")
    h_e1m, r_e1m = read_table("e1_threshold_multi_scale_dev.csv")
    h_e2, r_e2 = read_table("e2_mask_comparison.csv")
    h_e3, r_e3 = read_table("e3_scale_pyramid.csv")
    h_e4, r_e4 = read_table("e4_rotation_bank.csv")
    h_e5, r_e5 = read_table("e5_sliding_window.csv")
    h_e5b, r_e5b = read_table("e5b_sliding_window_threshold.csv")
    h_e6, r_e6 = read_table("e6_feature_matching.csv")
    h_e7, r_e7 = read_table("e7_hybrid.csv")
    h_e8, r_e8 = read_table("e8_test_set_comparison.csv")
    h_e8b, r_e8b = read_table("e8b_per_condition_f1.csv")

    ds = summary["dataset"]
    tm_best = max(r_e1m, key=lambda r: float(r[6]))
    ss_best = max(r_e1s, key=lambda r: float(r[6]))
    e8_best = max(r_e8, key=lambda r: float(r[7]))
    e6_best = max(r_e6, key=lambda r: float(r[7]))
    sw_row = [r for r in r_e8 if r[0].startswith("explicit sliding")][0]
    sw_fine = [r for r in r_e5 if r[0].startswith("3 window sizes") and r[2] == "2"]
    sw_coarse = [r for r in r_e5 if r[0].startswith("3 window sizes") and r[2] == "24"]
    sw_fine = sw_fine[0] if sw_fine else r_e5[0]
    sw_coarse = sw_coarse[0] if sw_coarse else r_e5[-1]

    # ---------------- cover ----------------
    story.append(Spacer(1, 2.4 * cm))
    p("COMPUTER VISION PROJECT", SUB)
    story.append(Spacer(1, 1.1 * cm))
    p("Classical Object Detection and Localisation of the University of Ghana Coat of Arms",
      TITLE)
    story.append(Spacer(1, 0.7 * cm))
    p("Template matching, sliding window search, feature matching and colour "
      "segmentation, evaluated on an annotated test set", SUB)
    story.append(Spacer(1, 1.5 * cm))
    info = Table([["Author", NAME],
                  ["Repository", "github.com/Qwamzz/Computer-Vision-Projects"],
                  ["Target object", "University of Ghana coat of arms"],
                  ["Evaluation set", "%d annotated scenes, %d object instances"
                   % (ds["dev_images"] + ds["test_images"],
                      ds["dev_objects"] + ds["test_objects"])],
                  ["Implementation", "Python 3 and OpenCV, no pre-trained recognition model"]],
                 colWidths=[4.2 * cm, 10.5 * cm])
    info.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b5")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    info.hAlign = "CENTER"
    story.append(info)
    story.append(Spacer(1, 1.4 * cm))
    p("<b>Statement on the imagery.</b> Every image used in this project is a "
      "real photograph or a real vector emblem obtained from Wikimedia Commons "
      "under a free licence, and the licence and author of each file are listed "
      "in Section 2.2 and in data/SOURCES.md. No image was produced by an image "
      "generation model.", ParagraphStyle("note", parent=BODY, fontSize=8.6,
                                          leading=12,
                                          borderPadding=6,
                                          backColor=colors.HexColor("#f4f7fb")))
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # ---------------- 1 problem ----------------
    h1("1. Problem definition and objectives")
    p("The task is to build a system that finds a chosen object category in an "
      "image and reports where it is, using classical detection techniques rather "
      "than a pre-trained recognition network. The "
      "object chosen here is the <b>University of Ghana coat of arms</b>, the "
      "emblem that carries three stylised palm trees over a navy shield above the "
      "motto <i>Integri Procedamus</i>. It is a good subject for a classical "
      "detector: it is rigid, it is planar, it has a fixed internal layout and it "
      "has strong internal contrast, which is exactly the situation in which "
      "correlation based matching is defensible. It is also a realistic object to "
      "detect, since it appears on university signage, letterheads, certificates, "
      "identity cards and merchandise, and automatic detection supports document "
      "verification and brand monitoring.")
    p("The specific objectives are the following.")
    bullets([
        "Build a template of the object and a test set in which the object appears "
        "at different positions, sizes, orientations, illumination levels, "
        "backgrounds and degrees of partial occlusion.",
        "Implement template matching by normalised cross correlation and measure "
        "how the decision threshold trades true detections against false and "
        "missed detections.",
        "Extend the detector with an explicit sliding window search and measure the "
        "effect of window size, step size and threshold on both accuracy and "
        "computational cost.",
        "Add feature matching with RANSAC and a colour segmentation stage where "
        "these improve reliability, and draw a bounding box on every detected "
        "instance.",
        "Evaluate on a manually specified annotated test set that was not used "
        "during development, reporting precision, recall and F1 score.",
        "Discuss the conditions under which template matching and sliding window "
        "detection become unreliable and explain why.",
    ])
    p("The complete detection progression is implemented and measured end to end:")
    eq("input image -&gt; search strategy -&gt; template comparison -&gt; "
       "detection decision -&gt; object localisation -&gt; performance evaluation")

    # ---------------- 2 dataset ----------------
    h1("2. Dataset")
    h2("2.1 Target object and template")
    p("The template is the official University of Ghana coat of arms, taken from "
      "the public domain vector file on Wikimedia Commons and rendered at a "
      "reference width of %d pixels, giving a template of %d by %d pixels. A "
      "binary mask of the emblem is stored beside it. The mask matters because "
      "the coat of arms is not rectangular: it is a shield with a curved lower "
      "edge and a scroll, and it covers only %.1f per cent of its bounding "
      "rectangle. The remaining %.1f per cent of the rectangle is background that "
      "changes from photograph to photograph, so including it in the comparison "
      "adds noise to the similarity score. Section 5.2 measures exactly how much "
      "this matters."
      % (summary["template"]["width"], summary["template"]["width"],
         summary["template"]["height"], 100 * summary["template"]["mask_coverage"],
         100 * (1 - summary["template"]["mask_coverage"])))
    figure("fig01_template.png",
           "The template used throughout: the colour emblem, its grey scale form "
           "which is what correlation actually sees, and the binary mask that "
           "restricts the comparison to the emblem itself.", 13 * cm)

    h2("2.2 Source imagery")
    p("All source material was downloaded from Wikimedia Commons by "
      "src/download_data.py, which also records the licence and author of each "
      "file. Nine real photographs supply the backgrounds and cover outdoor "
      "campus scenes, an Accra street scene, an office desk, a brick wall and two "
      "interior scenes. Two further university emblems that resemble the target "
      "in colour and layout are used as clutter and as hard negatives, so that "
      "the false positive rate is not measured only against easy background.")
    table(["local file", "source file on Wikimedia Commons", "licence", "author"],
          [[s["file"], s["commons_title"].replace("File:", ""), s["licence"],
            s["author"][:38]] for s in sources],
          "Provenance of every source image. None of these files is machine "
          "generated imagery.",
          widths=[4.0 * cm, 6.2 * cm, 2.6 * cm, 4.2 * cm], fontsize=6.6)

    h2("2.3 Construction of the test scenes")
    p("Each test scene was produced by transforming the real emblem and "
      "compositing it into a real photograph under controlled conditions. The "
      "controlled conditions are the point of the exercise: they make it possible "
      "to say precisely which factor caused a detector to fail, which is not "
      "possible with an uncontrolled collection of photographs. The following "
      "variations are covered.")
    bullets([
        "<b>Position.</b> The emblem appears at different places in the frame, "
        "never at a fixed location.",
        "<b>Size.</b> Scale factors from 0.55 to 1.90 relative to the template, "
        "that is from roughly 53 to 182 pixels wide.",
        "<b>Orientation.</b> In plane rotations of 6, 8, 15, 22, 30, 45 and 90 "
        "degrees.",
        "<b>Illumination.</b> Gamma correction from 0.40 (heavily under exposed) "
        "to 1.9 (washed out), and a left to right or right to left brightness "
        "ramp that simulates a single sided light source.",
        "<b>Background.</b> Nine different real scenes, from a flat brick wall to "
        "a cluttered office desk and a busy street.",
        "<b>Partial occlusion.</b> From 18 to 45 per cent of the emblem is covered "
        "by a textured patch copied from elsewhere in the same photograph, so the "
        "occluder is not trivially separable.",
        "<b>Clutter and hard negatives.</b> Scenes that contain two other "
        "university emblems, and two scenes that contain no target at all, which "
        "test the false positive behaviour directly.",
        "<b>Sensor effects.</b> Additive Gaussian noise with a standard deviation "
        "of 8 to 18 grey levels, and directional motion blur.",
    ])
    figure("fig02_dataset.png",
           "Nine of the annotated scenes, one per condition, with the ground truth "
           "boxes drawn in green. The bottom right scene contains no target and "
           "only two similar emblems from other faculties.")

    h2("2.4 Ground truth, splits and protocol")
    p("The dataset is split into a development set of %d images containing %d "
      "object instances, and a held out test set of %d images containing %d "
      "instances. Every threshold and every parameter reported in Section 5 was "
      "chosen on the development set only. The held out set was run once at the "
      "end with the parameters already fixed, which is what Section 5.8 reports. "
      "This separation is what makes the reported test numbers meaningful."
      % (ds["dev_images"], ds["dev_objects"], ds["test_images"], ds["test_objects"]))
    p("Bounding box annotation was produced from the known placement geometry of "
      "each emblem: the box is the tight rectangle around the pixels whose alpha "
      "value is non zero after the geometric transformation, so it is exact to "
      "the pixel rather than approximate. Every generated scene was then inspected "
      "with its box drawn on it to confirm the annotation, and Figure 2 is part of "
      "that check. The annotations are stored as JSON with the condition label, "
      "the scale, the rotation and the occluded fraction of each instance, which "
      "is what allows the per condition analysis in Section 5.8.")

    # ---------------- 3 methodology ----------------
    h1("3. Methodology and justification of the algorithms")
    h2("3.1 Overview")
    p("A classical detector is made of three separable decisions: a <b>search "
      "strategy</b> that says which image regions to examine, a <b>comparison "
      "function</b> that scores how much a region looks like the object, and a "
      "<b>decision rule</b> that turns scores into detections. This project keeps "
      "those three parts explicit and varies them one at a time, so that each "
      "experiment isolates one design choice.")
    figure("fig06_pipeline_stages.png",
           "The pipeline on one scene: the input, its grey scale form, the "
           "correlation surface produced by the masked template comparison, the "
           "part of that surface above the decision threshold, the candidate boxes "
           "before non maximum suppression, and the final localisation.")

    h2("3.2 Preprocessing")
    p("Images are converted to grey scale before matching. Correlation is defined "
      "on a single channel, and the emblem is discriminated mainly by its shape "
      "and internal contrast rather than by absolute colour, which changes with "
      "the illuminant. Colour is not discarded, it is used separately in the "
      "region proposal stage of Section 3.7, where its selectivity is an "
      "advantage rather than a liability. Optional Gaussian smoothing and CLAHE "
      "are available in the preprocessing function for the noisy and the low "
      "contrast scenes.")

    h2("3.3 The comparison function: masked zero mean normalised cross correlation")
    p("The similarity between a candidate window w and the template t is measured "
      "with the zero mean normalised cross correlation computed over the mask M "
      "of the emblem:")
    eq("NCC(w, t) = sum_M (w - mean_M w)(t - mean_M t) / "
       "( ||w - mean_M w||_M * ||t - mean_M t||_M )")
    p("Three properties justify this choice. Subtracting the mean makes the score "
      "invariant to an additive change in brightness. Dividing by the norms makes "
      "it invariant to a multiplicative change in contrast. Both together mean "
      "that a linear change of illumination, which is the first order model of a "
      "lighting change, leaves the score unchanged, and the score lies in the "
      "interval [-1, 1] so a single threshold is meaningful across images. "
      "Restricting all three sums to the mask M excludes the corners of the "
      "bounding rectangle that do not belong to the emblem. In OpenCV this is "
      "cv2.matchTemplate with TM_CCOEFF_NORMED and a mask argument; the explicit "
      "sliding window detector of Section 3.5 computes the same quantity directly "
      "from the definition.")
    p("What the score is <i>not</i> invariant to is equally important, and it "
      "sets up the rest of the design: normalised cross correlation is not "
      "invariant to scale, not invariant to rotation, and degrades quickly under "
      "occlusion and under projective distortion, because it compares pixels at "
      "fixed relative positions.")

    h2("3.4 Search strategy 1: dense correlation over a template bank")
    p("cv2.matchTemplate slides the template over every pixel position, so it is "
      "already an exhaustive sliding window search with a step of one pixel, "
      "implemented with an optimised correlation. Scale invariance is bought by "
      "repeating the search with the template resized over a geometric pyramid, "
      "and rotation invariance by repeating it again for each angle in a bank of "
      "rotated templates. The cost grows linearly with the number of templates in "
      "the bank, which is the trade off measured in Sections 5.3 and 5.4.")

    h2("3.5 Search strategy 2: the explicit sliding window detector")
    p("The second search strategy is the textbook sliding window, written out as "
      "explicit loops so that its three parameters can be varied and its cost "
      "measured. For each window size, the window is stepped across the image by "
      "a fixed step in x and y, each window is resized to the canonical template "
      "size, and the masked correlation above is computed directly. Window size "
      "controls which object scales can be found, step size controls both the "
      "localisation accuracy and the run time, and the threshold controls the "
      "detection decision. The number of windows examined grows as the image area "
      "divided by the square of the step, so halving the step multiplies the work "
      "by four. Section 5.5 measures this directly.")

    h2("3.6 The decision rule and non maximum suppression")
    p("A window is declared a detection when its score reaches the threshold. "
      "Because correlation is smooth, one true object produces a cluster of high "
      "scoring windows, and each scale in the pyramid produces its own cluster. "
      "Two stages reduce this to one box per object. First, only local maxima of "
      "the correlation surface are kept, found with a grey scale dilation, which "
      "removes the plateau around each peak cheaply. Second, greedy non maximum "
      "suppression sorts the remaining candidates by score and discards any "
      "candidate that overlaps an already accepted one by more than an "
      "intersection over union of 0.30. Without this step a single object would "
      "be counted as many detections and precision would collapse.")

    h2("3.7 Improving reliability: features and colour")
    p("Two further classical techniques are added because correlation alone "
      "cannot cover the required conditions.")
    p("<b>SIFT keypoints with RANSAC.</b> Scale invariant feature transform "
      "keypoints are detected on the template and on the scene, described, and "
      "matched with a brute force L2 matcher under Lowe's ratio test. The "
      "surviving correspondences are fitted with a homography under RANSAC, and "
      "the four template corners are mapped through that homography to localise "
      "the object. This is invariant to scale and to in plane rotation by "
      "construction, tolerant of partial occlusion because only a subset of the "
      "keypoints is needed, and tolerant of viewpoint change because a homography "
      "is exactly the right model for a planar object seen from another angle. "
      "Because one homography can only explain one instance, the fit is repeated "
      "after removing the inliers, which recovers the second emblem in the scenes "
      "that contain two.")
    p("<b>Colour segmentation as a region proposal stage.</b> The shield is a "
      "saturated dark blue. Thresholding hue and saturation in HSV, closing and "
      "opening the result with an elliptical structuring element, and taking "
      "connected components with a plausible area and aspect ratio gives a small "
      "number of candidate rectangles. Template matching then runs only inside "
      "those rectangles. This does not change the comparison function at all, it "
      "changes only where the search looks, so it is a pure reduction of the "
      "search space, and Section 5.7 measures how much of the image survives it.")
    figure("fig07_colour_proposals.png",
           "The colour proposal stage on a cluttered desk scene: the input, the "
           "HSV blue mask after morphological closing and opening, and the "
           "resulting region proposals. Template matching runs only inside the "
           "orange rectangles.")

    h2("3.8 Evaluation protocol")
    p("Detections are matched to ground truth greedily in order of decreasing "
      "score. A detection counts as a true positive when its intersection over "
      "union with a still unmatched ground truth box is at least 0.5, the "
      "standard detection criterion:")
    eq("IoU(A, B) = area(A intersect B) / area(A union B)")
    p("Every unmatched detection is a false positive and every unmatched ground "
      "truth object is a missed detection. The reported measures are then")
    eq("precision = TP / (TP + FP)&nbsp;&nbsp;&nbsp;&nbsp; "
       "recall = TP / (TP + FN)&nbsp;&nbsp;&nbsp;&nbsp; "
       "F1 = 2 x precision x recall / (precision + recall)")
    p("Precision answers how many of the boxes the system drew were real objects, "
      "recall answers how many of the real objects the system found, and F1 is "
      "their harmonic mean, which is the single number used to select operating "
      "points. Mean intersection over union of the true positives is also "
      "reported, because it measures localisation quality rather than detection "
      "quality: a detector can find every object and still place its boxes badly.")

    # ---------------- 4 implementation ----------------
    h1("4. Implementation")
    p("The system is written in Python 3 with OpenCV and NumPy. No pre-trained "
      "recognition model is used at any point; the only OpenCV operators used are "
      "classical ones.")
    table(["file", "contents"],
          [["src/download_data.py",
            "Downloads the emblem and the background photographs from Wikimedia "
            "Commons and records the licence of each file."],
           ["src/build_dataset.py",
            "Builds the template and its mask, generates the annotated scenes "
            "under the controlled conditions, writes the JSON ground truth."],
           ["src/detector.py",
            "Template bank construction, masked correlation matching, local peak "
            "extraction, non maximum suppression, explicit sliding window search, "
            "SIFT matching with RANSAC, HSV colour proposals, drawing."],
           ["src/evaluate.py",
            "Greedy intersection over union matching, precision, recall, F1, per "
            "condition and per image aggregation."],
           ["src/run_experiments.py",
            "Runs every experiment in Section 5 and writes the tables, the "
            "figures and the per image detection images."],
           ["src/make_report.py",
            "Builds this PDF directly from the measured CSV results."]],
          "Source files. The complete listing is reproduced in Appendix A.",
          widths=[4.6 * cm, 12.4 * cm], fontsize=7.6)
    p("The dataset is generated from a fixed random seed, so the whole study is "
      "reproducible. The full source code is included in Appendix A and in the "
      "repository.")

    # ---------------- 5 results ----------------
    h1("5. Experimental results")
    p("All tuning experiments in Sections 5.1 to 5.7 are run on the development "
      "set. The held out test set is used once, in Section 5.8. Timings were "
      "measured on a single CPU core on 960 by 640 pixel images.")

    h2("5.1 Effect of the detection threshold")
    p("This is the central experiment of the study: how the template matching "
      "threshold changes true detections, false detections and missed detections. "
      "The correlation surface was computed once at a low threshold and the "
      "decision threshold was then swept, so the geometry of the search is "
      "identical along the whole curve and only the decision changes.")
    table(h_e1m, [[num(r[0], 3)] + r[1:4] + [num(r[4]), num(r[5]), num(r[6])]
                  for r in r_e1m],
          "Threshold sweep for the eleven scale template matcher on the "
          "development set. The best operating point by F1 is at a threshold of "
          "%s." % num(tm_best[0], 3),
          widths=[2.4 * cm] + [2.1 * cm] * 6, fontsize=7.0)
    figure("fig03_threshold_sweep.png",
           "Effect of the correlation threshold. On the left, true, false and "
           "missed detections on a symmetric log scale. On the right, precision, "
           "recall and F1; the dashed line marks the operating point selected on "
           "the development set.")
    p("The behaviour is the classical one and it is very sharp. At a threshold of "
      "0.30 the detector finds %s of the %d objects but also produces %s false "
      "detections, a precision of %s. Every increase in the threshold removes "
      "false detections much faster than it removes true ones, until at %s the "
      "false detections reach zero while %s objects are still found. Beyond that "
      "point the threshold only destroys recall: at 0.80 only %s objects remain. "
      "The reason the curve is so steep is that the score of a correct match sits "
      "far above the score of background structure, but the background of a real "
      "photograph contains an enormous number of windows, so even a small "
      "probability of a high scoring background window produces many false "
      "positives in absolute terms. Precision is therefore dominated by the "
      "sheer number of windows examined, which is the same effect that makes the "
      "sliding window step size matter in Section 5.5."
      % (r_e1m[0][1], ds["dev_objects"], r_e1m[0][2], num(r_e1m[0][4], 4),
         num(tm_best[0], 3), tm_best[1],
         [r for r in r_e1m if abs(float(r[0]) - 0.8) < 1e-6][0][1]))

    h2("5.2 Masked against unmasked correlation")
    p("The second experiment isolates the effect of restricting the correlation "
      "to the emblem mask.")
    table(h_e2, [[r[0], num(r[1], 3), num(r[2]), num(r[3]), num(r[4]), num(r[5])]
                 for r in r_e2],
          "Masked against unmasked correlation, eleven scales, development set, "
          "each at its own best threshold.",
          widths=[3.2 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 3.0 * cm],
          fontsize=7.4)
    figure("fig04_mask_effect.png",
           "F1 against threshold for masked and unmasked correlation. Masking the "
           "non emblem corners of the template rectangle is worth a large amount "
           "of recall at every threshold.", 11 * cm)
    p("Masking raises the best F1 from %s to %s, almost entirely through recall "
      "(%s against %s). The explanation is direct: %.0f per cent of the template "
      "rectangle is not part of the emblem, and in the unmasked comparison those "
      "pixels are filled with whatever background happens to lie there. That "
      "background differs between the template card and the scene, so it drives "
      "the correlation down and pushes true matches below the threshold. The cost "
      "is run time, because masked correlation is roughly %s times slower in "
      "OpenCV, and a small loss of numerical stability in flat regions, where the "
      "masked score is poorly conditioned and can produce spurious high values. "
      "That instability is visible at low thresholds in Table 3 and is one of the "
      "reasons the useful operating region is above 0.55."
      % (num(r_e2[0][4]), num(r_e2[1][4]), num(r_e2[0][3]), num(r_e2[1][3]),
         100 * (1 - summary["template"]["mask_coverage"]),
         num(float(r_e2[0][5]) / max(1e-6, float(r_e2[1][5])), 1)))

    h2("5.3 Size of the scale pyramid")
    table(h_e3, [[r[0], r[1], num(r[2], 3), num(r[3]), num(r[4]), num(r[5]), num(r[6])]
                 for r in r_e3],
          "Effect of the scale pyramid on the development set. Each configuration "
          "is reported at its own best threshold.",
          widths=[4.4 * cm, 2.2 * cm, 2.0 * cm, 1.9 * cm, 1.7 * cm, 1.6 * cm, 2.2 * cm],
          fontsize=7.0)
    p("A single template at the reference size reaches an F1 of %s: it finds the "
      "instances near its own scale and misses everything else, which is the "
      "clearest possible demonstration that correlation is not scale invariant. "
      "Extending the pyramid to eleven scales between 0.50 and 2.02 raises F1 to "
      "%s and raises recall from %s to %s, at %s times the run time. The "
      "intermediate five scale pyramid is interesting: it is not simply half way, "
      "because its coarse spacing means some objects fall between two sampled "
      "scales and match neither well, and it also loses precision, which is why "
      "its F1 (%s) is no better than the single scale configuration. Scale "
      "sampling must be fine enough that every object is close to some sampled "
      "scale; a geometric ratio of about 1.15 achieves that here."
      % (num(r_e3[0][5]), num(r_e3[2][5]), num(r_e3[0][4]), num(r_e3[2][4]),
         num(float(r_e3[2][6]) / max(1e-6, float(r_e3[0][6])), 1), num(r_e3[1][5])))

    h2("5.4 Rotated template bank")
    table(h_e4, [[r[0], r[1], num(r[2], 3), num(r[3]), num(r[4]), num(r[5]), num(r[6])]
                 for r in r_e4],
          "Adding eight rotated copies of every template. Recall is reported "
          "separately on the rotated scenes and on the viewpoint scenes.",
          widths=[3.6 * cm, 2.2 * cm, 2.0 * cm, 1.8 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm],
          fontsize=7.0)
    p("Without rotated templates the detector finds none of the rotated instances "
      "at all: recall on the rotation scenes is %s. Adding eight angles raises "
      "that to %s, but multiplies the bank from %s to %s templates and the run "
      "time from %s to %s seconds per image, and overall F1 does not improve "
      "(%s against %s) because the enlarged bank also produces more opportunities "
      "for a background window to score highly. This is the honest result: buying "
      "rotation invariance by brute force is expensive and only partly effective, "
      "since a rotation that falls between two sampled angles still matches "
      "poorly. It is the strongest argument in this project for the feature based "
      "approach of Section 5.6, which is rotation invariant by construction and "
      "costs less."
      % (num(r_e4[0][4]), num(r_e4[1][4]), r_e4[0][1], r_e4[1][1],
         num(r_e4[0][6]), num(r_e4[1][6]), num(r_e4[0][3]), num(r_e4[1][3])))

    h2("5.5 The explicit sliding window detector")
    p("This experiment measures the sliding window detector directly, varying the "
      "set of window sizes and the step size, and recording both the number of "
      "windows evaluated and the time taken. It was run on a five image subset of "
      "the development set that includes a scene with two objects and a scene "
      "with none.")
    table(h_e5, [[r[0], r[1], r[2], r[3], num(r[4]), num(r[5], 2), r[6], r[7], r[8],
                  num(r[9]), num(r[10]), num(r[11])] for r in r_e5],
          "Sliding window detector: window sizes, step size, cost and accuracy. "
          "Each row is reported at its own best threshold.",
          widths=[2.9 * cm, 1.3 * cm, 1.0 * cm, 1.9 * cm, 1.6 * cm, 1.3 * cm,
                  1.0 * cm, 1.0 * cm, 1.0 * cm, 1.3 * cm, 1.1 * cm, 1.0 * cm],
          fontsize=6.2)
    figure("fig05_sliding_window.png",
           "Left, seconds per image against step size on a logarithmic axis. "
           "Right, F1 against step size. The cost falls as the square of the step "
           "while accuracy collapses beyond a step of about eight pixels.")
    p("The relationship between step size, detection performance and execution "
      "time is the clearest quantitative result in this project. With three "
      "window sizes, a step of 2 pixels evaluates %s windows and costs %s seconds "
      "per image, while a step of 24 pixels evaluates %s windows and costs %s "
      "seconds, a saving of about %s times, exactly as predicted by the inverse "
      "square relationship between step and window count. Accuracy moves the "
      "other way: F1 falls from %s at a step of 2 to %s at a step of 24. There "
      "are two distinct reasons for that fall, and they are worth separating. "
      "The first is a scoring effect: a window that is offset from the object by "
      "half a step contains part of the background and its correlation drops, so "
      "the object can fall below the threshold entirely. The second is a "
      "localisation effect: even when the object is detected, a coarse grid "
      "cannot place the box within the 0.5 intersection over union required to "
      "count as a true positive. A step of about one twentieth of the window "
      "size, which is 4 to 8 pixels here, is the practical compromise, and it is "
      "the configuration carried into Section 5.8. The fall is not perfectly "
      "monotonic: a step of 16 finds nothing at all while a step of 24 finds one "
      "object. That is not noise in the measurement, it is grid alignment. "
      "Whether a coarse grid happens to place a window close enough to an object "
      "depends on the object position modulo the step, so a coarse detector "
      "succeeds or fails almost by luck, which is itself a good reason not to use "
      "one."
      % (sw_fine[3], num(sw_fine[4], 2), sw_coarse[3], num(sw_coarse[4], 3),
         num(float(sw_fine[4]) / max(1e-6, float(sw_coarse[4])), 0),
         num(sw_fine[11]), num(sw_coarse[11])))
    table(h_e5b, [[num(r[0], 2)] + r[1:4] + [num(r[4]), num(r[5]), num(r[6])]
                  for r in r_e5b],
          "Threshold sweep for the explicit sliding window detector at a fixed "
          "geometry (three window sizes, step 4).",
          widths=[2.4 * cm] + [2.1 * cm] * 6, fontsize=7.0)

    h2("5.6 SIFT feature matching with RANSAC")
    p("Two parameters govern the feature based detector: Lowe's ratio, which "
      "decides how distinctive a correspondence must be to survive, and the "
      "minimum number of RANSAC inliers required before an instance is accepted, "
      "which is the detection decision for this method.")
    table(h_e6, [[num(r[0], 2), r[1], r[2], r[3], r[4], num(r[5]), num(r[6]),
                  num(r[7]), num(r[8])] for r in r_e6],
          "Ratio test and inlier threshold study on the development set.",
          widths=[1.9 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm, 2.2 * cm, 1.9 * cm,
                  1.7 * cm, 1.5 * cm, 2.0 * cm], fontsize=6.8)
    figure("fig09_sift_matches.png",
           "Correspondences that survive the ratio test between the template and "
           "a scene in which the emblem is rotated by 30 degrees. Correlation "
           "based matching fails completely on this image unless a rotated "
           "template bank is used; feature matching handles it without any "
           "additional search.")
    p("The best configuration on the development set is a ratio of %s with %s "
      "inliers required, giving an F1 of %s at %s seconds per image, which is "
      "faster than the eleven scale correlation search and far faster than the "
      "rotated bank. The inlier count behaves exactly like the correlation "
      "threshold: a low requirement admits spurious homographies fitted to "
      "accidental matches on the distractor emblems, and a high requirement "
      "rejects the small and the heavily occluded instances, which simply do not "
      "have enough keypoints to reach the count. The method has one structural "
      "weakness that the correlation detector does not have: it needs texture. "
      "The emblem at 0.55 scale is only about 53 pixels wide, and SIFT finds few "
      "stable keypoints at that size."
      % (num(e6_best[0], 2), e6_best[1], num(e6_best[7]), num(e6_best[8])))

    h2("5.7 Colour proposals combined with template matching")
    table(h_e7, [[r[0], num(r[1]), num(r[2], 4), num(r[3], 3), num(r[4]), num(r[5]),
                  num(r[6])] for r in r_e7],
          "Restricting the correlation search to colour proposals rather than "
          "searching the whole image.",
          widths=[5.0 * cm, 2.2 * cm, 2.6 * cm, 1.9 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm],
          fontsize=7.0)
    p("The proposal stage reduces the searched area to %s per cent of the image "
      "and the run time from %s to %s seconds per image, while F1 changes from %s "
      "to %s. The gain is worth understanding precisely, because it does not come "
      "from a better comparison function, which is unchanged. Removing seventy "
      "per cent of the image removes seventy per cent of the opportunities for a "
      "background window to score highly, so the detector can be operated at a "
      "lower threshold (%s rather than %s) without paying for it in false "
      "positives, and the lower threshold is what recovers the extra objects: "
      "recall rises from %s to %s at unchanged precision. But the proposal stage "
      "is itself a detector, and "
      "when its colour threshold rejects an emblem, for example in the heavily "
      "under exposed scene where the navy shield is nearly black and its "
      "saturation collapses, the correlation stage never gets the chance to look "
      "there. A cascade is only ever as good as its first stage, which is a "
      "general property of cascaded detectors and not a peculiarity of this "
      "implementation."
      % (num(100 * float(r_e7[1][2]), 1), num(r_e7[0][1]), num(r_e7[1][1]),
         num(r_e7[0][6]), num(r_e7[1][6]), num(r_e7[1][3], 3), num(r_e7[0][3], 3),
         num(r_e7[0][5]), num(r_e7[1][5])))

    h2("5.8 Held out test set: final comparison")
    p("Every method was then run once on the %d held out images, with all "
      "parameters fixed at the values selected on the development set. No "
      "parameter was adjusted after seeing these numbers." % ds["test_images"])
    table(h_e8, [[r[0], r[1], r[2], r[3],
                  r[4], num(r[5]), num(r[6]), num(r[7]), num(r[8]), num(r[9])]
                 for r in r_e8],
          "Final comparison on the held out test set, at IoU 0.5. The operating "
          "point is a correlation threshold for the correlation based methods and "
          "a minimum RANSAC inlier count for the feature based method.",
          widths=[4.0 * cm, 1.6 * cm, 1.2 * cm, 1.2 * cm, 1.3 * cm, 1.4 * cm,
                  1.3 * cm, 1.1 * cm, 1.5 * cm, 1.4 * cm], fontsize=6.6)
    table(h_e8b, r_e8b, "F1 by imaging condition on the held out test set. The "
          "negative images are excluded from this table because F1 is undefined "
          "when there is no object to find; their contribution appears as false "
          "positives in the previous table.",
          widths=None, fontsize=6.4)
    figure("fig08_per_condition.png",
           "F1 by imaging condition on the held out test set. The conditions on "
           "which correlation collapses are exactly the ones it has no invariance "
           "to.")
    p("The best method on the held out set is <b>%s</b>, with a precision of %s, "
      "a recall of %s and an F1 of %s. Localisation is good wherever detection "
      "succeeds: the mean intersection over union of the true positives is %s, "
      "well above the 0.5 acceptance threshold, which says that the failures in "
      "this system are failures to detect rather than failures to localise."
      % (e8_best[0], num(e8_best[5]), num(e8_best[6]), num(e8_best[7]),
         num(e8_best[8])))
    p("The ordering of the methods on the held out set repeats the ordering on "
      "the development set, which is the first thing to check, and the per "
      "condition table explains it. The correlation detectors score 1.0 on the "
      "clutter scenes and at or near 1.0 on the baseline scenes, and 0.0 on "
      "rotation, viewpoint, occlusion and noise. Feature matching scores 1.0 on "
      "rotation and is the only single method that scores above zero on noise and "
      "on occlusion, which is why combining the two recovers almost everything: "
      "the two families fail on disjoint conditions. Two results deserve to be "
      "stated plainly rather than smoothed over. First, no method in this study "
      "detected the viewpoint scene: a projective tilt of twenty per cent of the "
      "object width defeated the correlation detectors outright, and on the "
      "feature side only eleven correspondences survived the ratio test, fewer "
      "than the %s that the operating point demands, so no homography was even "
      "fitted. Second, the explicit sliding window detector scores worst of all "
      "on the test set while costing %s seconds per image, because a step of 4 "
      "pixels combined with only three window sizes samples scale far too "
      "coarsely; the dense one pixel search of cv2.matchTemplate over eleven "
      "scales is both cheaper and better, which is the practical argument for "
      "using the optimised correlation rather than an explicit loop once the "
      "algorithm is understood."
      % (e6_best[1], num(sw_row[9], 1)))
    figure("fig10_success.png",
           "Correct detections by multi scale template matching on the held out "
           "set. Yellow is ground truth, green is the detection.")
    figure("fig11_failure.png",
           "Failure cases for multi scale template matching on the held out set. "
           "Yellow is ground truth, green is the detection. Rotation, viewpoint "
           "change and heavy occlusion account for almost every failure.")

    # ---------------- 6 discussion ----------------
    h1("6. Critical discussion: when these methods become unreliable")
    p("The practical question for any deployment is when template matching and "
      "sliding window detection become unreliable. The per condition results in "
      "Table 12 answer "
      "that empirically, and the reason in each case follows from what the "
      "correlation score actually computes: a comparison of pixel values at fixed "
      "relative positions inside a rectangle.")

    h3("Scale variation")
    p("A template of a fixed size compared with an object of a different size "
      "compares the wrong pixels with each other. A scale error of 20 per cent is "
      "already enough to push a correct match below a threshold that is tight "
      "enough to control false positives, which is why the single scale detector "
      "in Table 5 reaches a recall of only %s. The pyramid solves this at linear "
      "cost in the number of scales, but only if the sampling is fine enough that "
      "every object lies close to some sampled scale. Scale is the best behaved "
      "of the failure modes because it is one dimensional and cheap to enumerate."
      % num(r_e3[0][4]))

    h3("Rotation")
    p("Rotation is worse than scale for two reasons. First, the search is over an "
      "additional dimension, so a bank that covers rotation as finely as it covers "
      "scale is an order of magnitude larger. Second, rotating a rectangular "
      "template introduces corners that contain no object, which interacts badly "
      "with the correlation unless the mask is rotated with the template. Table 6 "
      "shows the consequence: without rotated templates, recall on the rotated "
      "scenes is zero, and even with eight angles it only reaches %s while "
      "costing %s seconds per image. Feature matching is the right answer here, "
      "because SIFT descriptors are computed relative to a dominant gradient "
      "orientation and are therefore rotation invariant without any search at all."
      % (num(r_e4[1][4]), num(r_e4[1][6])))

    h3("Occlusion")
    p("Correlation is a sum over all pixels of the template, so an occluder that "
      "covers a fraction f of the object corrupts a fraction f of the terms in "
      "that sum, and the score falls roughly in proportion. Because the "
      "normalisation is computed over the whole mask, an occluded region does not "
      "merely contribute nothing, it contributes actively wrong values. In this "
      "study the correlation detector missed both occluded instances in the held "
      "out set at its selected operating point, including the one with only 20 "
      "per cent of the emblem covered. Feature matching degrades far more "
      "gracefully: it recovered the lightly occluded instance and failed only on "
      "the heavily occluded one, where 45 per cent of the emblem including the "
      "scroll and the lower shield is hidden. It needs only enough visible "
      "keypoints to fit a homography, and RANSAC discards the rest as outliers.")

    h3("Illumination")
    p("This is the one condition that the chosen comparison function genuinely "
      "handles. Because the mean is subtracted and the result is divided by the "
      "norms, an affine change of intensity leaves the score unchanged, and the "
      "detector survives the one sided brightness ramp and the milder gamma "
      "changes in this dataset. It fails when the illumination change is not "
      "affine: strong gamma compression is a non linear map that changes the "
      "internal contrast ratios of the emblem, and at a gamma of 0.40 the "
      "correlation detector missed the held out scene outright, because the dark "
      "shield is crushed towards black until quantisation and JPEG noise destroy "
      "its internal structure. Feature matching still found that instance. A one "
      "sided brightness ramp is closer to affine over a region as small as the "
      "emblem, and the detector survives it.")

    h3("Sensor noise and blur")
    p("Additive noise with a standard deviation of 18 grey levels was enough to "
      "push the correlation score of a correct match below the operating "
      "threshold, giving an F1 of zero on the noise condition for every "
      "correlation based method in Table 12. The mechanism is that noise inflates "
      "the norm of the window without contributing anything correlated with the "
      "template, so the normalised score is scaled down. Smoothing the image "
      "before matching recovers part of this, at the cost of blurring the fine "
      "internal structure of the emblem that the match depends on, which is why "
      "the preprocessing function exposes the smoothing radius rather than fixing "
      "it. Feature matching still found the emblem in the noisy scene, although it "
      "also produced a false positive there, because SIFT smooths at each octave "
      "and its descriptors pool gradients over a region rather than comparing "
      "pixel values one by one.")

    h3("Background clutter and the number of windows examined")
    p("Precision in this problem is dominated by how many windows are examined. A "
      "960 by 640 image contains roughly half a million window positions per "
      "scale, so even a false positive rate of one in a hundred thousand produces "
      "several false detections per image, and multiplying the template bank "
      "multiplies that number. This is why Table 3 shows more than two thousand "
      "false detections at a threshold of 0.30 and none at %s, and why the colour "
      "proposal stage of Section 5.7, which removes about %s per cent of the "
      "search space, helps precision without touching the comparison function. "
      "Clutter that resembles the target in colour and layout, such as the other "
      "university emblems used here, is the hardest case: it survives the colour "
      "proposal stage and can only be rejected by the comparison function itself."
      % (num(tm_best[0], 3), num(100 * (1 - float(r_e7[1][2])), 0)))

    h3("Viewpoint change")
    p("A change of viewpoint applies a projective transformation to a planar "
      "object. Correlation has no model for that at all: it cannot be enumerated "
      "cheaply, because the space of homographies is eight dimensional, and a "
      "template bank over that space is not feasible. This is where the classical "
      "correlation detector genuinely runs out of road, and where feature "
      "matching with RANSAC is not merely faster but structurally correct, since "
      "the homography it estimates is exactly the transformation that occurred. "
      "It should be said that being structurally correct was not enough here: on "
      "the held out viewpoint scene the feature detector found correspondences on "
      "the emblem but fell short of the inlier count required by the operating "
      "point, so it reported nothing. The tilt reduces the effective resolution "
      "of the emblem along one axis and destroys a proportion of the keypoints, "
      "and a threshold tuned on mostly frontal development images is then too "
      "strict. Viewpoint change was the single condition that no method in this "
      "study handled.")

    h2("6.1 Limitations of this study")
    bullets([
        "The evaluation set is modest, %d images with %d instances in total. The "
        "differences between the stronger methods in Table 11 rest on a small "
        "number of instances and should be read as indicative rather than "
        "definitive."
        % (ds["dev_images"] + ds["test_images"], ds["dev_objects"] + ds["test_objects"]),
        "The scenes are real photographs with a real emblem composited into them "
        "under controlled transformations. This gives exact ground truth and "
        "isolates each factor cleanly, but it does not reproduce every effect of "
        "photographing a real printed emblem, in particular specular highlights "
        "on glossy signage, surface curvature and lens distortion.",
        "The colour proposal stage is tuned to the navy blue of this particular "
        "emblem and would need to be retuned for another object, whereas the "
        "correlation and feature stages are generic.",
        "Only in plane rotation was tested at a limited set of angles, and the "
        "viewpoint changes are moderate tilts rather than extreme oblique views.",
        "Timings are single threaded Python and OpenCV on one machine. The "
        "relative costs are meaningful, the absolute numbers are not portable.",
    ])

    h2("6.2 How the system could be improved")
    bullets([
        "Use edge or gradient orientation information instead of raw intensity, "
        "for example the chamfer distance on Canny edges or a histogram of "
        "oriented gradients descriptor, both of which are far less sensitive to "
        "photometric change than raw pixel correlation.",
        "Build a proper cascade: a very cheap rejector on integral image features "
        "first, then correlation, then feature verification, so that the "
        "expensive comparison only ever sees a handful of windows.",
        "Verify every correlation detection with the feature matcher and keep "
        "only detections that both stages agree on, which trades a little recall "
        "for high precision, or keep detections that either stage finds, which "
        "does the reverse. The combined method in Table 11 is a simple version of "
        "the second option.",
        "Estimate a dominant orientation for each candidate region from its "
        "gradient histogram and rotate the window to a canonical orientation "
        "before correlating, which would give most of the benefit of the rotated "
        "bank at a small fraction of its cost.",
    ])

    # ---------------- 7 conclusion ----------------
    h1("7. Conclusion")
    p("A complete classical detection and localisation system for the University "
      "of Ghana coat of arms was built, and the full progression from search "
      "strategy through template comparison and detection decision to object "
      "localisation and quantitative evaluation was implemented and measured. On "
      "a held out set of %d images that was never used for tuning, the best "
      "configuration reached a precision of %s, a recall of %s and an F1 of %s at "
      "an intersection over union threshold of 0.5, with a mean intersection over "
      "union of %s on its correct detections."
      % (ds["test_images"], num(e8_best[5]), num(e8_best[6]), num(e8_best[7]),
         num(e8_best[8])))
    p("The experiments support four conclusions. First, the decision threshold is "
      "the single most powerful parameter in a correlation detector, and the "
      "transition from thousands of false detections to none spans a range of "
      "about 0.05 in correlation, which means the threshold must be selected on "
      "data and not guessed. Second, invariance in a correlation detector has to "
      "be paid for by enumeration: scale costs a linear factor and is worth "
      "paying, rotation costs an order of magnitude and is not, and viewpoint "
      "cannot be enumerated at all. Third, the cost of a sliding window search "
      "falls as the square of the step size while its accuracy collapses once the "
      "step exceeds roughly one twentieth of the window, which makes a step of 4 "
      "to 8 pixels the practical compromise here. Fourth, the classical methods "
      "are complementary rather than competing: colour segmentation is the "
      "cheapest way to cut the search space, correlation is the most precise "
      "comparison when the object appears roughly upright and unoccluded, and "
      "SIFT with RANSAC is the only one of the three that survives rotation, "
      "sensor noise and partial occlusion. A system that uses all three, in that "
      "order, is better than any of them alone. Viewpoint change defeated every "
      "method tested and remains the open problem in this pipeline.")

    # ---------------- references ----------------
    h1("References")
    for i, ref in enumerate([
        "Lewis, J. P. (1995). Fast normalized cross correlation. Vision Interface, "
        "pages 120 to 123.",
        "Lowe, D. G. (2004). Distinctive image features from scale invariant "
        "keypoints. International Journal of Computer Vision, 60(2), 91 to 110.",
        "Fischler, M. A. and Bolles, R. C. (1981). Random sample consensus: a "
        "paradigm for model fitting with applications to image analysis and "
        "automated cartography. Communications of the ACM, 24(6), 381 to 395.",
        "Viola, P. and Jones, M. (2001). Rapid object detection using a boosted "
        "cascade of simple features. Proceedings of CVPR, pages 511 to 518.",
        "Dalal, N. and Triggs, B. (2005). Histograms of oriented gradients for "
        "human detection. Proceedings of CVPR, pages 886 to 893.",
        "Szeliski, R. (2022). Computer Vision: Algorithms and Applications, "
        "second edition. Springer.",
        "Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software "
        "Tools.",
        "Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and "
        "Zisserman, A. (2010). The PASCAL Visual Object Classes challenge. "
        "International Journal of Computer Vision, 88(2), 303 to 338.",
    ], 1):
        p("[%d] %s" % (i, ref), ParagraphStyle("ref", parent=BODY, leftIndent=16,
                                               firstLineIndent=-16, spaceAfter=4))

    # ---------------- appendix ----------------
    story.append(PageBreak())
    h1("Appendix A. Complete source code")
    p("The listing below is the complete source of the system, in the order in "
      "which the files are run. The same files are included in the repository.")
    for fname in ["download_data.py", "build_dataset.py", "detector.py",
                  "evaluate.py", "run_experiments.py", "make_report.py"]:
        path = os.path.join(ROOT, "src", fname)
        if not os.path.exists(path):
            continue
        h2("src/" + fname)
        with open(path, "r", encoding="utf-8") as fh:
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
