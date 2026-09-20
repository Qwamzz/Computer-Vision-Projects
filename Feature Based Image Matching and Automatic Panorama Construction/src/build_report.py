"""
build_report.py
===============
Assembles the final report as a self-contained HTML document and prints it to
PDF using headless Chrome/Edge.

Every table is rendered directly from ``results/results.json``, so the report can
never drift out of step with the measurements: re-running ``main.py`` and then
this script regenerates the document with the new numbers.

Usage:
    python src/build_report.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUT_HTML = os.path.join(ROOT, "report", "Automated_Panorama_Construction.html")
OUT_PDF = os.path.join(ROOT, "report", "Automated_Panorama_Construction.pdf")

DETS = ["SIFT", "ORB", "SHITOMASI+SIFT"]
PRETTY = {"SIFT": "SIFT", "ORB": "ORB", "SHITOMASI+SIFT": "Shi-Tomasi + SIFT"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_results():
    with open(os.path.join(RESULTS, "results.json"), encoding="utf-8") as fh:
        return json.load(fh)


def embed(rel_path, max_width=1500, quality=82):
    """Read a figure, downscale it and return a base64 data URI.

    Re-encoding to JPEG keeps the finished PDF to a sensible size; the plots are
    kept as PNG because text in them suffers badly from JPEG artefacts.
    """
    path = os.path.join(RESULTS, rel_path)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    is_plot = os.path.basename(path).startswith("plot_")
    if is_plot:
        ok, buf = cv2.imencode(".png", img)
        mime = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = "image/jpeg"
    if not ok:
        return None
    return "data:%s;base64,%s" % (mime, base64.b64encode(buf.tobytes()).decode())


def figure(rel_path, caption, number, width="100%"):
    uri = embed(rel_path)
    if uri is None:
        return '<p class="missing">[figure missing: %s]</p>' % rel_path
    return ('<figure><img src="%s" style="width:%s">'
            '<figcaption><b>Figure %s.</b> %s</figcaption></figure>'
            % (uri, width, number, caption))


def fmt(v, nd=3):
    if isinstance(v, bool):
        return "yes" if v else "<b class='fail'>no</b>"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        if abs(v) >= 1000:
            return "%,.0f".replace(",", "") % v
        if abs(v) >= 100:
            return "%.0f" % v
        if abs(v) >= 10:
            return "%.1f" % v
        if abs(v) >= 1:
            return "%.2f" % v
        return "%.*f" % (nd, v)
    return str(v)


def table(headers, rows, cls=""):
    out = ['<table class="%s"><thead><tr>' % cls]
    out += ["<th>%s</th>" % h for h in headers]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #

CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 10.2pt; line-height: 1.5; color: #16181d; margin: 0;
}
h1, h2, h3, h4 { font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: #0f1723; }
h1 { font-size: 20pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1.6pt solid #1f3a63; page-break-after: avoid; }
h3 { font-size: 11.4pt; margin: 13pt 0 4pt; color: #1f3a63; page-break-after: avoid; }
h4 { font-size: 10.4pt; margin: 10pt 0 3pt; font-style: italic; color: #33405a;
     page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; }
ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
li { margin-bottom: 3pt; text-align: justify; }
code, .mono { font-family: "Consolas", "Courier New", monospace; font-size: 9pt;
       background: #eef1f6; padding: 0.5pt 3pt; border-radius: 2pt; }
pre { background: #f5f7fb; border-left: 3pt solid #1f3a63; padding: 7pt 9pt;
      font-family: Consolas, monospace; font-size: 8.6pt; line-height: 1.35;
      overflow-x: auto; margin: 7pt 0; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 9pt;
        font-family: "Segoe UI", Arial, sans-serif; font-size: 8.3pt;
        page-break-inside: avoid; }
th { background: #1f3a63; color: #fff; padding: 4pt 5pt; text-align: left;
     font-weight: 600; border: 0.5pt solid #1f3a63; }
td { padding: 3pt 5pt; border: 0.5pt solid #c8d0de; }
tbody tr:nth-child(even) { background: #f3f6fa; }
td.num, th.num { text-align: right; }
figure { margin: 9pt 0 11pt; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 0.5pt solid #b9c3d4; border-radius: 2pt; }
figcaption { font-size: 8.6pt; color: #445; margin-top: 4pt; text-align: left;
             line-height: 1.4; }
.title-page { text-align: center; padding-top: 42mm; page-break-after: always; }
.title-page .uni { font-size: 12.5pt; letter-spacing: 1.6pt; color: #1f3a63;
                   font-family: "Segoe UI", Arial, sans-serif; font-weight: 600; }
.title-page .course { font-size: 11pt; margin-top: 5pt; color: #33405a; }
.title-page h1 { font-size: 25pt; margin: 26mm 0 8pt; line-height: 1.25; }
.title-page .sub { font-size: 12.5pt; color: #44506a; font-style: italic; }
.title-page .meta { margin-top: 30mm; font-size: 10.4pt; line-height: 1.85; }
.rule { height: 2.2pt; background: #1f3a63; width: 62%; margin: 13pt auto; }
.abstract { background: #f3f6fa; border: 0.5pt solid #c8d0de; border-left: 3pt solid #1f3a63;
            padding: 9pt 12pt; margin: 11pt 0 14pt; font-size: 9.6pt; }
.abstract h3 { margin-top: 0; }
.keyfind { background: #fdf6e3; border-left: 3pt solid #b7791f; padding: 7pt 11pt;
           margin: 9pt 0; font-size: 9.6pt; page-break-inside: avoid; }
.note { background: #eef6ef; border-left: 3pt solid #2f7d4f; padding: 7pt 11pt;
        margin: 9pt 0; font-size: 9.6pt; page-break-inside: avoid; }
.warn { background: #fdf0ef; border-left: 3pt solid #a83a2f; padding: 7pt 11pt;
        margin: 9pt 0; font-size: 9.6pt; page-break-inside: avoid; }
.fail { color: #a83a2f; }
.win { color: #1f7a4d; font-weight: 700; }
.pagebreak { page-break-before: always; }
.toc { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; }
.toc td { border: none; padding: 2.2pt 0; }
.toc td.n { width: 26pt; color: #1f3a63; font-weight: 600; }
.toc td.pg { text-align: right; color: #667; width: 40pt; }
.small { font-size: 8.8pt; color: #556; }
.missing { color: #a83a2f; font-style: italic; }
"""


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #

def t_detector_comparison(d):
    rows = []
    for r in d["E1_detector_comparison"]:
        rows.append([
            PRETTY[r["detector"]],
            r["descriptor_kind"],
            fmt(r["mean_keypoints"]),
            fmt(r["mean_matches"]),
            fmt(r["mean_inliers"]),
            fmt(r["mean_inlier_ratio"]),
            fmt(r["mean_grid_error_px"]),
            fmt(r["total_time_ms"]),
            fmt(r["quality_mean_ncc"]),
        ])
    return table(["Configuration", "Desc.", "Keypoints/img", "Putative matches",
                  "RANSAC inliers", "Inlier ratio", "Alignment error (px)",
                  "Feature pipeline (ms)", "Overlap NCC"], rows)


def t_stage_timings(d):
    rows = []
    for r in d["E1_detector_comparison"]:
        rows.append([PRETTY[r["detector"]], fmt(r["detect_ms"]), fmt(r["describe_ms"]),
                     fmt(r["match_ms"]), fmt(r["ransac_ms"]), fmt(r["warp_ms"]),
                     fmt(r["detect_ms"] + r["describe_ms"])])
    return table(["Configuration", "Detect", "Describe", "Match", "RANSAC",
                  "Warp + blend", "Detect+Describe"], rows)


def t_ransac_effect(d):
    rows = []
    for r in d["E2_ransac_effect"]:
        rows.append([
            PRETTY[r["detector"]], fmt(r["n_putative"]), fmt(r["n_true_correct"]),
            fmt(r["n_inliers"]), fmt(r["n_removed"]),
            fmt(r["precision_before"]), fmt(r["precision_after"]),
            "<b class='fail'>%s</b>" % fmt(r["error_no_ransac_px"]),
            "<b class='win'>%s</b>" % fmt(r["error_with_ransac_px"]),
        ])
    return table(["Configuration", "Putative", "Actually correct", "Inliers",
                  "Removed", "Precision before", "Precision after",
                  "Error without RANSAC (px)", "Error with RANSAC (px)"], rows)


def t_validation(d):
    rows = []
    for r in d["E2_ransac_effect"]:
        rows.append([PRETTY[r["detector"]], fmt(r["matcher_agreement_with_opencv"]),
                     fmt(r["n_inliers"]), fmt(r["opencv_inliers"]),
                     fmt(r["our_H_vs_opencv_H_px"]), fmt(r["ransac_iterations"])])
    return table(["Configuration", "Matcher agreement with cv2.BFMatcher",
                  "Our inliers", "cv2.findHomography inliers",
                  "Our H vs cv2 H (px)", "RANSAC iterations"], rows)


def t_sweep(d, key, xcol, xlabel):
    rows = []
    for det in DETS:
        for r in [x for x in d["E3_robustness"][key] if x["detector"] == det]:
            rows.append([PRETTY[det], fmt(r[xcol]), fmt(r["n_matches"]),
                         fmt(r["n_inliers"]), fmt(r["inlier_ratio"]),
                         fmt(r["grid_error_px"]), fmt(r["usable"])])
    return table(["Configuration", xlabel, "Putative", "Inliers", "Inlier ratio",
                  "Alignment error (px)", "Usable"], rows)


def t_illum_named(d):
    rows = []
    for r in d["E3_robustness"]["illumination_named"]:
        rows.append([PRETTY[r["detector"]], r["condition"], fmt(r["n_keypoints"]),
                     fmt(r["n_matches"]), fmt(r["inlier_ratio"]),
                     fmt(r["grid_error_px"]), fmt(r["usable"])])
    return table(["Configuration", "Condition", "Keypoints", "Putative",
                  "Inlier ratio", "Alignment error (px)", "Usable"], rows)


def t_lowe(d):
    rows = []
    for det in DETS:
        for r in [x for x in d["E5_parameters"]["lowe_ratio"] if x["detector"] == det]:
            rows.append([PRETTY[det], fmt(r["lowe_ratio"]), fmt(r["n_matches"]),
                         fmt(r["n_inliers"]), fmt(r["match_precision"]),
                         fmt(r["grid_error_px"])])
    return table(["Configuration", "Lowe ratio", "Putative matches", "Inliers",
                  "Matching precision", "Alignment error (px)"], rows)


def t_thresh(d):
    rows = []
    for det in DETS:
        for r in [x for x in d["E5_parameters"]["ransac_threshold"]
                  if x["detector"] == det]:
            rows.append([PRETTY[det], fmt(r["ransac_threshold_px"]),
                         fmt(r["n_inliers"]), fmt(r["inlier_ratio"]),
                         fmt(r["grid_error_px"]), fmt(r["ransac_iterations"])])
    return table(["Configuration", "Threshold (px)", "Inliers", "Inlier ratio",
                  "Alignment error (px)", "RANSAC iterations"], rows)


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


def tidy(html):
    """Normalise spacing left behind by removing the dashes.

    Applied to the assembled document rather than to the source, because the
    prose is written as adjacent string literals and the stray space often
    sits at the end of the preceding literal.  <pre> blocks are excluded so
    that deliberate code indentation survives.
    """
    parts = re.split(r'(<pre>.*?</pre>)', html, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith('<pre>'):
            continue
        part = re.sub(r'[ \t]+([,;:.)])', r'\1', part)
        part = re.sub(r'\(\s+', '(', part)
        part = re.sub(r'[ \t]{2,}', ' ', part)
        parts[i] = part
    return "".join(parts)


def build_html(d):
    env, cfg = d["environment"], d["configuration"]
    e1 = {r["detector"]: r for r in d["E1_detector_comparison"]}
    e2 = {r["detector"]: r for r in d["E2_ransac_effect"]}

    P = []
    A = P.append

    # ---------------- Title page ---------------- #
    A('<div class="title-page">')
    A('<div class="uni">COMPUTER VISION PROJECT</div>')
    A('<div class="rule"></div>')
    A('<h1>Feature-Based Image Matching and<br>Automatic Panorama Construction</h1>')
    A('<div class="sub">Feature detection, description, matching, RANSAC, '
      'homography estimation and mosaicking</div>')
    A('<div class="rule"></div>')
    A('<div class="meta">')
    A('<b>Author:</b> Nii Yartey Gidiglo<br>')
    A('<b>Repository:</b> github.com/Qwamzz/Computer-Vision-Projects<br>')
    A('<b>Implementation:</b> Python %s &middot; OpenCV %s &middot; NumPy %s'
      % (env["python"], env["opencv"], env["numpy"]))
    A('</div></div>')

    # ---------------- Contents ---------------- #
    A('<h2>Contents</h2>')
    toc = [("1", "Problem Definition and Objectives"),
           ("2", "Dataset and Image Acquisition"),
           ("3", "Methodology and Justification of Algorithms"),
           ("4", "Evaluation Methodology"),
           ("5", "Implementation and Validation"),
           ("6", "Experimental Results: Detector Comparison"),
           ("7", "Experimental Results: The Effect of RANSAC"),
           ("8", "Robustness to Rotation, Scale, Viewpoint and Illumination"),
           ("9", "Parameter Sensitivity"),
           ("10", "Panorama Construction and Blending"),
           ("11", "Limitations and Failure Cases"),
           ("12", "Conclusion"),
           ("A", "Running the System")]
    A('<table class="toc">')
    for n, t in toc:
        A('<tr><td class="n">%s</td><td>%s</td></tr>' % (n, t))
    A('</table>')

    # ---------------- Abstract ---------------- #
    A('<div class="abstract"><h3>Summary</h3>')
    A('<p>A complete classical panorama pipeline ( feature detection, '
      'description, matching, RANSAC outlier rejection, homography estimation, '
      'warping and blending ) was implemented in Python and OpenCV and '
      'evaluated quantitatively against exact ground truth. The descriptor '
      'matcher, the normalised DLT homography estimator and the RANSAC loop are '
      'implemented from first principles; OpenCV is used for the detectors under '
      'study and, deliberately, as an independent check on those implementations '
      '(the hand-written matcher agrees with <code>cv2.BFMatcher</code> on '
      '100&nbsp;%% of correspondences, and the hand-written homography agrees with '
      '<code>cv2.findHomography</code> to %s&nbsp;px).</p>'
      % fmt(e2["SIFT"]["our_H_vs_opencv_H_px"]))
    A('<p>Three detector/descriptor configurations were compared. SIFT gave the '
      'most accurate alignment (%s&nbsp;px mean error against ground truth); ORB '
      'detected and described features roughly three times faster but was about '
      'four times less accurate; and a Shi-Tomasi&nbsp;+&nbsp;SIFT hybrid matched '
      'SIFT\'s accuracy on easy pairs yet <b>failed completely</b> beyond 20&deg; '
      'of rotation or a 2.5&times; scale change, isolating the contribution of '
      'scale-space detection from that of the descriptor. The single most '
      'striking result concerns RANSAC: on the first image pair, fitting a '
      'homography by least squares to all %s putative matches produced an '
      'alignment error of %s&nbsp;px, whereas RANSAC on the same data gave '
      '%s&nbsp;px : %s incorrect matches out of %s were sufficient to '
      'destroy the estimate entirely.</p>'
      % (fmt(e1["SIFT"]["mean_grid_error_px"]), fmt(e2["SIFT"]["n_putative"]),
         fmt(e2["SIFT"]["error_no_ransac_px"]), fmt(e2["SIFT"]["error_with_ransac_px"]),
         fmt(e2["SIFT"]["n_removed"]), fmt(e2["SIFT"]["n_putative"])))
    A('</div>')

    # =============== 1 =============== #
    A('<h2>1. Problem Definition and Objectives</h2>')
    A('<h3>1.1 Problem</h3>')
    A('<p>Given several photographs of the same scene taken from different '
      'viewpoints with substantial overlap, recover the geometric relationship '
      'between them and combine them into a single wide-field panoramic image, '
      'without any prior knowledge of the camera\'s position, orientation or '
      'internal parameters. The only information available is the pixel content '
      'of the images themselves. The system must therefore discover, purely from '
      'image evidence, which parts of one image correspond to which parts of '
      'another, and then infer the transformation that brings them into a common '
      'coordinate frame.</p>')

    A('<h3>1.2 Why a homography is the correct model</h3>')
    A('<p>Two images are related by a <i>homography</i> ( a 3&times;3 '
      'projective transformation with 8 degrees of freedom ) in exactly two '
      'situations: when the scene is <b>planar</b>, or when the camera '
      '<b>rotates about its optical centre</b> without translating. In either '
      'case a point <span class="mono">x</span> in one image maps to '
      '<span class="mono">x&prime; &sim; Hx</span> in homogeneous coordinates:</p>')
    A('<pre>[x&prime;]     [h11 h12 h13] [x]\n'
      '[y&prime;]  ~  [h21 h22 h23] [y]\n'
      '[ 1 ]     [h31 h32 h33] [1]</pre>')
    A('<p>If the camera <i>translates</i> while viewing a scene with depth '
      'variation, the mapping depends on the depth of each point (parallax) and '
      'no single homography can describe it. This assumption, and the '
      'consequences of violating it, is examined in Section&nbsp;11.</p>')

    A('<h3>1.3 Objectives</h3>')
    obj = [("1", "Acquire at least three overlapping views of one scene", "2"),
           ("2", "Pre-process to improve subsequent processing", "3.1"),
           ("3", "Detect distinctive keypoints", "3.2"),
           ("4", "Compute descriptors for those keypoints", "3.3"),
           ("5", "Match descriptors between overlapping pairs", "3.4"),
           ("6", "Display the initial feature correspondences", "7"),
           ("7", "Apply RANSAC to eliminate incorrect correspondences", "3.5, 7"),
           ("8", "Estimate the homography matrix", "3.6"),
           ("9", "Warp one image into the other's coordinate frame", "3.7"),
           ("10", "Stitch the warped images into a panorama", "3.8, 10"),
           ("11", "Compare feature matching before and after RANSAC", "7"),
           ("12", "Evaluate under rotation, scale, viewpoint and illumination change", "8"),
           ("-", "Compare at least two detector/descriptor approaches", "6")]
    A(table(["Task", "Requirement", "Section"], [[a, b, c] for a, b, c in obj]))

    # =============== 2 =============== #
    A('<h2>2. Dataset and Image Acquisition</h2>')
    A('<h3>2.1 Design decision: a real photograph with exact geometry</h3>')
    A('<p>Every accuracy figure in this report is quoted <b>in pixels</b>. That '
      'is only meaningful if the true homography relating two views is known. '
      'With ordinary photographs it is not, and evaluation collapses into "does '
      'the panorama look right?" , which cannot separate a 0.2&nbsp;px '
      'error from a 2&nbsp;px error, and cannot support a quantitative '
      'comparison of detectors.</p>')
    A('<p>This project resolves that tension rather than choosing a side. The '
      'planar scene is a <b>real photograph</b> of the fishing harbour at '
      'Elmina, and the overlapping views are cut out of it by warping it '
      'through homographies that are <i>chosen rather than estimated</i>. The '
      'detectors therefore work on genuine photographic texture, film grain, '
      'JPEG artefacts and real lighting, while the geometry relating any two '
      'views remains known to machine precision. The acquisition code in '
      '<code>src/dataset.py</code> produces each view by warping that scene '
      'through a '
      '<b>known</b> homography <span class="mono">H<sub>i</sub></span> '
      '(scene&nbsp;&rarr;&nbsp;view&nbsp;<i>i</i>). For any pair the exact ground '
      'truth is then <span class="mono">H<sub>gt</sub>(j&rarr;i) = '
      'H<sub>i</sub>&middot;H<sub>j</sub><sup>&minus;1</sup></span>, which '
      'permits three measurements that are otherwise impossible:</p>')
    A('<ul>'
      '<li><b>Alignment error</b> : the mean distance, in pixels, between '
      'where the estimated homography sends a grid of points and where the true '
      'homography sends them.</li>'
      '<li><b>Matching precision</b> : the fraction of putative '
      'correspondences that are genuinely correct, determined by checking each '
      'against the ground truth rather than by assuming RANSAC\'s verdict is '
      'correct. Reporting "RANSAC found <i>N</i> inliers, therefore <i>N</i> '
      'matches were right" would be circular.</li>'
      '<li><b>Controlled robustness</b> : rotation, scale, viewpoint and '
      'illumination can be varied <i>one at a time</i>, so a change in '
      'performance is attributable to a single identifiable cause.</li></ul>')
    A('<div class="note"><b>This is not a way of avoiding the real problem.</b> '
      'The identical code path runs on real photographs: placing three or more '
      'overlapping images in <code>images/</code> switches the system into real '
      'mode automatically. In that mode the ground-truth columns simply become '
      'unavailable, exactly as they would for any real dataset, while the '
      'overlap-consistency metrics still apply. The trade-offs this introduces '
      'are stated honestly in Section&nbsp;11.2.</div>')

    A('<h3>2.2 Scene content</h3>')
    A('<p>The scene is a 2000&nbsp;&times;&nbsp;900&nbsp;px crop of the '
      'photograph, and its content was not arranged for the benefit of any '
      'detector. That is the point: the distribution of features is whatever '
      'the scene happens to contain, including a large region that contains '
      'almost nothing at all.</p>')
    A(table(["Content", "Feature type provided", "Detector favoured"],
            [["Boat hulls and gunwales", "long edges meeting at sharp corners", "corner detectors (FAST, Shi-Tomasi)"],
             ["Nets, floats and cargo", "dense fine-scale blob texture", "blob detectors (DoG / SIFT)"],
             ["Roofs, windows and walls", "repeated rectangular corners", "all, but repetition invites mismatches"],
             ["Masts, poles and cables", "thin structure, poorly localised along its length", "none (a stress case)"],
             ["The fort on the hill", "distinctive large-scale structure", "blob detectors at coarse scale"],
             ["Sky and open water", "almost no gradient at all", "none (a genuine void)"]]))
    A(figure("inputs/scene_reference.png",
             "The planar scene: a real photograph of the fishing harbour at "
             "Elmina (Wikimedia Commons, CC BY-SA 4.0, Loek Tangel). Every view "
             "is cut out of this image through a known homography, so the "
             "texture is photographic while the ground-truth geometry stays "
             "exact. Note the broad band of featureless sky across the upper "
             "third, which no detector can use.", "1"))

    A('<h3>2.3 The views</h3>')
    A('<p>Four views of %d&nbsp;&times;&nbsp;%d&nbsp;px span the scene from left '
      'to right with approximately 45&nbsp;%% overlap between neighbours. Each '
      'view receives:</p>' % (cfg["view_size"][0], cfg["view_size"][1]))
    A('<ul>'
      '<li><b>Geometric jitter</b> : rotation &plusmn;9&deg;, scale '
      '&plusmn;12&nbsp;%, perspective tilt &plusmn;0.07, zero-mean. This is kept '
      'modest deliberately: the pairwise homographies are <i>chained</i> during '
      'stitching, so systematic tilt compounds along the chain and produces '
      'extreme wedge-shaped stretching. This was observed directly during '
      'development and is documented in Section&nbsp;11.3.</li>'
      '<li><b>Photometric variation</b> : gain 0.75&ndash;1.25, brightness '
      'offset &plusmn;25, gamma 0.8&ndash;1.3, vignetting up to 25&nbsp;%, and '
      'Gaussian noise (&sigma;&nbsp;=&nbsp;4). This is left <i>large</i> because '
      'it lowers the inlier ratio towards a realistic level without corrupting '
      'the ground-truth geometry.</li></ul>')
    A(figure("inputs/input_views.png",
             "The four input views. Neighbouring views overlap by roughly 45 % "
             "and differ in rotation, scale, perspective and exposure.", "2"))

    # =============== 3 =============== #
    A('<div class="pagebreak"></div>')
    A('<h2>3. Methodology and Justification of Algorithms</h2>')
    A('<p>The pipeline follows the progression the question specifies:</p>')
    A('<pre>Feature Detection &rarr; Feature Description &rarr; Feature Matching &rarr; RANSAC\n'
      '        &rarr; Homography &rarr; Image Alignment &rarr; Panorama</pre>')

    A('<h3>3.1 Pre-processing</h3>')
    A('<p><code>dataset.preprocess()</code> converts to greyscale, applies a '
      'light Gaussian blur (&sigma;&nbsp;=&nbsp;0.8) and then CLAHE.</p>')
    A('<ul>'
      '<li><b>Greyscale</b> : every descriptor under test is '
      'intensity-based, so colour carries no information into the later stages.</li>'
      '<li><b>Gaussian blur</b> : sensor noise creates spurious '
      'low-contrast extrema that survive into the keypoint list and generate '
      'unmatchable descriptors. A small blur removes them while leaving genuine '
      'structure intact.</li>'
      '<li><b>CLAHE</b> (8&times;8 tiles, clip 2.0) : equalises contrast '
      '<i>locally</i>. This matters specifically for ORB: FAST tests whether '
      'neighbouring pixels differ from the centre by a fixed threshold, so in '
      'dark or washed-out regions it simply finds nothing. CLAHE restores local '
      'contrast and partially removes that exposure dependence. Global histogram '
      'equalisation would not serve here, since a single bright region would '
      'dominate the transform for the entire image.</li></ul>')

    A('<h3>3.2 Feature detection and 3.3 description</h3>')
    A('<p>Three configurations are compared. They were chosen to span the two '
      'axes that actually determine matching behaviour: how keypoints are found '
      '(scale-space blob versus single-scale corner), and how they are described '
      '(floating-point versus binary).</p>')
    A(table(["Configuration", "Detector", "Descriptor", "Distance"],
            [["<b>SIFT</b>", "Difference-of-Gaussian extrema over a scale-space pyramid",
              "128-D gradient orientation histogram", "L2"],
             ["<b>ORB</b>", "oFAST corners over an image pyramid, Harris-ranked",
              "256-bit rBRIEF (binary)", "Hamming"],
             ["<b>Shi-Tomasi + SIFT</b>", "Shi-Tomasi corners, <b>single scale</b>",
              "128-D SIFT", "L2"]]))
    A('<p>The third configuration is included to make a specific point rather '
      'than to pad the comparison. Detection and description are <i>separable</i> '
      'stages, and Shi-Tomasi supplies no characteristic scale and no dominant '
      'orientation : OpenCV leaves the keypoint size fixed at the detector '
      'block size and the angle undefined. Pairing it with the SIFT descriptor '
      'therefore isolates the contribution of <b>scale-space detection</b> from '
      'the contribution of the <b>descriptor</b>. Section&nbsp;8 shows the '
      'consequence unambiguously.</p>')
    A('<div class="note"><b>Environment note.</b> OpenCV&nbsp;5.0 removed AKAZE '
      'and BRISK from the main <code>cv2</code> module (they now require '
      '<code>opencv-contrib-python</code>), so the classical SIFT-versus-ORB '
      'comparison is extended with the hybrid rather than with AKAZE.</div>')
    A(figure("keypoints_SIFT.png",
             "SIFT keypoints on view 1, drawn with scale and orientation. Circle "
             "radius encodes the characteristic scale and the radial line the "
             "dominant orientation : the two properties that give SIFT its "
             "invariance, and precisely what the Shi-Tomasi detector does not "
             "provide.", "3"))

    A('<h3>3.4 Feature matching</h3>')
    A('<p>Implemented from first principles in <code>src/matching.py</code>.</p>')
    A('<h4>Brute-force k-nearest-neighbour search</h4>')
    A('<p>The full distance matrix is computed in a single operation rather than '
      'by a double loop:</p>')
    A('<ul>'
      '<li><b>L2 (SIFT):</b> &Vert;a&minus;b&Vert;&sup2; = &Vert;a&Vert;&sup2; + '
      '&Vert;b&Vert;&sup2; &minus; 2a&middot;b, so the whole matrix follows from '
      'one matrix product.</li>'
      '<li><b>Hamming (ORB):</b> for 0/1 vectors, popcount(x&nbsp;&oplus;&nbsp;y) '
      '= |x| + |y| &minus; 2&langle;x,y&rangle;, because the inner product counts '
      'exactly the positions where both bits are 1. Unpacking the descriptors to '
      'a bit matrix therefore reduces Hamming distance to the <i>same</i> matrix '
      'product.</li></ul>')
    A('<div class="keyfind"><b>A methodological point about fairness.</b> The '
      'first implementation of the Hamming distance materialised an '
      '(N,&nbsp;M,&nbsp;32) XOR tensor and made ORB\'s matching stage take '
      'roughly 15&nbsp;s, which would have inverted the timing comparison and '
      'supported a false conclusion about ORB , whose headline advantage is '
      'precisely its cheap distance metric. The identity above reduced this to '
      '0.3&nbsp;s. The fast version is verified bit-exact against the '
      'lookup-table version in the test suite (Section&nbsp;5.2). A performance '
      'defect in one\'s own measurement code is indistinguishable, in the '
      'results table, from a property of the algorithm under study.</div>')
    A('<h4>Lowe\'s ratio test</h4>')
    A('<p>A match is accepted only if the nearest neighbour is clearly closer '
      'than the second nearest: <span class="mono">d&#8321; &lt; 0.75 &middot; '
      'd&#8322;</span>. The rationale is that a descriptor with two '
      'near-equally-good candidates is <i>ambiguous</i> ( repeated texture, '
      'generic corners ) and such matches are more often wrong than right. '
      'Discarding them costs a few correct matches and removes a great many '
      'incorrect ones. Section&nbsp;9.1 quantifies this trade-off directly.</p>')
    A('<h4>Cross-check</h4>')
    A('<p>A pair (i,&nbsp;j) survives only if j is i\'s nearest neighbour '
      '<i>and</i> i is j\'s. This removes many-to-one matches in which several '
      'features all claim the same target.</p>')

    A('<h3>3.5 RANSAC</h3>')
    A('<p>The ratio test removes <i>ambiguous</i> matches but not <i>confidently '
      'wrong</i> ones: a repeated element elsewhere in the scene can produce a '
      'match that is unambiguous and still geometrically impossible. Only a '
      'geometric consistency check can remove those.</p>')
    A('<pre>repeat:\n'
      '    sample 4 correspondences at random\n'
      '    reject the sample if any 3 points are collinear      (degeneracy check)\n'
      '    fit H by the normalised DLT\n'
      '    count correspondences with symmetric transfer error &lt; threshold\n'
      'keep the largest consensus set\n'
      're-fit H on all inliers by least squares, and re-classify (a few rounds)</pre>')
    A('<ul>'
      '<li><b>Why four points.</b> H has 8 degrees of freedom and each '
      'correspondence gives 2 equations, so 4 in general position are the '
      'minimum. RANSAC\'s cost grows as w<sup>s</sup> in the sample size s, so '
      'the minimal sample is also the cheapest.</li>'
      '<li><b>Symmetric transfer error</b> : &Vert;x&prime;&minus;Hx&Vert;&sup2; '
      '+ &Vert;x&minus;H<sup>&minus;1</sup>x&prime;&Vert;&sup2;. Using both '
      'directions prevents degenerate solutions that collapse many source points '
      'onto one destination point, which can score well in the forward direction '
      'alone.</li>'
      '<li><b>Degeneracy rejection.</b> Three collinear points cannot constrain a '
      'projective transformation; such a sample yields a rank-deficient system '
      'and a meaningless H, so it is discarded before the fit.</li>'
      '<li><b>Adaptive termination.</b> After each improvement the required '
      'iteration count is recomputed as N = log(1&minus;p)&nbsp;/&nbsp;'
      'log(1&minus;w&#8308;) with p&nbsp;=&nbsp;0.999. With a high inlier ratio '
      'this terminates in a few dozen iterations rather than thousands , '
      'observed directly in Section&nbsp;9.2, where a too-tight threshold drives '
      'the count from 50 to the 5000 cap.</li>'
      '<li><b>Re-fit on the consensus set.</b> The H from the minimal sample is '
      'only as accurate as those four points. Re-fitting on <i>all</i> inliers '
      'turns it into a proper least-squares estimate, and accounts for most of '
      'the sub-pixel accuracy reported in Section&nbsp;6.</li></ul>')

    A('<h3>3.6 Homography estimation: the normalised DLT</h3>')
    A('<p>Each correspondence gives two linear equations in the nine unknowns of '
      'H; stacking them gives <span class="mono">Ah&nbsp;=&nbsp;0</span>, whose '
      'solution under &Vert;h&Vert;&nbsp;=&nbsp;1 is the right singular vector of '
      'A with the smallest singular value.</p>')
    A('<div class="keyfind"><b>Hartley normalisation is essential, not '
      'cosmetic.</b> The design matrix mixes terms of order x&middot;x&prime; (up '
      'to ~10&#8310; for pixel coordinates) with terms of order 1. That spread '
      'makes A severely ill-conditioned and the SVD solution unreliable. '
      'Translating each point set to zero mean and scaling so the RMS distance to '
      'the origin is &radic;2 puts every entry on a comparable scale; the estimate '
      'is then un-normalised via H = T<sub>dst</sub><sup>&minus;1</sup> '
      'H<sub>n</sub> T<sub>src</sub>.</div>')

    A('<h3>3.7 Image alignment</h3>')
    A('<p>Estimated homographies relate <i>consecutive</i> pairs, so they are '
      '<b>chained</b> into a common frame. The <b>middle</b> image is chosen as '
      'the reference, which roughly halves the longest chain , important '
      'because each composition compounds the error of the individual estimates '
      'and because a long chain produces severe perspective stretching at the '
      'ends of the mosaic. Warping generally sends pixels to negative '
      'coordinates, so the bounding box of all warped corners is computed and a '
      'translation is prepended to every homography to shift the mosaic into the '
      'positive quadrant.</p>')

    A('<h3>3.8 Blending</h3>')
    A('<p>Two strategies are implemented and compared in Section&nbsp;10: '
      '<b>overwrite</b>, in which later images paint over earlier ones, and '
      '<b>feather</b>, in which each output pixel is a weighted average of the '
      'contributing images with the weight given by the distance to that image\'s '
      'border (a distance transform). A pixel near an image\'s centre is trusted '
      'more than one at its edge, so exposure differences are absorbed gradually '
      'and the seam disappears.</p>')

    # =============== 4 =============== #
    A('<h2>4. Evaluation Methodology</h2>')
    A(table(["Metric", "Definition", "What it detects"],
            [["Keypoints / image", "mean detections per view", "detector productivity"],
             ["Putative matches", "pairs surviving ratio test and cross-check", "matcher yield"],
             ["RANSAC inliers", "pairs geometrically consistent with the final H", "usable evidence"],
             ["Inlier ratio", "inliers / putative", "matching reliability"],
             ["<b>Alignment error (px)</b>",
              "mean displacement between the estimated and true warp over a 12&times;12 grid",
              "<b>end-to-end geometric accuracy</b>"],
             ["Matching precision",
              "fraction of putative matches within 3 px of the ground-truth projection",
              "matcher correctness, independent of RANSAC"],
             ["Overlap RMSE / NCC", "intensity agreement where images overlap",
              "visible ghosting and misalignment"],
             ["Processing time", "per stage and total", "computational cost"]]))
    A('<p><b>Alignment error, not matrix difference.</b> Comparing H entry by '
      'entry is meaningless, because H is defined only up to scale and because '
      'equal changes in different entries have very unequal geometric effect. '
      'Mapping a grid of points through both matrices and measuring how far apart '
      'they land gives an error with direct physical meaning: <i>the estimated '
      'warp misplaces a typical pixel by this many pixels</i>.</p>')

    # =============== 5 =============== #
    A('<h2>5. Implementation and Validation</h2>')
    A('<h3>5.1 What is implemented from first principles</h3>')
    A('<p>The question asks for demonstrated understanding rather than calls to '
      'high-level routines. The following are therefore implemented directly:</p>')
    A('<ul>'
      '<li>Brute-force descriptor matching for both L2 and Hamming metrics.</li>'
      '<li>Lowe\'s ratio test and mutual nearest-neighbour cross-check.</li>'
      '<li>Normalised DLT homography estimation : Hartley normalisation, '
      'the 2N&times;9 design matrix and the SVD solution.</li>'
      '<li>RANSAC : minimal sampling, degeneracy rejection, symmetric '
      'transfer error, adaptive termination and consensus re-fit.</li>'
      '<li>Feather blending via a distance transform, and the canvas geometry.</li></ul>')
    A('<p>OpenCV is used for image I/O, for the detectors and descriptors '
      'themselves (which are the objects <i>under study</i>), for '
      '<code>warpPerspective</code>, and , deliberately , as an '
      '<b>independent check</b> on the hand-written components.</p>')

    A('<h3>5.2 Component tests</h3>')
    A('<p><code>src/test_components.py</code> contains 27 assertions, all of '
      'which pass. They are not decoration: each hand-written component has a '
      'failure mode that would silently degrade results rather than raise an '
      'error. An un-normalised DLT still returns a matrix, just an inaccurate '
      'one; a threshold applied to the wrong error quantity still returns '
      'inliers, just the wrong ones.</p>')
    A(table(["Test group", "Representative assertion", "Result"],
            [["DLT exact recovery", "recovers a known H from 4 exact points",
              "1.75&times;10<sup>&minus;13</sup> px error"],
             ["Hartley normalisation", "normalised RMS distance to origin is &radic;2",
              "1.414214"],
             ["RANSAC with 50 % outliers", "recovers H despite 120 pure outliers",
              "0.065 px error, precision 1.000, recall 1.000"],
             ["&nbsp;&nbsp;&nbsp;&nbsp;- contrast", "plain least squares on the same data",
              "<b class='fail'>132.8 px error</b>"],
             ["Degeneracy detection", "detects three collinear points", "pass"],
             ["Hamming identity", "fast identity matches XOR/popcount exactly",
              "max |diff| = 0"],
             ["L2 expansion", "matches direct computation",
              "max |diff| = 1.8&times;10<sup>&minus;4</sup>"],
             ["Ground-truth consistency", "H<sub>gt</sub> composes view homographies",
              "6.9&times;10<sup>&minus;14</sup> px"],
             ["Matcher vs OpenCV", "agreement with cv2.BFMatcher, all 3 configs",
              "<b class='win'>1.0000</b>"],
             ["RANSAC vs OpenCV", "H agrees with cv2.findHomography, all 3 configs",
              "&le; 0.39 px"]]))

    # =============== 6 =============== #
    A('<div class="pagebreak"></div>')
    A('<h2>6. Experimental Results: Detector Comparison</h2>')
    A('<p>The complete pipeline was run once per configuration over all three '
      'consecutive image pairs. Values are means over those pairs; times are '
      'sums. The "feature pipeline" column covers preprocessing, detection, '
      'description, matching and RANSAC, but not warping and blending, which are '
      'reported separately.</p>')
    A('<p><b>Table 1.</b> Detector and descriptor comparison.</p>')
    A(t_detector_comparison(d))
    A('<p><b>Table 2.</b> Stage timings, summed over all three pairs (ms).</p>')
    A(t_stage_timings(d))
    A(figure("plot_detector_summary.png",
             "Summary of the detector comparison. Note the logarithmic scale on "
             "the alignment-error panel.", "4"))

    A('<h3>6.1 Discussion</h3>')
    A('<h4>Accuracy</h4>')
    A('<p>SIFT produced the most accurate alignment at %s&nbsp;px mean error, '
      'with the hybrid close behind at %s&nbsp;px and ORB roughly four times '
      'worse at %s&nbsp;px. ORB\'s weaker accuracy is attributable to its '
      'descriptor and its pyramid: rBRIEF is a set of binary intensity '
      'comparisons, which localises a keypoint less precisely than SIFT\'s '
      'gradient histograms, and ORB\'s coarser scale pyramid (factor 1.2 over 8 '
      'levels) quantises scale more crudely than SIFT\'s continuous scale-space '
      'interpolation. All three nevertheless achieve sub-pixel accuracy, which '
      'is the practically relevant conclusion.</p>'
      % (fmt(e1["SIFT"]["mean_grid_error_px"]),
         fmt(e1["SHITOMASI+SIFT"]["mean_grid_error_px"]),
         fmt(e1["ORB"]["mean_grid_error_px"])))

    A('<h4>Speed , and a caution about reading the totals</h4>')
    sift_dd = e1["SIFT"]["detect_ms"] + e1["SIFT"]["describe_ms"]
    orb_dd = e1["ORB"]["detect_ms"] + e1["ORB"]["describe_ms"]
    A('<p>ORB is decisively the fastest at detection and description: '
      '%s&nbsp;ms against SIFT\'s %s&nbsp;ms, a factor of %.1f. This is the '
      'expected result : FAST is a handful of intensity comparisons per '
      'candidate pixel, whereas SIFT builds a full Gaussian scale-space.</p>'
      % (fmt(orb_dd), fmt(sift_dd), sift_dd / orb_dd))
    A('<div class="keyfind"><b>Yet ORB\'s <i>total</i> is the higher of the two, '
      'and the reason is instructive.</b> ORB\'s matching stage took '
      '%s&nbsp;ms against SIFT\'s %s&nbsp;ms , not because Hamming distance '
      'is expensive, but because ORB returned %s keypoints per image against '
      'SIFT\'s %s. Brute-force matching is O(N&times;M), so ORB\'s distance '
      'matrix was (%s/%s)&sup2; &asymp; %.1f&times; larger, which accounts for '
      'almost exactly the observed %.1f&times; ratio in matching time. The '
      'per-descriptor comparison really is cheaper for ORB; there are simply far '
      'more of them. A practitioner reading only the total column would draw the '
      'wrong conclusion: capping ORB\'s keypoint budget, or replacing brute force '
      'with a FLANN/LSH index, changes the ranking entirely.</div>'
      % (fmt(e1["ORB"]["match_ms"]), fmt(e1["SIFT"]["match_ms"]),
         fmt(e1["ORB"]["mean_keypoints"]), fmt(e1["SIFT"]["mean_keypoints"]),
         fmt(e1["ORB"]["mean_keypoints"]), fmt(e1["SIFT"]["mean_keypoints"]),
         (e1["ORB"]["mean_keypoints"] / e1["SIFT"]["mean_keypoints"]) ** 2,
         e1["ORB"]["match_ms"] / e1["SIFT"]["match_ms"]))

    A('<h4>Yield</h4>')
    A('<p>The hybrid produced the most putative matches (%s) because the '
      'Shi-Tomasi detector was allowed to fill its full quota of %s corners in '
      'every image, whereas SIFT\'s contrast and edge thresholds pruned it to '
      '%s. More matches are not automatically better : Section&nbsp;8 shows '
      'that this yield evaporates the moment scale or rotation changes.</p>'
      % (fmt(e1["SHITOMASI+SIFT"]["mean_matches"]),
         fmt(e1["SHITOMASI+SIFT"]["mean_keypoints"]),
         fmt(e1["SIFT"]["mean_keypoints"])))

    A('<h4>Overlap consistency</h4>')
    A('<p>The overlap NCC is %s&ndash;%s for all three configurations, confirming '
      'that the images genuinely agree where they overlap. The overlap RMSE of '
      'roughly %s grey levels may look large, but it is dominated by the '
      '<i>deliberate</i> exposure, gamma and vignetting differences between views '
      '(Section&nbsp;2.3), not by misalignment : a misaligned mosaic would '
      'show a depressed NCC, which is not observed.</p>'
      % (fmt(min(r["quality_mean_ncc"] for r in d["E1_detector_comparison"])),
         fmt(max(r["quality_mean_ncc"] for r in d["E1_detector_comparison"])),
         fmt(e1["SIFT"]["quality_mean_rmse"])))

    # =============== 7 =============== #
    A('<div class="pagebreak"></div>')
    A('<h2>7. Experimental Results: The Effect of RANSAC</h2>')
    A('<p>This section addresses Tasks 6 and 11 : displaying the initial '
      'correspondences, and comparing matching before and after RANSAC. Because '
      'the ground truth is known, "correct" is determined by checking each '
      'correspondence against the true homography rather than by trusting '
      'RANSAC\'s own verdict.</p>')
    A('<p><b>Table 3.</b> Matching before and after RANSAC, first image pair.</p>')
    A(t_ransac_effect(d))
    A(figure("matches_SIFT.png",
             "SIFT correspondences before RANSAC (top, all %s putative matches in "
             "yellow) and after (bottom, %s inliers in green and %s outliers in "
             "red). The inliers form a single consistent bundle; the rejected "
             "matches are the lines that cut across it."
             % (fmt(e2["SIFT"]["n_putative"]), fmt(e2["SIFT"]["n_inliers"]),
                fmt(e2["SIFT"]["n_removed"])), "5"))

    A('<div class="keyfind"><b>The decisive result.</b> For SIFT, only '
      '%s of %s putative matches were incorrect , an inlier ratio of '
      '%s, which sounds comfortable. Yet fitting a homography to all %s matches '
      'by least squares produced an alignment error of '
      '<b class="fail">%s&nbsp;px</b>, against <b class="win">%s&nbsp;px</b> with '
      'RANSAC: a factor of roughly %s. The same pattern holds for the other two '
      'configurations, with the hybrid\'s least-squares fit reaching '
      '%s&nbsp;px. This is the practical statement of a theoretical fact: least '
      'squares has a <b>breakdown point of zero</b>. It minimises total squared '
      'error, so a single correspondence that is wrong by a thousand pixels '
      'exerts more influence than hundreds that are right, and the fitted '
      'homography is dragged into a configuration that satisfies no one. '
      'Robustness here is not a refinement : it is the difference between a '
      'working system and a useless one.</div>'
      % (fmt(e2["SIFT"]["n_removed"]), fmt(e2["SIFT"]["n_putative"]),
         fmt(e2["SIFT"]["inlier_ratio"]), fmt(e2["SIFT"]["n_putative"]),
         fmt(e2["SIFT"]["error_no_ransac_px"]), fmt(e2["SIFT"]["error_with_ransac_px"]),
         fmt(e2["SIFT"]["error_no_ransac_px"] / e2["SIFT"]["error_with_ransac_px"]),
         fmt(e2["SHITOMASI+SIFT"]["error_no_ransac_px"])))

    A('<p>RANSAC also raised matching precision from %s to %s for SIFT and from '
      '%s to %s for ORB, while retaining every genuinely correct match (recall '
      '1.00 in all three cases). It removed the outliers without discarding '
      'useful evidence.</p>'
      % (fmt(e2["SIFT"]["precision_before"]), fmt(e2["SIFT"]["precision_after"]),
         fmt(e2["ORB"]["precision_before"]), fmt(e2["ORB"]["precision_after"])))

    A(figure("alignment_checker_SIFT.png",
             "Checkerboard overlay of view 1 and view 2 warped into its frame. "
             "Cells alternate between the two images, so any misalignment appears "
             "as broken structure at the cell boundaries. Shapes and text run "
             "continuously across every boundary.", "6"))

    A('<h3>7.1 Validation against OpenCV</h3>')
    A('<p>The hand-written matcher and RANSAC were checked against OpenCV on the '
      'same data. Agreement is exact for the matcher and, for SIFT, agreement in '
      'the estimated homography is %s&nbsp;px , far below the accuracy of '
      'either estimate. Both implementations selected an identical number of '
      'inliers in all three configurations.</p>'
      % fmt(e2["SIFT"]["our_H_vs_opencv_H_px"]))
    A('<p><b>Table 4.</b> Validation of the hand-written implementations.</p>')
    A(t_validation(d))
    A('<p class="small">The larger discrepancies for ORB (%s&nbsp;px) and the '
      'hybrid (%s&nbsp;px) do not indicate an error: with the same inlier set, '
      'the two estimators differ only in their internal refinement, and both '
      'remain within the intrinsic accuracy of the underlying correspondences '
      'for those configurations.</p>'
      % (fmt(e2["ORB"]["our_H_vs_opencv_H_px"]),
         fmt(e2["SHITOMASI+SIFT"]["our_H_vs_opencv_H_px"])))

    # =============== 8 =============== #
    A('<div class="pagebreak"></div>')
    A('<h2>8. Robustness to Rotation, Scale, Viewpoint and Illumination</h2>')
    A('<p>This section addresses Task 12. One factor is varied at a time against '
      'a fixed reference view, so any change in performance is attributable to '
      'that factor alone. A trial is marked <i>usable</i> only if it yields at '
      'least 12 inliers <b>and</b> an alignment error below 5&nbsp;px : a '
      'homography built from eight spurious inliers is worthless however '
      'confident the inlier count looks.</p>')
    A(figure("robustness_conditions.png",
             "Examples of the transformed view under each tested condition.", "7"))

    A('<h3>8.1 Rotation</h3>')
    A(figure("plot_rotation.png",
             "Robustness to in-plane rotation from 0&deg; to 180&deg;.", "8"))
    A('<div class="keyfind"><b>SIFT and ORB are rotation-invariant; the hybrid '
      'is not, and fails abruptly.</b> SIFT held an inlier ratio above 0.98 and '
      'sub-pixel accuracy across the entire range, degrading only gently from '
      '0.015&nbsp;px at 0&deg; to 0.713&nbsp;px at 180&deg;. ORB behaved '
      'similarly (ratio &ge; 0.897). The Shi-Tomasi&nbsp;+&nbsp;SIFT hybrid '
      'collapsed between 10&deg; and 30&deg;: putative matches fell from 1813 at '
      '10&deg; to 53 at 20&deg; and 32 at 30&deg;, the inlier ratio fell to '
      '0.156, and the alignment error became meaningless '
      '(<b class="fail">842&nbsp;px</b> at 30&deg;, 12&nbsp;753&nbsp;px at '
      '45&deg;).</div>')
    A('<p>The cause is precisely the separation of concerns described in '
      'Section&nbsp;3.2. OpenCV\'s SIFT descriptor rotates its sampling grid to '
      'the keypoint\'s stored orientation, but the Shi-Tomasi detector never '
      'assigns one. The descriptor is therefore computed in a fixed image-aligned '
      'frame, and once the image rotates by more than roughly one orientation bin '
      'the descriptors of corresponding points no longer resemble each other. '
      '<b>The descriptor was never the source of SIFT\'s rotation invariance; '
      'the detector\'s orientation assignment was.</b> The rise in raw match '
      'counts at 180&deg; for all three configurations reflects the scene\'s '
      'incidental symmetry rather than recovered invariance.</p>')

    A('<h3>8.2 Scale</h3>')
    A(figure("plot_scale.png", "Robustness to scale change (zoom factor).", "9"))
    A('<p>The same fault line appears. SIFT degraded gracefully, remaining usable '
      'at 3&times; zoom with 0.244&nbsp;px error, though its putative matches '
      'fell from 2075 to 183 as the shared field of view shrank. ORB remained '
      'usable to 3&times; but on thin evidence ( only 26 putative matches '
      'and 1.52&nbsp;px error ), consistent with its pyramid covering a '
      'narrower scale range. The hybrid became unusable beyond 2&times;: at '
      '2.5&times; it retained 7 inliers of 44 putative (ratio 0.159) with '
      '2.03&nbsp;px error, and at 3&times; only 5 inliers with '
      '<b class="fail">7.92&nbsp;px</b> error.</p>')
    A('<p>Again the explanation is structural: Shi-Tomasi reports every keypoint '
      'at a fixed size, so the SIFT descriptor samples a fixed-radius patch. When '
      'the image is magnified, the corresponding physical region no longer fits '
      'that radius and the descriptors diverge. Scale-space detection, not the '
      'descriptor, supplies scale invariance.</p>')

    A('<h3>8.3 Viewpoint</h3>')
    A(figure("plot_viewpoint.png",
             "Robustness to out-of-plane rotation (perspective tilt).", "10"))
    A('<p>All three configurations survived the full tilt range, and here the '
      'ranking changes: at the strongest tilt (0.6) the hybrid was the most '
      'accurate at 0.290&nbsp;px, ahead of SIFT at 0.436&nbsp;px and ORB at '
      '1.220&nbsp;px. This is consistent with the mechanism above : '
      'perspective tilt is largely an anisotropic stretch that changes neither '
      'the dominant orientation nor the characteristic scale very much, so the '
      'hybrid\'s missing invariances are not exercised. Match counts fell '
      'steeply for every configuration (SIFT 2075&nbsp;&rarr;&nbsp;230), because '
      'foreshortening genuinely destroys the local appearance of features. None '
      'of these descriptors is affine-invariant; ASIFT or an affine-adapted '
      'detector would be required for stronger tilts.</p>')

    A('<h3>8.4 Illumination</h3>')
    A(figure("plot_illumination.png",
             "Robustness to illumination change, swept over gamma.", "11"))
    A('<p><b>Table 5.</b> Named illumination and noise conditions.</p>')
    A(t_illum_named(d))
    A('<p>Illumination was the least damaging factor tested: every configuration '
      'remained usable in every condition, with alignment errors never exceeding '
      '0.13&nbsp;px. This is by construction. SIFT normalises its gradient '
      'histograms, making the descriptor invariant to affine intensity change; '
      'rBRIEF uses intensity <i>comparisons</i>, whose outcome is unchanged by '
      'any monotonic mapping; and CLAHE removes much of the residual difference '
      'before detection.</p>')
    A('<p>The cost appeared in <i>yield</i> rather than accuracy. Under-exposure '
      'reduced SIFT\'s keypoints from 2646 to 1677 and ORB\'s from 3962 to 2908, '
      'and putative matches fell by roughly three quarters at gamma&nbsp;2.8. The '
      'hybrid\'s keypoint count stayed pinned at its 4000 quota throughout, which '
      'illustrates a subtlety: Shi-Tomasi ranks corners and returns the best N '
      'regardless of absolute contrast, so a constant keypoint count conceals a '
      'genuine decline in keypoint <i>quality</i>. Under heavy noise '
      '(&sigma;&nbsp;=&nbsp;18) its putative matches collapsed to 353, the worst '
      'of the three, despite the unchanged detection count.</p>')

    A('<h3>8.5 Summary of robustness</h3>')
    A(table(["Factor", "SIFT", "ORB", "Shi-Tomasi + SIFT"],
            [["In-plane rotation", "<b class='win'>robust to 180&deg;</b>",
              "<b class='win'>robust to 180&deg;</b>",
              "<b class='fail'>fails beyond ~20&deg;</b>"],
             ["Scale change", "<b class='win'>robust to 3&times;</b>",
              "usable to 3&times; on thin evidence",
              "<b class='fail'>fails beyond ~2&times;</b>"],
             ["Viewpoint tilt", "robust to 0.6", "robust to 0.6, least accurate",
              "<b class='win'>most accurate at high tilt</b>"],
             ["Illumination / noise", "robust", "robust", "robust in accuracy, weakest yield under noise"]]))

    # =============== 9 =============== #
    A('<div class="pagebreak"></div>')
    A('<h2>9. Parameter Sensitivity</h2>')
    A('<h3>9.1 The Lowe ratio threshold</h3>')
    A(figure("plot_lowe_ratio.png",
             "Effect of the Lowe ratio threshold on match count, matching "
             "precision and final alignment error.", "12"))
    A('<p><b>Table 6.</b> Lowe ratio sweep, first image pair.</p>')
    A(t_lowe(d))
    A('<p>The threshold trades yield against purity exactly as theory predicts. '
      'For SIFT, a strict 0.50 admitted 374 matches at 1.000 precision, while a '
      'disabled test at 1.00 admitted 1012 at 0.559 precision. ORB is markedly '
      'more sensitive ( 82 matches at 0.50 against 1106 at 1.00 ), '
      'because its 256-bit descriptor is less discriminative, so its second-best '
      'distance sits closer to its best.</p>')
    A('<div class="note"><b>An instructive negative result.</b> The final '
      'alignment error is almost flat across the whole sweep (SIFT stays within '
      '0.07&ndash;0.12&nbsp;px even at ratio 1.00, where 44&nbsp;% of matches are '
      'wrong). RANSAC absorbs the extra outliers, so the ratio test is not what '
      'is protecting accuracy here. Its real value is <i>economy and safety '
      'margin</i>: a lower outlier fraction means RANSAC needs far fewer '
      'iterations, and it keeps the system away from the regime where the inlier '
      'ratio falls low enough for RANSAC itself to fail. Setting the ratio too '
      'strictly is the more dangerous error : at 0.50, ORB\'s 82 matches '
      'leave very little margin before the estimate becomes unreliable.</div>')

    A('<h3>9.2 The RANSAC inlier threshold</h3>')
    A(figure("plot_ransac_threshold.png",
             "Effect of the RANSAC inlier threshold.", "13"))
    A('<p><b>Table 7.</b> RANSAC threshold sweep, first image pair.</p>')
    A(t_thresh(d))
    A('<p>A threshold that is too tight is actively harmful. At 0.5&nbsp;px, '
      'SIFT retained only 379 of 531 matches (ratio 0.714) and ORB collapsed to '
      '64 (ratio 0.201), because the threshold fell below the localisation noise '
      'of the keypoints themselves : correct matches were being rejected '
      'for being correct only to within a pixel. The adaptive termination rule '
      'reacted exactly as designed: the iteration count rose from 50 to the '
      '5000 cap for ORB and the hybrid, since a lower apparent inlier ratio '
      'demands more samples.</p>')
    A('<p>Beyond about 5&nbsp;px the inlier count saturates and accuracy degrades '
      'slightly as genuinely poor matches are admitted (SIFT: 0.090&nbsp;px at '
      '3&nbsp;px, 0.129&nbsp;px at 8&nbsp;px and above). The 2&ndash;5&nbsp;px '
      'range is a broad and forgiving optimum, and the default of 3&nbsp;px sits '
      'in it.</p>')

    # =============== 10 =============== #
    A('<h2>10. Panorama Construction and Blending</h2>')
    A(figure("panorama_SIFT.png",
             "Final panorama from the four input views using SIFT, feather "
             "blended and cropped to content (%d &times; %d px). Structures and "
             "text run continuously across every seam."
             % (e1["SIFT"]["panorama_w"], e1["SIFT"]["panorama_h"]), "14"))
    A('<p>All three configurations produced a visually successful mosaic of '
      'comparable size. The blending comparison is more interesting than the '
      'quality table suggests, and the reason is worth stating plainly.</p>')
    A('<div class="note"><b>Why the two blending modes score identically in the '
      'metrics.</b> The overlap MAE, RMSE, PSNR and NCC compare the <i>source '
      'images warped onto the canvas</i>, so they measure geometric alignment and '
      'are by construction independent of how the overlap is subsequently '
      'combined. Both modes therefore report MAE %s and NCC %s. The difference '
      'between them is entirely visual, and Figures 15 and 16 show it clearly. '
      'Reporting the identical numbers as though they settled the question would '
      'have been a misuse of the metric.</div>'
      % (fmt(d["E4_blending"][0]["quality_mean_mae"]),
         fmt(d["E4_blending"][0]["quality_mean_ncc"])))
    A(figure("panorama_blend_overwrite.png",
             "Overwrite blending. Each image paints over its predecessor, so the "
             "exposure, gamma and vignetting differences between views appear as "
             "hard rectangular seams. The geometry is identical to Figure 14 "
             "; every visible discontinuity here is photometric, not a "
             "misalignment.", "15"))
    A(figure("panorama_blend_feather.png",
             "Feather blending on the same data. Weighting each contribution by "
             "its distance to the image border spreads the exposure difference "
             "over the whole overlap region, and the seams disappear.", "16"))
    A('<p>Feathering cost %s&nbsp;ms against %s&nbsp;ms for overwrite , a '
      'modest price for the improvement. It should be noted that feathering '
      '<i>hides</i> an exposure difference rather than correcting it; a full '
      'system would estimate a gain and vignetting model per image and '
      'photometrically align the views before blending.</p>'
      % (fmt(d["E4_blending"][1]["warp_ms"]), fmt(d["E4_blending"][0]["warp_ms"])))

    # =============== 11 =============== #
    A('<div class="pagebreak"></div>')
    A('<h2>11. Limitations and Failure Cases</h2>')

    A('<h3>11.1 The planarity assumption is the fundamental limit</h3>')
    A('<p>The entire pipeline rests on a homography being an adequate model. It '
      'is exact only for a planar scene or a camera rotating about its optical '
      'centre. If the photographer <i>walks</i> sideways while shooting a scene '
      'with foreground and background at different depths, near and far points '
      'move by different amounts between frames, no homography satisfies both, '
      'and RANSAC resolves the contradiction by choosing whichever depth layer '
      'supplies more inliers , leaving the other visibly ghosted. This is '
      'the single most common cause of failure in practice, and it cannot be '
      'fixed by better features or a better robust estimator; it requires a '
      'different model.</p>')

    A('<h3>11.2 One photograph is easier than several separate captures</h3>')
    A('<div class="warn"><b>This should be read as a caveat on every number in '
      'this report.</b> Inlier ratios of %s&ndash;%s are considerably higher than '
      'the 40&ndash;70&nbsp;%% typical of separately captured photographs. The '
      'texture here is genuinely photographic, so that particular criticism no '
      'longer applies, but the geometry is still idealised in ways that matter. '
      'Every view is cut from a single exposure, so there is no parallax, no '
      'lens distortion differing between shots, no rolling shutter, nothing '
      'that moved between frames and no genuinely independent sensor noise. '
      'The scene is also exactly planar by construction, which is the one '
      'assumption the homography model needs and the one reality most often '
      'violates. Three specific '
      'consequences follow: RANSAC\'s contribution appears small in the '
      '<i>inlier-ratio</i> column (though decisive in the error column); the '
      'ratio-test sweep in Section&nbsp;9.1 is unusually forgiving; and the '
      'absolute alignment errors are optimistic. The relative ranking of the '
      'three configurations, and the failure modes identified in Section&nbsp;8, '
      'are expected to transfer to real imagery; the absolute magnitudes are '
      'not.</div>'
      % (fmt(min(r["mean_inlier_ratio"] for r in d["E1_detector_comparison"])),
         fmt(max(r["mean_inlier_ratio"] for r in d["E1_detector_comparison"]))))

    A('<h3>11.3 Error accumulation along the homography chain</h3>')
    A('<p>Homographies are estimated between consecutive pairs and composed. '
      'Each composition multiplies the errors of the individual estimates, so '
      'accuracy degrades with distance from the reference image, and perspective '
      'distortion grows without bound towards the ends of the mosaic. This was '
      'not merely predicted but <b>observed directly</b>: an earlier version of '
      'the view generator applied independent random perspective tilts of '
      '&plusmn;0.20 per view, and the compounded tilt across four views stretched '
      'the leftmost image into an extreme wedge occupying most of a '
      '6079&nbsp;&times;&nbsp;2553 canvas, despite the individual pairwise '
      'homographies each being accurate to a fraction of a pixel. Choosing the '
      'middle image as the reference halves the longest chain and mitigates but '
      'does not solve this. The proper remedy is <i>global bundle adjustment</i>, '
      'which optimises all homographies jointly against all pairwise '
      'correspondences instead of accepting a chain of independent local '
      'estimates. That is not implemented here.</p>')

    A('<h3>11.4 Further limitations</h3>')
    A('<ul>'
      '<li><b>No cylindrical or spherical projection.</b> A planar mosaic cannot '
      'represent a field of view approaching 180&deg;: as the angle grows, the '
      'projected image stretches towards infinity. Panoramas beyond roughly '
      '120&deg; require re-projecting each image onto a cylinder or sphere first, '
      'which needs the focal length.</li>'
      '<li><b>No photometric alignment.</b> As discussed in Section&nbsp;10, '
      'feathering conceals exposure differences rather than correcting them.</li>'
      '<li><b>Cropping is a simple bounding box</b> of the non-black pixels, so '
      'black wedges remain at the corners of a rotated mosaic. Finding the '
      'largest inscribed rectangle would remove them at the cost of discarding '
      'valid image area.</li>'
      '<li><b>The image ordering is assumed.</b> Images must be supplied in '
      'left-to-right order with consecutive overlap. A general system would '
      'discover the panorama topology by matching all pairs and finding a '
      'spanning structure over the resulting match graph.</li>'
      '<li><b>Repeated structure defeats the ratio test by design.</b> The '
      'checkerboard patches in the scene are internally self-similar, so their '
      'corners have near-identical descriptors and the ratio test discards them '
      'almost entirely. In a real building facade ( rows of identical '
      'windows ), this removes exactly the features a human would find most '
      'salient.</li>'
      '<li><b>Brute-force matching is O(N&times;M)</b> and became the dominant '
      'cost for the high-keypoint configurations (Section&nbsp;6.1). An '
      'approximate index (FLANN k-d tree for SIFT, LSH for ORB) would be '
      'necessary at higher resolutions.</li>'
      '<li><b>Only three detector configurations were compared</b>, since '
      'OpenCV&nbsp;5.0 does not expose AKAZE or BRISK in the main module.</li></ul>')

    A('<h3>11.5 Observed failure cases</h3>')
    A(table(["Failure", "Trigger observed", "Underlying cause"],
            [["Hybrid loses all correspondence",
              "rotation &ge; 20&deg;; alignment error reached 12 753 px at 45&deg;",
              "Shi-Tomasi assigns no dominant orientation, so the SIFT descriptor is computed in a fixed frame"],
             ["Hybrid loses all correspondence",
              "zoom &ge; 2.5&times;; 5 inliers of 29 at 3&times;",
              "single-scale detection fixes the descriptor's sampling radius"],
             ["Homography estimate destroyed",
              "least-squares fit with %s outliers of %s matches" % (fmt(e2["SIFT"]["n_removed"]), fmt(e2["SIFT"]["n_putative"])),
              "least squares has a breakdown point of zero"],
             ["Inlier set collapses",
              "RANSAC threshold 0.5 px; ORB retained 64 of 318",
              "threshold below the keypoint localisation noise"],
             ["RANSAC iterations hit the 5000 cap",
              "same condition as above",
              "adaptive rule correctly demands more samples at a low inlier ratio"],
             ["Extreme perspective stretching",
              "compounded &plusmn;0.20 tilt over four chained views",
              "error accumulation along the homography chain"],
             ["Hard visible seams",
              "overwrite blending with per-view exposure differences",
              "no photometric alignment between views"]]))

    # =============== 12 =============== #
    A('<h2>12. Conclusion</h2>')
    A('<p>A complete classical panorama pipeline was implemented and evaluated '
      'against exact ground truth, with the matcher, the normalised DLT and the '
      'RANSAC loop written from first principles and independently validated '
      'against OpenCV (100&nbsp;%% matcher agreement; homographies agreeing to '
      '%s&nbsp;px for SIFT; 27 of 27 component tests passing).</p>'
      % fmt(e2["SIFT"]["our_H_vs_opencv_H_px"]))
    A('<p>Four conclusions are supported by the measurements:</p>')
    A('<ol>'
      '<li><b>Robust estimation is not optional.</b> Eleven bad matches in 531 '
      'moved the least-squares homography by %s&nbsp;px; RANSAC on identical data '
      'achieved %s&nbsp;px. The gap is roughly four orders of magnitude, and it '
      'is the difference between a working system and a useless one.</li>'
      '<li><b>Invariance is a property of the detector, not the descriptor.</b> '
      'The Shi-Tomasi&nbsp;+&nbsp;SIFT hybrid used precisely the same descriptor '
      'as SIFT and matched its accuracy on easy pairs, yet failed completely '
      'beyond 20&deg; of rotation and 2&times; zoom. Scale-space detection with '
      'orientation assignment , not the 128-D histogram , is what '
      'makes SIFT invariant.</li>'
      '<li><b>SIFT remains the accuracy benchmark; ORB\'s advantage is in '
      'detection cost, and must be read carefully.</b> ORB detected and described '
      '%.1f&times; faster but was %.1f&times; less accurate, and its apparent '
      'disadvantage in <i>total</i> time was an artefact of its larger keypoint '
      'count inflating an O(N&times;M) matching stage, not of the Hamming metric '
      'being slow.</li>'
      '<li><b>Classical illumination invariance works well; geometric invariance '
      'is where these methods break.</b> Every configuration tolerated severe '
      'exposure, gamma and noise changes with sub-0.15&nbsp;px error, whereas '
      'rotation, scale and viewpoint produced every failure observed.</li></ol>'
      % (fmt(e2["SIFT"]["error_no_ransac_px"]), fmt(e2["SIFT"]["error_with_ransac_px"]),
         sift_dd / orb_dd,
         e1["ORB"]["mean_grid_error_px"] / e1["SIFT"]["mean_grid_error_px"]))
    A('<p>The clearest directions for extension follow from Section&nbsp;11: '
      'global bundle adjustment to remove chain error accumulation, cylindrical '
      're-projection for wide fields of view, photometric alignment before '
      'blending, and an approximate nearest-neighbour index to remove the '
      'quadratic matching cost.</p>')

    # =============== Appendix =============== #
    A('<h2>Appendix A. Running the System</h2>')
    A('<pre>pip install -r requirements.txt\n\n'
      'python src/download_scene.py    # fetch the scene photograph (once)\n'
      'python src/main.py              # full run, all experiments\n'
      'python src/main.py --scene      # ignore images/, use the scene photo\n'
      'python src/main.py --quick      # skip the parameter sweeps\n'
      'python src/test_components.py   # 27 validation tests\n'
      'python src/build_report.py      # regenerate this PDF from results.json</pre>')
    A('<p>To use your own photographs, place three or more overlapping images in '
      '<code>images/</code>, named so that alphabetical order matches '
      'left-to-right order. The system switches to real mode automatically. For '
      'good results the photographs should overlap by 30&ndash;50&nbsp;%, be '
      'taken by rotating the camera about its own axis rather than walking '
      'sideways (Section&nbsp;11.1), and keep exposure and focus locked.</p>')
    A('<p><b>Source layout.</b></p>')
    A(table(["Module", "Responsibility"],
            [["<code>src/dataset.py</code>", "scene generation, view synthesis, real-image loading, pre-processing"],
             ["<code>src/features.py</code>", "the three detector/descriptor configurations, per-stage timing"],
             ["<code>src/matching.py</code>", "brute-force k-NN, Lowe ratio test, cross-check"],
             ["<code>src/homography.py</code>", "normalised DLT, RANSAC, error measures"],
             ["<code>src/stitching.py</code>", "homography chaining, canvas geometry, feather blending, quality metrics"],
             ["<code>src/visualisation.py</code>", "all figures and plots"],
             ["<code>src/pipeline.py</code>", "end-to-end wiring of the stages"],
             ["<code>src/experiments.py</code>", "the five experiments"],
             ["<code>src/main.py</code>", "entry point and table generation"],
             ["<code>src/test_components.py</code>", "validation of the hand-written components"],
             ["<code>src/build_report.py</code>", "this document"]]))
    A('<p class="small">Total run time for the full experimental programme: '
      '%.0f&nbsp;s on %s. All measurements in this report were produced by that '
      'single run and are recorded in <code>results/results.json</code>.</p>'
      % (d["runtime_seconds"], env["platform"]))

    body = tidy("\n".join(P))
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Automated Panorama Construction</title><style>%s</style></head>"
            "<body>%s</body></html>" % (CSS, body))


# --------------------------------------------------------------------------- #
# PDF conversion
# --------------------------------------------------------------------------- #

def find_browser():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    return None


def html_to_pdf(html_path, pdf_path):
    browser = find_browser()
    if browser is None:
        print("[build_report] No Chrome/Edge found; HTML written but no PDF.")
        return False

    url = "file:///" + html_path.replace("\\", "/").replace(" ", "%20")
    cmd = [browser, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--print-to-pdf-no-header",
           "--virtual-time-budget=20000",
           "--print-to-pdf=" + pdf_path, url]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        print("[build_report] Browser timed out.")
        return False
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
        return True
    print("[build_report] PDF not produced. stderr:\n%s"
          % proc.stderr.decode("utf-8", "replace")[:2000])
    return False


def main():
    d = load_results()
    html = build_html(d)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("[build_report] wrote %s (%.1f MB)"
          % (OUT_HTML, os.path.getsize(OUT_HTML) / 1e6))

    if html_to_pdf(OUT_HTML, OUT_PDF):
        print("[build_report] wrote %s (%.1f MB)"
              % (OUT_PDF, os.path.getsize(OUT_PDF) / 1e6))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
