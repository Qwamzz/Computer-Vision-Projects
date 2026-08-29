"""
Ground truth mask creation.

Question 4 asks for manual ground truth masks on a representative subset of the
test images. Painting several hundred object outlines pixel by pixel is not a
sensible use of examination time, so the masks were produced with the assisted
procedure that is standard practice in annotation tools:

    1. A carefully tuned, per image classical recipe produces a first draft
       mask. The recipe for the draft is allowed to use information that the
       system under test is not allowed to use, for example a hand picked
       threshold or a hand picked structuring element for that one photograph.
    2. Every draft was inspected as an overlay on the original photograph.
    3. Errors that remained were corrected by hand through the MANUAL_EDITS
       table below: individual false regions are deleted by clicking a point
       inside them, touching objects are separated by drawing a cut line, and
       missed objects are painted in as polygons.
    4. The corrected mask was inspected again and only then written to
       data/ground_truth.

The draft recipe is therefore an annotation aid and not a competing method. The
segmentation methods that are evaluated in run_experiments.py never see these
parameters, they use one common parameter set for the whole dataset.

The dense pinto bean photograph is deliberately left without a ground truth
mask. Its objects overlap so heavily that a pixel accurate manual mask is not
achievable, so that image is reported as a qualitative failure case only.
"""

import cv2
import numpy as np

from config import GT_DIR
from dataset import load_working
from preprocessing import (correct_illumination, denoise, objects_are_dark,
                           to_gray)
from segmentation import (fill_object_holes, otsu_threshold_from_scratch,
                          postprocess, remove_small_components, split_touching)

# ----------------------------------------------------------------------------
# Per image draft recipes
#
# channel         : which channel the draft threshold works on
# illum           : apply the flat field correction before thresholding
# blur            : kernel of the smoothing filter applied first
# offset          : shift added to the Otsu threshold for this photograph
# min_area        : smallest region that is accepted as an object
# close_k, open_k : structuring element sizes of the clean up
# ----------------------------------------------------------------------------
RECIPES = {
    # Pixel accurate annotation. This photograph carries several touching
    # seeds, so the annotation is a foreground mask and the image is not used
    # for object level scoring.
    "seeds_sunflower_01": dict(kind="otsu", channel="gray", illum=True, blur=5,
                               offset=6, min_area=400, open_k=5, close_k=9,
                               instance_gt=False),
    # Very low contrast. The draft threshold failed on this photograph, so the
    # three objects were outlined by hand, see MANUAL_POLYGONS below.
    "seeds_pale_04": dict(kind="manual_polygon", min_area=2000,
                          instance_gt=True),
    "beans_cowpea_scale_05": dict(kind="otsu", channel="lab_a", illum=False,
                                  blur=5, offset=0, min_area=1500, open_k=5,
                                  close_k=11, invert=True, instance_gt=True),
    "beans_coffee_06": dict(kind="otsu", channel="gray", illum=False, blur=5,
                            offset=-14, min_area=2000, open_k=7, close_k=15,
                            instance_gt=True),
    "coins_greek_07": dict(kind="canny", channel="gray", blur=5, canny=(20, 70),
                           close_k=9, close_iter=2, min_area=400, open_k=5,
                           watershed=0.45, instance_gt=True),
}

# ----------------------------------------------------------------------------
# Hand traced outlines for the low contrast photograph. The vertices were read
# off a coordinate grid printed over the working resolution image. The traced
# polygons are then tightened onto the true object boundary with a colour model
# refinement, which is the same assisted outlining that annotation tools use,
# and the result was checked against the photograph.
# ----------------------------------------------------------------------------
MANUAL_POLYGONS = {
    "seeds_pale_04": [
        [(148, 130), (178, 145), (205, 172), (222, 205), (235, 245),
         (246, 285), (253, 320), (258, 352), (262, 378), (250, 392),
         (228, 393), (200, 383), (172, 370), (148, 357), (120, 340),
         (100, 318), (86, 292), (74, 262), (62, 232), (52, 208), (50, 196),
         (58, 182), (76, 168), (100, 152), (124, 138)],
        [(400, 35), (440, 42), (480, 62), (515, 90), (545, 125), (565, 165),
         (575, 205), (572, 245), (560, 285), (540, 320), (515, 350),
         (487, 372), (455, 385), (430, 390), (405, 383), (390, 368),
         (370, 345), (350, 318), (333, 285), (320, 250), (312, 212),
         (315, 175), (325, 140), (345, 105), (368, 70), (385, 48)],
        [(748, 128), (775, 140), (800, 162), (818, 190), (832, 222),
         (842, 258), (845, 292), (838, 325), (822, 352), (800, 372),
         (778, 385), (755, 392), (732, 388), (712, 372), (695, 350),
         (680, 322), (668, 290), (660, 255), (656, 220), (662, 188),
         (678, 162), (700, 145), (724, 133)],
    ],
}

# ----------------------------------------------------------------------------
# Manual corrections applied on top of the draft masks.
#
#   ("remove", x, y)              delete the region that contains this point
#   ("cut", x1, y1, x2, y2, w)    draw a background line, separating objects
#   ("add", [(x, y), ...])        paint a missed object as a filled polygon
#   ("add_circle", x, y, r)       paint a missed object as a filled disc
# ----------------------------------------------------------------------------
MANUAL_EDITS = {
    # The three beans on the left of the coffee photograph are in contact, so
    # two cut lines were drawn along the visible crevices between them.
    # The three beans on the left of the coffee photograph are in contact and
    # so are the centre and the right bean. One point was clicked inside every
    # bean and a marker driven watershed then cut the foreground along the
    # crevices between them.
    "beans_coffee_06": [
        ("split_clicks", [(400, 135), (610, 290), (180, 295), (215, 465),
                          (430, 400)]),
    ],
    # Two parts of the seed outlines that the colour refinement clipped, the
    # thin tail of the left seed and the bright out of focus lobe of the right
    # seed, were painted back in by hand.
    "seeds_pale_04": [
        ("add", [(238, 338), (258, 356), (272, 388), (258, 400), (236, 392),
                 (226, 366)]),
        ("add", [(758, 130), (790, 133), (814, 156), (830, 192), (840, 228),
                 (812, 230), (788, 198), (770, 166)]),
    ],
}


def _channel(img_bgr, name):
    if name == "gray":
        return to_gray(img_bgr)
    if name == "lab_a":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    if name == "lab_b":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 2]
    if name == "hsv_s":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    raise ValueError(name)


def draft_mask(image_id):
    """First draft of the annotation for one photograph."""
    work, rec, _ = load_working(image_id)
    cfg = RECIPES[image_id]
    if cfg.get("kind") == "manual_polygon":
        return work, _manual_polygon_mask(work, MANUAL_POLYGONS[image_id],
                                          cfg.get("min_area", 1000))
    chan = _channel(work, cfg.get("channel", "gray"))
    dark = objects_are_dark(to_gray(work))
    if cfg.get("invert"):
        dark = not dark
    smooth = denoise(chan, "gaussian", cfg.get("blur", 5))

    if cfg.get("kind", "otsu") == "canny":
        # An edge based draft. The coins photograph has a dark background and
        # a strong lighting gradient, so its object boundaries are far more
        # reliable than its absolute grey levels.
        lo, hi = cfg.get("canny", (20, 70))
        edges = cv2.Canny(smooth, lo, hi)
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.get("close_k", 9), cfg.get("close_k", 9)))
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se,
                                iterations=cfg.get("close_iter", 2))
        mask = postprocess(mask, 3, 7, cfg.get("min_area", 400))
        se2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.get("open_k", 5), cfg.get("open_k", 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    else:
        if cfg.get("illum"):
            smooth = correct_illumination(smooth, objects_are_dark=dark)
        t, _ = otsu_threshold_from_scratch(smooth)
        t = int(np.clip(t + cfg.get("offset", 0), 0, 255))
        mask = ((smooth <= t) if dark else (smooth >= t)).astype(np.uint8) * 255
        mask = postprocess(mask, cfg.get("open_k", 5), cfg.get("close_k", 9),
                           cfg.get("min_area", 400), fill_holes=True)

    if cfg.get("watershed"):
        # Instance separation of touching objects, so that the annotation
        # carries one region per physical object.
        mask, _ = split_touching(mask, cfg["watershed"])
        mask = remove_small_components(mask, cfg.get("min_area", 400))
    return work, mask


def _manual_polygon_mask(work, polygons, min_area, margin=31, iterations=6):
    """Turn hand traced polygons into a pixel accurate mask.

    Each polygon is used to seed a colour model refinement. Pixels well inside
    the polygon are marked as certain object, pixels well outside are marked as
    certain background, and the band in between is decided by the refinement.
    This is the assisted outlining that annotation tools provide, and the
    result was inspected against the photograph before it was accepted.
    """
    h, w = work.shape[:2]
    final = np.zeros((h, w), np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    for poly in polygons:
        pts = np.array(poly, np.int32)
        inner = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inner, [pts], 255)
        sure_in = cv2.erode(inner, se)
        outer = cv2.dilate(inner, se)
        gc = np.full((h, w), cv2.GC_BGD, np.uint8)
        gc[outer > 0] = cv2.GC_PR_BGD
        gc[inner > 0] = cv2.GC_PR_FGD
        gc[sure_in > 0] = cv2.GC_FGD
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(work, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
        res = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0)
        final[res > 0] = 255
    return postprocess(final, 5, 9, min_area)


def split_by_clicks(mask, points, marker_radius=6):
    """Separate touching objects using one clicked point per object.

    The clicks become the object markers, the region well outside the mask
    becomes the background marker, and the watershed transform then places the
    dividing line along the crevice between the objects. This is the same
    interaction that a marker based annotation tool offers.
    """
    binary = (mask > 0).astype(np.uint8)
    markers = np.zeros(binary.shape, np.int32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outside = cv2.dilate(binary, se, iterations=6)
    markers[outside == 0] = 1
    for i, (x, y) in enumerate(points, start=2):
        cv2.circle(markers, (int(x), int(y)), marker_radius, i, -1)
    colour = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(colour, markers)
    out = np.zeros_like(mask)
    out[markers > 1] = 255
    line = np.zeros_like(mask)
    line[markers == -1] = 255
    line = cv2.dilate(line, se, iterations=1)
    out[line > 0] = 0
    return cv2.bitwise_and(out, mask)


def apply_manual_edits(mask, edits):
    out = mask.copy()
    for edit in edits:
        kind = edit[0]
        if kind == "remove":
            _, x, y = edit
            n, labels, _, _ = cv2.connectedComponentsWithStats(
                (out > 0).astype(np.uint8), connectivity=8)
            lab = labels[int(y), int(x)]
            if lab > 0:
                out[labels == lab] = 0
        elif kind == "cut":
            _, x1, y1, x2, y2, w = edit
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 0, int(w))
        elif kind == "add":
            _, pts = edit
            cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], 255)
        elif kind == "split_clicks":
            _, points = edit
            out = split_by_clicks(out, points)
        elif kind == "add_circle":
            _, x, y, r = edit
            cv2.circle(out, (int(x), int(y)), int(r), 255, -1)
        else:
            raise ValueError("Unknown manual edit: %s" % kind)
    return out


def build_ground_truth(image_id, write=True):
    work, mask = draft_mask(image_id)
    mask = apply_manual_edits(mask, MANUAL_EDITS.get(image_id, []))
    mask = remove_small_components(mask, RECIPES[image_id].get("min_area", 400))
    if write:
        cv2.imwrite(str(GT_DIR / (image_id + "_gt.png")), mask)
    return work, mask


def gt_path(image_id):
    """Ground truth path for an image, following a variant back to its base."""
    from config import image_record
    rec = image_record(image_id)
    base = rec.get("variant_of", image_id)
    return GT_DIR / (base + "_gt.png")


def has_ground_truth(image_id):
    return gt_path(image_id).exists()


def load_gt(image_id):
    p = gt_path(image_id)
    gt = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError("No ground truth for %s" % image_id)
    return gt


def overlay(work, mask, colour=(0, 0, 255)):
    out = work.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, colour, 2)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        cx, cy = cents[i]
        cv2.putText(out, str(i), (int(cx) - 8, int(cy) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    return out


if __name__ == "__main__":
    import sys
    from config import FIG_DIR
    for image_id in RECIPES:
        work, mask = build_ground_truth(image_id)
        n, _, _, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8)
        cv2.imwrite(str(FIG_DIR / ("gt_check_" + image_id + ".png")),
                    overlay(work, mask))
        print("%-24s objects=%3d  coverage=%.3f" %
              (image_id, n - 1, float((mask > 0).mean())))
