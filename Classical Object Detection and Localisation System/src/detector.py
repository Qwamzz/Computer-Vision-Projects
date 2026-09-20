"""
detector.py
Classical Object Detection and Localisation

Classical detection and localisation of the University of Ghana coat of arms.
No pre-trained recognition network is used anywhere. Everything is built from
classical operators:

    * normalised cross correlation template matching (masked and unmasked)
    * an image pyramid for scale search
    * a rotated template bank for orientation search
    * an explicit sliding window search with configurable window and step
    * greedy non maximum suppression
    * SIFT keypoints, ratio test matching and RANSAC homography
    * HSV colour segmentation used as a region proposal stage

Author: Nii Yartey Gidiglo
"""

import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# basic geometry helpers
# --------------------------------------------------------------------------


def iou(box_a, box_b):
    """Intersection over union of two boxes given as [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = float((x1 - x0) * (y1 - y0))
    return inter / (aw * ah + bw * bh - inter)


def nms_indices(boxes, scores, iou_thr=0.30):
    """Greedy non maximum suppression, returning the indices that survive.

    Dense correlation produces a cluster of responses around every true
    location, so the raw peaks must be reduced to one detection per object.
    """
    if not boxes:
        return []
    order = list(np.argsort(scores)[::-1])
    keep = []
    suppressed = np.zeros(len(boxes), bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order:
            if not suppressed[j] and j != i and iou(boxes[i], boxes[j]) > iou_thr:
                suppressed[j] = True
    return keep


def non_max_suppression(boxes, scores, iou_thr=0.30):
    keep = nms_indices(boxes, scores, iou_thr)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------


def preprocess(bgr, clahe=False, blur=0):
    """Convert to grey scale, optionally equalise locally and denoise.

    Grey scale is used because normalised cross correlation is defined on a
    single channel and because the emblem is discriminated mainly by its
    shape and internal contrast rather than by absolute colour.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur:
        k = blur if blur % 2 == 1 else blur + 1
        grey = cv2.GaussianBlur(grey, (k, k), 0)
    if clahe:
        grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    return grey


# --------------------------------------------------------------------------
# template bank
# --------------------------------------------------------------------------


def rotate_with_mask(tmpl, mask, angle):
    """Rotate a template and its mask, expanding the canvas."""
    h, w = tmpl.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    rt = cv2.warpAffine(tmpl, M, (nw, nh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rm = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rt, rm


def build_template_bank(tmpl, mask, scales=(1.0,), angles=(0.0,), min_side=18):
    """Return a list of (template, mask, scale, angle) at every scale and angle.

    Template matching is invariant to neither scale nor rotation, so the
    invariance has to be bought explicitly by searching over a bank of
    transformed templates. The cost of the search grows linearly with the
    size of the bank, which is the trade off examined in the report.
    """
    bank = []
    for s in scales:
        w = max(min_side, int(round(tmpl.shape[1] * s)))
        h = max(min_side, int(round(tmpl.shape[0] * s)))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        ts = cv2.resize(tmpl, (w, h), interpolation=interp)
        ms = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        for a in angles:
            if abs(a) < 1e-6:
                bank.append((ts, ms, s, 0.0))
            else:
                rt, rm = rotate_with_mask(ts, ms, a)
                bank.append((rt, rm, s, a))
    return bank


# --------------------------------------------------------------------------
# method 1 and 2: dense template matching over a template bank
# --------------------------------------------------------------------------


def _peaks_from_response(resp, tw, th, threshold, offset=(0, 0), max_peaks=300):
    """Collect the local maxima of the correlation surface above the threshold.

    Taking every pixel above the threshold would return tens of thousands of
    positions around each response peak. A grey scale dilation keeps only
    positions that are the maximum of their neighbourhood, which is a cheap
    local peak detector and leaves non maximum suppression very little to do.
    """
    resp = np.nan_to_num(resp, nan=-1.0, posinf=-1.0, neginf=-1.0)
    k = max(3, int(min(tw, th) * 0.5) | 1)
    local_max = cv2.dilate(resp, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ys, xs = np.where((resp >= threshold) & (resp >= local_max - 1e-6))
    if len(xs) == 0:
        return [], []
    scores = resp[ys, xs]
    order = np.argsort(scores)[::-1][:max_peaks]
    boxes = [[int(xs[i]) + offset[0], int(ys[i]) + offset[1], tw, th] for i in order]
    return boxes, [float(scores[i]) for i in order]


def template_match(grey, bank, threshold, masked=True, nms_iou=0.30, rois=None):
    """Dense normalised cross correlation search over a template bank.

    cv2.matchTemplate slides the template over every pixel position, so this
    is itself an exhaustive sliding window search with a step of one pixel,
    implemented with an optimised correlation. The masked variant excludes
    the rectangular corners that do not belong to the emblem, which matters
    because the coat of arms is not a rectangle.
    """
    boxes, scores, meta = [], [], []
    regions = rois if rois else [[0, 0, grey.shape[1], grey.shape[0]]]
    for tmpl, mask, s, a in bank:
        th, tw = tmpl.shape[:2]
        for rx, ry, rw, rh in regions:
            sub = grey[ry:ry + rh, rx:rx + rw]
            if sub.shape[0] < th or sub.shape[1] < tw:
                continue
            if masked:
                resp = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
            else:
                resp = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
            b, sc = _peaks_from_response(resp, tw, th, threshold, (rx, ry))
            boxes.extend(b)
            scores.extend(sc)
            meta.extend([(s, a)] * len(b))
    if not boxes:
        return [], [], []
    keep = nms_indices(boxes, scores, nms_iou)
    return ([boxes[i] for i in keep], [scores[i] for i in keep],
            [meta[i] for i in keep])


def response_map(grey, tmpl, mask, masked=True):
    """Single correlation surface, used for the intermediate visualisations."""
    if masked:
        resp = cv2.matchTemplate(grey, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        resp = cv2.matchTemplate(grey, tmpl, cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(resp, nan=-1.0, posinf=-1.0, neginf=-1.0)


# --------------------------------------------------------------------------
# method 3: explicit sliding window search
# --------------------------------------------------------------------------


def sliding_window_detect(grey, tmpl, mask, win_sizes, step, threshold,
                          nms_iou=0.30):
    """Explicit sliding window detector.

    For every window position the patch is resized to the canonical template
    size and compared with the template using the masked zero mean normalised
    cross correlation

        NCC = sum(w' t') / (||w'|| ||t'||)

    where w' and t' are the mean removed pixel values inside the mask. This
    is the textbook detector: a search strategy (where to look), a comparison
    function (how similar), and a decision rule (the threshold). Window size,
    step size and threshold are exposed so their effect can be measured.
    """
    th, tw = tmpl.shape[:2]
    m = mask > 0
    t = tmpl.astype(np.float32)[m]
    t = t - t.mean()
    tn = np.linalg.norm(t) + 1e-9

    H, W = grey.shape[:2]
    boxes, scores, windows = [], [], 0
    t0 = time.perf_counter()
    for (wh, ww) in win_sizes:
        if wh > H or ww > W:
            continue
        for y in range(0, H - wh + 1, step):
            for x in range(0, W - ww + 1, step):
                patch = grey[y:y + wh, x:x + ww]
                if (wh, ww) != (th, tw):
                    patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
                v = patch.astype(np.float32)[m]
                v = v - v.mean()
                ncc = float(np.dot(v, t) / ((np.linalg.norm(v) + 1e-9) * tn))
                windows += 1
                if ncc >= threshold:
                    boxes.append([x, y, ww, wh])
                    scores.append(ncc)
    elapsed = time.perf_counter() - t0
    kept_b, kept_s = non_max_suppression(boxes, scores, nms_iou)
    return kept_b, kept_s, {"windows": windows, "seconds": elapsed}


def window_sizes_from_scales(tmpl_shape, scales):
    th, tw = tmpl_shape[:2]
    out = []
    for s in scales:
        out.append((max(8, int(round(th * s))), max(8, int(round(tw * s)))))
    return out


# --------------------------------------------------------------------------
# method 4: SIFT feature matching with RANSAC homography
# --------------------------------------------------------------------------


def feature_match_detect(grey, tmpl_grey, ratio=0.75, min_inliers=8,
                         ransac_thresh=4.0, max_instances=3):
    """Locate the emblem from local features instead of raw intensities.

    SIFT descriptors are invariant to scale and rotation and largely
    invariant to affine illumination change, so this stage recovers the
    instances that fixed template correlation loses. The homography from
    RANSAC gives the localisation: the four template corners are mapped into
    the scene and the axis aligned box around them is reported.

    A single homography can only explain one instance, so the search is
    repeated: after each accepted instance the scene keypoints that fall
    inside the detected quadrilateral are removed and the remaining matches
    are fitted again. This recovers the second emblem in the scenes that
    contain two.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tmpl_grey, None)
    kp2, des2 = sift.detectAndCompute(grey, None)
    info = {"template_keypoints": len(kp1), "scene_keypoints": len(kp2),
            "good_matches": 0, "inliers": 0, "instances": 0, "quads": []}
    if des1 is None or des2 is None or len(kp2) < 2:
        return [], [], info

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < ratio * n.distance]
    info["good_matches"] = len(good)
    info["good_match_objects"] = good
    info["template_kp"] = kp1
    info["scene_kp"] = kp2

    h, w = tmpl_grey.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    remaining = list(good)
    boxes, scores = [], []
    for _ in range(max_instances):
        if len(remaining) < min_inliers:
            break
        src = np.float32([kp1[m.queryIdx].pt for m in remaining]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in remaining]).reshape(-1, 1, 2)
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None or status is None:
            break
        inliers = int(status.sum())
        if inliers < min_inliers:
            break
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        x0, y0 = float(proj[:, 0].min()), float(proj[:, 1].min())
        bw = float(proj[:, 0].max()) - x0
        bh = float(proj[:, 1].max()) - y0
        keep_outside = [m for k, m in enumerate(remaining) if not status[k]]
        if 10 < bw < grey.shape[1] * 1.5 and 10 < bh < grey.shape[0] * 1.5:
            boxes.append([int(round(x0)), int(round(y0)),
                          int(round(bw)), int(round(bh))])
            scores.append(inliers / float(max(1, len(good))))
            info["quads"].append(proj.tolist())
            info["inliers"] += inliers
        remaining = keep_outside
    info["instances"] = len(boxes)
    if boxes:
        info["projected_quad"] = info["quads"][0]
    return boxes, scores, info


# --------------------------------------------------------------------------
# method 5: colour segmentation used as a region proposal stage
# --------------------------------------------------------------------------


def colour_proposals(bgr, pad=0.45, min_area=280, max_regions=12):
    """Propose regions that contain the navy blue of the shield.

    The emblem has a saturated dark blue field. Thresholding the hue and
    saturation channels of HSV, closing the result and taking connected
    components gives a small number of candidate rectangles. Template
    matching then runs only inside those rectangles, which cuts the search
    space by a large factor without changing the comparison function.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 70, 25]), np.array([140, 255, 220]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    H, W = mask.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ar = w / float(h)
        if ar < 0.25 or ar > 4.0:
            continue
        px, py = int(w * pad), int(h * pad)
        rx, ry = max(0, x - px), max(0, y - py)
        rw, rh = min(W - rx, w + 2 * px), min(H - ry, h + 2 * py)
        regions.append(([int(rx), int(ry), int(rw), int(rh)], int(area)))
    regions.sort(key=lambda r: -r[1])
    return [r[0] for r in regions[:max_regions]], mask


def hybrid_detect(bgr, grey, bank, threshold, nms_iou=0.30):
    """Colour proposals followed by masked template matching inside them."""
    rois, mask = colour_proposals(bgr)
    if not rois:
        return [], [], {"rois": 0, "search_fraction": 0.0, "mask": mask}
    covered = sum(r[2] * r[3] for r in rois)
    boxes, scores, _ = template_match(grey, bank, threshold, True, nms_iou, rois)
    frac = covered / float(grey.shape[0] * grey.shape[1])
    return boxes, scores, {"rois": len(rois), "search_fraction": frac, "mask": mask,
                           "roi_boxes": rois}


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def draw_boxes(img, boxes, scores=None, colour=(0, 200, 0), label=None, thickness=2):
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
        text = ""
        if label:
            text = label
        if scores is not None and i < len(scores):
            text = (text + " " if text else "") + "%.2f" % scores[i]
        if text:
            cv2.rectangle(out, (x, y - 18), (x + 8 * len(text) + 6, y), colour, -1)
            cv2.putText(out, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return out
