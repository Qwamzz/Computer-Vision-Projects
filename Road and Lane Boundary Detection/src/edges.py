"""
edges.py
Edge detection and the region of interest.

Edge detection. Two operators are implemented and compared. Sobel estimates
the image gradient with a small separable kernel and thresholds its magnitude,
which is the simplest thing that can work and returns thick, noisy responses.
Canny adds the three steps that make an edge detector usable: smoothing,
non maximum suppression across the gradient direction so that edges come out
one pixel wide, and hysteresis thresholding so that a weak edge is kept when it
connects to a strong one. Hysteresis is the reason Canny survives a faded lane
marking that a single threshold on Sobel magnitude either loses entirely or
buries in noise.

Region of interest. Almost everything above the horizon is irrelevant, and it
is also where most of the strong edges live: trees, buildings, poles and the
skyline. Restricting the search to a trapezoid that covers the carriageway in
front of the camera removes those distractors before they can be fitted, at the
cost of one strong assumption, that the camera points along the road. Section 6
of the report shows what that assumption costs on the curved scenes.
"""

from __future__ import annotations

import cv2
import numpy as np

EDGE_METHODS = ["canny", "sobel"]

# Trapezoid corners as fractions of the frame, (x, y), y downwards.
# Wide at the bottom of the frame, narrow near the horizon.
DEFAULT_ROI = dict(bottom_left=0.00, bottom_right=1.00,
                   top_left=0.28, top_right=0.72, top=0.45)


def roi_polygon(shape, roi=None):
    """Return the region of interest trapezoid in pixels."""
    roi = dict(DEFAULT_ROI, **(roi or {}))
    h, w = shape[:2]
    return np.array([[
        (int(roi["bottom_left"] * w), h - 1),
        (int(roi["top_left"] * w), int(roi["top"] * h)),
        (int(roi["top_right"] * w), int(roi["top"] * h)),
        (int(roi["bottom_right"] * w), h - 1),
    ]], np.int32)


def roi_mask(shape, roi=None):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, roi_polygon(shape, roi), 255)
    return m


def apply_roi(img, roi=None):
    return cv2.bitwise_and(img, img, mask=roi_mask(img.shape, roi))


def detect_edges(grey, method="canny", low=50, high=150, sobel_k=3,
                 sobel_thresh=60):
    """Return a binary edge map.

    canny : low and high are the hysteresis thresholds.
    sobel : the gradient magnitude is thresholded at sobel_thresh, with no
            thinning and no hysteresis, which is exactly the point of the
            comparison.
    """
    if method == "canny":
        return cv2.Canny(grey, low, high, apertureSize=3, L2gradient=True)
    if method == "sobel":
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=sobel_k)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=sobel_k)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return (mag >= sobel_thresh).astype(np.uint8) * 255
    raise ValueError("unknown edge method " + method)


def edge_density(edge_map, mask=None):
    """Fraction of pixels that the operator declared to be edges."""
    if mask is not None:
        sel = mask > 0
        if sel.sum() == 0:
            return 0.0
        return float((edge_map[sel] > 0).mean())
    return float((edge_map > 0).mean())


def boundary_precision(edge_map, gt_mask, tolerance=4, mask=None):
    """Of the edges the operator reported, how many lie on the true boundary?

    The complement of boundary_recall. Without it a detector can score well by
    simply declaring everything to be an edge, which is the degenerate case the
    Sobel operator falls into at a low threshold.
    """
    gt_edge = cv2.Canny((gt_mask > 0).astype(np.uint8) * 255, 50, 150)
    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    near_gt = cv2.dilate(gt_edge, k) > 0
    det = edge_map > 0
    if mask is not None:
        det = det & (mask > 0)
    total = det.sum()
    if total == 0:
        return 0.0
    return float((det & near_gt).sum() / total)


def boundary_f1(edge_map, gt_mask, tolerance=4, mask=None):
    """Harmonic mean of boundary recall and boundary precision."""
    r = boundary_recall(edge_map, gt_mask, tolerance)
    p = boundary_precision(edge_map, gt_mask, tolerance, mask)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def boundary_recall(edge_map, gt_mask, tolerance=4):
    """How much of the true road boundary did the operator actually find?

    The ground truth boundary is dilated by `tolerance` pixels and the
    fraction of it that coincides with a detected edge is reported. This
    measures the edge stage on its own terms, before any line fitting, and is
    what separates a detector that is merely busy from one that is useful.
    """
    gt_edge = cv2.Canny((gt_mask > 0).astype(np.uint8) * 255, 50, 150)
    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    found = cv2.dilate(edge_map, k) > 0
    total = (gt_edge > 0).sum()
    if total == 0:
        return 0.0
    return float((found & (gt_edge > 0)).sum() / total)
