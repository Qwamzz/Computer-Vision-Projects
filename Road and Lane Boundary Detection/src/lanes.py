"""
lanes.py
Identifying candidate lane and road boundaries from the edge map.

The probabilistic Hough transform turns a cloud of edge pixels into line
segments. Each segment votes for a line in parameter space, and a line is
reported when enough collinear pixels agree, which is what makes the method
tolerant of the gaps in a dashed centre line and of the pixels an occluding
vehicle removes.

Segments are then sorted into a left and a right boundary by the sign of their
slope, because in a forward facing view the left boundary recedes up and to the
right and the right boundary recedes up and to the left. Nearly horizontal
segments are discarded: they belong to shadows lying across the road, to the
horizon, and to the backs of vehicles, none of which is a lane boundary. Each
side is then reduced to a single straight line by a least squares fit weighted
by segment length, so that long confident segments outvote short ones.

The straight line model is the weak point of the whole stage and the report
does not hide it: on a curve the true boundary is not a line, and Section 6
measures exactly how much accuracy that costs.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_ABS_SLOPE = 0.35        # below this a segment is treated as horizontal
DEFAULT_HOUGH = dict(rho=1, theta=np.pi / 180, threshold=40,
                     min_line_len=40, max_line_gap=120)


def hough_segments(edge_map, params=None):
    """Return the raw probabilistic Hough segments as an (N, 4) array."""
    p = dict(DEFAULT_HOUGH, **(params or {}))
    lines = cv2.HoughLinesP(edge_map, p["rho"], p["theta"], p["threshold"],
                            minLineLength=p["min_line_len"],
                            maxLineGap=p["max_line_gap"])
    if lines is None:
        return np.zeros((0, 4), np.int32)
    return lines.reshape(-1, 4)


def split_by_slope(segments, width):
    """Split segments into left boundary, right boundary and rejected."""
    left, right, rejected = [], [], []
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            rejected.append((x1, y1, x2, y2))
            continue
        slope = (y2 - y1) / float(x2 - x1)
        if abs(slope) < MIN_ABS_SLOPE:
            rejected.append((x1, y1, x2, y2))
        elif slope < 0 and min(x1, x2) < 0.95 * width:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and max(x1, x2) > 0.05 * width:
            right.append((x1, y1, x2, y2))
        else:
            rejected.append((x1, y1, x2, y2))
    return left, right, rejected


def fit_boundary(segments, shape, y_top_frac=0.58):
    """Fit one straight boundary to a set of segments.

    Each segment contributes its two endpoints, weighted by its length, and a
    line x = m*y + c is fitted by weighted least squares. Fitting x as a
    function of y rather than the other way round keeps the fit stable for the
    near vertical boundaries that matter here.
    """
    # One stray segment is not evidence of a boundary. Requiring at least two
    # stops the fit from extrapolating a whole boundary out of a single edge on
    # a roadside bush.
    if len(segments) < 2:
        return None
    h, w = shape[0], shape[1]
    xs, ys, ws = [], [], []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        xs += [x1, x2]
        ys += [y1, y2]
        ws += [length, length]
    xs, ys, ws = np.array(xs, float), np.array(ys, float), np.array(ws, float)
    if len(xs) < 2 or ys.max() - ys.min() < 5:
        return None
    A = np.stack([ys, np.ones_like(ys)], axis=1)
    W = np.diag(ws)
    try:
        coef = np.linalg.lstsq(W @ A, W @ xs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    m, c = coef
    y_bot, y_top = h - 1, int(y_top_frac * h)
    x_bot, x_top = m * y_bot + c, m * y_top + c
    # A boundary that leaves the frame by more than half its width is not a
    # boundary, it is an artefact of extrapolating a short segment. Reporting
    # nothing is more useful than reporting that.
    if not (-0.5 * w <= x_bot <= 1.5 * w and -0.5 * w <= x_top <= 1.5 * w):
        return None
    return (int(x_bot), y_bot, int(x_top), y_top)


def detect_lanes(edge_map, shape, params=None, y_top_frac=0.58, restrict=None):
    """Full boundary identification stage.

    `restrict` optionally gates the edge map, for example with the dilated
    paint mask, so that only edges lying on road markings can vote.
    """
    if restrict is not None:
        edge_map = cv2.bitwise_and(edge_map, restrict)
    segs = hough_segments(edge_map, params)
    left, right, rejected = split_by_slope(segs, shape[1])
    return {
        "segments": segs,
        "left_segments": left,
        "right_segments": right,
        "rejected": rejected,
        "left": fit_boundary(left, shape, y_top_frac),
        "right": fit_boundary(right, shape, y_top_frac),
    }


def boundary_error(fitted, gt_mask, side, rows=12):
    """Mean horizontal error of a fitted boundary, in pixels.

    The ground truth boundary at a given row is the leftmost or rightmost road
    pixel in that row of the manual mask. The fitted line is evaluated at the
    same rows and the mean absolute difference is reported. Rows where the mask
    has no road are skipped. Returns None when nothing could be compared.
    """
    if fitted is None:
        return None
    h, w = gt_mask.shape[:2]
    x1, y1, x2, y2 = fitted
    if y1 == y2:
        return None
    m = (x2 - x1) / float(y2 - y1)
    errs = []
    for yf in np.linspace(0.55, 0.95, rows):
        y = int(yf * (h - 1))
        cols = np.where(gt_mask[y] > 0)[0]
        if len(cols) < 5:
            continue
        # A road that runs out of the frame has no visible boundary on that
        # side in that row, so there is nothing to compare the fit against.
        if side == "left" and cols.min() <= 2:
            continue
        if side == "right" and cols.max() >= w - 3:
            continue
        gt_x = cols.min() if side == "left" else cols.max()
        errs.append(abs(x1 + m * (y - y1) - gt_x))
    if len(errs) < 3:
        return None
    return float(np.mean(errs))


def draw_lanes(img, result, colour=(0, 0, 255), thickness=6):
    out = img.copy()
    for key in ("left", "right"):
        line = result.get(key)
        if line is not None:
            cv2.line(out, (line[0], line[1]), (line[2], line[3]), colour, thickness)
    return out


def draw_segments(img, result):
    """Colour coded Hough segments: kept left, kept right and rejected."""
    out = img.copy()
    for x1, y1, x2, y2 in result["rejected"]:
        cv2.line(out, (x1, y1), (x2, y2), (120, 120, 120), 2)
    for x1, y1, x2, y2 in result["left_segments"]:
        cv2.line(out, (x1, y1), (x2, y2), (0, 200, 255), 3)
    for x1, y1, x2, y2 in result["right_segments"]:
        cv2.line(out, (x1, y1), (x2, y2), (255, 120, 0), 3)
    return out

# --------------------------------------------------------------------------
# Lane marking isolation
# --------------------------------------------------------------------------
def paint_mask(bgr, white_l=170, white_s=90, yellow_lo=(15, 70, 70),
               yellow_hi=(38, 255, 255), dilate=9):
    """Isolate white and yellow road paint in HLS.

    Raw edges inside the region of interest are dominated by things that are
    not lane boundaries: the texture of the surface, the shadows of roadside
    trees, vegetation and the backs of vehicles. Road paint, however, is one of
    only two colours, and both are easy to state in HLS. White paint is high
    lightness and low saturation; yellow paint occupies a narrow hue band with
    a decent saturation.

    The result is dilated so that it can be used as a permissive gate on the
    edge map: an edge is kept when it lies on or beside paint. Roads with no
    paint at all return a nearly empty mask, which is the honest answer and is
    why the unsealed scenes fall back on the region derived boundary below.
    """
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    white = ((hls[:, :, 1] >= white_l) & (hls[:, :, 2] <= white_s))
    yellow = cv2.inRange(hls, np.array(yellow_lo, np.uint8),
                         np.array(yellow_hi, np.uint8)) > 0
    m = ((white | yellow).astype(np.uint8)) * 255
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k, iterations=1)
    return m


def boundaries_from_region(road_mask, y_top_frac=0.58, rows=24):
    """Derive the left and right boundaries from the segmented road region.

    For each sampled row the leftmost and rightmost road pixels are taken as
    boundary observations and a straight line is fitted to each side by the
    same weighted least squares used for the Hough segments. Rows where the
    region runs off the side of the frame are skipped, because there the
    boundary is the edge of the photograph and not the edge of the road.

    This path needs no paint and no edges at all, so it is the one that still
    works on the unsealed roads. What it cannot do is find a lane *within* a
    road: it only ever reports the outside of the drivable region.
    """
    h, w = road_mask.shape[:2]
    left_pts, right_pts = [], []
    for yf in np.linspace(y_top_frac, 0.99, rows):
        y = int(yf * (h - 1))
        cols = np.where(road_mask[y] > 0)[0]
        if len(cols) < 10:
            continue
        if cols.min() > 2:
            left_pts.append((int(cols.min()), y))
        if cols.max() < w - 3:
            right_pts.append((int(cols.max()), y))

    def fit(points):
        if len(points) < 4:
            return None
        xs = np.array([p[0] for p in points], float)
        ys = np.array([p[1] for p in points], float)
        if ys.max() - ys.min() < 5:
            return None
        A = np.stack([ys, np.ones_like(ys)], axis=1)
        m, c = np.linalg.lstsq(A, xs, rcond=None)[0]
        y_bot, y_top = h - 1, int(y_top_frac * h)
        return (int(m * y_bot + c), y_bot, int(m * y_top + c), y_top)

    return {"left": fit(left_pts), "right": fit(right_pts),
            "left_points": left_pts, "right_points": right_pts,
            "segments": np.zeros((0, 4), np.int32),
            "left_segments": [], "right_segments": [], "rejected": []}
