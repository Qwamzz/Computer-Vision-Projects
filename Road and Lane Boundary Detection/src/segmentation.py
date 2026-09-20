"""
segmentation.py
Separating the drivable road region from the rest of the scene.

Three classical approaches are implemented and compared. All three are given
the same input and the same post processing, so the comparison isolates the
segmentation rule itself.

1. Adaptive colour thresholding. A sample of pixels is taken from a small
   trapezoid immediately in front of the camera, which is the one place a
   forward facing photograph is almost certain to be road. The mean and spread
   of that sample in HLS define a per image acceptance band, and every pixel
   within it is called road. This is thresholding, but with the threshold set
   from the image rather than fixed in advance, which is what lets one rule
   survive both grey asphalt and orange laterite.

2. K means clustering in Lab. The image is clustered into K colours and the
   cluster that dominates the same near field sample is taken to be the road.
   No threshold is chosen at all: the data decides where the boundaries lie.
   The cost is that K must be chosen, and that a cluster can straddle road and
   shoulder when the two are similar.

3. Mahalanobis thresholding in Lab. The same near field sample is used, but
   instead of an independent band per channel the full 3x3 covariance of the
   sample is estimated and a pixel is accepted when its Mahalanobis distance
   from the sample mean is below a threshold. Treating the channels
   independently, as approach 1 does, implicitly assumes the acceptance region
   is an axis aligned box; in practice the road colours form a tilted,
   elongated cloud, and a box that covers it must also admit a great deal that
   is not road. This is the same idea as approach 1 with the correlation put
   back in, and experiment E6 shows it is worth a large amount of accuracy.

4. Region growing. A flood fill grows outwards from seeds in the near field,
   accepting a neighbour when it is within a tolerance of the seed region.
   Unlike the other two this enforces spatial connectivity, so a patch of grey
   roof on the far side of a hedge cannot be labelled road. That is its main
   advantage and, under a shadow that cuts the road in half, its main weakness.

All three are followed by the same morphological cleanup and by keeping the
connected component that touches the bottom of the frame, since the drivable
region must by definition reach the vehicle.
"""

from __future__ import annotations

import cv2
import numpy as np

METHODS = ["threshold", "mahalanobis", "kmeans", "region_growing"]


def near_field_sample(shape, width=0.22, height=0.10):
    """A trapezoid at the bottom centre, used as the road colour prior."""
    h, w = shape[:2]
    cx = w // 2
    half = int(width * w / 2)
    y0 = int((1.0 - height) * h)
    return np.array([[(cx - half, h - 1), (cx - int(half * 0.7), y0),
                      (cx + int(half * 0.7), y0), (cx + half, h - 1)]], np.int32)


def _sample_mask(shape, width=0.22, height=0.10):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, near_field_sample(shape, width, height), 255)
    return m


def horizon_cut(mask, frac=0.35):
    """Discard everything above the horizon.

    This is the region selection step applied to segmentation. Sky, distant
    buildings and tree canopy are never drivable, and a surprising amount of
    sky is close enough to grey asphalt in colour to be accepted by any of the
    rules below. Cutting the top of the frame removes that failure mode at the
    cost of one assumption: the camera is roughly level.
    """
    out = mask.copy()
    out[:int(frac * mask.shape[0])] = 0
    return out


def segment_mahalanobis(bgr, chi=3.0, cleanup=11, horizon=0.35, ridge=4.0):
    """Multivariate colour thresholding in Lab.

    The near field sample gives a mean and a full covariance, and a pixel is
    called road when

        sqrt( (x - mu)^T C^-1 (x - mu) )  <=  chi

    which accepts an ellipsoid oriented along the correlations actually present
    in the road colours rather than an axis aligned box. A small ridge is added
    to the diagonal of C so that a very uniform sample, which happens on a
    smooth asphalt surface, cannot produce a near singular covariance and an
    absurdly narrow acceptance region.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0),
                       cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w = bgr.shape[:2]
    sample = _sample_mask(bgr.shape) > 0
    if sample.sum() < 50:
        return np.zeros((h, w), np.uint8)
    X = lab[sample]
    mu = X.mean(axis=0)
    C = np.cov(X.T) + np.eye(3) * ridge
    try:
        Ci = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.zeros((h, w), np.uint8)
    d = lab.reshape(-1, 3) - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d)).reshape(h, w)
    mask = (md <= chi).astype(np.uint8) * 255
    if horizon:
        mask = horizon_cut(mask, horizon)
    return _cleanup(mask, kernel=cleanup)


def _cleanup(mask, keep_bottom=True, kernel=11):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    if not keep_bottom:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    h = mask.shape[0]
    bottom = lab[h - 1, :]
    ids = [i for i in np.unique(bottom) if i != 0]
    if not ids:                       # nothing reaches the camera
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ids = [idx]
    out = np.zeros_like(mask)
    for i in ids:
        out[lab == i] = 255
    return out


def segment_threshold(bgr, k_sigma=2.5, cleanup=11):
    """Adaptive colour band in HLS, set from the near field sample."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    sample = _sample_mask(bgr.shape) > 0
    if sample.sum() < 50:
        return np.zeros(bgr.shape[:2], np.uint8)
    mask = np.ones(bgr.shape[:2], bool)
    for c, spread_floor in ((0, 6.0), (1, 12.0), (2, 10.0)):
        ch = hls[:, :, c]
        vals = ch[sample]
        mu, sd = float(vals.mean()), max(float(vals.std()), spread_floor)
        mask &= np.abs(ch - mu) <= k_sigma * sd
    return _cleanup((mask.astype(np.uint8) * 255), kernel=cleanup)


def segment_kmeans(bgr, k=3, attempts=3, cleanup=11, downscale=0.5):
    """Cluster in Lab and keep the cluster that owns the near field."""
    small = cv2.resize(bgr, None, fx=downscale, fy=downscale,
                       interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(lab, k, None, crit, attempts,
                              cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])
    labels = cv2.resize(labels.astype(np.int32), (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    sample = _sample_mask(bgr.shape) > 0
    if sample.sum() < 50:
        return np.zeros(bgr.shape[:2], np.uint8)
    vals, counts = np.unique(labels[sample], return_counts=True)
    road_label = int(vals[int(np.argmax(counts))])
    return _cleanup(((labels == road_label).astype(np.uint8) * 255), kernel=cleanup)


def segment_region_growing(bgr, tolerance=14, cleanup=11, seeds=5):
    """Flood fill outwards from seeds on the near field road surface."""
    blur = cv2.GaussianBlur(bgr, (7, 7), 0)
    h, w = bgr.shape[:2]
    filled = blur.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    lo = (tolerance,) * 3
    hi = (tolerance,) * 3
    for i in range(seeds):
        fx = 0.5 + (i - (seeds - 1) / 2.0) * 0.04
        px = int(np.clip(fx, 0.02, 0.98) * w)
        py = int(0.96 * h)
        cv2.floodFill(filled, ff_mask, (px, py), (255, 255, 255), lo, hi,
                      cv2.FLOODFILL_FIXED_RANGE | 8 | (255 << 8))
    grown = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
    return _cleanup(grown, kernel=cleanup)


def segment(bgr, method="threshold", **kw):
    if method == "threshold":
        return segment_threshold(bgr, **kw)
    if method == "mahalanobis":
        return segment_mahalanobis(bgr, **kw)
    if method == "kmeans":
        return segment_kmeans(bgr, **kw)
    if method == "region_growing":
        return segment_region_growing(bgr, **kw)
    raise ValueError("unknown segmentation method " + method)


def restrict_to_roi(mask, roi_mask_img):
    """Optionally confine a segmentation to the region of interest."""
    return cv2.bitwise_and(mask, roi_mask_img)


def overlay_result(bgr, road_mask, lanes=None, colour=(0, 200, 0), alpha=0.40):
    """Final visualisation: road region tinted, boundaries drawn over it."""
    out = bgr.copy()
    sel = road_mask > 0
    out[sel] = (np.array(colour) * alpha + out[sel] * (1 - alpha)).astype(np.uint8)
    cnts, _ = cv2.findContours((road_mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 140, 0), 2)
    if lanes:
        for key in ("left", "right"):
            line = lanes.get(key)
            if line is not None:
                cv2.line(out, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 6)
    return out
