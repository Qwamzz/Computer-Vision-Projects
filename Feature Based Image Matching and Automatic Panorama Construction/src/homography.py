"""
homography.py
=============
Stages 4 and 5 of the pipeline: RANSAC outlier rejection and HOMOGRAPHY
estimation.

Everything here is implemented from first principles -- normalised DLT, the
RANSAC loop with adaptive termination, the symmetric transfer error and the
final re-fit.  ``cv2.findHomography`` is used only in :func:`compare_with_opencv`
as an independent reference, to confirm that the hand-written estimator gives
the same answer.

Theory summary
--------------
Two images of a *planar* scene, or two images taken by a camera rotating about
its optical centre, are related by a projective transformation (homography):

        [x']       [h11 h12 h13] [x]
        [y']  ~    [h21 h22 h23] [y]
        [ 1 ]      [h31 h32 h33] [1]

H has 8 degrees of freedom (it is defined only up to scale), and each point
correspondence supplies 2 equations, so 4 correspondences in general position
are the minimum needed.  That is why RANSAC samples exactly 4 points.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Hartley normalisation
# --------------------------------------------------------------------------- #

def normalise_points(pts: np.ndarray):
    """Translate to zero mean and scale so the RMS distance to the origin is
    sqrt(2).

    Why this matters: the DLT design matrix mixes terms of order x*x' (up to
    ~10^6 for pixel coordinates) with terms of order 1.  That spread makes the
    matrix badly conditioned and the SVD solution numerically unreliable.
    Normalisation puts every entry on a comparable scale.  Hartley showed this
    single step is the difference between a usable and an unusable DLT.
    """
    mean = pts.mean(axis=0)
    centred = pts - mean
    rms = np.sqrt((centred ** 2).sum(axis=1).mean())
    scale = np.sqrt(2.0) / rms if rms > 1e-12 else 1.0
    T = np.array([[scale, 0.0, -scale * mean[0]],
                  [0.0, scale, -scale * mean[1]],
                  [0.0, 0.0, 1.0]])
    return centred * scale, T


def to_homogeneous(pts):
    return np.hstack([pts, np.ones((len(pts), 1))])


def apply_homography(H, pts):
    """Map (N,2) points through H, returning (N,2) inhomogeneous points."""
    if len(pts) == 0:
        return pts.copy()
    ph = to_homogeneous(pts) @ H.T
    w = ph[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return ph[:, :2] / w


# --------------------------------------------------------------------------- #
# Direct Linear Transform
# --------------------------------------------------------------------------- #

def dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate H mapping ``src`` -> ``dst`` by the normalised DLT.

    For each correspondence (x, y) -> (x', y') the projective relation
    x' = (h11 x + h12 y + h13) / (h31 x + h32 y + h33) is rearranged into two
    linear equations in the 9 unknowns of h.  Stacking them gives A h = 0, whose
    least-squares solution under ||h|| = 1 is the right singular vector of A
    with the smallest singular value.
    """
    if len(src) < 4:
        raise ValueError("At least 4 correspondences are required")

    src_n, T_src = normalise_points(src)
    dst_n, T_dst = normalise_points(dst)

    n = len(src_n)
    A = np.zeros((2 * n, 9))
    x, y = src_n[:, 0], src_n[:, 1]
    xp, yp = dst_n[:, 0], dst_n[:, 1]

    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1.0
    A[0::2, 6] = x * xp
    A[0::2, 7] = y * xp
    A[0::2, 8] = xp

    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1.0
    A[1::2, 6] = x * yp
    A[1::2, 7] = y * yp
    A[1::2, 8] = yp

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    # Undo the normalisation:  x'_pixels = T_dst^-1 * H_n * T_src * x_pixels
    H = np.linalg.inv(T_dst) @ H_n @ T_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


# --------------------------------------------------------------------------- #
# Error measures
# --------------------------------------------------------------------------- #

def symmetric_transfer_error(H, src, dst):
    """Squared forward + backward transfer error, in squared pixels.

    Using both directions rather than the forward direction alone prevents
    degenerate solutions that collapse many source points onto one destination
    point, which can score well in one direction only.
    """
    fwd = apply_homography(H, src)
    d_fwd = ((fwd - dst) ** 2).sum(axis=1)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    bwd = apply_homography(H_inv, dst)
    d_bwd = ((bwd - src) ** 2).sum(axis=1)
    return d_fwd + d_bwd


def _is_degenerate(pts, tol=1e-6):
    """Reject a 4-point sample in which any three points are collinear.

    Three collinear points cannot constrain a projective transformation, so such
    a sample yields a rank-deficient system and a meaningless H.
    """
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                a, b, c = pts[i], pts[j], pts[k]
                area = abs((b[0] - a[0]) * (c[1] - a[1]) -
                           (b[1] - a[1]) * (c[0] - a[0]))
                if area < tol:
                    return True
    return False


# --------------------------------------------------------------------------- #
# RANSAC
# --------------------------------------------------------------------------- #

def ransac_homography(src, dst, threshold=3.0, confidence=0.999,
                      max_iters=5000, min_iters=50, seed=0, refine_rounds=5):
    """RANdom SAmple Consensus estimation of the homography.

    Parameters
    ----------
    threshold : float
        Inlier threshold in pixels, applied to the symmetric transfer distance.
    confidence : float
        Desired probability that at least one all-inlier sample is drawn.  Used
        for the adaptive stopping rule
        ``N = log(1 - p) / log(1 - w^s)`` with s = 4.
    refine_rounds : int
        After the best consensus set is found, H is re-fitted on all inliers and
        the inlier set is recomputed, a few times.  This matters: the H from the
        4-point minimal sample is only as accurate as those 4 points, whereas
        the re-fit uses every inlier and is a proper least-squares estimate.
    """
    t0 = time.perf_counter()
    n = len(src)
    result = {"H": None, "inliers": np.zeros(n, dtype=bool), "n_inliers": 0,
              "inlier_ratio": 0.0, "iterations": 0, "ransac_ms": 0.0,
              "mean_inlier_error": float("nan"), "success": False}

    if n < 4:
        result["ransac_ms"] = (time.perf_counter() - t0) * 1e3
        return result

    rng = np.random.default_rng(seed)
    thresh_sq = 2.0 * threshold ** 2   # symmetric error sums two squared terms

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_err = np.inf

    iters = max_iters
    i = 0
    while i < min(iters, max_iters) or i < min_iters:
        i += 1
        idx = rng.choice(n, size=4, replace=False)
        s4, d4 = src[idx], dst[idx]
        if _is_degenerate(s4) or _is_degenerate(d4):
            continue
        try:
            H = dlt_homography(s4, d4)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(H)):
            continue

        err = symmetric_transfer_error(H, src, dst)
        inliers = err < thresh_sq
        count = int(inliers.sum())
        err_sum = float(err[inliers].sum()) if count else np.inf

        # Prefer more inliers; break ties on total inlier error.
        if count > best_count or (count == best_count and err_sum < best_err):
            best_count, best_inliers, best_err = count, inliers, err_sum

            # Adaptive termination: update the estimated inlier ratio and with
            # it the number of samples still required.
            w = max(best_count / n, 1e-6)
            denom = np.log(max(1.0 - w ** 4, 1e-12))
            if denom < 0:
                iters = int(np.ceil(np.log(max(1.0 - confidence, 1e-12)) / denom))
                iters = max(min_iters, min(iters, max_iters))

    if best_count < 4:
        result["iterations"] = i
        result["ransac_ms"] = (time.perf_counter() - t0) * 1e3
        return result

    # --- Least-squares re-fit on the consensus set -------------------------- #
    inliers = best_inliers
    H = dlt_homography(src[inliers], dst[inliers])
    for _ in range(refine_rounds):
        err = symmetric_transfer_error(H, src, dst)
        new_inliers = err < thresh_sq
        if int(new_inliers.sum()) < 4 or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
        H = dlt_homography(src[inliers], dst[inliers])

    err = symmetric_transfer_error(H, src, dst)
    inliers = err < thresh_sq
    if int(inliers.sum()) >= 4:
        H = dlt_homography(src[inliers], dst[inliers])

    result.update({
        "H": H,
        "inliers": inliers,
        "n_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.sum()) / n,
        "iterations": i,
        "ransac_ms": (time.perf_counter() - t0) * 1e3,
        "mean_inlier_error": float(np.sqrt(err[inliers].mean() / 2.0))
        if inliers.any() else float("nan"),
        "success": True,
    })
    return result


# --------------------------------------------------------------------------- #
# Accuracy against ground truth
# --------------------------------------------------------------------------- #

def homography_grid_error(H_est, H_gt, width, height, grid=12):
    """Mean displacement (pixels) between H_est and H_gt over a grid of points.

    Comparing matrices entry by entry is meaningless because H is defined only
    up to scale and because equal changes in different entries have very unequal
    geometric effect.  Mapping a dense grid of image points through both
    matrices and measuring how far apart the results land gives an error with
    direct physical meaning: "the estimated warp misplaces a typical pixel by
    this many pixels".
    """
    if H_est is None or H_gt is None:
        return float("nan")
    xs = np.linspace(0, width - 1, grid)
    ys = np.linspace(0, height - 1, grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    a = apply_homography(H_est, pts)
    b = apply_homography(H_gt, pts)
    d = np.sqrt(((a - b) ** 2).sum(axis=1))
    return float(np.mean(d))


def corner_error(H_est, H_gt, width, height):
    """Same idea as above but using the four image corners only."""
    if H_est is None or H_gt is None:
        return float("nan")
    pts = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    a = apply_homography(H_est, pts)
    b = apply_homography(H_gt, pts)
    return float(np.mean(np.sqrt(((a - b) ** 2).sum(axis=1))))


def compare_with_opencv(src, dst, threshold=3.0):
    """Reference estimate from ``cv2.findHomography`` for validation."""
    if len(src) < 4:
        return None, 0
    H, mask = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32),
                                 cv2.RANSAC, threshold, maxIters=5000,
                                 confidence=0.999)
    n_in = int(mask.sum()) if mask is not None else 0
    return H, n_in


# --------------------------------------------------------------------------- #
# Least-squares fit with NO outlier rejection (for the before/after comparison)
# --------------------------------------------------------------------------- #

def least_squares_homography(src, dst):
    """Fit H to *all* correspondences, outliers included.

    This is the "before RANSAC" condition required by Task 11.  Because the DLT
    minimises an algebraic error over every correspondence, a single gross
    outlier can drag the whole solution away from the truth -- least squares has
    a breakdown point of 0%.
    """
    if len(src) < 4:
        return None
    try:
        return dlt_homography(src, dst)
    except (np.linalg.LinAlgError, ValueError):
        return None
