"""
test_components.py
==================
Correctness tests for the components implemented from first principles.

Run with:
    python src/test_components.py

These are not decoration.  The DLT, the RANSAC loop and the two distance
metrics are all hand-written, and each has a failure mode that would quietly
degrade the results rather than raise an error:

* an un-normalised DLT still returns a matrix, just an inaccurate one;
* a RANSAC threshold applied to the wrong error quantity still returns inliers,
  just the wrong ones;
* the fast Hamming identity would silently return wrong distances if the
  unpacking were mis-ordered.

Each test therefore checks against an independently known answer.
"""

from __future__ import annotations

import sys
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset
import features
import homography as hg
import matching

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    _results.append((name, status, detail))
    print("  [%s] %-52s %s" % (status, name, detail))
    return condition


# --------------------------------------------------------------------------- #

def test_dlt_exact_recovery():
    """The DLT must recover a known homography from exact correspondences."""
    print("\nDLT homography estimation")
    rng = np.random.default_rng(0)
    H_true = np.array([[1.2, 0.15, 40.0],
                       [-0.1, 0.95, -25.0],
                       [3e-4, -2e-4, 1.0]])
    src = rng.uniform(0, 600, (12, 2))
    dst = hg.apply_homography(H_true, src)

    H_est = hg.dlt_homography(src, dst)
    err = hg.homography_grid_error(H_est, H_true, 640, 480)
    check("recovers a known H from exact points", err < 1e-6,
          "grid error = %.2e px" % err)

    # Minimum case: exactly four correspondences.
    H4 = hg.dlt_homography(src[:4], dst[:4])
    err4 = hg.homography_grid_error(H4, H_true, 640, 480)
    check("recovers a known H from the minimal 4 points", err4 < 1e-6,
          "grid error = %.2e px" % err4)


def test_normalisation_matters():
    """Hartley normalisation must improve conditioning of the design matrix."""
    print("\nHartley normalisation")
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 2000, (30, 2))
    normed, T = hg.normalise_points(pts)

    rms = np.sqrt((normed ** 2).sum(axis=1).mean())
    check("normalised RMS distance to origin is sqrt(2)",
          abs(rms - np.sqrt(2)) < 1e-9, "rms = %.6f" % rms)
    check("normalised centroid is at the origin",
          np.allclose(normed.mean(axis=0), 0, atol=1e-9),
          "centroid = %s" % np.round(normed.mean(axis=0), 12))

    # T must reproduce the normalisation when applied as a homography.
    via_T = hg.apply_homography(T, pts)
    check("T reproduces the normalisation", np.allclose(via_T, normed, atol=1e-9))


def test_ransac_with_outliers():
    """RANSAC must recover H when half the correspondences are garbage."""
    print("\nRANSAC robustness")
    rng = np.random.default_rng(2)
    H_true = np.array([[1.05, 0.08, 30.0],
                       [-0.06, 1.02, -18.0],
                       [1.5e-4, 1.0e-4, 1.0]])

    n_in, n_out = 120, 120
    src_in = rng.uniform(0, 640, (n_in, 2))
    dst_in = hg.apply_homography(H_true, src_in) + rng.normal(0, 0.3, (n_in, 2))

    src_out = rng.uniform(0, 640, (n_out, 2))
    dst_out = rng.uniform(0, 640, (n_out, 2))       # unrelated: pure outliers

    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])
    truth = np.zeros(len(src), dtype=bool)
    truth[:n_in] = True

    res = hg.ransac_homography(src, dst, threshold=3.0, seed=0)
    err = hg.homography_grid_error(res["H"], H_true, 640, 480)

    check("succeeds with 50% outliers", res["success"])
    check("recovers H accurately despite outliers", err < 1.0,
          "grid error = %.4f px" % err)

    tp = int((res["inliers"] & truth).sum())
    fp = int((res["inliers"] & ~truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / n_in
    check("inlier precision > 0.95", precision > 0.95, "precision = %.3f" % precision)
    check("inlier recall > 0.90", recall > 0.90, "recall = %.3f" % recall)

    # The same data fitted without outlier rejection must be far worse: this is
    # the quantitative statement of "least squares has a 0% breakdown point".
    H_ls = hg.least_squares_homography(src, dst)
    err_ls = hg.homography_grid_error(H_ls, H_true, 640, 480)
    check("plain least squares is destroyed by the outliers", err_ls > 10 * max(err, 1e-6),
          "LS error = %.1f px vs RANSAC %.4f px" % (err_ls, err))


def test_degeneracy_rejection():
    """Collinear 4-point samples must be rejected."""
    print("\nDegeneracy detection")
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [5.0, 1.0]])
    general = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    check("detects three collinear points", hg._is_degenerate(collinear))
    check("accepts points in general position", not hg._is_degenerate(general))


def test_hamming_identity():
    """The fast bit-dot-product Hamming must equal the XOR/popcount version."""
    print("\nHamming distance")
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (300, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (400, 32), dtype=np.uint8)

    d_lut = matching._hamming_distance_matrix_lut(a, b).astype(np.float32)
    d_fast = matching._hamming_distance_matrix(a, b)
    check("fast identity matches XOR/popcount exactly",
          np.array_equal(d_lut, d_fast),
          "max |diff| = %g" % np.abs(d_lut - d_fast).max())

    # Cross-check one entry against a completely independent computation.
    manual = int(np.unpackbits(np.bitwise_xor(a[7], b[11])).sum())
    check("agrees with a directly computed entry", abs(d_fast[7, 11] - manual) < 1e-6,
          "computed %g, expected %d" % (d_fast[7, 11], manual))


def test_l2_distance():
    """The expanded L2 formula must equal a direct computation."""
    print("\nL2 distance")
    rng = np.random.default_rng(4)
    a = rng.normal(0, 50, (200, 128)).astype(np.float32)
    b = rng.normal(0, 50, (250, 128)).astype(np.float32)
    fast = matching._l2_distance_matrix(a, b)
    direct = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    check("expanded L2 matches the direct computation",
          np.allclose(fast, direct, atol=1e-2),
          "max |diff| = %.2e" % np.abs(fast - direct).max())


def test_matcher_against_opencv():
    """Our matcher must agree with cv2.BFMatcher on real descriptors."""
    print("\nMatcher validation against OpenCV")
    scene = dataset.build_scene()
    vs = dataset.make_panorama_views(scene, n_views=3)
    g1 = dataset.preprocess(vs.images[0])
    g2 = dataset.preprocess(vs.images[1])

    for cfg in features.DETECTOR_NAMES:
        _, d1, _, kind = features.detect_and_describe(g1, cfg, 1500)
        _, d2, _, _ = features.detect_and_describe(g2, cfg, 1500)
        agree = matching.verify_against_opencv(d1, d2, kind, ratio=0.75)
        check("%-16s matches cv2.BFMatcher" % cfg, agree > 0.99,
              "agreement = %.4f" % agree)


def test_ransac_against_opencv():
    """Our RANSAC homography must agree with cv2.findHomography."""
    print("\nRANSAC validation against OpenCV")
    import pipeline
    scene = dataset.build_scene()
    vs = dataset.make_panorama_views(scene, n_views=3)
    gt = vs.gt_homography(1, 0)

    for cfg in features.DETECTOR_NAMES:
        r = pipeline.match_pair(vs.images[1], vs.images[0], cfg,
                                n_features=2000, H_gt=gt, verify_matcher=True)
        check("%-16s H agrees with cv2.findHomography" % cfg,
              r["opencv_vs_ours_px"] < 1.0,
              "difference = %.4f px" % r["opencv_vs_ours_px"])
        check("%-16s sub-pixel accuracy vs ground truth" % cfg,
              r["grid_error_px"] < 2.0,
              "error = %.4f px" % r["grid_error_px"])


def test_ground_truth_consistency():
    """The synthetic ground truth must actually describe the rendered views."""
    print("\nGround-truth self-consistency")
    scene = dataset.build_scene()
    vs = dataset.make_panorama_views(scene, n_views=3)

    # A point in the scene, pushed into two views, must be related by H_gt.
    rng = np.random.default_rng(5)
    scene_pts = rng.uniform(200, 700, (50, 2))
    p0 = hg.apply_homography(vs.H_scene_to_view[0], scene_pts)
    p1 = hg.apply_homography(vs.H_scene_to_view[1], scene_pts)
    H_gt = vs.gt_homography(0, 1)
    p1_pred = hg.apply_homography(H_gt, p0)
    err = np.sqrt(((p1_pred - p1) ** 2).sum(1)).mean()
    check("gt_homography composes the view homographies correctly", err < 1e-8,
          "mean error = %.2e px" % err)


def test_stitching_geometry():
    """Canvas geometry must contain every warped corner."""
    print("\nStitching geometry")
    import stitching
    imgs = [np.zeros((100, 120, 3), np.uint8) for _ in range(3)]
    H = [np.eye(3),
         np.array([[1.0, 0.0, 90.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]]),
         np.array([[1.0, 0.0, 180.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]])]
    (w, h), T = stitching.canvas_geometry(imgs, H)

    ok = True
    for img, Hi in zip(imgs, H):
        ih, iw = img.shape[:2]
        c = hg.apply_homography(T @ Hi, stitching.image_corners(iw, ih))
        if c.min() < -0.5 or c[:, 0].max() > w + 0.5 or c[:, 1].max() > h + 0.5:
            ok = False
    check("canvas contains all warped corners", ok, "canvas = %dx%d" % (w, h))
    check("translation shifts into the positive quadrant", T[0, 2] >= 0 and T[1, 2] >= 0)


# --------------------------------------------------------------------------- #

def main():
    print("=" * 72)
    print("Component tests")
    print("=" * 72)

    test_dlt_exact_recovery()
    test_normalisation_matters()
    test_ransac_with_outliers()
    test_degeneracy_rejection()
    test_hamming_identity()
    test_l2_distance()
    test_ground_truth_consistency()
    test_stitching_geometry()
    test_matcher_against_opencv()
    test_ransac_against_opencv()

    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print("\n" + "=" * 72)
    print("%d tests, %d passed, %d failed" % (len(_results),
                                              len(_results) - n_fail, n_fail))
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
