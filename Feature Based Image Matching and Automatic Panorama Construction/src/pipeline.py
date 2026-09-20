"""
pipeline.py
===========
End-to-end wiring of the seven stages required by the question:

    Feature Detection -> Feature Description -> Feature Matching -> RANSAC
        -> Homography -> Image Alignment -> Panorama

:func:`match_pair` performs stages 1-5 on one image pair and returns every
intermediate quantity needed for the report tables.  :func:`build_panorama`
applies it to a sequence of overlapping images and produces the final mosaic.
"""

from __future__ import annotations

import time

import numpy as np

import dataset
import features
import homography as hg
import matching
import stitching


# --------------------------------------------------------------------------- #
# One image pair
# --------------------------------------------------------------------------- #

def match_pair(img_src, img_dst, detector="SIFT", ratio=0.75, ransac_thresh=3.0,
               n_features=4000, H_gt=None, cross_check=True, seed=0,
               verify_matcher=False):
    """Run detection, description, matching and RANSAC on one pair.

    ``H_gt`` is the ground-truth homography mapping ``img_src`` into
    ``img_dst``'s frame, when it is known; it is used only for evaluation and
    never for estimation.

    Returns a dictionary holding the estimated homography, the keypoints, the
    putative matches, the inlier mask and all timing/accuracy statistics.
    """
    t_start = time.perf_counter()

    gray_src = dataset.preprocess(img_src)
    gray_dst = dataset.preprocess(img_dst)

    kps1, desc1, t1, kind = features.detect_and_describe(gray_src, detector, n_features)
    kps2, desc2, t2, _ = features.detect_and_describe(gray_dst, detector, n_features)

    matches, minfo = matching.match_descriptors(desc1, desc2, kind, ratio=ratio,
                                                cross_check=cross_check)
    p_src, p_dst = matching.matched_points(kps1, kps2, matches)

    rres = hg.ransac_homography(p_src, p_dst, threshold=ransac_thresh, seed=seed)

    # "Before RANSAC" baseline: plain least squares over every putative match.
    H_ls = hg.least_squares_homography(p_src, p_dst)

    h_src, w_src = img_src.shape[:2]
    total_ms = (time.perf_counter() - t_start) * 1e3

    out = {
        "detector": detector,
        "kps1": kps1, "kps2": kps2,
        "desc_kind": kind,
        "n_kp1": len(kps1), "n_kp2": len(kps2),
        "mean_keypoints": 0.5 * (len(kps1) + len(kps2)),
        "kp_stats1": features.keypoint_statistics(kps1),
        "matches": matches,
        "p_src": p_src, "p_dst": p_dst,
        "n_raw_nn": minfo["n_raw"],
        "n_after_ratio": minfo["n_after_ratio"],
        "n_matches": len(matches),
        "H": rres["H"],
        "H_least_squares": H_ls,
        "inliers": rres["inliers"],
        "n_inliers": rres["n_inliers"],
        "inlier_ratio": rres["inlier_ratio"],
        "ransac_iterations": rres["iterations"],
        "mean_inlier_error_px": rres["mean_inlier_error"],
        "success": rres["success"],
        # timings
        "detect_ms": t1["detect_ms"] + t2["detect_ms"],
        "describe_ms": t1["describe_ms"] + t2["describe_ms"],
        "match_ms": minfo["match_ms"],
        "ransac_ms": rres["ransac_ms"],
        "total_time_ms": total_ms,
    }

    # --- Accuracy against ground truth ------------------------------------- #
    if H_gt is not None:
        out["grid_error_px"] = hg.homography_grid_error(rres["H"], H_gt, w_src, h_src)
        out["corner_error_px"] = hg.corner_error(rres["H"], H_gt, w_src, h_src)
        out["grid_error_no_ransac_px"] = hg.homography_grid_error(H_ls, H_gt,
                                                                 w_src, h_src)
        # A correspondence is "truly correct" if the ground-truth homography maps
        # its source point to within 3 px of its destination point.  This gives a
        # genuine precision/recall for the matching stage, independent of RANSAC.
        if len(p_src):
            gt_proj = hg.apply_homography(H_gt, p_src)
            gt_err = np.sqrt(((gt_proj - p_dst) ** 2).sum(axis=1))
            true_correct = gt_err < 3.0
            out["n_true_correct"] = int(true_correct.sum())
            out["match_precision"] = float(true_correct.mean())
            inl = rres["inliers"]
            if inl.any():
                out["ransac_precision"] = float(true_correct[inl].mean())
                out["ransac_recall"] = (float(true_correct[inl].sum() /
                                              max(true_correct.sum(), 1)))
            else:
                out["ransac_precision"] = float("nan")
                out["ransac_recall"] = 0.0
    else:
        for k in ("grid_error_px", "corner_error_px", "grid_error_no_ransac_px",
                  "match_precision", "ransac_precision", "ransac_recall"):
            out[k] = float("nan")
        out["n_true_correct"] = -1

    if verify_matcher:
        out["matcher_agreement"] = matching.verify_against_opencv(desc1, desc2,
                                                                  kind, ratio)
        H_cv, n_cv = hg.compare_with_opencv(p_src, p_dst, ransac_thresh)
        out["opencv_inliers"] = n_cv
        out["opencv_vs_ours_px"] = hg.homography_grid_error(rres["H"], H_cv,
                                                            w_src, h_src)
    return out


# --------------------------------------------------------------------------- #
# Multi-image panorama
# --------------------------------------------------------------------------- #

def build_panorama(images, detector="SIFT", ratio=0.75, ransac_thresh=3.0,
                   n_features=4000, blend="feather", ref_idx=None,
                   gt_pairs=None, seed=0, crop=True):
    """Stitch a left-to-right ordered sequence of overlapping images.

    Consecutive pairs are matched (image k+1 into image k), the resulting
    homographies are chained into a common reference frame, and all images are
    warped onto one canvas and blended.
    """
    n = len(images)
    if ref_idx is None:
        ref_idx = n // 2          # middle image: shortest chains, least distortion

    pair_results, pair_H = [], []
    for k in range(n - 1):
        H_gt = gt_pairs[k] if gt_pairs else None
        # map image k+1 into image k
        r = match_pair(images[k + 1], images[k], detector=detector, ratio=ratio,
                       ransac_thresh=ransac_thresh, n_features=n_features,
                       H_gt=H_gt, seed=seed)
        pair_results.append(r)
        pair_H.append(r["H"])

    H_to_ref = stitching.chain_homographies(pair_H, n, ref_idx)

    t0 = time.perf_counter()
    size, T = stitching.canvas_geometry(images, H_to_ref)
    panorama, masks = stitching.warp_all(images, H_to_ref, size, T, blend=blend)
    warp_ms = (time.perf_counter() - t0) * 1e3

    quality = stitching.overlap_quality(images, H_to_ref, size, T)
    summary = stitching.summarise_quality(quality)

    if crop:
        panorama = stitching.crop_to_content(panorama)

    return {
        "panorama": panorama,
        "pair_results": pair_results,
        "H_to_ref": H_to_ref,
        "canvas_size": size,
        "T": T,
        "ref_idx": ref_idx,
        "warp_ms": warp_ms,
        "quality_pairs": quality,
        "quality": summary,
        "detector": detector,
        "blend": blend,
    }


def aggregate_pair_results(pair_results):
    """Average the per-pair statistics into the single row used in Table 1."""
    def m(key):
        vals = [r.get(key, np.nan) for r in pair_results]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def s(key):
        return float(np.sum([r.get(key, 0) or 0 for r in pair_results]))

    return {
        "mean_keypoints": m("mean_keypoints"),
        "mean_matches": m("n_matches"),
        "mean_inliers": m("n_inliers"),
        "mean_inlier_ratio": m("inlier_ratio"),
        "mean_grid_error_px": m("grid_error_px"),
        "mean_grid_error_no_ransac_px": m("grid_error_no_ransac_px"),
        "mean_inlier_error_px": m("mean_inlier_error_px"),
        "mean_match_precision": m("match_precision"),
        "mean_ransac_precision": m("ransac_precision"),
        "mean_ransac_recall": m("ransac_recall"),
        "detect_ms": s("detect_ms"),
        "describe_ms": s("describe_ms"),
        "match_ms": s("match_ms"),
        "ransac_ms": s("ransac_ms"),
        "total_time_ms": s("total_time_ms"),
        "mean_ransac_iterations": m("ransac_iterations"),
    }
