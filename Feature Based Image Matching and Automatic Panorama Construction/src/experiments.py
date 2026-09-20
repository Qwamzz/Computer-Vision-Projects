"""
experiments.py
==============
The experimental programme required by the question.

E1  Detector / descriptor comparison on the full panorama sequence
    (keypoints, putative matches, RANSAC inliers, inlier ratio, processing time,
    panorama quality).
E2  Effect of RANSAC: matching results before and after outlier rejection.
E3  Robustness to rotation, scale, viewpoint and illumination.
E4  Effect of the blending strategy on the final mosaic.
E5  Parameter sensitivity: Lowe ratio and RANSAC inlier threshold.

Every experiment returns plain dictionaries/lists so that :mod:`main` can dump
them to JSON and render them as Markdown tables.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

import dataset
import features
import homography as hg
import pipeline
import stitching
import visualisation as vis

DETECTORS = features.DETECTOR_NAMES


def _gt_pairs(vs):
    """Ground-truth homographies for each consecutive pair (k+1 -> k)."""
    if not vs.synthetic:
        return None
    return [vs.gt_homography(k + 1, k) for k in range(len(vs) - 1)]


# --------------------------------------------------------------------------- #
# E1 -- detector comparison
# --------------------------------------------------------------------------- #

def experiment_detector_comparison(vs, out_dir, n_features=4000, ratio=0.75,
                                   ransac_thresh=3.0):
    """Run the complete pipeline once per detector and tabulate the results."""
    print("\n=== E1: detector / descriptor comparison ===")
    os.makedirs(out_dir, exist_ok=True)
    gt = _gt_pairs(vs)
    rows, panoramas = [], {}

    for det in DETECTORS:
        print("  [E1] %s ..." % det, flush=True)
        res = pipeline.build_panorama(vs.images, detector=det, ratio=ratio,
                                      ransac_thresh=ransac_thresh,
                                      n_features=n_features, blend="feather",
                                      gt_pairs=gt)
        agg = pipeline.aggregate_pair_results(res["pair_results"])
        agg.update({
            "detector": det,
            "descriptor_kind": res["pair_results"][0]["desc_kind"],
            "warp_ms": res["warp_ms"],
            "panorama_w": res["panorama"].shape[1],
            "panorama_h": res["panorama"].shape[0],
        })
        agg.update({"quality_" + k: v for k, v in res["quality"].items()})
        rows.append(agg)
        panoramas[det] = res["panorama"]

        safe = det.replace("+", "_")
        vis.save(res["panorama"], os.path.join(out_dir, "panorama_%s.png" % safe))

        # Keypoint illustration on the first view.
        gray = dataset.preprocess(vs.images[0])
        kps, _, _, _ = features.detect_and_describe(gray, det, n_features)
        vis.draw_keypoints(vs.images[0], kps,
                           os.path.join(out_dir, "keypoints_%s.png" % safe))

        # Before/after RANSAC figure for the first pair.
        pr = res["pair_results"][0]
        vis.draw_matches_before_after(
            vs.images[1], pr["kps1"], vs.images[0], pr["kps2"],
            pr["matches"], pr["inliers"],
            os.path.join(out_dir, "matches_%s.png" % safe), detector_name=det)

    vis.plot_detector_summary(rows, os.path.join(out_dir, "plot_detector_summary.png"))
    return rows, panoramas


# --------------------------------------------------------------------------- #
# E2 -- effect of RANSAC
# --------------------------------------------------------------------------- #

def experiment_ransac_effect(vs, out_dir, pair_idx=0, n_features=4000):
    """Quantify what RANSAC contributes, using the known ground truth.

    Three things are measured for each detector:

    * how many putative matches are actually correct (precision *before*),
    * how many of RANSAC's inliers are correct (precision *after*), and
    * the alignment error of a homography fitted with and without RANSAC.

    The third is the decisive number: least squares over all putative matches
    has a 0% breakdown point, so even a handful of gross outliers destroys it.
    """
    print("\n=== E2: effect of RANSAC ===")
    os.makedirs(out_dir, exist_ok=True)
    gt = vs.gt_homography(pair_idx + 1, pair_idx) if vs.synthetic else None
    rows = []

    for det in DETECTORS:
        r = pipeline.match_pair(vs.images[pair_idx + 1], vs.images[pair_idx],
                                detector=det, n_features=n_features, H_gt=gt,
                                verify_matcher=True)
        rows.append({
            "detector": det,
            "n_putative": r["n_matches"],
            "n_true_correct": r["n_true_correct"],
            "n_inliers": r["n_inliers"],
            "n_removed": r["n_matches"] - r["n_inliers"],
            "inlier_ratio": r["inlier_ratio"],
            "precision_before": r["match_precision"],
            "precision_after": r["ransac_precision"],
            "recall_after": r["ransac_recall"],
            "error_no_ransac_px": r["grid_error_no_ransac_px"],
            "error_with_ransac_px": r["grid_error_px"],
            "mean_inlier_error_px": r["mean_inlier_error_px"],
            "ransac_iterations": r["ransac_iterations"],
            "matcher_agreement_with_opencv": r.get("matcher_agreement", float("nan")),
            "opencv_inliers": r.get("opencv_inliers", -1),
            "our_H_vs_opencv_H_px": r.get("opencv_vs_ours_px", float("nan")),
        })

        # Visual proof of alignment: warp view k+1 into view k and compare.
        if r["H"] is not None:
            h, w = vs.images[pair_idx].shape[:2]
            warped = cv2.warpPerspective(vs.images[pair_idx + 1], r["H"], (w, h))
            safe = det.replace("+", "_")
            vis.save(vis.checkerboard_overlay(vs.images[pair_idx], warped),
                     os.path.join(out_dir, "alignment_checker_%s.png" % safe))
            vis.difference_map(vs.images[pair_idx], warped,
                               os.path.join(out_dir, "alignment_diff_%s.png" % safe))
    return rows


# --------------------------------------------------------------------------- #
# E3 -- robustness study
# --------------------------------------------------------------------------- #

def _run_controlled(scene, det, n_features, **kw):
    """One controlled-pair trial, returning the metrics of interest."""
    pair = dataset.make_controlled_pair(scene, **kw)
    gt = pair.gt_homography(1, 0)
    r = pipeline.match_pair(pair.images[1], pair.images[0], detector=det,
                            n_features=n_features, H_gt=gt)
    return {
        "detector": det,
        "n_keypoints": r["mean_keypoints"],
        "n_matches": r["n_matches"],
        "n_inliers": r["n_inliers"],
        "inlier_ratio": r["inlier_ratio"],
        "grid_error_px": r["grid_error_px"],
        "match_precision": r["match_precision"],
        "total_time_ms": r["total_time_ms"],
        # A trial "succeeds" only if enough inliers survive AND the resulting
        # warp is accurate; a homography from 8 spurious inliers is worthless.
        "usable": bool(r["n_inliers"] >= 12 and np.isfinite(r["grid_error_px"])
                       and r["grid_error_px"] < 5.0),
    }


def experiment_robustness(scene, out_dir, n_features=4000):
    """Vary one imaging factor at a time and record how matching degrades."""
    print("\n=== E3: robustness to rotation / scale / viewpoint / illumination ===")
    os.makedirs(out_dir, exist_ok=True)

    rotations = [0, 10, 20, 30, 45, 60, 90, 135, 180]
    zooms = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    tilts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    gammas = [0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.8]

    out = {"rotation": [], "scale": [], "viewpoint": [], "illumination": []}

    for det in DETECTORS:
        print("  [E3] %s" % det, flush=True)
        for a in rotations:
            row = _run_controlled(scene, det, n_features, rotation=a)
            row["rotation_deg"] = a
            out["rotation"].append(row)
        for z in zooms:
            row = _run_controlled(scene, det, n_features, scale=z)
            row["zoom_factor"] = z
            out["scale"].append(row)
        for t in tilts:
            row = _run_controlled(scene, det, n_features, tilt=t)
            row["tilt"] = t
            out["viewpoint"].append(row)
        for g in gammas:
            row = _run_controlled(scene, det, n_features, gamma=g)
            row["gamma"] = g
            out["illumination"].append(row)

    vis.plot_robustness(out["rotation"], "rotation_deg", "In-plane rotation (deg)",
                        os.path.join(out_dir, "plot_rotation.png"),
                        "Robustness to in-plane rotation")
    vis.plot_robustness(out["scale"], "zoom_factor", "Zoom factor",
                        os.path.join(out_dir, "plot_scale.png"),
                        "Robustness to scale change")
    vis.plot_robustness(out["viewpoint"], "tilt", "Perspective tilt",
                        os.path.join(out_dir, "plot_viewpoint.png"),
                        "Robustness to viewpoint change")
    vis.plot_robustness(out["illumination"], "gamma", "Gamma",
                        os.path.join(out_dir, "plot_illumination.png"),
                        "Robustness to illumination change")

    # Example images of the four conditions, for the report.
    examples = [
        ("rotation 45 deg", dict(rotation=45)),
        ("zoom x2.0", dict(scale=2.0)),
        ("tilt 0.5", dict(tilt=0.5)),
        ("gamma 2.8 (dark)", dict(gamma=2.8)),
        ("gain 0.4 (under-exposed)", dict(gain=0.4)),
        ("gain 1.6 + bias 40 (over-exposed)", dict(gain=1.6, bias=40)),
    ]
    imgs, titles = [], []
    for name, kw in examples:
        p = dataset.make_controlled_pair(scene, **kw)
        imgs.append(p.images[1])
        titles.append(name)
    vis.montage(imgs, titles, cols=3,
                path=os.path.join(out_dir, "robustness_conditions.png"))

    # Named illumination conditions (a table rather than a sweep).
    named = [("reference", {}),
             ("under-exposed (gain 0.4)", dict(gain=0.4)),
             ("over-exposed (gain 1.6, bias 40)", dict(gain=1.6, bias=40)),
             ("low contrast (gain 0.45, bias 90)", dict(gain=0.45, bias=90)),
             ("gamma 0.4", dict(gamma=0.4)),
             ("gamma 2.8", dict(gamma=2.8)),
             ("heavy noise (sigma 18)", dict(noise_sigma=18.0))]
    illum_rows = []
    for det in DETECTORS:
        for name, kw in named:
            row = _run_controlled(scene, det, n_features, **kw)
            row["condition"] = name
            illum_rows.append(row)
    out["illumination_named"] = illum_rows
    return out


# --------------------------------------------------------------------------- #
# E4 -- blending
# --------------------------------------------------------------------------- #

def experiment_blending(vs, out_dir, detector="SIFT", n_features=4000):
    """Compare hard overwrite against distance-weighted feathering."""
    print("\n=== E4: blending strategy ===")
    os.makedirs(out_dir, exist_ok=True)
    gt = _gt_pairs(vs)
    rows = []
    for blend in ("overwrite", "feather"):
        res = pipeline.build_panorama(vs.images, detector=detector,
                                      n_features=n_features, blend=blend,
                                      gt_pairs=gt)
        vis.save(res["panorama"], os.path.join(out_dir, "panorama_blend_%s.png" % blend))
        rows.append({"blend": blend, "warp_ms": res["warp_ms"],
                     **{"quality_" + k: v for k, v in res["quality"].items()}})
    return rows


# --------------------------------------------------------------------------- #
# E5 -- parameter sensitivity
# --------------------------------------------------------------------------- #

def experiment_parameters(vs, out_dir, pair_idx=0, n_features=4000):
    """Sweep the two parameters that most affect the outcome."""
    print("\n=== E5: parameter sensitivity ===")
    os.makedirs(out_dir, exist_ok=True)
    gt = vs.gt_homography(pair_idx + 1, pair_idx) if vs.synthetic else None

    ratio_rows = []
    for ratio in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        for det in DETECTORS:
            r = pipeline.match_pair(vs.images[pair_idx + 1], vs.images[pair_idx],
                                    detector=det, ratio=ratio,
                                    n_features=n_features, H_gt=gt)
            ratio_rows.append({"detector": det, "lowe_ratio": ratio,
                               "n_matches": r["n_matches"],
                               "n_inliers": r["n_inliers"],
                               "inlier_ratio": r["inlier_ratio"],
                               "match_precision": r["match_precision"],
                               "grid_error_px": r["grid_error_px"]})

    thresh_rows = []
    for th in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        for det in DETECTORS:
            r = pipeline.match_pair(vs.images[pair_idx + 1], vs.images[pair_idx],
                                    detector=det, ransac_thresh=th,
                                    n_features=n_features, H_gt=gt)
            thresh_rows.append({"detector": det, "ransac_threshold_px": th,
                                "n_inliers": r["n_inliers"],
                                "inlier_ratio": r["inlier_ratio"],
                                "grid_error_px": r["grid_error_px"],
                                "ransac_iterations": r["ransac_iterations"]})

    vis.plot_robustness(ratio_rows, "lowe_ratio", "Lowe ratio threshold",
                        os.path.join(out_dir, "plot_lowe_ratio.png"),
                        "Effect of the Lowe ratio threshold",
                        y_keys=(("n_matches", "Putative matches"),
                                ("match_precision", "Fraction actually correct"),
                                ("grid_error_px", "Alignment error (px)")))
    vis.plot_robustness(thresh_rows, "ransac_threshold_px", "RANSAC threshold (px)",
                        os.path.join(out_dir, "plot_ransac_threshold.png"),
                        "Effect of the RANSAC inlier threshold",
                        y_keys=(("n_inliers", "Inliers"),
                                ("inlier_ratio", "Inlier ratio"),
                                ("grid_error_px", "Alignment error (px)")))
    return {"lowe_ratio": ratio_rows, "ransac_threshold": thresh_rows}
