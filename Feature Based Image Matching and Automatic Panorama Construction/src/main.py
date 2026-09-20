"""
main.py
=======
Entry point.  Runs the whole project and writes every figure, table and JSON
record into ``results/``.

Usage
-----
    python src/main.py                 # full run (scene photograph, or images/ if populated)
    python src/main.py --quick         # skip the slower sweeps
    python src/main.py --scene         # ignore images/ and use the scene photograph
    python src/main.py --drawn-scene   # force the procedurally drawn fallback scene
    python src/main.py --views 5       # number of views to cut from the scene
    python src/main.py --detector ORB  # detector used for the single-detector outputs
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset            # noqa: E402
import experiments        # noqa: E402
import features           # noqa: E402
import pipeline           # noqa: E402
import visualisation as vis  # noqa: E402

ROOT = os.path.dirname(HERE)
IMAGES_DIR = os.path.join(ROOT, "images")
RESULTS_DIR = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------- #
# Markdown table rendering
# --------------------------------------------------------------------------- #

def _fmt(v):
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        if abs(v) >= 1000:
            return "%.0f" % v
        if abs(v) >= 10:
            return "%.1f" % v
        if abs(v) >= 1:
            return "%.2f" % v
        return "%.3f" % v
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def md_table(rows, columns=None, headers=None):
    """Render a list of dicts as a GitHub-flavoured Markdown table."""
    if not rows:
        return "_(no data)_\n"
    columns = columns or list(rows[0].keys())
    headers = headers or [c.replace("_", " ") for c in columns]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |")
    return "\n".join(out) + "\n"


def section(fh, title, level=2):
    fh.write("\n%s %s\n\n" % ("#" * level, title))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Feature-based panorama construction")
    ap.add_argument("--quick", action="store_true",
                    help="skip the parameter sweeps (faster run)")
    ap.add_argument("--scene", action="store_true",
                    help="ignore images/ and use the scene photograph")
    ap.add_argument("--drawn-scene", dest="drawn_scene", action="store_true",
                    help="force the procedurally drawn fallback scene")
    ap.add_argument("--views", type=int, default=4,
                    help="number of views to cut from the scene (default 4)")
    ap.add_argument("--features", type=int, default=4000,
                    help="keypoint budget per image (default 4000)")
    ap.add_argument("--detector", default="SIFT",
                    help="detector for the single-detector outputs")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac-thresh", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_run = time.time()

    # ---------------- Task 1: acquisition ---------------------------------- #
    print("=" * 70)
    print("Feature-Based Image Matching and Panorama Construction")
    print("=" * 70)
    vs, scene = dataset.acquire(IMAGES_DIR, n_views=args.views,
                                force_scene=args.scene,
                                force_drawn=args.drawn_scene)
    if len(vs) < 3:
        raise SystemExit("At least three overlapping images are required.")

    dataset.save_viewset(vs, os.path.join(RESULTS_DIR, "inputs"))
    if scene is not None:
        vis.save(scene, os.path.join(RESULTS_DIR, "inputs", "scene_reference.png"))
    vis.montage(vs.images, vs.names, cols=2,
                path=os.path.join(RESULTS_DIR, "inputs", "input_views.png"))

    record = {
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "configuration": {
            "mode": "scene" if vs.synthetic else "real",
            "n_views": len(vs),
            "view_size": list(vs.images[0].shape[:2][::-1]),
            "n_features_budget": args.features,
            "lowe_ratio": args.ratio,
            "ransac_threshold_px": args.ransac_thresh,
            "detectors": experiments.DETECTORS,
        },
    }

    # ---------------- E1 ---------------------------------------------------- #
    e1_rows, panoramas = experiments.experiment_detector_comparison(
        vs, RESULTS_DIR, n_features=args.features, ratio=args.ratio,
        ransac_thresh=args.ransac_thresh)
    record["E1_detector_comparison"] = e1_rows

    # ---------------- E2 ---------------------------------------------------- #
    e2_rows = experiments.experiment_ransac_effect(vs, RESULTS_DIR,
                                                   n_features=args.features)
    record["E2_ransac_effect"] = e2_rows

    # ---------------- E3 ---------------------------------------------------- #
    if scene is None:
        print("\n[E3] Robustness study needs the scene "
              "(it requires exact ground truth); loading one.")
        scene = dataset.build_scene(force_drawn=args.drawn_scene)
    e3 = experiments.experiment_robustness(scene, RESULTS_DIR,
                                           n_features=args.features)
    record["E3_robustness"] = e3

    # ---------------- E4 ---------------------------------------------------- #
    e4_rows = experiments.experiment_blending(vs, RESULTS_DIR,
                                              detector=args.detector,
                                              n_features=args.features)
    record["E4_blending"] = e4_rows

    # ---------------- E5 ---------------------------------------------------- #
    if not args.quick:
        e5 = experiments.experiment_parameters(vs, RESULTS_DIR,
                                               n_features=args.features)
        record["E5_parameters"] = e5
    else:
        e5 = None

    record["runtime_seconds"] = time.time() - t_run

    # ---------------- Persist ---------------------------------------------- #
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    write_results_markdown(record, e1_rows, e2_rows, e3, e4_rows, e5)

    print("\n" + "=" * 70)
    print("Done in %.1f s.  Outputs written to %s" % (record["runtime_seconds"],
                                                      RESULTS_DIR))
    print("=" * 70)


def write_results_markdown(record, e1, e2, e3, e4, e5):
    """Write results/RESULTS.md with every table filled from the actual run."""
    path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Experimental Results\n\n")
        fh.write("Auto-generated by `src/main.py`. Every number below comes from "
                 "the run recorded in `results.json`.\n\n")

        env, cfg = record["environment"], record["configuration"]
        fh.write("**Environment.** Python %s, OpenCV %s, NumPy %s, %s.\n\n"
                 % (env["python"], env["opencv"], env["numpy"], env["platform"]))
        fh.write("**Configuration.** mode=%s, %d views of %dx%d px, keypoint "
                 "budget %d, Lowe ratio %.2f, RANSAC threshold %.1f px.\n"
                 % (cfg["mode"], cfg["n_views"], cfg["view_size"][0],
                    cfg["view_size"][1], cfg["n_features_budget"],
                    cfg["lowe_ratio"], cfg["ransac_threshold_px"]))

        # ---- Table 1 ---- #
        section(fh, "Table 1 - Detector / descriptor comparison")
        fh.write(md_table(e1,
                          ["detector", "descriptor_kind", "mean_keypoints",
                           "mean_matches", "mean_inliers", "mean_inlier_ratio",
                           "mean_grid_error_px", "total_time_ms",
                           "quality_mean_rmse", "quality_mean_ncc"],
                          ["Detector", "Desc.", "Keypoints/img", "Putative matches",
                           "RANSAC inliers", "Inlier ratio", "Align. err (px)",
                           "Total time (ms)", "Overlap RMSE", "Overlap NCC"]))
        fh.write("\nStage timings (summed over all pairs, ms):\n\n")
        fh.write(md_table(e1,
                          ["detector", "detect_ms", "describe_ms", "match_ms",
                           "ransac_ms", "warp_ms", "total_time_ms"],
                          ["Detector", "Detect", "Describe", "Match", "RANSAC",
                           "Warp+blend", "Total"]))

        # ---- Table 2 ---- #
        section(fh, "Table 2 - Effect of RANSAC (first image pair)")
        fh.write(md_table(e2,
                          ["detector", "n_putative", "n_true_correct", "n_inliers",
                           "n_removed", "inlier_ratio", "precision_before",
                           "precision_after", "recall_after",
                           "error_no_ransac_px", "error_with_ransac_px"],
                          ["Detector", "Putative", "Actually correct", "Inliers",
                           "Removed", "Inlier ratio", "Precision before",
                           "Precision after", "Recall after",
                           "Error w/o RANSAC (px)", "Error with RANSAC (px)"]))
        fh.write("\nValidation of the hand-written implementations against OpenCV:\n\n")
        fh.write(md_table(e2,
                          ["detector", "matcher_agreement_with_opencv",
                           "n_inliers", "opencv_inliers", "our_H_vs_opencv_H_px",
                           "ransac_iterations"],
                          ["Detector", "Matcher agreement", "Our inliers",
                           "cv2 inliers", "Our H vs cv2 H (px)", "RANSAC iters"]))

        # ---- Table 3 ---- #
        section(fh, "Table 3 - Robustness")
        for key, xcol, label in [("rotation", "rotation_deg", "Rotation (deg)"),
                                 ("scale", "zoom_factor", "Zoom factor"),
                                 ("viewpoint", "tilt", "Perspective tilt"),
                                 ("illumination", "gamma", "Gamma")]:
            fh.write("\n**%s**\n\n" % label)
            fh.write(md_table(e3[key],
                              ["detector", xcol, "n_matches", "n_inliers",
                               "inlier_ratio", "grid_error_px", "usable"],
                              ["Detector", label, "Putative", "Inliers",
                               "Inlier ratio", "Align. err (px)", "Usable"]))

        fh.write("\n**Named illumination conditions**\n\n")
        fh.write(md_table(e3["illumination_named"],
                          ["detector", "condition", "n_keypoints", "n_matches",
                           "n_inliers", "inlier_ratio", "grid_error_px", "usable"],
                          ["Detector", "Condition", "Keypoints", "Putative",
                           "Inliers", "Inlier ratio", "Align. err (px)", "Usable"]))

        # ---- Table 4 ---- #
        section(fh, "Table 4 - Blending strategy")
        fh.write(md_table(e4,
                          ["blend", "quality_mean_mae", "quality_mean_rmse",
                           "quality_mean_psnr_db", "quality_mean_ncc", "warp_ms"],
                          ["Blend", "Overlap MAE", "Overlap RMSE", "PSNR (dB)",
                           "NCC", "Warp+blend (ms)"]))
        fh.write("\nNote: the overlap metrics compare the *source images* warped "
                 "onto the canvas, so they measure geometric alignment and are "
                 "identical for both blending modes by construction; the "
                 "difference between the modes is visual (seam visibility) and "
                 "is shown in `panorama_blend_overwrite.png` vs "
                 "`panorama_blend_feather.png`.\n")

        # ---- Table 5 ---- #
        if e5:
            section(fh, "Table 5 - Parameter sensitivity")
            fh.write("\n**Lowe ratio threshold**\n\n")
            fh.write(md_table(e5["lowe_ratio"],
                              ["detector", "lowe_ratio", "n_matches", "n_inliers",
                               "inlier_ratio", "match_precision", "grid_error_px"],
                              ["Detector", "Ratio", "Putative", "Inliers",
                               "Inlier ratio", "Precision", "Align. err (px)"]))
            fh.write("\n**RANSAC inlier threshold**\n\n")
            fh.write(md_table(e5["ransac_threshold"],
                              ["detector", "ransac_threshold_px", "n_inliers",
                               "inlier_ratio", "grid_error_px", "ransac_iterations"],
                              ["Detector", "Threshold (px)", "Inliers",
                               "Inlier ratio", "Align. err (px)", "Iterations"]))

    print("[main] wrote %s" % path)
    return path


if __name__ == "__main__":
    main()
