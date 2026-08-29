"""
Experiment driver.

Running this script reproduces every number and every figure that appears in
the report. It writes

    results/tables/*.csv        the quantitative tables
    results/measurements/*.csv  the per object measurements
    results/figures/*.png       the figures
    results/summary.json        a machine readable copy of the headline numbers

Usage
    python run_experiments.py
"""

import csv
import json
import time

import cv2
import numpy as np

import measurement as M
import preprocessing as P
import segmentation as S
import visualisation as V
from config import (BASE_IMAGES, FIG_DIR, MEAS_DIR, RESULTS_DIR, TABLE_DIR,
                    VARIANTS, all_image_ids, has_ground_truth,
                    has_instance_ground_truth, image_record)
from dataset import build_variants, load_working
from evaluation import evaluate, measurement_errors
from ground_truth import load_gt

# ----------------------------------------------------------------------------
# One common parameter set. The same values are used for every image, so no
# method is given an unfair per image advantage.
# ----------------------------------------------------------------------------
PREP = dict(denoise_method="gaussian", ksize=5, use_illumination_correction=True)

# The tolerance of the region growing and the configuration of the clustering
# are the values that performed best on average in the parameter studies of
# sections 4 and 5, so every method is compared at its own best common setting.
DEFAULTS = {
    "threshold": dict(mode="otsu", open_k=3, close_k=7, min_area=200),
    "region_growing": dict(tolerance=25, seed_strategy="distance",
                           criterion="running_mean", connectivity=8,
                           open_k=3, close_k=7, min_area=200),
    "kmeans": dict(k=3, feature_space="lab", open_k=3, close_k=7,
                   min_area=200),
}

METHOD_LABELS = {
    "threshold": "Thresholding (Otsu)",
    "region_growing": "Region growing (tol 25)",
    "kmeans": "K means (K = 3, Lab)",
}

# Marker distance ratio used when touching objects are separated.
SPLIT_RATIO = 0.4


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def prep_image(image_id, **overrides):
    work, rec, _ = load_working(image_id)
    cfg = dict(PREP)
    cfg.update(overrides)
    return work, rec, P.prepare(work, **cfg)


# ============================================================================
# Experiment 1: the three methods on every annotated image
# ============================================================================
def experiment_main_comparison():
    rows = []
    per_image_panels = []
    gt_ids = [r["id"] for r in BASE_IMAGES if has_ground_truth(r["id"])]

    for image_id in gt_ids:
        work, rec, prep = prep_image(image_id)
        gt = load_gt(image_id)
        panels = [work, gt]
        titles = ["photograph", "ground truth"]

        for method in ("threshold", "region_growing", "kmeans"):
            mask, info = S.run_method(method, prep, DEFAULTS[method])
            scores = evaluate(mask, gt)
            row = {
                "image": image_id,
                "title": rec["title"],
                "method": METHOD_LABELS[method],
                "iou": scores["iou"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "dice": scores["dice"],
                "time_s": round(info["time_s"], 3),
                "pred_count": scores["pred_count"],
                "gt_count": scores["gt_count"] if has_instance_ground_truth(image_id) else "",
                "count_error": (scores["count_error"]
                                if has_instance_ground_truth(image_id) else ""),
                "obj_precision": (scores["obj_precision"]
                                  if has_instance_ground_truth(image_id) else ""),
                "obj_recall": (scores["obj_recall"]
                               if has_instance_ground_truth(image_id) else ""),
                "obj_f1": (scores["obj_f1"]
                           if has_instance_ground_truth(image_id) else ""),
            }
            rows.append(row)
            panels.append(V.error_map(mask, gt))
            titles.append("%s\nIoU %.3f" % (METHOD_LABELS[method], scores["iou"]))

        V.panel(panels, titles, "main_%s.png" % image_id, cols=5,
                suptitle="%s. Green true positive, red false positive, blue missed"
                         % rec["title"])
        per_image_panels.append("main_%s.png" % image_id)

    write_csv(TABLE_DIR / "main_comparison.csv", rows)

    # Averages over the annotated images.
    summary = []
    for method in ("threshold", "region_growing", "kmeans"):
        sel = [r for r in rows if r["method"] == METHOD_LABELS[method]]
        summary.append({
            "method": METHOD_LABELS[method],
            "mean_iou": round(float(np.mean([r["iou"] for r in sel])), 4),
            "mean_precision": round(float(np.mean([r["precision"] for r in sel])), 4),
            "mean_recall": round(float(np.mean([r["recall"] for r in sel])), 4),
            "mean_f1": round(float(np.mean([r["f1"] for r in sel])), 4),
            "mean_time_s": round(float(np.mean([r["time_s"] for r in sel])), 3),
        })
    write_csv(TABLE_DIR / "main_comparison_mean.csv", summary)

    V.bar_chart([r["image"].replace("_", " ") for r in rows[::3]],
                {METHOD_LABELS[m]: [r["iou"] for r in rows
                                    if r["method"] == METHOD_LABELS[m]]
                 for m in ("threshold", "region_growing", "kmeans")},
                "iou_by_image.png", "IoU",
                "Segmentation accuracy per image", ylim=(0, 1.05))

    V.bar_chart([s["method"] for s in summary],
                {"mean processing time": [s["mean_time_s"] for s in summary]},
                "time_by_method.png", "seconds",
                "Mean processing time per image at 900 pixel working size")
    return rows, summary, per_image_panels


def experiment_dataset_figure():
    """One overview figure of the whole dataset."""
    imgs, titles = [], []
    for rec in BASE_IMAGES:
        work, _, _ = load_working(rec["id"])
        imgs.append(work)
        titles.append(rec["id"] + chr(10) + rec["domain"])
    V.panel(imgs, titles, "dataset_overview.png", cols=4,
            suptitle="The eight base photographs of the dataset")

    gts, gtitles = [], []
    for rec in BASE_IMAGES:
        if not has_ground_truth(rec["id"]):
            continue
        work, _, _ = load_working(rec["id"])
        gt = load_gt(rec["id"])
        gts.append(V.draw_boundaries(work, gt, (0, 220, 0), 2))
        gtitles.append(rec["id"])
    V.panel(gts, gtitles, "ground_truth_overview.png", cols=5,
            suptitle="Manually verified ground truth outlines")
    return len(imgs)


# ============================================================================
# Experiment 2: pipeline illustration
# ============================================================================
def experiment_pipeline_figure(image_id="seeds_sunflower_01"):
    work, rec, prep = prep_image(image_id)
    gray = prep["gray"]
    smooth = P.denoise(gray, "gaussian", 5)
    flat = P.correct_illumination(smooth, objects_are_dark=prep["objects_are_dark"])
    t, _ = S.otsu_threshold_from_scratch(flat)
    raw, _ = S.segment_threshold(flat, "otsu", prep["objects_are_dark"])
    clean = S.postprocess(raw, 3, 7, 200)
    bounds = M.boundaries(clean)
    rows = M.measure_objects(clean)
    measured = V.draw_measurements(work, clean, rows)

    V.panel([work, gray, smooth, flat, raw, clean, bounds, measured],
            ["1. input photograph", "2. grey level", "3. Gaussian smoothing",
             "4. illumination corrected", "5. Otsu threshold",
             "6. morphological clean up", "7. boundary identification",
             "8. object measurement"],
            "pipeline_stages.png", cols=4,
            suptitle="Complete pipeline on %s" % rec["title"])
    V.histogram_with_threshold(flat, t, "otsu_histogram.png",
                               "Histogram after illumination correction")
    return t


# ============================================================================
# Experiment 3: thresholding study
# ============================================================================
def experiment_threshold_study():
    rows = []
    cases = [
        ("Otsu, illumination corrected", dict(mode="otsu"), True),
        ("Otsu, no illumination correction", dict(mode="otsu"), False),
        ("Fixed threshold 100", dict(mode="global", manual_value=100), False),
        ("Fixed threshold 140", dict(mode="global", manual_value=140), False),
        ("Fixed threshold 180", dict(mode="global", manual_value=180), False),
        ("Adaptive Gaussian, block 51", dict(mode="adaptive", block_size=51, C=8), False),
        ("Adaptive Gaussian, block 101", dict(mode="adaptive", block_size=101, C=8), False),
    ]
    for image_id in ("seeds_sunflower_01", "seeds_sunflower_01_illum",
                     "coins_greek_07", "coins_greek_07_illum"):
        gt = load_gt(image_id)
        for label, params, use_illum in cases:
            work, rec, prep = prep_image(image_id,
                                         use_illumination_correction=use_illum)
            p = dict(DEFAULTS["threshold"])
            p.update(params)
            mask, info = S.run_method("threshold", prep, p)
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "setting": label,
                         "iou": sc["iou"], "precision": sc["precision"],
                         "recall": sc["recall"], "f1": sc["f1"],
                         "time_s": round(info["time_s"], 3)})
    write_csv(TABLE_DIR / "threshold_study.csv", rows)

    labels = [c[0] for c in cases]
    series = {}
    for image_id in ("seeds_sunflower_01", "seeds_sunflower_01_illum",
                     "coins_greek_07", "coins_greek_07_illum"):
        series[image_id.replace("seeds_sunflower_01", "seeds").replace(
            "coins_greek_07", "coins").replace("_", " ")] = [
            r["iou"] for r in rows if r["image"] == image_id]
    V.bar_chart(labels, series, "threshold_study.png", "IoU",
                "Effect of the threshold strategy", ylim=(0, 1.05), annotate=False)

    # Visual comparison on the image with the lighting gradient.
    work, rec, prep_on = prep_image("seeds_sunflower_01_illum",
                                    use_illumination_correction=True)
    _, _, prep_off = prep_image("seeds_sunflower_01_illum",
                                use_illumination_correction=False)
    m_off, _ = S.run_method("threshold", prep_off, DEFAULTS["threshold"])
    m_on, _ = S.run_method("threshold", prep_on, DEFAULTS["threshold"])
    V.panel([work, prep_off["prepared"], m_off, prep_on["prepared"], m_on],
            ["photograph with lighting gradient", "grey level, no correction",
             "Otsu without correction", "after flat field correction",
             "Otsu with correction"],
            "illumination_correction.png", cols=5)
    return rows


# ============================================================================
# Experiment 4: region growing study
# ============================================================================
def experiment_region_growing_study():
    rows = []
    tolerances = [6, 10, 14, 18, 25, 35, 50]
    for image_id in ("seeds_sunflower_01", "coins_greek_07"):
        gt = load_gt(image_id)
        work, rec, prep = prep_image(image_id)
        for tol in tolerances:
            p = dict(DEFAULTS["region_growing"])
            p["tolerance"] = tol
            mask, info = S.run_method("region_growing", prep, p)
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "study": "tolerance",
                         "setting": str(tol), "n_seeds": info["n_seeds"],
                         "iou": sc["iou"], "precision": sc["precision"],
                         "recall": sc["recall"], "f1": sc["f1"],
                         "pred_count": sc["pred_count"],
                         "time_s": round(info["time_s"], 3)})
        for strategy in ("distance", "grid", "random"):
            p = dict(DEFAULTS["region_growing"])
            p["seed_strategy"] = strategy
            p["n_seeds"] = 40
            mask, info = S.run_method("region_growing", prep, p)
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "study": "seed_strategy",
                         "setting": strategy, "n_seeds": info["n_seeds"],
                         "iou": sc["iou"], "precision": sc["precision"],
                         "recall": sc["recall"], "f1": sc["f1"],
                         "pred_count": sc["pred_count"],
                         "time_s": round(info["time_s"], 3)})
        for criterion in ("running_mean", "seed_value"):
            p = dict(DEFAULTS["region_growing"])
            p["criterion"] = criterion
            mask, info = S.run_method("region_growing", prep, p)
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "study": "criterion",
                         "setting": criterion, "n_seeds": info["n_seeds"],
                         "iou": sc["iou"], "precision": sc["precision"],
                         "recall": sc["recall"], "f1": sc["f1"],
                         "pred_count": sc["pred_count"],
                         "time_s": round(info["time_s"], 3)})
    write_csv(TABLE_DIR / "region_growing_study.csv", rows)

    V.line_chart(tolerances,
                 {image_id.replace("_", " "):
                  [r["iou"] for r in rows
                   if r["image"] == image_id and r["study"] == "tolerance"]
                  for image_id in ("seeds_sunflower_01", "coins_greek_07")},
                 "region_growing_tolerance.png", "similarity tolerance",
                 "IoU", "Region growing, effect of the similarity tolerance",
                 xticks=tolerances)

    V.bar_chart(["distance transform", "regular grid", "random"],
                {image_id.replace("_", " "):
                 [r["iou"] for r in rows
                  if r["image"] == image_id and r["study"] == "seed_strategy"]
                 for image_id in ("seeds_sunflower_01", "coins_greek_07")},
                "region_growing_seeds.png", "IoU",
                "Region growing, effect of the seed selection strategy",
                ylim=(0, 1.05))

    # Visual illustration of leaking and of under growing.
    work, rec, prep = prep_image("seeds_sunflower_01")
    panels, titles = [work], ["photograph"]
    for tol in (6, 18, 50):
        p = dict(DEFAULTS["region_growing"])
        p["tolerance"] = tol
        mask, info = S.run_method("region_growing", prep, p)
        panels.append(V.draw_boundaries(work, mask))
        titles.append("tolerance %d, %d seeds" % (tol, info["n_seeds"]))
    V.panel(panels, titles, "region_growing_visual.png", cols=4)
    return rows


# ============================================================================
# Experiment 5: K means study
# ============================================================================
def experiment_kmeans_study():
    rows = []
    ks = [2, 3, 4, 5, 6]
    for image_id in ("seeds_sunflower_01", "coins_greek_07", "beans_coffee_06"):
        gt = load_gt(image_id)
        work, rec, prep = prep_image(image_id)
        for k in ks:
            # The sweep over K is run on the intensity feature space, so that
            # the effect of K is not mixed with the effect of the features.
            p = dict(DEFAULTS["kmeans"])
            p["feature_space"] = "intensity"
            p["k"] = k
            mask, info = S.run_method("kmeans", prep, p)
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "study": "K", "setting": str(k),
                         "iou": sc["iou"], "precision": sc["precision"],
                         "recall": sc["recall"], "f1": sc["f1"],
                         "pred_count": sc["pred_count"],
                         "time_s": round(info["time_s"], 3)})
        for space in ("intensity", "lab", "lab_xy"):
            p = dict(DEFAULTS["kmeans"])
            p["feature_space"] = space
            p["k"] = 3
            mask, info = S.run_method("kmeans", prep, p)
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "study": "features",
                         "setting": space, "iou": sc["iou"],
                         "precision": sc["precision"], "recall": sc["recall"],
                         "f1": sc["f1"], "pred_count": sc["pred_count"],
                         "time_s": round(info["time_s"], 3)})
    write_csv(TABLE_DIR / "kmeans_study.csv", rows)

    V.line_chart(ks,
                 {image_id.replace("_", " "):
                  [r["iou"] for r in rows
                   if r["image"] == image_id and r["study"] == "K"]
                  for image_id in ("seeds_sunflower_01", "coins_greek_07",
                                   "beans_coffee_06")},
                 "kmeans_k.png", "number of clusters K", "IoU",
                 "K means, effect of the number of clusters", xticks=ks)

    # Cluster label maps for one image.
    work, rec, prep = prep_image("beans_coffee_06")
    panels, titles = [work], ["photograph"]
    for k in (2, 3, 5):
        mask, info = S.segment_kmeans(prep["bgr"], prep["prepared"], k=k,
                                      objects_dark=prep["objects_are_dark"])
        label_img = info["label_image"].astype(np.float32)
        label_img = cv2.normalize(label_img, None, 0, 255, cv2.NORM_MINMAX)
        panels.append(cv2.applyColorMap(label_img.astype(np.uint8),
                                        cv2.COLORMAP_VIRIDIS))
        titles.append("K = %d cluster map" % k)
    V.panel(panels, titles, "kmeans_clusters.png", cols=4)
    return rows


# ============================================================================
# Experiment 6: filtering study
# ============================================================================
def experiment_filter_study():
    rows = []
    for image_id in ("seeds_sunflower_01_noise", "coins_greek_07_noise"):
        gt = load_gt(image_id)
        for method in ("none", "gaussian", "median", "bilateral"):
            work, rec, prep = prep_image(image_id, denoise_method=method, ksize=5)
            t0 = time.perf_counter()
            mask, _ = S.run_method("threshold", prep, DEFAULTS["threshold"])
            elapsed = time.perf_counter() - t0
            sc = evaluate(mask, gt)
            rows.append({"image": image_id, "filter": method,
                         "iou": sc["iou"], "precision": sc["precision"],
                         "recall": sc["recall"], "f1": sc["f1"],
                         "pred_count": sc["pred_count"],
                         "time_s": round(elapsed, 3)})
    write_csv(TABLE_DIR / "filter_study.csv", rows)
    V.bar_chart(["no filter", "Gaussian", "median", "bilateral"],
                {image_id.replace("_", " "):
                 [r["iou"] for r in rows if r["image"] == image_id]
                 for image_id in ("seeds_sunflower_01_noise", "coins_greek_07_noise")},
                "filter_study.png", "IoU",
                "Effect of the smoothing filter on noisy images", ylim=(0, 1.05))
    return rows


# ============================================================================
# Experiment 7: robustness to imaging conditions
# ============================================================================
def experiment_conditions():
    rows = []
    groups = [("seeds_sunflower_01", ["seeds_sunflower_01", "seeds_sunflower_01_illum",
                                      "seeds_sunflower_01_dim",
                                      "seeds_sunflower_01_noise",
                                      "seeds_sunflower_01_blur"]),
              ("coins_greek_07", ["coins_greek_07", "coins_greek_07_illum",
                                  "coins_greek_07_noise"])]
    for base, ids in groups:
        for image_id in ids:
            gt = load_gt(image_id)
            rec = image_record(image_id)
            work, _, prep = prep_image(image_id)
            for method in ("threshold", "region_growing", "kmeans"):
                mask, info = S.run_method(method, prep, DEFAULTS[method])
                sc = evaluate(mask, gt)
                rows.append({
                    "base": base, "image": image_id,
                    "condition": rec.get("variant_label", "original photograph"),
                    "method": METHOD_LABELS[method],
                    "iou": sc["iou"], "f1": sc["f1"],
                    "precision": sc["precision"], "recall": sc["recall"],
                    "pred_count": sc["pred_count"],
                    "time_s": round(info["time_s"], 3)})
    write_csv(TABLE_DIR / "conditions.csv", rows)

    for base, ids in groups:
        conditions = [image_record(i).get("variant_label", "original")
                      for i in ids]
        series = {}
        for method in ("threshold", "region_growing", "kmeans"):
            series[METHOD_LABELS[method]] = [
                r["iou"] for r in rows
                if r["base"] == base and r["method"] == METHOD_LABELS[method]]
        V.bar_chart(conditions, series, "conditions_%s.png" % base, "IoU",
                    "Robustness to imaging conditions, %s" % base.replace("_", " "),
                    ylim=(0, 1.05), annotate=False)

    # Visual strip of the conditions.
    imgs, titles = [], []
    for image_id in groups[0][1]:
        work, rec, _ = load_working(image_id)
        imgs.append(work)
        titles.append(rec.get("variant_label", "original photograph"))
    V.panel(imgs, titles, "conditions_strip.png", cols=5)
    return rows


# ============================================================================
# Experiment 8: separating touching objects and object measurement
# ============================================================================
def experiment_split_study():
    """Effect of the marker threshold used when touching objects are split."""
    rows = []
    ratios = [0.25, 0.35, 0.45, 0.55, 0.65]
    for image_id in ("beans_coffee_06", "coins_greek_07",
                     "beans_cowpea_scale_05", "seeds_sunflower_01"):
        rec = image_record(image_id)
        work, _, prep = prep_image(image_id)
        mask, _ = S.run_method("threshold", prep, DEFAULTS["threshold"])
        base_count = len(M.measure_objects(mask))
        for ratio in ratios:
            split, markers = S.split_touching(mask, ratio)
            split = S.remove_small_components(split, DEFAULTS["threshold"]["min_area"])
            rows.append({"image": image_id,
                         "expected": rec.get("expected_count") or "",
                         "marker_ratio": ratio,
                         "count_without_split": base_count,
                         "count_with_split": len(M.measure_objects(split))})
    write_csv(TABLE_DIR / "split_study.csv", rows)
    V.line_chart(ratios,
                 {image_id.replace("_", " "):
                  [r["count_with_split"] for r in rows if r["image"] == image_id]
                  for image_id in ("beans_coffee_06", "coins_greek_07",
                                   "beans_cowpea_scale_05", "seeds_sunflower_01")},
                 "split_study.png", "marker distance ratio",
                 "objects detected",
                 "Separating touching objects, effect of the marker threshold",
                 xticks=ratios)
    return rows


def experiment_measurement():
    rows_all = []
    counts = []
    for image_id in ("beans_cowpea_scale_05", "beans_coffee_06", "coins_greek_07",
                     "seeds_pale_04", "seeds_sunflower_01"):
        rec = image_record(image_id)
        work, _, prep = prep_image(image_id)
        mask, _ = S.run_method("threshold", prep, DEFAULTS["threshold"])
        split, n_markers = S.split_touching(mask, SPLIT_RATIO)
        split = S.remove_small_components(split, DEFAULTS["threshold"]["min_area"])

        px_per_mm = rec.get("px_per_mm")
        rows_before = M.measure_objects(mask, px_per_mm)
        rows_after = M.measure_objects(split, px_per_mm)

        write_csv(MEAS_DIR / ("measurements_%s.csv" % image_id), rows_after)
        for r in rows_after:
            r2 = dict(r)
            r2["image"] = image_id
            rows_all.append(r2)

        counts.append({
            "image": image_id,
            "expected": rec.get("expected_count") or "",
            "count_before_split": len(rows_before),
            "count_after_split": len(rows_after),
            "summary_after": M.summarise(rows_after, px_per_mm),
            "size_classes": M.size_classes(rows_after),
        })

        # A full resolution copy of the annotated result, easier to read than
        # the four panel summary.
        cv2.imwrite(str(FIG_DIR / ("measured_full_%s.png" % image_id)),
                    V.draw_measurements(work, split, rows_after, px_per_mm,
                                        font_scale=0.55))

        V.panel([work,
                 V.draw_boundaries(work, mask),
                 V.draw_boundaries(work, split),
                 V.draw_measurements(work, split, rows_after, px_per_mm)],
                ["photograph", "segmented regions",
                 "after separating touching objects",
                 "measurements"],
                "measure_%s.png" % image_id, cols=4,
                suptitle=rec["title"])

    write_csv(TABLE_DIR / "measurement_counts.csv",
              [{"image": c["image"], "expected": c["expected"],
                "count_before_split": c["count_before_split"],
                "count_after_split": c["count_after_split"]} for c in counts])

    cowpea = [r for r in rows_all if r["image"] == "beans_cowpea_scale_05"]
    if cowpea:
        V.size_histogram([r["major_axis_mm"] for r in cowpea],
                         "cowpea_length_histogram.png",
                         "seed length in millimetres",
                         "Length distribution of the cowpea seeds", bins=8)
    seeds = [r for r in rows_all if r["image"] == "seeds_sunflower_01"]
    if seeds:
        V.size_histogram([r["area_px"] for r in seeds],
                         "seed_area_histogram.png", "area in pixels",
                         "Area distribution of the sunflower seeds", bins=10)
    return counts, rows_all


# ============================================================================
# Experiment 9: failure cases
# ============================================================================
def experiment_measurement_accuracy():
    """How accurate are the reported numbers for the objects that are found."""
    rows = []
    for image_id in ("beans_cowpea_scale_05", "beans_coffee_06", "coins_greek_07",
                     "seeds_pale_04"):
        rec = image_record(image_id)
        if not has_instance_ground_truth(image_id):
            continue
        gt = load_gt(image_id)
        work, _, prep = prep_image(image_id)
        for method in ("threshold", "kmeans"):
            mask, _ = S.run_method(method, prep, DEFAULTS[method])
            split, _ = S.split_touching(mask, SPLIT_RATIO)
            split = S.remove_small_components(split, DEFAULTS[method]["min_area"])
            err = measurement_errors(split, gt, rec.get("px_per_mm"))
            rows.append({
                "image": image_id, "method": METHOD_LABELS[method],
                "objects_in_ground_truth": rec.get("expected_count") or "",
                "matched_objects": err.get("matched", 0),
                "mean_abs_area_error_pct": err.get("mean_abs_area_error_pct", ""),
                "max_abs_area_error_pct": err.get("max_abs_area_error_pct", ""),
                "mean_abs_length_error_pct": err.get("mean_abs_major_error_pct", ""),
                "max_abs_length_error_pct": err.get("max_abs_major_error_pct", ""),
            })
    write_csv(TABLE_DIR / "measurement_accuracy.csv", rows)
    return rows


def experiment_failures():
    notes = []

    # Dense pile of beans.
    work, rec, prep = prep_image("beans_pinto_dense_08")
    mask, _ = S.run_method("threshold", prep, DEFAULTS["threshold"])
    split, _ = S.split_touching(mask, SPLIT_RATIO)
    split = S.remove_small_components(split, 200)
    n_regions = len(M.measure_objects(mask))
    n_split = len(M.measure_objects(split))
    V.panel([work, V.draw_boundaries(work, mask), V.draw_boundaries(work, split)],
            ["photograph, about 200 beans",
             "one region for the whole pile, %d regions" % n_regions,
             "after watershed, %d regions" % n_split],
            "failure_dense.png", cols=3)
    notes.append({"case": "dense pile", "regions": n_regions,
                  "regions_after_split": n_split})

    # Low contrast objects.
    work, rec, prep = prep_image("seeds_pale_04")
    gt = load_gt("seeds_pale_04")
    panels, titles = [work, gt], ["low contrast photograph", "ground truth"]
    for method in ("threshold", "region_growing", "kmeans"):
        mask, _ = S.run_method(method, prep, DEFAULTS[method])
        sc = evaluate(mask, gt)
        panels.append(V.error_map(mask, gt))
        titles.append("%s, IoU %.3f" % (METHOD_LABELS[method], sc["iou"]))
        notes.append({"case": "low contrast", "method": METHOD_LABELS[method],
                      "iou": sc["iou"]})
    V.panel(panels, titles, "failure_low_contrast.png", cols=5)

    # Shadows treated as object.
    work, rec, prep = prep_image("beans_coffee_06")
    gt = load_gt("beans_coffee_06")
    mask, _ = S.run_method("threshold", prep, DEFAULTS["threshold"])
    V.panel([work, V.error_map(mask, gt)],
            ["beans with soft cast shadows",
             "red shows shadow pixels accepted as object"],
            "failure_shadow.png", cols=2)
    return notes


# ============================================================================
# Driver
# ============================================================================
def main():
    build_variants()
    started = time.time()
    summary = {}

    print("1. main comparison")
    rows, means, panels = experiment_main_comparison()
    summary["main"] = rows
    summary["main_mean"] = means

    print("1b. dataset figure")
    experiment_dataset_figure()

    print("2. pipeline figure")
    summary["otsu_threshold_example"] = experiment_pipeline_figure()

    print("3. threshold study")
    summary["threshold_study"] = experiment_threshold_study()

    print("4. region growing study")
    summary["region_growing_study"] = experiment_region_growing_study()

    print("5. k means study")
    summary["kmeans_study"] = experiment_kmeans_study()

    print("6. filter study")
    summary["filter_study"] = experiment_filter_study()

    print("7. imaging conditions")
    summary["conditions"] = experiment_conditions()

    print("8. separating touching objects")
    summary["split_study"] = experiment_split_study()

    print("9. measurement")
    counts, meas = experiment_measurement()
    summary["measurement_counts"] = counts

    print("10. measurement accuracy")
    summary["measurement_accuracy"] = experiment_measurement_accuracy()

    print("11. failure cases")
    summary["failures"] = experiment_failures()

    summary["runtime_s"] = round(time.time() - started, 1)
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("done in %.1f s" % summary["runtime_s"])


if __name__ == "__main__":
    main()
