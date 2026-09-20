"""
run_experiments.py
Classical Object Detection and Localisation

Runs every experiment reported in the write up and saves the tables,
figures and annotated detection images under results/.

    python src/run_experiments.py

Experiments
    E1  detection threshold sweep (true, false and missed detections)
    E2  masked against unmasked normalised cross correlation
    E3  size of the scale pyramid against accuracy and run time
    E4  rotated template bank against accuracy and run time
    E5  explicit sliding window: window size, step size, threshold, cost
    E6  SIFT feature matching with RANSAC, ratio and inlier study
    E7  colour segmentation proposals combined with template matching
    E8  final comparison on the held out test set
    E9  qualitative figures and per image detection output

Author: Nii Yartey Gidiglo
"""

import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import detector as D
import evaluate as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
DET = os.path.join(RES, "detections")
for d in (RES, FIG, TAB, DET):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

# scale pyramids, expressed as multiples of the reference template size
SCALES_1 = [1.0]
SCALES_5 = [0.6, 0.8, 1.0, 1.35, 1.8]
SCALES_11 = [round(0.5 * 1.15 ** i, 4) for i in range(11)]
ANGLES = [-45, -30, -15, 0, 15, 30, 45, 90]

THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(25)]   # 0.300 to 0.900
BASE_THRESHOLD = 0.30      # detections are generated once at this low value
SUMMARY = {}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_split(split):
    folder = "dev_set" if split == "dev" else "test_set"
    ann = E.load_annotations(os.path.join(DATA, "annotations", "%s_annotations.json" % split))
    images = {}
    for name in ann:
        bgr = cv2.imread(os.path.join(DATA, folder, name), cv2.IMREAD_COLOR)
        images[name] = {"bgr": bgr, "grey": D.preprocess(bgr)}
    return ann, images


def scored_detections(images, bank, masked=True, threshold=BASE_THRESHOLD):
    """Run the dense matcher once per image and keep boxes with scores."""
    out, total_time = {}, 0.0
    for name, im in images.items():
        t0 = time.perf_counter()
        boxes, scores, meta = D.template_match(im["grey"], bank, threshold, masked)
        total_time += time.perf_counter() - t0
        out[name] = {"boxes": boxes, "scores": scores, "meta": meta}
    return out, total_time / max(1, len(images))


def at_threshold(scored, thr):
    return {n: [b for b, s in zip(v["boxes"], v["scores"]) if s >= thr]
            for n, v in scored.items()}


def sweep(scored, ann, thresholds=THRESHOLDS):
    rows = []
    for thr in thresholds:
        r = E.evaluate_split(at_threshold(scored, thr), ann)
        rows.append((thr, r["tp"], r["fp"], r["fn"], r["precision"], r["recall"], r["f1"]))
    return rows


def best_row(rows):
    return max(rows, key=lambda r: (r[6], r[4]))


def fmt(rows):
    return [[r[0], r[1], r[2], r[3], round(r[4], 4), round(r[5], 4), round(r[6], 4)]
            for r in rows]


SWEEP_HEADER = ["threshold", "true_positives", "false_positives",
                "missed_detections", "precision", "recall", "f1"]


# --------------------------------------------------------------------------
def main():
    tmpl = cv2.imread(os.path.join(DATA, "template", "ug_logo_template.png"), 0)
    mask = cv2.imread(os.path.join(DATA, "template", "ug_logo_mask.png"), 0)
    tmpl_bgr = cv2.imread(os.path.join(DATA, "template", "ug_logo_template.png"))
    print("template %dx%d, mask covers %.1f%% of the rectangle"
          % (tmpl.shape[1], tmpl.shape[0], 100.0 * (mask > 0).mean()))
    SUMMARY["template"] = {"width": int(tmpl.shape[1]), "height": int(tmpl.shape[0]),
                           "mask_coverage": float((mask > 0).mean())}

    dev_ann, dev_img = load_split("dev")
    test_ann, test_img = load_split("test")
    SUMMARY["dataset"] = {
        "dev_images": len(dev_ann),
        "dev_objects": sum(len(v["objects"]) for v in dev_ann.values()),
        "test_images": len(test_ann),
        "test_objects": sum(len(v["objects"]) for v in test_ann.values()),
    }

    bank1 = D.build_template_bank(tmpl, mask, SCALES_1, [0.0])
    bank5 = D.build_template_bank(tmpl, mask, SCALES_5, [0.0])
    bank11 = D.build_template_bank(tmpl, mask, SCALES_11, [0.0])
    bank_rot = D.build_template_bank(tmpl, mask, SCALES_11, ANGLES)

    # ------------------------------------------------------------------
    # figure: template and mask
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax[0].imshow(cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2RGB)); ax[0].set_title("template (colour)")
    ax[1].imshow(tmpl, cmap="gray"); ax[1].set_title("template (grey scale)")
    ax[2].imshow(mask, cmap="gray"); ax[2].set_title("correlation mask")
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.savefig(os.path.join(FIG, "fig01_template.png")); plt.close(fig)

    # ------------------------------------------------------------------
    # figure: dataset overview
    # ------------------------------------------------------------------
    show = ["dev01_baseline_a.jpg", "dev04_scale_small.jpg", "dev09_rot_large.jpg",
            "dev10_illum_dark.jpg", "dev13_view_tilt.jpg", "dev16_occ_heavy.jpg",
            "dev17_clutter.jpg", "dev19_blur.jpg", "dev20_negative.jpg"]
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 6.4))
    for a, name in zip(axes.ravel(), show):
        img = dev_img[name]["bgr"].copy()
        for o in dev_ann[name]["objects"]:
            x, y, w, h = o["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title("%s\n%s" % (name.replace(".jpg", ""), dev_ann[name]["condition"]), fontsize=7)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.savefig(os.path.join(FIG, "fig02_dataset.png")); plt.close(fig)

    # ==================================================================
    # E1  threshold sweep, single scale template matching, development set
    # ==================================================================
    print("\nE1 threshold sweep")
    dev1, t_dev1 = scored_detections(dev_img, bank1, True)
    rows1 = sweep(dev1, dev_ann)
    E.write_csv(os.path.join(TAB, "e1_threshold_single_scale_dev.csv"), SWEEP_HEADER, fmt(rows1))
    b1 = best_row(rows1)
    print("  best single scale: thr=%.3f F1=%.3f (P=%.3f R=%.3f)" % (b1[0], b1[6], b1[4], b1[5]))

    dev11, t_dev11 = scored_detections(dev_img, bank11, True)
    rows11 = sweep(dev11, dev_ann)
    E.write_csv(os.path.join(TAB, "e1_threshold_multi_scale_dev.csv"), SWEEP_HEADER, fmt(rows11))
    b11 = best_row(rows11)
    print("  best multi scale : thr=%.3f F1=%.3f (P=%.3f R=%.3f)" % (b11[0], b11[6], b11[4], b11[5]))

    a = np.array(rows11, float)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax[0].plot(a[:, 0], a[:, 1], "o-", label="true detections")
    ax[0].plot(a[:, 0], a[:, 2], "s-", label="false detections")
    ax[0].plot(a[:, 0], a[:, 3], "^-", label="missed detections")
    ax[0].set_yscale("symlog"); ax[0].set_xlabel("correlation threshold")
    ax[0].set_ylabel("count"); ax[0].legend(); ax[0].set_title("effect of the threshold on the counts")
    ax[1].plot(a[:, 0], a[:, 4], "o-", label="precision")
    ax[1].plot(a[:, 0], a[:, 5], "s-", label="recall")
    ax[1].plot(a[:, 0], a[:, 6], "^-", label="F1")
    ax[1].axvline(b11[0], color="k", ls="--", lw=1)
    ax[1].set_xlabel("correlation threshold"); ax[1].set_ylim(-0.02, 1.05)
    ax[1].legend(); ax[1].set_title("precision, recall and F1, multi scale, dev set")
    fig.savefig(os.path.join(FIG, "fig03_threshold_sweep.png")); plt.close(fig)

    SUMMARY["E1"] = {"single_scale_best": list(b1), "multi_scale_best": list(b11),
                     "single_scale_seconds_per_image": t_dev1,
                     "multi_scale_seconds_per_image": t_dev11}

    # ==================================================================
    # E2  masked against unmasked correlation
    # ==================================================================
    print("E2 masked against unmasked correlation")
    dev11u, t_dev11u = scored_detections(dev_img, bank11, False)
    rows11u = sweep(dev11u, dev_ann)
    E.write_csv(os.path.join(TAB, "e2_threshold_multi_scale_unmasked_dev.csv"),
                SWEEP_HEADER, fmt(rows11u))
    b11u = best_row(rows11u)
    print("  unmasked best    : thr=%.3f F1=%.3f" % (b11u[0], b11u[6]))
    au = np.array(rows11u, float)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(a[:, 0], a[:, 6], "o-", label="masked correlation")
    ax.plot(au[:, 0], au[:, 6], "s-", label="unmasked correlation")
    ax.set_xlabel("correlation threshold"); ax.set_ylabel("F1")
    ax.set_ylim(-0.02, 1.05); ax.legend(); ax.set_title("effect of the correlation mask")
    fig.savefig(os.path.join(FIG, "fig04_mask_effect.png")); plt.close(fig)
    E.write_csv(os.path.join(TAB, "e2_mask_comparison.csv"),
                ["variant", "best_threshold", "precision", "recall", "f1", "seconds_per_image"],
                [["masked", b11[0], round(b11[4], 4), round(b11[5], 4), round(b11[6], 4),
                  round(t_dev11, 3)],
                 ["unmasked", b11u[0], round(b11u[4], 4), round(b11u[5], 4), round(b11u[6], 4),
                  round(t_dev11u, 3)]])
    SUMMARY["E2"] = {"masked_best": list(b11), "unmasked_best": list(b11u)}

    # ==================================================================
    # E3  size of the scale pyramid
    # ==================================================================
    print("E3 scale pyramid")
    dev5, t_dev5 = scored_detections(dev_img, bank5, True)
    rows5 = sweep(dev5, dev_ann)
    b5 = best_row(rows5)
    e3 = [["1 scale (reference size only)", len(bank1), b1[0], round(b1[4], 4), round(b1[5], 4),
           round(b1[6], 4), round(t_dev1, 3)],
          ["5 scales (0.6 to 1.8)", len(bank5), b5[0], round(b5[4], 4), round(b5[5], 4),
           round(b5[6], 4), round(t_dev5, 3)],
          ["11 scales (0.50 to 2.02)", len(bank11), b11[0], round(b11[4], 4), round(b11[5], 4),
           round(b11[6], 4), round(t_dev11, 3)]]
    E.write_csv(os.path.join(TAB, "e3_scale_pyramid.csv"),
                ["configuration", "templates_in_bank", "best_threshold", "precision",
                 "recall", "f1", "seconds_per_image"], e3)
    for r in e3:
        print("  %-32s F1=%.3f  %.2f s/image" % (r[0], r[5], r[6]))
    SUMMARY["E3"] = e3

    # ==================================================================
    # E4  rotated template bank
    # ==================================================================
    print("E4 rotated template bank")
    devr, t_devr = scored_detections(dev_img, bank_rot, True)
    rowsr = sweep(devr, dev_ann)
    E.write_csv(os.path.join(TAB, "e4_threshold_rotation_bank_dev.csv"), SWEEP_HEADER, fmt(rowsr))
    br = best_row(rowsr)
    rot_no = E.evaluate_split(at_threshold(dev11, b11[0]), dev_ann)["per_condition"]
    rot_yes = E.evaluate_split(at_threshold(devr, br[0]), dev_ann)["per_condition"]
    e4 = []
    for label, bank, best, per in (("11 scales without rotation", bank11, b11, rot_no),
                                   ("11 scales with 8 angles", bank_rot, br, rot_yes)):
        rc = per.get("rotation", {"recall": 0.0, "f1": 0.0})
        vc = per.get("viewpoint", {"recall": 0.0, "f1": 0.0})
        e4.append([label, len(bank), best[0], round(best[6], 4), round(rc["recall"], 4),
                   round(vc["recall"], 4),
                   round(t_dev11 if "without rotation" in label else t_devr, 3)])
    E.write_csv(os.path.join(TAB, "e4_rotation_bank.csv"),
                ["configuration", "templates_in_bank", "best_threshold", "f1_overall",
                 "recall_rotated_images", "recall_viewpoint_images", "seconds_per_image"], e4)
    for r in e4:
        print("  %-24s F1=%.3f rot recall=%.3f  %.2f s/image" % (r[0], r[3], r[4], r[6]))
    SUMMARY["E4"] = e4

    # ==================================================================
    # E5  explicit sliding window detector
    # ==================================================================
    print("E5 sliding window")
    subset = ["dev01_baseline_a.jpg", "dev03_baseline_two.jpg", "dev04_scale_small.jpg",
              "dev11_illum_bright.jpg", "dev20_negative.jpg"]
    sub_ann = {k: dev_ann[k] for k in subset}
    win_sets = {
        "3 window sizes (0.6 / 1.0 / 1.6)": [0.6, 1.0, 1.6],
        "5 window sizes (0.6 to 1.8)": SCALES_5,
        "1 window size (1.0)": [1.0],
    }
    steps = [2, 4, 8, 16, 24]
    e5_rows = []
    for wname, wscales in win_sets.items():
        sizes = D.window_sizes_from_scales(tmpl.shape, wscales)
        for step in steps:
            per_image, windows, secs = {}, 0, 0.0
            raw = {}
            for name in subset:
                b, s, info = D.sliding_window_detect(
                    dev_img[name]["grey"], tmpl, mask, sizes, step, 0.50)
                raw[name] = (b, s)
                windows += info["windows"]
                secs += info["seconds"]
            best = None
            for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                res = {n: [bb for bb, ss in zip(*raw[n]) if ss >= thr] for n in subset}
                r = E.evaluate_split(res, sub_ann)
                if best is None or r["f1"] > best[1]["f1"]:
                    best = (thr, r)
            thr, r = best
            e5_rows.append([wname, len(sizes), step, windows, round(secs / len(subset), 3),
                            thr, r["tp"], r["fp"], r["fn"], round(r["precision"], 4),
                            round(r["recall"], 4), round(r["f1"], 4)])
            print("  %-24s step=%2d windows=%7d %.2f s/image F1=%.3f"
                  % (wname, step, windows, secs / len(subset), r["f1"]))
    E.write_csv(os.path.join(TAB, "e5_sliding_window.csv"),
                ["window_set", "window_sizes", "step_px", "windows_evaluated",
                 "seconds_per_image", "best_threshold", "true_positives",
                 "false_positives", "missed_detections", "precision", "recall", "f1"], e5_rows)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for wname in win_sets:
        r = [x for x in e5_rows if x[0] == wname]
        ax[0].plot([x[2] for x in r], [x[4] for x in r], "o-", label=wname)
        ax[1].plot([x[2] for x in r], [x[11] for x in r], "o-", label=wname)
    ax[0].set_xlabel("step size in pixels"); ax[0].set_ylabel("seconds per image")
    ax[0].set_yscale("log"); ax[0].set_title("cost of the sliding window search"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("step size in pixels"); ax[1].set_ylabel("F1")
    ax[1].set_ylim(-0.02, 1.05); ax[1].set_title("accuracy against step size"); ax[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "fig05_sliding_window.png")); plt.close(fig)
    SUMMARY["E5"] = e5_rows

    # sliding window threshold study at a fixed geometry
    sizes = D.window_sizes_from_scales(tmpl.shape, [0.6, 1.0, 1.6])
    raw = {}
    for name in subset:
        b, s, info = D.sliding_window_detect(dev_img[name]["grey"], tmpl, mask, sizes, 4, 0.40)
        raw[name] = (b, s)
    rows_sw = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        res = {n: [bb for bb, ss in zip(*raw[n]) if ss >= thr] for n in subset}
        r = E.evaluate_split(res, sub_ann)
        rows_sw.append([thr, r["tp"], r["fp"], r["fn"], round(r["precision"], 4),
                        round(r["recall"], 4), round(r["f1"], 4)])
    E.write_csv(os.path.join(TAB, "e5b_sliding_window_threshold.csv"), SWEEP_HEADER, rows_sw)

    # ==================================================================
    # E6  SIFT feature matching with RANSAC
    # ==================================================================
    print("E6 feature matching")
    e6_rows, best_fm = [], None
    for ratio in [0.70, 0.75, 0.80]:
        for min_inl in [8, 10, 12, 15, 20]:
            res, secs = {}, 0.0
            for name, im in dev_img.items():
                t0 = time.perf_counter()
                b, s, info = D.feature_match_detect(im["grey"], tmpl, ratio=ratio,
                                                    min_inliers=min_inl)
                secs += time.perf_counter() - t0
                res[name] = b
            r = E.evaluate_split(res, dev_ann)
            e6_rows.append([ratio, min_inl, r["tp"], r["fp"], r["fn"], round(r["precision"], 4),
                            round(r["recall"], 4), round(r["f1"], 4),
                            round(secs / len(dev_img), 3)])
            if best_fm is None or r["f1"] > best_fm[2]:
                best_fm = (ratio, min_inl, r["f1"])
    E.write_csv(os.path.join(TAB, "e6_feature_matching.csv"),
                ["lowe_ratio", "min_inliers", "true_positives", "false_positives",
                 "missed_detections", "precision", "recall", "f1", "seconds_per_image"], e6_rows)
    print("  best feature configuration: ratio=%.2f min_inliers=%d F1=%.3f" % best_fm)
    SUMMARY["E6"] = {"best": list(best_fm), "rows": e6_rows}

    # ==================================================================
    # E7  colour proposals combined with template matching
    # ==================================================================
    print("E7 colour proposals and template matching")
    hyb, secs, fracs, roi_counts = {}, 0.0, [], []
    for name, im in dev_img.items():
        t0 = time.perf_counter()
        b, s, info = D.hybrid_detect(im["bgr"], im["grey"], bank11, BASE_THRESHOLD)
        secs += time.perf_counter() - t0
        hyb[name] = {"boxes": b, "scores": s}
        fracs.append(info["search_fraction"])
        roi_counts.append(info["rois"])
    rows7 = sweep(hyb, dev_ann)
    E.write_csv(os.path.join(TAB, "e7_threshold_hybrid_dev.csv"), SWEEP_HEADER, fmt(rows7))
    b7 = best_row(rows7)
    e7 = [["multi scale template matching over the whole image", round(t_dev11, 3), 1.0,
           b11[0], round(b11[4], 4), round(b11[5], 4), round(b11[6], 4)],
          ["colour proposals then template matching", round(secs / len(dev_img), 3),
           round(float(np.mean(fracs)), 4), b7[0], round(b7[4], 4), round(b7[5], 4),
           round(b7[6], 4)]]
    E.write_csv(os.path.join(TAB, "e7_hybrid.csv"),
                ["configuration", "seconds_per_image", "fraction_of_image_searched",
                 "best_threshold", "precision", "recall", "f1"], e7)
    print("  proposals cover %.1f%% of the image, %.2f s/image, F1=%.3f"
          % (100 * np.mean(fracs), secs / len(dev_img), b7[6]))
    SUMMARY["E7"] = {"rows": e7, "mean_rois": float(np.mean(roi_counts))}

    # ==================================================================
    # E8  final evaluation on the held out test set
    # ==================================================================
    print("E8 held out test set")
    methods = {}

    t1, s1 = scored_detections(test_img, bank1, True)
    methods["single scale template matching"] = (at_threshold(t1, b1[0]), s1, b1[0])

    t11, s11 = scored_detections(test_img, bank11, True)
    methods["multi scale template matching"] = (at_threshold(t11, b11[0]), s11, b11[0])

    tr, sr = scored_detections(test_img, bank_rot, True)
    methods["multi scale and rotation bank"] = (at_threshold(tr, br[0]), sr, br[0])

    sizes = D.window_sizes_from_scales(tmpl.shape, [0.6, 1.0, 1.6])
    sw_res, sw_secs = {}, 0.0
    for name, im in test_img.items():
        b, s, info = D.sliding_window_detect(im["grey"], tmpl, mask, sizes, 4, 0.55)
        sw_res[name] = b
        sw_secs += info["seconds"]
    methods["explicit sliding window (step 4)"] = (sw_res, sw_secs / len(test_img), 0.55)

    fm_res, fm_secs, fm_info = {}, 0.0, {}
    for name, im in test_img.items():
        t0 = time.perf_counter()
        b, s, info = D.feature_match_detect(im["grey"], tmpl, ratio=best_fm[0],
                                            min_inliers=best_fm[1])
        fm_secs += time.perf_counter() - t0
        fm_res[name] = b
        fm_info[name] = {"boxes": b, "scores": s, "info": info}
    methods["SIFT matching with RANSAC"] = (fm_res, fm_secs / len(test_img), best_fm[1])

    hy_res, hy_secs = {}, 0.0
    for name, im in test_img.items():
        t0 = time.perf_counter()
        b, s, info = D.hybrid_detect(im["bgr"], im["grey"], bank11, b7[0])
        hy_secs += time.perf_counter() - t0
        hy_res[name] = b
    methods["colour proposals and template matching"] = (hy_res, hy_secs / len(test_img), b7[0])

    # combination: colour guided template matching plus feature verification
    comb_res, comb_secs = {}, 0.0
    for name, im in test_img.items():
        t0 = time.perf_counter()
        b, s, info = D.hybrid_detect(im["bgr"], im["grey"], bank11, b7[0])
        fb = fm_info[name]["boxes"]
        merged_b = list(b) + list(fb)
        merged_s = list(s) + [0.6 + 0.4 * x for x in fm_info[name]["scores"]]
        keep = D.nms_indices(merged_b, merged_s, 0.30)
        comb_res[name] = [merged_b[i] for i in keep]
        comb_secs += time.perf_counter() - t0
    methods["combined colour and template and features"] = (
        comb_res, comb_secs / len(test_img) + fm_secs / len(test_img), None)

    e8_rows, per_cond_table = [], {}
    for label, (res, secs, thr) in methods.items():
        r = E.evaluate_split(res, test_ann)
        e8_rows.append([label, thr if thr is not None else "n/a", r["tp"], r["fp"], r["fn"],
                        round(r["precision"], 4), round(r["recall"], 4), round(r["f1"], 4),
                        round(r["mean_iou_of_true_positives"], 4), round(secs, 3)])
        per_cond_table[label] = r["per_condition"]
        print("  %-42s P=%.3f R=%.3f F1=%.3f  %.2f s/image"
              % (label, r["precision"], r["recall"], r["f1"], secs))
    E.write_csv(os.path.join(TAB, "e8_test_set_comparison.csv"),
                ["method", "operating_threshold", "true_positives", "false_positives",
                 "missed_detections", "precision", "recall", "f1",
                 "mean_iou_of_true_positives", "seconds_per_image"], e8_rows)
    SUMMARY["E8"] = e8_rows

    conds = sorted({c for v in per_cond_table.values() for c in v if c != "negative"})
    rows_pc = []
    for label, per in per_cond_table.items():
        rows_pc.append([label] + [round(per.get(c, {"f1": 0.0})["f1"], 3) for c in conds])
    E.write_csv(os.path.join(TAB, "e8b_per_condition_f1.csv"), ["method"] + conds, rows_pc)

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    width = 0.8 / len(rows_pc)
    xs = np.arange(len(conds))
    for i, row in enumerate(rows_pc):
        ax.bar(xs + i * width, row[1:], width, label=row[0])
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
    ax.set_title("F1 by imaging condition, held out test set")
    ax.legend(fontsize=6.5, ncol=2)
    fig.savefig(os.path.join(FIG, "fig08_per_condition.png")); plt.close(fig)

    # ==================================================================
    # E9  qualitative material
    # ==================================================================
    print("E9 figures and per image output")

    # pipeline stages on one image
    name = "dev01_baseline_a.jpg"
    grey = dev_img[name]["grey"]
    resp = D.response_map(grey, tmpl, mask, True)
    raw_boxes, raw_scores = D._peaks_from_response(resp, tmpl.shape[1], tmpl.shape[0], 0.45)
    kept_b, kept_s = D.non_max_suppression(raw_boxes, raw_scores, 0.30)
    stage_pre = D.draw_boxes(dev_img[name]["bgr"], raw_boxes[:60], None, (0, 165, 255), None, 1)
    stage_post = D.draw_boxes(dev_img[name]["bgr"], kept_b, kept_s, (0, 220, 0), "logo")
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.4))
    ax[0][0].imshow(cv2.cvtColor(dev_img[name]["bgr"], cv2.COLOR_BGR2RGB)); ax[0][0].set_title("1. input scene")
    ax[0][1].imshow(grey, cmap="gray"); ax[0][1].set_title("2. grey scale input")
    im2 = ax[0][2].imshow(resp, cmap="inferno"); ax[0][2].set_title("3. correlation surface")
    fig.colorbar(im2, ax=ax[0][2], fraction=0.035)
    ax[1][0].imshow(resp >= 0.45, cmap="gray"); ax[1][0].set_title("4. surface above the threshold")
    ax[1][1].imshow(cv2.cvtColor(stage_pre, cv2.COLOR_BGR2RGB)); ax[1][1].set_title("5. candidates before suppression")
    ax[1][2].imshow(cv2.cvtColor(stage_post, cv2.COLOR_BGR2RGB)); ax[1][2].set_title("6. final localisation")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.savefig(os.path.join(FIG, "fig06_pipeline_stages.png")); plt.close(fig)

    # colour proposal figure
    name = "dev17_clutter.jpg"
    rois, cmask = D.colour_proposals(dev_img[name]["bgr"])
    vis = dev_img[name]["bgr"].copy()
    for x, y, w, h in rois:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 160, 0), 2)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    ax[0].imshow(cv2.cvtColor(dev_img[name]["bgr"], cv2.COLOR_BGR2RGB)); ax[0].set_title("scene with clutter")
    ax[1].imshow(cmask, cmap="gray"); ax[1].set_title("HSV blue mask after morphology")
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[2].set_title("region proposals")
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.savefig(os.path.join(FIG, "fig07_colour_proposals.png")); plt.close(fig)

    # SIFT match figure with RANSAC inliers
    name = "test06_rot_medium.jpg"
    g = test_img[name]["grey"]
    b, s, info = D.feature_match_detect(g, tmpl, ratio=best_fm[0], min_inliers=best_fm[1])
    good = info.get("good_match_objects", [])
    vis = cv2.drawMatches(tmpl, info["template_kp"], test_img[name]["bgr"], info["scene_kp"],
                          good[:60], None,
                          matchColor=(0, 220, 0), singlePointColor=(200, 200, 200),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title("SIFT correspondences kept by the ratio test, %s" % name.replace(".jpg", ""))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.savefig(os.path.join(FIG, "fig09_sift_matches.png")); plt.close(fig)

    # per image detection output for the whole test set, best method
    best_label = max(e8_rows, key=lambda r: r[7])[0]
    best_res = methods[best_label][0]
    for name, im in test_img.items():
        vis = im["bgr"].copy()
        for o in test_ann[name]["objects"]:
            x, y, w, h = o["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        vis = D.draw_boxes(vis, best_res.get(name, []), None, (0, 200, 0), "UG logo")
        cv2.imwrite(os.path.join(DET, name), vis)
    print("  best method on the test set: %s" % best_label)

    # success and failure montage
    def montage(names, results, path, title):
        cols = 3
        rows = int(np.ceil(len(names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.4 * rows))
        for a, nm in zip(np.ravel(axes), names):
            vis = test_img[nm]["bgr"].copy()
            for o in test_ann[nm]["objects"]:
                x, y, w, h = o["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
            vis = D.draw_boxes(vis, results.get(nm, []), None, (0, 200, 0), None)
            a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            a.set_title("%s (%s)" % (nm.replace(".jpg", ""), test_ann[nm]["condition"]), fontsize=7)
        for a in np.ravel(axes):
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.suptitle(title, fontsize=10)
        fig.savefig(path); plt.close(fig)

    tm_res = methods["multi scale template matching"][0]
    per_img = E.evaluate_split(tm_res, test_ann)["per_image"]
    fails = [n for n, v in per_img.items() if v["fn"] > 0 or v["fp"] > 0][:6]
    wins = [n for n, v in per_img.items() if v["fn"] == 0 and v["fp"] == 0][:6]
    if wins:
        montage(wins, tm_res, os.path.join(FIG, "fig10_success.png"),
                "multi scale template matching: correct cases (yellow is ground truth)")
    if fails:
        montage(fails, tm_res, os.path.join(FIG, "fig11_failure.png"),
                "multi scale template matching: failure cases (yellow is ground truth)")

    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(SUMMARY, fh, indent=2, default=str)
    print("\nall results written to results/")


if __name__ == "__main__":
    main()
