"""
run_experiments.py
Runs every experiment and writes the tables, figures and overlays.

    python src/run_experiments.py

Protocol. The twelve photographs are split in half. The six images whose name
ends in _01 are the tuning set: every parameter reported below was chosen on
those and only those. The six ending in _02 are held out and were run once,
at the end, with the parameters already fixed. Both halves cover all six
imaging conditions, so the held out half is not easier than the tuning half.

Experiments
    E1  colour representation, measured by class separability
    E2  filtering: four filters against boundary survival and noise
    E3  edge detection: Canny against Sobel, and their threshold sweeps
    E4  the region of interest: what it removes and what it costs
    E5  lane boundary fitting and the Hough parameters
    E6  segmentation: three approaches and their parameters
    E7  the assembled pipeline, on the tuning half and the held out half
"""

from __future__ import annotations

import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import edges as E
import evaluate as EV
import lanes as L
import preprocessing as P
import segmentation as S

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "data", "images")
ANN_DIR = os.path.join(ROOT, "data", "annotations")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
OVL = os.path.join(RES, "overlays")
for d in (RES, FIG, TAB, OVL):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

CONDITION = {
    "daylight": "good daylight", "shadows": "shadows",
    "poor_illumination": "poor illumination", "curve": "curved road",
    "worn_markings": "worn or absent markings", "occlusion": "occluding traffic",
}
SUMMARY = {}


# --------------------------------------------------------------------------
def load_dataset():
    with open(os.path.join(ROOT, "data", "sources.json"), encoding="utf-8") as fh:
        meta = {r["file"]: r for r in json.load(fh)}
    items = []
    for name in sorted(os.listdir(IMG_DIR)):
        if not name.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(ANN_DIR, name.replace(".jpg", "_road.png")), 0)
        items.append({
            "name": name, "stem": name.replace(".jpg", ""), "img": img, "gt": gt,
            "condition": meta[name]["condition"], "note": meta[name]["note"],
            "split": "tuning" if name.endswith("_01.jpg") else "held out",
        })
    return items


def grey_for(img, filt="bilateral", space="HLS", strength=1.0):
    return P.road_channel(P.apply_filter(img, filt, strength), space)


# --------------------------------------------------------------------------
def e1_colour(items):
    print("E1 colour representation")
    rows = []
    for it in items:
        row = [it["stem"], it["condition"]]
        for sp in P.COLOUR_SPACES:
            row.append(P.separability(it["img"], it["gt"], sp))
        rows.append(row)
    EV.write_csv(os.path.join(TAB, "e1_colour_separability.csv"),
                 ["image", "condition"] + P.COLOUR_SPACES, rows)
    means = {sp: float(np.mean([r[2 + i] for r in rows]))
             for i, sp in enumerate(P.COLOUR_SPACES)}
    best = max(means, key=means.get)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(list(means), [means[s] for s in means], color="#3b6ea5")
    ax.set_ylabel("mean Fisher separability")
    ax.set_title("Colour representation: road against everything else")
    fig.savefig(os.path.join(FIG, "fig02_colour.png")); plt.close(fig)
    print("   best representation: %s (%.2f)" % (best, means[best]))
    SUMMARY["E1"] = {"means": means, "best": best}
    return best


def e2_filters(items):
    """Four filters, scored the same way the edge operators are scored.

    The tempting measure is boundary recall divided by edge density, on the
    grounds that a good filter keeps the boundary and suppresses the rest. It
    is a trap, and an instructive one: a filter that blurs the image into
    porridge produces almost no edges at all, so the denominator collapses and
    the ratio goes up. Heavy Gaussian smoothing wins that measure while
    destroying nine tenths of the road boundary. Boundary F1 cannot be gamed
    that way, because a filter that removes the boundary is punished by the
    recall term.
    """
    print("E2 filtering")
    rows = []
    for filt in P.FILTERS:
        rec, prec, dens, times = [], [], [], []
        for it in items:
            t0 = time.perf_counter()
            g = grey_for(it["img"], filt)
            em = E.detect_edges(g, "canny", 50, 150)
            times.append(time.perf_counter() - t0)
            roi = E.roi_mask(it["img"].shape)
            em = E.apply_roi(em)
            rec.append(E.boundary_recall(em, it["gt"]))
            prec.append(E.boundary_precision(em, it["gt"], mask=roi))
            dens.append(E.edge_density(em, roi))
        r, p = float(np.mean(rec)), float(np.mean(prec))
        f1 = 0.0 if (r + p) == 0 else 2 * r * p / (r + p)
        rows.append([filt, r, p, f1, float(np.mean(dens)), float(np.mean(times))])
    EV.write_csv(os.path.join(TAB, "e2_filters.csv"),
                 ["filter", "boundary_recall", "boundary_precision",
                  "boundary_f1", "edge_density", "seconds_per_image"], rows)
    best = max(rows, key=lambda r: r[3])[0]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax[0].bar([r[0] for r in rows], [r[1] for r in rows], color="#3b6ea5")
    ax[0].set_ylabel("boundary recall")
    ax[0].set_title("how much of the boundary survives")
    ax[1].bar([r[0] for r in rows], [r[3] for r in rows], color="#a5643b")
    ax[1].set_ylabel("boundary F1")
    ax[1].set_title("recall and precision together")
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    fig.savefig(os.path.join(FIG, "fig04_filters.png"))
    plt.close(fig)
    print("   best filter by boundary F1: %s" % best)
    SUMMARY["E2"] = {"rows": rows, "best": best}
    return best


def e3_edges(items, filt):
    """Canny against Sobel, scored by boundary F1 rather than recall alone.

    Recall on its own is a trap: an operator that marks every pixel as an edge
    scores a perfect recall and is useless. Boundary precision, the fraction of
    reported edges that actually lie on the road boundary, is what stops that,
    and the harmonic mean of the two is what selects the operating point.
    """
    print("E3 edge detection")
    rows = []
    for method in E.EDGE_METHODS:
        settings = ([(30, 90), (50, 150), (80, 200), (120, 260)]
                    if method == "canny" else [(30,), (50,), (70,), (100,)])
        for setting in settings:
            rec, prec, dens = [], [], []
            for it in items:
                g = grey_for(it["img"], filt)
                roi = E.roi_mask(it["img"].shape)
                if method == "canny":
                    em = E.detect_edges(g, "canny", setting[0], setting[1])
                    label = "canny %d/%d" % setting
                else:
                    em = E.detect_edges(g, "sobel", sobel_thresh=setting[0])
                    label = "sobel %d" % setting[0]
                em = E.apply_roi(em)
                rec.append(E.boundary_recall(em, it["gt"]))
                prec.append(E.boundary_precision(em, it["gt"], mask=roi))
                dens.append(E.edge_density(em, roi))
            r, p = float(np.mean(rec)), float(np.mean(prec))
            f1 = 0.0 if (r + p) == 0 else 2 * r * p / (r + p)
            rows.append([label, method, r, p, f1, float(np.mean(dens))])
    EV.write_csv(os.path.join(TAB, "e3_edges.csv"),
                 ["setting", "method", "boundary_recall", "boundary_precision",
                  "boundary_f1", "edge_density"], rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for method, marker in (("canny", "o-"), ("sobel", "s--")):
        sel = [r for r in rows if r[1] == method]
        ax.plot([r[2] for r in sel], [r[3] for r in sel], marker, label=method)
        for r in sel:
            ax.annotate(r[0].split()[-1], (r[2], r[3]), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("boundary recall"); ax.set_ylabel("boundary precision")
    ax.set_title("Edge operating curves: Canny against Sobel")
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig05_edges.png")); plt.close(fig)
    best = max(rows, key=lambda r: r[4])
    print("   best operating point: %s (boundary F1 %.3f)" % (best[0], best[4]))
    SUMMARY["E3"] = {"rows": rows, "best": best[0]}
    return best[0]


def adaptive_roi_mask(img):
    """Derive the search region from a first pass segmentation.

    The fixed trapezoid encodes an assumption about where the camera points.
    An alternative is to let the image say where the road is: segment once,
    dilate the result generously, and search for boundaries only there. This
    costs one extra segmentation and removes the geometric assumption.
    """
    rough = S.segment(img, "threshold", k_sigma=3.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    return cv2.dilate(rough, k, iterations=1)


def e4_roi(items, filt):
    print("E4 region of interest")
    rows = []
    for label in ("whole frame", "fixed trapezoid", "adaptive from segmentation"):
        kept, errs, found = [], [], 0
        for it in items:
            g = grey_for(it["img"], filt)
            em = E.detect_edges(g, "canny", 50, 150)
            if label == "fixed trapezoid":
                em = E.apply_roi(em)
            elif label.startswith("adaptive"):
                em = cv2.bitwise_and(em, adaptive_roi_mask(it["img"]))
            res = L.detect_lanes(em, it["img"].shape)
            kept.append(len(res["segments"]))
            for side in ("left", "right"):
                if res[side] is not None:
                    found += 1
                    e = L.boundary_error(res[side], it["gt"], side)
                    if e is not None:
                        errs.append(e)
        rows.append([label, float(np.mean(kept)), found, 2 * len(items),
                     float(np.mean(errs)) if errs else float("nan"), len(errs)])
    EV.write_csv(os.path.join(TAB, "e4_roi.csv"),
                 ["configuration", "mean_hough_segments", "boundaries_fitted",
                  "boundaries_possible", "mean_boundary_error_px",
                  "boundaries_scored"], rows)
    for r in rows:
        print("   %-27s %6.0f segments, %d/%d fitted, error %.1f px"
              % (r[0], r[1], r[2], r[3], r[4]))
    SUMMARY["E4"] = rows
    return rows


def e5_boundaries(items, filt, edge_cfg, seg_method, seg_kw):
    """Boundary identification: three strategies, and the Hough parameters.

    Strategy A fits straight lines to Hough segments taken from the edge map
    inside the region of interest, which is the textbook lane finder.
    Strategy B first gates those edges with the dilated road paint mask, so
    that only edges lying on white or yellow markings can vote.
    Strategy C ignores the edge map and reads the boundary straight off the
    segmented road region, taking the extreme road pixel in each row.
    """
    print("E5 boundary identification")

    sweep = []
    for thr in (20, 25, 40, 60):
        for mll in (25, 40, 70):
            errs, found = [], 0
            for it in items:
                em = E.apply_roi(edge_map_for(it["img"], filt, edge_cfg))
                res = L.detect_lanes(em, it["img"].shape,
                                     {"threshold": thr, "min_line_len": mll})
                for side in ("left", "right"):
                    if res[side] is not None:
                        found += 1
                        e = L.boundary_error(res[side], it["gt"], side)
                        if e is not None:
                            errs.append(e)
            sweep.append([thr, mll, found, 2 * len(items),
                          float(np.mean(errs)) if errs else float("nan")])
    EV.write_csv(os.path.join(TAB, "e5_hough_sweep.csv"),
                 ["hough_threshold", "min_line_length", "boundaries_found",
                  "boundaries_possible", "mean_boundary_error_px"], sweep)
    ok = [r for r in sweep if not np.isnan(r[4]) and r[2] >= 2]
    best_h = min(ok, key=lambda r: r[4]) if ok else sweep[0]
    hough = {"threshold": int(best_h[0]), "min_line_len": int(best_h[1])}

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for mll in (25, 40, 70):
        sel = [r for r in sweep if r[1] == mll]
        ax.plot([r[0] for r in sel], [r[4] for r in sel], "o-",
                label="min length %d" % mll)
    ax.set_xlabel("Hough accumulator threshold")
    ax.set_ylabel("mean boundary error (px)")
    ax.set_title("Lane boundary accuracy against the Hough threshold")
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig06_hough.png"))
    plt.close(fig)

    rows = []
    for label in ("hough on ROI edges", "hough on paint edges",
                  "from segmented region"):
        errs, found, possible = [], 0, 2 * len(items)
        for it in items:
            em = E.apply_roi(edge_map_for(it["img"], filt, edge_cfg))
            if label == "hough on ROI edges":
                res = L.detect_lanes(em, it["img"].shape, hough)
            elif label == "hough on paint edges":
                res = L.detect_lanes(em, it["img"].shape, hough,
                                     restrict=L.paint_mask(it["img"]))
            else:
                res = L.boundaries_from_region(
                    S.segment(it["img"], seg_method, **seg_kw))
            for side in ("left", "right"):
                if res[side] is not None:
                    found += 1
                    e = L.boundary_error(res[side], it["gt"], side)
                    if e is not None:
                        errs.append(e)
        rows.append([label, found, possible, len(errs),
                     float(np.mean(errs)) if errs else float("nan")])
    EV.write_csv(os.path.join(TAB, "e5_boundary_strategies.csv"),
                 ["strategy", "boundaries_found", "boundaries_possible",
                  "boundaries_scored", "mean_boundary_error_px"], rows)
    for r in rows:
        txt = ("%.1f px" % r[4]) if not np.isnan(r[4]) else "n/a"
        print("   %-24s found %2d/%d  error %s" % (r[0], r[1], r[2], txt))
    scored = [r for r in rows if not np.isnan(r[4])]
    best = min(scored, key=lambda r: r[4]) if scored else rows[0]
    print("   chosen: %s (Hough threshold=%d, min length=%d)"
          % (best[0], hough["threshold"], hough["min_line_len"]))
    SUMMARY["E5"] = {"sweep": sweep, "strategies": rows, "best": best[0],
                     "hough": hough}
    return hough, best[0]


def e6_segmentation(items):
    print("E6 segmentation")
    configs = []
    for ks in (2.0, 2.5, 3.0):
        configs.append(("threshold", {"k_sigma": ks}, "threshold k=%.1f" % ks))
    for chi in (2.5, 3.0, 3.5):
        configs.append(("mahalanobis", {"chi": chi},
                        "mahalanobis chi=%.1f" % chi))
    for k in (2, 3, 4, 5):
        configs.append(("kmeans", {"k": k}, "kmeans K=%d" % k))
    for tol in (8, 14, 20, 28):
        configs.append(("region_growing", {"tolerance": tol},
                        "region growing tol=%d" % tol))
    rows = []
    for method, kw, label in configs:
        per, times = [], []
        for it in items:
            t0 = time.perf_counter()
            m = S.segment(it["img"], method, **kw)
            times.append(time.perf_counter() - t0)
            per.append(EV.scores(m, it["gt"]))
        mean = EV.mean_scores(per)
        rows.append([label, method, mean["mean_iou"], mean["mean_precision"],
                     mean["mean_recall"], mean["mean_f1"], float(np.mean(times))])
    EV.write_csv(os.path.join(TAB, "e6_segmentation.csv"),
                 ["configuration", "method", "mean_iou", "mean_precision",
                  "mean_recall", "mean_f1", "seconds_per_image"], rows)
    best = max(rows, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    colours = {"threshold": "#3b6ea5", "mahalanobis": "#6b3ba5",
               "kmeans": "#a5643b", "region_growing": "#4f8a4f"}
    ax.bar([r[0] for r in rows], [r[2] for r in rows],
           color=[colours[r[1]] for r in rows])
    ax.set_ylabel("mean IoU"); ax.tick_params(axis="x", rotation=30)
    ax.set_title("Segmentation approaches and their parameters (tuning set)")
    fig.savefig(os.path.join(FIG, "fig07_segmentation.png")); plt.close(fig)
    print("   best: %s (IoU %.3f)" % (best[0], best[2]))
    SUMMARY["E6"] = {"rows": rows, "best": best}
    return best


def edge_from_label(label):
    """Turn the winning label from E3 back into an edge configuration."""
    if label.startswith("canny"):
        lo, hi = label.split()[1].split("/")
        return {"method": "canny", "low": int(lo), "high": int(hi)}
    return {"method": "sobel", "sobel_thresh": int(label.split()[1])}


def edge_map_for(img, filt, edge_cfg):
    """Filtering and edge detection, exactly as the pipeline does it."""
    g = grey_for(img, filt)
    if edge_cfg["method"] == "canny":
        return E.detect_edges(g, "canny", edge_cfg["low"], edge_cfg["high"])
    return E.detect_edges(g, "sobel", sobel_thresh=edge_cfg["sobel_thresh"])


def seg_params(seg_best):
    """Turn the winning label from E6 back into keyword arguments."""
    method, label = seg_best[1], seg_best[0]
    value = label.split("=")[1]
    if method == "threshold":
        return {"k_sigma": float(value)}
    if method == "mahalanobis":
        return {"chi": float(value)}
    if method == "kmeans":
        return {"k": int(value)}
    return {"tolerance": int(value)}


def build_pipeline_config(filt, edge_label, hough, seg_best, boundary):
    method = seg_best[1]
    kw = seg_params(seg_best)
    edge = edge_from_label(edge_label)
    return {"filter": filt, "colour_space": "HLS", "edge": edge,
            "hough": hough, "boundary_strategy": boundary,
            "segmentation": {"method": method, "params": kw}}


def run_pipeline(img, cfg):
    """Road image -> filtering -> edges -> ROI -> boundaries -> segmentation."""
    filtered = P.apply_filter(img, cfg["filter"])
    grey = P.road_channel(filtered, cfg["colour_space"])
    e = cfg["edge"]
    if e["method"] == "canny":
        em = E.detect_edges(grey, "canny", e["low"], e["high"])
    else:
        em = E.detect_edges(grey, "sobel", sobel_thresh=e["sobel_thresh"])
    em_roi = E.apply_roi(em)
    road = S.segment(filtered, cfg["segmentation"]["method"],
                     **cfg["segmentation"]["params"])

    hough_lanes = L.detect_lanes(em_roi, img.shape, cfg["hough"])
    paint = L.paint_mask(img)
    paint_lanes = L.detect_lanes(em_roi, img.shape, cfg["hough"],
                                 restrict=paint)
    region_lanes = L.boundaries_from_region(road)
    chosen = {"hough on ROI edges": hough_lanes,
              "hough on paint edges": paint_lanes,
              "from segmented region": region_lanes}[cfg["boundary_strategy"]]

    return {"filtered": filtered, "grey": grey, "edges": em,
            "edges_roi": em_roi, "paint": paint, "lanes": chosen,
            "hough_lanes": hough_lanes, "region_lanes": region_lanes,
            "road": road}


def e7_final(items, cfg):
    print("E7 assembled pipeline")
    rows, per_cond = [], {}
    for it in items:
        t0 = time.perf_counter()
        out = run_pipeline(it["img"], cfg)
        secs = time.perf_counter() - t0
        sc = EV.scores(out["road"], it["gt"])
        el = L.boundary_error(out["lanes"]["left"], it["gt"], "left")
        er = L.boundary_error(out["lanes"]["right"], it["gt"], "right")
        berr = [v for v in (el, er) if v is not None]
        rows.append([it["stem"], it["condition"], it["split"], sc["iou"],
                     sc["precision"], sc["recall"], sc["f1"],
                     float(np.mean(berr)) if berr else float("nan"),
                     len(berr), secs])
        c = per_cond.setdefault(it["condition"], [])
        c.append(sc)
        cv2.imwrite(os.path.join(OVL, "final_%s.jpg" % it["stem"]),
                    S.overlay_result(it["img"], out["road"], out["lanes"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    EV.write_csv(os.path.join(TAB, "e7_final_per_image.csv"),
                 ["image", "condition", "split", "iou", "precision", "recall",
                  "f1", "boundary_error_px", "boundaries_found",
                  "seconds"], rows)

    cond_rows = []
    for cond, lst in sorted(per_cond.items()):
        m = EV.mean_scores(lst)
        cond_rows.append([CONDITION.get(cond, cond), len(lst), m["mean_iou"],
                          m["mean_precision"], m["mean_recall"], m["mean_f1"]])
    EV.write_csv(os.path.join(TAB, "e7_by_condition.csv"),
                 ["condition", "images", "mean_iou", "mean_precision",
                  "mean_recall", "mean_f1"], cond_rows)

    split_rows = []
    for split in ("tuning", "held out"):
        sel = [r for r in rows if r[2] == split]
        if not sel:
            continue
        split_rows.append([split, len(sel),
                           float(np.mean([r[3] for r in sel])),
                           float(np.mean([r[4] for r in sel])),
                           float(np.mean([r[5] for r in sel])),
                           float(np.mean([r[6] for r in sel])),
                           (float(np.nanmean([r[7] for r in sel]))
                            if np.any(~np.isnan([r[7] for r in sel]))
                            else float("nan"))])
    EV.write_csv(os.path.join(TAB, "e7_by_split.csv"),
                 ["split", "images", "mean_iou", "mean_precision", "mean_recall",
                  "mean_f1", "mean_boundary_error_px"], split_rows)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar([r[0] for r in cond_rows], [r[2] for r in cond_rows], color="#3b6ea5")
    ax.set_ylabel("mean IoU"); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Road region IoU by imaging condition")
    fig.savefig(os.path.join(FIG, "fig09_by_condition.png")); plt.close(fig)

    for r in split_rows:
        print("   %-9s IoU %.3f  P %.3f  R %.3f  F1 %.3f  boundary %.1f px"
              % (r[0], r[2], r[3], r[4], r[5], r[6]))
    SUMMARY["E7"] = {"per_image": rows, "by_condition": cond_rows,
                     "by_split": split_rows, "config": cfg}
    return rows, cond_rows, split_rows


# --------------------------------------------------------------------------
def figures(items, cfg):
    print("figures")
    # dataset montage
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 6.6))
    for a, it in zip(axes.ravel(), items):
        vis = it["img"].copy()
        sel = it["gt"] > 0
        vis[sel] = (np.array((0, 210, 0)) * 0.34 + vis[sel] * 0.66).astype(np.uint8)
        a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        a.set_title("%s\n%s" % (it["stem"], CONDITION.get(it["condition"], "")),
                    fontsize=7)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("The dataset with the manual road annotation in green", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig01_dataset.png")); plt.close(fig)

    # pipeline stages
    it = next(x for x in items if x["stem"] == "daylight_02")
    out = run_pipeline(it["img"], cfg)
    roi_vis = it["img"].copy()
    cv2.polylines(roi_vis, E.roi_polygon(it["img"].shape), True, (0, 180, 255), 3)
    panels = [
        (cv2.cvtColor(it["img"], cv2.COLOR_BGR2RGB), "1. road image"),
        (cv2.cvtColor(out["filtered"], cv2.COLOR_BGR2RGB), "2. filtering"),
        (out["edges"], "3. edge detection"),
        (cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB), "4. region selection"),
        (out["edges_roi"], "5. edges inside the ROI"),
        (cv2.cvtColor(L.draw_segments(it["img"], out["lanes"]), cv2.COLOR_BGR2RGB),
         "6. Hough segments"),
        (out["road"], "7. segmentation"),
        (cv2.cvtColor(S.overlay_result(it["img"], out["road"], out["lanes"]),
                      cv2.COLOR_BGR2RGB), "8. final visualisation"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 4.6))
    for a, (im, title) in zip(axes.ravel(), panels):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray")
        a.set_title(title, fontsize=8)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.savefig(os.path.join(FIG, "fig03_pipeline.png")); plt.close(fig)

    # segmentation methods side by side on two images
    fig, axes = plt.subplots(2, 1 + len(S.METHODS), figsize=(15.5, 4.8))
    for row, stem in enumerate(["shadow_01", "worn_02"]):
        it = next(x for x in items if x["stem"] == stem)
        axes[row][0].imshow(cv2.cvtColor(it["img"], cv2.COLOR_BGR2RGB))
        axes[row][0].set_title("%s" % stem, fontsize=8)
        for col, method in enumerate(S.METHODS, start=1):
            m = S.segment(it["img"], method)
            sc = EV.scores(m, it["gt"])
            axes[row][col].imshow(cv2.cvtColor(
                S.overlay_result(it["img"], m), cv2.COLOR_BGR2RGB))
            axes[row][col].set_title("%s, IoU %.2f" % (method, sc["iou"]), fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.savefig(os.path.join(FIG, "fig08_methods.png")); plt.close(fig)

    # failure montage: the four worst images by IoU
    scored = []
    for it in items:
        out = run_pipeline(it["img"], cfg)
        scored.append((EV.scores(out["road"], it["gt"])["iou"], it, out))
    scored.sort(key=lambda t: t[0])
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 2.9))
    for a, (iou, it, out) in zip(axes, scored[:4]):
        a.imshow(cv2.cvtColor(S.overlay_result(it["img"], out["road"], out["lanes"]),
                              cv2.COLOR_BGR2RGB))
        a.set_title("%s (%s)\nIoU %.2f" % (it["stem"],
                                           CONDITION.get(it["condition"], ""), iou),
                    fontsize=8)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Where the system fails", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig10_failures.png")); plt.close(fig)


def main():
    items = load_dataset()
    tuning = [it for it in items if it["split"] == "tuning"]
    print("%d images, %d tuning, %d held out\n"
          % (len(items), len(tuning), len(items) - len(tuning)))
    SUMMARY["dataset"] = {"images": len(items), "tuning": len(tuning),
                          "held_out": len(items) - len(tuning)}

    best_space = e1_colour(items)
    best_filter = e2_filters(tuning)
    best_edge = e3_edges(tuning, best_filter)
    e4_roi(tuning, best_filter)
    seg_best = e6_segmentation(tuning)
    hough, boundary = e5_boundaries(tuning, best_filter,
                                    edge_from_label(best_edge),
                                    seg_best[1], seg_params(seg_best))

    cfg = build_pipeline_config(best_filter, best_edge, hough, seg_best,
                                boundary)
    EV.write_json(os.path.join(RES, "pipeline_config.json"), cfg)
    print("\npipeline: %s\n" % json.dumps(cfg))

    e7_final(items, cfg)
    figures(items, cfg)

    SUMMARY["colour_note"] = best_space
    EV.write_json(os.path.join(RES, "summary.json"), SUMMARY)
    print("\nall results written to results/")


if __name__ == "__main__":
    main()
