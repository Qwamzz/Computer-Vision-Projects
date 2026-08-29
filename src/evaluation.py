"""
Quantitative evaluation against manually verified ground truth masks.

Pixel level measures
    Intersection over Union, precision, recall, F1 score and pixel accuracy.

Object level measures
    Predicted count, ground truth count, count error, and a greedy one to one
    matching of predicted regions to ground truth regions at an IoU threshold,
    from which object level precision, recall and F1 are computed.
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Pixel level
# ----------------------------------------------------------------------------
def pixel_scores(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    accuracy = (tp + tn) / float(pred.size)
    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "dice": round(dice, 4),
        "pixel_accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ----------------------------------------------------------------------------
# Object level
# ----------------------------------------------------------------------------
def _components(mask, min_area=30):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    keep = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep.append(labels == i)
    return keep


def object_scores(pred_mask, gt_mask, iou_threshold=0.5, min_area=30):
    """Greedy one to one matching of predicted regions to ground truth regions."""
    preds = _components(pred_mask, min_area)
    gts = _components(gt_mask, min_area)
    if not preds or not gts:
        return {
            "pred_count": len(preds), "gt_count": len(gts),
            "count_error": len(preds) - len(gts),
            "obj_precision": 0.0, "obj_recall": 0.0, "obj_f1": 0.0,
            "mean_matched_iou": 0.0,
        }

    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            inter = np.logical_and(p, g).sum()
            if inter == 0:
                continue
            union = np.logical_or(p, g).sum()
            pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)

    used_p, used_g, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_threshold or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append(iou)

    tp = len(matched)
    precision = tp / len(preds)
    recall = tp / len(gts)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "pred_count": len(preds), "gt_count": len(gts),
        "count_error": len(preds) - len(gts),
        "obj_precision": round(precision, 4),
        "obj_recall": round(recall, 4),
        "obj_f1": round(f1, 4),
        "mean_matched_iou": round(float(np.mean(matched)) if matched else 0.0, 4),
    }


def evaluate(pred_mask, gt_mask, iou_threshold=0.5, min_area=30):
    out = pixel_scores(pred_mask, gt_mask)
    out.update(object_scores(pred_mask, gt_mask, iou_threshold, min_area))
    return out


# ----------------------------------------------------------------------------
# Measurement accuracy
# ----------------------------------------------------------------------------
def measurement_errors(pred_mask, gt_mask, px_per_mm=None, iou_threshold=0.5,
                       min_area=30):
    """Compare the measurements of matched objects with the ground truth.

    Predicted regions are matched to ground truth regions with the same greedy
    one to one rule as object_scores, and for every matched pair the relative
    error of the area and of the major axis length is computed. This answers
    the question that matters for a measurement system: given that an object
    was found, how accurate is the number reported for it.
    """
    import measurement as _M

    pred_rows = _M.measure_objects(pred_mask, px_per_mm, min_area=min_area)
    gt_rows = _M.measure_objects(gt_mask, px_per_mm, min_area=min_area)
    preds = _components(pred_mask, min_area)
    gts = _components(gt_mask, min_area)

    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            inter = np.logical_and(p, g).sum()
            if inter == 0:
                continue
            pairs.append((inter / np.logical_or(p, g).sum(), i, j))
    pairs.sort(reverse=True)

    used_p, used_g, records = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_threshold or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pr, gr = pred_rows[i], gt_rows[j]
        records.append({
            "iou": round(float(iou), 4),
            "area_pred": pr["area_px"], "area_gt": gr["area_px"],
            "area_error_pct": round(100.0 * (pr["area_px"] - gr["area_px"])
                                    / max(1, gr["area_px"]), 2),
            "major_pred": pr["major_axis_px"], "major_gt": gr["major_axis_px"],
            "major_error_pct": round(100.0 * (pr["major_axis_px"] - gr["major_axis_px"])
                                     / max(1e-6, gr["major_axis_px"]), 2),
        })
    if not records:
        return {"matched": 0}
    area_err = np.array([abs(r["area_error_pct"]) for r in records])
    major_err = np.array([abs(r["major_error_pct"]) for r in records])
    return {
        "matched": len(records),
        "mean_abs_area_error_pct": round(float(area_err.mean()), 2),
        "max_abs_area_error_pct": round(float(area_err.max()), 2),
        "mean_abs_major_error_pct": round(float(major_err.mean()), 2),
        "max_abs_major_error_pct": round(float(major_err.max()), 2),
        "records": records,
    }
