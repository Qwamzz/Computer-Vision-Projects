"""
evaluate.py
Quantitative evaluation against the manual ground truth.

Segmentation is scored per pixel against the hand traced masks:

    IoU       = TP / (TP + FP + FN)
    precision = TP / (TP + FP)        of the pixels called road, how many were
    recall    = TP / (TP + FN)        of the road pixels, how many were found
    F1        = 2PR / (P + R)

IoU is reported first because it is the measure that punishes both kinds of
mistake at once and cannot be inflated by predicting a very large or a very
small region, which precision and recall can be individually.

Boundary accuracy is scored separately in lanes.boundary_error, in pixels,
because a detector can enclose the right area while placing its edges badly.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np


def confusion(pred, gt):
    p = pred > 0
    g = gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return tp, fp, fn, tn


def scores(pred, gt):
    tp, fp, fn, tn = confusion(pred, gt)
    iou = tp / float(tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def mean_scores(rows, keys=("iou", "precision", "recall", "f1")):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out["mean_" + k] = float(np.mean(vals)) if vals else 0.0
    return out


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([("%.4f" % v) if isinstance(v, float) else v for v in r])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
