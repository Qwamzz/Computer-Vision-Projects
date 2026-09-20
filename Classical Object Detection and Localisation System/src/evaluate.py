"""
evaluate.py
Classical Object Detection and Localisation

Detection evaluation utilities: greedy matching of detections to ground
truth at a fixed intersection over union, and the standard detection
measures precision, recall and F1 score.

A detection counts as a true positive when its intersection over union with
an unmatched ground truth box is at least IOU_MATCH. Every remaining
detection is a false positive and every unmatched ground truth object is a
false negative, that is a missed detection.
"""

import csv
import json
import os

from detector import iou

IOU_MATCH = 0.5


def match_detections(dets, gts, iou_match=IOU_MATCH):
    """Greedily match detections (highest score first) to ground truth boxes.

    Returns (tp, fp, fn, per_detection_flags, matched_ious).
    """
    used = [False] * len(gts)
    flags, ious = [], []
    tp = 0
    for d in dets:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(d, g)
            if v > best:
                best, best_j = v, j
        if best >= iou_match and best_j >= 0:
            used[best_j] = True
            tp += 1
            flags.append(True)
            ious.append(best)
        else:
            flags.append(False)
            ious.append(best)
    fp = len(dets) - tp
    fn = used.count(False)
    return tp, fp, fn, flags, ious


def prf(tp, fp, fn):
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_split(results, annotations, iou_match=IOU_MATCH):
    """Aggregate over a whole split and also per condition.

    `results` maps image name to a list of detection boxes ordered by score.
    """
    tot = {"tp": 0, "fp": 0, "fn": 0}
    per_cond = {}
    per_image = {}
    all_ious = []
    for name, meta in annotations.items():
        gts = [o["bbox"] for o in meta["objects"]]
        dets = results.get(name, [])
        tp, fp, fn, flags, ious = match_detections(dets, gts, iou_match)
        all_ious.extend([v for v, f in zip(ious, flags) if f])
        tot["tp"] += tp
        tot["fp"] += fp
        tot["fn"] += fn
        c = per_cond.setdefault(meta["condition"], {"tp": 0, "fp": 0, "fn": 0})
        c["tp"] += tp
        c["fp"] += fp
        c["fn"] += fn
        per_image[name] = {"tp": tp, "fp": fp, "fn": fn,
                           "condition": meta["condition"]}
    p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
    out = {"tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"],
           "precision": p, "recall": r, "f1": f,
           "mean_iou_of_true_positives": (sum(all_ious) / len(all_ious)) if all_ious else 0.0,
           "per_condition": {}, "per_image": per_image}
    for cond, c in sorted(per_cond.items()):
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        out["per_condition"][cond] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                                      "precision": p, "recall": r, "f1": f}
    return out


def load_annotations(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path, header, rows):
    """Write a table, quoting any field that contains a comma."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([str(v) for v in row])
