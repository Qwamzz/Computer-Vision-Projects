"""
visualisation.py
================
Figures for the report: keypoints, correspondences before and after RANSAC,
warped-image overlays and the robustness plots.

Task 6 ("display the initial feature correspondences") and Task 11 ("compare the
matching results before and after RANSAC") are served by
:func:`draw_matches_before_after`, which puts the two states side by side and
colour-codes inliers and outliers.
"""

from __future__ import annotations

import os

import cv2
import matplotlib
matplotlib.use("Agg")            # headless backend: write files, open no window
import matplotlib.pyplot as plt
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 210, 235)


def _ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save(img, path):
    cv2.imwrite(_ensure(path), img)
    return path


# --------------------------------------------------------------------------- #
# Keypoints
# --------------------------------------------------------------------------- #

def draw_keypoints(img, kps, path=None, rich=True):
    """Draw keypoints; ``rich`` also shows each keypoint's scale and orientation.

    The rich form is worth including in the report because it makes the
    difference between the detectors visible at a glance: SIFT circles vary
    greatly in radius (many scales), whereas Shi-Tomasi circles are all the same
    size (single scale).
    """
    flags = (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich
             else cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    out = cv2.drawKeypoints(img, kps, None, color=GREEN, flags=flags)
    if path:
        save(out, path)
    return out


# --------------------------------------------------------------------------- #
# Correspondences
# --------------------------------------------------------------------------- #

def draw_correspondences(img1, kps1, img2, kps2, matches, inlier_mask=None,
                         max_draw=200, title=None, seed=0):
    """Side-by-side image with lines joining matched keypoints.

    If ``inlier_mask`` is given, RANSAC inliers are drawn green and outliers red,
    which is the clearest way to show what RANSAC actually removed.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H, W = max(h1, h2), w1 + w2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    n = len(matches)
    if n == 0:
        return canvas

    idx = np.arange(n)
    if n > max_draw:
        rng = np.random.default_rng(seed)
        if inlier_mask is not None:
            # Sample inliers and outliers proportionally so that both remain
            # visible rather than one class swamping the picture.
            inl = idx[inlier_mask]
            outl = idx[~inlier_mask]
            k_in = min(len(inl), int(max_draw * 0.7))
            k_out = min(len(outl), max_draw - k_in)
            idx = np.concatenate([rng.choice(inl, k_in, replace=False) if k_in else [],
                                  rng.choice(outl, k_out, replace=False) if k_out else []]
                                 ).astype(int)
        else:
            idx = rng.choice(idx, max_draw, replace=False)

    for i in idx:
        p1 = tuple(np.round(kps1[matches[i, 0]].pt).astype(int))
        p2 = np.round(kps2[matches[i, 1]].pt).astype(int)
        p2 = (int(p2[0] + w1), int(p2[1]))
        if inlier_mask is None:
            colour = YELLOW
        else:
            colour = GREEN if inlier_mask[i] else RED
        cv2.line(canvas, p1, p2, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, colour, 1, cv2.LINE_AA)

    if title:
        cv2.rectangle(canvas, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_matches_before_after(img1, kps1, img2, kps2, matches, inlier_mask,
                              path, detector_name=""):
    """Stack the 'all matches' and 'RANSAC inliers only' figures vertically."""
    n_all = len(matches)
    n_in = int(inlier_mask.sum())
    ratio = 100.0 * n_in / n_all if n_all else 0.0

    before = draw_correspondences(
        img1, kps1, img2, kps2, matches, None,
        title="%s  BEFORE RANSAC: %d putative matches" % (detector_name, n_all))
    after = draw_correspondences(
        img1, kps1, img2, kps2, matches, inlier_mask,
        title="%s  AFTER RANSAC: %d inliers (green) / %d outliers (red), %.1f%% inliers"
              % (detector_name, n_in, n_all - n_in, ratio))

    gap = np.full((12, before.shape[1], 3), 40, dtype=np.uint8)
    out = np.vstack([before, gap, after])
    save(out, path)
    return out


# --------------------------------------------------------------------------- #
# Alignment check
# --------------------------------------------------------------------------- #

def checkerboard_overlay(img_a, img_b_warped, cell=48):
    """Interleave two aligned images in a checkerboard pattern.

    Misalignment shows up as broken lines at the cell boundaries, which is a far
    more sensitive test of alignment quality than looking at a blended result
    where feathering hides small errors.
    """
    h, w = img_a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy // cell) + (xx // cell)) % 2 == 0)[..., None]
    return np.where(mask, img_a, img_b_warped).astype(np.uint8)


def difference_map(img_a, img_b_warped, path=None, amplify=3.0):
    """Absolute difference between two aligned images, contrast-amplified."""
    a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(img_b_warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.clip(np.abs(a - b) * amplify, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    if path:
        save(out, path)
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_robustness(results, x_key, x_label, out_path, title,
                    y_keys=(("inlier_ratio", "Inlier ratio"),
                            ("n_inliers", "Number of inliers"),
                            ("grid_error_px", "Alignment error (px)"))):
    """One figure per robustness factor, with one panel per metric.

    ``results`` is a list of dicts each containing 'detector', ``x_key`` and the
    metric keys.
    """
    detectors = sorted({r["detector"] for r in results})
    fig, axes = plt.subplots(1, len(y_keys), figsize=(5.2 * len(y_keys), 4.0))
    if len(y_keys) == 1:
        axes = [axes]

    for ax, (key, ylabel) in zip(axes, y_keys):
        for det in detectors:
            rows = sorted([r for r in results if r["detector"] == det],
                          key=lambda r: r[x_key])
            xs = [r[x_key] for r in rows]
            ys = [r.get(key, np.nan) for r in rows]
            ax.plot(xs, ys, marker="o", label=det, linewidth=1.8, markersize=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if key == "grid_error_px":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), dpi=130)
    plt.close(fig)
    return out_path


def plot_detector_summary(rows, out_path):
    """Bar charts summarising the main detector-comparison table."""
    dets = [r["detector"] for r in rows]
    x = np.arange(len(dets))
    metrics = [("mean_keypoints", "Keypoints per image", None),
               ("mean_inlier_ratio", "Inlier ratio", None),
               ("total_time_ms", "Total pipeline time (ms)", None),
               ("mean_grid_error_px", "Alignment error vs GT (px)", "log")]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.0))
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (key, label, scale) in zip(axes, metrics):
        vals = [r.get(key, np.nan) for r in rows]
        ax.bar(x, vals, color=colours[:len(dets)])
        ax.set_xticks(x)
        ax.set_xticklabels(dets, rotation=12, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if scale == "log" and np.nanmax(vals) > 0:
            ax.set_yscale("log")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, ("%.3g" % v), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Detector / descriptor comparison")
    fig.tight_layout()
    fig.savefig(_ensure(out_path), dpi=130)
    plt.close(fig)
    return out_path


def montage(images, titles=None, cols=2, cell_w=420, path=None):
    """Simple grid montage used for the input views and the condition tests."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    thumbs = []
    for i, im in enumerate(images):
        s = cell_w / im.shape[1]
        t = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if titles and i < len(titles):
            cv2.rectangle(t, (0, 0), (t.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(t, titles[i], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)

    cell_h = max(t.shape[0] for t in thumbs)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * cell_h:r * cell_h + t.shape[0],
               c * cell_w:c * cell_w + t.shape[1]] = t
    if path:
        save(canvas, path)
    return canvas
