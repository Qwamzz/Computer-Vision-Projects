"""
Visualisation helpers.

All figures written to results/figures are produced here, so that the report
generator only has to place finished images on the page.
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DIR

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.grid": False,
})

OBJECT_COLOUR = (0, 0, 255)      # BGR, red outline
GT_COLOUR = (0, 200, 0)          # BGR, green outline


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_boundaries(img_bgr, mask, colour=OBJECT_COLOUR, thickness=2):
    out = img_bgr.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, colour, thickness)
    return out


def draw_measurements(img_bgr, mask, rows, px_per_mm=None, show_bbox=True,
                      font_scale=0.42):
    """Draw the boundary, the bounding box, the centroid and a label with the
    main measurements of every detected object."""
    out = draw_boundaries(img_bgr, mask, OBJECT_COLOUR, 2)
    for r in rows:
        x, y, w, h = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
        if show_bbox:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 1)
        cx, cy = int(round(r["centroid_x"])), int(round(r["centroid_y"]))
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 9, 2)
        if px_per_mm:
            label = "%d: %.1fx%.1fmm" % (r["object_id"], r["major_axis_mm"],
                                         r["minor_axis_mm"])
        else:
            label = "%d: %dpx" % (r["object_id"], r["area_px"])
        cv2.putText(out, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def error_map(pred, gt):
    """Colour coded comparison: green true positive, red false positive,
    blue false negative."""
    h, w = gt.shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    p, g = pred > 0, gt > 0
    out[np.logical_and(p, g)] = (0, 180, 0)
    out[np.logical_and(p, ~g)] = (0, 0, 220)
    out[np.logical_and(~p, g)] = (220, 120, 0)
    return out


def panel(images, titles, path, cols=None, figsize_scale=3.1, suptitle=None):
    """Write a grid of images as one figure."""
    n = len(images)
    cols = cols or min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_scale * cols, figsize_scale * rows * 0.85))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(rgb(img))
        ax.set_title(title)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / path), bbox_inches="tight")
    plt.close(fig)
    return FIG_DIR / path


def histogram_with_threshold(gray, threshold, path, title="Grey level histogram"):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(gray.ravel(), bins=256, range=(0, 255), color="#4a6fa5")
    ax.axvline(threshold, color="crimson", linewidth=1.6,
               label="Otsu threshold = %d" % threshold)
    ax.set_xlabel("grey level")
    ax.set_ylabel("pixel count")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / path))
    plt.close(fig)
    return FIG_DIR / path


def bar_chart(categories, series, path, ylabel, title, ylim=None, annotate=True):
    """Grouped bar chart. series is a dictionary of label to list of values."""
    fig, ax = plt.subplots(figsize=(max(4.6, 1.05 * len(categories) + 1.6), 2.9))
    n = len(series)
    width = 0.8 / max(1, n)
    x = np.arange(len(categories))
    for i, (label, values) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, values, width=width, label=label)
        if annotate:
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, v, ("%.2f" % v).lstrip("0"),
                        ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / path))
    plt.close(fig)
    return FIG_DIR / path


def line_chart(x, series, path, xlabel, ylabel, title, xticks=None):
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for label, values in series.items():
        ax.plot(x, values, marker="o", markersize=3.5, linewidth=1.3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / path))
    plt.close(fig)
    return FIG_DIR / path


def size_histogram(values, path, xlabel, title, bins=12):
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.hist(values, bins=bins, color="#5a8f5a", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of objects")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / path))
    plt.close(fig)
    return FIG_DIR / path
