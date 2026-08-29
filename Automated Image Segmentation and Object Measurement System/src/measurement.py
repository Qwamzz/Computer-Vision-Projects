"""
Object analysis stage: boundary identification, object extraction and the
measurements requested by Question 4.

For every segmented region the system reports the object count, area,
perimeter, centroid, bounding box and object size. When a scale is available
the pixel measurements are converted into millimetres.
"""

import cv2
import numpy as np


def boundaries(mask, thickness=1):
    """Return the boundary pixels of a binary mask.

    The morphological gradient, that is the dilation minus the erosion, gives a
    one pixel wide outline of every region and is the classical way to turn a
    region representation into a boundary representation.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = (mask > 0).astype(np.uint8) * 255
    grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, se)
    if thickness > 1:
        grad = cv2.dilate(grad, se, iterations=thickness - 1)
    return grad


def canny_edges(gray, low=60, high=160, blur=5):
    """Canny edges of the intensity image, used for the visual comparison
    between an edge based and a region based description of the objects."""
    smooth = cv2.GaussianBlur(gray, (blur, blur), 0)
    return cv2.Canny(smooth, low, high)


def measure_objects(mask, px_per_mm=None, drop_border_objects=False,
                    min_area=1):
    """Measure every connected component of a binary mask.

    Returned per object:
        id, area_px, perimeter_px, centroid, bounding box, width, height,
        equivalent diameter, major and minor axis of the fitted ellipse,
        orientation, circularity, aspect ratio, extent, solidity and, when a
        scale is supplied, the area in square millimetres and the lengths in
        millimetres.

    Circularity is 4 * pi * area / perimeter^2, which is 1 for a perfect disc
    and falls towards 0 for elongated or ragged shapes.
    """
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    h, w = binary.shape[:2]
    rows = []
    kept = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if drop_border_objects and touches_border:
            continue

        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        if len(contour) >= 5:
            (_, _), (ax1, ax2), angle = cv2.fitEllipse(contour)
            major, minor = max(ax1, ax2), min(ax1, ax2)
        else:
            major, minor, angle = float(max(bw, bh)), float(min(bw, bh)), 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / float(bw * bh) if bw * bh > 0 else 0.0
        eq_diameter = float(np.sqrt(4.0 * area / np.pi))

        kept += 1
        row = {
            "object_id": kept,
            "area_px": area,
            "perimeter_px": round(perimeter, 2),
            "centroid_x": round(float(centroids[i][0]), 2),
            "centroid_y": round(float(centroids[i][1]), 2),
            "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
            "equivalent_diameter_px": round(eq_diameter, 2),
            "major_axis_px": round(float(major), 2),
            "minor_axis_px": round(float(minor), 2),
            "orientation_deg": round(float(angle), 1),
            "aspect_ratio": round(float(major / minor), 3) if minor > 0 else 0.0,
            "circularity": round(float(circularity), 3),
            "solidity": round(float(solidity), 3),
            "extent": round(float(extent), 3),
            "touches_border": int(touches_border),
        }
        if px_per_mm:
            row["area_mm2"] = round(area / (px_per_mm ** 2), 2)
            row["perimeter_mm"] = round(perimeter / px_per_mm, 2)
            row["major_axis_mm"] = round(float(major) / px_per_mm, 2)
            row["minor_axis_mm"] = round(float(minor) / px_per_mm, 2)
            row["equivalent_diameter_mm"] = round(eq_diameter / px_per_mm, 2)
        rows.append(row)
    return rows


def summarise(rows, px_per_mm=None):
    """Aggregate statistics over all measured objects of one image."""
    if not rows:
        return {"count": 0}
    areas = np.array([r["area_px"] for r in rows], dtype=np.float64)
    majors = np.array([r["major_axis_px"] for r in rows], dtype=np.float64)
    minors = np.array([r["minor_axis_px"] for r in rows], dtype=np.float64)
    circs = np.array([r["circularity"] for r in rows], dtype=np.float64)
    out = {
        "count": len(rows),
        "area_px_mean": round(float(areas.mean()), 1),
        "area_px_std": round(float(areas.std()), 1),
        "area_px_min": round(float(areas.min()), 1),
        "area_px_max": round(float(areas.max()), 1),
        "major_axis_px_mean": round(float(majors.mean()), 2),
        "minor_axis_px_mean": round(float(minors.mean()), 2),
        "circularity_mean": round(float(circs.mean()), 3),
    }
    if px_per_mm:
        out["area_mm2_mean"] = round(float(areas.mean()) / (px_per_mm ** 2), 2)
        out["major_axis_mm_mean"] = round(float(majors.mean()) / px_per_mm, 2)
        out["minor_axis_mm_mean"] = round(float(minors.mean()) / px_per_mm, 2)
    return out


def size_classes(rows, n_classes=3):
    """Group objects into small, medium and large by equivalent diameter.

    This supports the object size requirement of the question and it is what a
    grading application would report for a batch of seeds.
    """
    if not rows:
        return {}
    d = np.array([r["equivalent_diameter_px"] for r in rows])
    edges = np.quantile(d, np.linspace(0, 1, n_classes + 1))
    names = ["small", "medium", "large"][:n_classes]
    counts = {}
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        if i == n_classes - 1:
            counts[name] = int(((d >= lo) & (d <= hi)).sum())
        else:
            counts[name] = int(((d >= lo) & (d < hi)).sum())
    return counts
