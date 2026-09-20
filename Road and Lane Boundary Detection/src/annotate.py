"""
annotate.py
Automated road and lane boundary detection and scene segmentation.

Builds the manually annotated ground truth masks of the drivable road region.

Method. Each road surface was traced by hand as a polygon, read off the
photograph against a printed coordinate grid, and then checked by drawing the
resulting mask back over the photograph and correcting whatever was wrong. The
vertices live in this file rather than in a binary mask, so the annotation is
readable, reviewable and reproducible, and anyone can see exactly what was
claimed to be road.

Two lists describe each image:

    road : the outline of the drivable surface, as it appears in the image.
           For a divided carriageway there is one polygon per carriageway.
    cut  : anything standing on that surface which is not road, such as a
           vehicle or an animal. These are removed from the mask, so the
           ground truth is the *visible* drivable surface. A segmentation
           method cannot be expected to report road where a bus is parked.

Coordinates are normalised, x to the right and y downwards, so they do not
depend on the stored resolution.

Nothing in the evaluated pipeline had any hand in producing these masks: the
pipeline never sees the polygons.

Usage:
    python src/annotate.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "data", "images")
ANN_DIR = os.path.join(ROOT, "data", "annotations")
CHECK_DIR = os.path.join(ROOT, "results", "overlays")

# --------------------------------------------------------------------------
# Hand traced ground truth
# --------------------------------------------------------------------------
ANNOTATION = {
    # Straight sealed road, white edge lines, low sun. A baboon stands on the
    # carriageway and is cut out of the mask.
    "daylight_01.jpg": dict(
        road=[[(0.00, 0.560), (0.10, 0.455), (0.20, 0.355), (0.26, 0.305),
               (0.315, 0.275), (0.38, 0.325), (0.50, 0.420), (0.62, 0.500),
               (0.74, 0.575), (0.86, 0.640), (1.00, 0.720), (1.00, 1.000),
               (0.00, 1.000)]],
        cut=[[(0.495, 0.325), (0.585, 0.325), (0.585, 0.445), (0.495, 0.445)]]),

    # Two lane road with a painted centre line, bright even light.
    "daylight_02.jpg": dict(
        road=[[(0.00, 0.760), (0.15, 0.700), (0.30, 0.640), (0.42, 0.570),
               (0.50, 0.510), (0.525, 0.485), (0.56, 0.510), (0.615, 0.580),
               (0.66, 0.680), (0.70, 0.790), (0.735, 0.900), (0.760, 1.000),
               (0.00, 1.000)]],
        cut=[]),

    # Avenue of trees, hard dappled shadow bands across the carriageway.
    "shadow_01.jpg": dict(
        road=[[(0.05, 1.000), (0.18, 0.850), (0.28, 0.720), (0.36, 0.600),
               (0.41, 0.500), (0.435, 0.450), (0.455, 0.450), (0.52, 0.520),
               (0.60, 0.620), (0.68, 0.740), (0.75, 0.870), (0.82, 1.000)]],
        cut=[]),

    # Unsealed road under trees; shadow contrast rivals the road edge.
    "shadow_02.jpg": dict(
        road=[[(0.10, 1.000), (0.25, 0.850), (0.38, 0.700), (0.50, 0.580),
               (0.57, 0.520), (0.595, 0.500), (0.615, 0.500), (0.66, 0.550),
               (0.73, 0.640), (0.81, 0.760), (0.88, 0.880), (0.95, 1.000)]],
        cut=[]),

    # Haze and flat overcast light, very low edge contrast.
    "lowlight_01.jpg": dict(
        road=[[(0.03, 1.000), (0.18, 0.900), (0.30, 0.820), (0.38, 0.770),
               (0.435, 0.735), (0.465, 0.735), (0.55, 0.780), (0.68, 0.850),
               (0.82, 0.920), (0.95, 1.000)]],
        cut=[]),

    # Overcast, weak markings, road runs away up and slightly to the left.
    "lowlight_02.jpg": dict(
        road=[[(0.170, 1.000), (0.235, 0.900), (0.300, 0.800), (0.350, 0.700),
               (0.395, 0.600), (0.425, 0.520), (0.433, 0.490), (0.438, 0.490),
               (0.450, 0.600), (0.485, 0.700), (0.525, 0.800), (0.575, 0.900),
               (0.620, 1.000)]],
        cut=[[(0.415, 0.870), (0.505, 0.870), (0.505, 1.000), (0.415, 1.000)]]),

    # Strong single bend with clear markings.
    "curve_01.jpg": dict(
        road=[[(0.00, 0.660), (0.12, 0.645), (0.25, 0.632), (0.36, 0.620),
               (0.45, 0.612), (0.52, 0.620), (0.58, 0.645), (0.65, 0.685),
               (0.72, 0.745), (0.79, 0.830), (0.85, 0.920), (0.88, 1.000),
               (0.00, 1.000)]],
        cut=[]),

    # Sweeping curve seen from the inside of the bend.
    "curve_02.jpg": dict(
        road=[[(0.42, 1.000), (0.50, 0.900), (0.555, 0.800), (0.585, 0.720),
               (0.605, 0.655), (0.635, 0.595), (0.70, 0.585), (0.78, 0.605),
               (0.88, 0.635), (1.00, 0.665), (1.00, 1.000)]],
        cut=[]),

    # Unsealed laterite road, no markings at all; a car is parked far ahead.
    "worn_01.jpg": dict(
        road=[[(0.10, 1.000), (0.22, 0.880), (0.33, 0.770), (0.43, 0.680),
               (0.49, 0.630), (0.52, 0.605), (0.545, 0.605), (0.60, 0.640),
               (0.68, 0.710), (0.78, 0.800), (0.88, 0.900), (0.97, 1.000)]],
        cut=[[(0.505, 0.560), (0.585, 0.560), (0.585, 0.615), (0.505, 0.615)]]),

    # Straight laterite road; the boundary is a colour change, not an edge line.
    "worn_02.jpg": dict(
        road=[[(0.280, 1.000), (0.330, 0.900), (0.390, 0.800), (0.450, 0.700),
               (0.495, 0.645), (0.515, 0.620), (0.535, 0.620), (0.565, 0.645),
               (0.620, 0.700), (0.700, 0.800), (0.760, 0.900), (0.800, 1.000)]],
        cut=[]),

    # A queue of minibuses standing on the carriageway.
    "occlusion_01.jpg": dict(
        road=[[(0.00, 0.600), (0.12, 0.645), (0.24, 0.680), (0.34, 0.720),
               (0.44, 0.775), (0.52, 0.820), (0.60, 0.870), (0.68, 0.920),
               (0.75, 0.955), (0.80, 0.900), (0.85, 0.800), (0.88, 0.720),
               (0.92, 0.700), (1.00, 0.700), (1.00, 1.000), (0.00, 1.000)]],
        cut=[]),

    # A tractor and a pickup occupying the carriageway ahead.
    "occlusion_02.jpg": dict(
        road=[[(0.13, 1.000), (0.15, 0.850), (0.17, 0.740), (0.22, 0.660),
               (0.30, 0.620), (0.42, 0.600), (0.55, 0.600), (0.62, 0.620),
               (0.70, 0.640), (0.80, 0.660), (0.90, 0.680), (1.00, 0.700),
               (1.00, 1.000)]],
        cut=[[(0.330, 0.590), (0.595, 0.590), (0.595, 0.760), (0.330, 0.760)]]),
}


# --------------------------------------------------------------------------
def to_px(poly, w, h):
    return np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], np.int32)


def build_mask(shape, spec):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for poly in spec["road"]:
        cv2.fillPoly(mask, [to_px(poly, w, h)], 255)
    for poly in spec.get("cut", []):
        cv2.fillPoly(mask, [to_px(poly, w, h)], 0)
    return mask


def overlay(img, mask, colour=(0, 200, 0), alpha=0.40):
    out = img.copy()
    out[mask > 0] = (np.array(colour) * alpha
                     + out[mask > 0] * (1 - alpha)).astype(np.uint8)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    return out


def main():
    os.makedirs(ANN_DIR, exist_ok=True)
    os.makedirs(CHECK_DIR, exist_ok=True)
    summary = {}
    for name, spec in ANNOTATION.items():
        img = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("cannot read " + name)
        mask = build_mask(img.shape, spec)
        cv2.imwrite(os.path.join(ANN_DIR, name.replace(".jpg", "_road.png")), mask)
        cv2.imwrite(os.path.join(CHECK_DIR, "gt_" + name),
                    overlay(img, mask), [cv2.IMWRITE_JPEG_QUALITY, 92])
        frac = float((mask > 0).mean())
        summary[name] = {"road_fraction": round(frac, 4),
                         "pixels": int((mask > 0).sum()),
                         "size": [int(img.shape[1]), int(img.shape[0])],
                         "polygons": len(spec["road"]), "cuts": len(spec.get("cut", []))}
        print("%-18s road covers %5.1f%% of the frame" % (name, 100 * frac))
    with open(os.path.join(ANN_DIR, "annotation_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
