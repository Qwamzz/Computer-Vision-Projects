"""
features.py
===========
Stage 1 and Stage 2 of the pipeline: FEATURE DETECTION and FEATURE DESCRIPTION.

Three configurations are compared in this project.  They were chosen so that the
comparison spans the two design axes that matter for matching:

    +---------------------+----------------+------------------+-------------+
    | Configuration       | Detector       | Descriptor       | Distance    |
    +---------------------+----------------+------------------+-------------+
    | SIFT                | DoG blobs      | 128-D gradient   | L2 (float)  |
    |                     | (scale-space)  | histogram        |             |
    | ORB                 | oFAST corners  | 256-bit rBRIEF   | Hamming     |
    |                     | (image pyramid)| (binary)         |             |
    | SHITOMASI+SIFT      | Shi-Tomasi     | 128-D SIFT       | L2 (float)  |
    |                     | corners        |                  |             |
    |                     | (single scale) |                  |             |
    +---------------------+----------------+------------------+-------------+

The third configuration exists to make an explicit teaching point: *detection and
description are separable stages*.  Shi-Tomasi is a single-scale corner detector
with no notion of characteristic scale or dominant orientation, so pairing it
with the SIFT descriptor isolates the contribution of scale-space detection from
the contribution of the descriptor itself.

Note on the environment: OpenCV 5.0 no longer exposes AKAZE/BRISK in the main
`cv2` module (they now live in opencv-contrib), so the classic SIFT-vs-ORB
comparison is extended with the hybrid above rather than with AKAZE.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

# Descriptor "norm kind": determines which distance the matcher must use.
FLOAT_DESC = "float"     # L2 distance
BINARY_DESC = "binary"   # Hamming distance


# --------------------------------------------------------------------------- #
# Detector / descriptor factory
# --------------------------------------------------------------------------- #

def create_detector(name: str, n_features: int = 4000):
    """Return ``(detector, descriptor, kind)`` for the named configuration.

    ``detector`` and ``descriptor`` may be the same object (SIFT, ORB) or two
    different objects (the hybrid), which is exactly how OpenCV models the
    separation between the two stages.
    """
    name = name.upper()

    if name == "SIFT":
        # nfeatures=0 -> keep every keypoint that survives the contrast and edge
        # tests.  contrastThreshold filters low-contrast DoG extrema (noise);
        # edgeThreshold rejects extrema lying along an edge, where the position
        # is poorly localised along the edge direction (the aperture problem).
        sift = cv2.SIFT_create(nfeatures=n_features,
                               nOctaveLayers=3,
                               contrastThreshold=0.04,
                               edgeThreshold=10,
                               sigma=1.6)
        return sift, sift, FLOAT_DESC

    if name == "ORB":
        # scaleFactor/nlevels define the image pyramid that gives ORB its
        # (limited) scale invariance.  fastThreshold is the FAST intensity
        # difference required for a pixel to be declared a corner.
        orb = cv2.ORB_create(nfeatures=n_features,
                             scaleFactor=1.2,
                             nlevels=8,
                             edgeThreshold=31,
                             firstLevel=0,
                             WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE,
                             patchSize=31,
                             fastThreshold=20)
        return orb, orb, BINARY_DESC

    if name in ("SHITOMASI+SIFT", "GFTT+SIFT", "HYBRID"):
        gftt = cv2.GFTTDetector_create(maxCorners=n_features,
                                       qualityLevel=0.01,
                                       minDistance=5,
                                       blockSize=3,
                                       useHarrisDetector=False)
        sift = cv2.SIFT_create()
        return gftt, sift, FLOAT_DESC

    raise ValueError("Unknown detector configuration: %s" % name)


DETECTOR_NAMES = ["SIFT", "ORB", "SHITOMASI+SIFT"]


# --------------------------------------------------------------------------- #
# Detection + description with timing
# --------------------------------------------------------------------------- #

def detect_and_describe(gray, config: str, n_features: int = 4000):
    """Run detection then description, timing each stage separately.

    Returns
    -------
    keypoints : list of cv2.KeyPoint
    descriptors : ndarray (N, D)
    timings : dict with 'detect_ms', 'describe_ms', 'total_ms'
    kind : FLOAT_DESC or BINARY_DESC
    """
    detector, describer, kind = create_detector(config, n_features)

    t0 = time.perf_counter()
    kps = detector.detect(gray, None)
    t1 = time.perf_counter()
    kps, desc = describer.compute(gray, kps)
    t2 = time.perf_counter()

    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)

    timings = {"detect_ms": (t1 - t0) * 1e3,
               "describe_ms": (t2 - t1) * 1e3,
               "total_ms": (t2 - t0) * 1e3}
    return kps, desc, timings, kind


def keypoints_to_array(kps) -> np.ndarray:
    """(N, 2) float array of keypoint pixel coordinates."""
    if len(kps) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in kps], dtype=np.float64)


def keypoint_statistics(kps) -> dict:
    """Summary statistics used in the report's detector-comparison table."""
    if len(kps) == 0:
        return {"count": 0, "mean_size": 0.0, "mean_response": 0.0,
                "n_scales": 0}
    sizes = np.array([kp.size for kp in kps])
    resp = np.array([kp.response for kp in kps])
    return {"count": int(len(kps)),
            "mean_size": float(sizes.mean()),
            "mean_response": float(resp.mean()),
            "n_scales": int(len(np.unique(np.round(sizes, 2))))}
