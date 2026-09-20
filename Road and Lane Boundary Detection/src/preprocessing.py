"""
preprocessing.py
Colour representation and noise suppression.

Two questions are settled here, and both are settled by measurement rather than
by assertion:

    which colour representation separates road from not road, and
    which filter suppresses noise without destroying the road boundary.

Colour. Road surfaces in these photographs are grey asphalt or orange laterite.
What they have in common is not a colour but a *lack* of one: compared with the
vegetation, sky and buildings around them they are weakly saturated and their
lightness is mid range. HLS exposes exactly those two quantities on separate
axes, and the L channel of HLS is also where white and yellow lane paint stands
out most strongly. Lab is kept as the alternative because its a and b axes are
approximately perceptually uniform, which matters for clustering. RGB is kept
only as the baseline that mixes brightness and chroma together and therefore
moves under every change of illumination.

Filtering. A road boundary is a step edge that the edge detector must keep,
while the texture of chippings, gravel and tree shade is noise that it must
lose. Gaussian smoothing treats both the same way. A bilateral filter weights
by intensity difference as well as distance, so it smooths within a region and
stops at the boundary between regions. The two are compared directly in
experiment E2.
"""

from __future__ import annotations

import cv2
import numpy as np

# Names used consistently in the tables and the report.
COLOUR_SPACES = ["BGR", "HLS", "Lab", "HSV"]
FILTERS = ["none", "gaussian", "median", "bilateral"]


def to_space(bgr, space):
    """Convert to one of the colour representations under test."""
    if space == "BGR":
        return bgr.copy()
    if space == "HLS":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    if space == "Lab":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    if space == "HSV":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raise ValueError("unknown colour space " + space)


def apply_filter(bgr, kind, strength=1.0):
    """Suppress noise. `strength` scales the kernel so it can be swept.

    Gaussian   : separable, isotropic, cheap, blurs edges along with noise.
    Median     : removes impulsive speckle, preserves step edges, slower.
    Bilateral  : edge preserving, the most expensive of the three.
    """
    if kind == "none":
        return bgr.copy()
    if kind == "gaussian":
        k = int(max(1, round(5 * strength)) * 2 + 1)
        return cv2.GaussianBlur(bgr, (k, k), 0)
    if kind == "median":
        k = int(max(1, round(2 * strength)) * 2 + 1)
        return cv2.medianBlur(bgr, k)
    if kind == "bilateral":
        d = int(max(3, round(9 * strength)))
        return cv2.bilateralFilter(bgr, d, 75 * strength, 75 * strength)
    raise ValueError("unknown filter " + kind)


def road_channel(bgr, space="HLS"):
    """Return the single channel in which the road is most distinctive.

    For HLS that is lightness, which carries the road surface and the paint.
    For Lab it is L. For HSV it is V. For BGR the grey scale conversion is
    used, which is the honest baseline.
    """
    if space == "BGR":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    conv = to_space(bgr, space)
    if space == "HLS":
        return conv[:, :, 1]
    if space == "Lab":
        return conv[:, :, 0]
    if space == "HSV":
        return conv[:, :, 2]
    raise ValueError(space)


def separability(bgr, mask, space):
    """How well does this colour representation separate road from not road?

    Reports the Fisher ratio between the road and non road populations,

        J = (mu_road - mu_other)^2 / (var_road + var_other)

    averaged over the channels of the representation. A larger value means the
    two populations are further apart relative to their spread, so a threshold
    or a clustering step has an easier job. This is computed with the manual
    ground truth and is therefore a property of the *representation*, not of any
    particular segmentation method.
    """
    conv = to_space(bgr, space).astype(np.float32)
    road = mask > 0
    other = ~road
    if road.sum() < 50 or other.sum() < 50:
        return 0.0
    scores = []
    for c in range(conv.shape[2]):
        ch = conv[:, :, c]
        a, b = ch[road], ch[other]
        denom = a.var() + b.var()
        if denom < 1e-6:
            continue
        scores.append((a.mean() - b.mean()) ** 2 / denom)
    return float(np.mean(scores)) if scores else 0.0


def illumination_correction(bgr, clip=2.0, tiles=8):
    """Local contrast equalisation on the lightness channel only.

    Applied to the hazy and heavily shaded photographs. Working on L in HLS
    rather than on the three BGR channels avoids shifting the hue, which would
    defeat the colour based segmentation that follows.
    """
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    hls[:, :, 1] = clahe.apply(hls[:, :, 1])
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
