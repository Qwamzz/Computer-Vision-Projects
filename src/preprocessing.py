"""
Image preparation stage of the pipeline.

Everything in this module is deliberately classical: colour conversion,
smoothing filters, contrast handling and a morphological illumination
correction. No learned model is used anywhere in the project.
"""

import cv2
import numpy as np

from config import WORK_MAX_SIDE


# ----------------------------------------------------------------------------
# Basic input handling
# ----------------------------------------------------------------------------
def load_bgr(path):
    """Read an image from disk as BGR, raising a clear error if it is missing."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not read image: %s" % path)
    return img


def resize_max_side(img, max_side=WORK_MAX_SIDE):
    """Scale an image so that its longest side equals max_side.

    Images are only ever shrunk, never enlarged, so no detail is invented.
    The scale factor is returned so that pixel measurements can be traced back
    to the original photograph if needed.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img.copy(), 1.0
    scale = max_side / float(longest)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def colour_channels(img_bgr):
    """Return a dictionary of candidate single channel representations.

    Question 4 asks for an appropriate colour representation to be selected.
    The channels below are the ones examined in the report: grey level, the
    saturation and value channels of HSV, and the a and b channels of CIE Lab.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return {
        "gray": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        "hsv_s": hsv[:, :, 1],
        "hsv_v": hsv[:, :, 2],
        "lab_l": lab[:, :, 0],
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
    }


def channel_separability(channel, sample_border=12):
    """Rough measure of how well a channel separates object from background.

    The border ring of the image is assumed to be mostly background. The score
    is the absolute difference between the mean of the border ring and the mean
    of the image centre, divided by the pooled standard deviation. It is only
    used to justify the choice of channel, never inside the segmentation.
    """
    h, w = channel.shape[:2]
    b = sample_border
    ring = np.concatenate([
        channel[:b, :].ravel(), channel[-b:, :].ravel(),
        channel[:, :b].ravel(), channel[:, -b:].ravel()])
    centre = channel[h // 4: 3 * h // 4, w // 4: 3 * w // 4].ravel()
    pooled = np.sqrt(0.5 * (ring.std() ** 2 + centre.std() ** 2)) + 1e-6
    return float(abs(float(ring.mean()) - float(centre.mean())) / pooled)


# ----------------------------------------------------------------------------
# Noise suppression
# ----------------------------------------------------------------------------
def denoise(gray, method="gaussian", ksize=5, sigma_colour=45, sigma_space=9):
    """Apply one of three classical smoothing filters.

    gaussian  : linear low pass filter, fastest, blurs edges as well as noise
    median    : rank filter, removes impulse noise while keeping step edges
    bilateral : edge preserving filter, slowest of the three
    """
    if method == "none":
        return gray.copy()
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)
    if method == "median":
        return cv2.medianBlur(gray, ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, ksize, sigma_colour, sigma_space)
    raise ValueError("Unknown denoise method: %s" % method)


# ----------------------------------------------------------------------------
# Contrast and illumination handling
# ----------------------------------------------------------------------------
def clahe(gray, clip=2.0, tiles=8):
    """Contrast limited adaptive histogram equalisation."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return op.apply(gray)


def estimate_background(gray, kernel_frac=0.25, objects_are_dark=True):
    """Estimate the slowly varying background using a large morphological op.

    A structuring element much larger than any object is used, so the operation
    removes the objects and leaves the illumination field. When the objects are
    darker than the background a closing recovers the background, and when they
    are brighter an opening is used instead.
    """
    h, w = gray.shape[:2]
    k = int(max(15, kernel_frac * min(h, w)))
    if k % 2 == 0:
        k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    op = cv2.MORPH_CLOSE if objects_are_dark else cv2.MORPH_OPEN
    return cv2.morphologyEx(gray, op, se)


def correct_illumination(gray, objects_are_dark=True, kernel_frac=0.25):
    """Flatten an uneven illumination field.

    The estimated background is divided out and the result is rescaled to the
    full eight bit range. This is the classical flat field correction and it is
    what allows a single global threshold to work on images with a lighting
    gradient.
    """
    bg = estimate_background(gray, kernel_frac, objects_are_dark).astype(np.float32)
    bg[bg < 1.0] = 1.0
    flat = gray.astype(np.float32) / bg
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)
    return flat.astype(np.uint8)


def objects_are_dark(gray, border=10):
    """Decide the polarity of the scene.

    The border ring of the image is treated as a background sample. If the
    interior is darker than that ring the objects are dark on a bright
    background, which is true for every photograph in this dataset except the
    coins image.
    """
    h, w = gray.shape[:2]
    ring = np.concatenate([
        gray[:border, :].ravel(), gray[-border:, :].ravel(),
        gray[:, :border].ravel(), gray[:, -border:].ravel()])
    return float(gray.mean()) < float(ring.mean())


# ----------------------------------------------------------------------------
# One call preparation used by the experiment driver
# ----------------------------------------------------------------------------
def prepare(img_bgr, denoise_method="gaussian", ksize=5,
            use_illumination_correction=True, use_clahe=False):
    """Run the standard preparation chain and return the working images.

    Returns a dictionary with the resized colour image, the raw grey image and
    the prepared grey image that the segmentation methods consume.
    """
    work, scale = resize_max_side(img_bgr)
    gray = to_gray(work)
    dark = objects_are_dark(gray)
    prepared = denoise(gray, denoise_method, ksize)
    if use_illumination_correction:
        prepared = correct_illumination(prepared, objects_are_dark=dark)
    if use_clahe:
        prepared = clahe(prepared)
    return {
        "bgr": work,
        "gray": gray,
        "prepared": prepared,
        "objects_are_dark": dark,
        "scale": scale,
    }
