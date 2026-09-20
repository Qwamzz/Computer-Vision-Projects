"""
stitching.py
============
Stages 6 and 7 of the pipeline: IMAGE ALIGNMENT and PANORAMA CONSTRUCTION.

Given the pairwise homographies produced by :mod:`homography`, this module

1. chains them so that every image is expressed in one common reference frame,
2. computes the size of the canvas that contains all the warped images,
3. warps each image onto that canvas,
4. blends the overlapping regions, and
5. measures how well the images actually agree in the overlap.

Two blending strategies are implemented so that their effect can be compared:

* ``overwrite`` -- later images simply paint over earlier ones.  Fast, and it
  makes any misalignment or exposure difference immediately visible as a hard
  seam.  Used in the report to show *why* blending is needed.
* ``feather``   -- each pixel is a weighted average of the contributing images,
  the weight being the distance to that image's border (a distance transform).
  A pixel near the centre of an image is trusted more than one at its edge, so
  the transition is gradual and the seam disappears.
"""

from __future__ import annotations

import cv2
import numpy as np

from homography import apply_homography


# --------------------------------------------------------------------------- #
# Geometry bookkeeping
# --------------------------------------------------------------------------- #

def image_corners(w, h):
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    dtype=np.float64)


def chain_homographies(pair_H, n_images, ref_idx):
    """Convert consecutive pairwise homographies into per-image transforms into
    the reference frame.

    ``pair_H[k]`` must map image ``k+1`` into image ``k``.  Composing along the
    chain gives, for example, H(3 -> 1) = H(3 -> 2) @ ... no: with our
    convention H(2->1) @ H(3->2) maps a point of image 3 first into image 2 and
    then into image 1.

    Choosing the *middle* image as the reference roughly halves the length of
    the longest chain, which matters because each composition compounds the
    error of the individual estimates and because a long chain produces extreme
    perspective stretching at the ends of the panorama.
    """
    H_to_ref = [None] * n_images
    H_to_ref[ref_idx] = np.eye(3)

    # Walk right from the reference.
    for k in range(ref_idx, n_images - 1):
        if pair_H[k] is None or H_to_ref[k] is None:
            break
        H_to_ref[k + 1] = H_to_ref[k] @ pair_H[k]

    # Walk left from the reference.
    for k in range(ref_idx - 1, -1, -1):
        if pair_H[k] is None or H_to_ref[k + 1] is None:
            break
        H_to_ref[k] = H_to_ref[k + 1] @ np.linalg.inv(pair_H[k])

    return H_to_ref


def canvas_geometry(images, H_to_ref, max_pixels=40_000_000):
    """Bounding box of all warped images, plus the translation that shifts it to
    the positive quadrant.

    Warping generally sends pixels to negative coordinates, which cannot be
    stored in an image, so a translation T is prepended to every homography.
    """
    all_pts = []
    for img, H in zip(images, H_to_ref):
        if H is None:
            continue
        h, w = img.shape[:2]
        all_pts.append(apply_homography(H, image_corners(w, h)))
    if not all_pts:
        raise RuntimeError("No valid homographies -- cannot build a canvas")

    pts = np.vstack(all_pts)
    x_min, y_min = np.floor(pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(pts.max(axis=0)).astype(int)

    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)

    # Safety valve: a badly estimated homography can demand an enormous canvas.
    if width * height > max_pixels or width <= 0 or height <= 0:
        raise RuntimeError("Canvas of %dx%d px is implausible; the homography "
                           "estimate is probably degenerate" % (width, height))

    T = np.array([[1.0, 0.0, -x_min],
                  [0.0, 1.0, -y_min],
                  [0.0, 0.0, 1.0]])
    return (width, height), T


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #

def _feather_weight(h, w):
    """Weight map that is 0 on the image border and rises towards the centre.

    ``cv2.distanceTransform`` on a mask that is zero only on the boundary gives
    exactly the "distance to the nearest edge of this image" that feathering
    needs.
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[0, :] = mask[-1, :] = 0
    mask[:, 0] = mask[:, -1] = 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32) + 1e-6


def warp_all(images, H_to_ref, size, T, blend="feather"):
    """Warp every image onto the shared canvas and blend.

    Returns the panorama and the list of per-image validity masks (useful for
    the overlap-quality measurements).
    """
    width, height = size
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    masks = []

    for img, H in zip(images, H_to_ref):
        if H is None:
            masks.append(None)
            continue
        h, w = img.shape[:2]
        M = T @ H

        warped = cv2.warpPerspective(img, M, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        valid = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, M,
                                    (width, height), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        masks.append(valid > 127)

        if blend == "overwrite":
            m = (valid > 127)
            accum[m] = warped[m].astype(np.float32)
            weight[m] = 1.0
        else:
            wmap = cv2.warpPerspective(_feather_weight(h, w), M, (width, height),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            wmap = np.where(valid > 127, wmap, 0.0).astype(np.float32)[..., None]
            accum += warped.astype(np.float32) * wmap
            weight += wmap

    out = np.where(weight > 1e-6, accum / np.maximum(weight, 1e-6), 0.0)
    return np.clip(out, 0, 255).astype(np.uint8), masks


def crop_to_content(panorama, masks=None, border_tol=8):
    """Remove the black margin left by warping.

    A simple bounding box of the non-black pixels is used.  (A more elaborate
    approach searches for the largest inscribed rectangle, which removes the
    ragged edges too but also discards valid image area; the bounding box is the
    conventional choice.)
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray > border_tol)
    if len(xs) == 0:
        return panorama
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return panorama[y0:y1 + 1, x0:x1 + 1]


# --------------------------------------------------------------------------- #
# Quantitative panorama quality
# --------------------------------------------------------------------------- #

def overlap_quality(images, H_to_ref, size, T):
    """Measure agreement between images where they overlap.

    This is the objective substitute for "does the panorama look right?".  If
    the homographies are accurate, the two images should carry nearly identical
    intensities in the shared region, so:

    * **MAE / RMSE** should be small (only exposure differences and noise), and
    * **NCC** (normalised cross-correlation) should be close to 1.

    A visible ghosting or double-edge artefact shows up as a large RMSE and a
    depressed NCC, so these numbers rank panoramas the same way the eye does.
    """
    width, height = size
    warped_grays, valids = [], []
    for img, H in zip(images, H_to_ref):
        if H is None:
            warped_grays.append(None)
            valids.append(None)
            continue
        h, w = img.shape[:2]
        M = T @ H
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        wg = cv2.warpPerspective(g, M, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v = cv2.warpPerspective(np.ones((h, w), np.uint8) * 255, M, (width, height),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
        warped_grays.append(wg.astype(np.float64))
        valids.append(v)

    results = []
    n = len(images)
    for i in range(n):
        for j in range(i + 1, n):
            if valids[i] is None or valids[j] is None:
                continue
            ov = valids[i] & valids[j]
            n_px = int(ov.sum())
            if n_px < 500:
                continue
            a, b = warped_grays[i][ov], warped_grays[j][ov]
            diff = a - b
            mae = float(np.abs(diff).mean())
            rmse = float(np.sqrt((diff ** 2).mean()))
            psnr = float(20 * np.log10(255.0 / rmse)) if rmse > 1e-9 else float("inf")
            a0, b0 = a - a.mean(), b - b.mean()
            den = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum())
            ncc = float((a0 * b0).sum() / den) if den > 1e-9 else float("nan")
            results.append({"pair": (i, j), "overlap_px": n_px, "mae": mae,
                            "rmse": rmse, "psnr_db": psnr, "ncc": ncc})
    return results


def summarise_quality(quality_list):
    """Average the per-pair overlap metrics into one row for the report table."""
    if not quality_list:
        return {"mean_mae": float("nan"), "mean_rmse": float("nan"),
                "mean_psnr_db": float("nan"), "mean_ncc": float("nan"),
                "n_pairs": 0, "total_overlap_px": 0}
    w = np.array([q["overlap_px"] for q in quality_list], dtype=np.float64)
    w = w / w.sum()
    return {
        "mean_mae": float(np.sum(w * [q["mae"] for q in quality_list])),
        "mean_rmse": float(np.sum(w * [q["rmse"] for q in quality_list])),
        "mean_psnr_db": float(np.sum(w * [min(q["psnr_db"], 99) for q in quality_list])),
        "mean_ncc": float(np.sum(w * [q["ncc"] for q in quality_list])),
        "n_pairs": len(quality_list),
        "total_overlap_px": int(sum(q["overlap_px"] for q in quality_list)),
    }
