"""
matching.py
===========
Stage 3 of the pipeline: FEATURE MATCHING.

The brute-force matcher is implemented from first principles in NumPy rather
than simply calling ``cv2.BFMatcher``, because the exam requires demonstrated
understanding of the technique.  The OpenCV matcher is still used, but only as
an independent check: :func:`verify_against_opencv` reports the fraction of
matches on which the two implementations agree, which validates the hand-written
code.

Two filters are applied to the raw nearest-neighbour matches:

* **Lowe's ratio test.**  A match is accepted only if the best distance is
  clearly smaller than the second-best distance, i.e. ``d1 < ratio * d2``.  The
  reasoning is that a descriptor with many near-equally-good candidates is
  ambiguous (repeated texture, generic corners), and such a match is more likely
  to be wrong than right.  ratio = 0.75 is Lowe's original recommendation.

* **Cross-check (mutual nearest neighbour).**  A match (i, j) is kept only if j
  is the nearest neighbour of i *and* i is the nearest neighbour of j.  This
  removes many-to-one matches, where several features in one image all claim the
  same feature in the other.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from features import BINARY_DESC, FLOAT_DESC

# Lookup table: number of set bits in each possible byte value (0..255).
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)


# --------------------------------------------------------------------------- #
# Distance matrices
# --------------------------------------------------------------------------- #

def _l2_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of ``a`` and every row of ``b``.

    Uses the expansion  ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y  so that the
    whole matrix is obtained with a single matrix product, which is orders of
    magnitude faster than a Python double loop.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(d2, 0, out=d2)          # clamp small negatives from round-off
    return np.sqrt(d2)


def _hamming_distance_matrix_lut(a: np.ndarray, b: np.ndarray,
                                 chunk: int = 256) -> np.ndarray:
    """Hamming distance by explicit XOR and population count.

    The XOR of two descriptors has a 1 wherever the bits differ, so the Hamming
    distance is the population count of the XOR.  A 256-entry lookup table turns
    that population count into a single array indexing operation, and the
    computation is chunked over the rows of ``a`` to bound peak memory.

    This is the textbook formulation and is kept as the reference
    implementation, but it materialises an (n, m, 32) intermediate array and is
    therefore slow for large descriptor sets.  :func:`_hamming_distance_matrix`
    below computes the identical result far faster.
    """
    n, m = a.shape[0], b.shape[0]
    out = np.empty((n, m), dtype=np.uint16)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        xor = np.bitwise_xor(a[i0:i1, None, :], b[None, :, :])
        out[i0:i1] = _POPCOUNT[xor].sum(axis=2)
    return out


def _hamming_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance via a single matrix product.

    For binary vectors written as 0/1 entries,

        popcount(x XOR y) = popcount(x) + popcount(y) - 2 * <x, y>

    because the inner product counts exactly the positions where both bits are
    1.  Unpacking the descriptors to a 0/1 matrix therefore turns the whole
    distance matrix into one BLAS matrix multiplication -- structurally the same
    trick used for the L2 case above, and roughly two orders of magnitude faster
    than materialising the XOR tensor.

    Making this fast matters for the fairness of the experiment: ORB's headline
    advantage is its cheap distance metric, and a naive Hamming implementation
    would hide that advantage behind an artefact of our own code.
    """
    A = np.unpackbits(a, axis=1).astype(np.float32)
    B = np.unpackbits(b, axis=1).astype(np.float32)
    a1 = A.sum(1)[:, None]
    b1 = B.sum(1)[None, :]
    d = a1 + b1 - 2.0 * (A @ B.T)
    np.maximum(d, 0, out=d)
    return d


def distance_matrix(desc1, desc2, kind):
    if kind == BINARY_DESC:
        return _hamming_distance_matrix(desc1, desc2).astype(np.float32)
    return _l2_distance_matrix(desc1, desc2)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def match_descriptors(desc1, desc2, kind=FLOAT_DESC, ratio=0.75,
                      cross_check=True):
    """Brute-force k-NN matching followed by the ratio test and cross-check.

    Returns
    -------
    matches : (M, 2) int array of (index_in_image1, index_in_image2)
    info : dict of diagnostic counts and timings
    """
    t0 = time.perf_counter()

    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=int), {
            "n_raw": 0, "n_after_ratio": 0, "n_after_cross": 0,
            "match_ms": 0.0, "ratio": ratio}

    D = distance_matrix(desc1, desc2, kind)

    # Two nearest neighbours in image 2 for every descriptor of image 1.
    # argpartition is O(N) per row versus O(N log N) for a full sort.
    nn_idx = np.argpartition(D, kth=1, axis=1)[:, :2]
    nn_val = np.take_along_axis(D, nn_idx, axis=1)
    order = np.argsort(nn_val, axis=1)
    nn_idx = np.take_along_axis(nn_idx, order, axis=1)
    nn_val = np.take_along_axis(nn_val, order, axis=1)

    best_idx, second_idx = nn_idx[:, 0], nn_idx[:, 1]
    best_val, second_val = nn_val[:, 0], nn_val[:, 1]
    n_raw = len(best_idx)

    # --- Lowe's ratio test -------------------------------------------------- #
    # Guard against a zero second-best distance (identical descriptors).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ok = best_val < ratio * second_val
    ratio_ok &= np.isfinite(best_val)
    idx1 = np.nonzero(ratio_ok)[0]
    idx2 = best_idx[ratio_ok]
    n_after_ratio = len(idx1)

    # --- Cross-check -------------------------------------------------------- #
    if cross_check and n_after_ratio > 0:
        back_best = np.argmin(D, axis=0)          # nearest in image 1 for each of image 2
        keep = back_best[idx2] == idx1
        idx1, idx2 = idx1[keep], idx2[keep]

    matches = np.stack([idx1, idx2], axis=1)
    t1 = time.perf_counter()

    info = {"n_raw": int(n_raw),
            "n_after_ratio": int(n_after_ratio),
            "n_after_cross": int(len(matches)),
            "match_ms": (t1 - t0) * 1e3,
            "ratio": ratio}
    return matches, info


def matched_points(kps1, kps2, matches):
    """Convert index pairs into two aligned (M, 2) coordinate arrays."""
    if len(matches) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p1 = np.array([kps1[i].pt for i in matches[:, 0]], dtype=np.float64)
    p2 = np.array([kps2[j].pt for j in matches[:, 1]], dtype=np.float64)
    return p1, p2


# --------------------------------------------------------------------------- #
# Validation of the hand-written matcher
# --------------------------------------------------------------------------- #

def verify_against_opencv(desc1, desc2, kind, ratio=0.75):
    """Compare our matcher with ``cv2.BFMatcher`` (ratio test only, no
    cross-check, so that the two are directly comparable).

    Returns the agreement fraction in [0, 1].  A value of 1.0 means the two
    implementations produced exactly the same set of correspondences.
    """
    if len(desc1) < 2 or len(desc2) < 2:
        return float("nan")

    norm = cv2.NORM_HAMMING if kind == BINARY_DESC else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    cv_pairs = set()
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            cv_pairs.add((pair[0].queryIdx, pair[0].trainIdx))

    ours, _ = match_descriptors(desc1, desc2, kind, ratio=ratio, cross_check=False)
    our_pairs = set(map(tuple, ours.tolist()))

    if not cv_pairs and not our_pairs:
        return 1.0
    union = cv_pairs | our_pairs
    return len(cv_pairs & our_pairs) / len(union)
