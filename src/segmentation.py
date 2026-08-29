"""
The three segmentation techniques required by Question 4.

    1. Threshold based segmentation, including an Otsu implementation written
       from first principles so that the criterion itself is visible.
    2. Region growing, written from scratch with an explicit seed set, an
       explicit similarity predicate and an explicit queue.
    3. K means clustering, written from scratch as Lloyd iterations with a
       k means plus plus initialisation.

Each function returns a binary mask in which object pixels are 255 and
background pixels are 0.
"""

import time

import cv2
import numpy as np

from config import RANDOM_SEED


# ============================================================================
# Shared helpers
# ============================================================================
def _as_mask(bool_array):
    return (bool_array.astype(np.uint8)) * 255


def postprocess(mask, open_k=3, close_k=5, min_area=60, fill_holes=True,
                clear_border=False):
    """Morphological clean up of a raw binary segmentation.

    Opening removes isolated noise pixels, closing repairs small gaps in the
    object interiors, hole filling closes the specular highlights that appear
    inside glossy objects, and the area filter removes fragments that are far
    too small to be a real object.
    """
    out = mask.copy()
    if open_k and open_k > 1:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se)
    if close_k and close_k > 1:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se)
    if fill_holes:
        out = fill_object_holes(out)
    if min_area and min_area > 0:
        out = remove_small_components(out, min_area)
    if clear_border:
        out = remove_border_components(out)
    return out


def fill_object_holes(mask):
    """Fill every hole that is fully enclosed by an object.

    The mask is first padded with a one pixel background border so that the
    flood always starts in background, even when an object touches the corner
    of the image. Whatever background the flood reaches stays background, and
    every enclosed background pocket becomes object.
    """
    h, w = mask.shape[:2]
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    flood = padded.copy()
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:h + 1, 1:w + 1]
    return cv2.bitwise_or(mask, holes)


def remove_small_components(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def remove_border_components(mask):
    n, labels, _, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | \
        set(labels[:, 0]) | set(labels[:, -1])
    out = np.zeros_like(mask)
    for i in range(1, n):
        if i not in border_labels:
            out[labels == i] = 255
    return out


# ============================================================================
# 1. Threshold based segmentation
# ============================================================================
def otsu_threshold_from_scratch(gray):
    """Compute Otsu's threshold directly from the definition.

    Otsu chooses the grey level t that maximises the between class variance

        sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2

    where w0 and w1 are the class probabilities and mu0 and mu1 the class
    means. The implementation below evaluates that expression for all 256
    candidate thresholds using cumulative sums, so it runs in linear time in
    the number of grey levels.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0, np.zeros(256)
    p = hist / total
    levels = np.arange(256)
    w0 = np.cumsum(p)                      # probability of the low class
    w1 = 1.0 - w0                          # probability of the high class
    mu_cum = np.cumsum(p * levels)
    mu_total = mu_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, mu_cum / w0, 0.0)
        mu1 = np.where(w1 > 0, (mu_total - mu_cum) / w1, 0.0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = 0.0
    t = int(np.argmax(between))
    return t, between


def segment_threshold(gray, method="otsu", objects_dark=True, manual_value=128,
                      block_size=51, C=8, offset=0):
    """Threshold based segmentation.

    method = "otsu"     : global threshold from the from scratch Otsu routine
    method = "global"   : fixed threshold supplied by the caller
    method = "adaptive" : locally adaptive Gaussian threshold, which handles a
                          lighting gradient without an explicit flat field step
    """
    if method == "otsu":
        t, _ = otsu_threshold_from_scratch(gray)
        t = int(np.clip(t + offset, 0, 255))
        mask = gray <= t if objects_dark else gray >= t
        return _as_mask(mask), {"threshold": t}
    if method == "global":
        t = int(manual_value)
        mask = gray <= t if objects_dark else gray >= t
        return _as_mask(mask), {"threshold": t}
    if method == "adaptive":
        bs = block_size if block_size % 2 == 1 else block_size + 1
        ttype = cv2.THRESH_BINARY_INV if objects_dark else cv2.THRESH_BINARY
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     ttype, bs, C)
        return mask, {"block_size": bs, "C": C}
    raise ValueError("Unknown threshold method: %s" % method)


# ============================================================================
# 2. Region growing
# ============================================================================
def _neighbour_offsets(connectivity):
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return ((-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1))


def pick_seeds(gray, strategy="distance", n_seeds=40, objects_dark=True,
               coarse_mask=None, min_distance=12):
    """Choose the seed points for region growing.

    distance : threshold the image coarsely, take the distance transform of the
               object mask and place one seed at the deepest point of every
               connected component. This is an informed seeding strategy and it
               places one seed near the centre of each object.
    grid     : place seeds on a regular lattice and keep only those that fall
               on the object side of a coarse threshold. This is an uninformed
               strategy and it is used in the report to show how sensitive the
               method is to seed placement.
    random   : sample seeds uniformly at random from the object side of a
               coarse threshold.
    """
    if coarse_mask is None:
        coarse_mask, _ = segment_threshold(gray, "otsu", objects_dark)
        coarse_mask = postprocess(coarse_mask, 3, 5, 40)
    binary = (coarse_mask > 0).astype(np.uint8)

    if strategy == "distance":
        n, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        seeds = []
        for i in range(1, n):
            comp = labels == i
            if comp.sum() < 20:
                continue
            d = np.where(comp, dist, 0)
            y, x = np.unravel_index(int(np.argmax(d)), d.shape)
            seeds.append((int(y), int(x)))
        return seeds

    if strategy == "grid":
        h, w = gray.shape[:2]
        step = max(4, int(np.sqrt(h * w / float(max(1, n_seeds)))))
        seeds = []
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                if binary[y, x]:
                    seeds.append((y, x))
        return seeds

    if strategy == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        ys, xs = np.nonzero(binary)
        if len(ys) == 0:
            return []
        idx = rng.choice(len(ys), size=min(n_seeds, len(ys)), replace=False)
        return [(int(ys[i]), int(xs[i])) for i in idx]

    raise ValueError("Unknown seed strategy: %s" % strategy)


def region_growing(gray, seeds, tolerance=18, connectivity=8,
                   criterion="running_mean", max_pixels_per_region=None):
    """Grow regions from the given seed points.

    The similarity predicate is the absolute difference between the intensity
    of the candidate pixel and a reference intensity of the region. Two
    predicates are supported:

    running_mean : the reference is the mean intensity of the pixels accepted
                   so far, which lets a region follow a slow intensity drift
    seed_value   : the reference is fixed at the intensity of the seed, which
                   is stricter and stops the region leaking through a shading
                   gradient

    The queue is a plain first in first out list, so the traversal is a breadth
    first flood of the similarity region. Every pixel is visited at most once,
    which makes the routine linear in the number of pixels.
    """
    h, w = gray.shape[:2]
    img = gray.astype(np.float32)
    visited = np.zeros((h, w), dtype=bool)
    labels = np.zeros((h, w), dtype=np.int32)
    offsets = _neighbour_offsets(connectivity)

    from collections import deque

    for label_id, (sy, sx) in enumerate(seeds, start=1):
        if not (0 <= sy < h and 0 <= sx < w) or visited[sy, sx]:
            continue
        seed_value = float(img[sy, sx])
        total = seed_value
        count = 1
        visited[sy, sx] = True
        labels[sy, sx] = label_id
        queue = deque()
        queue.append((sy, sx))
        while queue:
            y, x = queue.popleft()
            reference = (total / count) if criterion == "running_mean" else seed_value
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx]:
                    continue
                if abs(float(img[ny, nx]) - reference) <= tolerance:
                    visited[ny, nx] = True
                    labels[ny, nx] = label_id
                    total += float(img[ny, nx])
                    count += 1
                    queue.append((ny, nx))
                    if max_pixels_per_region and count > max_pixels_per_region:
                        queue.clear()
                        break
    return labels


def segment_region_growing(gray, objects_dark=True, tolerance=18,
                           seed_strategy="distance", n_seeds=40,
                           connectivity=8, criterion="running_mean"):
    """Region growing wrapped as a binary segmentation."""
    seeds = pick_seeds(gray, seed_strategy, n_seeds, objects_dark)
    labels = region_growing(gray, seeds, tolerance, connectivity, criterion)
    mask = _as_mask(labels > 0)
    return mask, {"n_seeds": len(seeds), "tolerance": tolerance,
                  "seed_strategy": seed_strategy, "criterion": criterion,
                  "seeds": seeds}


# ============================================================================
# 3. K means clustering
# ============================================================================
def _kmeans_plus_plus_init(X, k, rng):
    """k means plus plus seeding: spread the initial centres apart."""
    n = X.shape[0]
    centres = np.empty((k, X.shape[1]), dtype=np.float32)
    centres[0] = X[rng.integers(n)]
    closest = np.sum((X - centres[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:
            centres[i] = X[rng.integers(n)]
        else:
            probs = closest / total
            centres[i] = X[rng.choice(n, p=probs)]
        d = np.sum((X - centres[i]) ** 2, axis=1)
        closest = np.minimum(closest, d)
    return centres


def kmeans_from_scratch(X, k, max_iter=40, tol=1e-4, seed=RANDOM_SEED):
    """Lloyd's algorithm.

    Assignment step: every sample is attached to the nearest centre.
    Update step   : every centre moves to the mean of the samples attached to
                    it. The loop stops when no centre moves further than tol.
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    centres = _kmeans_plus_plus_init(X, k, rng)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        # Assignment step, computed in blocks to keep memory bounded.
        block = 200000
        for start in range(0, X.shape[0], block):
            chunk = X[start:start + block]
            d = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            labels[start:start + block] = np.argmin(d, axis=1)
        # Update step.
        shift = 0.0
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                new_centre = X[rng.integers(X.shape[0])]
            else:
                new_centre = members.mean(axis=0)
            shift = max(shift, float(np.linalg.norm(new_centre - centres[j])))
            centres[j] = new_centre
        if shift < tol:
            break
    return labels, centres


def build_features(img_bgr, gray_prepared, feature_space="intensity",
                   spatial_weight=0.15):
    """Assemble the per pixel feature vectors used by the clustering."""
    h, w = gray_prepared.shape[:2]
    if feature_space == "intensity":
        return gray_prepared.reshape(-1, 1).astype(np.float32)
    if feature_space == "lab":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        return lab.reshape(-1, 3).astype(np.float32)
    if feature_space == "lab_xy":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ys, xs = np.mgrid[0:h, 0:w]
        sx = (xs.astype(np.float32) / w) * 255.0 * spatial_weight
        sy = (ys.astype(np.float32) / h) * 255.0 * spatial_weight
        return np.concatenate(
            [lab.reshape(-1, 3), sx.reshape(-1, 1), sy.reshape(-1, 1)], axis=1)
    raise ValueError("Unknown feature space: %s" % feature_space)


def segment_kmeans(img_bgr, gray_prepared, k=2, feature_space="intensity",
                   objects_dark=True, border=10, use_opencv=False):
    """K means segmentation with an automatic foreground cluster decision.

    After clustering, the clusters that dominate the border ring of the image
    are declared background, since the border of every image in this dataset is
    background. Any remaining cluster is foreground. When every cluster touches
    the border, the single cluster whose mean intensity is furthest from the
    border mean in the expected direction is taken as the object cluster.
    """
    h, w = gray_prepared.shape[:2]
    X = build_features(img_bgr, gray_prepared, feature_space)

    if use_opencv:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
        _, lab, centres = cv2.kmeans(X, k, None, criteria, 3,
                                     cv2.KMEANS_PP_CENTERS)
        labels = lab.ravel()
    else:
        labels, centres = kmeans_from_scratch(X, k)

    label_img = labels.reshape(h, w)

    # Border ring statistics decide which clusters are background.
    ring = np.zeros((h, w), dtype=bool)
    ring[:border, :] = True
    ring[-border:, :] = True
    ring[:, :border] = True
    ring[:, -border:] = True
    ring_labels = label_img[ring]

    mean_intensity = np.array(
        [float(gray_prepared[label_img == j].mean()) if np.any(label_img == j)
         else 0.0 for j in range(k)])

    background = set()
    for j in range(k):
        share_of_ring = float((ring_labels == j).sum()) / max(1, ring_labels.size)
        if share_of_ring > 0.25:
            background.add(j)
    if len(background) == 0 or len(background) == k:
        # Fall back to the intensity ordering of the cluster means.
        order = np.argsort(mean_intensity)
        object_cluster = order[0] if objects_dark else order[-1]
        background = set(range(k)) - {int(object_cluster)}

    mask = np.isin(label_img, list(set(range(k)) - background))
    return _as_mask(mask), {"k": k, "feature_space": feature_space,
                            "cluster_means": mean_intensity.tolist(),
                            "background_clusters": sorted(background),
                            "label_image": label_img}


# ============================================================================
# Optional improvement stage: splitting touching objects
# ============================================================================
def split_touching(mask, dist_ratio=0.45):
    """Separate touching objects with a distance transform and watershed.

    The distance transform of the object mask peaks at the centre of every
    object. Thresholding it produces one marker per object even when the
    objects share a boundary, and the watershed transform then grows those
    markers back to the object boundaries.
    """
    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask.copy(), 0

    # The marker threshold is applied inside each connected component
    # separately. A single global threshold would erase every object that is
    # smaller than the largest one in the image, which silently loses objects
    # when the scene contains objects of different sizes.
    n_comp, comp_labels, _, _ = cv2.connectedComponentsWithStats(binary,
                                                                connectivity=8)
    sure_fg = np.zeros_like(binary)
    for i in range(1, n_comp):
        comp = comp_labels == i
        local_max = float(dist[comp].max())
        if local_max <= 0:
            continue
        sure_fg[np.logical_and(comp, dist >= dist_ratio * local_max)] = 255
    n_markers, markers = cv2.connectedComponents(sure_fg)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, se, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg // 255)
    markers = markers + 1
    markers[unknown > 0] = 0
    colour = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(colour, markers)
    out = np.zeros_like(mask)
    out[markers > 1] = 255
    # The watershed lines are one pixel wide, which is not enough to break an
    # eight connected diagonal link between two neighbouring objects, so the
    # lines are widened by one pixel before they are cut out of the mask.
    line = np.zeros_like(mask)
    line[markers == -1] = 255
    line = cv2.dilate(line, se, iterations=1)
    out[line > 0] = 0
    out = cv2.bitwise_and(out, mask)
    return out, max(0, n_markers - 1)


# ============================================================================
# Timed wrapper used by the experiment driver
# ============================================================================
def run_method(name, prep, params):
    """Run one segmentation method and report the wall clock time.

    prep is the dictionary returned by preprocessing.prepare.
    """
    gray = prep["prepared"]
    dark = prep["objects_are_dark"]
    t0 = time.perf_counter()

    if name == "threshold":
        mask, info = segment_threshold(
            gray, params.get("mode", "otsu"), dark,
            manual_value=params.get("manual_value", 128),
            block_size=params.get("block_size", 51),
            C=params.get("C", 8),
            offset=params.get("offset", 0))
    elif name == "region_growing":
        mask, info = segment_region_growing(
            gray, dark,
            tolerance=params.get("tolerance", 18),
            seed_strategy=params.get("seed_strategy", "distance"),
            n_seeds=params.get("n_seeds", 40),
            connectivity=params.get("connectivity", 8),
            criterion=params.get("criterion", "running_mean"))
    elif name == "kmeans":
        mask, info = segment_kmeans(
            prep["bgr"], gray,
            k=params.get("k", 2),
            feature_space=params.get("feature_space", "intensity"),
            objects_dark=dark,
            use_opencv=params.get("use_opencv", False))
    else:
        raise ValueError("Unknown method: %s" % name)

    mask = postprocess(mask,
                       open_k=params.get("open_k", 3),
                       close_k=params.get("close_k", 5),
                       min_area=params.get("min_area", 80),
                       fill_holes=params.get("fill_holes", True))
    elapsed = time.perf_counter() - t0
    info["time_s"] = elapsed
    return mask, info
