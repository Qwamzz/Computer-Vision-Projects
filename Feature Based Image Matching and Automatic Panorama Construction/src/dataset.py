"""
dataset.py
==========
Image acquisition for the panorama project.

Two acquisition modes are supported:

1. REAL MODE  -- if the ``images/`` folder contains >= 3 photographs, they are
   loaded directly.  This is the mode to use for photographs taken with a phone
   or camera.

2. SCENE MODE -- a single large planar "scene" is used, and a set of overlapping
   views is produced by warping that scene with *known* homographies.  Because
   the ground-truth homography relating any two views is known exactly, the
   accuracy of the estimated homography can be measured in pixels rather than
   merely inspected by eye.  This is what makes the quantitative evaluation in
   this project genuinely quantitative.

   The scene itself is a real photograph, a view of the fishing harbour at
   Elmina downloaded from Wikimedia Commons by ``download_scene.py``.  Using a
   photograph rather than a drawn pattern means the detectors are tested on
   genuine photographic texture -- boat hulls, masts, nets and roofs, and a wide
   band of almost featureless sky -- while the ground truth stays exact.  If the
   photograph has not been downloaded, a procedurally drawn scene is used
   instead so that the pipeline still runs offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SCENE_W, SCENE_H = 2000, 900     # size of the virtual planar scene
VIEW_W, VIEW_H = 800, 600        # size of each simulated camera view
RNG_SEED = 20252026              # fixed so that every run is reproducible

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# The real photograph used as the planar scene, fetched by download_scene.py.
SCENE_PHOTO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "scene", "harbour_scene.jpg")
# Vertical placement of the crop taken from that photograph. Slightly below the
# middle, which keeps the fort and the full width of the moored boats while
# trimming some of the featureless sky.
SCENE_CROP_BIAS = 0.62


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #

@dataclass
class ViewSet:
    """A set of overlapping views of one scene.

    Attributes
    ----------
    images : list of BGR uint8 arrays
    names  : human readable name of each view
    H_scene_to_view : list of 3x3 arrays or None
        Ground-truth homography mapping scene coordinates to the coordinates of
        each view.  ``None`` in REAL mode, where no ground truth exists.
    synthetic : whether ground truth is available
    """
    images: list = field(default_factory=list)
    names: list = field(default_factory=list)
    H_scene_to_view: list = field(default_factory=list)
    synthetic: bool = True

    def __len__(self):
        return len(self.images)

    def gt_homography(self, src_idx: int, dst_idx: int):
        """Ground-truth homography mapping points of view ``src_idx`` into the
        coordinate frame of view ``dst_idx``.

        If  x_view_i = H_i * x_scene  then  x_view_j = H_j * H_i^-1 * x_view_i.
        """
        if not self.synthetic:
            return None
        H_src = self.H_scene_to_view[src_idx]
        H_dst = self.H_scene_to_view[dst_idx]
        return H_dst @ np.linalg.inv(H_src)


# --------------------------------------------------------------------------- #
# Procedural scene generation
# --------------------------------------------------------------------------- #

def _texture_background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """A smoothly varying, mildly textured background.

    A perfectly flat background would give the detectors nothing to lock onto
    over large parts of the image, which is unrealistic; real scenes carry
    texture almost everywhere.
    """
    # Low-frequency colour gradient (sky/ground feel).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 40 + 90 * (yy / h) + 40 * np.sin(xx / 260.0)
    g = 70 + 70 * (yy / h) + 30 * np.cos(xx / 310.0 + 1.0)
    b = 150 - 80 * (yy / h) + 25 * np.sin(xx / 190.0 + 2.0)
    bg = np.stack([b, g, r], axis=-1)

    # Band-limited noise adds fine texture (blurred white noise).
    noise = rng.normal(0, 42, (h, w, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.6)
    bg += noise
    return np.clip(bg, 0, 255).astype(np.uint8)


def _random_colour(rng):
    return tuple(int(c) for c in rng.integers(0, 256, 3))


def load_scene_photo(path: str = SCENE_PHOTO, w: int = SCENE_W,
                     h: int = SCENE_H) -> "np.ndarray | None":
    """Load the real photograph and cut the scene canvas out of it.

    The photograph is wider than it is tall but not as wide as the scene
    canvas, so a horizontal strip of the correct aspect ratio is taken and
    scaled down to the canvas size.  Downscaling rather than upscaling keeps
    the fine detail that the detectors rely on.
    """
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    src_h, src_w = img.shape[:2]
    want = w / float(h)
    strip_h = int(round(src_w / want))
    if strip_h <= src_h:
        y0 = int(round((src_h - strip_h) * SCENE_CROP_BIAS))
        img = img[y0:y0 + strip_h, :]
    else:                                   # photograph is taller than needed
        strip_w = int(round(src_h * want))
        x0 = (src_w - strip_w) // 2
        img = img[:, x0:x0 + strip_w]
    interp = cv2.INTER_AREA if img.shape[1] >= w else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def build_scene(w: int = SCENE_W, h: int = SCENE_H, seed: int = RNG_SEED,
                force_drawn: bool = False) -> np.ndarray:
    """Return the planar scene: the real photograph when it is available.

    Falls back to the procedurally drawn scene below when the photograph has
    not been downloaded, so the project still runs without a network.  Pass
    ``force_drawn`` to use the drawn scene even when the photograph is present.
    """
    if not force_drawn:
        photo = load_scene_photo(w=w, h=h)
        if photo is not None:
            return photo
    return build_synthetic_scene(w, h, seed)


def build_synthetic_scene(w: int = SCENE_W, h: int = SCENE_H,
                          seed: int = RNG_SEED) -> np.ndarray:
    """Procedurally build one wide, feature-rich planar scene (offline fallback)."""
    rng = np.random.default_rng(seed)
    img = _texture_background(w, h, rng)

    # --- Large coloured polygons: strong edges and corners ------------------ #
    for _ in range(55):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        n_pts = int(rng.integers(3, 7))
        radius = int(rng.integers(25, 120))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
        pts = np.stack([cx + radius * np.cos(ang) * rng.uniform(0.5, 1.3, n_pts),
                        cy + radius * np.sin(ang) * rng.uniform(0.5, 1.3, n_pts)], axis=-1)
        cv2.fillPoly(img, [pts.astype(np.int32)], _random_colour(rng), lineType=cv2.LINE_AA)

    # --- Circles and ellipses: blob-like structures ------------------------- #
    for _ in range(45):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(8, 55)), int(rng.integers(8, 55)))
        cv2.ellipse(img, c, axes, float(rng.uniform(0, 180)), 0, 360,
                    _random_colour(rng), -1, lineType=cv2.LINE_AA)

    # --- Checkerboard patches: ideal corner features ------------------------ #
    for _ in range(9):
        cell = int(rng.integers(12, 26))
        nx, ny = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        x0, y0 = int(rng.integers(0, w - nx * cell)), int(rng.integers(0, h - ny * cell))
        c1, c2 = _random_colour(rng), _random_colour(rng)
        for iy in range(ny):
            for ix in range(nx):
                col = c1 if (ix + iy) % 2 == 0 else c2
                cv2.rectangle(img, (x0 + ix * cell, y0 + iy * cell),
                              (x0 + (ix + 1) * cell, y0 + (iy + 1) * cell), col, -1)

    # --- Text: high-frequency, highly distinctive detail -------------------- #
    words = ["HARBOUR", "VISION", "SIFT", "ORB", "RANSAC", "HOMOGRAPHY",
             "PANORAMA", "OPENCV", "KEYPOINT", "DESCRIPTOR", "INLIER", "STITCH",
             "MOSAIC", "GRADIENT", "FEATURE", "MATCHING", "WARP", "BLEND"]
    for wd in words:
        org = (int(rng.integers(20, w - 320)), int(rng.integers(40, h - 20)))
        scale = float(rng.uniform(0.9, 2.2))
        cv2.putText(img, wd, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    _random_colour(rng), int(rng.integers(2, 4)), cv2.LINE_AA)

    # --- Thin lines: elongated structures that challenge corner detectors --- #
    for _ in range(40):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (p1[0] + int(rng.integers(-220, 220)), p1[1] + int(rng.integers(-220, 220)))
        cv2.line(img, p1, p2, _random_colour(rng), int(rng.integers(1, 4)), cv2.LINE_AA)

    # A touch of blur so that edges are not perfectly aliased, as with a real lens.
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.7)
    return img


# --------------------------------------------------------------------------- #
# View synthesis
# --------------------------------------------------------------------------- #

def _quad_from_params(cx, cy, half_w, half_h, angle_deg, scale, tilt_x, tilt_y):
    """Build a quadrilateral in scene coordinates.

    The quadrilateral is a rectangle that has been rotated, scaled and given a
    perspective 'tilt'.  Mapping this quadrilateral onto the rectangular view
    produces a homography containing rotation, scale and out-of-plane rotation
    exactly as a real hand-held camera would.
    """
    hw, hh = half_w / scale, half_h / scale
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)

    # Perspective tilt: shrink one side of the rectangle towards a vanishing point.
    base[:, 0] *= (1.0 + tilt_x * base[:, 1] / max(hh, 1e-6))
    base[:, 1] *= (1.0 + tilt_y * base[:, 0] / max(hw, 1e-6))

    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (base @ R.T) + np.array([cx, cy])


def render_view(scene, quad, view_w=VIEW_W, view_h=VIEW_H):
    """Warp the scene quadrilateral into a rectangular view.

    Returns the rendered view and the homography scene -> view.
    """
    dst = np.array([[0, 0], [view_w - 1, 0],
                    [view_w - 1, view_h - 1], [0, view_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    view = cv2.warpPerspective(scene, H, (view_w, view_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return view, H.astype(np.float64)


def apply_photometric(img, gain=1.0, bias=0.0, gamma=1.0, vignette=0.0,
                      noise_sigma=0.0, seed=0):
    """Change the *appearance* of a view without changing its geometry.

    Used for the illumination-robustness experiment: because the geometry is
    untouched, the ground-truth homography remains valid, so any loss of
    accuracy is attributable purely to the photometric change.
    """
    out = img.astype(np.float32)
    if gamma != 1.0:
        out = 255.0 * np.power(np.clip(out / 255.0, 0, 1), gamma)
    out = out * gain + bias
    if vignette > 0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        mask = np.clip(1.0 - vignette * r ** 2, 0, 1)[..., None]
        out = out * mask
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_panorama_views(scene, n_views=4, overlap=0.45, seed=RNG_SEED):
    """Generate ``n_views`` overlapping views spanning the scene left to right.

    ``overlap`` is the approximate fraction of a view shared with its neighbour.
    Each view is given a small random rotation, scale and perspective tilt so
    that the problem is a genuine homography estimation problem and not a pure
    translation.
    """
    rng = np.random.default_rng(seed + 7)
    h, w = scene.shape[:2]

    half_w = w / (n_views - (n_views - 1) * overlap) / 2.0
    half_h = half_w * VIEW_H / VIEW_W
    half_h = min(half_h, h / 2 * 0.92)

    step = 2 * half_w * (1 - overlap)
    x0 = half_w + 0.02 * w

    vs = ViewSet(synthetic=True)
    for k in range(n_views):
        cx = x0 + k * step
        cy = h / 2 + rng.uniform(-0.035, 0.035) * h
        # Geometric jitter is kept modest and zero-mean.  Two reasons:
        #  * it mimics a hand-held camera panning across a scene, where the
        #    relative perspective change between neighbouring shots is small;
        #  * the homographies are *chained* during stitching, so any systematic
        #    tilt compounds along the chain and produces the extreme wedge-shaped
        #    stretching that is characteristic of planar panorama mosaics.
        # Photometric variation (below) is left large, since it lowers the inlier
        # ratio to a realistic level without corrupting the geometry.
        quad = _quad_from_params(cx, cy, half_w, half_h,
                                 angle_deg=rng.uniform(-9, 9),
                                 scale=rng.uniform(0.88, 1.12),
                                 tilt_x=rng.uniform(-0.07, 0.07),
                                 tilt_y=rng.uniform(-0.04, 0.04))
        view, H = render_view(scene, quad)
        view = apply_photometric(view, gain=rng.uniform(0.75, 1.25),
                                 bias=rng.uniform(-25, 25), gamma=rng.uniform(0.8, 1.3),
                                 vignette=rng.uniform(0.0, 0.25),
                                 noise_sigma=4.0, seed=seed + k)
        vs.images.append(view)
        vs.names.append("view_%d" % (k + 1))
        vs.H_scene_to_view.append(H)
    return vs


def make_controlled_pair(scene, rotation=0.0, scale=1.0, tilt=0.0,
                         gain=1.0, bias=0.0, gamma=1.0, noise_sigma=2.0,
                         seed=RNG_SEED):
    """Build a *reference* view and a *transformed* view of the same scene patch.

    Exactly one factor (rotation / scale / viewpoint tilt / illumination) is
    varied at a time, which is what allows the robustness study to attribute a
    change in performance to a single cause.
    """
    h, w = scene.shape[:2]
    cx, cy = w / 2, h / 2
    half_w, half_h = w * 0.24, w * 0.24 * VIEW_H / VIEW_W

    quad_a = _quad_from_params(cx, cy, half_w, half_h, 0.0, 1.0, 0.0, 0.0)
    view_a, H_a = render_view(scene, quad_a)
    view_a = apply_photometric(view_a, noise_sigma=noise_sigma, seed=seed)

    quad_b = _quad_from_params(cx, cy, half_w, half_h, rotation, scale, tilt, 0.0)
    view_b, H_b = render_view(scene, quad_b)
    view_b = apply_photometric(view_b, gain=gain, bias=bias, gamma=gamma,
                               noise_sigma=noise_sigma, seed=seed + 1)

    vs = ViewSet(synthetic=True)
    vs.images = [view_a, view_b]
    vs.names = ["reference", "transformed"]
    vs.H_scene_to_view = [H_a, H_b]
    return vs


# --------------------------------------------------------------------------- #
# Real-image loading
# --------------------------------------------------------------------------- #

def load_real_images(folder: str, max_width: int = 1000) -> ViewSet:
    """Load user-supplied photographs from ``folder`` (sorted by filename).

    Images are downscaled so that the pipeline runs in a reasonable time; the
    ordering of the filenames must correspond to the left-to-right ordering of
    the views.
    """
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(IMAGE_EXTS))
    vs = ViewSet(synthetic=False)
    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        vs.images.append(img)
        vs.names.append(os.path.splitext(f)[0])
        vs.H_scene_to_view.append(None)
    return vs


def acquire(images_folder: str, n_views: int = 4, force_scene: bool = False,
            force_drawn: bool = False):
    """Top-level acquisition: prefer the user's own photographs when present."""
    if not (force_scene or force_drawn) and os.path.isdir(images_folder):
        vs = load_real_images(images_folder)
        if len(vs) >= 3:
            print("[dataset] REAL mode: loaded %d photographs from %s"
                  % (len(vs), images_folder))
            return vs, None
    scene = build_scene(force_drawn=force_drawn)
    vs = make_panorama_views(scene, n_views=n_views)
    kind = "drawn scene" if force_drawn or load_scene_photo() is None else "photograph"
    print("[dataset] SCENE mode: cut %d overlapping views out of the %s, "
          "with known ground truth" % (len(vs), kind))
    return vs, scene


# --------------------------------------------------------------------------- #
# Pre-processing (Task 2: "image preparation")
# --------------------------------------------------------------------------- #

def preprocess(img, use_clahe: bool = True, denoise: bool = True):
    """Convert to greyscale and improve local contrast.

    Rationale
    ---------
    * Greyscale: all descriptors under test are intensity based, so colour is
      discarded before detection.
    * A light Gaussian blur suppresses sensor noise that would otherwise create
      spurious low-contrast keypoints, without removing the structures that
      carry the useful features.
    * CLAHE (Contrast Limited Adaptive Histogram Equalisation) equalises
      contrast *locally*.  This matters because ORB's FAST detector uses a fixed
      intensity threshold and therefore finds very few keypoints in dark or
      washed-out regions; CLAHE partially removes that dependence on exposure.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def save_viewset(vs: ViewSet, out_dir: str, prefix: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    for name, img in zip(vs.names, vs.images):
        cv2.imwrite(os.path.join(out_dir, "%s%s.png" % (prefix, name)), img)
    if vs.synthetic:
        meta = {n: H.tolist() for n, H in zip(vs.names, vs.H_scene_to_view)}
        with open(os.path.join(out_dir, "%sground_truth_homographies.json" % prefix),
                  "w") as fh:
            json.dump(meta, fh, indent=2)
