"""
Dataset handling.

The eight base images are real photographs that are shipped with this
submission in data/images. This module can re download them from their source
if they are missing, and it generates the controlled degradations that are used
for the robustness study.
"""

import subprocess

import cv2
import numpy as np

from config import BASE_IMAGES, IMAGE_DIR, VARIANTS, RANDOM_SEED, image_record
from preprocessing import load_bgr, resize_max_side

DOWNLOAD_URLS = {
    "seeds_sunflower_01.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/"
        "Sunflower_seeds._img_007.jpg/1280px-Sunflower_seeds._img_007.jpg",
    "seeds_sunflower_02.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/"
        "Sunflower_seeds._img_008.jpg/1280px-Sunflower_seeds._img_008.jpg",
    "seeds_sunflower_03.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/"
        "Sunflower_seeds._img_009.jpg/1280px-Sunflower_seeds._img_009.jpg",
    "seeds_pale_04.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/"
        "Sunflower_seeds_-_white._img_09.jpg/1280px-Sunflower_seeds_-_white._img_09.jpg",
    "beans_cowpea_scale_05.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/"
        "Vigna_unguiculata_03.jpg/1280px-Vigna_unguiculata_03.jpg",
    "beans_coffee_06.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/5/5b/"
        "Dark_roasted_coffee_on_white_background.jpg",
    "coins_greek_07.png":
        "https://raw.githubusercontent.com/scikit-image/scikit-image/"
        "v0.24.0/skimage/data/coins.png",
    "beans_pinto_dense_08.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "Pinto_Beans_Seeds.jpg/1280px-Pinto_Beans_Seeds.jpg",
}


def ensure_base_images():
    """Download any base photograph that is not already present."""
    missing = [r for r in BASE_IMAGES if not (IMAGE_DIR / r["file"]).exists()]
    for rec in missing:
        url = DOWNLOAD_URLS[rec["file"]]
        target = IMAGE_DIR / rec["file"]
        print("downloading", rec["file"])
        subprocess.run(["curl", "-sL", "-o", str(target),
                        "-A", "CSCD608-coursework/1.0", url], check=True)
    return [IMAGE_DIR / r["file"] for r in BASE_IMAGES]


# ----------------------------------------------------------------------------
# Controlled degradations
# ----------------------------------------------------------------------------
def apply_illumination_gradient(img, low=0.30, high=1.20):
    """Multiply the image by a diagonal brightness ramp.

    This reproduces the very common situation of a scene lit from one side,
    and it is the condition that breaks a single global threshold.
    """
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xs / max(1, w - 1) + ys / max(1, h - 1)) / 2.0
    field = low + (high - low) * ramp
    out = img.astype(np.float32) * field[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_dim(img, gain=0.45, lift=25):
    """Under expose the image and lift the black point, which compresses the
    contrast between object and background."""
    out = img.astype(np.float32) * gain + lift
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(img, sigma=18.0, seed=RANDOM_SEED):
    """Additive zero mean Gaussian noise, the standard sensor noise model."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_blur(img, ksize=15):
    """Defocus blur, modelled with a Gaussian point spread function."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


_VARIANT_FUNCS = {
    "illumination_gradient": apply_illumination_gradient,
    "dim": apply_dim,
    "noise": apply_noise,
    "blur": apply_blur,
}


def build_variants(force=False):
    """Create every degraded image listed in the manifest."""
    made = []
    for var in VARIANTS:
        target = IMAGE_DIR / (var["id"] + ".png")
        if target.exists() and not force:
            made.append(target)
            continue
        base = image_record(var["base"])
        img = load_bgr(IMAGE_DIR / base["file"])
        out = _VARIANT_FUNCS[var["kind"]](img)
        cv2.imwrite(str(target), out)
        made.append(target)
    return made


def load_working(image_id):
    """Load one image of the dataset at the common working resolution."""
    rec = image_record(image_id)
    img = load_bgr(IMAGE_DIR / rec["file"])
    work, scale = resize_max_side(img)
    return work, rec, scale


if __name__ == "__main__":
    ensure_base_images()
    paths = build_variants(force=True)
    print("variants written:", len(paths))
