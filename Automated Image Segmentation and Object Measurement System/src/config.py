"""
Project wide configuration and the dataset manifest.

CSCD608 Advanced Computer Vision, Final Examination, Question 4
Automated Image Segmentation and Object Measurement System

Author : Nii Yartey Gidiglo
ID     : 22424650
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
GT_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MEAS_DIR = RESULTS_DIR / "measurements"
REPORT_DIR = ROOT_DIR / "report"

for _d in (IMAGE_DIR, GT_DIR, FIG_DIR, TABLE_DIR, MEAS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Longest side that images are resized to before processing. Keeping every
# image at a common working resolution makes the timing comparison fair and
# keeps the morphological kernel sizes meaningful across the dataset.
WORK_MAX_SIDE = 900

RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Dataset manifest
#
# Every base image is a real photograph. Nothing in this dataset was generated
# by an image generation model. The source URL and licence of each photograph
# are recorded here and reproduced in the report.
#
# The "variant" images are controlled degradations that are computed from the
# base photographs by src/dataset.py (illumination gradient, dim exposure,
# sensor noise, defocus blur). They are used for the robustness study, and
# they inherit the ground truth mask of the base image because the geometry of
# the objects is unchanged.
# ----------------------------------------------------------------------------

BASE_IMAGES = [
    {
        "id": "seeds_sunflower_01",
        "file": "seeds_sunflower_01.jpg",
        "title": "Sunflower seeds, scattered",
        "domain": "Counting and measuring seeds",
        "expected_count": None,
        "px_per_mm": None,
        "ground_truth": "pixel",
        "notes": (
            "Dark seeds on a bright uniform surface. Most seeds are well "
            "separated, a cluster on the right is touching, and several seeds "
            "are cut by the image border."
        ),
        "source": "Wikimedia Commons, File:Sunflower seeds. img 007.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Sunflower_seeds._img_007.jpg",
        "licence": "CC BY-SA 4.0",
    },
    {
        "id": "seeds_sunflower_02",
        "file": "seeds_sunflower_02.jpg",
        "title": "Sunflower seeds, second layout",
        "domain": "Counting and measuring seeds",
        "expected_count": None,
        "px_per_mm": None,
        "ground_truth": None,
        "notes": "Same object class, different arrangement, more touching pairs.",
        "source": "Wikimedia Commons, File:Sunflower seeds. img 008.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Sunflower_seeds._img_008.jpg",
        "licence": "CC BY-SA 4.0",
    },
    {
        "id": "seeds_sunflower_03",
        "file": "seeds_sunflower_03.jpg",
        "title": "Sunflower seeds, third layout",
        "domain": "Counting and measuring seeds",
        "expected_count": None,
        "px_per_mm": None,
        "ground_truth": None,
        "notes": "Shallow depth of field, part of the field is defocused.",
        "source": "Wikimedia Commons, File:Sunflower seeds. img 009.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Sunflower_seeds._img_009.jpg",
        "licence": "CC BY-SA 4.0",
    },
    {
        "id": "seeds_pale_04",
        "file": "seeds_pale_04.jpg",
        "title": "Pale sunflower seeds of three sizes",
        "domain": "Counting and measuring seeds",
        "expected_count": 3,
        "px_per_mm": None,
        "ground_truth": "instance",
        "notes": (
            "Low contrast case. The seeds are almost the same brightness as "
            "the background, and the three objects differ strongly in size."
        ),
        "source": "Wikimedia Commons, File:Sunflower seeds - white. img 09.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Sunflower_seeds_-_white._img_09.jpg",
        "licence": "CC BY-SA 4.0",
    },
    {
        "id": "beans_cowpea_scale_05",
        "file": "beans_cowpea_scale_05.jpg",
        "title": "Cowpea seeds photographed with a centimetre scale",
        "domain": "Analysing objects in agricultural images",
        "expected_count": 8,
        # Measured in src/calibration.py from the printed ruler in the photo.
        "px_per_mm": 13.0,
        "ground_truth": "instance",
        "notes": (
            "Eight well separated cowpea seeds photographed next to a printed "
            "centimetre ruler, which allows pixel measurements to be converted "
            "into millimetres."
        ),
        "source": "Wikimedia Commons, File:Vigna unguiculata 03.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Vigna_unguiculata_03.jpg",
        "licence": "CC BY-SA 3.0",
        "has_ruler": True,
    },
    {
        "id": "beans_coffee_06",
        "file": "beans_coffee_06.jpg",
        "title": "Roasted coffee beans with cast shadows",
        "domain": "Separating foreground objects from a simple background",
        "expected_count": 5,
        "px_per_mm": None,
        "ground_truth": "instance",
        "notes": (
            "Five glossy beans on a cream surface. Two beans touch, all of "
            "them cast soft shadows, and specular highlights sit inside the "
            "objects."
        ),
        "source": "Wikimedia Commons, File:Dark roasted coffee on white background.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Dark_roasted_coffee_on_white_background.jpg",
        "licence": "CC BY-SA 4.0",
    },
    {
        "id": "coins_greek_07",
        "file": "coins_greek_07.png",
        "title": "Twenty four Greek coins",
        "domain": "Identifying coins or circular objects",
        "expected_count": 24,
        "px_per_mm": None,
        "ground_truth": "instance",
        "notes": (
            "The classic coins photograph distributed with scikit-image. The "
            "background is dark, the illumination falls off towards the "
            "bottom left corner, and coin brightness overlaps background "
            "brightness in the darkest corner."
        ),
        "source": "scikit-image sample data, coins.png, Brooklyn Museum collection",
        "url": "https://github.com/scikit-image/scikit-image/blob/main/skimage/data/coins.png",
        "licence": "Public domain",
    },
    {
        "id": "beans_pinto_dense_08",
        "file": "beans_pinto_dense_08.jpg",
        "title": "Densely piled pinto beans",
        "domain": "Counting and measuring seeds, hard case",
        "expected_count": None,
        "px_per_mm": None,
        "ground_truth": None,
        "notes": (
            "A deliberate stress case. Roughly two hundred beans are piled so "
            "that almost every bean touches or overlaps a neighbour, which "
            "breaks the one region per object assumption."
        ),
        "source": "Wikimedia Commons, File:Pinto Beans Seeds.jpg",
        "url": "https://commons.wikimedia.org/wiki/File:Pinto_Beans_Seeds.jpg",
        "licence": "CC BY-SA 4.0",
    },
]

# Controlled degradations applied to two of the base photographs.
VARIANTS = [
    {"id": "seeds_sunflower_01_illum", "base": "seeds_sunflower_01",
     "kind": "illumination_gradient", "label": "Strong lighting gradient"},
    {"id": "seeds_sunflower_01_dim", "base": "seeds_sunflower_01",
     "kind": "dim", "label": "Under exposed, low contrast"},
    {"id": "seeds_sunflower_01_noise", "base": "seeds_sunflower_01",
     "kind": "noise", "label": "Additive Gaussian sensor noise"},
    {"id": "seeds_sunflower_01_blur", "base": "seeds_sunflower_01",
     "kind": "blur", "label": "Defocus blur"},
    {"id": "coins_greek_07_illum", "base": "coins_greek_07",
     "kind": "illumination_gradient", "label": "Strong lighting gradient"},
    {"id": "coins_greek_07_noise", "base": "coins_greek_07",
     "kind": "noise", "label": "Additive Gaussian sensor noise"},
]


def has_ground_truth(image_id):
    """True when the image, or the image a variant was derived from, has a
    manually verified ground truth mask."""
    rec = image_record(image_id)
    return rec.get("ground_truth") is not None


def has_instance_ground_truth(image_id):
    """True when the ground truth carries one region per physical object, so
    that object level counting scores are meaningful."""
    rec = image_record(image_id)
    return rec.get("ground_truth") == "instance"


def image_record(image_id):
    """Return the manifest record for a base image or a variant."""
    for rec in BASE_IMAGES:
        if rec["id"] == image_id:
            return rec
    for var in VARIANTS:
        if var["id"] == image_id:
            base = image_record(var["base"])
            rec = dict(base)
            rec.update({
                "id": var["id"],
                "file": var["id"] + ".png",
                "title": base["title"] + ", " + var["label"].lower(),
                "variant_of": var["base"],
                "variant_kind": var["kind"],
                "variant_label": var["label"],
            })
            return rec
    raise KeyError(image_id)


def all_image_ids(include_variants=True):
    ids = [r["id"] for r in BASE_IMAGES]
    if include_variants:
        ids += [v["id"] for v in VARIANTS]
    return ids
