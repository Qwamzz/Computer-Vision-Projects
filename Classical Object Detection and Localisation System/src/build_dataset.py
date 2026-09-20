"""
build_dataset.py
Classical Object Detection and Localisation

Builds the template image and the annotated evaluation set for the
"University of Ghana coat of arms" detection task.

Every pixel of source material is real: the emblem is the public domain
University of Ghana coat of arms and the scenes are real photographs
(campus, street, office and library interiors) downloaded by
src/download_data.py. Each test scene is produced by geometrically and
photometrically transforming the emblem and compositing it into a real
photograph under controlled conditions. Because the placement geometry is
known exactly, the bounding box ground truth is exact rather than
hand-drawn, which removes annotator error from the evaluation. Every
generated scene was afterwards inspected visually and the drawn ground
truth boxes were confirmed to sit on the emblem.

Outputs
    data/template/ug_logo_template.png   reference template (BGR on white)
    data/template/ug_logo_mask.png       binary mask of the emblem
    data/dev_set/*.jpg                   development scenes (tuning allowed)
    data/test_set/*.jpg                  held out scenes (never used for tuning)
    data/annotations/dev_annotations.json
    data/annotations/test_annotations.json

Usage:
    python src/build_dataset.py
"""

import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

SCENE_W, SCENE_H = 960, 640
TEMPLATE_W = 96                      # reference width of the emblem in pixels
SEED = 20260920                      # fixed seed, so the dataset is reproducible


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("cannot read " + path)
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, np.uint8)
        img = np.dstack([img, alpha])
    return img


def trim_alpha(rgba, thr=8):
    ys, xs = np.where(rgba[:, :, 3] > thr)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def scale_to_width(rgba, width):
    h, w = rgba.shape[:2]
    height = max(1, int(round(h * width / float(w))))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
    return cv2.resize(rgba, (width, height), interpolation=interp)


def rotate_rgba(rgba, angle):
    """Rotate about the centre, expanding the canvas so nothing is clipped."""
    if abs(angle) < 1e-6:
        return rgba
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def perspective_rgba(rgba, tilt_x=0.0, tilt_y=0.0):
    """Simulate a change of viewpoint by a mild projective warp.

    tilt_x and tilt_y are fractions of the width and height by which the far
    edge of the emblem is pulled in, which is what happens when a flat object
    is photographed off axis.
    """
    if abs(tilt_x) < 1e-6 and abs(tilt_y) < 1e-6:
        return rgba
    h, w = rgba.shape[:2]
    dx, dy = tilt_x * w, tilt_y * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx, 0], [w, h], [dx, h - dy]])
    xs, ys = dst[:, 0].copy(), dst[:, 1].copy()
    dst[:, 0] -= xs.min()
    dst[:, 1] -= ys.min()
    nw = int(np.ceil(xs.max() - xs.min()))
    nh = int(np.ceil(ys.max() - ys.min()))
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def alpha_paste(scene, rgba, x, y):
    """Alpha blend rgba onto scene with its top left corner at (x, y)."""
    h, w = rgba.shape[:2]
    H, W = scene.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi = scene[y0:y1, x0:x1].astype(np.float32)
    scene[y0:y1, x0:x1] = (patch[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
    ys, xs = np.where(patch[:, :, 3] > 16)
    if len(xs) == 0:
        return None
    return [int(x0 + xs.min()), int(y0 + ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def occlude(scene, box, fraction, rng, side=None):
    """Hide part of the object with a patch copied from elsewhere in the scene.

    A copied patch is used rather than a flat rectangle so that the occluder
    has realistic texture and is not trivially separable from the object.
    """
    x, y, w, h = box
    side = side or str(rng.choice(["bottom", "top", "left", "right"]))
    if side in ("bottom", "top"):
        oh = int(round(h * fraction))
        ox, oy, ow = x, (y + h - oh if side == "bottom" else y), w
    else:
        ow = int(round(w * fraction))
        ox, oy, oh = (x + w - ow if side == "right" else x), y, h
    H, W = scene.shape[:2]
    sx, sy = 0, 0
    for _ in range(40):
        sx = int(rng.integers(0, max(1, W - ow)))
        sy = int(rng.integers(0, max(1, H - oh)))
        if abs(sx - x) > w or abs(sy - y) > h:
            break
    src = scene[sy:sy + oh, sx:sx + ow]
    if src.shape[0] != oh or src.shape[1] != ow:
        src = cv2.resize(src, (ow, oh))
    scene[oy:oy + oh, ox:ox + ow] = src
    return side


def apply_gamma(img, gamma):
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def apply_light_gradient(img, strength=0.55, direction="left"):
    """Uneven lighting, as produced by a lamp or by direct sun from one side."""
    h, w = img.shape[:2]
    n = w if direction in ("left", "right") else h
    ramp = np.linspace(1.0 + strength, 1.0 - strength, n)
    if direction == "right":
        ramp = ramp[::-1]
    field = np.tile(ramp, (h, 1)) if direction in ("left", "right") else np.tile(ramp[:, None], (1, w))
    out = img.astype(np.float32) * field[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_noise(img, sigma, rng):
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def motion_blur(img, k=9, angle=0):
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= kern.sum()
    return cv2.filter2D(img, -1, kern)


def prepare_background(path, rng):
    bg = cv2.imread(path, cv2.IMREAD_COLOR)
    if bg is None:
        raise IOError("cannot read " + path)
    h, w = bg.shape[:2]
    s = max(SCENE_W / float(w), SCENE_H / float(h))
    bg = cv2.resize(bg, (int(np.ceil(w * s)), int(np.ceil(h * s))), interpolation=cv2.INTER_AREA)
    h, w = bg.shape[:2]
    x = int(rng.integers(0, w - SCENE_W + 1))
    y = int(rng.integers(0, h - SCENE_H + 1))
    return bg[y:y + SCENE_H, x:x + SCENE_W].copy()


# --------------------------------------------------------------------------
# scene specification
# --------------------------------------------------------------------------
def scene_plan():
    """Explicit, reproducible description of every scene in the dataset.

    scale is relative to the reference template width of TEMPLATE_W pixels,
    rot is in degrees, tilt is the projective viewpoint change, pos is the
    object centre as a fraction of the scene, occ is the occluded fraction.
    """
    B = ["bg_ug_campus.jpg", "bg_balme_library.jpg", "bg_mensah_sarbah.jpg",
         "bg_accra_street.jpg", "bg_accra_scene.jpg", "bg_office_desk.jpg",
         "bg_brick_wall.jpg", "bg_library_interior.jpg", "bg_lecture_interior.jpg"]

    def inst(scale=1.0, rot=0.0, tilt=(0.0, 0.0), pos=(0.5, 0.5), occ=0.0):
        return {"scale": scale, "rot": rot, "tilt": tilt, "pos": pos, "occ": occ}

    dev = [
        ("dev01_baseline_a", B[0], "baseline", [inst(1.0, 0, (0, 0), (0.30, 0.45))], {}),
        ("dev02_baseline_b", B[6], "baseline", [inst(1.15, 0, (0, 0), (0.62, 0.40))], {}),
        ("dev03_baseline_two", B[7], "baseline", [inst(1.0, 0, (0, 0), (0.25, 0.35)),
                                                  inst(1.0, 0, (0, 0), (0.72, 0.62))], {}),
        ("dev04_scale_small", B[1], "scale", [inst(0.60, 0, (0, 0), (0.40, 0.55))], {}),
        ("dev05_scale_large", B[2], "scale", [inst(1.75, 0, (0, 0), (0.55, 0.50))], {}),
        ("dev06_scale_mixed", B[8], "scale", [inst(0.70, 0, (0, 0), (0.22, 0.30)),
                                              inst(1.45, 0, (0, 0), (0.70, 0.60))], {}),
        ("dev07_rot_small", B[3], "rotation", [inst(1.0, 8, (0, 0), (0.45, 0.45))], {}),
        ("dev08_rot_medium", B[4], "rotation", [inst(1.1, 22, (0, 0), (0.55, 0.50))], {}),
        ("dev09_rot_large", B[5], "rotation", [inst(1.0, 45, (0, 0), (0.40, 0.50))], {}),
        ("dev10_illum_dark", B[0], "illumination", [inst(1.05, 0, (0, 0), (0.50, 0.50))],
         {"gamma": 0.45}),
        ("dev11_illum_bright", B[6], "illumination", [inst(1.05, 0, (0, 0), (0.45, 0.45))],
         {"gamma": 1.9}),
        ("dev12_illum_gradient", B[7], "illumination", [inst(1.1, 0, (0, 0), (0.62, 0.50))],
         {"gradient": ("left", 0.55)}),
        ("dev13_view_tilt", B[1], "viewpoint", [inst(1.2, 0, (0.16, 0.05), (0.45, 0.50))], {}),
        ("dev14_view_tilt_rot", B[2], "viewpoint", [inst(1.2, 12, (0.22, 0.08), (0.55, 0.50))], {}),
        ("dev15_occ_light", B[3], "occlusion", [inst(1.1, 0, (0, 0), (0.40, 0.50), 0.18)], {}),
        ("dev16_occ_heavy", B[4], "occlusion", [inst(1.1, 0, (0, 0), (0.55, 0.50), 0.42)], {}),
        ("dev17_clutter", B[5], "clutter", [inst(1.0, 0, (0, 0), (0.30, 0.55))],
         {"distractors": 2}),
        ("dev18_noise", B[8], "noise", [inst(1.1, 0, (0, 0), (0.50, 0.45))], {"noise": 14}),
        ("dev19_blur", B[0], "noise", [inst(1.1, 0, (0, 0), (0.45, 0.55))], {"blur": (9, 20)}),
        ("dev20_negative", B[6], "negative", [], {"distractors": 2}),
    ]
    test = [
        ("test01_baseline", B[4], "baseline", [inst(1.05, 0, (0, 0), (0.35, 0.42))], {}),
        ("test02_baseline_two", B[5], "baseline", [inst(1.0, 0, (0, 0), (0.24, 0.60)),
                                                   inst(1.2, 0, (0, 0), (0.70, 0.35))], {}),
        ("test03_scale_small", B[7], "scale", [inst(0.55, 0, (0, 0), (0.60, 0.40))], {}),
        ("test04_scale_large", B[3], "scale", [inst(1.90, 0, (0, 0), (0.45, 0.52))], {}),
        ("test05_rot_small", B[8], "rotation", [inst(1.05, 6, (0, 0), (0.38, 0.48))], {}),
        ("test06_rot_medium", B[1], "rotation", [inst(1.15, 30, (0, 0), (0.55, 0.45))], {}),
        ("test07_rot_ninety", B[2], "rotation", [inst(1.0, 90, (0, 0), (0.45, 0.50))], {}),
        ("test08_illum_dark", B[6], "illumination", [inst(1.1, 0, (0, 0), (0.52, 0.48))],
         {"gamma": 0.40}),
        ("test09_illum_gradient", B[0], "illumination", [inst(1.05, 0, (0, 0), (0.30, 0.50))],
         {"gradient": ("right", 0.60)}),
        ("test10_view_tilt", B[4], "viewpoint", [inst(1.25, 0, (0.20, 0.07), (0.50, 0.50))], {}),
        ("test11_occ_light", B[5], "occlusion", [inst(1.1, 0, (0, 0), (0.42, 0.45), 0.20)], {}),
        ("test12_occ_heavy", B[7], "occlusion", [inst(1.15, 0, (0, 0), (0.58, 0.52), 0.45)], {}),
        ("test13_clutter", B[8], "clutter", [inst(1.05, 0, (0, 0), (0.68, 0.55))],
         {"distractors": 3}),
        ("test14_noise", B[3], "noise", [inst(1.1, 0, (0, 0), (0.47, 0.47))], {"noise": 18}),
        ("test15_combined", B[1], "combined", [inst(0.80, 15, (0.12, 0.04), (0.35, 0.45), 0.22),
                                               inst(1.30, 0, (0, 0), (0.72, 0.58))],
         {"gamma": 0.65, "noise": 8}),
        ("test16_negative", B[2], "negative", [], {"distractors": 2}),
    ]
    plan = []
    for name, bg, cond, insts, photo in dev:
        plan.append((name, bg, "dev", cond, insts, photo))
    for name, bg, cond, insts, photo in test:
        plan.append((name, bg, "test", cond, insts, photo))
    return plan


def main():
    rng = np.random.default_rng(SEED)

    master = trim_alpha(load_rgba(os.path.join(DATA, "template", "ug_logo_master.png")))
    ref = scale_to_width(master, TEMPLATE_W)

    # The stored template is the emblem composited on a white card, which is
    # how it appears on real printed and web material.
    a = ref[:, :, 3:4].astype(np.float32) / 255.0
    tmpl = (ref[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    cv2.imwrite(os.path.join(DATA, "template", "ug_logo_template.png"), tmpl)
    cv2.imwrite(os.path.join(DATA, "template", "ug_logo_mask.png"),
                (ref[:, :, 3] > 16).astype(np.uint8) * 255)
    print("template size %dx%d" % (tmpl.shape[1], tmpl.shape[0]))

    distractors = [trim_alpha(load_rgba(os.path.join(DATA, "distractors", f)))
                   for f in sorted(os.listdir(os.path.join(DATA, "distractors")))]

    ann = {"dev": {}, "test": {}}
    for name, bgfile, split, cond, insts, photo in scene_plan():
        scene = prepare_background(os.path.join(DATA, "backgrounds", bgfile), rng)
        boxes = []

        for extra in range(photo.get("distractors", 0)):
            d = distractors[extra % len(distractors)]
            d = scale_to_width(d, int(rng.integers(70, 130)))
            d = rotate_rgba(d, float(rng.uniform(-15, 15)))
            dx = int(rng.integers(20, SCENE_W - d.shape[1] - 20))
            dy = int(rng.integers(20, SCENE_H - d.shape[0] - 20))
            alpha_paste(scene, d, dx, dy)

        for spec in insts:
            obj = scale_to_width(master, max(16, int(round(TEMPLATE_W * spec["scale"]))))
            obj = perspective_rgba(obj, spec["tilt"][0], spec["tilt"][1])
            obj = rotate_rgba(obj, spec["rot"])
            oh, ow = obj.shape[:2]
            cx = int(spec["pos"][0] * SCENE_W)
            cy = int(spec["pos"][1] * SCENE_H)
            x = int(np.clip(cx - ow // 2, 4, SCENE_W - ow - 4))
            y = int(np.clip(cy - oh // 2, 4, SCENE_H - oh - 4))
            box = alpha_paste(scene, obj, x, y)
            if box is None:
                continue
            record = {"bbox": box, "scale": spec["scale"], "rotation": spec["rot"],
                      "tilt": list(spec["tilt"]), "occlusion": spec["occ"]}
            if spec["occ"] > 0:
                record["occluded_side"] = occlude(scene, box, spec["occ"], rng)
            boxes.append(record)

        if "gamma" in photo:
            scene = apply_gamma(scene, photo["gamma"])
        if "gradient" in photo:
            d, s = photo["gradient"]
            scene = apply_light_gradient(scene, s, d)
        if "blur" in photo:
            scene = motion_blur(scene, photo["blur"][0], photo["blur"][1])
        if "noise" in photo:
            scene = add_noise(scene, photo["noise"], rng)

        folder = "dev_set" if split == "dev" else "test_set"
        out = os.path.join(DATA, folder, name + ".jpg")
        cv2.imwrite(out, scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ann[split][name + ".jpg"] = {
            "condition": cond, "background": bgfile,
            "photometric": {k: v for k, v in photo.items() if k != "distractors"},
            "distractors": photo.get("distractors", 0),
            "objects": boxes,
        }
        print("%-24s %-12s objects=%d" % (name, cond, len(boxes)))

    os.makedirs(os.path.join(DATA, "annotations"), exist_ok=True)
    for split, key in (("dev", "dev_annotations.json"), ("test", "test_annotations.json")):
        with open(os.path.join(DATA, "annotations", key), "w", encoding="utf-8") as fh:
            json.dump(ann[split], fh, indent=2)
    n_dev = sum(len(v["objects"]) for v in ann["dev"].values())
    n_test = sum(len(v["objects"]) for v in ann["test"].values())
    print("dev  : %d images, %d annotated objects" % (len(ann["dev"]), n_dev))
    print("test : %d images, %d annotated objects" % (len(ann["test"]), n_test))


if __name__ == "__main__":
    main()
