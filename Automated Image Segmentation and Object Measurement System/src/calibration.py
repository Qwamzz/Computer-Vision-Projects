"""
Spatial calibration from the printed ruler in the cowpea photograph.

One image of the dataset was photographed next to a millimetre ruler. The tick
marks are found by projecting the ruler band onto the horizontal axis, and the
median distance between neighbouring ticks gives the number of pixels per
millimetre. Every pixel measurement of that image can then be reported in
millimetres, which is what an agricultural grading application needs.
"""

import cv2
import numpy as np

# Vertical band of the working resolution image that contains the ruler, and
# the physical distance between two neighbouring tick marks.
RULER_BAND = (500, 570)
MM_PER_TICK = 1.0


def tick_positions(img_bgr, band=RULER_BAND, ink_threshold=128, min_ink=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    strip = gray[band[0]:band[1], :]
    ink_per_column = (strip < ink_threshold).sum(axis=0)
    is_tick = ink_per_column > min_ink
    centres, start = [], None
    for i, v in enumerate(is_tick):
        if v and start is None:
            start = i
        if not v and start is not None:
            centres.append((start + i - 1) / 2.0)
            start = None
    if start is not None:
        centres.append((start + len(is_tick) - 1) / 2.0)
    return centres


def pixels_per_mm(img_bgr, band=RULER_BAND, mm_per_tick=MM_PER_TICK):
    centres = tick_positions(img_bgr, band)
    if len(centres) < 5:
        raise RuntimeError("Ruler not found, only %d ticks detected" % len(centres))
    spacing = np.diff(np.array(centres))
    # The median is used so that the wider marks at the five millimetre
    # positions cannot bias the estimate.
    return float(np.median(spacing) / mm_per_tick), len(centres)


if __name__ == "__main__":
    from dataset import load_working
    work, rec, _ = load_working("beans_cowpea_scale_05")
    scale, n = pixels_per_mm(work)
    print("ticks detected : %d" % n)
    print("pixels per mm  : %.2f" % scale)
    print("field of view  : %.1f mm wide" % (work.shape[1] / scale))
