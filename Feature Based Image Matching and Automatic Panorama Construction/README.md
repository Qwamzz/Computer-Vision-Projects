# Feature Based Image Matching and Automatic Panorama Construction

Finding corresponding regions between overlapping photographs and stitching them
into a single mosaic, using classical computer vision only. Every stage is
implemented from the definition rather than called from a high level stitching
API, and each one is validated against the OpenCV equivalent.

The full write up is **`report/Automated_Panorama_Construction.pdf`**.

## The pipeline

```
feature detection -> feature description -> feature matching -> RANSAC
                  -> homography -> image alignment -> panorama
```

Implemented from first principles: the ratio test matcher, the RANSAC loop with
an adaptive iteration count, the normalised DLT homography solve, the warping and
the feathered blending. `src/test_components.py` checks each stage against
OpenCV, and all 27 tests pass.

## Results

Three detector and descriptor configurations, averaged over the neighbouring
view pairs:

| Configuration | Keypoints | Matches | Inliers | Inlier ratio | Alignment error | Time |
|---|---|---|---|---|---|---|
| **SIFT** | 4000 | 1103 | 1090 | 0.987 | **0.070 px** | 569 ms |
| ORB | 4000 | 551 | 527 | 0.957 | 0.645 px | 284 ms |
| Shi-Tomasi + SIFT | 3930 | 995 | 977 | 0.982 | 0.145 px | 100 ms |

Alignment error is the mean distance, in pixels, between where the estimated
homography sends a grid of points and where the true homography sends them. It
can be quoted in pixels because the ground truth is known exactly, which is the
whole point of the dataset design below.

The effect of RANSAC on the same pairs:

| Detector | Putative matches | Removed | Error before | Error after |
|---|---|---|---|---|
| SIFT | 1138 | 11 | 287.7 px | 0.074 px |
| ORB | 531 | 27 | 2706.9 px | 0.288 px |
| Shi-Tomasi + SIFT | 1183 | 24 | 3461.2 px | 0.124 px |

Discarding roughly 1 to 5 per cent of the matches changes the alignment error by
three to four orders of magnitude. A least squares fit over all putative matches
is destroyed by a handful of outliers, because squared error lets one gross
mismatch outvote a thousand good ones.

The report also covers robustness to rotation, scale, viewpoint and illumination
varied one at a time, sensitivity to the Lowe ratio and the RANSAC threshold, and
a comparison of overwrite against feathered blending.

## Dataset design

The scene is a **real photograph** of the fishing harbour at Elmina, downloaded
from Wikimedia Commons under CC BY-SA 4.0. Nothing here is machine generated
imagery, and `data/SOURCES.md` records the licence and author.

The overlapping views are cut out of that photograph by warping it through
homographies that are *chosen rather than estimated*, each with its own rotation,
scale, perspective tilt and exposure. This is what lets the project have both
things at once:

- the detectors work on genuine photographic texture, film grain and real
  lighting, including a wide band of sky where there is nothing to find;
- the homography relating any two views is known to machine precision, so
  alignment error is measurable in pixels, matching precision can be checked
  against the truth instead of assuming RANSAC was right, and rotation, scale,
  viewpoint and illumination can be varied one at a time.

Section 11.2 of the report is explicit about what this still does not capture:
because every view comes from one exposure there is no parallax, no per shot lens
distortion and nothing that moved between frames, so the inlier ratios are higher
than separately captured photographs would give.

To run on your own photographs instead, drop three or more overlapping images
into `images/`. The system switches to real mode automatically; the ground truth
columns then become unavailable, as they would for any real dataset.

## Folder layout

```
Feature Based Image Matching and Automatic Panorama Construction/
├── README.md
├── requirements.txt
├── report/
│   └── Automated_Panorama_Construction.pdf
├── data/
│   ├── scene/harbour_scene.jpg   the real photograph used as the planar scene
│   └── SOURCES.md                its licence and author
├── images/                       drop your own overlapping photos here
├── results/
│   ├── RESULTS.md                every experimental table, written by the run
│   ├── results.json              the same numbers in machine readable form
│   ├── inputs/                   the input views and their ground truth
│   ├── panorama_<DETECTOR>.png   final mosaic from each configuration
│   ├── matches_<DETECTOR>.png    correspondences before and after RANSAC
│   ├── keypoints_<DETECTOR>.png  detected keypoints with scale and orientation
│   ├── alignment_*.png           checkerboard and difference overlays
│   └── plot_*.png                robustness and parameter sensitivity plots
└── src/
    ├── download_scene.py         fetches the scene photograph from Commons
    ├── dataset.py                acquisition, view synthesis, preprocessing
    ├── features.py               detectors and descriptors
    ├── matching.py               ratio test matching
    ├── homography.py             normalised DLT and the RANSAC loop
    ├── stitching.py              warping, canvas layout, blending
    ├── pipeline.py               the end to end pipeline
    ├── experiments.py            the five experiments
    ├── visualisation.py          all figures
    ├── test_components.py        27 validation tests against OpenCV
    ├── main.py                   the driver
    └── build_report.py           renders the report from results.json
```

## Reproducing

```bash
pip install -r requirements.txt
python src/download_scene.py    # fetch the scene photograph, run once
python src/main.py              # full run, all experiments
python src/test_components.py   # 27 validation tests
python src/build_report.py      # rebuild the PDF from results.json
```

A full run takes roughly twenty minutes on one CPU core; `--quick` skips the
parameter sweeps. `--scene` ignores `images/` and uses the scene photograph,
and `--drawn-scene` forces a procedurally drawn fallback scene so the project
still runs with no network access.

Every table in the report is rendered directly from `results/results.json`, so
the document cannot drift out of step with the measurements.
