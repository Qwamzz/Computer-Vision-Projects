# Classical Object Detection and Localisation System

Detecting and localising a rigid planar emblem, the University of Ghana coat of
arms, in real photographs using classical computer vision only. No pre-trained
recognition network is used at any point.

The full write up is **`report/Classical_Object_Detection_and_Localisation.pdf`**.

## What the system does

```
input image -> search strategy -> template comparison -> detection decision
            -> object localisation -> performance evaluation
```

Six detection strategies are implemented from the classical toolbox and
measured against each other on the same annotated data:

| Strategy | Idea |
|---|---|
| Masked template matching | Zero mean normalised cross correlation restricted to the emblem mask |
| Scale pyramid | The same correlation repeated over an eleven level geometric pyramid |
| Rotated template bank | Eight rotated copies of every template, to buy rotation invariance by enumeration |
| Explicit sliding window | Textbook window loop with configurable window size, step size and threshold |
| SIFT with RANSAC | Keypoints, Lowe ratio test, homography fitted under RANSAC, repeated for multiple instances |
| Colour proposals | HSV segmentation of the navy shield used to cut the search space before correlation |

Supporting machinery: local peak extraction by grey scale dilation, greedy non
maximum suppression, and greedy intersection over union matching against ground
truth for scoring.

## Results

Held out test set of 16 images and 17 object instances, never used for tuning,
at an intersection over union of 0.5.

| Method | Precision | Recall | F1 | s/image |
|---|---|---|---|---|
| Single scale template matching | 1.000 | 0.294 | 0.455 | 0.09 |
| Multi scale template matching | 1.000 | 0.412 | 0.583 | 1.16 |
| Multi scale and rotation bank | 0.750 | 0.529 | 0.621 | 9.96 |
| Explicit sliding window (step 4) | 1.000 | 0.118 | 0.211 | 8.92 |
| SIFT matching with RANSAC | 0.800 | 0.706 | 0.750 | 0.14 |
| Colour proposals and template matching | 1.000 | 0.471 | 0.640 | 0.39 |
| **Combined colour, template and features** | **0.933** | **0.824** | **0.875** | 0.52 |

Every table in `results/` is regenerated end to end by `run_experiments.py`, and
the report is built directly from those CSV files, so the document cannot drift
away from the measurements.

What the experiments show:

- The decision threshold moves the detector from 2004 false detections to zero
  across a span of about 0.05 in correlation, so it has to be selected on data.
- Masking the non emblem corners of the template rectangle lifts F1 from 0.43 to
  0.69, almost entirely through recall.
- Invariance bought by enumeration is priced very differently: scale costs a
  linear factor and is worth paying, rotation costs an order of magnitude and
  does not improve overall F1, and viewpoint cannot be enumerated at all.
- Sliding window cost falls as the square of the step size, from 36.2 to 0.27
  seconds per image between step 2 and step 24, while F1 collapses from 1.00 to
  0.33 once the step passes roughly one twentieth of the window.
- Colour proposals cut the searched area to 29.8 per cent of the image, which
  allows a lower threshold at unchanged precision and raises F1 from 0.69 to
  0.76 on the development set.
- Correlation and feature matching fail on disjoint conditions, which is why
  combining them beats either alone. No method handled the viewpoint tilt.

## Dataset

Every source image is a real photograph or a real vector emblem downloaded from
Wikimedia Commons under a free licence. Nothing was produced by an image
generation model. `data/SOURCES.md` lists the licence and the author of each
file.

The emblem is the public domain University of Ghana coat of arms. Nine real
photographs supply the backgrounds: campus buildings, the Balme Library, Mensah
Sarbah Hall, Accra street scenes, an office desk, a brick wall and two interior
scenes. Two other university emblems act as clutter and as hard negatives.

Each scene is built by transforming the emblem and compositing it into a real
photograph under controlled conditions: position, scale from 0.55 to 1.90,
rotation from 6 to 90 degrees, gamma from 0.40 to 1.9, one sided lighting ramps,
projective tilt, occlusion from 18 to 45 per cent by a textured patch copied
from the same photograph, clutter, Gaussian noise and motion blur. Because the
placement geometry is known, the bounding box ground truth is exact to the pixel
rather than hand drawn. Two scenes contain no target at all, so false positive
behaviour is measured directly.

The 36 scenes are split into 20 development images used for tuning and 16 held
out images used once, with every parameter already fixed.

## Folder layout

```
Classical Object Detection and Localisation System/
├── README.md
├── requirements.txt
├── report/
│   └── Classical_Object_Detection_and_Localisation.pdf
├── data/
│   ├── template/         the emblem template and its correlation mask
│   ├── backgrounds/      nine real photographs used as scene backgrounds
│   ├── distractors/      similar emblems used as clutter and hard negatives
│   ├── dev_set/          20 development scenes
│   ├── test_set/         16 held out scenes
│   ├── annotations/      bounding box ground truth as JSON
│   └── SOURCES.md        licence and author of every downloaded image
├── results/
│   ├── figures/          every figure used in the report
│   ├── tables/           every quantitative table as CSV
│   ├── detections/       final detections drawn on each test image
│   └── summary.json      machine readable copy of the results
└── src/
    ├── download_data.py  fetches the source imagery from Wikimedia Commons
    ├── build_dataset.py  builds the template and the annotated scenes
    ├── detector.py       all detection algorithms
    ├── evaluate.py       matching, precision, recall and F1
    ├── run_experiments.py runs every experiment and writes results/
    └── make_report.py    builds the report PDF from the measured results
```

## Reproducing

```bash
pip install -r requirements.txt
python src/download_data.py      # needs an internet connection, run once
python src/build_dataset.py
python src/run_experiments.py
python src/make_report.py
```

`run_experiments.py` takes roughly half an hour on one CPU core. The dataset is
generated from a fixed random seed, so it is identical on every run.
