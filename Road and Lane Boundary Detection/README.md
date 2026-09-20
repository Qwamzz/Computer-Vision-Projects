# Road and Lane Boundary Detection

Finding the drivable road surface and its boundaries in ordinary road
photographs, using classical computer vision only. No learned model is used
anywhere in the pipeline.

The full write up is **`report/Road_and_Lane_Boundary_Detection.pdf`**.

## What the system does

```
road image -> filtering -> edge detection -> region selection
           -> segmentation -> boundary identification -> visualisation
```

Every stage offers a real choice, and each one is settled by measurement:

| Stage | Options compared |
|---|---|
| Colour representation | BGR, HLS, Lab, HSV, scored by Fisher separability against the ground truth |
| Filtering | none, Gaussian, median, bilateral |
| Edge detection | Canny and Sobel, across four thresholds each |
| Region selection | whole frame, fixed trapezoid, adaptive region from a first pass segmentation |
| Segmentation | adaptive threshold, Mahalanobis threshold, K means, region growing |
| Boundary identification | Hough on ROI edges, Hough gated by a road paint mask, boundaries read from the segmented region |

## Results

Twelve real photographs, six imaging conditions, two images each. The six
images ending `_01` are the tuning set; the six ending `_02` were held out and
run once with every parameter already fixed.

| Split | IoU | Precision | Recall | F1 | Boundary error |
|---|---|---|---|---|---|
| Tuning | 0.710 | 0.806 | 0.865 | 0.828 | 70 px |
| **Held out** | **0.597** | **0.761** | **0.772** | **0.733** | **137 px** |

Boundary error is the mean horizontal distance from the true road edge on
images 1280 px wide. The drop across the split is real: with six images to tune
on, some of the apparent accuracy was fitted to those six photographs, and the
held out figure is the one to believe.

By condition, over all twelve images:

| Condition | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Good daylight | 0.778 | 0.856 | 0.898 | 0.875 |
| Worn or absent markings | 0.733 | 0.855 | 0.844 | 0.844 |
| Curved road | 0.723 | 0.752 | 0.942 | 0.835 |
| Shadows | 0.673 | 0.770 | 0.847 | 0.803 |
| Occluding traffic | 0.548 | 0.926 | 0.567 | 0.697 |
| Poor illumination | 0.467 | 0.542 | 0.811 | 0.629 |

### Three findings worth reporting

**The textbook lane finder loses, and badly.** Fitting straight lines to Hough
segments gives a mean boundary error of 356 px. Gating those edges with a white
and yellow paint mask does not rescue it (348 px) and finds fewer boundaries,
because half the dataset has no usable paint. Reading the boundary off the
segmented region instead gives **76 px**. The reason is measurable upstream:
boundary precision for the edge operators never exceeds about 0.09, because
almost every edge inside the region of interest is a real edge of something that
is not the road boundary.

| Boundary strategy | Found | Mean error |
|---|---|---|
| Hough on ROI edges | 9/12 | 356 px |
| Hough on paint edges | 7/12 | 348 px |
| **From segmented region** | **9/12** | **76 px** |

**Putting the channel correlation back into a threshold was the single largest
gain.** A per channel acceptance band in HLS reaches an IoU of 0.572; the same
idea using the full covariance of the colour sample, so the acceptance region is
an oriented ellipsoid rather than an axis aligned box, reaches **0.710**.

| Segmentation | Best IoU | Precision | Recall |
|---|---|---|---|
| **Mahalanobis threshold (Lab)** | **0.710** | 0.806 | 0.865 |
| Region growing (tol 28) | 0.593 | 0.824 | 0.732 |
| Adaptive threshold (HLS) | 0.572 | 0.737 | 0.744 |
| K means (K = 3) | 0.493 | 0.788 | 0.672 |

**The selection metric matters as much as the stage.** Scoring filters by
boundary recall divided by edge density selects Gaussian smoothing, which
destroys about nine tenths of the road boundary (recall 0.034 against 0.365 for
no filtering) purely because its denominator collapses faster than its
numerator. The same trap catches the edge operator comparison. Boundary F1 is
used instead, and the report documents the mistake rather than hiding it.

## Dataset and ground truth

Twelve real photographs of real roads from Wikimedia Commons, all under free
licences, listed with author and licence in `data/SOURCES.md`. None was produced
by an image generation model. They were taken by different people with different
cameras for unrelated purposes, which is a harder and more honest test than a
dashcam sequence: no geometry is shared between images, so no hidden constant can
be tuned.

The drivable surface of every image was traced by hand as a polygon, checked by
drawing the mask back over the photograph, and corrected until it was right. The
vertices live in `src/annotate.py`, so the annotation is readable and
reproducible rather than hidden in a binary mask. Two rules: the region is the
sealed carriageway bounded by the painted edge lines where they exist, and
anything standing on it (a vehicle, an animal) is cut out, so the ground truth is
the *visible* drivable surface.

## Folder layout

```
Road and Lane Boundary Detection/
├── README.md
├── requirements.txt
├── report/
│   └── Road_and_Lane_Boundary_Detection.pdf
├── data/
│   ├── images/          12 road photographs at 1280 px wide
│   ├── annotations/     rasterised ground truth masks
│   └── SOURCES.md       licence and author of every image
├── results/
│   ├── figures/         every figure used in the report
│   ├── tables/          every quantitative table as CSV
│   ├── overlays/        ground truth and final output for each image
│   ├── pipeline_config.json
│   └── summary.json
└── src/
    ├── download_data.py    fetches the photographs from Wikimedia Commons
    ├── annotate.py         the hand traced polygons and the mask builder
    ├── preprocessing.py    colour representations, filters, separability
    ├── edges.py            Canny, Sobel, the ROI, boundary recall and precision
    ├── lanes.py            Hough, slope splitting, paint mask, region boundaries
    ├── segmentation.py     the four segmentation rules and the overlay
    ├── evaluate.py         IoU, precision, recall, F1
    ├── run_experiments.py  runs the seven experiments and writes results/
    └── make_report.py      builds the report PDF from the measured results
```

## Reproducing

```bash
pip install -r requirements.txt
python src/download_data.py     # needs an internet connection, run once
python src/annotate.py          # rebuild the ground truth masks
python src/run_experiments.py   # all seven experiments, tables and figures
python src/make_report.py       # rebuild the PDF from the measured results
```

A full run takes about four minutes on one CPU core. Every number in the report
is read from the CSV files in `results/tables/`, so the document cannot drift
away from the measurements.
