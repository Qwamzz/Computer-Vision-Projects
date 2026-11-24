# Computer Vision Projects

End to end computer vision systems built with **classical techniques only**.
None of them uses a learned or pre-trained model. Each stage is implemented from
its definition, compared against the alternatives on annotated data, and written
up in a full report.
 
| Project | Task | Headline result |
|---|---|---|
| [Automated Image Segmentation and Object Measurement](Automated%20Image%20Segmentation%20and%20Object%20Measurement%20System) | Segment, count and measure seeds, beans and coins | K means (Lab, K = 3): **IoU 0.815, F1 0.893** |
| [Classical Object Detection and Localisation](Classical%20Object%20Detection%20and%20Localisation%20System) | Find and box a planar emblem in cluttered real scenes | Combined colour, template and SIFT: **F1 0.875** on a held out set |
| [Feature Based Image Matching and Panorama Construction](Feature%20Based%20Image%20Matching%20and%20Automatic%20Panorama%20Construction) | Match overlapping views and stitch them into a mosaic | SIFT + RANSAC: **0.070 px** mean alignment error |
| [Road and Lane Boundary Detection](Road%20and%20Lane%20Boundary%20Detection) | Find the drivable road surface and its edges | Mahalanobis segmentation: **F1 0.733** on held out images |

## Projects

### 1. Automated Image Segmentation and Object Measurement System

<img src="Automated%20Image%20Segmentation%20and%20Object%20Measurement%20System/results/figures/main_coins_greek_07.png" width="600" alt="Segmentation and measurement of coins">

I compare three segmentation methods, all written from scratch: Otsu
thresholding, region growing, and K means with k-means++ initialisation. The
system then extracts each object and reports its area, perimeter, centroid, axes,
orientation, circularity, solidity and extent. For the photograph with a printed
ruler, it also gives these in millimetres. A robustness study checks behaviour
under lighting gradients, under exposure, noise and blur.

**Techniques:** Otsu, region growing, K means, morphology, watershed, illumination correction, camera calibration

[README](Automated%20Image%20Segmentation%20and%20Object%20Measurement%20System/README.md) ·
[Report (PDF)](Automated%20Image%20Segmentation%20and%20Object%20Measurement%20System/report/Automated_Image_Segmentation_and_Object_Measurement.pdf)

### 2. Classical Object Detection and Localisation System

<img src="Classical%20Object%20Detection%20and%20Localisation%20System/results/figures/fig10_success.png" width="600" alt="Detections of the emblem in test scenes">

The system detects the University of Ghana coat of arms in real photographs. I
compare six strategies: masked template matching, a scale pyramid, a rotated
template bank, an explicit sliding window, SIFT with RANSAC, and colour based
region proposals. Ground truth boxes are exact to the pixel because every
placement is known. Tuning used 20 development scenes. The 16 test scenes were
run once, after every parameter was fixed.

**Techniques:** normalised cross correlation, image pyramids, non maximum suppression, SIFT, RANSAC homography, HSV segmentation

[README](Classical%20Object%20Detection%20and%20Localisation%20System/README.md) ·
[Report (PDF)](Classical%20Object%20Detection%20and%20Localisation%20System/report/Classical_Object_Detection_and_Localisation.pdf)

### 3. Feature Based Image Matching and Automatic Panorama Construction

<img src="Feature%20Based%20Image%20Matching%20and%20Automatic%20Panorama%20Construction/results/panorama_SIFT.png" width="600" alt="Stitched panorama of Elmina harbour">

This is a full stitching pipeline with no high level stitching API. I wrote the
ratio test matcher, the adaptive RANSAC loop, the normalised DLT homography
solve, the warping and the feathered blending myself. 27 tests check each stage
against OpenCV. The views come from a real photograph through known
homographies, so alignment error can be measured in pixels. The pipeline also
runs on your own overlapping photos placed in `images/`.

**Techniques:** SIFT, ORB, Shi-Tomasi, Lowe ratio test, RANSAC, DLT homography, image warping, feathered blending

[README](Feature%20Based%20Image%20Matching%20and%20Automatic%20Panorama%20Construction/README.md) ·
[Report (PDF)](Feature%20Based%20Image%20Matching%20and%20Automatic%20Panorama%20Construction/report/Automated_Panorama_Construction.pdf)

### 4. Road and Lane Boundary Detection

<img src="Road%20and%20Lane%20Boundary%20Detection/results/figures/fig03_pipeline.png" width="600" alt="Road detection pipeline stages">

This project finds the drivable road surface in twelve real photographs that
cover six conditions: daylight, worn markings, curves, shadows, occluding
traffic and poor lighting. Each stage is chosen by measurement. That covers the
colour space, filter, edge detector, region of interest, segmentation rule and
boundary strategy. One main finding: reading boundaries off the segmented region
gives 76 px of error, while the textbook Hough line fit gives 356 px.

**Techniques:** colour space analysis, Canny, Sobel, Hough transform, Mahalanobis thresholding, K means, region growing

[README](Road%20and%20Lane%20Boundary%20Detection/README.md) ·
[Report (PDF)](Road%20and%20Lane%20Boundary%20Detection/report/Road_and_Lane_Boundary_Detection.pdf)

## Common approach

- **Classical methods only.** No neural networks and no pre-trained weights.
- **Measured, not assumed.** I compare each design choice on annotated data with
  IoU, precision, recall, F1 or pixel error.
- **Honest evaluation.** Where it applies, parameters are tuned on one split and
  reported on a held out split that was only run once.
- **Real data.** Images are real photographs from Wikimedia Commons or
  scikit-image, under free licences. Each project lists its sources and none of
  the images were machine generated.
- **Reproducible.** The scripts regenerate every table and figure, and each
  report is built from those results so the numbers always match.

## Repository layout

Each project is self contained and follows the same structure:

```
<Project>/
├── README.md          project details and full results
├── requirements.txt
├── report/            full write up as a PDF
├── data/              images, annotations and source licences
├── results/           figures, CSV tables and summary JSON
└── src/               the implementation and experiment scripts
```

## Getting started

Each project has its own dependencies (mostly `numpy`, `opencv-python`,
`matplotlib` and `reportlab`). Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Qwamzz/Computer-Vision-Projects.git
cd "Computer-Vision-Projects/Road and Lane Boundary Detection"
pip install -r requirements.txt
python src/run_experiments.py
```

See each project's README for the exact steps and run times.
