# CSCD608 Advanced Computer Vision, Final, Question 4

**Automated Image Segmentation and Object Measurement System**

| | |
|---|---|

| Application domain | Counting and measuring seeds, beans and coins |

The report is **`report/CSCD608_Q4_Report.pdf`**.

## What the system does

Input image -> image preparation -> segmentation -> boundary identification ->
object extraction -> object measurement -> performance evaluation

Three segmentation techniques are implemented from first principles and
compared: threshold based segmentation with an Otsu implementation written from
the definition, region growing with an explicit seed set, similarity predicate
and queue, and K means clustering implemented as Lloyd iterations with a k
means plus plus initialisation. No learned or pre-trained model is used
anywhere in the project.

For every segmented object the system reports the count, area, perimeter,
centroid, bounding box, equivalent diameter, major and minor axis, orientation,
circularity, solidity and extent. One photograph was taken next to a printed
ruler, so its measurements are also reported in millimetres.

## Folder layout

```
Q4_Segmentation_Measurement_22424650/
├── README.md
├── requirements.txt
├── report/
│   └── CSCD608_Q4_Report.pdf        the full report
├── data/
│   ├── images/                      8 real photographs plus 6 controlled variants
│   └── ground_truth/                manually verified annotation masks
├── results/
│   ├── figures/                     every figure used in the report
│   ├── tables/                      every quantitative table as CSV
│   ├── measurements/                per object measurements as CSV
│   └── summary.json                 machine readable copy of the results
└── src/
    ├── config.py                    paths, working resolution, dataset manifest
    ├── dataset.py                   image retrieval and controlled degradations
    ├── preprocessing.py             colour, filtering, illumination correction
    ├── segmentation.py              Otsu, region growing, K means, morphology, watershed
    ├── measurement.py               boundaries, object extraction, measurements
    ├── evaluation.py                IoU, precision, recall, F1, object matching
    ├── calibration.py               pixels per millimetre from the printed ruler
    ├── ground_truth.py              assisted annotation and the manual edit list
    ├── visualisation.py             all figures
    ├── run_experiments.py           the experiment driver
    └── make_report.py               builds the PDF report
```

## Requirements

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## How to reproduce everything

```bash
cd src
python dataset.py            # fetch the photographs if missing, build the variants
python ground_truth.py       # rebuild the annotation masks
python run_experiments.py    # all experiments, tables and figures
python make_report.py        # rebuild the PDF report
```

The full experiment run takes roughly six minutes on an ordinary laptop.
Individual modules can also be run on their own, for example
`python calibration.py` prints the pixels per millimetre measured from the
ruler in the cowpea photograph.

## Dataset

All eight base images are real photographs. Seven come from Wikimedia Commons
under Creative Commons licences and one is the coins photograph distributed as
sample data with scikit-image. None of them was produced by an image generation
model. The source URL and licence of every image are recorded in
`src/config.py` and reproduced in Table 1 and Table 15 of the report.

The six variant images are controlled degradations computed from two of the
photographs, a lighting gradient, an under exposure, additive Gaussian noise
and a defocus blur. They are used for the robustness study.

## Headline results

Mean scores over the five annotated photographs, using one common parameter set:

| method | mean IoU | mean F1 | mean time per image |
|---|---|---|---|
| Thresholding (Otsu) | 0.710 | 0.785 | 0.015 s |
| Region growing (tolerance 25) | 0.365 | 0.512 | 1.37 s |
| K means (K = 3, Lab colour) | 0.815 | 0.893 | 4.05 s |

The exact numbers are in `results/tables/` and are discussed in sections 5 and
6 of the report.
