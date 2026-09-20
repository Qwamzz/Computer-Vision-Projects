# Automated Image Segmentation and Object Measurement System

Segmenting objects of interest from a photograph and computing meaningful
measurements about them, using classical computer vision only. No learned or
pre-trained model is used anywhere in the project.

Application domain: counting and measuring seeds, beans and coins.

The full write up is
**`report/Automated_Image_Segmentation_and_Object_Measurement.pdf`**.

## What the system does

```
input image -> image preparation -> segmentation -> boundary identification
            -> object extraction -> object measurement -> performance evaluation
```

Three segmentation techniques are implemented from first principles and
compared: threshold based segmentation with an Otsu implementation written from
the definition, region growing with an explicit seed set, similarity predicate
and queue, and K means clustering implemented as Lloyd iterations with a k
means plus plus initialisation.

For every segmented object the system reports the count, area, perimeter,
centroid, bounding box, equivalent diameter, major and minor axis, orientation,
circularity, solidity and extent. One photograph was taken next to a printed
ruler, so its measurements are also reported in millimetres.

## Headline results

Mean scores over the five annotated photographs, using one common parameter set:

| Method | Mean IoU | Mean F1 | Mean time per image |
|---|---|---|---|
| Thresholding (Otsu) | 0.710 | 0.785 | 0.03 s |
| Region growing (tolerance 25) | 0.365 | 0.512 | 3.14 s |
| **K means (K = 3, Lab colour)** | **0.815** | **0.893** | 7.12 s |

The exact numbers are in `results/tables/` and are discussed in sections 5 and 6
of the report.

## Dataset

All eight base images are real photographs. Seven come from Wikimedia Commons
under Creative Commons licences and one is the coins photograph distributed as
sample data with scikit-image. None of them was produced by an image generation
model. The source URL and licence of every image are recorded in `src/config.py`
and reproduced in Table 1 and Table 15 of the report.

The six variant images are controlled degradations computed from two of the
photographs: a lighting gradient, an under exposure, additive Gaussian noise and
a defocus blur. They are used for the robustness study.

## Folder layout

```
Automated Image Segmentation and Object Measurement System/
├── README.md
├── requirements.txt
├── report/
│   └── Automated_Image_Segmentation_and_Object_Measurement.pdf
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
    ├── report_content.py            the report text, written once as blocks
    ├── make_report.py               renders the blocks into the PDF
    └── make_report_docx.py          renders the same blocks into Word
```

## Requirements

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Reproducing everything

```bash
cd src
python dataset.py            # fetch the photographs if missing, build the variants
python ground_truth.py       # rebuild the annotation masks
python run_experiments.py    # all experiments, tables and figures
python make_report.py        # rebuild the PDF report
python make_report_docx.py   # optional, the same report as a Word document
```

The full experiment run takes roughly six minutes on an ordinary laptop.
Individual modules can also be run on their own, for example
`python calibration.py` prints the pixels per millimetre measured from the ruler
in the cowpea photograph.

## One source, two output formats

The report is written once, in `src/report_content.py`, as a sequence of neutral
content blocks. `make_report.py` renders those blocks into the PDF and
`make_report_docx.py` renders the same blocks into a Word document, so the two
files always carry the same text, the same tables and the same figures. The Word
file uses real headings, so the navigation pane and any table of contents Word
generates will work.
