# Dataset Verification Notes

Checked on 2026-05-25.

## Verdict

The downloaded A-Z folder appears to be the same public Kaggle-style Indian Sign Language alphabet dataset that is commonly used in ISL recognition projects. It is usable for a class project, but it should be treated as a community/open dataset, not as an official ISLRTC-certified ground-truth corpus.

For training, use `manifests/training_manifest.csv` instead of walking the raw folders directly.

## Web Check

- Kaggle dataset referenced in papers: `https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl`
- TechScience IASC 2023 paper reports the Kaggle source as 26 alphabet classes, 31,200 RGB images, 1,200 per class, 128 x 128 pixels: `https://www.techscience.com/iasc/v37n3/54116/html`
- A related GitHub ISL recognition project describes the older/full dataset as A-Z plus 1-9, 35 classes, 1,200 images per class: `https://github.com/Karthikeyu/Indian-sign-language-recognition`
- Official ISL reference source: ISLRTC maintains the national ISL dictionary and dataset links, so it is the better authority for sign correctness checks: `https://islrtc.nic.in/isl-dictionary/`

## Local Audit

- Expected alphabet subset: 26 folders, `A` through `Z`
- Expected count: 1,200 images per class
- Current canonical total: 31,200 images
- Current quarantine total: 0 images
- Image size spot-check: 128 x 128 JPG
- Raw folder typo retained: `DTASET A-Z`

Originally removed extra files:

- `C`: 247 extras
- `I`: 179 extras
- `O`: 229 extras
- `V`: 90 extras

## MediaPipe Warning

The images are small for MediaPipe. The extractor therefore upscales each 128 x 128 image to 512 px on the largest side before detection, but upscaling cannot recover detail that is not present in the original image.

The extractor accepts one-hand and two-hand signs by default and pads the missing second hand with zeros. Several alphabet images appear to show one visible hand, so using a strict two-hand rule would remove whole classes such as one-hand C/D-style signs.
