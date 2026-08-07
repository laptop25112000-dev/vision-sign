# Visual Voice

This project is starting with Indian Sign Language alphabet recognition using MediaPipe landmarks and a TensorFlow or scikit-learn classifier. Text-to-speech can be added later once the text prediction path is stable.

## Current Dataset

The local dataset is in `DTASET A-Z`. It contains A-Z image folders from a Kaggle-style ISL alphabet dataset.

Do not train by blindly reading every file in each folder. Use the generated manifest:

```bash
python tools/audit_dataset.py
```

Training input:

- `manifests/training_manifest.csv`: clean 26 x 1,200 image list
- `manifests/quarantine_manifest.csv`: extra files ignored from training
- `manifests/dataset_audit.json`: repeatable dataset audit summary

## MediaPipe Landmark Extraction

After installing dependencies:

```bash
pip install -r requirements.txt
python src/extract_landmarks.py
```

By default, the extractor now processes only `A` and `B`:

```bash
python src/extract_landmarks.py
```

To choose classes:

```bash
python src/extract_landmarks.py --classes A,B
```

If classes are split across folders:

```bash
python src/extract_landmarks.py --classes A,B,C,D --dataset-dirs "DTASET A-Z,continuation dataset" --output data/landmarks_abcd.npz --failures-out manifests/landmark_failures_abcd.csv
```

To process every class:

```bash
python src/extract_landmarks.py --classes ALL
```

For a quick timing test:

```bash
python src/extract_landmarks.py --classes A,B --limit 1000
```

The extractor upscales the 128 x 128 images to 512 px on the largest side before MediaPipe detection. It now accepts one-hand and two-hand signs by default and pads the missing second hand with zeros, keeping every sample at 128 features. Use `--required-hands 2` only when you intentionally want to reject one-hand signs.

Output:

- `data/landmarks_az.npz`: fixed-size landmark features and labels
- `manifests/landmark_failures.csv`: images where MediaPipe did not detect hands

## Train A/B Classifier

```bash
python src/train_landmark_classifier.py
```

The trainer balances A and B automatically before training. Output:

- `models/isl_ab_svm.joblib`: saved sklearn model
- `models/isl_ab_metrics.json`: accuracy, report, and confusion matrix

## Test With Webcam

```bash
python src/webcam_predict.py
```

Show both hands to the camera and press `q` to quit. The live predictor uses confidence thresholding and a short smoothing window before showing a stable letter.

To speak stable letters:

```bash
python src/webcam_predict.py --model models/isl_az_live_svm.joblib --speak
```

## Browser Website

```bash
python src/web_app.py --model models/isl_az_live_svm.joblib
```

Open `http://127.0.0.1:7860`. The website uses the browser webcam, sends frames to the Python backend for prediction, builds a sentence from stable letters, and can speak the sentence using browser speech synthesis.

Low-confidence frames are shown as `Not detectable` by default when confidence is below `0.60`.

The live detector searches for up to four hands but only uses the two largest hands by landmark bounding-box area. This helps ignore smaller background hands and focus on the nearest person.

## Collect Webcam Samples

```bash
python src/collect_webcam_landmarks.py --classes A,B,C,D
```

Press number keys to choose the current label, for example `1=Q`, `2=R`, `3=S`, `4=T` for a Q-R-S-T run. Press `Enter` to arm auto-capture, `Space` to save one detected sample, `S` to save the file, and `Esc` to quit. Auto-capture waits until the required hand count is detected, then starts saving automatically. The collector defaults to two hands; use `--required-hands 1` only for one-hand signs. When one label reaches its target, auto-capture stops, switches to the next label, and waits for you to press `Enter` again.

Train with both dataset and webcam samples:

```bash
python src/train_landmark_classifier.py --classes A,B,C,D --landmarks "data/landmarks_abcd.npz,data/webcam_landmarks_abcd.npz" --model-out models/isl_abcd_live_svm.joblib --metrics-out models/isl_abcd_live_metrics.json
```

The saved model also stores class-shape distance thresholds. During webcam prediction, samples that do not look close enough to the predicted class are shown as `Uncertain` instead of forcing a wrong letter.

## Analyze Landmark Data

```bash
python src/analyze_landmarks.py --landmarks data/landmarks_abcd.npz --classes A,B,C,D
```

This prints class counts, one-hand/two-hand presence, centroid distances, and likely outlier samples.

See `docs/dataset_verification.md` for the web verification notes and source links.
