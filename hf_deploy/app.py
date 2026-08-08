#!/usr/bin/env python3
"""Visual Voice — ISL Translator on Hugging Face Spaces.

Gradio interface for real-time Indian Sign Language A-Z recognition
using MediaPipe hand landmarks and an SVM classifier.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import urllib.request
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np

# ── MediaPipe setup ─────────────────────────────────────────────────

import mediapipe as mp

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

if not HAND_MODEL_PATH.exists():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    print(f"Downloading hand_landmarker.task...", flush=True)
    urllib.request.urlretrieve(url, str(HAND_MODEL_PATH))
    print("Download complete.", flush=True)


class PointWrapper:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class HandWrapper:
    def __init__(self, landmark):
        self.landmark = landmark


def create_detector():
    """Create a MediaPipe hand landmarker (IMAGE mode for Gradio)."""
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


detector = create_detector()


def detect_hands(rgb_image):
    """Run MediaPipe on an RGB numpy array, return list of HandWrapper."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = detector.detect(mp_image)
    hands = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            landmark_list = [PointWrapper(pt.x, pt.y, pt.z) for pt in hand]
            hands.append(HandWrapper(landmark_list))
    return hands


# ── Feature engineering (same as original) ──────────────────────────

def normalized_hand_vector(hand_landmarks):
    landmarks = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32
    )
    landmarks = landmarks - landmarks[0]
    scale = float(np.max(np.linalg.norm(landmarks, axis=1)))
    if scale > 0:
        landmarks = landmarks / scale
    return landmarks.reshape(-1)


def hand_box_area(hand_landmarks):
    xs = [p.x for p in hand_landmarks.landmark]
    ys = [p.y for p in hand_landmarks.landmark]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def select_primary_hands(hand_landmarks, max_hands=2, min_area=0.015):
    hands = [h for h in (hand_landmarks or []) if hand_box_area(h) >= min_area]
    return sorted(hands, key=hand_box_area, reverse=True)[:max_hands]


def hands_to_feature(hand_landmarks, max_hands=2):
    ordered = sorted(
        hand_landmarks or [],
        key=lambda h: sum(p.x for p in h.landmark) / len(h.landmark),
    )
    parts = []
    for hand in ordered[:max_hands]:
        parts.append(np.array([1.0], dtype=np.float32))
        parts.append(normalized_hand_vector(hand))
    for _ in range(max_hands - len(ordered[:max_hands])):
        parts.append(np.zeros(1 + 21 * 3, dtype=np.float32))
    return np.concatenate(parts)


def passes_distance_gate(feature, predicted_label_id, bundle, slack=1.3):
    centroids = bundle.get("centroids")
    thresholds = bundle.get("distance_thresholds")
    if not centroids or not thresholds:
        return True, 0.0, 0.0
    centroid = centroids[predicted_label_id]
    threshold = float(thresholds[predicted_label_id]) * slack
    distance = float(np.linalg.norm(feature.reshape(-1) - centroid))
    return distance <= threshold, distance, threshold


# ── Load SVM model ──────────────────────────────────────────────────

SVM_PATH = MODEL_DIR / "isl_az_live_svm.joblib"
bundle = joblib.load(SVM_PATH)
model = bundle["model"]
class_names = bundle["class_names"].astype(str)
print(f"Model loaded: {len(class_names)} classes", flush=True)

CONFIDENCE_THRESHOLD = 0.70
NOT_DETECTABLE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.10
DISTANCE_SLACK = 1.3
MIN_HAND_AREA = 0.015


# ── Prediction function ────────────────────────────────────────────

def predict_sign(image):
    """Accept a webcam frame, return prediction results."""
    if image is None:
        return "No image received", "", ""

    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image

    # Detect hands
    all_hands = detect_hands(rgb)
    selected = select_primary_hands(all_hands, max_hands=2, min_area=MIN_HAND_AREA)

    if len(selected) < 1:
        return "👋 No hand detected — show your hand to the camera", "", ""

    # Build feature vector and predict
    feature = hands_to_feature(selected).reshape(1, -1)
    probabilities = model.predict_proba(feature)[0]
    sorted_idx = np.argsort(probabilities)[::-1]
    best_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else best_idx

    confidence = float(probabilities[best_idx])
    gap = confidence - float(probabilities[second_idx])
    predicted_label_id = int(model.classes_[best_idx])
    label = str(class_names[predicted_label_id])

    passes_dist, distance, threshold = passes_distance_gate(
        feature, predicted_label_id, bundle, DISTANCE_SLACK
    )

    accepted = confidence >= CONFIDENCE_THRESHOLD and gap >= MARGIN_THRESHOLD and passes_dist

    if confidence < NOT_DETECTABLE_THRESHOLD:
        return "🤔 Sign not clear — try a cleaner hand position", f"{confidence:.0%}", f"{len(selected)} hand(s)"

    if not accepted:
        return f"❓ Maybe **{label}** (low confidence)", f"{confidence:.0%}", f"{len(selected)} hand(s)"

    return f"# ✅ {label}", f"{confidence:.0%}", f"{len(selected)} hand(s)"


# ── Gradio UI ───────────────────────────────────────────────────────

DESCRIPTION = """
# 🤟 Visual Voice — ISL Translator

**Indian Sign Language A-Z Alphabet Recognition** using MediaPipe hand landmarks and an SVM classifier.

Made by **Divyansh** | Made under **Navin Sir's guidance**

### How to use:
1. Allow camera access when prompted
2. Show an ISL alphabet sign (A-Z) to your webcam
3. The model will predict the letter in real-time

> **Tip:** Keep your hand clearly visible with good lighting for best results.
"""

with gr.Blocks(
    title="Visual Voice — ISL Translator",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
    ),
    css="""
    .main-title { text-align: center; }
    .prediction-box { font-size: 2rem !important; text-align: center; min-height: 80px; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown(DESCRIPTION, elem_classes="main-title")

    with gr.Row():
        with gr.Column(scale=2):
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="📷 Your Webcam",
                streaming=True,
                mirror_webcam=True,
            )
        with gr.Column(scale=1):
            prediction = gr.Markdown(
                value="Waiting for camera...",
                label="Predicted Letter",
                elem_classes="prediction-box",
            )
            confidence = gr.Textbox(label="🎯 Confidence", interactive=False)
            hands_count = gr.Textbox(label="🖐️ Hands Detected", interactive=False)

    with gr.Row():
        gr.Markdown("""
        ### 📊 Model Info
        | Detail | Value |
        |--------|-------|
        | **Model** | SVM (RBF kernel) |
        | **Features** | 128 MediaPipe landmarks |
        | **Classes** | 26 (A-Z) |
        | **Accuracy** | 99.8% (offline) |
        """)

    webcam.stream(
        fn=predict_sign,
        inputs=[webcam],
        outputs=[prediction, confidence, hands_count],
    )

if __name__ == "__main__":
    demo.launch()
