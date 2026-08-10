#!/usr/bin/env python3
"""Run the isl_mnop_svm.joblib model on your webcam in real-time.

Usage:
  python tools/run_mnop.py
  python tools/run_mnop.py --model models/isl_mnop_svm.joblib
  python tools/run_mnop.py --model models/isl_mnop_live_svm.joblib
"""

import sys
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mediapipe_wrapper import ISLHandDetector
from extract_landmarks import resize_for_detection, hands_to_feature

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run an ISL SVM model on webcam.")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "isl_mnop_svm.joblib"),
        help="Path to the .joblib model file.",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)."
    )
    args = parser.parse_args()

    # ── Load Model ────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    bundle = joblib.load(args.model)

    model = bundle["model"]
    class_names = bundle["class_names"]
    label_ids = bundle["label_ids"]
    feature_size = bundle["feature_size"]
    centroids = bundle.get("centroids", {})
    thresholds = bundle.get("distance_thresholds", {})
    slack = bundle.get("distance_slack", 1.3)

    letter_map = {lid: class_names[lid] for lid in label_ids}
    print(f"Model recognises: {', '.join(letter_map.values())}")
    print(f"Feature size: {feature_size}")

    # ── Init MediaPipe ────────────────────────────────────────────────
    detector = ISLHandDetector(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
    )

    # ── Open Webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Error: Cannot open camera {args.camera}")

    print("\n── Webcam running ── Press 'q' to quit ──\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = detector.process(rgb)

        label_text = "No hand"
        confidence = 0.0
        color = (100, 100, 100)

        if result.multi_hand_landmarks:
            # Draw hand landmarks
            for hand_lms in result.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    h, w = frame.shape[:2]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 200), -1)

            # Select primary hands by area
            hand_data = []
            for hand in result.multi_hand_landmarks:
                xs = [pt.x for pt in hand.landmark]
                ys = [pt.y for pt in hand.landmark]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                hand_data.append((hand, area))
            hand_data.sort(key=lambda x: x[1], reverse=True)
            selected = [h[0] for h in hand_data[:2]]

            # Extract feature vector
            feature = hands_to_feature(selected, np)

            # Predict
            proba = model.predict_proba(feature.reshape(1, -1))[0]
            pred_idx = np.argmax(proba)
            pred_label_id = model.classes_[pred_idx]
            confidence = float(proba[pred_idx])

            letter = class_names[pred_label_id]

            # Distance gate check
            passed_gate = True
            if pred_label_id in centroids and pred_label_id in thresholds:
                centroid = centroids[pred_label_id]
                if hasattr(centroid, "__len__"):
                    centroid = np.array(centroid)
                dist = float(np.linalg.norm(feature - centroid))
                threshold = thresholds[pred_label_id]
                if dist > threshold * slack:
                    passed_gate = False

            if confidence >= 0.70 and passed_gate:
                label_text = f"{letter} ({confidence:.0%})"
                color = (0, 255, 100)
            elif confidence >= 0.50:
                label_text = f"{letter}? ({confidence:.0%})"
                color = (0, 200, 255)
            else:
                label_text = f"Unsure ({confidence:.0%})"
                color = (0, 100, 255)

        # Draw prediction on frame
        cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Show controls
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("ISL MNOP Model", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done.")


if __name__ == "__main__":
    main()
