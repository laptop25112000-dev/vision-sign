#!/usr/bin/env python3
"""Collect webcam landmark samples for custom live-environment training."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def require_dependencies():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe_wrapper import ISLHandDetector
    except ImportError as exc:
        import sys
        print(f"\n[DIAGNOSTIC] Python Executable: {sys.executable}", file=sys.stderr)
        print(f"[DIAGNOSTIC] Python Version: {sys.version}", file=sys.stderr)
        print(f"[DIAGNOSTIC] Import Error: {exc}\n", file=sys.stderr)
        raise SystemExit(
            "Missing dependency. Install project requirements first:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return cv2, mp, np, ISLHandDetector


def normalized_hand_vector(hand_landmarks, np):
    landmarks = np.array(
        [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
        dtype=np.float32,
    )
    landmarks = landmarks - landmarks[0]
    scale = float(np.max(np.linalg.norm(landmarks, axis=1)))
    if scale > 0:
        landmarks = landmarks / scale
    return landmarks.reshape(-1)


def hands_to_feature(hand_landmarks, np, max_hands: int = 2):
    ordered_hands = sorted(
        hand_landmarks or [],
        key=lambda hand: sum(point.x for point in hand.landmark) / len(hand.landmark),
    )
    feature_parts = []
    for hand in ordered_hands[:max_hands]:
        feature_parts.append(np.array([1.0], dtype=np.float32))
        feature_parts.append(normalized_hand_vector(hand, np))
    missing_hands = max_hands - len(ordered_hands[:max_hands])
    for _ in range(missing_hands):
        feature_parts.append(np.zeros(1 + 21 * 3, dtype=np.float32))
    return np.concatenate(feature_parts)


def hand_box_area(hand_landmarks):
    xs = [point.x for point in hand_landmarks.landmark]
    ys = [point.y for point in hand_landmarks.landmark]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def select_primary_hands(hand_landmarks, max_hands: int, min_area: float):
    hands = [
        hand
        for hand in (hand_landmarks or [])
        if hand_box_area(hand) >= min_area
    ]
    return sorted(hands, key=hand_box_area, reverse=True)[:max_hands]


def save_dataset(path, features, labels, class_names, np):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.array(features, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
        paths=np.array([f"webcam_sample_{index}" for index in range(len(labels))]),
        class_names=np.array(class_names),
    )


def draw_hands(frame, hand_landmarks, mp, cv2):
    has_legacy_drawing = hasattr(mp, "solutions") and hasattr(mp.solutions, "drawing_utils")
    if has_legacy_drawing:
        for hand in hand_landmarks or []:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
            )
    else:
        # Fallback to custom drawing using cv2
        height, width = frame.shape[:2]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        for hand in hand_landmarks or []:
            points = []
            for lm in hand.landmark:
                px = int(lm.x * width)
                py = int(lm.y * height)
                points.append((px, py))
            for start_idx, end_idx in connections:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], (200, 50, 50), 2)
            for idx, (px, py) in enumerate(points):
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(frame, (px, py), 5, (50, 230, 50), -1)
                else:
                    cv2.circle(frame, (px, py), 3, (240, 240, 240), -1)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(project_root / "data" / "webcam_landmarks_abcd.npz"),
    )
    parser.add_argument("--classes", default="A,B,C,D")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--required-hands", type=int, default=2, choices=(1, 2))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-area", type=float, default=0.015)
    parser.add_argument("--target-per-class", type=int, default=50)
    parser.add_argument("--capture-interval", type=float, default=0.25)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview frame. Off by default so that coordinates match dataset orientation.",
    )
    args = parser.parse_args()

    cv2, mp, np, ISLHandDetector = require_dependencies()
    selected_classes = [
        item.strip().upper()
        for item in args.classes.split(",")
        if item.strip()
    ]
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    features = []
    labels = []
    counts = {label: 0 for label in selected_classes}
    current_label = selected_classes[0]
    auto_capture = False
    last_capture_time = 0.0

    # Use DirectShow backend on Windows to fix connection latency/freezing bugs
    if os.name == "nt":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    # Set camera properties for responsiveness and size stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    hands = ISLHandDetector(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=args.min_detection_confidence,
    )

    print(
        "Keys: 1/2/3/4 select label, ENTER toggles auto-capture, "
        "SPACE saves one sample, S saves file, ESC quits."
    )
    print(
        "Label keys: "
        + ", ".join(
            f"{index + 1}={label}" for index, label in enumerate(selected_classes[:9])
        )
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # NOTE: We do not mirror prior to detection so landmark features
            # are saved in standard orientation compatible with local dataset training.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            selected_hands = select_primary_hands(
                result.multi_hand_landmarks,
                max_hands=2,
                min_area=args.min_hand_area,
            )
            detected_hands = len(selected_hands)
            feature = None
            if detected_hands >= args.required_hands:
                feature = hands_to_feature(selected_hands, np)

            now = time.perf_counter()
            if (
                auto_capture
                and feature is not None
                and counts[current_label] < args.target_per_class
                and now - last_capture_time >= args.capture_interval
            ):
                features.append(feature)
                labels.append(ord(current_label) - ord("A"))
                counts[current_label] += 1
                last_capture_time = now
                print(f"Auto saved {current_label}: {counts[current_label]}", flush=True)

                if counts[current_label] >= args.target_per_class:
                    print(f"{current_label} target reached.", flush=True)
                    auto_capture = False
                    current_index = selected_classes.index(current_label)
                    if current_index + 1 < len(selected_classes):
                        next_label = selected_classes[current_index + 1]
                        current_label = next_label
                        print(
                            f"Switched to {current_label}. Show the sign, then press ENTER.",
                            flush=True,
                        )
                    else:
                        print("All targets reached. Press S or ESC to save.", flush=True)

            # Draw annotations in standard unmirrored coordinate space
            draw_hands(frame, selected_hands, mp, cv2)

            # Mirror the visualization frame if requested, flipping landmarks too
            if args.mirror:
                frame = cv2.flip(frame, 1)

            if auto_capture and detected_hands < args.required_hands:
                auto_status = f"WAITING {detected_hands}/{args.required_hands}"
            elif auto_capture:
                auto_status = "CAPTURING"
            else:
                auto_status = "OFF"

            status = (
                f"Label: {current_label} | auto: {auto_status} | "
                f"hands: {detected_hands} | "
                + " ".join(f"{label}:{counts[label]}" for label in selected_classes)
            )
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect Webcam Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (10, 13):
                auto_capture = not auto_capture
                last_capture_time = 0.0
                if auto_capture:
                    print(
                        f"Auto-capture armed. Show {args.required_hands} hand(s) to start.",
                        flush=True,
                    )
                else:
                    print("Auto-capture: OFF", flush=True)
            if key == ord("s"):
                save_dataset(args.output, features, labels, class_names, np)
                print(f"Saved {len(labels)} samples to {args.output}", flush=True)
            if key == ord(" ") and feature is not None:
                features.append(feature)
                labels.append(ord(current_label) - ord("A"))
                counts[current_label] += 1
                print(f"Saved {current_label}: {counts[current_label]}", flush=True)
            elif key == ord(" ") and feature is None:
                print("No usable hand landmarks to save.", flush=True)
            else:
                if ord("1") <= key <= ord("9"):
                    selected_index = key - ord("1")
                    if selected_index < len(selected_classes):
                        current_label = selected_classes[selected_index]
                    print(f"Current label: {current_label}", flush=True)
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    save_dataset(args.output, features, labels, class_names, np)
    print(f"Final saved {len(labels)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
