#!/usr/bin/env python3
"""Extract MediaPipe hand landmarks from images and save as .npz for training.

Usage:
  python tools/extract_from_images.py --label M --images img1.jpg img2.jpg --output data/webcam_landmarks_my_M.npz
  python tools/extract_from_images.py --label N --images n1.jpg n2.jpg n3.jpg --output data/webcam_landmarks_my_N.npz --augment 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def augment_landmarks(feature: np.ndarray, count: int = 200, noise_std: float = 0.012, seed: int = 42) -> np.ndarray:
    """Generate augmented copies of a 128-d landmark vector by adding Gaussian noise to coordinates."""
    rng = np.random.default_rng(seed)
    augmented = []
    for i in range(count):
        feat_copy = feature.copy()
        for hand_idx in range(2):
            offset = hand_idx * 64
            if feat_copy[offset] > 0.5:  # hand is present
                noise = rng.normal(loc=0.0, scale=noise_std, size=63)
                feat_copy[offset + 1 : offset + 64] += noise
        augmented.append(feat_copy)
        # Vary seed per sample so we get different noise each time
        rng = np.random.default_rng(seed + i + 1)
    return np.array(augmented, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from images and save as .npz for ISL model training."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Single letter A-Z that this hand sign represents.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more image file paths.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file path (e.g. data/webcam_landmarks_my_M.npz).",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=200,
        help="Number of augmented samples to generate per image (default: 200).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.012,
        help="Gaussian noise std dev for augmentation (default: 0.012).",
    )
    args = parser.parse_args()

    # Validate label
    label = args.label.strip().upper()
    if len(label) != 1 or not label.isalpha():
        sys.exit(f"Error: --label must be a single letter A-Z, got '{args.label}'")

    label_id = ord(label) - ord("A")

    # Import dependencies
    try:
        import cv2
        import mediapipe as mp
        from mediapipe_wrapper import ISLHandDetector
        from extract_landmarks import resize_for_detection, hands_to_feature
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Make sure you have opencv-python and mediapipe installed:\n"
            "  pip install opencv-python mediapipe"
        )

    # Init detector
    detector = ISLHandDetector(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.3,
    )

    all_features = []
    all_labels = []

    for img_path_str in args.images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"  SKIP: {img_path} does not exist")
            continue

        print(f"Processing: {img_path}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  SKIP: Failed to read image")
            continue

        # Resize for better detection
        resized = resize_for_detection(image, cv2, target_size=1024)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        result = detector.process(rgb)

        if not result.multi_hand_landmarks:
            print(f"  WARNING: No hands detected!")
            continue

        print(f"  Detected {len(result.multi_hand_landmarks)} hand(s)")

        # Select primary hands by size
        hand_data = []
        for hand in result.multi_hand_landmarks:
            h_xs = [pt.x for pt in hand.landmark]
            h_ys = [pt.y for pt in hand.landmark]
            area = (max(h_xs) - min(h_xs)) * (max(h_ys) - min(h_ys))
            hand_data.append((hand, area))

        hand_data.sort(key=lambda x: x[1], reverse=True)
        selected = [h[0] for h in hand_data[:2]]

        # Extract 128-d feature vector
        feature = hands_to_feature(selected, np)

        # Augment
        augmented = augment_landmarks(
            feature,
            count=args.augment,
            noise_std=args.noise,
            seed=42 + len(all_features),
        )

        all_features.append(augmented)
        all_labels.extend([label_id] * len(augmented))
        print(f"  Generated {len(augmented)} augmented samples for '{label}'")

    if not all_features:
        sys.exit("No landmarks extracted from any image. Check your image files.")

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels, dtype=np.int64)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        paths=np.array([f"user_sample_{i}" for i in range(len(labels))]),
        class_names=np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    )

    print(f"\n✅ Saved {len(labels)} samples to {output_path}")
    print(f"\nNext step: python tools/retrain.py --classes ALL")


if __name__ == "__main__":
    main()
