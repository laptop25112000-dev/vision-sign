#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np

# Ensure we can import modules from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    import cv2
    import mediapipe as mp
    from mediapipe_wrapper import ISLHandDetector
    from extract_landmarks import resize_for_detection, hands_to_feature
except ImportError as exc:
    sys.exit(f"Failed to import dependencies: {exc}")

UPLOAD_DIR = Path(r"C:\Users\Divyansh\.gemini\antigravity\brain\464d20de-de65-4af4-9578-086494ff316a\.user_uploaded")
OUTPUT_NPZ = Path(r"C:\Users\Divyansh\Downloads\ISL TRANSLATOR\data\webcam_landmarks_mn_extra.npz")

# Map of label to image filename
IMAGES = {
    "M": "media_1786306394680.jpg",
    "N": "media_1786306402221.jpg"
}

def augment_landmarks(feature, count=200, noise_std=0.015, seed=42):
    """Augment a landmark feature vector by adding small random noise to coordinates."""
    rng = np.random.default_rng(seed)
    augmented = []
    
    # feature shape: (128,)
    # The 128 elements consist of two hand parts (64 elements each).
    # Each hand part has: 1.0 (presence flag) + 63 (normalized landmarks)
    # We should only add noise to the landmark coordinates (where presence flag is 1.0)
    for _ in range(count):
        feat_copy = feature.copy()
        for hand_idx in range(2):
            offset = hand_idx * 64
            if feat_copy[offset] > 0.5: # hand is present
                # coordinates are at offset + 1 to offset + 63
                noise = rng.normal(loc=0.0, scale=noise_std, size=63)
                feat_copy[offset+1 : offset+64] += noise
        augmented.append(feat_copy)
        
    return np.array(augmented, dtype=np.float32)

def main():
    detector = ISLHandDetector(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.3
    )
    
    all_features = []
    all_labels = []
    
    for label, filename in IMAGES.items():
        img_path = UPLOAD_DIR / filename
        if not img_path.exists():
            print(f"Error: {img_path} does not exist!")
            continue
            
        print(f"Processing {label} image: {img_path} ...")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Failed to read image {img_path}")
            continue
            
        # Resize image for better MediaPipe detection
        resized = resize_for_detection(image, cv2, target_size=1024)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        result = detector.process(rgb)
        
        # Check if hands detected
        if not result.multi_hand_landmarks:
            print(f"Warning: MediaPipe failed to detect hands for {label}!")
            continue
            
        print(f"  Detected {len(result.multi_hand_landmarks)} hand(s) for {label}.")
        
        # Select up to 2 primary hands based on size (min area 0.005)
        xs = []
        for hand in result.multi_hand_landmarks:
            h_xs = [point.x for point in hand.landmark]
            h_ys = [point.y for point in hand.landmark]
            area = (max(h_xs) - min(h_xs)) * (max(h_ys) - min(h_ys))
            xs.append((hand, area))
            
        xs.sort(key=lambda x: x[1], reverse=True)
        selected_hands = [x[0] for x in xs[:2]]
        
        # Convert hands to 128 feature vector
        feature = hands_to_feature(selected_hands, np)
        
        # Augment feature to generate robust samples
        augmented_feats = augment_landmarks(feature, count=250, noise_std=0.012)
        
        label_id = ord(label) - ord("A")
        
        all_features.append(augmented_feats)
        all_labels.extend([label_id] * len(augmented_feats))
        print(f"  Successfully generated {len(augmented_feats)} augmented landmark samples for {label}.")

    if not all_features:
        print("No features extracted. Exiting.")
        return 1
        
    features_concat = np.concatenate(all_features, axis=0)
    labels_concat = np.array(all_labels, dtype=np.int64)
    
    # Save npz file
    np.savez_compressed(
        OUTPUT_NPZ,
        features=features_concat,
        labels=labels_concat,
        paths=np.array([f"user_image_sample_{i}" for i in range(len(labels_concat))]),
        class_names=np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    )
    print(f"Saved extracted and augmented landmarks to {OUTPUT_NPZ}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
