#!/usr/bin/env python3
"""Retrain a classifier on a subset of letters and auto-deploy to the web model directory.

Usage:
  python tools/retrain.py --classes A,B,C
  python tools/retrain.py --classes ALL
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
WEB_MODEL_DIR = PROJECT_ROOT / "web" / "model"

def check_dependencies():
    try:
        import joblib
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        sys.exit(
            f"Missing base dependency: {exc}\n"
            "Install requirements first:\n"
            "  pip install -r requirements.txt"
        )
    
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        sys.exit(
            "Missing skl2onnx for ONNX conversion.\n"
            "Please install it:\n"
            "  pip install skl2onnx"
        )
        
    return {
        "joblib": joblib,
        "np": np,
        "accuracy_score": accuracy_score,
        "train_test_split": train_test_split,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
        "convert_sklearn": convert_sklearn,
        "FloatTensorType": FloatTensorType,
    }

def parse_classes(value: str) -> list[str]:
    if value.strip().upper() == "ALL":
        return [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    classes = [
        item.strip().upper()
        for item in value.replace(" ", "").split(",")
        if item.strip()
    ]
    invalid = [item for item in classes if len(item) != 1 or not item.isalpha()]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid class labels: {', '.join(invalid)}"
        )
    return classes

def find_landmark_files():
    base_files = glob.glob(str(DATA_DIR / "landmarks_*.npz"))
    webcam_files = glob.glob(str(DATA_DIR / "webcam_landmarks_*.npz"))
    all_files = base_files + webcam_files
    # Sort files to ensure deterministic loading order
    all_files.sort()
    return [Path(p) for p in all_files]

def load_data(files: list[Path], np):
    if not files:
        sys.exit(f"No .npz files found in {DATA_DIR}. Please make sure dataset landmarks exist.")
        
    features_list = []
    labels_list = []
    class_names = None
    
    for f_path in files:
        data = np.load(f_path, allow_pickle=True)
        if class_names is None:
            class_names = data["class_names"].astype(str)
            
        features_list.append(data["features"])
        labels_list.append(data["labels"])
        
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels, class_names

def balanced_subset(features, labels, selected_label_ids, np, seed: int):
    rng = np.random.default_rng(seed)
    indexes_by_label = {
        label_id: np.where(labels == label_id)[0]
        for label_id in selected_label_ids
    }
    
    # Filter empty labels
    active_label_ids = [lid for lid, idxs in indexes_by_label.items() if len(idxs) > 0]
    if not active_label_ids:
        sys.exit("No samples found for any of the selected classes.")
        
    min_count = min(len(indexes_by_label[lid]) for lid in active_label_ids)
    
    selected_indexes = []
    for label_id in active_label_ids:
        indexes = indexes_by_label[label_id]
        selected_indexes.append(rng.choice(indexes, size=min_count, replace=False))
        
    selected_indexes = np.concatenate(selected_indexes)
    rng.shuffle(selected_indexes)
    return features[selected_indexes], labels[selected_indexes], min_count, active_label_ids

def class_counts(labels, label_ids, class_names, np):
    return {
        class_names[label_id]: int(np.sum(labels == label_id))
        for label_id in label_ids
    }

def class_distance_profile(features, labels, label_ids, np, percentile: float, slack: float):
    centroids = {}
    thresholds = {}
    for label_id in label_ids:
        class_features = features[labels == label_id]
        if len(class_features) == 0:
            continue
        centroid = class_features.mean(axis=0)
        distances = np.linalg.norm(class_features - centroid, axis=1)
        centroids[label_id] = centroid
        thresholds[label_id] = float(np.percentile(distances, percentile) * slack)
    return centroids, thresholds

def main():
    parser = argparse.ArgumentParser(description="Retrain ISL SVM model and auto-deploy to Web UI.")
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default="ALL",
        help="Comma-separated letters to train (e.g. A,B,C) or ALL (default)."
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing subset selection."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random state seed."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="isl_az_live_svm",
        help="Base name for the output model files (e.g., isl_mnop_svm)."
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip converting to ONNX and deploying to web/model directory."
    )
    args = parser.parse_args()

    # 1. Check Dependencies
    deps = check_dependencies()
    np = deps["np"]
    joblib = deps["joblib"]
    
    # 2. Find and Load Landmark files
    npz_files = find_landmark_files()
    print("Found landmark files:")
    for f in npz_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")
        
    features, labels, class_names = load_data(npz_files, np)
    
    # 3. Resolve classes to Label IDs
    label_ids = []
    for class_name in args.classes:
        matches = np.where(class_names == class_name)[0]
        if len(matches) == 0:
            print(f"Warning: Class {class_name} not found in dataset. Skipping.")
            continue
        label_ids.append(int(matches[0]))
        
    if not label_ids:
        sys.exit("No valid training classes resolved. Exiting.")
        
    mask = np.isin(labels, label_ids)
    features = features[mask]
    labels = labels[mask]
    
    counts_before_balance = class_counts(labels, label_ids, class_names, np)
    print("\nSample counts before balancing:")
    for cls, cnt in counts_before_balance.items():
        print(f"  {cls}: {cnt}")
        
    # 4. Class balancing
    if not args.no_balance:
        features, labels, per_class_count, active_label_ids = balanced_subset(
            features, labels, label_ids, np, args.seed
        )
        print(f"\nBalanced dataset count: {per_class_count} samples per class.")
    else:
        active_label_ids = [lid for lid in label_ids if np.sum(labels == lid) > 0]
        per_class_count = None
        
    # 5. Train Split
    x_train, x_test, y_train, y_test = deps["train_test_split"](
        features,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    
    # 6. Fit Model
    print(f"\nTraining SVM on {len(active_label_ids)} classes ({len(x_train)} training samples) ...")
    model = deps["make_pipeline"](
        deps["StandardScaler"](),
        deps["SVC"](kernel="rbf", C=10, gamma="scale", probability=True),
    )
    model.fit(x_train, y_train)
    
    # 7. Evaluate
    predictions = model.predict(x_test)
    accuracy = float(deps["accuracy_score"](y_test, predictions))
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    # 8. Distance profile calculations
    centroids, thresholds = class_distance_profile(
        features, labels, active_label_ids, np, percentile=95.0, slack=1.35
    )
    
    # 9. Save Joblib Model Bundle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib_path = MODELS_DIR / f"{args.output_name}.joblib"
    
    bundle = {
        "model": model,
        "class_names": class_names,
        "label_ids": active_label_ids,
        "feature_size": int(features.shape[1]),
        "required_hands": 1,
        "centroids": centroids,
        "distance_thresholds": thresholds,
        "distance_percentile": 95.0,
        "distance_slack": 1.35,
    }
    joblib.dump(bundle, joblib_path)
    print(f"\nSaved sklearn model bundle: {joblib_path}")
    
    # 10. ONNX conversion and Web deployment
    if args.skip_onnx:
        print("\nSkipping ONNX deployment to web/model directory.")
        print("\nRetraining completed successfully!")
        return

    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\nConverting SVM model to ONNX ...")
    
    n_features = 128
    initial_type = [("features", deps["FloatTensorType"]([None, n_features]))]
    
    onnx_model = deps["convert_sklearn"](
        model,
        initial_types=initial_type,
        target_opset=15,
        options={id(model): {"zipmap": False}},
    )
    
    # Export ONNX file
    onnx_path = WEB_MODEL_DIR / f"{args.output_name}.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  -> Deployed ONNX: {onnx_path} ({onnx_path.stat().st_size:,} bytes)")
    
    # Export class names
    class_names_path = WEB_MODEL_DIR / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names.tolist(), f)
    print(f"  -> Deployed Class Names: {class_names_path}")
    
    # Export centroids
    centroids_serializable = {
        str(k): v.tolist() if hasattr(v, "tolist") else v
        for k, v in centroids.items()
    }
    centroids_path = WEB_MODEL_DIR / "centroids.json"
    with open(centroids_path, "w") as f:
        json.dump(centroids_serializable, f)
    print(f"  -> Deployed Centroids: {centroids_path}")
    
    # Export thresholds
    thresholds_serializable = {
        str(k): float(v) for k, v in thresholds.items()
    }
    thresholds_path = WEB_MODEL_DIR / "distance_thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_serializable, f)
    print(f"  -> Deployed Distance Thresholds: {thresholds_path}")
    
    # Export model classes map
    classes_path = WEB_MODEL_DIR / "model_classes.json"
    with open(classes_path, "w") as f:
        json.dump(model.classes_.tolist(), f)
    print(f"  -> Deployed Model Classes Map: {classes_path}")
    
    print("\nRetraining and deployment completed successfully!")

if __name__ == "__main__":
    main()
