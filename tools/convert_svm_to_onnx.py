#!/usr/bin/env python3
"""Convert the SVM joblib model to ONNX format for browser inference.

Outputs:
  web/model/isl_az_live_svm.onnx   – the SVM model in ONNX format
  web/model/class_names.json       – ordered class name list
  web/model/centroids.json         – per-class centroid vectors (for distance gating)
  web/model/distance_thresholds.json – per-class distance thresholds
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    sys.exit(
        "Missing skl2onnx. Install it:\n"
        "  pip install skl2onnx"
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "isl_az_live_svm.joblib"
    output_dir = project_root / "web" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_path} ...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    class_names = bundle["class_names"].astype(str).tolist()

    # Convert SVM to ONNX
    n_features = 128  # 2 hands × (1 flag + 21 landmarks × 3 coords)
    initial_type = [("features", FloatTensorType([None, n_features]))]

    print("Converting to ONNX ...")
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
        options={id(model): {"zipmap": False}},
    )

    onnx_path = output_dir / "isl_az_live_svm.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  -> {onnx_path}  ({onnx_path.stat().st_size:,} bytes)")

    # Export class names
    class_names_path = output_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    print(f"  -> {class_names_path}")

    # Export centroids
    centroids = bundle.get("centroids")
    if centroids:
        centroids_serializable = {
            str(k): v.tolist() if hasattr(v, "tolist") else v
            for k, v in centroids.items()
        }
        centroids_path = output_dir / "centroids.json"
        with open(centroids_path, "w") as f:
            json.dump(centroids_serializable, f)
        print(f"  -> {centroids_path}")
    else:
        print("  (no centroids in bundle)")

    # Export distance thresholds
    thresholds = bundle.get("distance_thresholds")
    if thresholds:
        thresholds_serializable = {
            str(k): float(v) for k, v in thresholds.items()
        }
        thresholds_path = output_dir / "distance_thresholds.json"
        with open(thresholds_path, "w") as f:
            json.dump(thresholds_serializable, f)
        print(f"  -> {thresholds_path}")
    else:
        print("  (no distance_thresholds in bundle)")

    # Also export model.classes_ so the browser knows label ID mapping
    classes_path = output_dir / "model_classes.json"
    with open(classes_path, "w") as f:
        json.dump(model.classes_.tolist(), f)
    print(f"  -> {classes_path}")

    print("\nDone! Your web/model/ directory is ready for Vercel deployment.")


if __name__ == "__main__":
    main()
