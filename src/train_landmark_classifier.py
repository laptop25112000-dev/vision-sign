#!/usr/bin/env python3
"""Train a sklearn classifier from extracted MediaPipe landmark features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_dependencies():
    try:
        import joblib
        import numpy as np
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install project requirements first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    return {
        "joblib": joblib,
        "np": np,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "train_test_split": train_test_split,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def parse_classes(value: str) -> list[str]:
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


def parse_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def balanced_subset(features, labels, selected_label_ids, np, seed: int):
    rng = np.random.default_rng(seed)
    indexes_by_label = {
        label_id: np.where(labels == label_id)[0]
        for label_id in selected_label_ids
    }
    min_count = min(len(indexes) for indexes in indexes_by_label.values())
    selected_indexes = []

    for label_id in selected_label_ids:
        indexes = indexes_by_label[label_id]
        selected_indexes.append(rng.choice(indexes, size=min_count, replace=False))

    selected_indexes = np.concatenate(selected_indexes)
    rng.shuffle(selected_indexes)
    return features[selected_indexes], labels[selected_indexes], min_count


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
        centroid = class_features.mean(axis=0)
        distances = np.linalg.norm(class_features - centroid, axis=1)
        centroids[label_id] = centroid
        thresholds[label_id] = float(np.percentile(distances, percentile) * slack)
    return centroids, thresholds


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landmarks",
        type=parse_paths,
        default=str(project_root / "data" / "landmarks_az.npz"),
        help="Comma-separated .npz landmark files to combine.",
    )
    parser.add_argument(
        "--model-out",
        default=str(project_root / "models" / "isl_ab_svm.joblib"),
    )
    parser.add_argument(
        "--metrics-out",
        default=str(project_root / "models" / "isl_ab_metrics.json"),
    )
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("A,B"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-hands", type=int, default=1, choices=(1, 2))
    parser.add_argument("--distance-percentile", type=float, default=95.0)
    parser.add_argument("--distance-slack", type=float, default=1.35)
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()

    deps = require_dependencies()
    joblib = deps["joblib"]
    np = deps["np"]

    loaded = [np.load(path, allow_pickle=True) for path in args.landmarks]
    class_names = loaded[0]["class_names"].astype(str)
    features = np.concatenate([data["features"] for data in loaded], axis=0)
    labels = np.concatenate([data["labels"] for data in loaded], axis=0)

    label_ids = []
    for class_name in args.classes:
        matches = np.where(class_names == class_name)[0]
        if len(matches) == 0:
            raise SystemExit(f"Class {class_name} not found in {args.landmarks}")
        label_ids.append(int(matches[0]))

    mask = np.isin(labels, label_ids)
    features = features[mask]
    labels = labels[mask]
    counts_before_balance = class_counts(labels, label_ids, class_names, np)

    missing_or_empty = [
        class_name
        for class_name, count in counts_before_balance.items()
        if count == 0
    ]
    if missing_or_empty:
        raise SystemExit(
            "No usable samples found for: "
            + ", ".join(missing_or_empty)
            + "\nLandmark file: "
            + ", ".join(args.landmarks)
            + "\nCounts in this file for requested classes: "
            + json.dumps(counts_before_balance)
            + "\nRun extraction again for these classes, or train only on classes with samples."
        )

    if not args.no_balance:
        features, labels, per_class_count = balanced_subset(
            features,
            labels,
            label_ids,
            np,
            args.seed,
        )
    else:
        per_class_count = None

    x_train, x_test, y_train, y_test = deps["train_test_split"](
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model = deps["make_pipeline"](
        deps["StandardScaler"](),
        deps["SVC"](kernel="rbf", C=10, gamma="scale", probability=True),
    )
    model.fit(x_train, y_train)
    centroids, distance_thresholds = class_distance_profile(
        features,
        labels,
        label_ids,
        np,
        args.distance_percentile,
        args.distance_slack,
    )

    predictions = model.predict(x_test)
    target_names = [class_names[label_id] for label_id in label_ids]
    accuracy = float(deps["accuracy_score"](y_test, predictions))
    report = deps["classification_report"](
        y_test,
        predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = deps["confusion_matrix"](
        y_test,
        predictions,
        labels=label_ids,
    )

    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "class_names": class_names,
            "label_ids": label_ids,
            "feature_size": int(features.shape[1]),
            "required_hands": args.required_hands,
            "centroids": centroids,
            "distance_thresholds": distance_thresholds,
            "distance_percentile": args.distance_percentile,
            "distance_slack": args.distance_slack,
        },
        model_out,
    )

    metrics = {
        "classes": target_names,
        "landmarks": args.landmarks,
        "counts_before_balance": counts_before_balance,
        "balanced": not args.no_balance,
        "required_hands": args.required_hands,
        "distance_percentile": args.distance_percentile,
        "distance_slack": args.distance_slack,
        "distance_thresholds": {
            class_names[label_id]: distance_thresholds[label_id]
            for label_id in label_ids
        },
        "per_class_count": per_class_count,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "model_out": str(model_out),
    }
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
