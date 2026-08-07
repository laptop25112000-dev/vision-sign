#!/usr/bin/env python3
"""Analyze extracted landmark features for class balance and separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_dependencies():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing numpy. Install requirements first.") from exc
    return np


def centroid_distances(features, labels, label_ids, class_names, np):
    centroids = {
        label_id: features[labels == label_id].mean(axis=0)
        for label_id in label_ids
    }
    distances = {}
    for left in label_ids:
        for right in label_ids:
            if left >= right:
                continue
            key = f"{class_names[left]}-{class_names[right]}"
            distances[key] = float(np.linalg.norm(centroids[left] - centroids[right]))
    return distances, centroids


def nearest_centroid_confusions(features, labels, paths, label_ids, class_names, centroids, np):
    confusion = {
        class_names[label_id]: {class_names[other]: 0 for other in label_ids}
        for label_id in label_ids
    }
    outlier_scores = []
    centroid_matrix = np.array([centroids[label_id] for label_id in label_ids])

    for index, feature in enumerate(features):
        true_label = int(labels[index])
        distances = np.linalg.norm(centroid_matrix - feature, axis=1)
        predicted_label = label_ids[int(np.argmin(distances))]
        confusion[class_names[true_label]][class_names[predicted_label]] += 1

        own_distance = float(np.linalg.norm(feature - centroids[true_label]))
        nearest_wrong = float(
            min(
                np.linalg.norm(feature - centroids[label_id])
                for label_id in label_ids
                if label_id != true_label
            )
        )
        outlier_scores.append(
            {
                "path": str(paths[index]),
                "label": class_names[true_label],
                "nearest_centroid_prediction": class_names[predicted_label],
                "own_distance": own_distance,
                "nearest_wrong_distance": nearest_wrong,
                "margin": nearest_wrong - own_distance,
            }
        )

    outlier_scores.sort(key=lambda row: row["margin"])
    return confusion, outlier_scores[:25]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landmarks",
        default=str(project_root / "data" / "landmarks_abcd.npz"),
    )
    parser.add_argument("--classes", default="A,B,C,D")
    parser.add_argument(
        "--out",
        default=str(project_root / "models" / "landmark_analysis_abcd.json"),
    )
    args = parser.parse_args()

    np = require_dependencies()
    data = np.load(args.landmarks, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    paths = data["paths"] if "paths" in data.files else np.array([""] * len(labels))
    class_names = data["class_names"].astype(str)

    requested_classes = [
        item.strip().upper()
        for item in args.classes.split(",")
        if item.strip()
    ]
    label_ids = [int(np.where(class_names == label)[0][0]) for label in requested_classes]
    mask = np.isin(labels, label_ids)
    features = features[mask]
    labels = labels[mask]
    paths = paths[mask]

    counts = {
        class_names[label_id]: int(np.sum(labels == label_id))
        for label_id in label_ids
    }
    hand_presence = {}
    for label_id in label_ids:
        class_features = features[labels == label_id]
        hand_presence[class_names[label_id]] = {
            "first_hand_present": int(np.sum(class_features[:, 0] > 0.5)),
            "second_hand_present": int(np.sum(class_features[:, 64] > 0.5)),
        }

    distances, centroids = centroid_distances(features, labels, label_ids, class_names, np)
    confusion, outliers = nearest_centroid_confusions(
        features,
        labels,
        paths,
        label_ids,
        class_names,
        centroids,
        np,
    )

    result = {
        "landmarks": args.landmarks,
        "counts": counts,
        "hand_presence": hand_presence,
        "centroid_distances": distances,
        "nearest_centroid_confusion": confusion,
        "closest_or_outlier_samples": outliers,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
