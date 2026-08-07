#!/usr/bin/env python3
"""Extract MediaPipe two-hand landmarks from the clean dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
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


def resize_for_detection(image, cv2, target_size: int):
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if largest_side >= target_size:
        return image

    scale = target_size / largest_side
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )


def hands_to_feature(hand_landmarks, np, max_hands: int = 2):
    feature_parts = []
    ordered_hands = sorted(
        hand_landmarks or [],
        key=lambda hand: sum(point.x for point in hand.landmark) / len(hand.landmark),
    )

    for hand in ordered_hands[:max_hands]:
        feature_parts.append(np.array([1.0], dtype=np.float32))
        feature_parts.append(normalized_hand_vector(hand, np))

    missing_hands = max_hands - len(ordered_hands[:max_hands])
    for _ in range(missing_hands):
        feature_parts.append(np.zeros(1 + 21 * 3, dtype=np.float32))

    return np.concatenate(feature_parts)


def read_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8") as file:
        yield from csv.DictReader(file)


def parse_dataset_dirs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def numeric_sort_key(path: Path):
    if path.stem.isdigit():
        return (0, int(path.stem), path.name.lower())
    return (1, path.name.lower())


def read_dataset_dirs(project_root: Path, dataset_dirs: list[str], selected_classes: set[str]):
    for label in sorted(selected_classes):
        label_index = ord(label) - ord("A")
        class_dir = None
        for dataset_dir in dataset_dirs:
            candidate = project_root / dataset_dir / label
            if candidate.exists():
                class_dir = candidate
                break

        if class_dir is None:
            continue

        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=numeric_sort_key,
        )
        for sample_index, image_path in enumerate(image_paths):
            yield {
                "label": label,
                "label_index": str(label_index),
                "sample_index": str(sample_index),
                "path": image_path.relative_to(project_root).as_posix(),
            }


def read_rows(project_root: Path, manifest_path: Path, dataset_dirs: list[str], selected_classes: set[str]):
    if dataset_dirs:
        yield from read_dataset_dirs(project_root, dataset_dirs, selected_classes)
    else:
        yield from read_manifest(manifest_path)


def print_progress(processed, accepted, failed, start_time, total):
    elapsed = time.perf_counter() - start_time
    rate = processed / elapsed if elapsed else 0
    remaining = max(total - processed, 0)
    eta_minutes = (remaining / rate / 60) if rate else 0
    print(
        f"processed={processed} accepted={accepted} failed={failed} "
        f"rate={rate:.2f}/s eta={eta_minutes:.1f}m",
        flush=True,
    )


def parse_classes(value: str) -> set[str]:
    if not value:
        return set()

    classes = {
        item.strip().upper()
        for item in value.replace(" ", "").split(",")
        if item.strip()
    }
    invalid = sorted(item for item in classes if len(item) != 1 or not item.isalpha())
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid class labels: {', '.join(invalid)}"
        )
    return classes


def count_selected_rows(project_root: Path, manifest_path: Path, dataset_dirs: list[str], selected_classes: set[str], limit: int) -> int:
    count = 0
    for row in read_rows(project_root, manifest_path, dataset_dirs, selected_classes):
        if selected_classes and row["label"] not in selected_classes:
            continue
        count += 1
        if limit and count >= limit:
            return limit
    return count


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(project_root / "manifests" / "training_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "data" / "landmarks_az.npz"),
    )
    parser.add_argument(
        "--failures-out",
        default=str(project_root / "manifests" / "landmark_failures.csv"),
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--detection-size", type=int, default=512)
    parser.add_argument("--required-hands", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N manifest rows. Use 0 for all rows.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after this many processed images.",
    )
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=parse_classes("A,B"),
        help="Comma-separated labels to process, for example A,B. Use ALL for every class.",
    )
    parser.add_argument(
        "--dataset-dirs",
        type=parse_dataset_dirs,
        default=[],
        help="Comma-separated dataset folders to scan instead of using the manifest.",
    )
    args = parser.parse_args()

    if args.classes == {"ALL"}:
        args.classes = set()

    cv2, mp, np, ISLHandDetector = require_dependencies()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    failures_path = Path(args.failures_out)

    features = []
    labels = []
    sample_paths = []
    failures = []
    accepted_by_label = Counter()
    failed_by_label = Counter()
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    start_time = time.perf_counter()
    processed = 0
    interrupted = False
    total_rows = count_selected_rows(
        project_root,
        manifest_path,
        args.dataset_dirs,
        args.classes,
        args.limit,
    )

    hands = ISLHandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
    )

    print(
        f"Starting extraction: total={total_rows} "
        f"classes={','.join(sorted(args.classes)) if args.classes else 'ALL'} "
        f"required_hands={args.required_hands} detection_size={args.detection_size}",
        flush=True,
    )

    try:
        for row in read_rows(project_root, manifest_path, args.dataset_dirs, args.classes):
            if args.classes and row["label"] not in args.classes:
                continue

            if args.limit and processed >= args.limit:
                break

            processed += 1
            image_path = project_root / row["path"]
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append({**row, "reason": "image_not_readable"})
                failed_by_label[row["label"]] += 1
            else:
                try:
                    image = resize_for_detection(image, cv2, args.detection_size)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb_image)
                    detected_hands = len(result.multi_hand_landmarks or [])
                    if detected_hands < args.required_hands:
                        failures.append(
                            {
                                **row,
                                "reason": f"insufficient_hands_detected:{detected_hands}",
                            }
                        )
                        failed_by_label[row["label"]] += 1
                    else:
                        features.append(hands_to_feature(result.multi_hand_landmarks, np))
                        labels.append(int(row["label_index"]))
                        sample_paths.append(row["path"])
                        accepted_by_label[row["label"]] += 1
                except Exception as exc:
                    failures.append({**row, "reason": f"processing_error:{type(exc).__name__}"})
                    failed_by_label[row["label"]] += 1

            if args.progress_every > 0 and processed % args.progress_every == 0:
                print_progress(processed, len(features), len(failures), start_time, total_rows)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving partial output...", flush=True)
    finally:
        hands.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.array(features, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
        paths=np.array(sample_paths),
        class_names=np.array(class_names),
    )

    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["label", "label_index", "sample_index", "path", "reason"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)

    print(json.dumps(
        {
            "output": str(output_path),
            "processed": processed,
            "samples": len(features),
            "failures": len(failures),
            "feature_size": int(features[0].shape[0]) if features else 0,
            "required_hands": args.required_hands,
            "detection_size": args.detection_size,
            "classes": sorted(args.classes) if args.classes else "ALL",
            "dataset_dirs": args.dataset_dirs,
            "interrupted": interrupted,
            "accepted_by_label": dict(sorted(accepted_by_label.items())),
            "failed_by_label": dict(sorted(failed_by_label.items())),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
