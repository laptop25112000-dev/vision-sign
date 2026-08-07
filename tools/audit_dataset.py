#!/usr/bin/env python3
"""Audit the local ISL A-Z image dataset and build training manifests.

The script does not modify image files. It creates:
- manifests/training_manifest.csv: the canonical 26 x 1200 sample list
- manifests/quarantine_manifest.csv: extra or suspicious files to ignore
- manifests/dataset_audit.json: counts and warnings for repeatable checks
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


EXPECTED_CLASSES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXPECTED_PER_CLASS = 1200
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def numeric_index(path: Path) -> int | None:
    if re.fullmatch(r"\d+", path.stem):
        return int(path.stem)
    return None


def canonical_score(path: Path, index: int) -> tuple[int, int, int, str]:
    """Prefer plain names like 12.jpg over 012.jpg or other duplicates."""
    exact_numeric_name = path.stem == str(index)
    has_copy_marker = "copy" in path.stem.lower()
    return (
        0 if exact_numeric_name else 1,
        1 if has_copy_marker else 0,
        len(path.stem),
        path.name.lower(),
    )


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda item: item.as_posix().lower(),
    )


def audit_dataset(dataset_dir: Path, project_root: Path) -> tuple[list[dict], list[dict], dict]:
    manifest_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    class_counts: dict[str, dict] = {}

    existing_classes = {
        path.name: path
        for path in dataset_dir.iterdir()
        if path.is_dir() and len(path.name) == 1 and path.name.isalpha()
    }
    unexpected_dirs = sorted(
        path.name
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name not in EXPECTED_CLASSES
    )

    for label_index, class_name in enumerate(EXPECTED_CLASSES):
        class_dir = existing_classes.get(class_name)
        if class_dir is None:
            class_counts[class_name] = {
                "total_images": 0,
                "canonical_images": 0,
                "quarantined_images": 0,
                "missing_indexes": list(range(EXPECTED_PER_CLASS)),
                "duplicate_numeric_indexes": [],
            }
            continue

        files = image_files(class_dir)
        grouped_by_index: dict[int, list[Path]] = defaultdict(list)
        non_numeric: list[Path] = []

        for path in files:
            index = numeric_index(path)
            if index is None:
                non_numeric.append(path)
            else:
                grouped_by_index[index].append(path)

        selected_paths: set[Path] = set()
        missing_indexes: list[int] = []
        duplicate_indexes: list[int] = []

        for sample_index in range(EXPECTED_PER_CLASS):
            candidates = grouped_by_index.get(sample_index, [])
            if not candidates:
                missing_indexes.append(sample_index)
                continue

            if len(candidates) > 1:
                duplicate_indexes.append(sample_index)

            selected = sorted(
                candidates,
                key=lambda path: canonical_score(path, sample_index),
            )[0]
            selected_paths.add(selected)
            manifest_rows.append(
                {
                    "label": class_name,
                    "label_index": label_index,
                    "sample_index": sample_index,
                    "path": rel(selected, project_root),
                }
            )

            for duplicate in candidates:
                if duplicate == selected:
                    continue
                quarantine_rows.append(
                    {
                        "label": class_name,
                        "reason": "duplicate_numeric_index",
                        "sample_index": sample_index,
                        "path": rel(duplicate, project_root),
                    }
                )

        for index, candidates in grouped_by_index.items():
            if 0 <= index < EXPECTED_PER_CLASS:
                continue
            for path in candidates:
                quarantine_rows.append(
                    {
                        "label": class_name,
                        "reason": "outside_expected_index_range",
                        "sample_index": index,
                        "path": rel(path, project_root),
                    }
                )

        for path in non_numeric:
            quarantine_rows.append(
                {
                    "label": class_name,
                    "reason": "non_numeric_filename",
                    "sample_index": "",
                    "path": rel(path, project_root),
                }
            )

        class_counts[class_name] = {
            "total_images": len(files),
            "canonical_images": sum(1 for path in files if path in selected_paths),
            "quarantined_images": sum(
                1 for row in quarantine_rows if row["label"] == class_name
            ),
            "missing_indexes": missing_indexes,
            "duplicate_numeric_indexes": duplicate_indexes,
        }

    audit = {
        "dataset_dir": rel(dataset_dir, project_root),
        "expected_classes": list(EXPECTED_CLASSES),
        "expected_images_per_class": EXPECTED_PER_CLASS,
        "canonical_total": len(manifest_rows),
        "quarantine_total": len(quarantine_rows),
        "unexpected_directories": unexpected_dirs,
        "classes": class_counts,
        "verdict": (
            "usable_with_manifest"
            if len(manifest_rows) == len(EXPECTED_CLASSES) * EXPECTED_PER_CLASS
            else "needs_dataset_fix"
        ),
    }
    return manifest_rows, quarantine_rows, audit


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=str(project_root / "DTASET A-Z"),
        help="Path to the downloaded Kaggle A-Z folder.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(project_root / "manifests"),
        help="Directory where audit outputs are written.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    manifest_rows, quarantine_rows, audit = audit_dataset(dataset_dir, project_root)

    write_csv(
        out_dir / "training_manifest.csv",
        manifest_rows,
        ["label", "label_index", "sample_index", "path"],
    )
    write_csv(
        out_dir / "quarantine_manifest.csv",
        quarantine_rows,
        ["label", "reason", "sample_index", "path"],
    )
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "verdict": audit["verdict"],
            "canonical_total": audit["canonical_total"],
            "quarantine_total": audit["quarantine_total"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
