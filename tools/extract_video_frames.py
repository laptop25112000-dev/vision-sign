#!/usr/bin/env python3
"""Extract evenly spaced frames from a screen recording for visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out-dir", default="video_frames")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    import cv2

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration = total_frames / fps if fps else 0

    if total_frames <= 0:
        raise SystemExit("Video has no readable frames.")

    selected = []
    if args.count <= 1:
        frame_indexes = [total_frames // 2]
    else:
        frame_indexes = [
            int(round(i * (total_frames - 1) / (args.count - 1)))
            for i in range(args.count)
        ]

    for frame_index in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        timestamp = frame_index / fps if fps else 0
        out_path = out_dir / f"frame_{len(selected):02d}_{timestamp:06.2f}s.jpg"
        cv2.imwrite(str(out_path), frame)
        selected.append(
            {
                "frame": frame_index,
                "timestamp_seconds": round(timestamp, 2),
                "path": str(out_path),
            }
        )

    cap.release()
    print(
        json.dumps(
            {
                "video": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "extracted": selected,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
