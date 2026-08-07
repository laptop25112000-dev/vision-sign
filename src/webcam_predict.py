#!/usr/bin/env python3
"""Run live A/B prediction from webcam using the trained landmark classifier."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, deque
from pathlib import Path


def require_dependencies():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import cv2
        import joblib
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
    return cv2, joblib, mp, np, ISLHandDetector


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


def stable_prediction(history: deque[str], min_votes: int) -> str:
    if len(history) < min_votes:
        return ""
    label, count = Counter(history).most_common(1)[0]
    return label if count >= min_votes else ""


def passes_distance_gate(feature, predicted_label_id, bundle, np, slack: float):
    centroids = bundle.get("centroids")
    thresholds = bundle.get("distance_thresholds")
    if not centroids or not thresholds:
        return True, 0.0, 0.0

    centroid = centroids[predicted_label_id]
    threshold = float(thresholds[predicted_label_id]) * slack
    distance = float(np.linalg.norm(feature.reshape(-1) - centroid))
    return distance <= threshold, distance, threshold


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


def create_tts(enabled: bool, rate: int):
    if not enabled:
        return None
    try:
        import pyttsx3
    except ImportError as exc:
        raise SystemExit("pyttsx3 is not installed. Run: pip install pyttsx3") from exc

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    return engine


def speak_letter(engine, label: str):
    if engine is None:
        return
    engine.stop()
    engine.say(label)
    engine.runAndWait()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(project_root / "models" / "isl_ab_svm.joblib"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--not-detectable-confidence", type=float, default=0.60)
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument("--distance-slack", type=float, default=1.0)
    parser.add_argument("--disable-distance-gate", action="store_true")
    parser.add_argument("--show-uncertain", action="store_true")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--min-votes", type=int, default=8)
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--speak-cooldown", type=float, default=1.2)
    parser.add_argument("--speech-rate", type=int, default=145)
    parser.add_argument("--required-hands", type=int, default=1, choices=(1, 2))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-area", type=float, default=0.015)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the webcam frame before prediction. Off by default to match dataset orientation.",
    )
    parser.add_argument(
        "--auto-space-delay",
        type=float,
        default=2.0,
        help="Delay in seconds of 'no hand' to automatically append a space. Use 0 to disable.",
    )
    parser.add_argument(
        "--reset-delay-frames",
        type=int,
        default=10,
        help="Consecutive frames without a stable letter required to reset the last appended letter and allow duplicates.",
    )
    args = parser.parse_args()

    cv2, joblib, mp, np, ISLHandDetector = require_dependencies()
    tts = create_tts(args.speak, args.speech_rate)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    class_names = bundle["class_names"].astype(str)

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

    history: deque[str] = deque(maxlen=args.window)
    last_spoken = ""
    last_spoken_at = 0.0
    
    # Word Construction state
    spelling_buffer: list[str] = []
    last_appended_letter = ""
    unstable_frames_count = 0
    last_hand_seen_time = time.perf_counter()

    hands = ISLHandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
    )

    print("Keyboard Controls:", flush=True)
    print("  - SPACE: Insert manual space", flush=True)
    print("  - BACKSPACE: Delete last letter", flush=True)
    print("  - C: Clear spelling buffer", flush=True)
    print("  - ENTER: Speak spelling buffer via TTS", flush=True)
    print("  - Q: Quit application", flush=True)
    print("Press q or ESC in the webcam window to quit.", flush=True)
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.", flush=True)
                break

            # NOTE: We do NOT flip the image before MediaPipe process. Doing so negates
            # the coordinates relative to the wrist origin and degrades model accuracy.
            # We process the unmirrored frame and flip the canvas afterwards for user preview.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            selected_hands = select_primary_hands(
                result.multi_hand_landmarks,
                max_hands=2,
                min_area=args.min_hand_area,
            )
            detected_hands = len(selected_hands)
            label_text = "No hand detected"
            stable = ""

            if detected_hands >= args.required_hands:
                last_hand_seen_time = time.perf_counter()
                feature = hands_to_feature(selected_hands, np).reshape(1, -1)
                probabilities = model.predict_proba(feature)[0]
                sorted_indexes = np.argsort(probabilities)[::-1]
                best_index = int(sorted_indexes[0])
                second_index = int(sorted_indexes[1]) if len(sorted_indexes) > 1 else best_index
                confidence = float(probabilities[best_index])
                second_confidence = float(probabilities[second_index])
                margin = confidence - second_confidence
                predicted_label_id = int(model.classes_[best_index])
                predicted_label = class_names[predicted_label_id]
                passes_distance, distance, threshold = passes_distance_gate(
                    feature,
                    predicted_label_id,
                    bundle,
                    np,
                    args.distance_slack,
                )
                if args.disable_distance_gate:
                    passes_distance = True

                if (
                    confidence >= args.confidence
                    and margin >= args.margin
                    and passes_distance
                ):
                    history.append(predicted_label)
                    label_text = f"{predicted_label} {confidence:.2f}"
                else:
                    history.clear()
                    if confidence < args.not_detectable_confidence:
                        label_text = "Not detectable"
                    else:
                        label_text = f"{predicted_label} {confidence:.2f}"
                    if args.show_uncertain:
                        reason = "conf"
                        if margin < args.margin:
                            reason = "margin"
                        if not passes_distance:
                            reason = "shape"
                        label_text = (
                            f"Uncertain {predicted_label} {confidence:.2f} "
                            f"gap {margin:.2f} {reason}"
                        )

                stable = stable_prediction(history, args.min_votes)
                if stable:
                    unstable_frames_count = 0
                    
                    # Auto-append stable letter if it is different from the last one
                    if stable != last_appended_letter:
                        spelling_buffer.append(stable)
                        last_appended_letter = stable
                        print(f"Appended: {stable} | Buffer: {''.join(spelling_buffer)}", flush=True)

                    now = time.perf_counter()
                    if stable != last_spoken or now - last_spoken_at >= args.speak_cooldown:
                        speak_letter(tts, stable)
                        last_spoken = stable
                        last_spoken_at = now
                else:
                    unstable_frames_count += 1
                    if unstable_frames_count >= args.reset_delay_frames:
                        last_appended_letter = ""  # Reset active lock to allow consecutive duplicates
            else:
                history.clear()
                unstable_frames_count += 1
                if unstable_frames_count >= args.reset_delay_frames:
                    last_appended_letter = ""
                
                if detected_hands > 0:
                    last_hand_seen_time = time.perf_counter()
                    label_text = (
                        f"Detected {detected_hands}, need {args.required_hands}"
                    )
                else:
                    # Inactivity check for auto-spacing
                    if args.auto_space_delay > 0:
                        now = time.perf_counter()
                        if now - last_hand_seen_time >= args.auto_space_delay:
                            if spelling_buffer and spelling_buffer[-1] != " ":
                                spelling_buffer.append(" ")
                                last_appended_letter = ""
                                print("Auto-spaced buffer due to inactivity", flush=True)
                            last_hand_seen_time = now  # Reset countdown until next activity

            # Draw skeletons on the original frame coordinates
            draw_hands(frame, selected_hands, mp, cv2)

            # Flip the frame for preview display if mirror argument is enabled.
            # Flips hand annotations along with frame, maintaining alignment.
            if args.mirror:
                frame = cv2.flip(frame, 1)

            # Draw prediction text info
            if label_text:
                cv2.putText(
                    frame,
                    label_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if stable:
                cv2.putText(
                    frame,
                    f"Stable: {stable}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Render Spelling Buffer Glassmorphic Bar at the bottom
            height, width = frame.shape[:2]
            box_height = 60
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, height - box_height),
                (width, height),
                (15, 15, 15),
                -1,
            )
            # Add transparency blend
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            word_str = "".join(spelling_buffer)
            if not word_str:
                buffer_display = "Spelling Buffer: (sign letters to build word)"
                text_color = (130, 130, 130)
            else:
                buffer_display = f"Spelling Buffer: {word_str}"
                text_color = (50, 255, 50)

            cv2.putText(
                frame,
                buffer_display,
                (20, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                text_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ISL Translator", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # ESC or q
                break
            elif key == ord(" "):  # Manual space
                if not spelling_buffer or spelling_buffer[-1] != " ":
                    spelling_buffer.append(" ")
                    last_appended_letter = ""
            elif key == 8:  # Backspace
                if spelling_buffer:
                    spelling_buffer.pop()
                    last_appended_letter = ""
            elif key in (ord("c"), ord("C")):  # Clear
                spelling_buffer.clear()
                last_appended_letter = ""
            elif key in (10, 13):  # Enter key
                phrase = "".join(spelling_buffer).strip()
                if phrase:
                    print(f"Speaking: '{phrase}'", flush=True)
                    speak_letter(tts, phrase)
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
