#!/usr/bin/env python3
"""Local web UI for Vision Sign.

This uses only Python's standard HTTP server for routing. The browser captures
webcam frames and posts them to /predict; Python runs MediaPipe + the saved SVM.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


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


def passes_distance_gate(feature, predicted_label_id, bundle, np, slack: float):
    centroids = bundle.get("centroids")
    thresholds = bundle.get("distance_thresholds")
    if not centroids or not thresholds:
        return True, 0.0, 0.0

    centroid = centroids[predicted_label_id]
    threshold = float(thresholds[predicted_label_id]) * slack
    distance = float(np.linalg.norm(feature.reshape(-1) - centroid))
    return distance <= threshold, distance, threshold


class Predictor:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        not_detectable_confidence: float,
        margin: float,
        distance_slack: float,
        min_hand_area: float,
    ):
        self.cv2, joblib, self.mp, self.np, ISLHandDetector = require_dependencies()
        self.joblib = joblib
        self.model_path = model_path
        self.model_mtime = 0.0
        self.load_model()
        self.confidence = confidence
        self.not_detectable_confidence = not_detectable_confidence
        self.margin = margin
        self.distance_slack = distance_slack
        self.min_hand_area = min_hand_area
        self.hands = ISLHandDetector(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
        )

    def load_model(self):
        self.bundle = self.joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.class_names = self.bundle["class_names"].astype(str)
        self.model_mtime = self.model_path.stat().st_mtime

    def reload_model_if_changed(self):
        current_mtime = self.model_path.stat().st_mtime
        if current_mtime != self.model_mtime:
            self.load_model()

    def predict_data_url(self, data_url: str):
        self.reload_model_if_changed()
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(data_url)
        encoded = self.np.frombuffer(image_bytes, dtype=self.np.uint8)
        frame = self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "reason": "image_not_readable"}

        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        selected_hands = select_primary_hands(
            result.multi_hand_landmarks,
            max_hands=2,
            min_area=self.min_hand_area,
        )
        detected_hands = len(selected_hands)
        if detected_hands < 1:
            return {"ok": False, "reason": "no_hand_detected", "hands": detected_hands}

        feature = hands_to_feature(selected_hands, self.np).reshape(1, -1)
        probabilities = self.model.predict_proba(feature)[0]
        sorted_indexes = self.np.argsort(probabilities)[::-1]
        best_index = int(sorted_indexes[0])
        second_index = int(sorted_indexes[1]) if len(sorted_indexes) > 1 else best_index
        confidence = float(probabilities[best_index])
        second_confidence = float(probabilities[second_index])
        gap = confidence - second_confidence
        predicted_label_id = int(self.model.classes_[best_index])
        label = str(self.class_names[predicted_label_id])
        passes_distance, distance, threshold = passes_distance_gate(
            feature,
            predicted_label_id,
            self.bundle,
            self.np,
            self.distance_slack,
        )
        accepted = confidence >= self.confidence and gap >= self.margin and passes_distance
        not_detectable = confidence < self.not_detectable_confidence

        return {
            "ok": True,
            "accepted": accepted,
            "not_detectable": not_detectable,
            "label": label,
            "confidence": confidence,
            "gap": gap,
            "hands": detected_hands,
            "distance": distance,
            "threshold": threshold,
        }


def speak_text(text: str) -> tuple[bool, str]:
    text = "".join(ch for ch in text.strip() if ch.isalnum() or ch.isspace())[:40]
    if not text:
        return False, "empty_text"

    def worker():
        import pyttsx3

        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        preferred = ("zira", "david", "english", "india")
        for voice in voices:
            name = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
            if any(item in name for item in preferred):
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 118)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    try:
        threading.Thread(target=worker, daemon=True).start()
    except Exception as exc:
        return False, type(exc).__name__
    return True, "speaking"


class ISLRequestHandler(SimpleHTTPRequestHandler):
    predictor: Predictor
    web_root: Path

    def translate_path(self, path: str):
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])
        if path == "/":
            path = "/index.html"
        requested = (self.web_root / path.lstrip("/")).resolve()
        if not str(requested).startswith(str(self.web_root.resolve())):
            return str(self.web_root / "index.html")
        return str(requested)

    def do_POST(self):
        if self.path == "/speak":
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            ok, reason = speak_text(str(payload.get("text", "")))
            response = {"ok": ok, "reason": reason}
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.safe_write(body)
            return

        if self.path != "/predict":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            response = self.predictor.predict_data_url(payload["image"])
        except Exception as exc:
            response = {"ok": False, "reason": type(exc).__name__}

        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.safe_write(body)

    def safe_write(self, body: bytes):
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError):
            pass


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default=str(project_root / "models" / "isl_az_live_svm.joblib"))
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--not-detectable-confidence", type=float, default=0.60)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--distance-slack", type=float, default=1.3)
    parser.add_argument("--min-hand-area", type=float, default=0.015)
    args = parser.parse_args()

    ISLRequestHandler.web_root = project_root / "web"
    ISLRequestHandler.predictor = Predictor(
        Path(args.model),
        args.confidence,
        args.not_detectable_confidence,
        args.margin,
        args.distance_slack,
        args.min_hand_area,
    )
    server = ThreadingHTTPServer((args.host, args.port), ISLRequestHandler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        ISLRequestHandler.predictor.hands.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())                          
