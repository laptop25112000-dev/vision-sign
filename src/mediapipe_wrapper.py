import os
import urllib.request
import time
from pathlib import Path

class PointWrapper:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class HandWrapper:
    def __init__(self, landmark):
        self.landmark = landmark

class ProcessResult:
    def __init__(self, multi_hand_landmarks):
        self.multi_hand_landmarks = multi_hand_landmarks

class ISLHandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        import mediapipe as mp
        self.mp = mp
        
        # Check if legacy solutions are available
        has_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        
        if has_legacy:
            print("Using legacy MediaPipe solutions API", flush=True)
            self.mode = "legacy"
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            print("Using modern MediaPipe Tasks API", flush=True)
            self.mode = "modern"
            
            project_root = Path(__file__).resolve().parents[1]
            model_dir = project_root / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "hand_landmarker.task"
            
            if not model_path.exists():
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                print(f"Downloading hand_landmarker.task model from Google to {model_path}...", flush=True)
                try:
                    urllib.request.urlretrieve(url, str(model_path))
                    print("Download complete.", flush=True)
                except Exception as exc:
                    raise SystemExit(
                        f"Failed to download hand_landmarker.task model: {exc}\n"
                        f"Please manually download the model from {url} and place it in {model_path}"
                    )
            
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Choose running mode based on static_image_mode
            running_mode = VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO
            
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
            )
            self.detector = HandLandmarker.create_from_options(options)
            self.running_mode = running_mode
            self.start_time = time.perf_counter()

    def process(self, rgb_image):
        if self.mode == "legacy":
            return self.hands.process(rgb_image)
        else:
            # Modern tasks API requires mp.Image
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect based on running mode
            if self.running_mode == self.mp.tasks.vision.RunningMode.VIDEO:
                # Video mode requires a monotonically increasing timestamp in milliseconds
                timestamp_ms = int((time.perf_counter() - self.start_time) * 1000)
                result = self.detector.detect_for_video(mp_image, timestamp_ms)
            else:
                result = self.detector.detect(mp_image)
                
            multi_hand_landmarks = []
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    landmark_list = [PointWrapper(pt.x, pt.y, pt.z) for pt in hand]
                    multi_hand_landmarks.append(HandWrapper(landmark_list))
                    
            return ProcessResult(multi_hand_landmarks if multi_hand_landmarks else None)

    def close(self):
        if self.mode == "legacy":
            self.hands.close()
        else:
            self.detector.close()

def draw_hands(frame, hand_landmarks, mp, cv2):
    """Draw landmarks with support for legacy MediaPipe drawing or an optimized custom fallback."""
    has_legacy_drawing = hasattr(mp, "solutions") and hasattr(mp.solutions, "drawing_utils")
    
    if has_legacy_drawing:
        for hand in hand_landmarks or []:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
            )
    else:
        # Custom drawing using cv2
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
                
            # Draw connections (lines)
            for start_idx, end_idx in connections:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], (200, 50, 50), 2)
                    
            # Draw joints (circles)
            for idx, (px, py) in enumerate(points):
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(frame, (px, py), 5, (50, 230, 50), -1)  # green tips
                else:
                    cv2.circle(frame, (px, py), 3, (240, 240, 240), -1)  # white joints
