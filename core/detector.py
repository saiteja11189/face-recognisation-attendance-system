"""
Face Detection module using OpenCV DNN YuNet.
Detects faces and extracts 5 key facial landmarks (eyes, nose, mouth corners)
with high speed and robust pose/illumination invariance.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from core import config
from core.model_loader import ensure_models


@dataclass
class DetectedFace:
    """Represents a detected face with coordinates, landmarks, and confidence score."""
    box: Tuple[int, int, int, int]  # x, y, width, height
    landmarks: np.ndarray  # shape (5, 2): right_eye, left_eye, nose, right_mouth, left_mouth
    confidence: float
    raw_face: np.ndarray  # raw 15-element array required for SFace alignment

    @property
    def x(self) -> int:
        return self.box[0]

    @property
    def y(self) -> int:
        return self.box[1]

    @property
    def width(self) -> int:
        return self.box[2]

    @property
    def height(self) -> int:
        return self.box[3]

    @property
    def right_eye(self) -> Tuple[float, float]:
        return float(self.landmarks[0][0]), float(self.landmarks[0][1])

    @property
    def left_eye(self) -> Tuple[float, float]:
        return float(self.landmarks[1][0]), float(self.landmarks[1][1])


class FaceDetector:
    """
    High-performance Deep Learning Face Detector wrapping OpenCV FaceDetectorYN.
    Automatically handles dynamic input image resolutions and filters detections
    based on confidence and non-maximum suppression (NMS).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = config.DETECTION_SCORE_THRESHOLD,
        nms_threshold: float = config.DETECTION_NMS_THRESHOLD,
        top_k: int = config.DETECTION_TOP_K,
    ):
        ensure_models()
        self.model_path = str(model_path or config.YUNET_MODEL_PATH)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)

        # Initialize detector with a default size; will dynamically adjust to input frames
        self._current_size = (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            self._current_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def set_score_threshold(self, threshold: float):
        """Update detection confidence threshold dynamically."""
        self.score_threshold = float(threshold)
        self.detector.setScoreThreshold(self.score_threshold)

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in an input BGR or RGB image.

        Args:
            image: numpy ndarray (H, W, 3) image in BGR or RGB format.

        Returns:
            List of DetectedFace instances, sorted by bounding box area (largest first).
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # Update input size if frame dimensions have changed
        if (w, h) != self._current_size:
            self._current_size = (w, h)
            self.detector.setInputSize((w, h))

        # Perform forward inference
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        results: List[DetectedFace] = []
        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, bw, bh = map(int, face[0:4])
            score = float(face[14])

            # Clamp bounding box within image boundaries
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            if bw <= 0 or bh <= 0:
                continue

            # Extract 5 landmarks
            landmarks = np.array([
                [face[4], face[5]],    # Right eye
                [face[6], face[7]],    # Left eye
                [face[8], face[9]],    # Nose tip
                [face[10], face[11]],  # Right corner mouth
                [face[12], face[13]],  # Left corner mouth
            ], dtype=np.float32)

            results.append(DetectedFace(
                box=(x, y, bw, bh),
                landmarks=landmarks,
                confidence=score,
                raw_face=face.astype(np.float32),
            ))

        # Sort detected faces: largest face first (most relevant face in frame)
        results.sort(key=lambda f: f.width * f.height, reverse=True)
        return results

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: List[DetectedFace],
        draw_landmarks: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and facial landmarks on an image."""
        annotated = image.copy()
        for face in faces:
            x, y, w, h = face.box
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

            # Draw confidence text badge
            badge_text = f"Face: {face.confidence * 100:.1f}%"
            cv2.putText(
                annotated,
                badge_text,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            # Draw landmarks
            if draw_landmarks:
                for lx, ly in face.landmarks:
                    cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        return annotated
