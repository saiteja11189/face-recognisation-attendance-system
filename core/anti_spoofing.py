"""
Anti-Spoofing and Liveness Verification module.
Protects the attendance system against presentation attacks (printed photos, phone screen replays).
Uses Laplacian texture sharpness and landmark micro-motion dynamics across frames.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np

from core.detector import DetectedFace


@dataclass
class LivenessResult:
    """Represents the liveness verification outcome."""
    is_live: bool
    confidence: float  # [0.0, 100.0]
    reason: str
    sharpness_score: float
    motion_score: float


class AntiSpoofingDetector:
    """
    Lightweight, real-time passive liveness detector.
    Analyzes:
    1. High-frequency texture sharpness (detects printed paper blur or screen pixelation).
    2. Landmark micro-motion variance over a sliding temporal window (detects static photos).
    """

    def __init__(
        self,
        history_length: int = 10,
        min_sharpness: float = 40.0,
        min_motion_variance: float = 0.80,
    ):
        self.history_length = history_length
        self.min_sharpness = min_sharpness
        self.min_motion_variance = min_motion_variance

        # Sliding window of recent landmark positions for motion tracking: {track_id: deque}
        self._landmark_history: Deque[np.ndarray] = deque(maxlen=history_length)

    def compute_texture_sharpness(self, face_crop: np.ndarray) -> float:
        """
        Computes the Laplacian variance of the face crop.
        Low values (<40) indicate out-of-focus prints or low-res display replays.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    def compute_motion_variance(self, current_landmarks: np.ndarray) -> float:
        """
        Tracks micro-movements of facial landmarks across recent frames.
        Natural human faces exhibit involuntary saccades and breathing micro-shifts (variance > 0.8).
        A static printed photo held in hand has near-zero relative landmark variation.
        """
        self._landmark_history.append(current_landmarks.copy())

        if len(self._landmark_history) < 3:
            # Not enough frames yet, grant benefit of doubt
            return 1.5

        history_array = np.array(list(self._landmark_history))  # Shape: (T, 5, 2)
        # Compute standard deviation of landmarks across time window
        std_per_point = np.std(history_array, axis=0)  # Shape: (5, 2)
        mean_motion = float(np.mean(std_per_point))
        return mean_motion

    def reset_history(self):
        """Clears sliding motion window."""
        self._landmark_history.clear()

    def check_liveness(self, face_crop: np.ndarray, face: DetectedFace) -> LivenessResult:
        """
        Evaluates face liveness based on texture sharpness and landmark dynamics.
        """
        sharpness = self.compute_texture_sharpness(face_crop)
        motion = self.compute_motion_variance(face.landmarks)

        # Check conditions
        sharpness_ok = sharpness >= self.min_sharpness
        motion_ok = motion >= self.min_motion_variance

        is_live = sharpness_ok and motion_ok
        confidence = 0.0
        reason = "Live Subject"

        if not sharpness_ok:
            is_live = False
            reason = f"Texture blur / screen replay detected (score: {sharpness:.1f} < {self.min_sharpness})"
            confidence = max(10.0, (sharpness / self.min_sharpness) * 40.0)
        elif not motion_ok and len(self._landmark_history) >= 5:
            is_live = False
            reason = f"Static photo spoof detected (motion: {motion:.2f} < {self.min_motion_variance})"
            confidence = max(10.0, (motion / self.min_motion_variance) * 40.0)
        else:
            is_live = True
            # Compute confidence score
            s_factor = min(1.0, sharpness / 100.0)
            m_factor = min(1.0, motion / 2.0)
            confidence = float(min(100.0, 60.0 + (s_factor * 20.0) + (m_factor * 20.0)))

        return LivenessResult(
            is_live=is_live,
            confidence=round(confidence, 1),
            reason=reason,
            sharpness_score=round(sharpness, 1),
            motion_score=round(motion, 2),
        )
