"""
Camera and Video Stream wrapper module.
Manages OpenCV VideoCapture, FPS computation, frame resolution,
and camera hardware discovery with robust error recovery.
"""

import time
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np

from core import config


class Camera:
    """
    Robust VideoCapture wrapper with real-time FPS estimation and resolution controls.
    """

    def __init__(
        self,
        source: int = config.DEFAULT_CAMERA_INDEX,
        width: int = config.DEFAULT_FRAME_WIDTH,
        height: int = config.DEFAULT_FRAME_HEIGHT,
        fps: int = config.DEFAULT_FPS,
    ):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

        # FPS calculation state
        self._prev_time = 0.0
        self._fps = 0.0
        self._fps_alpha = 0.9  # Smoothing factor

    def start(self) -> bool:
        """Opens camera and sets resolution properties."""
        if self.cap is not None and self.cap.isOpened():
            return True

        # Use CAP_DSHOW on Windows for fast webcam initialization
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._prev_time = time.time()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame and updates current FPS rate."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Update smoothed FPS
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            current_instant_fps = 1.0 / dt
            self._fps = (self._fps_alpha * self._fps) + ((1.0 - self._fps_alpha) * current_instant_fps)

        return True, frame

    @property
    def fps(self) -> float:
        """Returns the current smoothed frame rate."""
        return round(self._fps, 1)

    def stop(self):
        """Releases the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def list_available_cameras(max_tested: int = 4) -> List[int]:
        """Scans and detects working webcam indexes on the host system."""
        available = []
        for i in range(max_tested):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    available.append(i)
                temp_cap.release()
        return available
