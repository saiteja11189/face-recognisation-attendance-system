"""
Face Preprocessing module.
Handles face alignment (affine transformation based on eye coordinates),
illumination equalization (CLAHE on luminance channel), and normalization.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from core.detector import DetectedFace


class FacePreprocessor:
    """
    Normalizes, enhances, and aligns face crops before feature extraction.
    Ensures invariant representation across head tilts, lighting variations, and scales.
    """

    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def align_face_affine(
        self,
        image: np.ndarray,
        right_eye: Tuple[float, float],
        left_eye: Tuple[float, float],
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Custom affine alignment using eye landmark coordinates.
        Calculates rotation angle and affine transformation matrix so that
        the line connecting both eyes is strictly horizontal.

        Args:
            image: Source image.
            right_eye: (x, y) coordinates of the subject's right eye.
            left_eye: (x, y) coordinates of the subject's left eye.
            crop_box: Optional (x, y, w, h) bounding box to constrain rotation center.

        Returns:
            Rotated and aligned face crop resized to target_size.
        """
        # Calculate angle of the eyes line relative to horizontal
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Center of rotation is midpoint between eyes
        eyes_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)

        # Get rotation matrix (scale = 1.0)
        m = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        # Warp whole image
        h, w = image.shape[:2]
        rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # If bounding box is given, crop around the rotated face
        if crop_box is not None:
            bx, by, bw, bh = crop_box
            # Pad crop slightly for context
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            cropped = rotated[y1:y2, x1:x2]
            if cropped.size > 0:
                return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

        return cv2.resize(rotated, self.target_size, interpolation=cv2.INTER_AREA)

    def enhance_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
        of LAB color space. Improves facial contrast in harsh shadows or dim lighting.
        """
        if image is None or image.size == 0:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def extract_crop(
        self,
        image: np.ndarray,
        face: DetectedFace,
        pad_ratio: float = 0.15
    ) -> np.ndarray:
        """
        Extracts a rectangular crop around the detected face with optional safety padding.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = face.box
        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)

        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w, x + bw + px)
        y2 = min(h, y + bh + py)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image[y:y + bh, x:x + bw]
        return crop

    def normalize_tensor(self, face_image: np.ndarray) -> np.ndarray:
        """
        Standardizes face crop pixels to range [-1.0, 1.0] for CNN ingestion.
        """
        float_img = face_image.astype(np.float32)
        normalized = (float_img - 127.5) / 128.0
        return normalized
