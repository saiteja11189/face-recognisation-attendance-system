"""
Facial Feature Extractor (Embedder) using CNN / FaceNet / SFace.
Transforms aligned face crops into compact 128-dimensional latent vectors (embeddings)
for robust identity representation.
"""

from typing import List, Optional
import cv2
import numpy as np

from core import config
from core.detector import DetectedFace
from core.model_loader import ensure_models


class FaceEmbedder:
    """
    Deep CNN Embedding Extractor wrapping OpenCV FaceRecognizerSF.
    Computes 128-dimensional L2-normalized identity vectors.
    """

    def __init__(self, model_path: Optional[str] = None):
        ensure_models()
        self.model_path = str(model_path or config.SFACE_MODEL_PATH)
        self.recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

    def align_and_crop(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """
        Uses SFace's native landmark-based similarity transform to produce
        a canonical 112x112 aligned face crop.
        """
        aligned = self.recognizer.alignCrop(image, face.raw_face)
        return aligned

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts a 128-D feature vector from an aligned 112x112 face crop.

        Args:
            aligned_face: (112, 112, 3) BGR image of aligned face.

        Returns:
            1D numpy array of shape (128,) with float32 values, L2-normalized.
        """
        feature = self.recognizer.feature(aligned_face)
        # Ensure 1D shape
        embedding = feature.flatten().astype(np.float32)

        # Ensure unit length (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding

    def process_face(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """
        Convenience end-to-end pipeline:
        Takes the original image and a DetectedFace, performs landmark alignment,
        crops to 112x112, and computes the 128-D normalized embedding.
        """
        aligned = self.align_and_crop(image, face)
        return self.extract_embedding(aligned)

    @staticmethod
    def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Computes the centroid of multiple face embeddings (e.g. from a video burst).
        Filters outliers and normalizes the resulting mean vector to unit norm.

        Args:
            embeddings: List of 1D numpy arrays (128,).

        Returns:
            Normalized 1D array representing the composite identity vector.
        """
        if not embeddings:
            raise ValueError("Cannot average empty list of embeddings.")

        stacked = np.stack(embeddings, axis=0)  # Shape: (N, 128)
        mean_vec = np.mean(stacked, axis=0)

        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            mean_vec = mean_vec / norm

        return mean_vec.astype(np.float32)
