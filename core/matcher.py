"""
Similarity Matching module.
Computes distance metrics (Cosine Similarity, Euclidean Distance / L2 Norm)
and matches query embeddings against registered gallery embeddings with threshold validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from core import config


@dataclass
class MatchResult:
    """Represents the outcome of comparing a query embedding against the gallery."""
    is_match: bool
    user_id: Optional[str]
    name: Optional[str]
    score: float           # Raw similarity or distance value
    confidence: float      # Normalized percentage [0.0, 100.0]
    metric: str            # 'cosine' or 'l2'
    top_matches: List[dict]  # Candidate rankings


class SimilarityMatcher:
    """
    High-speed vectorized embedding matcher with multi-metric support.
    """

    def __init__(
        self,
        metric: str = config.DEFAULT_METRIC,
        cosine_threshold: float = config.COSINE_SIMILARITY_THRESHOLD,
        l2_threshold: float = config.L2_DISTANCE_THRESHOLD,
    ):
        self.metric = metric.lower()
        self.cosine_threshold = float(cosine_threshold)
        self.l2_threshold = float(l2_threshold)

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1D vectors:
        cos(theta) = (u . v) / (||u|| * ||v||)
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    @staticmethod
    def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute Euclidean (L2) distance between two 1D vectors:
        ||u - v||_2 = sqrt(sum((u_i - v_i)^2))
        """
        return float(np.linalg.norm(u - v))

    def compute_confidence(self, score: float, metric: str) -> float:
        """
        Converts raw distance/similarity metric into an intuitive confidence percentage [0, 100].
        """
        if metric == "cosine":
            # Cosine similarity: range [-1, 1], higher is closer
            # At threshold -> ~50% confidence. At 1.0 -> 100% confidence.
            th = self.cosine_threshold
            if score < th:
                # Sub-threshold: scaled from 0% to 50%
                conf = max(0.0, (score / max(th, 0.001)) * 50.0)
            else:
                # Above threshold: scaled from 50% to 100%
                conf = 50.0 + ((score - th) / max(1.0 - th, 0.001)) * 50.0
            return float(np.clip(conf, 0.0, 100.0))
        else:
            # L2 distance: lower is closer. Perfect match = 0.0.
            # At threshold -> ~50% confidence. At 0.0 -> 100% confidence.
            th = self.l2_threshold
            if score > th:
                # Worse than threshold
                conf = max(0.0, 50.0 - ((score - th) / max(th, 0.001)) * 50.0)
            else:
                # Better than threshold
                conf = 50.0 + ((th - score) / max(th, 0.001)) * 50.0
            return float(np.clip(conf, 0.0, 100.0))

    def match(
        self,
        query_embedding: np.ndarray,
        gallery: Dict[str, dict],  # {user_id: {"name": ..., "embedding": np.ndarray, ...}}
        top_k: int = 3,
    ) -> MatchResult:
        """
        Compare query embedding against registered gallery using vectorized matrix operations.

        Args:
            query_embedding: 1D array of shape (128,)
            gallery: Mapping of user_id to user metadata including 'embedding'
            top_k: Number of candidate matches to return

        Returns:
            MatchResult containing match decision, recognized user, and ranking.
        """
        if not gallery:
            return MatchResult(
                is_match=False,
                user_id=None,
                name="Unknown",
                score=0.0,
                confidence=0.0,
                metric=self.metric,
                top_matches=[],
            )

        user_ids = list(gallery.keys())
        # Stack gallery embeddings into matrix (N, 128)
        matrix = np.stack([gallery[uid]["embedding"] for uid in user_ids], axis=0)

        # Ensure query vector is unit normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 1e-6:
            q = (query_embedding / query_norm).astype(np.float32)
        else:
            q = query_embedding.astype(np.float32)

        candidates = []

        if self.metric == "cosine":
            # Vectorized dot product
            scores = np.dot(matrix, q)  # Shape (N,)
            for idx, uid in enumerate(user_ids):
                sc = float(scores[idx])
                conf = self.compute_confidence(sc, "cosine")
                candidates.append({
                    "user_id": uid,
                    "name": gallery[uid].get("name", uid),
                    "department": gallery[uid].get("department", "General"),
                    "score": sc,
                    "confidence": conf,
                })
            # Higher score is better
            candidates.sort(key=lambda c: c["score"], reverse=True)
            best = candidates[0]
            is_match = best["score"] >= self.cosine_threshold

        else:  # L2 distance
            # Vectorized Euclidean distance
            diffs = matrix - q  # Shape (N, 128)
            distances = np.linalg.norm(diffs, axis=1)  # Shape (N,)
            for idx, uid in enumerate(user_ids):
                dist = float(distances[idx])
                conf = self.compute_confidence(dist, "l2")
                candidates.append({
                    "user_id": uid,
                    "name": gallery[uid].get("name", uid),
                    "department": gallery[uid].get("department", "General"),
                    "score": dist,
                    "confidence": conf,
                })
            # Lower distance is better
            candidates.sort(key=lambda c: c["score"], reverse=False)
            best = candidates[0]
            is_match = best["score"] <= self.l2_threshold

        return MatchResult(
            is_match=is_match,
            user_id=best["user_id"] if is_match else None,
            name=best["name"] if is_match else "Unknown",
            score=best["score"],
            confidence=best["confidence"],
            metric=self.metric,
            top_matches=candidates[:top_k],
        )
