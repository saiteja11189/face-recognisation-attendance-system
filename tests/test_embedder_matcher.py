"""
Unit tests for Embedder and Similarity Matcher.
"""

import unittest
import numpy as np
from core.matcher import SimilarityMatcher
from core.embedder import FaceEmbedder


class TestEmbedderMatcher(unittest.TestCase):
    def test_similarity_math(self):
        matcher = SimilarityMatcher()

        # Identical vectors
        v1 = np.random.randn(128).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        cos_sim = matcher.cosine_similarity(v1, v1)
        self.assertTrue(np.isclose(cos_sim, 1.0, atol=1e-5))

        l2_dist = matcher.euclidean_distance(v1, v1)
        self.assertTrue(np.isclose(l2_dist, 0.0, atol=1e-5))

        # Orthogonal vectors
        v2 = np.zeros(128, dtype=np.float32)
        v3 = np.zeros(128, dtype=np.float32)
        v2[0] = 1.0
        v3[1] = 1.0

        cos_ortho = matcher.cosine_similarity(v2, v3)
        self.assertTrue(np.isclose(cos_ortho, 0.0, atol=1e-5))

        l2_ortho = matcher.euclidean_distance(v2, v3)
        self.assertTrue(np.isclose(l2_ortho, np.sqrt(2.0), atol=1e-5))

    def test_gallery_matching_cosine(self):
        matcher = SimilarityMatcher(metric="cosine", cosine_threshold=0.38)

        # Mock gallery
        vec_alice = np.random.randn(128).astype(np.float32)
        vec_alice /= np.linalg.norm(vec_alice)

        vec_bob = np.random.randn(128).astype(np.float32)
        vec_bob /= np.linalg.norm(vec_bob)

        gallery = {
            "EMP001": {"name": "Alice Smith", "department": "AI Lab", "embedding": vec_alice},
            "EMP002": {"name": "Bob Jones", "department": "DevOps", "embedding": vec_bob},
        }

        # Query with slightly perturbed Alice vector
        noise = np.random.randn(128).astype(np.float32) * 0.05
        query_alice = vec_alice + noise
        query_alice /= np.linalg.norm(query_alice)

        res = matcher.match(query_alice, gallery)
        self.assertTrue(res.is_match)
        self.assertEqual(res.user_id, "EMP001")
        self.assertEqual(res.name, "Alice Smith")
        self.assertGreater(res.score, 0.80)
        self.assertGreater(res.confidence, 75.0)

        # Query with orthogonal un-enrolled vector
        vec_stranger = np.random.randn(128).astype(np.float32)
        vec_stranger -= np.dot(vec_stranger, vec_alice) * vec_alice
        vec_stranger -= np.dot(vec_stranger, vec_bob) * vec_bob
        vec_stranger /= np.linalg.norm(vec_stranger)

        res_unknown = matcher.match(vec_stranger, gallery)
        self.assertFalse(res_unknown.is_match)
        self.assertEqual(res_unknown.name, "Unknown")

    def test_gallery_matching_l2(self):
        matcher = SimilarityMatcher(metric="l2", l2_threshold=1.10)

        vec_charlie = np.random.randn(128).astype(np.float32)
        vec_charlie /= np.linalg.norm(vec_charlie)

        gallery = {
            "EMP003": {"name": "Charlie", "department": "Design", "embedding": vec_charlie},
        }

        res = matcher.match(vec_charlie, gallery)
        self.assertTrue(res.is_match)
        self.assertEqual(res.user_id, "EMP003")
        self.assertTrue(np.isclose(res.score, 0.0, atol=1e-5))
        self.assertTrue(np.isclose(res.confidence, 100.0, atol=1e-2))

    def test_embedder_average(self):
        v1 = np.ones(128, dtype=np.float32)
        v2 = np.ones(128, dtype=np.float32)
        avg = FaceEmbedder.average_embeddings([v1, v2])

        self.assertEqual(avg.shape, (128,))
        self.assertTrue(np.isclose(np.linalg.norm(avg), 1.0))


if __name__ == "__main__":
    unittest.main()
