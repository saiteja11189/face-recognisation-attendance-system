"""
Unit tests for Face Preprocessor.
"""

import unittest
import numpy as np
from core.preprocessor import FacePreprocessor


class TestFacePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = FacePreprocessor(target_size=(112, 112))

    def test_clahe_enhancement(self):
        dark_img = np.ones((112, 112, 3), dtype=np.uint8) * 30
        enhanced = self.preprocessor.enhance_illumination(dark_img)

        self.assertEqual(enhanced.shape, dark_img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreaterEqual(np.mean(enhanced), np.mean(dark_img))

    def test_affine_alignment(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        right_eye = (80.0, 90.0)
        left_eye = (120.0, 110.0)

        aligned = self.preprocessor.align_face_affine(img, right_eye, left_eye, crop_box=(70, 70, 60, 60))
        self.assertEqual(aligned.shape, (112, 112, 3))
        self.assertEqual(aligned.dtype, np.uint8)

    def test_normalize_tensor(self):
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        norm = self.preprocessor.normalize_tensor(img)

        self.assertEqual(norm.shape, (112, 112, 3))
        self.assertEqual(norm.dtype, np.float32)
        self.assertTrue(np.isclose(norm[0, 0, 0], -127.5 / 128.0))


if __name__ == "__main__":
    unittest.main()
