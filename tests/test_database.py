"""
Unit tests for Database module.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from core.database import Database


class TestDatabase(unittest.TestCase):
    def test_database_crud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_attendance.db"
            db = Database(db_path=db_path)

            # Enrolled identity
            vec = np.random.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)

            ok = db.add_user(
                user_id="STU101",
                name="John Student",
                department="Computer Science",
                email="john@university.edu",
                embedding=vec,
                photo_path="data/faces/STU101.jpg",
            )
            self.assertTrue(ok)

            # Fetch user
            user = db.get_user("STU101")
            self.assertIsNotNone(user)
            self.assertEqual(user["name"], "John Student")
            self.assertEqual(user["department"], "Computer Science")
            self.assertTrue(np.allclose(user["embedding"], vec))

            # Test gallery caching
            gallery = db.get_gallery()
            self.assertIn("STU101", gallery)
            self.assertEqual(gallery["STU101"]["name"], "John Student")

            # Test attendance log
            rec_id = db.log_attendance(
                user_id="STU101",
                name="John Student",
                department="Computer Science",
                status="Present",
                confidence=95.5,
                metric="cosine",
            )
            self.assertGreater(rec_id, 0)

            # Query attendance
            records = db.get_attendance_records(user_id="STU101")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "Present")

            # Check last attendance
            last = db.get_last_attendance("STU101")
            self.assertIsNotNone(last)
            self.assertEqual(last["id"], rec_id)

            # Delete user
            deleted = db.delete_user("STU101")
            self.assertTrue(deleted)
            self.assertIsNone(db.get_user("STU101"))
            self.assertNotIn("STU101", db.get_gallery())


if __name__ == "__main__":
    unittest.main()
