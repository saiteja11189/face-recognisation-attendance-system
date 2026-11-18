"""
Unit tests for Attendance Manager module.
"""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core.database import Database
from core.attendance_manager import AttendanceManager


class TestAttendanceManager(unittest.TestCase):
    def test_attendance_marking_and_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_att.db"
            db = Database(db_path=db_path)
            mgr = AttendanceManager(db=db, cooldown_minutes=30)

            # Enroll test user
            vec = np.random.randn(128).astype(np.float32)
            db.add_user("EMP99", "Sarah Connor", "Security", vec)

            now = datetime(2026, 11, 17, 8, 45, 0)

            # 1. First check-in: Should succeed (Status: Present, before 9:15)
            ok, msg, rec = mgr.mark_attendance("EMP99", confidence=96.0, metric="cosine", now=now)
            self.assertTrue(ok)
            self.assertEqual(rec["status"], "Present")
            self.assertIn("Marked Present", msg)

            # 2. Immediate duplicate check-in (5 mins later): Should be blocked by cooldown
            later = now + timedelta(minutes=5)
            ok2, msg2, rec2 = mgr.mark_attendance("EMP99", confidence=97.0, metric="cosine", now=later)
            self.assertFalse(ok2)
            self.assertIn("cooldown active", msg2)

            # 3. Check-in after cooldown (35 mins later, 9:20 AM): Should succeed (Status: Late)
            after_cooldown = now + timedelta(minutes=35)
            ok3, msg3, rec3 = mgr.mark_attendance("EMP99", confidence=95.0, metric="cosine", now=after_cooldown)
            self.assertTrue(ok3)
            self.assertEqual(rec3["status"], "Late")

    def test_daily_summary_and_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_att.db"
            db = Database(db_path=db_path)
            mgr = AttendanceManager(db=db)

            # Enroll 2 users
            db.add_user("U1", "User One", "CS", np.zeros(128, dtype=np.float32))
            db.add_user("U2", "User Two", "Math", np.zeros(128, dtype=np.float32))

            # Mark U1 present
            mgr.mark_attendance("U1", confidence=92.0, metric="cosine")

            summary = mgr.get_daily_summary()
            self.assertEqual(summary["total_registered"], 2)
            self.assertEqual(summary["total_present"], 1)
            self.assertEqual(summary["total_absent"], 1)
            self.assertEqual(summary["attendance_rate"], 50.0)
            self.assertEqual(len(summary["absent_users"]), 1)
            self.assertEqual(summary["absent_users"][0]["user_id"], "U2")

            # Test CSV export
            csv_path = Path(tmpdir) / "export.csv"
            mgr.export_to_csv(output_path=csv_path)
            self.assertTrue(csv_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)

            # Test Excel export
            xlsx_path = Path(tmpdir) / "export.xlsx"
            mgr.export_to_excel(output_path=xlsx_path)
            self.assertTrue(xlsx_path.exists())
            self.assertGreater(xlsx_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
