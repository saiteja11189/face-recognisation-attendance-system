"""
Unit tests for Timetable and Subject-wise Attendance with dynamic user-defined subjects.
"""

from datetime import datetime
from pathlib import Path
import tempfile
import unittest
import numpy as np

from core.database import Database
from core.attendance_manager import AttendanceManager
from core.schedule_manager import ScheduleManager


class TestScheduleAndSubjectAttendance(unittest.TestCase):
    def test_schedule_management(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_sched.db"
            db = Database(db_path=db_path)
            sched = ScheduleManager(db=db)

            # Initially empty (no hardcoded subjects)
            slots = sched.get_all_slots()
            self.assertEqual(len(slots), 0)

            # Add user-defined subject slot
            slot_id = sched.add_slot(
                day_of_week="Monday",
                period_name="Period 1",
                subject_code="CSE316",
                subject_name="Operating Systems",
                department="Computer Science",
                start_time="09:00",
                end_time="10:00",
                faculty_phone="9876543210",
            )
            self.assertGreater(slot_id, 0)

            # Query slots by day
            mon_slots = sched.get_slots_by_day("Monday")
            self.assertEqual(len(mon_slots), 1)
            self.assertEqual(mon_slots[0]["subject_name"], "Operating Systems")

            # Test delete slot
            deleted = sched.delete_slot(slot_id)
            self.assertTrue(deleted)
            self.assertEqual(len(sched.get_slots_by_day("Monday")), 0)

    def test_period_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_sched.db"
            db = Database(db_path=db_path)
            sched = ScheduleManager(db=db)

            # Add faculty subject slot
            sched.add_slot(
                day_of_week="Monday",
                period_name="Period 1",
                subject_code="CSE316",
                subject_name="Operating Systems",
                department="Computer Science",
                start_time="09:00",
                end_time="10:00",
                faculty_phone="9876543210",
            )

            # Mock a Monday at 09:30 AM (Inside Period 1)
            mock_dt = datetime(2026, 11, 16, 9, 30, 0)  # Monday
            curr = sched.get_current_period(
                department="Computer Science",
                faculty_phone="9876543210",
                subject_name="Operating Systems",
                dt=mock_dt,
            )
            self.assertIsNotNone(curr)
            self.assertEqual(curr["subject_name"], "Operating Systems")
            self.assertEqual(curr["period_name"], "Period 1")

    def test_multi_subject_attendance_same_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_sched.db"
            db = Database(db_path=db_path)
            mgr = AttendanceManager(db=db, cooldown_minutes=30)

            # Register student
            vec = np.random.randn(128).astype(np.float32)
            db.add_user("STU501", "Emma Watson", "Computer Science", vec)

            dt_period1 = datetime(2026, 11, 17, 9, 5, 0)
            dt_period2 = datetime(2026, 11, 17, 10, 5, 0)

            # 1. Mark Period 1 (Operating Systems)
            ok1, msg1, r1 = mgr.mark_attendance(
                user_id="STU501",
                confidence=95.0,
                metric="cosine",
                subject="Operating Systems",
                period="Period 1",
                period_start_time="09:00",
                now=dt_period1,
            )
            self.assertTrue(ok1)
            self.assertEqual(r1["subject"], "Operating Systems")
            self.assertEqual(r1["status"], "Present")

            # 2. Duplicate mark for same subject -> blocked by cooldown
            ok_dup, msg_dup, _ = mgr.mark_attendance(
                user_id="STU501",
                confidence=96.0,
                metric="cosine",
                subject="Operating Systems",
                period="Period 1",
                period_start_time="09:00",
                now=datetime(2026, 11, 17, 9, 10, 0),
            )
            self.assertFalse(ok_dup)
            self.assertIn("Already marked for Operating Systems", msg_dup)

            # 3. Mark Period 2 (Computer Networks) 1 hour later -> SUCCEEDS!
            ok2, msg2, r2 = mgr.mark_attendance(
                user_id="STU501",
                confidence=94.5,
                metric="cosine",
                subject="Computer Networks",
                period="Period 2",
                period_start_time="10:00",
                now=dt_period2,
            )
            self.assertTrue(ok2)
            self.assertEqual(r2["subject"], "Computer Networks")


if __name__ == "__main__":
    unittest.main()
