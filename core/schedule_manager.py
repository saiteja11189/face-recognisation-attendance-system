"""
Timetable and Schedule Manager module (Dynamic User-Defined Subjects).
Manages weekly subject timetables without pre-set hardcoded subjects.
Faculties register and define their own subjects and schedule slots.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

from core.database import Database

logger = logging.getLogger(__name__)

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


@dataclass
class PeriodSlot:
    """Represents a scheduled academic subject period."""
    id: Optional[int]
    day_of_week: str
    period_name: str
    subject_code: str
    subject_name: str
    department: str
    start_time: str
    end_time: str
    faculty_phone: str = ""

    @property
    def display_title(self) -> str:
        return f"{self.period_name}: {self.subject_name} ({self.start_time} - {self.end_time})"


class ScheduleManager:
    """
    Manages timetable schedules with dynamic faculty-defined subjects.
    No hardcoded subjects are injected.
    """

    def __init__(self, db: Database):
        self.db = db

    def add_slot(
        self,
        day_of_week: str,
        period_name: str,
        subject_code: str,
        subject_name: str,
        department: str,
        start_time: str,
        end_time: str,
        faculty_phone: str = "",
    ) -> int:
        """Adds a new subject period slot to the timetable."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO timetable (day_of_week, period_name, subject_code, subject_name, department, start_time, end_time, faculty_phone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                day_of_week.strip().capitalize(),
                period_name.strip(),
                subject_code.strip().upper(),
                subject_name.strip(),
                department.strip(),
                start_time.strip(),
                end_time.strip(),
                faculty_phone.strip(),
            ))
            slot_id = cursor.lastrowid
            conn.commit()
            return slot_id

    def delete_slot(self, slot_id: int) -> bool:
        """Removes a period slot from the timetable."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM timetable WHERE id = ?", (slot_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def get_all_slots(
        self,
        department: Optional[str] = None,
        faculty_phone: Optional[str] = None,
        subject_name: Optional[str] = None,
    ) -> List[dict]:
        """Fetches scheduled periods ordered by day and start time."""
        query = "SELECT * FROM timetable WHERE 1=1"
        params = []
        if department and department != "All":
            query += " AND (department = ? OR department = 'General')"
            params.append(department)
        if faculty_phone:
            query += " AND (faculty_phone = ? OR faculty_phone = '')"
            params.append(faculty_phone)
        if subject_name and subject_name != "All":
            query += " AND subject_name = ?"
            params.append(subject_name)

        query += " ORDER BY CASE day_of_week " \
                 "WHEN 'Monday' THEN 1 " \
                 "WHEN 'Tuesday' THEN 2 " \
                 "WHEN 'Wednesday' THEN 3 " \
                 "WHEN 'Thursday' THEN 4 " \
                 "WHEN 'Friday' THEN 5 " \
                 "WHEN 'Saturday' THEN 6 " \
                 "ELSE 7 END, start_time ASC"

        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_slots_by_day(
        self,
        day_of_week: str,
        department: Optional[str] = None,
        faculty_phone: Optional[str] = None,
        subject_name: Optional[str] = None,
    ) -> List[dict]:
        """Fetches timetable slots for a specific day of the week."""
        query = "SELECT * FROM timetable WHERE LOWER(day_of_week) = LOWER(?)"
        params = [day_of_week]
        if department and department != "All":
            query += " AND (department = ? OR department = 'General')"
            params.append(department)
        if faculty_phone:
            query += " AND (faculty_phone = ? OR faculty_phone = '')"
            params.append(faculty_phone)
        if subject_name and subject_name != "All":
            query += " AND subject_name = ?"
            params.append(subject_name)

        query += " ORDER BY start_time ASC"

        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_current_period(
        self,
        department: Optional[str] = None,
        faculty_phone: Optional[str] = None,
        subject_name: Optional[str] = None,
        dt: Optional[datetime] = None,
    ) -> Optional[dict]:
        """Determines active period based on wall-clock time."""
        now = dt or datetime.now()
        day_name = now.strftime("%A")
        time_str = now.strftime("%H:%M")

        slots = self.get_slots_by_day(
            day_of_week=day_name,
            department=department,
            faculty_phone=faculty_phone,
            subject_name=subject_name,
        )
        for slot in slots:
            if slot["start_time"] <= time_str <= slot["end_time"]:
                return slot
        return None

    def get_upcoming_period(
        self,
        department: Optional[str] = None,
        faculty_phone: Optional[str] = None,
        subject_name: Optional[str] = None,
        dt: Optional[datetime] = None,
    ) -> Optional[dict]:
        """Finds next upcoming period today."""
        now = dt or datetime.now()
        day_name = now.strftime("%A")
        time_str = now.strftime("%H:%M")

        slots = self.get_slots_by_day(
            day_of_week=day_name,
            department=department,
            faculty_phone=faculty_phone,
            subject_name=subject_name,
        )
        for slot in slots:
            if slot["start_time"] > time_str:
                return slot
        return None

    def get_all_distinct_subjects(
        self,
        department: Optional[str] = None,
        faculty_phone: Optional[str] = None,
    ) -> List[str]:
        """Returns unique subject names currently defined in the database."""
        slots = self.get_all_slots(department=department, faculty_phone=faculty_phone)
        subjects = sorted(list({s["subject_name"] for s in slots if s.get("subject_name")}))
        return subjects
