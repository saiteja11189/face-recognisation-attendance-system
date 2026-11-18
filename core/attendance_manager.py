"""
Attendance Manager module with Subject and Period awareness.
Enforces duplicate prevention scoped per subject/period, tracks punctuality against period start times,
computes subject-wise analytics, and exports comprehensive multi-subject reports to CSV and Excel.
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from core import config
from core.database import Database

logger = logging.getLogger(__name__)


class AttendanceManager:
    """
    Manages attendance logging, period-specific cooldown verification,
    subject-wise analytics, and multi-sheet report exports.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        cooldown_minutes: int = config.ATTENDANCE_COOLDOWN_MINUTES,
        grace_period_minutes: int = 10,
    ):
        self.db = db or Database()
        self.cooldown_minutes = int(cooldown_minutes)
        self.grace_period_minutes = int(grace_period_minutes)

    def determine_status_for_period(
        self,
        current_dt: datetime,
        period_start_time: Optional[str] = None,
    ) -> str:
        """
        Determines if check-in is On Time (Present) or Late based on the period's start time.
        If period_start_time is "10:00", arrival up to "10:10" is Present; after is Late.
        """
        if not period_start_time:
            # Fallback to standard morning cutoff
            cutoff = current_dt.replace(
                hour=config.WORK_START_HOUR,
                minute=config.WORK_START_MINUTE,
                second=0,
                microsecond=0
            )
            return "Late" if current_dt > cutoff else "Present"

        try:
            parts = list(map(int, period_start_time.split(":")))
            period_start = current_dt.replace(
                hour=parts[0],
                minute=parts[1],
                second=0,
                microsecond=0
            )
            cutoff = period_start + timedelta(minutes=self.grace_period_minutes)
            return "Late" if current_dt > cutoff else "Present"
        except Exception:
            return "Present"

    def mark_attendance(
        self,
        user_id: str,
        confidence: float,
        metric: str,
        subject: str = "General",
        period: str = "General",
        period_start_time: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        Marks attendance for a student/employee in a specific subject period.
        Deduplication is scoped per subject: a student can attend multiple distinct periods in a day!

        Returns:
            (success: bool, message: str, record: Optional[dict])
        """
        current_time = now or datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")

        user = self.db.get_user(user_id)
        if not user:
            return False, f"User {user_id} not registered in system.", None

        # Check last attendance for this user IN THIS SPECIFIC SUBJECT on this day
        last_rec = self.db.get_last_attendance(
            user_id=user_id,
            date_str=date_str,
            subject=subject,
            period=period,
        )
        if last_rec:
            last_ts = datetime.strptime(last_rec["timestamp"], "%Y-%m-%d %H:%M:%S")
            diff = current_time - last_ts
            if diff < timedelta(minutes=self.cooldown_minutes):
                mins_left = self.cooldown_minutes - int(diff.total_seconds() / 60)
                msg = f"Already marked for {subject} at {last_ts.strftime('%H:%M:%S')} (cooldown active: {mins_left}m remaining)"
                return False, msg, last_rec

        # Determine status (Present vs Late)
        status = self.determine_status_for_period(current_time, period_start_time=period_start_time)

        # Log into database
        rec_id = self.db.log_attendance(
            user_id=user["user_id"],
            name=user["name"],
            department=user["department"],
            status=status,
            confidence=confidence,
            metric=metric,
            subject=subject,
            period=period,
            timestamp=current_time,
        )

        record = {
            "id": rec_id,
            "user_id": user["user_id"],
            "name": user["name"],
            "department": user["department"],
            "date": date_str,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "confidence": confidence,
            "metric": metric,
            "subject": subject,
            "period": period,
        }

        # Auto-export updated logs
        try:
            self.export_to_csv(date_str)
        except Exception as e:
            logger.warning(f"Auto-export CSV failed: {e}")

        msg = f"Marked {status} for {subject}! {user['name']} ({user['user_id']})"
        return True, msg, record

    def get_daily_summary(
        self,
        date_str: Optional[str] = None,
        subject: Optional[str] = None,
        period: Optional[str] = None,
        department: Optional[str] = None,
    ) -> dict:
        """
        Calculates attendance KPIs with subject and period filtering.
        """
        query_date = date_str or datetime.now().strftime("%Y-%m-%d")
        all_users = self.db.get_all_users()
        if department and department != "All":
            all_users = [u for u in all_users if u.get("department") == department or u.get("department") == "General"]

        records = self.db.get_attendance_records(
            date_str=query_date,
            department=department,
            subject=subject if subject != "All" else None,
            period=period if period != "All" else None,
            limit=10000,
        )

        present_uids = {rec["user_id"] for rec in records}
        total_registered = len(all_users)
        total_present = len(present_uids)
        total_absent = max(0, total_registered - total_present)
        attendance_rate = (total_present / total_registered * 100.0) if total_registered > 0 else 0.0

        present_count = sum(1 for r in records if r["status"] == "Present")
        late_count = sum(1 for r in records if r["status"] == "Late")

        # Subject breakdown
        subject_breakdown = {}
        for r in records:
            sb = r.get("subject", "General")
            if sb not in subject_breakdown:
                subject_breakdown[sb] = {"count": 0, "on_time": 0, "late": 0}
            subject_breakdown[sb]["count"] += 1
            if r["status"] == "Present":
                subject_breakdown[sb]["on_time"] += 1
            else:
                subject_breakdown[sb]["late"] += 1

        absent_users = [u for u in all_users if u["user_id"] not in present_uids]

        return {
            "date": query_date,
            "subject": subject or "All",
            "period": period or "All",
            "total_registered": total_registered,
            "total_present": total_present,
            "total_absent": total_absent,
            "attendance_rate": round(attendance_rate, 1),
            "present_count": present_count,
            "late_count": late_count,
            "subject_breakdown": subject_breakdown,
            "absent_users": absent_users,
            "records": records,
        }

    def get_student_subject_analytics(self, department: Optional[str] = None) -> List[dict]:
        """
        Computes overall and per-subject attendance percentages for all enrolled students.
        Flags students with attendance below 75% threshold.
        """
        all_users = self.db.get_all_users()
        if department and department != "All":
            all_users = [u for u in all_users if u.get("department") == department or u.get("department") == "General"]

        all_records = self.db.get_attendance_records(limit=50000)

        # Distinct dates/sessions per subject
        subject_sessions: Dict[str, set] = {}
        # Records per (user_id, subject)
        user_subject_attended: Dict[Tuple[str, str], int] = {}
        total_user_attended: Dict[str, int] = {}

        for r in all_records:
            sb = r.get("subject", "General")
            uid = r["user_id"]
            d = r["date"]

            if sb not in subject_sessions:
                subject_sessions[sb] = set()
            subject_sessions[sb].add(d)

            pair = (uid, sb)
            user_subject_attended[pair] = user_subject_attended.get(pair, 0) + 1
            total_user_attended[uid] = total_user_attended.get(uid, 0) + 1

        total_possible_sessions = sum(len(dates) for dates in subject_sessions.values()) or 1

        results = []
        for user in all_users:
            uid = user["user_id"]
            name = user["name"]
            dept = user.get("department", "General")
            attended_total = total_user_attended.get(uid, 0)

            overall_pct = (attended_total / total_possible_sessions) * 100.0 if total_possible_sessions > 0 else 0.0

            # Breakdown per subject
            subj_details = {}
            for sb, dates in subject_sessions.items():
                conducted = len(dates)
                attended = user_subject_attended.get((uid, sb), 0)
                rate = (attended / conducted * 100.0) if conducted > 0 else 0.0
                subj_details[sb] = {
                    "attended": attended,
                    "conducted": conducted,
                    "percentage": round(rate, 1),
                }

            is_shortage = overall_pct < 75.0

            results.append({
                "user_id": uid,
                "name": name,
                "department": dept,
                "total_attended": attended_total,
                "overall_percentage": round(overall_pct, 1),
                "is_shortage": is_shortage,
                "subject_breakdown": subj_details,
            })

        return results

    def export_to_csv(self, date_str: Optional[str] = None, output_path: Optional[Path] = None) -> Path:
        """Exports attendance records to CSV."""
        query_date = date_str or datetime.now().strftime("%Y-%m-%d")
        records = self.db.get_attendance_records(date_str=query_date, limit=10000)

        dest = output_path or (config.ATTENDANCE_LOGS_DIR / f"attendance_{query_date}.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            df = pd.DataFrame(columns=[
                "ID", "User ID", "Name", "Department", "Date", "Timestamp", "Status", "Confidence (%)", "Metric", "Subject", "Period"
            ])
        else:
            df = pd.DataFrame(records)
            df.columns = [col.replace("_", " ").title() for col in df.columns]

        df.to_csv(dest, index=False)
        return dest

    def export_to_excel(self, date_str: Optional[str] = None, output_path: Optional[Path] = None) -> Path:
        """Exports multi-sheet styled Excel report with Subject and Period details."""
        query_date = date_str or datetime.now().strftime("%Y-%m-%d")
        summary = self.get_daily_summary(query_date)
        records = summary["records"]

        dest = output_path or (config.ATTENDANCE_LOGS_DIR / f"attendance_{query_date}.xlsx")
        dest.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(dest, engine="xlsxwriter") as writer:
            # Sheet 1: Detailed Logs
            if records:
                df_records = pd.DataFrame(records)
                cols_order = ["id", "user_id", "name", "department", "subject", "period", "timestamp", "status", "confidence", "metric"]
                df_records = df_records[[c for c in cols_order if c in df_records.columns]]
                df_records.columns = [c.replace("_", " ").title() for c in df_records.columns]
            else:
                df_records = pd.DataFrame(columns=["User ID", "Name", "Department", "Subject", "Period", "Timestamp", "Status", "Confidence"])

            df_records.to_excel(writer, sheet_name="Subject Attendance Logs", index=False)

            # Sheet 2: Daily KPI Summary
            summary_data = [
                {"Metric": "Date", "Value": query_date},
                {"Metric": "Total Registered", "Value": summary["total_registered"]},
                {"Metric": "Total Present", "Value": summary["total_present"]},
                {"Metric": "Total Absent", "Value": summary["total_absent"]},
                {"Metric": "Attendance Rate", "Value": f"{summary['attendance_rate']}%"},
                {"Metric": "On Time (Present)", "Value": summary["present_count"]},
                {"Metric": "Late Check-ins", "Value": summary["late_count"]},
            ]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary KPIs", index=False)

            # Sheet 3: Subject Breakdown
            subj_rows = []
            for sb, data in summary["subject_breakdown"].items():
                subj_rows.append({
                    "Subject": sb,
                    "Total Check-ins": data["count"],
                    "On Time": data["on_time"],
                    "Late": data["late"],
                })
            pd.DataFrame(subj_rows or [{"Subject": "None", "Total Check-ins": 0, "On Time": 0, "Late": 0}]).to_excel(
                writer, sheet_name="Subject Breakdown", index=False
            )

        return dest
