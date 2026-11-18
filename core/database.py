"""
Database module using SQLite.
Stores registered user identities, 128D facial embedding vectors,
faculty profiles with mobile numbers, timetable schedules, persistent OTP vault,
and subject attendance records.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, Generator, List, Optional
import numpy as np

from core import config


class Database:
    """
    SQLite Database Manager for Face Recognition Attendance.
    Handles user enrollment, faculty auth profiles, weekly timetables,
    persistent OTP verification vault, and period-specific subject attendance transaction logs.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or config.DB_PATH)
        self._init_tables()
        self._gallery_cache: Optional[Dict[str, dict]] = None

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """Creates user, faculty, otp_vault, attendance, and timetable tables with migrations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Persistent OTP Vault for strict verification across sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otp_vault (
                    phone_number TEXT PRIMARY KEY,
                    otp_code TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Faculty table for Staff ID & Password authentication
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faculty (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    faculty_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    phone_number TEXT NOT NULL DEFAULT '',
                    department TEXT NOT NULL,
                    subject_name TEXT NOT NULL,
                    subject_code TEXT NOT NULL,
                    password_hash TEXT DEFAULT '',
                    password_salt TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)

            # Users (Students/Staff) table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL DEFAULT 'General',
                    email TEXT DEFAULT '',
                    embedding BLOB NOT NULL,
                    photo_path TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)

            # Attendance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metric TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT 'General',
                    period TEXT NOT NULL DEFAULT 'General',
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)

            # Safe column migration for existing databases
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='faculty'")
            fac_row = cursor.fetchone()
            if fac_row and "phone_number TEXT UNIQUE" in fac_row[0]:
                cursor.execute("""
                    CREATE TABLE faculty_mig (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        faculty_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        phone_number TEXT NOT NULL DEFAULT '',
                        department TEXT NOT NULL,
                        subject_name TEXT NOT NULL,
                        subject_code TEXT NOT NULL,
                        password_hash TEXT DEFAULT '',
                        password_salt TEXT DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    INSERT INTO faculty_mig (id, faculty_id, name, phone_number, department, subject_name, subject_code, password_hash, password_salt, created_at)
                    SELECT id, faculty_id, name, phone_number, department, subject_name, subject_code, password_hash, password_salt, created_at FROM faculty
                """)
                cursor.execute("DROP TABLE faculty")
                cursor.execute("ALTER TABLE faculty_mig RENAME TO faculty")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_id ON faculty(faculty_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_phone ON faculty(phone_number)")

            cursor.execute("PRAGMA table_info(faculty)")
            fac_cols = [row["name"] for row in cursor.fetchall()]
            if "password_hash" not in fac_cols:
                cursor.execute("ALTER TABLE faculty ADD COLUMN password_hash TEXT DEFAULT ''")
            if "password_salt" not in fac_cols:
                cursor.execute("ALTER TABLE faculty ADD COLUMN password_salt TEXT DEFAULT ''")

            cursor.execute("PRAGMA table_info(attendance)")
            cols = [row["name"] for row in cursor.fetchall()]
            if "subject" not in cols:
                cursor.execute("ALTER TABLE attendance ADD COLUMN subject TEXT NOT NULL DEFAULT 'General'")
            if "period" not in cols:
                cursor.execute("ALTER TABLE attendance ADD COLUMN period TEXT NOT NULL DEFAULT 'General'")

            # Timetable table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timetable (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day_of_week TEXT NOT NULL,
                    period_name TEXT NOT NULL,
                    subject_code TEXT NOT NULL,
                    subject_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    faculty_phone TEXT DEFAULT ''
                )
            """)

            cursor.execute("PRAGMA table_info(timetable)")
            tt_cols = [row["name"] for row in cursor.fetchall()]
            if "faculty_phone" not in tt_cols:
                cursor.execute("ALTER TABLE timetable ADD COLUMN faculty_phone TEXT DEFAULT ''")

            # Indexes for ultra-fast query execution
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_phone ON faculty(phone_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_uid ON users(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_uid ON attendance(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_subj ON attendance(subject)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tt_day ON timetable(day_of_week)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tt_dept ON timetable(department)")
            conn.commit()

    # --- PERSISTENT OTP VAULT ---

    def save_otp(self, phone_number: str, otp_code: str, expiry_minutes: int = 5):
        """Saves or replaces active OTP in SQLite with expiration timestamp."""
        now = datetime.now()
        expires = now + timedelta(minutes=expiry_minutes)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        exp_str = expires.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO otp_vault (phone_number, otp_code, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(phone_number) DO UPDATE SET
                    otp_code = excluded.otp_code,
                    expires_at = excluded.expires_at,
                    created_at = excluded.created_at
            """, (phone_number, otp_code, exp_str, now_str))
            conn.commit()

    def get_active_otp(self, phone_number: str) -> Optional[dict]:
        """Retrieves active OTP for phone number."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM otp_vault WHERE phone_number = ?", (phone_number,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_otp(self, phone_number: str):
        """Removes consumed or expired OTP."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM otp_vault WHERE phone_number = ?", (phone_number,))
            conn.commit()

    # --- FACULTY AUTHENTICATION CRUD ---

    def add_faculty(
        self,
        faculty_id: str,
        name: str,
        phone_number: str,
        department: str,
        subject_name: str,
        subject_code: str,
        password_hash: str = "",
        password_salt: str = "",
    ) -> bool:
        """Registers a new faculty member with Staff ID, mobile number, subject, and password hash."""
        phone = str(phone_number).strip().replace(" ", "").replace("-", "")
        if len(phone) > 10:
            phone = phone[-10:]

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO faculty (faculty_id, name, phone_number, department, subject_name, subject_code, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(faculty_id) DO UPDATE SET
                    name = excluded.name,
                    phone_number = excluded.phone_number,
                    department = excluded.department,
                    subject_name = excluded.subject_name,
                    subject_code = excluded.subject_code,
                    password_hash = excluded.password_hash,
                    password_salt = excluded.password_salt
            """, (faculty_id.strip(), name.strip(), phone, department.strip(), subject_name.strip(), subject_code.strip(), password_hash, password_salt, now_str))
            conn.commit()
            return True

    def get_faculty_by_id(self, login_identifier: str) -> Optional[dict]:
        """Look up faculty profile by faculty/staff ID or mobile phone number (case-insensitive)."""
        ident = str(login_identifier).strip()
        phone_clean = ident.replace(" ", "").replace("-", "")
        if len(phone_clean) > 10:
            phone_clean = phone_clean[-10:]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM faculty
                WHERE LOWER(faculty_id) = LOWER(?)
                   OR phone_number = ?
                   OR phone_number = ?
            """, (ident, ident, phone_clean))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_faculty_password(self, faculty_id: str, password_hash: str, password_salt: str) -> bool:
        """Updates password hash and salt for a faculty member."""
        fid = str(faculty_id).strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE faculty
                SET password_hash = ?, password_salt = ?
                WHERE LOWER(faculty_id) = LOWER(?) OR phone_number = ?
            """, (password_hash, password_salt, fid, fid))
            conn.commit()
            return cursor.rowcount > 0

    def get_faculty_by_phone(self, phone_number: str) -> Optional[dict]:
        """Look up faculty profile by mobile number."""
        phone = str(phone_number).strip().replace(" ", "").replace("-", "")
        if len(phone) > 10:
            phone = phone[-10:]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM faculty WHERE phone_number = ?", (phone,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_faculty(self) -> List[dict]:
        """Fetch all registered faculty profiles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM faculty ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    # --- STUDENT / USER GALLERY CRUD ---

    @staticmethod
    def _embedding_to_blob(embedding: np.ndarray) -> bytes:
        """Serializes 1D float32 numpy array to raw bytes."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def _blob_to_embedding(blob: bytes) -> np.ndarray:
        """Deserializes raw bytes back into 1D float32 numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    def add_user(
        self,
        user_id: str,
        name: str,
        department: str,
        embedding: np.ndarray,
        email: str = "",
        photo_path: str = "",
    ) -> bool:
        """Enrolls a student or staff identity."""
        user_id = str(user_id).strip()
        name = str(name).strip()
        department = str(department).strip() or "General"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        blob = self._embedding_to_blob(embedding)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, name, department, email, embedding, photo_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    department = excluded.department,
                    email = excluded.email,
                    embedding = excluded.embedding,
                    photo_path = CASE WHEN excluded.photo_path != '' THEN excluded.photo_path ELSE users.photo_path END
            """, (user_id, name, department, email, blob, photo_path, now_str))
            conn.commit()

        self._gallery_cache = None
        return True

    def get_user(self, user_id: str) -> Optional[dict]:
        """Fetch a single user profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["embedding"] = self._blob_to_embedding(res["embedding"])
            return res

    def get_all_users(self) -> List[dict]:
        """Fetch list of all registered students."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_id, name, department, email, photo_path, created_at FROM users ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_user(self, user_id: str) -> bool:
        """Remove a registered student and their enrolled identity."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

        self._gallery_cache = None
        return deleted

    def get_gallery(self, force_reload: bool = False) -> Dict[str, dict]:
        """Loads all enrolled embeddings into an in-memory dictionary for sub-millisecond matching."""
        if self._gallery_cache is not None and not force_reload:
            return self._gallery_cache

        gallery = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, name, department, photo_path, embedding FROM users")
            for row in cursor.fetchall():
                gallery[row["user_id"]] = {
                    "name": row["name"],
                    "department": row["department"],
                    "photo_path": row["photo_path"],
                    "embedding": self._blob_to_embedding(row["embedding"]),
                }

        self._gallery_cache = gallery
        return gallery

    # --- ATTENDANCE CRUD ---

    def log_attendance(
        self,
        user_id: str,
        name: str,
        department: str,
        status: str,
        confidence: float,
        metric: str,
        subject: str = "General",
        period: str = "General",
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Logs a period-specific subject attendance record into the database."""
        ts = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO attendance (user_id, name, department, date, timestamp, status, confidence, metric, subject, period)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, name, department, date_str, time_str, status, float(confidence), metric, subject, period))
            rec_id = cursor.lastrowid
            conn.commit()
            return rec_id

    def get_last_attendance(
        self,
        user_id: str,
        date_str: Optional[str] = None,
        subject: Optional[str] = None,
        period: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Retrieves the most recent attendance entry for duplicate/cooldown checking.
        Scoped to subject and period so students can be marked for multiple distinct subjects on the same day.
        """
        date_filter = date_str or datetime.now().strftime("%Y-%m-%d")
        query = "SELECT * FROM attendance WHERE user_id = ? AND date = ?"
        params = [user_id, date_filter]

        if subject and subject != "General":
            query += " AND subject = ?"
            params.append(subject)
        elif period and period != "General":
            query += " AND period = ?"
            params.append(period)

        query += " ORDER BY timestamp DESC LIMIT 1"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_attendance_records(
        self,
        date_str: Optional[str] = None,
        department: Optional[str] = None,
        subject: Optional[str] = None,
        period: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Fetch attendance records with multi-dimensional filtering."""
        query = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if date_str:
            query += " AND date = ?"
            params.append(date_str)
        if department and department != "All":
            query += " AND department = ?"
            params.append(department)
        if subject and subject != "All":
            query += " AND subject = ?"
            params.append(subject)
        if period and period != "All":
            query += " AND period = ?"
            params.append(period)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def clear_attendance(self, date_str: Optional[str] = None, subject: Optional[str] = None) -> int:
        """Clears attendance entries, optionally restricted to a specific date or subject."""
        query = "DELETE FROM attendance WHERE 1=1"
        params = []
        if date_str:
            query += " AND date = ?"
            params.append(date_str)
        if subject and subject != "All":
            query += " AND subject = ?"
            params.append(subject)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            count = cursor.rowcount
            conn.commit()
            return count
