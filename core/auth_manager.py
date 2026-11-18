"""
Faculty Authentication and Access Control Manager.
Provides secure Staff ID / Username and Password authentication,
salted PBKDF2-HMAC-SHA256 password hashing, and subject profile management.
"""

from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import re
import secrets
from typing import Dict, Optional, Tuple

from core.database import Database

logger = logging.getLogger(__name__)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Cryptographically hashes a plaintext password using PBKDF2-HMAC-SHA256
    with a 16-byte random salt and 100,000 iterations.
    Returns (hex_hash, salt_hex).
    """
    if not salt:
        salt = secrets.token_hex(16)
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    pwd_hash = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000).hex()
    return pwd_hash, salt


def verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    """
    Verifies a plaintext password against the stored hash and salt
    using constant-time comparison to prevent timing attacks.
    """
    if not pwd_hash or not salt:
        return False
    calc_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(calc_hash, pwd_hash)


class AuthManager:
    """
    Manages Faculty / Staff ID and Password authentication, registration,
    and profile access control.
    """

    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def clean_phone(phone: str) -> str:
        """Extracts 10 digits from mobile number string."""
        digits = re.sub(r"\D", "", str(phone))
        if len(digits) > 10:
            return digits[-10:]
        return digits

    def login(self, login_identifier: str, password: str) -> Tuple[bool, Optional[dict], str]:
        """
        Authenticates a faculty member using their Staff/User ID or Mobile Number
        and their created password.
        Returns:
            (success: bool, faculty_profile: Optional[dict], message: str)
        """
        ident = str(login_identifier).strip()
        pwd = str(password).strip()

        if not ident or not pwd:
            return False, None, "Please enter both your Staff ID and password."

        # Look up by Staff ID or Phone
        faculty = self.db.get_faculty_by_id(ident)
        if not faculty:
            # Constant-time dummy computation to mitigate user enumeration timing
            hash_password("dummy_password_timing_protection")
            return False, None, "Invalid username or password. Please verify your Staff ID and password."

        stored_hash = faculty.get("password_hash") or ""
        stored_salt = faculty.get("password_salt") or ""

        # If existing legacy account without password, check if they provide a password to set
        if not stored_hash or not stored_salt:
            # Upgrade account by saving this password
            new_hash, new_salt = hash_password(pwd)
            self.db.update_faculty_password(faculty["faculty_id"], new_hash, new_salt)
            refreshed = self.db.get_faculty_by_id(ident)
            return True, refreshed, "Password created and login successful!"

        # Verify password
        if not verify_password(pwd, stored_hash, stored_salt):
            return False, None, "Invalid username or password. Please verify your Staff ID and password."

        return True, faculty, f"Welcome back, {faculty['name']}!"

    def register_faculty(
        self,
        faculty_id: str,
        name: str,
        phone_number: str,
        department: str,
        subject_name: str,
        subject_code: str,
        password: str,
    ) -> Tuple[bool, Optional[dict], str]:
        """
        Registers a new faculty profile with Staff ID, assigned subject, and secure password.
        """
        fid = str(faculty_id).strip()
        fname = str(name).strip()
        phone = self.clean_phone(phone_number)
        dept = str(department).strip()
        subj = str(subject_name).strip()
        code = str(subject_code).strip() or "N/A"
        pwd = str(password).strip()

        if not fid or not fname or not subj or not pwd:
            return False, None, "Please fill in all mandatory fields (Staff ID, Name, Subject, and Password)."

        if len(phone) != 10:
            return False, None, "Please enter a valid 10-digit mobile number."

        if len(pwd) < 4:
            return False, None, "Password must be at least 4 characters long."

        # Check if Staff ID is already registered
        existing = self.db.get_faculty_by_id(fid)
        if existing and existing["faculty_id"].lower() == fid.lower():
            return False, None, f"Staff ID '{fid}' is already registered. Please sign in or use another Staff ID."

        # Hash password
        pwd_hash, salt = hash_password(pwd)

        # Save into SQLite database
        try:
            self.db.add_faculty(
                faculty_id=fid,
                name=fname,
                phone_number=phone,
                department=dept,
                subject_name=subj,
                subject_code=code,
                password_hash=pwd_hash,
                password_salt=salt,
            )
        except Exception as e:
            logger.error(f"Error registering faculty {fid}: {e}")
            return False, None, f"Registration could not be completed: {e}"

        created = self.db.get_faculty_by_id(fid)
        logger.info(f"Registered new faculty: {fname} (ID: {fid}, Subject: {subj})")
        return True, created, f"Account successfully created for {fname} ({subj})!"

    def change_password(self, faculty_id: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Allows an authenticated faculty to update their password."""
        fid = str(faculty_id).strip()
        old_pwd = str(old_password).strip()
        new_pwd = str(new_password).strip()

        if not fid or not old_pwd or not new_pwd:
            return False, "Please fill in both old and new passwords."

        if len(new_pwd) < 4:
            return False, "New password must be at least 4 characters long."

        faculty = self.db.get_faculty_by_id(fid)
        if not faculty:
            return False, "Faculty account not found."

        stored_hash = faculty.get("password_hash") or ""
        stored_salt = faculty.get("password_salt") or ""

        if stored_hash and stored_salt:
            if not verify_password(old_pwd, stored_hash, stored_salt):
                return False, "Incorrect current password."

        # Save new hashed password
        new_hash, new_salt = hash_password(new_pwd)
        self.db.update_faculty_password(fid, new_hash, new_salt)
        return True, "Password successfully updated!"
