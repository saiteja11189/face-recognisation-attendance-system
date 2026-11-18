"""
Unit tests for Faculty Staff ID and Password Authentication.
Validates:
1. PBKDF2-HMAC-SHA256 password hashing and secure salt verification.
2. Faculty registration with user-defined subject and password.
3. Strict credential checking: Rejection of incorrect usernames and passwords.
4. Support for logging in via Staff ID or registered Mobile Number.
5. Password update and duplicate Staff ID prevention.
"""

import os
import unittest
from core.database import Database
from core.auth_manager import AuthManager, hash_password, verify_password


class TestAuthManager(unittest.TestCase):
    def setUp(self):
        self.db_path = "data/test_auth_pwd.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = Database(self.db_path)
        self.auth = AuthManager(self.db)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_password_hashing_and_verification(self):
        """Tests that PBKDF2-HMAC-SHA256 hashes are unique, salted, and verify correctly."""
        pwd = "SecretPassword123!"
        h1, s1 = hash_password(pwd)
        h2, s2 = hash_password(pwd)

        # Unique salts should produce different hashes for same password
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(h1, h2)

        # Correct password verification
        self.assertTrue(verify_password(pwd, h1, s1))
        self.assertTrue(verify_password(pwd, h2, s2))

        # Incorrect password verification
        self.assertFalse(verify_password("WrongPassword!", h1, s1))
        self.assertFalse(verify_password("123456", h1, s1))
        self.assertFalse(verify_password("", h1, s1))

    def test_faculty_registration_and_login_success(self):
        """Tests registering a new faculty with custom subject and logging in."""
        r_ok, fac, msg = self.auth.register_faculty(
            faculty_id="LPU-CSE-801",
            name="Dr. Sunita Sharma",
            phone_number="9876543210",
            department="Computer Science & Engineering",
            subject_name="Deep Neural Networks",
            subject_code="CSE522",
            password="FacultyPassword#2026",
        )
        self.assertTrue(r_ok)
        self.assertIsNotNone(fac)
        self.assertEqual(fac["faculty_id"], "LPU-CSE-801")
        self.assertEqual(fac["subject_name"], "Deep Neural Networks")

        # Login with correct Staff ID & password
        l_ok, profile, l_msg = self.auth.login("LPU-CSE-801", "FacultyPassword#2026")
        self.assertTrue(l_ok)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], "Dr. Sunita Sharma")

        # Case-insensitive Staff ID login
        l_ok2, profile2, _ = self.auth.login("lpu-cse-801", "FacultyPassword#2026")
        self.assertTrue(l_ok2)

        # Login via registered mobile number
        l_ok3, profile3, _ = self.auth.login("9876543210", "FacultyPassword#2026")
        self.assertTrue(l_ok3)

    def test_login_invalid_credentials(self):
        """Tests rejection of invalid username or password."""
        self.auth.register_faculty(
            faculty_id="EMP-100",
            name="Prof. John",
            phone_number="9123456789",
            department="IT",
            subject_name="Cloud Computing",
            subject_code="IT301",
            password="CorrectPassword1",
        )

        # 1. Invalid password
        ok_p, _, err_p = self.auth.login("EMP-100", "WrongPassword")
        self.assertFalse(ok_p)
        self.assertIn("Invalid username or password", err_p)

        # 2. Invalid username / Staff ID
        ok_u, _, err_u = self.auth.login("NON_EXISTENT_ID", "CorrectPassword1")
        self.assertFalse(ok_u)
        self.assertIn("Invalid username or password", err_u)

        # 3. Empty fields
        ok_e, _, _ = self.auth.login("", "")
        self.assertFalse(ok_e)

    def test_duplicate_staff_id_prevention(self):
        """Tests that registering an already existing Staff ID is rejected."""
        self.auth.register_faculty(
            faculty_id="EMP-DUP",
            name="First User",
            phone_number="9000000001",
            department="CSE",
            subject_name="Math",
            subject_code="M101",
            password="Pass1",
        )
        r_ok, _, err = self.auth.register_faculty(
            faculty_id="EMP-DUP",
            name="Second User",
            phone_number="9000000002",
            department="CSE",
            subject_name="Science",
            subject_code="S101",
            password="Pass2",
        )
        self.assertFalse(r_ok)
        self.assertIn("already registered", err)

    def test_change_password(self):
        """Tests updating faculty password."""
        self.auth.register_faculty(
            faculty_id="FAC-CP",
            name="Dr. Change",
            phone_number="9111111111",
            department="CSE",
            subject_name="Algorithms",
            subject_code="CSE201",
            password="InitialPassword#1",
        )

        # Wrong old password
        c_ok_w, err_w = self.auth.change_password("FAC-CP", "BadOldPass", "BrandNewPassword#2")
        self.assertFalse(c_ok_w)

        # Correct old password
        c_ok, msg = self.auth.change_password("FAC-CP", "InitialPassword#1", "BrandNewPassword#2")
        self.assertTrue(c_ok)

        # Old password must now fail
        l_old, _, _ = self.auth.login("FAC-CP", "InitialPassword#1")
        self.assertFalse(l_old)

        # New password must succeed
        l_new, _, _ = self.auth.login("FAC-CP", "BrandNewPassword#2")
        self.assertTrue(l_new)


if __name__ == "__main__":
    unittest.main()
