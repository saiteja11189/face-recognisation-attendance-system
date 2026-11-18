"""
Core package exports for Face Recognition Attendance System.
"""

from core.detector import FaceDetector, DetectedFace
from core.preprocessor import FacePreprocessor
from core.embedder import FaceEmbedder
from core.matcher import SimilarityMatcher, MatchResult
from core.database import Database
from core.attendance_manager import AttendanceManager
from core.schedule_manager import ScheduleManager, PeriodSlot
from core.auth_manager import AuthManager
from core.camera import Camera
from core.anti_spoofing import AntiSpoofingDetector, LivenessResult
import core.config as config

__all__ = [
    "FaceDetector",
    "DetectedFace",
    "FacePreprocessor",
    "FaceEmbedder",
    "SimilarityMatcher",
    "MatchResult",
    "Database",
    "AttendanceManager",
    "ScheduleManager",
    "PeriodSlot",
    "AuthManager",
    "Camera",
    "AntiSpoofingDetector",
    "LivenessResult",
    "config",
]
