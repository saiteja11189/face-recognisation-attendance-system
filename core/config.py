"""
Configuration settings for Face Recognition Attendance System.
Centralizes paths, model hyperparameters, thresholds, and operational flags.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ATTENDANCE_LOGS_DIR = DATA_DIR / "attendance_logs"
FACES_DIR = DATA_DIR / "faces"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ATTENDANCE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "attendance.db"

# Model file paths
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
HAAR_FACE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
HAAR_EYE_PATH = MODELS_DIR / "haarcascade_eye.xml"

# Face Detector settings
DEFAULT_DETECTOR = "yunet"  # 'yunet' or 'haar'
DETECTION_SCORE_THRESHOLD = 0.80  # Filter out low-confidence face detections
DETECTION_NMS_THRESHOLD = 0.30    # Non-maximum suppression threshold
DETECTION_TOP_K = 5000

# Face Recognizer / Feature Extractor settings
# SFace model produces 128-dimensional L2-normalized embeddings
EMBEDDING_DIMENSION = 128

# Similarity Matching Thresholds (Calibrated for SFace)
# Cosine Similarity: higher is more similar (range [-1, 1], typically [0, 1] for face embeddings)
# Standard recommended threshold: 0.363; 0.40 provides stricter false-positive rejection.
COSINE_SIMILARITY_THRESHOLD = 0.38

# Euclidean (L2) Norm Distance: lower is more similar (range [0, 2] for unit vectors)
# Standard recommended threshold: 1.128; 1.05 provides stricter matching.
L2_DISTANCE_THRESHOLD = 1.10

# Default Metric: 'cosine' or 'l2'
DEFAULT_METRIC = "cosine"

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_FPS = 30

# Attendance Business Logic
# Cooldown in minutes: prevent duplicate logs for the same user within this time window
ATTENDANCE_COOLDOWN_MINUTES = 60

# Attendance time windows (for Late vs Present status)
WORK_START_HOUR = 9     # 9:00 AM
WORK_START_MINUTE = 15  # 15 minutes grace period
