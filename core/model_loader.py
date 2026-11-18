"""
Model loader utility for Face Recognition Attendance System.
Ensures required ONNX and XML weights are present, downloading them automatically if needed.
"""

import os
import sys
import logging
from pathlib import Path
import urllib.request
import ssl
from typing import Tuple

from core import config

logger = logging.getLogger(__name__)

# Pre-configured download sources
MODEL_URLS = {
    "yunet": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "sface": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
    "haar_face": [
        "https://cdn.jsdelivr.net/gh/opencv/opencv@4.x/data/haarcascades/haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
    ],
    "haar_eye": [
        "https://cdn.jsdelivr.net/gh/opencv/opencv@4.x/data/haarcascades/haarcascade_eye.xml",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_eye.xml",
    ],
}


def download_file(urls, dest_path: Path) -> bool:
    """Download a file from a list of URLs with fallback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True

    # SSL context with verification disabled if corporate proxy interferes
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in urls:
        try:
            logger.info(f"Downloading {dest_path.name} from {url}...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if dest_path.exists() and dest_path.stat().st_size > 1024:
                logger.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")
                return True
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
    return False


def ensure_models() -> dict:
    """Verify that required models exist, downloading missing ones."""
    status = {
        "yunet": config.YUNET_MODEL_PATH.exists() and config.YUNET_MODEL_PATH.stat().st_size > 10000,
        "sface": config.SFACE_MODEL_PATH.exists() and config.SFACE_MODEL_PATH.stat().st_size > 1000000,
        "haar_face": config.HAAR_FACE_PATH.exists() and config.HAAR_FACE_PATH.stat().st_size > 10000,
    }

    if not status["yunet"]:
        status["yunet"] = download_file(MODEL_URLS["yunet"], config.YUNET_MODEL_PATH)
    if not status["sface"]:
        status["sface"] = download_file(MODEL_URLS["sface"], config.SFACE_MODEL_PATH)
    if not status["haar_face"]:
        status["haar_face"] = download_file(MODEL_URLS["haar_face"], config.HAAR_FACE_PATH)

    return status
