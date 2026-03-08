# -*- coding: utf-8 -*-
import datetime
import logging
import os
import threading
from logging.handlers import RotatingFileHandler

import cv2


_MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
_LEGACY_ROOT = os.path.dirname(_MODULE_ROOT)
_PREFERRED_DATA_ROOT = os.environ.get("LOABOT_DATA_ROOT", r"D:\LoABot_Training_Data")


def project_path(rel_path: str) -> str:
    """
    Local-first path resolution for v5.9:
    1) C:/LoABot/LoABoTv5.9/<rel_path>
    2) C:/LoABot/<rel_path> (legacy fallback)
    """
    local_path = os.path.join(_MODULE_ROOT, rel_path)
    legacy_path = os.path.join(_LEGACY_ROOT, rel_path)
    if os.path.exists(local_path):
        return local_path
    if os.path.exists(legacy_path):
        return legacy_path
    return local_path


IMAGE_DIR = project_path("image")
LEGACY_IMAGE_DIR = os.path.join(_LEGACY_ROOT, "image")

CONFIG_CANDIDATES = [
    project_path("config/GeminiProConfig.yaml"),
    project_path("GeminiProConfig.YAML"),
    project_path("GeminiProConfig.yaml"),
]
CONFIG_FILE = next((p for p in CONFIG_CANDIDATES if os.path.exists(p)), CONFIG_CANDIDATES[0])

if os.path.isdir(_PREFERRED_DATA_ROOT):
    _runtime_log_dir = os.path.join(_PREFERRED_DATA_ROOT, "runtime_logs")
    os.makedirs(_runtime_log_dir, exist_ok=True)
    LOG_FILE = os.path.join(_runtime_log_dir, "game_monitor_v59.log")
else:
    LOG_FILE = os.path.join(_MODULE_ROOT, "game_monitor_v59.log")
BOT_IS_CLICKING_EVENT = threading.Event()


# ── Performanslı loglama altyapısı ───────────────────────────────────────
# Eski kod her log çağrısında os.makedirs + open/close yapıyordu.
# Yoğun I/O trafiğinde (YOLO + SeqRec + Reward eşzamanlı) ciddi bottleneck.
# RotatingFileHandler: tek dosya handle, 10 MB rotasyon, 5 yedek.
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

_log_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(_log_formatter)

_logger = logging.getLogger("LoABot")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_file_handler)


def log_to_file(msg):
    """Thread-safe, buffered dosya loglama. Eski open/close yerine geçer."""
    _logger.info(msg)


def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def load_image(filename: str):
    path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    legacy_path = os.path.join(LEGACY_IMAGE_DIR, filename)
    if os.path.exists(legacy_path):
        return cv2.imread(legacy_path, cv2.IMREAD_GRAYSCALE)

    return None
