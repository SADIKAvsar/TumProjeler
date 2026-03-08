"""
vision_manager.py — Görsel Algılama Motoru (v2.0 — Lean)
=========================================================
Shadow mode (JPEG resim toplama) tamamen kaldırıldı.
Video tabanlı kayıt sistemi (video_recorder.py) aktif.

Kalan sorumluluklar:
  - Template matching (cv2.matchTemplate)
  - YOLOv10 inference
  - Ekran yakalama (mss)
  - Görsel cache yönetimi
"""

import os
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from utils import IMAGE_DIR, load_image, _safe_int

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from core.event_types import EVT
except Exception:
    class _FallbackEVT:
        VISION_UPDATE = "VISION_UPDATE"
        OBJECT_FOUND = "OBJECT_FOUND"
    EVT = _FallbackEVT()


class VisionManager:
    def __init__(self, bot):
        self.bot = bot
        self.image_cache = {}

        self.screen_width = int(self.bot.settings.get("SCREEN_WIDTH", 2560))
        self.screen_height = int(self.bot.settings.get("SCREEN_HEIGHT", 1440))

        # YOLOv10
        self.yolo_model = None
        self.yolo_device = "cpu"
        self._load_yolo_model()

        self.bot.log("VisionManager v2: YOLO + Template Matching motoru hazır.")

    # ══════════════════════════════════════════════════════════════════
    #  YOLO MODEL
    # ══════════════════════════════════════════════════════════════════

    def _load_yolo_model(self):
        """YOLOv10n modelini tercihen GPU (cuda) üzerinde yükler."""
        model_path = self.bot.settings.get("YOLO_MODEL_PATH", "yolov10n.pt")
        requested_device = str(self.bot.settings.get("YOLO_DEVICE", "cuda")).lower()

        if YOLO is None:
            self.bot.log("YOLO yuklenemedi: ultralytics import hatasi.", level="WARNING")
            return

        if not os.path.isfile(model_path):
            self.bot.log(
                f"YOLO model dosyasi bulunamadi: {model_path}. "
                "YOLO devre disi, template-matching fallback aktif.",
                level="ERROR",
            )
            self.yolo_model = None
            return

        device = "cpu"
        if requested_device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"

        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(device)
            self.yolo_device = device
            self.bot.log(f"YOLO yüklendi: {model_path} (device={device})")
            if requested_device == "cuda" and device != "cuda":
                self.bot.log(
                    "UYARI: CUDA istenmis ama aktif degil; YOLO CPU'da calisiyor.",
                    level="WARNING",
                )
        except Exception as exc:
            self.yolo_model = None
            self.yolo_device = "cpu"
            self.bot.log(f"YOLO yukleme hatasi: {exc}", level="WARNING")

    # ══════════════════════════════════════════════════════════════════
    #  IMAGE CACHE
    # ══════════════════════════════════════════════════════════════════

    def preload_all(self):
        """Image klasöründeki tüm PNG dosyalarını RAM'e yükler."""
        self.bot.log("VisionManager: Görsel hafıza yükleniyor...")
        loaded_count = 0
        try:
            for filename in os.listdir(IMAGE_DIR):
                if filename.lower().endswith(".png"):
                    img = load_image(filename)
                    if img is not None:
                        self.image_cache[filename] = img
                        loaded_count += 1
            self.bot.log(f"VisionManager: {loaded_count} adet resim RAM'e yuklendi.")
        except Exception as e:
            self.bot.log(f"VisionManager yukleme hatasi: {e}")

    def get_image(self, filename: str):
        if filename not in self.image_cache:
            self.image_cache[filename] = load_image(filename)
        return self.image_cache.get(filename)

    # ══════════════════════════════════════════════════════════════════
    #  BOSS VISUAL DATA
    # ══════════════════════════════════════════════════════════════════

    def get_boss_visual_data(self, boss_data, key_type="spawn_check"):
        """
        Boss YAML içindeki görsel bloklarını normalize eder.
        Desteklenen bloklar: anchor, area_check, spawn_check, victory.
        """
        data = boss_data.get(key_type, {})

        if key_type == "anchor" and not isinstance(data, dict):
            data = boss_data.get("gorsel_dogrulama", {})

        if key_type == "victory":
            if not isinstance(data, dict):
                data = {}
            if not data.get("image_file") and boss_data.get("victory_image"):
                data["image_file"] = boss_data.get("victory_image")

        if not isinstance(data, dict):
            data = {}

        default_conf = 0.70
        if key_type == "anchor":
            default_conf = 0.80
        elif key_type == "victory":
            default_conf = 0.75

        return {
            "image_file": data.get("image_file", "default.png"),
            "confidence": float(data.get("confidence", default_conf)),
            "region_key": data.get("region_key", "region_full_screen"),
            "timeout_sn": float(data.get("timeout_sn", 6.0)),
            "poll_interval_sn": float(data.get("poll_interval_sn", 0.4)),
            "required": bool(data.get("required", False)),
            "enabled": bool(data.get("enabled", True)),
            "pre_window_sn": float(data.get("pre_window_sn", 4.0)),
        }

    # ══════════════════════════════════════════════════════════════════
    #  SCREEN CAPTURE
    # ══════════════════════════════════════════════════════════════════

    def capture_region(self, region: dict):
        """Verilen bölgeden BGR screenshot döndürür."""
        monitor = {
            "top": _safe_int(region.get("y"), 0),
            "left": _safe_int(region.get("x"), 0),
            "width": _safe_int(region.get("w"), self.screen_width),
            "height": _safe_int(region.get("h"), self.screen_height),
        }
        try:
            with mss.mss() as sct:
                bgra = np.array(sct.grab(monitor))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def capture_full_screen(self):
        """Tam ekran BGR screenshot."""
        full_region = self.bot.ui_regions.get(
            "region_full_screen",
            {"x": 0, "y": 0, "w": self.screen_width, "h": self.screen_height},
        )
        return self.capture_region(full_region)

    # ══════════════════════════════════════════════════════════════════
    #  YOLO INFERENCE
    # ══════════════════════════════════════════════════════════════════

    def detect_with_yolo(self, frame: np.ndarray, conf: float = 0.25, classes=None):
        """YOLOv10 inferansı."""
        if self.yolo_model is None or frame is None:
            return None
        try:
            return self.yolo_model.predict(
                source=frame,
                conf=conf,
                classes=classes,
                device=self.yolo_device,
                verbose=False,
            )
        except Exception as exc:
            self.bot.log(f"YOLO inference hatasi: {exc}", level="WARNING")
            return None

    # ══════════════════════════════════════════════════════════════════
    #  TEMPLATE MATCHING — find()
    # ══════════════════════════════════════════════════════════════════

    def find(
        self,
        template_name: str,
        region: dict,
        confidence: float = 0.7,
        target_data: dict = None,
        stage: str = None,
        shadow_force_capture: bool = False,
    ):
        """
        Belirtilen bölgede şablon eşleştirmesi yapar.

        Döndürür:
            (cx, cy, max_val) — eşleşme merkezi ve güven skoru
            None — bulunamadıysa
        """
        template = self.get_image(template_name)
        if template is None:
            return None

        monitor = {
            "top": _safe_int(region.get("y"), 0),
            "left": _safe_int(region.get("x"), 0),
            "width": _safe_int(region.get("w"), self.screen_width),
            "height": _safe_int(region.get("h"), self.screen_height),
        }

        try:
            with mss.mss() as sct:
                bgra = np.array(sct.grab(monitor))
            gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)

            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= confidence:
                x1 = monitor["left"] + max_loc[0]
                y1 = monitor["top"] + max_loc[1]
                w = int(template.shape[1])
                h = int(template.shape[0])
                cx = x1 + (w // 2)
                cy = y1 + (h // 2)
                self._publish_find_events(
                    template_name=template_name,
                    region=region,
                    confidence=confidence,
                    score=max_val,
                    center_x=cx,
                    center_y=cy,
                    stage=stage,
                    target_data=target_data,
                )
                return (cx, cy, max_val)

        except Exception as exc:
            self.bot.log(
                f"[VISION] find() hatasi: sablon={template_name!r} bolge={region} - {exc}",
                level="WARNING",
            )
            return None

        return None

    def _publish_find_events(
        self,
        template_name: str,
        region: dict,
        confidence: float,
        score: float,
        center_x: int,
        center_y: int,
        stage: str = None,
        target_data: dict = None,
    ) -> None:
        """
        v6.0 EventBus yayini (backward-compatible):
        - EventBus yoksa sessizce atlanir.
        """
        bus = getattr(self.bot, "event_bus", None)
        if bus is None:
            return

        target_id = ""
        if isinstance(target_data, dict):
            target_id = str(target_data.get("aciklama", target_data.get("boss_id", "")))

        found_payload = {
            "template": str(template_name),
            "stage": str(stage or ""),
            "target_id": target_id,
            "confidence_threshold": float(confidence),
            "score": float(score),
            "center": {"x": int(center_x), "y": int(center_y)},
            "region": dict(region or {}),
            "frame_ts": time.time(),
        }

        try:
            bus.publish(EVT.OBJECT_FOUND, found_payload, source="vision_manager.find")
            bus.publish(
                EVT.VISION_UPDATE,
                {
                    "objects": [found_payload],
                    "frame_ts": found_payload["frame_ts"],
                    "stage": found_payload["stage"],
                },
                source="vision_manager.find",
            )
        except Exception as exc:
            self.bot.log(f"[VISION] EventBus publish hatasi: {exc}", level="WARNING")
