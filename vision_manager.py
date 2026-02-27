import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import mss
import numpy as np

from utils import IMAGE_DIR, load_image, _safe_int

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class VisionManager:
    def __init__(self, bot):
        self.bot = bot
        self.image_cache = {}

        self.screen_width = int(self.bot.settings.get("SCREEN_WIDTH", 2560))
        self.screen_height = int(self.bot.settings.get("SCREEN_HEIGHT", 1440))

        # Shadow mode settings (E:\ drive)
        self.shadow_mode_enabled = bool(self.bot.settings.get("SHADOW_MODE_ENABLED", True))
        self.shadow_data_root = Path(
            self.bot.settings.get("SHADOW_DATA_ROOT", r"E:\LoABot_Training_Data")
        )
        self.shadow_jpeg_quality = int(self.bot.settings.get("SHADOW_JPEG_QUALITY", 92))
        self.shadow_queue_maxsize = int(self.bot.settings.get("SHADOW_QUEUE_MAXSIZE", 256))
        self.shadow_worker_count = int(self.bot.settings.get("SHADOW_WORKER_COUNT", 2))
        self.shadow_save_min_interval_sn = float(self.bot.settings.get("SHADOW_SAVE_MIN_INTERVAL_SN", 0.15))
        self.shadow_save_on_miss = bool(self.bot.settings.get("SHADOW_SAVE_ON_MISS", True))
        self.shadow_miss_min_interval_sn = float(self.bot.settings.get("SHADOW_MISS_MIN_INTERVAL_SN", 2.0))
        self.shadow_system_run_yolo = bool(self.bot.settings.get("SHADOW_SYSTEM_RUN_YOLO", True))
        self.shadow_buffer_ram_gb = float(self.bot.settings.get("SHADOW_BUFFER_RAM_GB", 32.0))
        self.mission_recorder_enabled = bool(self.bot.settings.get("MISSION_RECORDER_ENABLED", True))
        self.mission_capture_fps = float(self.bot.settings.get("MISSION_CAPTURE_FPS", 1.0))
        self.mission_run_yolo = bool(self.bot.settings.get("MISSION_RUN_YOLO", False))
        self.recording_enabled = bool(self.bot.settings.get("RECORDING_ENABLED", True))

        # 32GB tampon hedefi icin queue kapasitesi RAM'e gore hesaplanir.
        approx_frame_bytes = max(1, self.screen_width * self.screen_height * 3)
        ram_queue_cap = int((self.shadow_buffer_ram_gb * (1024**3)) / approx_frame_bytes)
        self.shadow_queue_maxsize = max(16, self.shadow_queue_maxsize, min(max(16, ram_queue_cap), 10000))

        self._shadow_last_saved = {}
        self._shadow_lock = threading.Lock()
        self._shadow_queue = Queue(maxsize=max(16, self.shadow_queue_maxsize))
        self._shadow_stop_event = threading.Event()
        self._shadow_workers = []

        self._mission_state_lock = threading.Lock()
        self._mission_active = False
        self._mission_session_id = ""
        self._mission_namespace = "NAV_PHASE"
        self._mission_stage = "mission_loop"
        self._mission_reason = ""
        self._mission_extra = {}
        self._mission_thread = None

        # Class map for YOLO labels
        self._class_map = {}
        self._class_map_path = self.shadow_data_root / "class_map.json"

        # YOLOv10
        self.yolo_model = None
        self.yolo_device = "cpu"
        self._load_yolo_model()

        self._prepare_shadow_mode()

    def _load_yolo_model(self):
        """YOLOv10n modelini tercihen GPU (cuda) uzerinde yukler."""
        model_path = self.bot.settings.get("YOLO_MODEL_PATH", "yolov10n.pt")
        requested_device = str(self.bot.settings.get("YOLO_DEVICE", "cuda")).lower()

        if YOLO is None:
            self.bot.log("YOLO yuklenemedi: ultralytics import hatasi.", level="WARNING")
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
                self.bot.log("UYARI: CUDA istenmis ama aktif degil; YOLO CPU'da calisiyor.", level="WARNING")
        except Exception as exc:
            self.yolo_model = None
            self.yolo_device = "cpu"
            self.bot.log(f"YOLO yukleme hatasi: {exc}", level="WARNING")

    def _prepare_shadow_mode(self):
        if not self.shadow_mode_enabled:
            self.recording_enabled = False
            self.bot.log("Shadow mode pasif.", level="DEBUG")
            return

        try:
            self.shadow_data_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.bot.log(f"Shadow mode klasor olusturma hatasi: {exc}", level="WARNING")
            self.shadow_mode_enabled = False
            return

        self._load_class_map()

        for idx in range(max(1, self.shadow_worker_count)):
            t = threading.Thread(
                target=self._shadow_worker_loop,
                daemon=True,
                name=f"ShadowSaveWorker-{idx+1}",
            )
            t.start()
            self._shadow_workers.append(t)

        if self.mission_recorder_enabled:
            self._mission_thread = threading.Thread(
                target=self._mission_recorder_loop,
                daemon=True,
                name="MissionRecorderThread",
            )
            self._mission_thread.start()

        if self.recording_enabled:
            self.bot.log(f"Shadow mode aktif. Kayit yolu: {self.shadow_data_root}")
        else:
            self.bot.log("Shadow mode aktif ama kayit baslangicta kapali.", level="INFO")

    def is_recording_enabled(self) -> bool:
        return bool(self.shadow_mode_enabled and self.recording_enabled)

    def set_recording_enabled(self, enabled: bool) -> bool:
        if not self.shadow_mode_enabled:
            self.recording_enabled = False
            return False
        self.recording_enabled = bool(enabled)
        return self.recording_enabled

    def toggle_recording(self) -> bool:
        return self.set_recording_enabled(not self.is_recording_enabled())

    def _load_class_map(self):
        if self._class_map_path.exists():
            try:
                self._class_map = json.loads(self._class_map_path.read_text(encoding="utf-8"))
            except Exception:
                self._class_map = {}
        else:
            self._class_map = {}

    def _save_class_map(self):
        try:
            self._class_map_path.write_text(
                json.dumps(self._class_map, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _get_or_create_class_id(self, boss_id: str) -> int:
        key = str(boss_id)
        with self._shadow_lock:
            if key not in self._class_map:
                self._class_map[key] = len(self._class_map)
                self._save_class_map()
            return int(self._class_map[key])

    def _shadow_worker_loop(self):
        while not self._shadow_stop_event.is_set():
            try:
                payload = self._shadow_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self.capture_and_save(
                    frame=payload["frame"],
                    target_data=payload["target_data"],
                    stage=payload["stage"],
                )
            except Exception as exc:
                self.bot.log(f"Shadow kayit hatasi: {exc}", level="WARNING")
            finally:
                self._shadow_queue.task_done()

    def start_global_mission(self, namespace: str = "NAV_PHASE", stage: str = "mission_loop", reason: str = "", extra: dict = None):
        if not self.shadow_mode_enabled or not self.mission_recorder_enabled:
            return

        with self._mission_state_lock:
            if not self._mission_session_id:
                self._mission_session_id = datetime.now().strftime("mission_%Y%m%d_%H%M%S")
            self._mission_active = True
            self._mission_namespace = str(namespace or "NAV_PHASE").strip().upper()
            self._mission_stage = str(stage or "mission_loop").strip().replace(" ", "_")
            self._mission_reason = str(reason or "")
            self._mission_extra = dict(extra or {})

    def set_global_mission_phase(self, namespace: str, stage: str = "mission_loop", reason: str = "", extra: dict = None):
        if not self.shadow_mode_enabled or not self.mission_recorder_enabled:
            return

        with self._mission_state_lock:
            if not self._mission_session_id:
                self._mission_session_id = datetime.now().strftime("mission_%Y%m%d_%H%M%S")
            self._mission_active = True
            self._mission_namespace = str(namespace or "NAV_PHASE").strip().upper()
            self._mission_stage = str(stage or "mission_loop").strip().replace(" ", "_")
            if reason:
                self._mission_reason = str(reason)
            if extra is not None:
                self._mission_extra = dict(extra)

    def stop_global_mission(self, reason: str = ""):
        if not self.shadow_mode_enabled or not self.mission_recorder_enabled:
            return
        with self._mission_state_lock:
            self._mission_active = False
            self._mission_reason = str(reason or self._mission_reason)
            self._mission_extra = {}
            self._mission_stage = "mission_loop"
            self._mission_namespace = "NAV_PHASE"

    def is_global_mission_active(self) -> bool:
        with self._mission_state_lock:
            return bool(self._mission_active)

    def _mission_recorder_loop(self):
        fps = max(0.2, float(self.mission_capture_fps))
        interval = 1.0 / fps

        while not self._shadow_stop_event.is_set():
            with self._mission_state_lock:
                active = bool(self._mission_active)
                session_id = self._mission_session_id
                namespace = self._mission_namespace
                stage = self._mission_stage
                reason = self._mission_reason
                extra = dict(self._mission_extra or {})

            if not self.is_recording_enabled():
                time.sleep(0.2)
                continue

            if not active:
                # Kayit acik ama misyon yok: bot calisiyorsa standby frame kaydet.
                bot_running = getattr(self.bot, "running", None)
                if bot_running is not None and bot_running.is_set():
                    frame = self.capture_full_screen()
                    if frame is not None:
                        standby_session = session_id or datetime.now().strftime("standby_%Y%m%d_%H%M%S")
                        self.save_ai_training_data(
                            frame=frame,
                            target_data={
                                "dataset_namespace": "STANDBY_PHASE",
                                "boss_id": "STANDBY_PHASE",
                                "negative": True,
                                "mission_recorder": True,
                                "run_yolo": bool(self.mission_run_yolo),
                                "mission_session_id": standby_session,
                                "reason": "standby_recording",
                            },
                            stage="standby_loop",
                            force=True,
                        )
                time.sleep(interval)
                continue

            frame = self.capture_full_screen()
            if frame is not None:
                meta = {
                    "dataset_namespace": namespace,
                    "boss_id": namespace,
                    "negative": True,
                    "mission_recorder": True,
                    "run_yolo": bool(self.mission_run_yolo),
                    "mission_session_id": session_id,
                    "reason": reason,
                }
                meta.update(extra)
                self.save_ai_training_data(
                    frame=frame,
                    target_data=meta,
                    stage=stage or "mission_loop",
                    force=True,
                )

                if hasattr(self.bot, "log_training_state"):
                    try:
                        self.bot.log_training_state(
                            "mission_tick",
                            {
                                "mission_phase": namespace,
                                "mission_stage": stage,
                                "mission_reason": reason,
                                "mission_session_id": session_id,
                            },
                        )
                    except Exception:
                        pass

            time.sleep(interval)

    def preload_all(self):
        """Image klasorundeki tum PNG dosyalarini RAM'e yukler."""
        self.bot.log("VisionManager: Gorsel hafiza yukleniyor...")
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

    def get_boss_visual_data(self, boss_data, key_type="spawn_check"):
        """
        Boss YAML icindeki gorsel bloklarini normalize eder.
        Desteklenen bloklar: anchor, area_check, spawn_check, victory.
        Geriye ortak bir sozluk dondurur.
        """
        data = boss_data.get(key_type, {})

        # Geriye donuk uyumluluk
        if key_type == "anchor" and not isinstance(data, dict):
            data = boss_data.get("gorsel_dogrulama", {})

        if key_type == "victory":
            if not isinstance(data, dict):
                data = {}
            if not data.get("image_file") and boss_data.get("victory_image"):
                data["image_file"] = boss_data.get("victory_image")

        if not isinstance(data, dict):
            data = {}

        # Varsayilan confidence degerleri blok tipine gore ayarlanir.
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

    def capture_region(self, region: dict):
        """
        Verilen bolgeden BGR screenshot dondurur.
        AI Engine tarafinda freeze/boss kararlarinda kullanilir.
        """
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
        full_region = self.bot.ui_regions.get(
            "region_full_screen",
            {"x": 0, "y": 0, "w": self.screen_width, "h": self.screen_height},
        )
        return self.capture_region(full_region)

    def save_system_shadow_frame(
        self,
        stage: str,
        namespace: str = "system_recovery",
        reason: str = "",
        extra: dict = None,
        force: bool = True,
    ):
        """
        System-level shadow kaydi:
        - Namespace: system_recovery / startup_sync vb.
        - Stage: freeze_check_triggered, server_selection_screen vb.
        - BBox yok; etiket txt bos yazilir.
        """
        if not self.is_recording_enabled():
            return

        frame = self.capture_full_screen()
        if frame is None:
            return

        stage_name = str(stage or "unknown_stage").strip().replace(" ", "_")
        namespace_name = str(namespace or "system").strip().replace(" ", "_")

        meta = {
            "dataset_namespace": namespace_name,
            "boss_id": namespace_name,
            "negative": True,
            "reason": str(reason or ""),
            "run_yolo": bool(self.shadow_system_run_yolo),
        }
        if isinstance(extra, dict):
            meta.update(extra)

        self.save_ai_training_data(
            frame=frame,
            target_data=meta,
            stage=stage_name,
            force=force,
        )

    def save_action_decision_frame(
        self,
        action_name: str,
        payload: dict = None,
        phase: str = "ACTION_PHASE",
        stage: str = "action_decision",
        force: bool = True,
    ) -> str:
        """
        Yerel karar verildigi anda ekrani kaydeder.
        Elde edilen decision_id, TrainingLogger action satirlariyla esitlenir.
        """
        if not self.is_recording_enabled():
            return ""

        frame = self.capture_full_screen()
        if frame is None:
            return ""

        action = str(action_name or "unknown_action").strip().replace(" ", "_")
        phase_name = str(phase or "ACTION_PHASE").strip().upper().replace(" ", "_")
        stage_name = str(stage or "action_decision").strip().replace(" ", "_")
        decision_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        meta = {
            "dataset_namespace": phase_name,
            "boss_id": phase_name,
            "negative": True,
            "reason": "local_action_decision",
            "action_name": action,
            "decision_id": decision_id,
            "mission_session_id": decision_id,
            "run_yolo": bool(self.shadow_system_run_yolo),
        }
        if isinstance(payload, dict):
            meta.update(payload)

        self.save_ai_training_data(
            frame=frame,
            target_data=meta,
            stage=f"{stage_name}_{action}",
            force=force,
        )
        return decision_id

    def detect_with_yolo(self, frame: np.ndarray, conf: float = 0.25, classes=None):
        """
        YOLOv10 inferansi (simdilik yardimci API, ana akisi bozmaz).
        """
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

    def save_ai_training_data(self, frame: np.ndarray, target_data: dict, stage: str, force: bool = False):
        """
        Shadow Mode enqueue:
        - OpenCV find basarili oldugunda cagrilir.
        - Asenkron kayit icin queue'ya yazar.
        """
        if not self.is_recording_enabled():
            return
        if frame is None or not isinstance(target_data, dict):
            return

        boss_id = str(
            target_data.get("boss_id")
            or target_data.get("aciklama")
            or target_data.get("boss_name")
            or "unknown"
        )
        stage_name = str(stage or "unknown_stage").strip().replace(" ", "_")
        namespace = str(target_data.get("dataset_namespace") or "")

        key = f"{namespace}:{boss_id}:{stage_name}"
        now = time.monotonic()
        with self._shadow_lock:
            if not force:
                last = self._shadow_last_saved.get(key, 0.0)
                if now - last < self.shadow_save_min_interval_sn:
                    return
            self._shadow_last_saved[key] = now

        payload = {
            "frame": frame.copy(),
            "target_data": dict(target_data),
            "stage": stage_name,
        }
        try:
            self._shadow_queue.put_nowait(payload)
        except Full:
            self.bot.log("Shadow queue dolu, veri kaydi atlandi.", level="WARNING")

    def save_ai_training_miss(self, frame: np.ndarray, target_data: dict, stage: str, force: bool = False):
        """
        Negatif ornek (object yok) kaydi:
        - jpg yazilir
        - txt bos yazilir (YOLO no-object)
        """
        if not self.is_recording_enabled() or not self.shadow_save_on_miss:
            return
        if frame is None or not isinstance(target_data, dict):
            return

        boss_id = str(
            target_data.get("boss_id")
            or target_data.get("aciklama")
            or target_data.get("boss_name")
            or "unknown"
        )
        stage_name = str(stage or "unknown_stage").strip().replace(" ", "_")
        namespace = str(target_data.get("dataset_namespace") or "")
        key = f"{namespace}:{boss_id}:{stage_name}:miss"
        now = time.monotonic()

        with self._shadow_lock:
            if not force:
                last = self._shadow_last_saved.get(key, 0.0)
                if now - last < self.shadow_miss_min_interval_sn:
                    return
            self._shadow_last_saved[key] = now

        payload = {
            "frame": frame.copy(),
            "target_data": dict(target_data),
            "stage": stage_name,
        }
        payload["target_data"]["negative"] = True

        try:
            self._shadow_queue.put_nowait(payload)
        except Full:
            self.bot.log("Shadow queue dolu, miss kaydi atlandi.", level="WARNING")

    def _save_yolo_snapshot_metadata(self, frame: np.ndarray, out_dir: Path, base: str):
        if self.yolo_model is None:
            return

        preds = self.detect_with_yolo(frame, conf=0.25)
        if not preds:
            return

        items = []
        try:
            result0 = preds[0]
            boxes = getattr(result0, "boxes", None)
            names = getattr(result0, "names", {}) or {}
            if boxes is None:
                return

            for b in boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].tolist() if b.xyxy is not None else None
                if not xyxy or len(xyxy) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                items.append(
                    {
                        "class_id": cls_id,
                        "class_name": str(names.get(cls_id, cls_id)),
                        "confidence": conf,
                        "xyxy": [x1, y1, x2, y2],
                    }
                )
        except Exception as exc:
            self.bot.log(f"YOLO metadata parse hatasi: {exc}", level="WARNING")
            return

        json_path = out_dir / f"{base}.yolo.json"
        try:
            json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def capture_and_save(self, frame: np.ndarray, target_data: dict, stage: str):
        """
        E drive altinda su yapida kayit:
        E:/LoABot_Training_Data/{boss_id}/{stage}/
        - .jpg goruntu
        - .txt YOLO etiketi yazar.
        """
        if not self.is_recording_enabled():
            return
        if frame is None or not isinstance(target_data, dict):
            return

        bbox = target_data.get("bbox")
        negative = bool(target_data.get("negative", False))
        namespace = str(target_data.get("dataset_namespace") or "").strip()
        mission_recorder = bool(target_data.get("mission_recorder", False))
        mission_session = str(target_data.get("mission_session_id") or "").strip()

        boss_id = str(
            target_data.get("boss_id")
            or target_data.get("aciklama")
            or target_data.get("boss_name")
            or "unknown"
        )
        stage_name = str(stage or "unknown_stage").strip().replace(" ", "_")
        if namespace:
            # Mission recorder fazlari tek seviyede tutulur:
            # E:\LoABot_Training_Data\NAV_PHASE\
            out_dir = (self.shadow_data_root / namespace) if mission_recorder else (self.shadow_data_root / namespace / stage_name)
            name_prefix = namespace
        else:
            out_dir = self.shadow_data_root / boss_id / stage_name
            name_prefix = boss_id
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_suffix = f"_{mission_session}" if mission_session else ""
        base = f"{stamp}_{name_prefix}{session_suffix}_{stage_name}"
        jpg_path = out_dir / f"{base}.jpg"
        txt_path = out_dir / f"{base}.txt"

        cv2.imwrite(
            str(jpg_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.shadow_jpeg_quality)],
        )

        if bool(target_data.get("run_yolo", False)):
            self._save_yolo_snapshot_metadata(frame=frame, out_dir=out_dir, base=base)

        # Negatif ornekse bos label yaz.
        if negative:
            txt_path.write_text("", encoding="utf-8")
            self.bot.log(f"Shadow save (NEG): {jpg_path}", level="DEBUG")
            return

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            txt_path.write_text("", encoding="utf-8")
            self.bot.log(f"Shadow save (NO_BBOX): {jpg_path}", level="DEBUG")
            return

        class_id = target_data.get("class_id")
        if class_id is None:
            class_id = self._get_or_create_class_id(boss_id)
        class_id = int(class_id)

        x, y, w, h = [float(v) for v in bbox]
        if w <= 1 or h <= 1:
            txt_path.write_text("", encoding="utf-8")
            self.bot.log(f"Shadow save (SMALL_BBOX): {jpg_path}", level="DEBUG")
            return

        # 2560x1440 normalize (configteki screen_width/height)
        x_center = (x + (w / 2.0)) / float(self.screen_width)
        y_center = (y + (h / 2.0)) / float(self.screen_height)
        ww = w / float(self.screen_width)
        hh = h / float(self.screen_height)

        # clamp
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        ww = max(0.0, min(1.0, ww))
        hh = max(0.0, min(1.0, hh))

        txt_path.write_text(f"{class_id} {x_center:.6f} {y_center:.6f} {ww:.6f} {hh:.6f}\n", encoding="utf-8")
        self.bot.log(f"Shadow save (POS): {jpg_path}", level="DEBUG")

    def find(
        self,
        template_name: str,
        region: dict,
        confidence: float = 0.7,
        target_data: dict = None,
        stage: str = None,
        shadow_force_capture: bool = False,
    ):
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

                if self.is_recording_enabled() and stage:
                    full_region = self.bot.ui_regions.get(
                        "region_full_screen",
                        {"x": 0, "y": 0, "w": self.screen_width, "h": self.screen_height},
                    )
                    is_full_capture = (
                        monitor["left"] == _safe_int(full_region.get("x"), 0)
                        and monitor["top"] == _safe_int(full_region.get("y"), 0)
                        and monitor["width"] == _safe_int(full_region.get("w"), self.screen_width)
                        and monitor["height"] == _safe_int(full_region.get("h"), self.screen_height)
                    )

                    if is_full_capture:
                        full_frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                    else:
                        full_frame = self.capture_region(full_region)

                    meta = dict(target_data or {})
                    meta["bbox"] = [x1, y1, w, h]
                    if "boss_id" not in meta:
                        meta["boss_id"] = str(meta.get("aciklama", "unknown"))
                    self.save_ai_training_data(
                        frame=full_frame,
                        target_data=meta,
                        stage=stage,
                        force=shadow_force_capture,
                    )

                return (cx, cy, max_val)

            # Match bulunamadiysa negatif veri kaydi (opsiyonel)
            if self.is_recording_enabled() and stage and self.shadow_save_on_miss:
                full_region = self.bot.ui_regions.get(
                    "region_full_screen",
                    {"x": 0, "y": 0, "w": self.screen_width, "h": self.screen_height},
                )
                is_full_capture = (
                    monitor["left"] == _safe_int(full_region.get("x"), 0)
                    and monitor["top"] == _safe_int(full_region.get("y"), 0)
                    and monitor["width"] == _safe_int(full_region.get("w"), self.screen_width)
                    and monitor["height"] == _safe_int(full_region.get("h"), self.screen_height)
                )

                if is_full_capture:
                    full_frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                else:
                    full_frame = self.capture_region(full_region)

                miss_meta = dict(target_data or {})
                if "boss_id" not in miss_meta:
                    miss_meta["boss_id"] = str(miss_meta.get("aciklama", "unknown"))
                self.save_ai_training_miss(
                    frame=full_frame,
                    target_data=miss_meta,
                    stage=f"{stage}_miss",
                    force=shadow_force_capture,
                )
        except Exception as exc:
            self.bot.log(
                f"[VISION] find() hatasi: sablon={template_name!r} bolge={region} - {exc}",
                level="WARNING",
            )
            return None
        return None
