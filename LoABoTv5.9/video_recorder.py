# -*- coding: utf-8 -*-
"""
video_recorder.py — LoABoTv5.9 Video Tabanlı Eğitim Verisi Kaydedici
=====================================================================
Amaç:
  Bot bir boss'a saldırmak için EXP_FARM'dan ayrıldığı andan geri
  döndüğü ana kadar kesintisiz video (.mp4) kaydeder. Kayıt esnasında
  tüm mouse tıklamaları ve klavye tuşlamaları video frame index'i ile
  eşleştirilip actions.jsonl dosyasına yazılır.

Mimari:
  ┌─────────────┐   frame    ┌──────────────┐   .mp4
  │ VisionMgr   │──────────▶│ VideoRecorder │──────────▶ disk
  │ (ekran yak.)│           │              │
  └─────────────┘           │  ActionLog   │──────────▶ actions.jsonl
  ┌─────────────┐  event    │              │
  │ pynput hook │──────────▶│              │
  │ (fare+klv)  │           └──────────────┘
  └─────────────┘

Çıktı yapısı:
  D:/LoABot_Training_Data/videos/
  └── SESSION_20250228_143022/
      ├── video.mp4           # Sürekli ekran kaydı (H.264)
      ├── actions.jsonl       # Her satır bir aksiyon eventi
      └── session_meta.json   # Oturum özet bilgisi

actions.jsonl satır formatı:
  {
    "frame_idx": 142,
    "ts_video_sec": 14.2,
    "ts_unix": 1740750622.345,
    "event_type": "mouse_click",
    "data": {"x": 845, "y": 320, "button": "left"},
    "source": "bot" | "user",
    "phase": "COMBAT_PHASE",
    "action_label": "attack_boss"
  }

Performans:
  - Capture thread ayrı çalışır, main thread bloklanmaz
  - cv2.VideoWriter donanım kodlayıcı destekler (varsa)
  - Frame copy sadece gerektiğinde (zero-copy mümkünse)
  - Bellek: sadece 1 frame buffer (rolling window yok)
  - JPEG yerine video codec → ~5-10x daha az disk alanı

Entegrasyon:
  boss_manager.py → automation_thread() içinde:
    self.bot.video_recorder.start("boss_attack")
    ...saldırı akışı...
    self.bot.video_recorder.stop(success=True/False)
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Sabitler ──────────────────────────────────────────────────────────────
TARGET_FPS: int = 10
FRAME_INTERVAL_SEC: float = 1.0 / TARGET_FPS
DEFAULT_VIDEO_ROOT: str = r"D:\LoABot_Training_Data\videos"
# H.264 codec fourcc — geniş uyumluluk, iyi sıkıştırma
FOURCC_H264: int = cv2.VideoWriter_fourcc(*"mp4v")
# Yedek codec (H.264 başarısızsa)
FOURCC_FALLBACK: int = cv2.VideoWriter_fourcc(*"XVID")
# Aksiyonlar için disk yazma kuyruğu limiti
ACTION_QUEUE_LIMIT: int = 2048
# Debounce süresi (ms) — aynı olayı tekrar loglamamak için
DEBOUNCE_MS: float = 80.0


@dataclass
class ActionEvent:
    """Tek bir kullanıcı/bot aksiyonu."""
    frame_idx: int
    ts_video_sec: float
    ts_unix: float
    event_type: str        # mouse_click | key_press | bot_click | bot_key
    data: Dict[str, Any]
    source: str            # "bot" | "user"
    phase: str
    action_label: str = ""

    # Opsiyonel: 1 kare öncesi ve sonrası referans index'leri
    frame_before: int = -1
    frame_after: int = -1


class VideoRecorder:
    """
    Sürekli video kaydedici + senkronize aksiyon logger.

    Dışa açık API:
      start(trigger_type)  — Kaydı başlat (EXP_FARM ayrılışı)
      stop(success)        — Kaydı durdur (EXP_FARM dönüşü)
      log_action(...)      — Aksiyon logla (tıklama/tuş)
      shutdown()           — Graceful kapanış

    Thread güvenli: Tüm public API'ler thread-safe'dir.
    """

    def __init__(self, bot):
        self.bot = bot

        # ── Ayarları oku ──────────────────────────────────────────────
        settings = getattr(bot, "settings", {}) or {}
        general_cfg = getattr(bot, "general_cfg", {}) or {}
        video_cfg = (
            general_cfg.get("recording", {}).get("video", {})
            if isinstance(general_cfg, dict) else {}
        )

        self._enabled: bool = self._coerce_bool(
            video_cfg.get("enabled", settings.get("VIDEO_RECORDER_ENABLED", True))
        )
        self._data_root: Path = Path(str(
            video_cfg.get("output_path",
                          settings.get("VIDEO_RECORDER_ROOT", DEFAULT_VIDEO_ROOT))
        ))
        self._target_fps: int = int(
            video_cfg.get("fps", settings.get("VIDEO_RECORDER_FPS", TARGET_FPS))
        )
        self._frame_input_color: str = str(
            video_cfg.get("input_color_order",
                          settings.get("VIDEO_INPUT_COLOR_ORDER", "BGR"))
        ).strip().upper()

        # ── Çözünürlük esnekliği ─────────────────────────────────────
        # İlk frame'den dinamik algılanır, sabit değer gerekmez
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._resolution_locked: bool = False

        # ── Oturum durumu ─────────────────────────────────────────────
        self._recording: bool = False
        self._session_id: str = ""
        self._session_dir: Optional[Path] = None
        self._trigger_type: str = ""
        self._start_ts: float = 0.0
        self._frame_count: int = 0
        self._state_lock = threading.Lock()

        # ── Video writer ──────────────────────────────────────────────
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_lock = threading.Lock()

        # ── Aksiyon kuyruğu (capture thread → disk writer) ───────────
        self._action_queue: Queue[ActionEvent] = Queue(maxsize=ACTION_QUEUE_LIMIT)
        self._action_file = None
        self._action_file_lock = threading.Lock()

        # ── Debounce state ────────────────────────────────────────────
        self._last_mouse_ts: float = 0.0
        self._last_key_ts: float = 0.0
        self._debounce_lock = threading.Lock()

        # ── Thread kontrol ────────────────────────────────────────────
        self._stop_event = threading.Event()

        # Capture thread: ekran → video
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="VideoRec-Capture",
        )
        self._capture_thread.start()

        # Action writer thread: kuyruk → jsonl
        self._action_writer_thread = threading.Thread(
            target=self._action_writer_loop,
            daemon=True,
            name="VideoRec-ActionWriter",
        )
        self._action_writer_thread.start()

        # pynput dinleyicileri → Artık UserInputMonitor üzerinden subscriber
        self._mouse_listener = None      # Geriye uyumluluk (shutdown kontrolü)
        self._keyboard_listener = None   # Geriye uyumluluk (shutdown kontrolü)
        self._subscribe_to_input_hub()

        bot.log(
            f"VideoRecorder: Hazır (enabled={self._enabled}, "
            f"fps={self._target_fps}, root={self._data_root})"
        )

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, trigger_type: str = "boss_attack") -> bool:
        """
        Yeni video kayıt oturumu başlat.

        Returns:
            True: Kayıt başarıyla başladı
            False: Devre dışı veya zaten kayıtta
        """
        if not self._enabled:
            self.bot.log("VideoRecorder: Devre dışı, kayıt başlatılmadı.", level="DEBUG")
            return False

        with self._state_lock:
            if self._recording:
                self.bot.log("VideoRecorder: Zaten kayıtta, tekrar başlatılamaz.", level="DEBUG")
                return False

            now = datetime.now()
            self._session_id = now.strftime("VID_%Y%m%d_%H%M%S")
            self._session_dir = self._data_root / self._session_id
            self._trigger_type = trigger_type
            self._start_ts = time.time()
            self._frame_count = 0
            self._resolution_locked = False

        # Oturum klasörünü oluştur
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.bot.log(
                f"VideoRecorder: Klasör oluşturulamadı: {exc}",
                level="WARNING",
            )
            return False

        # Action log dosyasını aç
        self._open_action_file()

        with self._state_lock:
            self._recording = True

        self.bot.log(
            f"VideoRecorder: ▶ Kayıt başladı | oturum={self._session_id} | "
            f"tetikleyici={trigger_type}"
        )
        return True

    def stop(self, success: bool = True, reason: str = "") -> bool:
        """
        Aktif kaydı durdur, video ve metadata'yı finalize et.

        Args:
            success: Boss saldırısı başarılı mı?
            reason: Ek açıklama
        """
        with self._state_lock:
            if not self._recording:
                return False
            self._recording = False
            session_id = self._session_id
            session_dir = self._session_dir
            frame_count = self._frame_count
            start_ts = self._start_ts

        # Video writer'ı kapat
        self._close_writer()

        # Kuyruktaki kalan aksiyonları diske yaz
        self._flush_action_queue()
        self._close_action_file()

        # Oturum metadata'sını yaz
        end_ts = time.time()
        duration_sec = end_ts - start_ts
        self._write_session_meta(
            session_dir=session_dir,
            session_id=session_id,
            success=success,
            reason=reason,
            frame_count=frame_count,
            duration_sec=duration_sec,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        self.bot.log(
            f"VideoRecorder: ⏹ Kayıt durduruldu | oturum={session_id} | "
            f"{'✓ BAŞARILI' if success else '✗ BAŞARISIZ'} | "
            f"{frame_count} kare | {duration_sec:.1f}s | sebep={reason}"
        )
        return True

    def log_action(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "bot",
        action_label: str = "",
        phase: Optional[str] = None,
    ) -> None:
        """
        Bir aksiyonu video frame'i ile eşleştirerek logla.

        Bu metod bot'un kendi aksiyonları için kullanılır (tıklama, tuş).
        Kullanıcı eylemleri pynput hook'ları tarafından otomatik loglanır.

        Args:
            event_type: "bot_click", "bot_key", "bot_action", vs.
            data: {"x": 100, "y": 200} veya {"key": "a"} gibi ek veri
            source: "bot" veya "user"
            action_label: "attack_boss", "select_layer" gibi etiket
            phase: Oyun fazı (None ise otomatik algılanır)
        """
        if not self._recording:
            return

        with self._state_lock:
            frame_idx = self._frame_count
            start_ts = self._start_ts

        ts_now = time.time()
        ts_video = ts_now - start_ts
        resolved_phase = phase or self._get_current_phase()

        event = ActionEvent(
            frame_idx=frame_idx,
            ts_video_sec=round(ts_video, 4),
            ts_unix=round(ts_now, 6),
            event_type=event_type,
            data=data,
            source=source,
            phase=resolved_phase,
            action_label=action_label,
            # 1 kare öncesi ve sonrası (eğitim verisi için)
            frame_before=max(0, frame_idx - 1),
            frame_after=frame_idx + 1,  # sonraki kare henüz yok, tahmini
        )

        try:
            self._action_queue.put_nowait(event)
        except Exception:
            # Kuyruk doluysa logla ama çökme
            self.bot.log(
                "VideoRecorder: Aksiyon kuyruğu dolu, event atlandı.",
                level="WARNING",
            )

    def shutdown(self) -> None:
        """Graceful kapanış — uygulama çıkışında çağrılır."""
        if self._recording:
            self.stop(success=False, reason="SHUTDOWN")

        self._stop_event.set()

        # UserInputMonitor'dan aboneliği kaldır
        monitor = getattr(self.bot, "user_monitor", None)
        if monitor is not None:
            try:
                monitor.unsubscribe("video_recorder")
            except Exception:
                pass

        self.bot.log("VideoRecorder: Kapatıldı.")

    def get_statistics(self) -> Dict[str, Any]:
        """Anlık istatistikler (GUI için)."""
        with self._state_lock:
            return {
                "recording": self._recording,
                "session_id": self._session_id,
                "frame_count": self._frame_count,
                "fps": self._target_fps,
                "resolution": f"{self._frame_width}x{self._frame_height}",
                "action_queue_size": self._action_queue.qsize(),
                "elapsed_sec": round(time.time() - self._start_ts, 1) if self._recording else 0,
            }

    # ══════════════════════════════════════════════════════════════════
    #  CAPTURE THREAD
    # ══════════════════════════════════════════════════════════════════

    def _capture_loop(self) -> None:
        """
        Ana capture döngüsü — hedef FPS'te ekran yakalar ve videoya yazar.

        Performans notları:
        - sleep ile FPS sabitleme (busy-wait değil)
        - Frame copy sadece writer aktifse
        - Çözünürlük değişirse writer yeniden oluşturulur
        """
        interval = 1.0 / self._target_fps

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            if self._recording and self._has_vision():
                try:
                    frame = self.bot.vision.capture_full_screen()
                except Exception:
                    frame = None

                if frame is not None:
                    self._process_frame(frame)

            # FPS sabitleme: süre farkı kadar bekle
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Tek bir frame'i işle: renk dönüşümü, çözünürlük kontrolü, yazma.
        """
        # BGR'ye dönüştür (gerekirse)
        bgr_frame = self._ensure_bgr(frame)

        h, w = bgr_frame.shape[:2]

        # Çözünürlük değişimi kontrolü
        if not self._resolution_locked:
            self._frame_width = w
            self._frame_height = h
            self._resolution_locked = True
            self._init_writer()
        elif w != self._frame_width or h != self._frame_height:
            # Çözünürlük değişti — writer'ı yeniden oluştur
            self.bot.log(
                f"VideoRecorder: Çözünürlük değişti "
                f"({self._frame_width}x{self._frame_height} → {w}x{h}), "
                f"writer yeniden başlatılıyor.",
                level="INFO",
            )
            self._close_writer()
            self._frame_width = w
            self._frame_height = h
            self._init_writer()

        # Frame'i videoya yaz
        with self._writer_lock:
            if self._writer is not None and self._writer.isOpened():
                try:
                    self._writer.write(bgr_frame)
                    with self._state_lock:
                        self._frame_count += 1
                except Exception as exc:
                    self.bot.log(
                        f"VideoRecorder: Frame yazma hatası: {exc}",
                        level="WARNING",
                    )

    def _ensure_bgr(self, frame: np.ndarray) -> np.ndarray:
        """Frame'i VideoWriter için BGR uint8 formatına dönüştür."""
        out = frame
        if out.dtype != np.uint8:
            out = out.astype(np.uint8, copy=False)

        if out.ndim == 3:
            ch = out.shape[2]
            if ch == 3 and self._frame_input_color == "RGB":
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            elif ch == 4:
                code = (
                    cv2.COLOR_RGBA2BGR
                    if self._frame_input_color == "RGB"
                    else cv2.COLOR_BGRA2BGR
                )
                out = cv2.cvtColor(out, code)
        return out

    # ══════════════════════════════════════════════════════════════════
    #  VIDEO WRITER YÖNETİMİ
    # ══════════════════════════════════════════════════════════════════

    def _init_writer(self) -> None:
        """VideoWriter'ı mevcut çözünürlükle başlat."""
        if self._session_dir is None:
            return

        video_path = str(self._session_dir / "video.mp4")

        with self._writer_lock:
            # Önce H.264 dene
            self._writer = cv2.VideoWriter(
                video_path,
                FOURCC_H264,
                self._target_fps,
                (self._frame_width, self._frame_height),
            )
            if not self._writer.isOpened():
                # Fallback: XVID → .avi
                self.bot.log(
                    "VideoRecorder: H.264 başarısız, XVID fallback deneniyor.",
                    level="WARNING",
                )
                video_path_avi = str(self._session_dir / "video.avi")
                self._writer = cv2.VideoWriter(
                    video_path_avi,
                    FOURCC_FALLBACK,
                    self._target_fps,
                    (self._frame_width, self._frame_height),
                )
                if not self._writer.isOpened():
                    self.bot.log(
                        "VideoRecorder: Video writer açılamadı!",
                        level="ERROR",
                    )
                    self._writer = None

        if self._writer is not None:
            self.bot.log(
                f"VideoRecorder: Writer başlatıldı "
                f"({self._frame_width}x{self._frame_height} @ {self._target_fps}fps)",
                level="DEBUG",
            )

    def _close_writer(self) -> None:
        """VideoWriter'ı güvenle kapat."""
        with self._writer_lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception as exc:
                    self.bot.log(
                        f"VideoRecorder: Writer kapatma hatası: {exc}",
                        level="WARNING",
                    )
                self._writer = None

    # ══════════════════════════════════════════════════════════════════
    #  ACTION LOG YÖNETİMİ
    # ══════════════════════════════════════════════════════════════════

    def _open_action_file(self) -> None:
        """actions.jsonl dosyasını aç."""
        with self._action_file_lock:
            if self._session_dir is None:
                return
            try:
                path = self._session_dir / "actions.jsonl"
                self._action_file = open(path, "w", encoding="utf-8", buffering=1)
            except OSError as exc:
                self.bot.log(
                    f"VideoRecorder: Action dosyası açılamadı: {exc}",
                    level="WARNING",
                )
                self._action_file = None

    def _close_action_file(self) -> None:
        """actions.jsonl dosyasını kapat."""
        with self._action_file_lock:
            if self._action_file is not None:
                try:
                    self._action_file.close()
                except Exception:
                    pass
                self._action_file = None

    def _action_writer_loop(self) -> None:
        """Aksiyonları kuyruktan alıp jsonl'e yazan arka plan thread'i."""
        while not self._stop_event.is_set():
            try:
                event: ActionEvent = self._action_queue.get(timeout=0.3)
            except Empty:
                continue

            try:
                self._write_action_event(event)
            except Exception as exc:
                self.bot.log(
                    f"VideoRecorder: Action yazma hatası: {exc}",
                    level="WARNING",
                )
            finally:
                self._action_queue.task_done()

    def _write_action_event(self, event: ActionEvent) -> None:
        """Tek bir ActionEvent'i jsonl satırı olarak yaz."""
        record = {
            "frame_idx": event.frame_idx,
            "frame_before": event.frame_before,
            "frame_after": event.frame_after,
            "ts_video_sec": event.ts_video_sec,
            "ts_unix": event.ts_unix,
            "event_type": event.event_type,
            "data": event.data,
            "source": event.source,
            "phase": event.phase,
            "action_label": event.action_label,
        }

        with self._action_file_lock:
            if self._action_file is not None:
                try:
                    self._action_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                except Exception:
                    pass

    def _flush_action_queue(self) -> None:
        """Kuyruktaki tüm bekleyen aksiyonları hemen diske yaz."""
        flushed = 0
        while True:
            try:
                event = self._action_queue.get_nowait()
            except Empty:
                break
            try:
                self._write_action_event(event)
            finally:
                self._action_queue.task_done()
                flushed += 1

        if flushed > 0:
            self.bot.log(
                f"VideoRecorder: {flushed} bekleyen aksiyon diske yazıldı.",
                level="DEBUG",
            )

    # ══════════════════════════════════════════════════════════════════
    #  SESSION METADATA
    # ══════════════════════════════════════════════════════════════════

    def _write_session_meta(
        self,
        session_dir: Path,
        session_id: str,
        success: bool,
        reason: str,
        frame_count: int,
        duration_sec: float,
        start_ts: float,
        end_ts: float,
    ) -> None:
        """Oturum sonunda özet metadata dosyası yaz."""
        meta = {
            "session_id": session_id,
            "trigger_type": self._trigger_type,
            "success": success,
            "reason": reason,
            "frame_count": frame_count,
            "fps": self._target_fps,
            "duration_sec": round(duration_sec, 2),
            "resolution": f"{self._frame_width}x{self._frame_height}",
            "start_ts_unix": round(start_ts, 6),
            "end_ts_unix": round(end_ts, 6),
            "start_ts_iso": datetime.fromtimestamp(start_ts).isoformat(
                timespec="milliseconds"
            ),
            "end_ts_iso": datetime.fromtimestamp(end_ts).isoformat(
                timespec="milliseconds"
            ),
            "video_file": self._find_video_filename(session_dir),
        }

        try:
            meta_path = session_dir / "session_meta.json"
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self.bot.log(
                f"VideoRecorder: Meta yazma hatası: {exc}",
                level="WARNING",
            )

    def _find_video_filename(self, session_dir: Path) -> str:
        """Oturum klasöründeki video dosyasını bul."""
        for ext in ("mp4", "avi"):
            path = session_dir / f"video.{ext}"
            if path.exists():
                return f"video.{ext}"
        return "video.mp4"

    # ══════════════════════════════════════════════════════════════════
    #  INPUT SUBSCRIPTION (UserInputMonitor Observer)
    # ══════════════════════════════════════════════════════════════════

    def _subscribe_to_input_hub(self) -> None:
        """
        UserInputMonitor'a abone ol.
        Tüm fare/klavye olayları _on_input_event callback'ine gelir.
        Not: UserInputMonitor, bot init sırasında bu modülden ÖNCE
        başlatılmıyorsa, gecikmeli bağlantı denenecektir.
        """
        monitor = getattr(self.bot, "user_monitor", None)
        if monitor is not None and hasattr(monitor, "subscribe"):
            monitor.subscribe("video_recorder", self._on_input_event)
            self.bot.log("VideoRecorder: UserInputMonitor'a subscribe oldu.", level="DEBUG")
        else:
            # Bot init sırası: user_monitor henüz oluşmamışsa
            # loabot_main.py'de video_recorder user_monitor'dan sonra
            # bağlanacak şekilde _late_subscribe tetiklenir.
            self.bot.log(
                "VideoRecorder: UserInputMonitor henüz hazır değil — "
                "late_subscribe bekliyor.",
                level="DEBUG",
            )

    def late_subscribe(self) -> None:
        """
        Bot init sırası sorunu için: user_monitor hazır olduktan
        sonra loabot_main.py tarafından çağrılır.
        """
        self._subscribe_to_input_hub()

    def _on_input_event(self, event) -> None:
        """
        UserInputMonitor'dan gelen InputEvent callback'i.
        Kayıt aktifse video frame'i ile eşleştirerek loglar.
        """
        if not self._recording:
            return

        # Kullanıcı olaylarını sadece müdahale niteliğindeyse veri setine yaz.
        if event.source == "user":
            data = dict(getattr(event, "data", {}) or {})
            if not bool(data.get("is_intervention", False)):
                return

        action_label = f"{event.source}_{event.event_type}"
        if event.source == "user":
            action_label = f"user_intervention_{event.event_type}"

        # InputEvent -> log_action dönüşümü
        self.log_action(
            event_type=event.event_type,
            data=dict(event.data),
            source=event.source,
            action_label=action_label,
            phase=event.phase,
        )

    # ══════════════════════════════════════════════════════════════════
    #  YARDIMCILAR
    # ══════════════════════════════════════════════════════════════════

    def _has_vision(self) -> bool:
        """VisionManager erişilebilir mi?"""
        return hasattr(self.bot, "vision") and self.bot.vision is not None

    def _get_current_phase(self) -> str:
        """Aktif oyun fazını al."""
        try:
            if self._has_vision():
                ns = getattr(self.bot.vision, "_mission_namespace", None)
                if ns:
                    return str(ns)
        except Exception:
            pass
        return getattr(self.bot, "_global_phase", "UNKNOWN_PHASE")

    @staticmethod
    def _coerce_bool(value) -> bool:
        """Farklı tiplerden bool dönüşümü."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
