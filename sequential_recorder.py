"""
sequential_recorder.py — 10 FPS Video Sekans Tabanlı Eğitim Veri Kaydedici
===========================================================================
Sabit 10 FPS ile döngüsel bir çerçeve tamponu (sliding window) tutar.
Bir aksiyon algılandığında mevcut [T-9...T] penceresini mühürler (primary seal).
Ardından, 1 saniyelik 10 çerçevelik süreklilik (persistence) bloğu aynı
etiketle otomatik kaydedilir.

Dizin yapısı (CNN+LSTM / TimeSformer'a hazır):
  <data_root>/sequences/
  └── SESSION_YYYYMMDD_HHMMSS/
      ├── Sequence_0001/
      │   ├── frame_00.jpg ... frame_09.jpg
      │   └── metadata.json
      ├── Sequence_0002/
      ...

metadata.json şeması:
  {
    "sequence_id"   : "Sequence_0001",
    "session_id"    : "SESSION_...",
    "trigger_type"  : "manual_user",  # manual_user | auto_rule_engine
    "phase"         : "NAV_PHASE",
    "action_label"  : "mouse_click",
    "action_source" : "primary",     # primary | persist
    "stuck"         : false,         # Error_Stuck
    "idle"          : false,
    "ts_seal_unix"  : 1234567890.123,
    "ts_seal_iso"   : "2026-02-21T12:34:56.789",
    "num_frames"    : 10,
    "fps"           : 10,
    "char_position" : {"x": 0.0, "y": 0.0},
    "frames"        : [
      {"filename": "frame_00.jpg", "ts_unix": ..., "ts_iso": "..."},
      ...
    ]
  }
"""

from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Sabitler ──────────────────────────────────────────────────────────────

TARGET_FPS: int = 10
FRAME_INTERVAL_SEC: float = 1.0 / TARGET_FPS       # 100 ms
WINDOW_SIZE: int = 10                               # Kayar pencere genişliği
PERSIST_FRAMES: int = 10                            # Aksiyon sonrası süreklilik karesi
IDLE_THRESHOLD_SEC: float = 2.0                     # Boşta eşiği
STUCK_THRESHOLD_SEC: float = 2.0                    # Takılma eşiği
STUCK_MIN_PIXEL_DELTA: float = 3.0                  # Minimum hareket (piksel)
JPEG_QUALITY: int = 92                              # Kayıt kalitesi

# Koordinat değişikliği = "yürüme niyeti" olan aksiyonlar
_WALK_ACTIONS: frozenset = frozenset({
    "mouse_click",
    "seq_boss_secimi",
    "seq_katman_secimi",
})


# ── Veri Yapıları ─────────────────────────────────────────────────────────

@dataclass
class SealEvent:
    """Bir mühürleme olayını temsil eder; seal_worker thread'i tarafından işlenir."""
    frames: List[Tuple[np.ndarray, float]]  # [(frame_bgr, ts_unix), ...]
    action_label: str
    phase: str
    action_source: str                       # "primary" | "persist"
    ts_seal_unix: float
    char_position: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
#  Ana Kaydedici
# ══════════════════════════════════════════════════════════════════════════

class SequentialRecorder:
    """
    10 FPS video sekansı tabanlı eğitim veri kaydedici.

    Entegrasyon noktaları:
      - bot.seq_recorder.start()           → kayıt başlat (toggle_recording)
      - bot.seq_recorder.stop()            → kayıt durdur (toggle_recording)
      - bot.seq_recorder.seal_action(...)  → aksiyon oldu (Automator)
      - bot.seq_recorder.update_position(x, y) → koordinat güncelle (opsiyonel)
      - bot.seq_recorder.shutdown()        → bot kapanıyor
    """

    def __init__(self, bot):
        self.bot = bot

        # Ayarlardan konfigürasyon
        self._data_root = Path(
            bot.settings.get("SEQUENTIAL_RECORDER_ROOT",
                             r"E:\LoABot_Training_Data\sequences")
        )
        self._jpeg_quality = int(bot.settings.get("SHADOW_JPEG_QUALITY", JPEG_QUALITY))

        # Kayıt durumu
        self._recording: bool = False
        self._trigger_type: str = "manual_user"
        self._session_id: str = ""
        self._session_dir: Optional[Path] = None
        self._seq_counter: int = 0
        self._state_lock = threading.Lock()

        # Döngüsel çerçeve tamponu: deque[(frame_bgr, ts_unix)]
        self._buffer: Deque[Tuple[np.ndarray, float]] = collections.deque(maxlen=WINDOW_SIZE)
        self._buffer_lock = threading.Lock()

        # Seal kuyruğu (üretici: capture/seal_action → tüketici: seal_worker)
        self._seal_queue: Queue[SealEvent] = Queue(maxsize=512)

        # Persist durumu
        self._persist_action: Optional[str] = None
        self._persist_phase: str = "UNKNOWN_PHASE"
        self._persist_frames: List[Tuple[np.ndarray, float]] = []
        self._persist_lock = threading.Lock()

        # Boşta / takılma takibi
        self._last_input_ts: float = 0.0
        self._last_pos_change_ts: float = 0.0
        self._char_x: float = 0.0
        self._char_y: float = 0.0
        self._pos_lock = threading.Lock()

        # Durdurma işareti
        self._stop_event = threading.Event()

        # Fiziksel fare dinleyicisi (pynput)
        self._mouse_listener = None

        # Daemon thread'leri başlat
        threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="SeqRec-Capture",
        ).start()
        threading.Thread(
            target=self._seal_worker_loop,
            daemon=True,
            name="SeqRec-SealWorker",
        ).start()

        self._start_mouse_listener()
        bot.log("SequentialRecorder: 10 FPS sekans kaydedici hazir.")

    # ── Dışa Açık API ─────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, trigger_type: str = "manual_user"):
        """Yeni bir kayıt oturumu başlatır.

        Args:
            trigger_type: Oturumu kim baslatti.
                          "manual_user"      — GUI butonu (usta)
                          "auto_rule_engine" — Kural motoru (bot)
        """
        with self._state_lock:
            if self._recording:
                return

            self._recording = True
            self._trigger_type = trigger_type
            now = datetime.now()
            self._session_id = now.strftime("SESSION_%Y%m%d_%H%M%S")
            self._session_dir = self._data_root / self._session_id
            self._seq_counter = 0

        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.bot.log(f"SequentialRecorder: Klasor olusturulamadi: {exc}", level="ERROR")
            with self._state_lock:
                self._recording = False
            return

        # Tampon ve persist durumunu sıfırla
        with self._buffer_lock:
            self._buffer.clear()
        with self._persist_lock:
            self._persist_action = None
            self._persist_frames = []

        self._last_input_ts = time.monotonic()
        self._last_pos_change_ts = 0.0

        self.bot.log(f"SequentialRecorder: Kayit basladi. Oturum: {self._session_id}")

    def stop(self):
        """Kaydı durdurur; kalan seal olaylarını işler."""
        with self._state_lock:
            if not self._recording:
                return
            self._recording = False

        # Persist temizliği
        with self._persist_lock:
            self._persist_action = None
            self._persist_frames = []

        # --- DEĞİŞTİRİLEN KISIM: GUI'yi kilitlememek için join() kaldırıldı ---
        # Artık GUI'yi bekletmiyoruz. _seal_worker_loop daemon thread olduğu için
        # kuyrukta kalanlar arka planda diske yazılmaya devam edecek.
        # self._seal_queue.join() komutunu SİLDİK.
        # -----------------------------------------------------------------------

        with self._state_lock:
            seq_count = self._seq_counter
            session_dir = self._session_dir

        self.bot.log(
            f"SequentialRecorder: Kayit durduruldu. "
            f"{seq_count} sekans kaydedildi. "
            f"Klasor: {session_dir}"
        )

    def seal_action(
        self,
        action_label: str,
        phase: Optional[str] = None,
        extra: Optional[Dict] = None,
    ):
        """
        Bir aksiyon gerçekleştiğinde Automator tarafından çağrılır.

        1. Mevcut 10 karelik tamponu mühürler (primary seal).
        2. Sonraki 10 karelik persist bloğunu otomatik etiketler.
        """
        if not self._recording:
            return

        self._last_input_ts = time.monotonic()

        with self._buffer_lock:
            frames = list(self._buffer)

        if not frames:
            return  # Tampon henüz dolmadı

        resolved_phase = phase or self._get_current_phase()
        char_pos = self._get_char_position()
        idle = self._is_idle()

        event = SealEvent(
            frames=frames,
            action_label=action_label,
            phase=resolved_phase,
            action_source="primary",
            ts_seal_unix=time.time(),
            char_position=char_pos,
            extra={"idle": idle, **(extra or {})},
        )

        try:
            self._seal_queue.put_nowait(event)
        except Full:
            self.bot.log(
                "SequentialRecorder: Seal kuyrugu dolu, primary event atlandi.",
                level="WARNING",
            )
            return

        # Persist modunu aktifleştir — capture_loop devam edecek
        with self._persist_lock:
            self._persist_action = action_label
            self._persist_phase = resolved_phase
            self._persist_frames = []

    def update_position(self, x: float, y: float):
        """
        Karakter koordinatı güncellemesi.
        Yeterince büyük hareket varsa last_pos_change_ts güncellenir.
        """
        with self._pos_lock:
            dx = abs(x - self._char_x)
            dy = abs(y - self._char_y)
            if dx > STUCK_MIN_PIXEL_DELTA or dy > STUCK_MIN_PIXEL_DELTA:
                self._char_x = x
                self._char_y = y
                self._last_pos_change_ts = time.monotonic()

    def shutdown(self):
        """Bot kapanırken temiz kapatma."""
        self.stop()
        self._stop_event.set()
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass

    def get_statistics(self) -> Dict:
        with self._state_lock:
            return {
                "recording": self._recording,
                "session_id": self._session_id,
                "sequences_saved": self._seq_counter,
                "seal_queue_size": self._seal_queue.qsize(),
            }

    # ── Fiziksel Fare Dinleyicisi ─────────────────────────────────────────

    def _start_mouse_listener(self):
        """
        pynput ile fiziksel sol tıklamaları dinler.
        Bot kendi tıklamalarını utils.BOT_IS_CLICKING_EVENT ile işaretler;
        o event set'liyken gelen tıklamalar filtrelenir.
        """
        try:
            from pynput import mouse as _pynput_mouse
            from utils import BOT_IS_CLICKING_EVENT as _BOT_EVT

            def on_click(x, y, button, pressed):
                if not pressed:
                    return  # Sadece basış anı
                if button != _pynput_mouse.Button.left:
                    return  # Sadece sol tık
                if not self._recording:
                    return  # Kayıt aktif değil
                if _BOT_EVT.is_set():
                    return  # Bot'un kendi pyautogui tıklaması — filtrele

                # Kullanıcının fiziksel tıklaması → anında sekans mühürle
                self.seal_action(
                    action_label="manual_mouse_click",
                    extra={
                        "trigger_type": "manual_user",
                        "click_point": {"x": int(x), "y": int(y)},
                    },
                )

            listener = _pynput_mouse.Listener(on_click=on_click)
            listener.daemon = True
            listener.start()
            self._mouse_listener = listener
            self.bot.log("SequentialRecorder: Fiziksel tik dinleyicisi aktif.")
        except ImportError:
            self.bot.log(
                "pynput yuklu degil — 'pip install pynput' ile kurun. "
                "Fiziksel tik kaydedilmeyecek.",
                level="WARNING",
            )
        except Exception as exc:
            self.bot.log(f"Mouse listener baslatilamadi: {exc}", level="WARNING")

    # ── Capture Thread (10 FPS) ───────────────────────────────────────────

    def _capture_loop(self):
        """
        10 FPS'te çalışan ana yakalama döngüsü.
        - Rolling frame buffer'ı doldurur.
        - Persist mode aktifse yeni kareleri toplar; 10'a ulaşınca otomatik mühürler.
        """
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            if self._recording and self._has_vision():
                try:
                    frame = self.bot.vision.capture_full_screen()
                except Exception:
                    frame = None

                if frame is not None:
                    ts = time.time()

                    # Rolling buffer
                    with self._buffer_lock:
                        self._buffer.append((frame.copy(), ts))

                    # Persist collector
                    with self._persist_lock:
                        if self._persist_action is not None:
                            self._persist_frames.append((frame.copy(), ts))

                            if len(self._persist_frames) >= PERSIST_FRAMES:
                                self._flush_persist_block()

            # Sabit 10 FPS ritmi koru
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, FRAME_INTERVAL_SEC - elapsed)
            time.sleep(sleep_for)

    def _flush_persist_block(self):
        """
        Persist tamponu dolduğunda çağrılır (_persist_lock alınmış olmalı).
        """
        frames = list(self._persist_frames)
        action = self._persist_action
        phase = self._persist_phase

        # Persist durumunu sıfırla
        self._persist_action = None
        self._persist_frames = []

        # Takılma kontrolü
        stuck = self._is_stuck(action)

        event = SealEvent(
            frames=frames,
            action_label=action,
            phase=phase,
            action_source="persist",
            ts_seal_unix=time.time(),
            char_position=self._get_char_position(),
            extra={"stuck": stuck},
        )

        try:
            self._seal_queue.put_nowait(event)
        except Full:
            self.bot.log(
                "SequentialRecorder: Seal kuyrugu dolu, persist block atlandi.",
                level="WARNING",
            )

    # ── Seal Worker Thread ────────────────────────────────────────────────

    def _seal_worker_loop(self):
        """Seal kuyruğunu işler ve dizinlere yazar."""
        while not self._stop_event.is_set():
            try:
                event: SealEvent = self._seal_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self._package_sequence(event)
            except Exception as exc:
                self.bot.log(
                    f"SequentialRecorder: Sekans yazma hatasi: {exc}",
                    level="WARNING",
                )
            finally:
                self._seal_queue.task_done()

    def _package_sequence(self, event: SealEvent):
        """
        SealEvent → Sequence_XXXX/ klasörü + metadata.json yazar.
        """
        if not event.frames:
            return

        with self._state_lock:
            self._seq_counter += 1
            seq_num = self._seq_counter
            session_dir = self._session_dir

        if session_dir is None:
            return

        seq_dir = session_dir / f"Sequence_{seq_num:04d}"
        try:
            seq_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.bot.log(f"SequentialRecorder: Klasor olusturulamadi: {exc}", level="WARNING")
            return

        # Kare kayıtları
        frames_meta: List[Dict] = []
        for i, (frame_bgr, ts) in enumerate(event.frames):
            filename = f"frame_{i:02d}.jpg"
            out_path = seq_dir / filename
            try:
                cv2.imwrite(
                    str(out_path),
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
                )
            except Exception:
                pass
            frames_meta.append({
                "filename": filename,
                "ts_unix": round(ts, 6),
                "ts_iso": datetime.fromtimestamp(ts).isoformat(timespec="milliseconds"),
            })

        # Takılma etiketini normalleştir
        action_final = event.action_label
        if event.extra.get("stuck") and event.action_label in _WALK_ACTIONS:
            action_final = "Error_Stuck"

        # extra["trigger_type"] varsa oturum düzeyini geçersiz kılar
        # (fiziksel tıklamalar oto-oturumda bile "manual_user" olarak işaretlenir)
        effective_trigger = event.extra.get("trigger_type", self._trigger_type)

        metadata = {
            "sequence_id": f"Sequence_{seq_num:04d}",
            "session_id": self._session_id,
            "trigger_type": effective_trigger,
            "phase": event.phase,
            "action_label": action_final,
            "action_source": event.action_source,
            "stuck": bool(event.extra.get("stuck", False)),
            "idle": bool(event.extra.get("idle", False)),
            "ts_seal_unix": round(event.ts_seal_unix, 6),
            "ts_seal_iso": datetime.fromtimestamp(event.ts_seal_unix).isoformat(timespec="milliseconds"),
            "num_frames": len(event.frames),
            "fps": TARGET_FPS,
            "char_position": event.char_position,
            "frames": frames_meta,
        }

        # Fiziksel tık koordinatı — sadece manual_mouse_click seallerinde mevcuttur
        if event.extra.get("click_point"):
            metadata["click_point"] = event.extra["click_point"]

        try:
            (seq_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self.bot.log(f"SequentialRecorder: metadata.json yazma hatasi: {exc}", level="WARNING")
            return

        self.bot.log(
            f"[SeqRec] Sequence_{seq_num:04d} | "
            f"phase={event.phase} | "
            f"aksiyon={action_final}({event.action_source}) | "
            f"{len(event.frames)} kare"
            + (" [STUCK]" if event.extra.get("stuck") else "")
            + (" [IDLE]" if event.extra.get("idle") else ""),
            level="DEBUG",
        )

    # ── Yardımcı Metodlar ─────────────────────────────────────────────────

    def _has_vision(self) -> bool:
        return hasattr(self.bot, "vision") and self.bot.vision is not None

    def _get_current_phase(self) -> str:
        """Vision manager'dan aktif faz adını okur."""
        try:
            if self._has_vision():
                ns = getattr(self.bot.vision, "_mission_namespace", None)
                if ns:
                    return str(ns)
        except Exception:
            pass
        return "UNKNOWN_PHASE"

    def _get_char_position(self) -> Dict[str, float]:
        with self._pos_lock:
            return {"x": self._char_x, "y": self._char_y}

    def _is_idle(self) -> bool:
        """Hem koordinat hem girdi 2 saniyedir değişmemişse True.
        Pozisyon hiç güncellenmemişse (update_position çağrılmadıysa)
        yalnızca girdi timeout'una göre karar verir.
        """
        now = time.monotonic()
        input_idle = now - self._last_input_ts > IDLE_THRESHOLD_SEC
        if self._last_pos_change_ts <= 0:
            # Pozisyon verisi yok — sadece girdi zaman aşımına bak
            return input_idle
        pos_idle = now - self._last_pos_change_ts > IDLE_THRESHOLD_SEC
        return input_idle and pos_idle

    def _is_stuck(self, action_label: str) -> bool:
        """
        Nav aksiyonu aktifken son {STUCK_THRESHOLD_SEC} saniyede
        pozisyon değişmemişse True döner.
        """
        if action_label not in _WALK_ACTIONS:
            return False
        if self._last_pos_change_ts <= 0:
            return False
        with self._pos_lock:
            return time.monotonic() - self._last_pos_change_ts > STUCK_THRESHOLD_SEC
