"""
sequential_recorder.py

RAM-first sequential recorder for Agentic AI training data.

Core behavior:
- Captures frames at fixed FPS into a rolling window.
- On action seal, stores T-window + T+persist events in RAM (no disk write).
- Writes to disk asynchronously only after SUCCESS/FLUSH signal.
- On FAIL, clears RAM and pending seal queue without touching disk.
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

TARGET_FPS: int = 10
FRAME_INTERVAL_SEC: float = 1.0 / TARGET_FPS
WINDOW_SIZE: int = 10
PERSIST_FRAMES: int = 10
IDLE_THRESHOLD_SEC: float = 2.0
STUCK_THRESHOLD_SEC: float = 2.0
STUCK_MIN_PIXEL_DELTA: float = 3.0
JPEG_QUALITY: int = 92
DEFAULT_RECORDER_ROOT: str = r"D:\LoABot_Training_Data\sequences"
DEFAULT_RAM_FRAME_LIMIT: int = 100

# Actions expected to move character position.
_WALK_ACTIONS: frozenset[str] = frozenset({
    "mouse_click",
    "seq_boss_secimi",
    "seq_katman_secimi",
})


@dataclass
class SealEvent:
    """Single action event captured from rolling/persist windows."""

    session_id: str
    frames: List[Tuple[np.ndarray, float]]  # (frame_bgr, ts_unix)
    action_label: str
    phase: str
    action_source: str  # primary | persist
    ts_seal_unix: float
    char_position: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlushJob:
    """Async disk-write job containing staged RAM events."""

    session_id: str
    session_dir: Path
    trigger_type: str
    reason: str
    events: List[SealEvent]


class SequentialRecorder:
    """
    10 FPS sequence recorder.

    External API:
    - start(trigger_type)
    - stop()
    - seal_action(action_label, phase=None, extra=None)
    - mark_action_start(action_label, extra=None)   # Aksiyon baslangici: T-window dondurur
    - mark_action_end(action_label, result_status)   # Aksiyon bitisi: tek kare 'termination'
    - update_position(x, y)
    - signal_success(reason)
    - signal_flush(reason)
    - signal_fail(reason)
    - shutdown()
    """

    def __init__(self, bot):
        self.bot = bot

        settings = getattr(bot, "settings", {}) or {}
        general_cfg = getattr(bot, "general_cfg", {}) or {}
        recording_cfg = general_cfg.get("recording", {}) if isinstance(general_cfg, dict) else {}
        seq_cfg = recording_cfg.get("sequential", {}) if isinstance(recording_cfg, dict) else {}

        self._enabled = self._coerce_bool(
            seq_cfg.get("enabled", settings.get("SEQUENTIAL_RECORDER_ENABLED", True))
        )
        self._data_root = Path(
            str(seq_cfg.get("output_path", settings.get("SEQUENTIAL_RECORDER_ROOT", DEFAULT_RECORDER_ROOT)))
        )
        self._jpeg_quality = int(seq_cfg.get("jpeg_quality", settings.get("SHADOW_JPEG_QUALITY", JPEG_QUALITY)))
        self._max_ram_frames = max(
            10,
            int(seq_cfg.get("max_ram_frames", settings.get("SEQUENTIAL_MAX_RAM_FRAMES", DEFAULT_RAM_FRAME_LIMIT))),
        )
        self._flush_on_success_only = self._coerce_bool(
            seq_cfg.get(
                "flush_on_success_only",
                settings.get("SEQUENTIAL_FLUSH_ON_SUCCESS_ONLY", True),
            )
        )
        self._frame_input_color_order = str(
            seq_cfg.get(
                "input_color_order",
                settings.get("SEQUENTIAL_INPUT_COLOR_ORDER", "BGR"),
            )
        ).strip().upper()

        # Session state
        self._recording: bool = False
        self._trigger_type: str = "manual_user"
        self._session_id: str = ""
        self._session_dir: Optional[Path] = None
        self._seq_counter: int = 0
        self._session_accepting_events: bool = False
        self._state_lock = threading.Lock()
        self._session_seq_counters: Dict[str, int] = {}

        # Rolling capture buffer
        self._buffer: Deque[Tuple[np.ndarray, float]] = collections.deque(maxlen=WINDOW_SIZE)
        self._buffer_lock = threading.Lock()

        # Seal queue (capture -> staging)
        self._seal_queue: Queue[SealEvent] = Queue(maxsize=512)

        # Persist state
        self._persist_action: Optional[str] = None
        self._persist_phase: str = "UNKNOWN_PHASE"
        self._persist_frames: List[Tuple[np.ndarray, float]] = []
        self._persist_lock = threading.Lock()

        # RAM staging buffer (circular by frame budget)
        self._ram_events: Deque[SealEvent] = collections.deque()
        self._ram_frames_total: int = 0
        self._ram_lock = threading.Lock()

        # Flush queue (staging -> disk)
        self._flush_queue: Queue[FlushJob] = Queue(maxsize=32)

        # Idle/stuck tracking
        self._last_input_ts: float = 0.0
        self._last_pos_change_ts: float = 0.0
        self._char_x: float = 0.0
        self._char_y: float = 0.0
        self._pos_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._mouse_listener = None

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
        threading.Thread(
            target=self._flush_writer_loop,
            daemon=True,
            name="SeqRec-FlushWriter",
        ).start()

        self._start_mouse_listener()
        bot.log(
            "SequentialRecorder: 10 FPS kaydedici hazir "
            f"(enabled={self._enabled}, RAM limit={self._max_ram_frames} kare, "
            f"flush_on_success_only={self._flush_on_success_only})."
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, trigger_type: str = "manual_user"):
        """Start a new capture session (RAM-first)."""
        if not self._enabled:
            self.bot.log("SequentialRecorder: disabled=true, kayit baslatilmadi.", level="DEBUG")
            return

        with self._state_lock:
            if self._recording:
                return

            self._recording = True
            self._session_accepting_events = True
            self._trigger_type = trigger_type
            now = datetime.now()
            self._session_id = now.strftime("SESSION_%Y%m%d_%H%M%S")
            self._session_dir = self._data_root / self._session_id
            self._seq_counter = 0
            self._session_seq_counters[self._session_id] = 0

        with self._buffer_lock:
            self._buffer.clear()
        with self._persist_lock:
            self._persist_action = None
            self._persist_phase = "UNKNOWN_PHASE"
            self._persist_frames = []
        self._clear_ram_events(log_drop=False)
        self._purge_pending_seal_events(log_drop=False)

        self._last_input_ts = time.monotonic()
        self._last_pos_change_ts = 0.0

        self.bot.log(f"SequentialRecorder: Kayit basladi. Oturum: {self._session_id}")

    def stop(self):
        """Stop session; default behavior is FAIL-safe drop unless flushed."""
        with self._state_lock:
            if not self._recording:
                return
            self._recording = False
            self._session_accepting_events = False

        with self._persist_lock:
            self._persist_action = None
            self._persist_phase = "UNKNOWN_PHASE"
            self._persist_frames = []

        if self._flush_on_success_only:
            dropped = self.signal_fail(reason="STOP", log_result=False)
            stop_msg = f"RAM temizlenen event={dropped}"
        else:
            self._purge_pending_seal_events(log_drop=False)
            flushed = self.signal_flush(reason="STOP_AUTO_FLUSH")
            stop_msg = f"stop_auto_flush={'ok' if flushed else 'empty'}"

        with self._state_lock:
            seq_count = self._seq_counter
            session_dir = self._session_dir

        self.bot.log(
            f"SequentialRecorder: Kayit durduruldu. {seq_count} sekans kaydedildi. "
            f"Klasor: {session_dir} | {stop_msg}"
        )

    def signal_success(self, reason: str = "SUCCESS") -> bool:
        """Success signal from boss/event flow. Flush RAM to async disk writer."""
        return self.signal_flush(reason=reason)

    def signal_flush(self, reason: str = "FLUSH") -> bool:
        """Flush staged RAM events to disk asynchronously."""
        # Seal worker thread gecikmesinde event kaybi olmamasi icin kuyruktaki
        # olaylari flush oncesi senkron sekilde RAM'e tasiyoruz.
        self._drain_seal_queue_to_ram()

        with self._state_lock:
            session_id = self._session_id
            session_dir = self._session_dir
            trigger_type = self._trigger_type

        if not session_id or session_dir is None:
            return False

        with self._ram_lock:
            if not self._ram_events:
                self.bot.log(
                    f"SequentialRecorder: FLUSH atlandi (RAM bos). reason={reason}",
                    level="DEBUG",
                )
                return False
            events = list(self._ram_events)
            event_count = len(events)
            frame_count = self._ram_frames_total
            self._ram_events.clear()
            self._ram_frames_total = 0

        job = FlushJob(
            session_id=session_id,
            session_dir=session_dir,
            trigger_type=trigger_type,
            reason=reason,
            events=events,
        )

        try:
            self._flush_queue.put_nowait(job)
        except Full:
            self.bot.log(
                "SequentialRecorder: Flush queue dolu, veri kaybi riski.",
                level="WARNING",
            )
            return False

        self.bot.log(
            f"SequentialRecorder: FLUSH kuyruga alindi. reason={reason} "
            f"event={event_count} frame={frame_count}",
            level="DEBUG",
        )
        return True

    def signal_fail(self, reason: str = "FAIL", log_result: bool = True) -> int:
        """Fail signal: clear RAM and pending seal events, no disk write."""
        with self._state_lock:
            self._session_accepting_events = False

        dropped_events, dropped_frames = self._clear_ram_events(log_drop=False)
        q_events, q_frames = self._purge_pending_seal_events(log_drop=False)
        total_events = dropped_events + q_events
        total_frames = dropped_frames + q_frames

        if log_result:
            self.bot.log(
                f"SequentialRecorder: FAIL temizligi. reason={reason} "
                f"dropped_events={total_events} dropped_frames={total_frames}",
                level="DEBUG",
            )
        return total_events

    def seal_action(
        self,
        action_label: str,
        phase: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Seal current rolling window and arm T+persist collector."""
        if not self._recording:
            return

        self._last_input_ts = time.monotonic()

        with self._buffer_lock:
            frames = list(self._buffer)

        if not frames:
            return

        with self._state_lock:
            session_id = self._session_id

        resolved_phase = phase or self._get_current_phase()
        char_pos = self._get_char_position()
        idle = self._is_idle()

        event = SealEvent(
            session_id=session_id,
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

        with self._persist_lock:
            self._persist_action = action_label
            self._persist_phase = resolved_phase
            self._persist_frames = []

    def mark_action_start(
        self,
        action_label: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Aksiyon emri verilmeden hemen ONCE cagrılır (fare hareketinden önce).

        O anki rolling buffer'daki son WINDOW_SIZE kareyi 'Action_Initiation'
        etiketiyle dondurur. Olay asenkron seal kuyruğuna yazılır; main thread
        BLOKLANMAZ.

        Disk yazimi yalnizca SUCCESS sinyalinde gerceklesir (SATA korumasi).
        """
        if not self._recording:
            return

        self._last_input_ts = time.monotonic()
        ts_start = time.time()

        with self._buffer_lock:
            frames = [(f.copy(), ts) for f, ts in self._buffer]

        if not frames:
            return

        with self._state_lock:
            session_id = self._session_id

        event = SealEvent(
            session_id=session_id,
            frames=frames,
            action_label=action_label,
            phase=self._get_current_phase(),
            action_source="initiation",
            ts_seal_unix=ts_start,
            char_position=self._get_char_position(),
            extra={
                "Start_TS": round(ts_start, 6),
                "boundary_type": "start",
                **(extra or {}),
            },
        )

        try:
            self._seal_queue.put_nowait(event)
        except Full:
            self.bot.log(
                "SequentialRecorder: Seal kuyrugu dolu, initiation event atlandi.",
                level="WARNING",
            )

    def mark_action_end(
        self,
        action_label: str,
        result_status: str = "unknown",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Aksiyon tamamlanip karakter tekrar stabil/idle hale geldiginde cagrılır.

        Buffer'daki en son TEK kareyi 'Action_Termination' etiketiyle kaydeder.
        Bu tek kare, 'termination anının' görsel kanıtıdır. Main thread BLOKLANMAZ.

        result_status: 'success' | 'fail' | 'unknown'
        Disk yazimi yalnizca SUCCESS sinyalinde gerceklesir (SATA korumasi).
        """
        if not self._recording:
            return

        ts_end = time.time()

        with self._buffer_lock:
            if not self._buffer:
                return
            # Termination anı = buffer'daki en güncel (son) kare
            end_frame, end_frame_ts = self._buffer[-1]
            end_frame = end_frame.copy()

        with self._state_lock:
            session_id = self._session_id

        event = SealEvent(
            session_id=session_id,
            frames=[(end_frame, end_frame_ts)],   # Bilinçli: tek kare yeterli
            action_label=action_label,
            phase=self._get_current_phase(),
            action_source="termination",
            ts_seal_unix=ts_end,
            char_position=self._get_char_position(),
            extra={
                "End_TS": round(ts_end, 6),
                "result_status": str(result_status),
                "boundary_type": "end",
                **(extra or {}),
            },
        )

        try:
            self._seal_queue.put_nowait(event)
        except Full:
            self.bot.log(
                "SequentialRecorder: Seal kuyrugu dolu, termination event atlandi.",
                level="WARNING",
            )

    def update_position(self, x: float, y: float):
        """Update character position and movement timestamp."""
        with self._pos_lock:
            dx = abs(x - self._char_x)
            dy = abs(y - self._char_y)
            if dx > STUCK_MIN_PIXEL_DELTA or dy > STUCK_MIN_PIXEL_DELTA:
                self._char_x = x
                self._char_y = y
                self._last_pos_change_ts = time.monotonic()

    def shutdown(self):
        """Graceful shutdown for app exit."""
        self.stop()
        self._stop_event.set()
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        with self._state_lock:
            session_id = self._session_id
            recording = self._recording
            seq_saved = self._seq_counter
        with self._ram_lock:
            ram_events = len(self._ram_events)
            ram_frames = self._ram_frames_total

        return {
            "recording": recording,
            "session_id": session_id,
            "sequences_saved": seq_saved,
            "seal_queue_size": self._seal_queue.qsize(),
            "flush_queue_size": self._flush_queue.qsize(),
            "ram_events": ram_events,
            "ram_frames": ram_frames,
            "ram_frame_limit": self._max_ram_frames,
        }

    def _start_mouse_listener(self):
        """
        Mouse/keyboard listener was delegated to UserInputMonitor.
        This method remains as compatibility no-op.
        """
        self.bot.log(
            "SequentialRecorder: Fare/klavye dinleyicisi UserInputMonitor'a devredildi.",
            level="DEBUG",
        )

    def _capture_loop(self):
        """Capture loop running at fixed TARGET_FPS."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            if self._recording and self._has_vision():
                frame = None
                try:
                    frame = self.bot.vision.capture_full_screen()
                except Exception:
                    frame = None

                if frame is not None:
                    ts = time.time()

                    with self._buffer_lock:
                        self._buffer.append((frame.copy(), ts))

                    with self._persist_lock:
                        if self._persist_action is not None:
                            self._persist_frames.append((frame.copy(), ts))
                            if len(self._persist_frames) >= PERSIST_FRAMES:
                                self._flush_persist_block()

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, FRAME_INTERVAL_SEC - elapsed))

    def _flush_persist_block(self):
        """Convert persist window into a seal event and queue it."""
        frames = list(self._persist_frames)
        action = self._persist_action
        phase = self._persist_phase

        self._persist_action = None
        self._persist_phase = "UNKNOWN_PHASE"
        self._persist_frames = []

        if not action or not frames:
            return

        with self._state_lock:
            session_id = self._session_id

        stuck = self._is_stuck(action)
        event = SealEvent(
            session_id=session_id,
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

    def _seal_worker_loop(self):
        """Consume seal queue and stage events in RAM (no disk I/O here)."""
        while not self._stop_event.is_set():
            try:
                event: SealEvent = self._seal_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self._stage_event(event)
            except Exception as exc:
                self.bot.log(
                    f"SequentialRecorder: RAM stage hatasi: {exc}",
                    level="WARNING",
                )
            finally:
                self._seal_queue.task_done()

    def _flush_writer_loop(self):
        """Consume flush jobs and write sequences to disk asynchronously."""
        while not self._stop_event.is_set():
            try:
                job: FlushJob = self._flush_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self._write_flush_job(job)
            except Exception as exc:
                self.bot.log(
                    f"SequentialRecorder: Flush yazma hatasi: {exc}",
                    level="WARNING",
                )
            finally:
                self._flush_queue.task_done()

    def _stage_event(self, event: SealEvent):
        """Append event to RAM circular buffer with frame-count cap."""
        if not event.frames:
            return

        with self._state_lock:
            active_session = self._session_id
            accepting = self._session_accepting_events
            recording = self._recording

        # Drop stale events (old session, fail state, or stopped state)
        if (not accepting) or (not recording) or (event.session_id != active_session):
            return

        dropped_events = 0
        dropped_frames = 0
        with self._ram_lock:
            self._ram_events.append(event)
            self._ram_frames_total += len(event.frames)

            # Keep only latest frames in RAM.
            while self._ram_frames_total > self._max_ram_frames and self._ram_events:
                old = self._ram_events.popleft()
                dropped_events += 1
                dropped_frames += len(old.frames)
                self._ram_frames_total -= len(old.frames)

        if dropped_events:
            self.bot.log(
                f"SequentialRecorder: RAM circular trim. dropped_events={dropped_events} "
                f"dropped_frames={dropped_frames} limit={self._max_ram_frames}",
                level="DEBUG",
            )

    def _write_flush_job(self, job: FlushJob):
        """Write all staged events of a flush job to disk."""
        if not job.events:
            return

        try:
            job.session_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.bot.log(f"SequentialRecorder: Klasor olusturulamadi: {exc}", level="WARNING")
            return

        wrote = 0
        for event in job.events:
            seq_num = self._next_sequence_number(job.session_id)
            if seq_num <= 0:
                continue
            if self._package_sequence(
                event=event,
                seq_num=seq_num,
                session_id=job.session_id,
                session_dir=job.session_dir,
                trigger_type=job.trigger_type,
            ):
                wrote += 1

        self.bot.log(
            f"SequentialRecorder: FLUSH tamamlandi. session={job.session_id} "
            f"reason={job.reason} yazilan={wrote}",
            level="DEBUG",
        )

    def _next_sequence_number(self, session_id: str) -> int:
        with self._state_lock:
            n = self._session_seq_counters.get(session_id, 0) + 1
            self._session_seq_counters[session_id] = n
            if session_id == self._session_id:
                self._seq_counter = n
            return n

    def _package_sequence(
        self,
        event: SealEvent,
        seq_num: int,
        session_id: str,
        session_dir: Path,
        trigger_type: str,
    ) -> bool:
        """SealEvent -> Sequence_XXXX folder + metadata.json."""
        if not event.frames:
            return False

        seq_dir = session_dir / f"Sequence_{seq_num:04d}"
        try:
            seq_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.bot.log(f"SequentialRecorder: Klasor olusturulamadi: {exc}", level="WARNING")
            return False

        frames_meta: List[Dict[str, Any]] = []
        for i, (frame, ts) in enumerate(event.frames):
            filename = f"frame_{i:02d}.jpg"
            out_path = seq_dir / filename
            try:
                # NumPy/OpenCV uyumlulugu: contiguous + uint8 formatina zorla.
                save_ready_frame = np.ascontiguousarray(frame)
                if save_ready_frame.dtype != np.uint8:
                    save_ready_frame = save_ready_frame.astype(np.uint8, copy=False)

                # Giris renk duzeni RGB ise OpenCV yazimi oncesi BGR'e cevir.
                if save_ready_frame.ndim == 3:
                    channels = int(save_ready_frame.shape[2])
                    if channels == 3 and self._frame_input_color_order == "RGB":
                        save_ready_frame = cv2.cvtColor(save_ready_frame, cv2.COLOR_RGB2BGR)
                    elif channels == 4:
                        code = (
                            cv2.COLOR_RGBA2BGR
                            if self._frame_input_color_order == "RGB"
                            else cv2.COLOR_BGRA2BGR
                        )
                        save_ready_frame = cv2.cvtColor(save_ready_frame, code)

                ok = cv2.imwrite(
                    str(out_path),
                    save_ready_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
                )
                if not ok:
                    self.bot.log(
                        f"[SeqRec] Resim kaydedilemedi (cv2 False): {out_path}",
                        level="WARNING",
                    )
            except Exception as exc:
                self.bot.log(f"[SeqRec] Yazma hatasi: {exc}", level="WARNING")

            frames_meta.append(
                {
                    "filename": filename,
                    "ts_unix": round(ts, 6),
                    "ts_iso": datetime.fromtimestamp(ts).isoformat(timespec="milliseconds"),
                }
            )

        action_final = event.action_label
        if event.extra.get("stuck") and event.action_label in _WALK_ACTIONS:
            action_final = "Error_Stuck"

        effective_trigger = event.extra.get("trigger_type", trigger_type)
        boundary_type = event.extra.get("boundary_type")  # "start" | "end" | None

        # --- Boundary frame yazimi (asenkron, SUCCESS sonrasi) ---
        # initiation → start_frame.jpg = buffer'in SON karesi (aksiyona en yakin)
        # termination → end_frame.jpg  = tek kare (termination ani)
        if boundary_type == "start" and event.frames:
            self._write_boundary_frame(
                seq_dir, "start_frame.jpg", event.frames[-1][0]
            )
        elif boundary_type == "end" and event.frames:
            self._write_boundary_frame(
                seq_dir, "end_frame.jpg", event.frames[0][0]
            )

        metadata = {
            "sequence_id": f"Sequence_{seq_num:04d}",
            "session_id": session_id,
            "trigger_type": effective_trigger,
            "phase": event.phase,
            "action_label": action_final,
            "action_source": event.action_source,
            "boundary_type": boundary_type,
            "stuck": bool(event.extra.get("stuck", False)),
            "idle": bool(event.extra.get("idle", False)),
            "ts_seal_unix": round(event.ts_seal_unix, 6),
            "ts_seal_iso": datetime.fromtimestamp(event.ts_seal_unix).isoformat(timespec="milliseconds"),
            "num_frames": len(event.frames),
            "fps": TARGET_FPS,
            "char_position": event.char_position,
            "frames": frames_meta,
        }

        # Boundary zaman damgalari - egitim setinde dogrudan erisilebilir
        if boundary_type == "start":
            metadata["Start_TS"] = event.extra.get("Start_TS", round(event.ts_seal_unix, 6))
        elif boundary_type == "end":
            metadata["End_TS"] = event.extra.get("End_TS", round(event.ts_seal_unix, 6))
            metadata["result_status"] = event.extra.get("result_status", "unknown")

        if event.extra.get("click_point"):
            metadata["click_point"] = event.extra["click_point"]

        try:
            (seq_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self.bot.log(f"SequentialRecorder: metadata.json yazma hatasi: {exc}", level="WARNING")
            return False

        boundary_tag = ""
        if boundary_type == "start":
            boundary_tag = " [BOUNDARY:START]"
        elif boundary_type == "end":
            result_tag = event.extra.get("result_status", "unknown")
            boundary_tag = f" [BOUNDARY:END result={result_tag}]"

        self.bot.log(
            f"[SeqRec] Sequence_{seq_num:04d} | phase={event.phase} | "
            f"aksiyon={action_final}({event.action_source}) | {len(event.frames)} kare"
            + (" [STUCK]" if event.extra.get("stuck") else "")
            + (" [IDLE]" if event.extra.get("idle") else "")
            + boundary_tag,
            level="DEBUG",
        )
        return True

    def _write_boundary_frame(
        self,
        seq_dir: Path,
        filename: str,
        frame: np.ndarray,
    ) -> None:
        """
        Tek bir boundary karesini (start_frame.jpg / end_frame.jpg) diske yazar.
        Calisan thread: FlushWriter (async, main thread bloklanmaz).
        """
        try:
            out_path = seq_dir / filename
            save_frame = np.ascontiguousarray(frame)
            if save_frame.dtype != np.uint8:
                save_frame = save_frame.astype(np.uint8, copy=False)
            if save_frame.ndim == 3:
                ch = int(save_frame.shape[2])
                if ch == 3 and self._frame_input_color_order == "RGB":
                    save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
                elif ch == 4:
                    code = (
                        cv2.COLOR_RGBA2BGR
                        if self._frame_input_color_order == "RGB"
                        else cv2.COLOR_BGRA2BGR
                    )
                    save_frame = cv2.cvtColor(save_frame, code)
            cv2.imwrite(
                str(out_path),
                save_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
        except Exception as exc:
            self.bot.log(f"[SeqRec] Boundary kare yazma hatasi ({filename}): {exc}", level="WARNING")

    def _clear_ram_events(self, log_drop: bool = True) -> Tuple[int, int]:
        with self._ram_lock:
            ev = len(self._ram_events)
            fr = self._ram_frames_total
            self._ram_events.clear()
            self._ram_frames_total = 0

        if log_drop and ev:
            self.bot.log(
                f"SequentialRecorder: RAM temizlendi. events={ev} frames={fr}",
                level="DEBUG",
            )
        return ev, fr

    def _purge_pending_seal_events(self, log_drop: bool = True) -> Tuple[int, int]:
        dropped_events = 0
        dropped_frames = 0
        while True:
            try:
                ev = self._seal_queue.get_nowait()
            except Empty:
                break

            dropped_events += 1
            dropped_frames += len(getattr(ev, "frames", []) or [])
            self._seal_queue.task_done()

        if log_drop and dropped_events:
            self.bot.log(
                f"SequentialRecorder: Seal queue temizlendi. events={dropped_events} "
                f"frames={dropped_frames}",
                level="DEBUG",
            )
        return dropped_events, dropped_frames

    def _drain_seal_queue_to_ram(self, max_items: int = 4096) -> int:
        """
        Seal queue'da bekleyen olaylari anlik olarak RAM stage'e tasir.
        Flush oncesi senkronizasyon icin kullanilir.
        """
        moved = 0
        for _ in range(max_items):
            try:
                ev = self._seal_queue.get_nowait()
            except Empty:
                break

            try:
                self._stage_event(ev)
            finally:
                self._seal_queue.task_done()
                moved += 1
        return moved

    def _has_vision(self) -> bool:
        return hasattr(self.bot, "vision") and self.bot.vision is not None

    def _get_current_phase(self) -> str:
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
        now = time.monotonic()
        input_idle = now - self._last_input_ts > IDLE_THRESHOLD_SEC
        if self._last_pos_change_ts <= 0:
            return input_idle
        pos_idle = now - self._last_pos_change_ts > IDLE_THRESHOLD_SEC
        return input_idle and pos_idle

    def _is_stuck(self, action_label: str) -> bool:
        if action_label not in _WALK_ACTIONS:
            return False
        if self._last_pos_change_ts <= 0:
            return False
        with self._pos_lock:
            return time.monotonic() - self._last_pos_change_ts > STUCK_THRESHOLD_SEC

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
