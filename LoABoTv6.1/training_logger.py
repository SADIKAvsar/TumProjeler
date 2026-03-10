# -*- coding: utf-8 -*-
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue


class TrainingLogger:
    """
    Agentic egitim icin aksiyon/durum/sonuc olaylarini asenkron JSONL olarak kaydeder.
    """

    def __init__(self, bot):
        self.bot = bot
        self.enabled = bool(self.bot.settings.get("TRAINING_LOGGER_ENABLED", True))
        self.root = Path(self.bot.settings.get("TRAINING_LOGGER_ROOT", r"D:\LoABot_Training_Data\AGENTIC_LOGS"))
        self.queue_maxsize = int(self.bot.settings.get("TRAINING_LOGGER_QUEUE_MAXSIZE", 50000))

        self._queue = Queue(maxsize=max(1024, self.queue_maxsize))
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._episode = None
        self._worker = None
        self._dropped = 0

        if not self.enabled:
            return

        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.enabled = False
            self.bot.log(f"TrainingLogger devre disi (klasor hatasi): {exc}", level="WARNING")
            return

        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="TrainingLoggerWorker")
        self._worker.start()
        self.bot.log(f"TrainingLogger aktif. Kayit yolu: {self.root}", level="DEBUG")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue

            path = item.pop("_path", None)
            if path:
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as exc:
                    self.bot.log(f"TrainingLogger yazma hatasi: {exc}", level="WARNING")
            self._queue.task_done()

    def _enqueue(self, event_type: str, payload: dict):
        if not self.enabled:
            return

        with self._lock:
            ep = dict(self._episode or {})
        if not ep:
            return

        row = {
            "ts_unix": time.time(),
            "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
            "episode_id": ep.get("id"),
            "event_type": str(event_type),
            "payload": dict(payload or {}),
            "_path": ep.get("path"),
        }
        try:
            self._queue.put_nowait(row)
        except Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                self.bot.log(f"TrainingLogger queue dolu, event atlandi (drop={self._dropped})", level="WARNING")

    def start_episode(self, episode_type: str = "mission", context: dict = None):
        if not self.enabled:
            return None

        with self._lock:
            if self._episode:
                return self._episode.get("id")

            now = datetime.now()
            episode_id = now.strftime("%Y%m%d_%H%M%S_%f")
            day_dir = self.root / now.strftime("%Y%m%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            path = day_dir / f"{episode_id}.jsonl"
            self._episode = {
                "id": episode_id,
                "path": str(path),
                "type": str(episode_type or "mission"),
                "start_ts": time.time(),
            }

        self._enqueue(
            "episode_start",
            {
                "episode_type": str(episode_type or "mission"),
                "context": dict(context or {}),
            },
        )
        return episode_id

    def end_episode(self, status: str = "completed", reason: str = "", metrics: dict = None):
        if not self.enabled:
            return

        with self._lock:
            ep = dict(self._episode or {})
            self._episode = None
        if not ep:
            return

        row = {
            "ts_unix": time.time(),
            "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
            "episode_id": ep.get("id"),
            "event_type": "episode_end",
            "payload": {
                "status": str(status or "completed"),
                "reason": str(reason or ""),
                "duration_sn": max(0.0, time.time() - float(ep.get("start_ts", time.time()))),
                "metrics": dict(metrics or {}),
            },
            "_path": ep.get("path"),
        }
        try:
            self._queue.put_nowait(row)
        except Full:
            pass

    def log_action(self, action_name: str, payload: dict = None):
        self._enqueue("action", {"name": str(action_name or "unknown_action"), **dict(payload or {})})

    def log_state(self, state_name: str, payload: dict = None):
        self._enqueue("state", {"name": str(state_name or "unknown_state"), **dict(payload or {})})

    def log_outcome(self, outcome_name: str, payload: dict = None):
        self._enqueue("outcome", {"name": str(outcome_name or "unknown_outcome"), **dict(payload or {})})