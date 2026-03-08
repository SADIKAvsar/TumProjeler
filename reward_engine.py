"""
reward_engine.py — Ödül Hesaplama Motoru
==========================================
Agentic YZ ajanının öğrenmesi için her sekans / eylem adına
sayısal ödül (reward) üretir.

Ödül Sinyalleri:
  +1.0  boss öldürme başarısı
  +0.5  etkinliğe giriş başarısı
  -1.0  ölüm / fail
  -0.5  oyun dondu / restart gerekti
  +0.1  her başarılı sekans adımı
  -0.1  sekans adımı başarısız
  ±0.0–0.3  kill time bonus (hızlı kill → +, yavaş → -)
  -0.3  Error_Stuck tespiti

Reward, SequentialRecorder metadata.json'a "reward" alanı olarak
eklenir ve TrainingLogger episode_end payload'una yazılır.
Böylece train_agentic.py sırasında hem sequence hem de episode
seviyesinde reward kullanılabilir.

Kullanım:
    from reward_engine import RewardEngine
    re = RewardEngine(bot)
    re.on_boss_killed("boss_800", kill_time=38.5)
    re.on_stuck_detected()
    # Otomatik olarak son aktif sequence'a reward yazar.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RewardRecord:
    """Tek bir ödül kaydı."""
    ts: float
    value: float
    reason: str
    phase: str = "UNKNOWN"
    extra: Dict = field(default_factory=dict)


# ── Ödül Sabitleri ────────────────────────────────────────────────────────

class R:
    BOSS_KILL        = +1.0
    EVENT_ENTRY      = +0.5
    DEATH            = -1.0
    RESTART          = -0.5
    SEQ_STEP_OK      = +0.1
    SEQ_STEP_FAIL    = -0.1
    STUCK            = -0.3
    FAST_KILL_BONUS  = +0.3   # kill_time < best_time * 1.1
    SLOW_KILL_PENALT = -0.2   # kill_time > avg_kill_time * 1.5


class RewardEngine:
    """
    Bot olaylarından ödül üretir ve aktif sekansa / episode'a yazar.
    """

    def __init__(self, bot):
        self.bot = bot
        self._lock = threading.Lock()
        self._episode_rewards: List[RewardRecord] = []
        self._session_total: float = 0.0

    # ── Dışa Açık Olay API'leri ──────────────────────────────────────────

    def on_boss_killed(self, boss_name: str, kill_time: float):
        """
        Boss başarıyla öldürüldüğünde çağrılır.
        CombatManager.boss_loop() içinden tetiklenmeli.
        """
        value = R.BOSS_KILL

        # Kill time bonus/ceza: MemoryManager'daki istatistikleri kullan
        mem = self._get_memory()
        if mem:
            stats = mem.get_boss_stats(boss_name)
            avg = stats.get("avg_kill_time", 0.0)
            best = stats.get("best_time", float("inf"))
            if avg > 0 and kill_time < avg * 0.85:
                value += R.FAST_KILL_BONUS
            elif avg > 0 and kill_time > avg * 1.5:
                value += R.SLOW_KILL_PENALT

        record = RewardRecord(
            ts=time.time(),
            value=round(value, 3),
            reason="boss_killed",
            phase=self._current_phase(),
            extra={"boss_name": boss_name, "kill_time": round(kill_time, 2)},
        )
        self._record(record)
        self._write_to_last_sequence(record)
        self.bot.log(
            f"[Reward] boss_killed={boss_name}, t={kill_time:.1f}s → {value:+.2f}",
            level="DEBUG",
        )

    def on_death(self, boss_name: str = ""):
        record = RewardRecord(
            ts=time.time(),
            value=R.DEATH,
            reason="death",
            phase=self._current_phase(),
            extra={"boss_name": boss_name},
        )
        self._record(record)
        self._write_to_last_sequence(record)
        self.bot.log(f"[Reward] death → {R.DEATH:+.2f}", level="DEBUG")

    def on_restart(self, reason: str = ""):
        record = RewardRecord(
            ts=time.time(),
            value=R.RESTART,
            reason="restart",
            phase=self._current_phase(),
            extra={"reason": reason},
        )
        self._record(record)
        self._write_to_last_sequence(record)
        self.bot.log(f"[Reward] restart → {R.RESTART:+.2f}", level="DEBUG")

    def on_event_entry(self, event_name: str, success: bool):
        value = R.EVENT_ENTRY if success else R.SEQ_STEP_FAIL
        record = RewardRecord(
            ts=time.time(),
            value=value,
            reason="event_entry",
            phase="EVENT_PHASE",
            extra={"event_name": event_name, "success": success},
        )
        self._record(record)
        self._write_to_last_sequence(record)

    def on_stuck_detected(self):
        record = RewardRecord(
            ts=time.time(),
            value=R.STUCK,
            reason="stuck",
            phase=self._current_phase(),
        )
        self._record(record)
        self._write_to_last_sequence(record)
        self.bot.log(f"[Reward] stuck → {R.STUCK:+.2f}", level="DEBUG")

    def on_sequence_step(self, success: bool, action: str = ""):
        value = R.SEQ_STEP_OK if success else R.SEQ_STEP_FAIL
        record = RewardRecord(
            ts=time.time(),
            value=value,
            reason=f"seq_step_{'ok' if success else 'fail'}",
            phase=self._current_phase(),
            extra={"action": action},
        )
        self._record(record)

    # ── Episode Özeti ────────────────────────────────────────────────────

    def get_episode_summary(self) -> Dict:
        """Episode bittiğinde TrainingLogger'a gönderilecek özet."""
        with self._lock:
            total = sum(r.value for r in self._episode_rewards)
            by_reason: Dict[str, float] = {}
            for r in self._episode_rewards:
                by_reason[r.reason] = round(by_reason.get(r.reason, 0.0) + r.value, 3)
            count = len(self._episode_rewards)
            rewards = [{"ts": r.ts, "value": r.value, "reason": r.reason} for r in self._episode_rewards]
        return {
            "total_reward": round(total, 3),
            "event_count": count,
            "by_reason": by_reason,
            "rewards": rewards,
        }

    def reset_episode(self):
        """Yeni episode başlarken sıfırla."""
        with self._lock:
            self._episode_rewards.clear()

    def get_session_total(self) -> float:
        return round(self._session_total, 3)

    # ── Dahili ───────────────────────────────────────────────────────────

    def _record(self, record: RewardRecord):
        with self._lock:
            self._episode_rewards.append(record)
            self._session_total += record.value

    def _write_to_last_sequence(self, record: RewardRecord):
        """
        SequentialRecorder'ın en son yazılan Sequence_XXXX/metadata.json
        dosyasına "reward" alanı ekler.

        Not: seq_recorder._session_dir + seq_counter ile son klasörü bulur.
        Bu hafif bir yaklaşımdır — daha güçlü bağlantı için seq_recorder'a
        bir callback mekanizması eklenebilir.
        """
        try:
            seq = getattr(self.bot, "seq_recorder", None)
            if seq is None:
                return
            with seq._state_lock:
                session_dir = seq._session_dir
                seq_count = seq._seq_counter
            if session_dir is None or seq_count == 0:
                return

            meta_path = session_dir / f"Sequence_{seq_count:04d}" / "metadata.json"
            if not meta_path.exists():
                # Son yazılan sekans henüz tamamlanmamış olabilir — bir öncekine bak
                if seq_count > 1:
                    meta_path = session_dir / f"Sequence_{seq_count - 1:04d}" / "metadata.json"
                if not meta_path.exists():
                    return

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Reward alanı ekle/güncelle
            if "reward" not in meta:
                meta["reward"] = {"value": 0.0, "events": []}
            meta["reward"]["value"] = round(meta["reward"]["value"] + record.value, 3)
            meta["reward"]["events"].append({
                "ts": round(record.ts, 3),
                "value": record.value,
                "reason": record.reason,
            })

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            self.bot.log(f"[Reward] metadata.json reward yazma hatasi: {exc}", level="DEBUG")

    def _current_phase(self) -> str:
        return str(getattr(self.bot, "_global_phase", "UNKNOWN"))

    def _get_memory(self):
        """MemoryManager'a erişim — brain.ai_engine.memory üzerinden."""
        try:
            return self.bot.brain.ai_engine.memory
        except Exception:
            return None
