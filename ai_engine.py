"""
Local-first decision engine.

Cloud/API calls are intentionally removed. All decisions are made with
deterministic local rules and local memory statistics.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from memory_manager import MemoryManager
from utils import project_path


class DecisionHistoryLogger:
    """JSONL audit trail for local decisions."""

    def __init__(self, log_path: str = None):
        default_path = project_path(r"E:\LoABot_Training_Data\runtime_data\decision_history.jsonl")
        self.log_path = Path(log_path or default_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, decision_type: str, decision: Dict, context: Dict, source: str = "local_rules"):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "type": str(decision_type),
                "source": str(source),
                "decision": dict(decision or {}),
                "context": dict(context or {}),
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_stats(self) -> Dict:
        if not self.log_path.exists():
            return {"total_logged": 0, "local_decisions": 0}

        total = 0
        local = 0
        by_type = {}
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = line.strip()
                    if not row:
                        continue
                    item = json.loads(row)
                    total += 1
                    if item.get("source") == "local_rules":
                        local += 1
                    t = str(item.get("type", "unknown"))
                    by_type[t] = by_type.get(t, 0) + 1
        except Exception:
            pass

        return {
            "total_logged": total,
            "local_decisions": local,
            "local_ratio": round((local / max(1, total)) * 100.0, 1),
            "by_type": by_type,
        }


class AIEngine:
    """
    Local decision engine that preserves the same interface expected by the bot.
    """

    def __init__(self, bot, config_path: str = None):
        del config_path  # Backward-compatible signature.
        self.bot = bot

        s = getattr(self.bot, "settings", {}) or {}
        self.enabled = bool(s.get("LOCAL_AI_ENABLED", True))
        self.observer_mode = bool(s.get("LOCAL_AI_OBSERVER_MODE", False))
        self.fallback_to_rules = True

        memory_path = s.get("LOCAL_AI_MEMORY_PATH") or r"E:\LoABot_Training_Data\runtime_data\ai_memory.json"
        history_path = s.get("LOCAL_AI_HISTORY_PATH") or r"E:\LoABot_Training_Data\runtime_data\decision_history.jsonl"
        self.memory = MemoryManager(memory_path)
        self.history = DecisionHistoryLogger(history_path)

        self.decisions_made = 0
        self.ai_calls_made = 0
        self.fallback_count = 0
        self.cache_hits = 0

        self.bot.log(f"Local AI Engine initialized (enabled={self.enabled}, observer={self.observer_mode})")

    def _region_name(self) -> str:
        if hasattr(self.bot, "location_manager"):
            return str(self.bot.location_manager.get_region_name() or "UNKNOWN")
        if hasattr(self.bot, "location"):
            return str(self.bot.location.get_region_name() or "UNKNOWN")
        return "UNKNOWN"

    def _boss_ready_delta(self, boss: Dict, now: float) -> float:
        spawn_ts = boss.get("spawn_time")
        if not isinstance(spawn_ts, (int, float)):
            return 10**9
        head_start = float(boss.get("head_start_saniye", 0.0))
        return max(0.0, (float(spawn_ts) - head_start) - now)

    def _is_same_map(self, current_region: str, boss: Dict) -> bool:
        current = str(current_region or "").upper()
        katman = str(boss.get("katman_id", "")).lower()
        if current == "KATMAN_1":
            return "katman_1" in katman
        if current == "KATMAN_2":
            return "katman_2" in katman
        return False

    def decide_next_boss(self, ready_bosses: List[Dict]) -> Optional[str]:
        if not self.enabled or not ready_bosses:
            return None

        self.ai_calls_made += 1
        self.decisions_made += 1
        now = time.time()
        current_region = self._region_name()

        def sort_key(boss: Dict):
            same_map_priority = 0 if self._is_same_map(current_region, boss) else 1
            ready_delta = self._boss_ready_delta(boss, now)
            success_rate = self.memory.get_boss_success_rate(str(boss.get("aciklama", "")))
            return (same_map_priority, ready_delta, -success_rate, str(boss.get("aciklama", "")))

        chosen = sorted(ready_bosses, key=sort_key)[0]
        boss_name = str(chosen.get("aciklama", ""))
        confidence = 0.90 if self._is_same_map(current_region, chosen) else 0.75
        reason = (
            "same_map_priority_then_earliest_spawn"
            if self._is_same_map(current_region, chosen)
            else "earliest_spawn_with_success_rate_tiebreak"
        )

        context = {
            "current_location": current_region,
            "ready_bosses": [str(b.get("aciklama", "")) for b in ready_bosses],
            "ready_count": len(ready_bosses),
        }
        decision = {
            "decision": boss_name,
            "confidence": confidence,
            "reasoning": reason,
        }
        self.history.log_decision("boss_selection", decision, context, source="local_rules")
        self.memory.record_decision("boss_selection")
        self.bot.log(f"[LOCAL_AI] Boss decision: {boss_name} (confidence: {confidence:.2f}, reason: {reason})")
        return boss_name

    def diagnose_freeze(
        self,
        duration: float,
        last_action: str = "unknown",
        expected_result: str = "unknown",
    ) -> Optional[Dict]:
        if not self.enabled:
            return None

        self.ai_calls_made += 1
        self.decisions_made += 1

        freeze_limit = float(self.bot.settings.get("MAX_FREEZE_DURATION_SN", 60.0))
        interval = float(self.bot.settings.get("FREEZE_CHECK_INTERVAL_SN", 15.0))
        duration = float(max(0.0, duration))

        if duration < freeze_limit * 1.20:
            decision = {
                "is_frozen": True,
                "freeze_type": "likely_temporary_stall",
                "recommended_action": "wait",
                "wait_time_seconds": max(5.0, min(interval, 20.0)),
                "confidence": 0.70,
                "visual_evidence": "low_motion_detected",
            }
        elif duration < freeze_limit * 1.80:
            decision = {
                "is_frozen": True,
                "freeze_type": "possible_popup_or_path_stuck",
                "recommended_action": "close_popup",
                "wait_time_seconds": 0.0,
                "confidence": 0.78,
                "visual_evidence": "persistent_low_motion",
            }
        else:
            decision = {
                "is_frozen": True,
                "freeze_type": "hard_stall",
                "recommended_action": "restart",
                "wait_time_seconds": 0.0,
                "confidence": 0.92,
                "visual_evidence": "long_static_duration",
            }

        context = {
            "duration": duration,
            "last_action": str(last_action),
            "expected_result": str(expected_result),
            "location": self._region_name(),
        }
        self.history.log_decision("freeze_diagnosis", decision, context, source="local_rules")
        self.memory.record_freeze_diagnosis(diagnosis=decision, duration=duration, was_correct=None)
        return decision

    def evaluate_strategic_wait(
        self,
        current_boss: Dict,
        next_boss: Dict,
        time_until_next: float,
    ) -> Optional[Dict]:
        if not self.enabled or not current_boss or not next_boss:
            return None

        self.ai_calls_made += 1
        self.decisions_made += 1

        same_map = str(current_boss.get("katman_id", "")) == str(next_boss.get("katman_id", ""))
        switch_threshold = float(self.bot.settings.get("BOSS_SWITCH_THRESHOLD_SN", 91.0))
        fast_chain_threshold = float(self.bot.settings.get("FAST_CHAIN_THRESHOLD_SN", 90.0))
        ttn = float(max(0.0, time_until_next))

        walk_time = 0.0
        walk_time_enabled = bool(self.bot.settings.get("WALK_TIME_ENABLED", False))
        if walk_time_enabled and hasattr(self.bot, "combat"):
            try:
                walk_time = float(
                    self.bot.combat.get_walk_time(
                        str(current_boss.get("aciklama", "")),
                        str(next_boss.get("aciklama", "")),
                    )
                )
            except Exception:
                walk_time = float(self.bot.settings.get("WALK_TIME_DEFAULT_SN", 40))

        if same_map and ttn <= max(switch_threshold, fast_chain_threshold):
            action = "wait"
            confidence = 0.90
            reason = "same_map_and_short_timer"
        elif walk_time_enabled and walk_time < ttn and ttn <= switch_threshold:
            action = "wait"
            confidence = 0.75
            reason = "walk_time_advantage"
        else:
            action = "return_to_farm"
            confidence = 0.90
            reason = "timer_too_long_or_map_switch"

        decision = {
            "decision": action,
            "reasoning": reason,
            "confidence": confidence,
            "time_impact": round(max(0.0, switch_threshold - ttn), 2) if action == "wait" else 0.0,
            "risk_assessment": "low" if confidence >= 0.85 else "medium",
        }
        context = {
            "current_boss": str(current_boss.get("aciklama", "")),
            "next_boss": str(next_boss.get("aciklama", "")),
            "time_until_next": ttn,
            "same_map": same_map,
            "walk_time": walk_time,
        }
        self.history.log_decision("strategic_wait", decision, context, source="local_rules")
        self.memory.record_strategic_decision(
            decision_type="wait_same_map",
            outcome=(action == "wait"),
            time_saved=float(decision.get("time_impact", 0.0)),
            context=context,
        )
        return decision

    def get_statistics(self) -> Dict:
        history_stats = self.history.get_stats()
        memory_stats = self.memory.get_summary_stats()
        return {
            "enabled": self.enabled,
            "observer_mode": self.observer_mode,
            "decisions_made": self.decisions_made,
            "ai_calls_made": self.ai_calls_made,
            "fallback_count": self.fallback_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": 0.0,
            "daily_cost_usd": 0.0,
            "daily_limit_usd": 0.0,
            **memory_stats,
            "history": history_stats,
        }

    def toggle_ai(self, enabled: bool = None):
        if enabled is None:
            self.enabled = not self.enabled
        else:
            self.enabled = bool(enabled)
        self.bot.log(f"[LOCAL_AI] {'aktif' if self.enabled else 'devre disi'}")
