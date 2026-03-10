"""
game_state.py — Thread-Safe Centralized Game State
=====================================================
v6.1: Sadeleştirildi — kullanılmayan dunder metotlar kaldırıldı,
lazy import yerine doğrudan event_bus import kullanılıyor.

Kullanım:
    from game_state import GameState

    state = GameState()
    state.update({"phase": "COMBAT_PHASE", "target": "Dragon"})
    phase = state.get("phase")
    snap = state.snapshot()
"""

from __future__ import annotations

import copy
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

if TYPE_CHECKING:
    from event_bus import EventBus


# ══════════════════════════════════════════════════════════════════════
#  VARSAYILAN DURUM ŞABLONU
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_STATE: Dict[str, Any] = {
    # ── Bot Durumu ────────────────────────────────────────────────
    "running": False,
    "paused": False,

    # ── Phase Machine ─────────────────────────────────────────────
    "phase": "IDLE_PHASE",
    "stage": "waiting",
    "phase_reason": "",
    "phase_extra": {},
    "phase_ts": 0.0,

    # ── Saldırı Durumu ────────────────────────────────────────────
    "attacking_target": None,
    "active_event": None,
    "in_combat": False,

    # ── Konum ─────────────────────────────────────────────────────
    "current_region": "UNKNOWN",
    "current_location": "UNKNOWN",

    # ── Vision ────────────────────────────────────────────────────
    "last_vision_ts": 0.0,
    "detected_objects": [],
    "enemy_visible": False,

    # ── Boss ──────────────────────────────────────────────────────
    "next_boss_name": None,
    "next_boss_spawn_ts": 0.0,
    "last_boss_killed": None,
    "last_boss_kill_ts": 0.0,

    # ── Kayıt ─────────────────────────────────────────────────────
    "recording_active": False,

    # ── Performans ────────────────────────────────────────────────
    "inference_ms": 0.0,
    "fps": 0.0,
}


# ══════════════════════════════════════════════════════════════════════
#  GAME STATE
# ══════════════════════════════════════════════════════════════════════

class GameState:
    """
    Thread-safe merkezi oyun durumu.
    Tüm modüller bu nesne üzerinden durum okur/yazar.
    """

    def __init__(self, event_bus: Optional["EventBus"] = None):
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = dict(_DEFAULT_STATE)
        self._event_bus = event_bus
        self._update_count: int = 0
        self._dirty_keys: Set[str] = set()

    # ── BAĞLANTI ──────────────────────────────────────────────────

    def set_event_bus(self, bus: "EventBus") -> None:
        self._event_bus = bus

    # ── OKUMA ─────────────────────────────────────────────────────

    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        with self._lock:
            if key is None:
                return dict(self._data)
            return self._data.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Thread-safe derin kopya — uzun süreli analizler için."""
        with self._lock:
            return copy.deepcopy(self._data)

    # ── YAZMA ─────────────────────────────────────────────────────

    def update(self, data: Dict[str, Any], source: str = "") -> None:
        if not data:
            return

        changed: Dict[str, Any] = {}

        with self._lock:
            for key, value in data.items():
                old = self._data.get(key)
                if old != value:
                    changed[key] = value
                self._data[key] = value

            self._update_count += 1
            self._dirty_keys = set(changed.keys())

        if changed and self._event_bus:
            self._publish_changes(changed, source)

    def set(self, key: str, value: Any, source: str = "") -> None:
        self.update({key: value}, source=source)

    # ── PHASE YÖNETIMI ────────────────────────────────────────────

    def set_phase(
        self,
        phase: str,
        stage: str = "",
        reason: str = "",
        extra: Optional[Dict] = None,
        source: str = "",
    ) -> None:
        self.update(
            {
                "phase": phase,
                "stage": stage,
                "phase_reason": reason,
                "phase_extra": dict(extra or {}),
                "phase_ts": time.time(),
            },
            source=source,
        )

    # ── SORGULAMA KISA YOLLARI ────────────────────────────────────

    @property
    def phase(self) -> str:
        return str(self.get("phase", "IDLE_PHASE"))

    @property
    def is_idle(self) -> bool:
        return self.phase in {"IDLE_PHASE", "IDLE", ""}

    @property
    def is_in_combat(self) -> bool:
        phase = self.phase.upper()
        attacking = bool(self.get("attacking_target"))
        return attacking or phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}

    @property
    def is_running(self) -> bool:
        return bool(self.get("running", False))

    @property
    def is_paused(self) -> bool:
        return bool(self.get("paused", False))

    # ── AUTO-PUBLISH ──────────────────────────────────────────────

    def _publish_changes(self, changed: Dict[str, Any], source: str) -> None:
        from event_bus import EVT  # v6.1: artık event_bus'tan import

        bus = self._event_bus
        if not bus:
            return

        if "phase" in changed:
            bus.publish(
                EVT.PHASE_CHANGED,
                {
                    "new_phase": changed["phase"],
                    "stage": self.get("stage", ""),
                    "reason": self.get("phase_reason", ""),
                },
                source=source or "game_state",
            )

        if "attacking_target" in changed:
            target = changed["attacking_target"]
            if target:
                bus.publish(
                    EVT.COMBAT_STARTED,
                    {"target": target},
                    source=source or "game_state",
                )
            else:
                bus.publish(
                    EVT.COMBAT_FINISHED,
                    {"reason": self.get("phase_reason", "")},
                    source=source or "game_state",
                )

        if "current_region" in changed:
            bus.publish(
                EVT.LOCATION_CHANGED,
                {"region": changed["current_region"]},
                source=source or "game_state",
            )

    # ── İSTATİSTİK ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "update_count": self._update_count,
                "key_count": len(self._data),
                "dirty_keys": list(self._dirty_keys),
                "phase": self._data.get("phase", "?"),
                "attacking": self._data.get("attacking_target"),
            }

    # ── RESET ─────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._data = dict(_DEFAULT_STATE)
            self._update_count = 0
            self._dirty_keys.clear()

    def __repr__(self) -> str:
        return f"<GameState phase={self.phase} combat={self.is_in_combat}>"
