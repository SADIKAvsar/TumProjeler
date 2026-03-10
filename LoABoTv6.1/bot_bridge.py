"""
bot_bridge.py — Backward Compatibility Bridge + Bootstrap
============================================================
v6.1: bootstrap.py bu dosyaya birlestirildi.
      _ATTR_MAP sadeleştirildi, sync interval artırıldı.

Strangler Fig Pattern:
    - Yeni modüller GameState + EventBus kullanır
    - Eski modüller self.bot.* kullanmaya devam eder
    - BotBridge ikisini senkronize tutar

Kullanım:
    from bot_bridge import bootstrap_v6

    self.game_state, self.event_bus, self.bridge = bootstrap_v6(self)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from event_bus import EventBus
    from game_state import GameState


class BotBridge:
    """
    self.bot.* ↔ GameState/EventBus iki yönlü köprü.
    """

    # v6.1: Sadeleştirildi — reason/extra/phase_ts interceptor üzerinden
    # zaten GameState'e yazılıyor, tekrar sync'e gerek yok.
    _ATTR_MAP: Dict[str, str] = {
        "_global_phase":             "phase",
        "_global_stage":             "stage",
        "attacking_target_aciklama": "attacking_target",
        "active_event":              "active_event",
        "paused":                    "paused",
    }

    def __init__(
        self,
        bot,
        game_state: "GameState",
        event_bus: "EventBus",
    ):
        self.bot = bot
        self.state = game_state
        self.bus = event_bus
        self._sync_lock = threading.Lock()

        # İlk senkronizasyon: bot → state
        self.sync_from_bot()

        running = getattr(bot, "running", None)
        if running and hasattr(running, "is_set"):
            self.state.set("running", running.is_set(), source="bridge_init")

    # ══════════════════════════════════════════════════════════════
    #  BOT → STATE
    # ══════════════════════════════════════════════════════════════

    def sync_from_bot(self) -> None:
        updates = {}
        for bot_attr, state_key in self._ATTR_MAP.items():
            val = getattr(self.bot, bot_attr, None)
            if val is not None:
                updates[state_key] = val

        running = getattr(self.bot, "running", None)
        if running and hasattr(running, "is_set"):
            updates["running"] = running.is_set()

        if updates:
            self.state.update(updates, source="bridge_from_bot")

    # ══════════════════════════════════════════════════════════════
    #  STATE → BOT
    # ══════════════════════════════════════════════════════════════

    def sync_to_bot(self) -> None:
        with self._sync_lock:
            for bot_attr, state_key in self._ATTR_MAP.items():
                val = self.state.get(state_key)
                if val is not None and hasattr(self.bot, bot_attr):
                    setattr(self.bot, bot_attr, val)

    # ══════════════════════════════════════════════════════════════
    #  WRAPPED METHODS
    # ══════════════════════════════════════════════════════════════

    def set_global_mission_phase(
        self,
        phase: str,
        stage: str,
        reason: str = "",
        extra: Optional[Dict] = None,
    ) -> None:
        if hasattr(self.bot, "_original_set_global_mission_phase"):
            self.bot._original_set_global_mission_phase(phase, stage, reason, extra)
        else:
            self.bot._global_phase = phase
            self.bot._global_stage = stage
            self.bot._global_mission_reason = reason
            self.bot._global_mission_extra = dict(extra or {})

        self.state.set_phase(
            phase=phase,
            stage=stage,
            reason=reason,
            extra=extra,
            source="bridge",
        )

    def start_global_mission(
        self,
        phase: str,
        stage: str,
        reason: str = "",
        extra: Optional[Dict] = None,
    ) -> None:
        from event_bus import EVT

        if hasattr(self.bot, "_original_start_global_mission"):
            self.bot._original_start_global_mission(phase, stage, reason, extra)
        else:
            self.set_global_mission_phase(phase, stage, reason, extra)

        self.bus.publish(
            EVT.MISSION_STARTED,
            {"phase": phase, "stage": stage, "reason": reason, "extra": extra or {}},
            source="bridge",
        )

    def stop_global_mission(self, reason: str = "") -> None:
        from event_bus import EVT

        if hasattr(self.bot, "_original_stop_global_mission"):
            self.bot._original_stop_global_mission(reason)
        else:
            self.set_global_mission_phase("IDLE_PHASE", "waiting", reason)

        self.bus.publish(
            EVT.MISSION_ENDED,
            {"reason": reason},
            source="bridge",
        )

    # ══════════════════════════════════════════════════════════════
    #  MONKEY-PATCH
    # ══════════════════════════════════════════════════════════════

    def install_interceptors(self) -> None:
        bot = self.bot

        if hasattr(bot, "set_global_mission_phase"):
            bot._original_set_global_mission_phase = bot.set_global_mission_phase
        if hasattr(bot, "start_global_mission"):
            bot._original_start_global_mission = bot.start_global_mission
        if hasattr(bot, "stop_global_mission"):
            bot._original_stop_global_mission = bot.stop_global_mission

        bot.set_global_mission_phase = self.set_global_mission_phase
        bot.start_global_mission = self.start_global_mission
        bot.stop_global_mission = self.stop_global_mission

        if hasattr(bot, "log"):
            bot.log("[BotBridge] Interceptor'lar kuruldu. Eski API -> EventBus koprusu aktif.")

    def uninstall_interceptors(self) -> None:
        bot = self.bot
        for method in ("set_global_mission_phase", "start_global_mission", "stop_global_mission"):
            original = f"_original_{method}"
            if hasattr(bot, original):
                setattr(bot, method, getattr(bot, original))
                delattr(bot, original)

    # ══════════════════════════════════════════════════════════════
    #  PERİYODİK SYNC
    # ══════════════════════════════════════════════════════════════

    def periodic_sync(self, interval: float = 2.0) -> None:
        while True:
            try:
                self.sync_from_bot()
            except Exception:
                pass
            time.sleep(interval)

    def start_sync_thread(self, interval: float = 2.0) -> None:
        t = threading.Thread(
            target=self.periodic_sync,
            args=(interval,),
            daemon=True,
            name="BotBridge-Sync",
        )
        t.start()


# ══════════════════════════════════════════════════════════════════════
#  BOOTSTRAP  (eski bootstrap.py)
# ══════════════════════════════════════════════════════════════════════

def bootstrap_v6(bot) -> Tuple["GameState", "EventBus", "BotBridge"]:
    """
    v6.0 çekirdek altyapısını kurar ve bot'a bağlar.

    Returns:
        (game_state, event_bus, bridge)
    """
    from event_bus import EventBus, EVT
    from game_state import GameState

    log_fn = getattr(bot, "log", print)

    # 1. EventBus
    event_bus = EventBus(log_fn=log_fn, debug=False)

    # 2. GameState (EventBus bağlı)
    game_state = GameState(event_bus=event_bus)

    # 3. BotBridge
    bridge = BotBridge(bot, game_state, event_bus)

    # 4. Interceptor'lar
    bridge.install_interceptors()

    # 5. Debug event logger
    _critical = {
        EVT.PHASE_CHANGED,
        EVT.COMBAT_STARTED,
        EVT.COMBAT_FINISHED,
        EVT.BOSS_KILLED,
        EVT.STUCK_DETECTED,
        EVT.MISSION_STARTED,
        EVT.MISSION_ENDED,
    }

    def _event_logger(event):
        if event.event_type in _critical:
            log_fn(
                f"[EventBus] {event.event_type} "
                f"from={event.source} "
                f"payload={dict(list(event.payload.items())[:4])}"
            )

    event_bus.subscribe("*", _event_logger, name="debug_logger", priority=999)

    # 6. Sync thread — v6.1: interval 2.0s (interceptor'lar anında sync yapar)
    bridge.start_sync_thread(interval=2.0)

    log_fn("[v6.1] GameState + EventBus + BotBridge aktif.")

    return game_state, event_bus, bridge
