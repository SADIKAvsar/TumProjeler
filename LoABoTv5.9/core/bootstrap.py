"""
core/bootstrap.py — v6.0 Altyapı Başlatma
============================================
loabot_main.py'nin __init__() sonunda çağrılır.
GameState + EventBus + BotBridge kurar ve interceptor'ları yerleştirir.

Kullanım (loabot_main.py LoABot.__init__'e eklenecek 3 satır):

    # ── v6.0 altyapı ───────────────────────────────────
    from core.bootstrap import bootstrap_v6
    self.game_state, self.event_bus, self.bridge = bootstrap_v6(self)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from compat.bot_bridge import BotBridge
    from core.event_bus import EventBus
    from core.game_state import GameState


def bootstrap_v6(bot) -> Tuple["GameState", "EventBus", "BotBridge"]:
    """
    v6.0 çekirdek altyapısını kurar ve bot'a bağlar.

    Sıra:
      1. EventBus oluştur (log_fn = bot.log)
      2. GameState oluştur (event_bus bağlı)
      3. BotBridge oluştur (bot ↔ state senkronizasyonu)
      4. Interceptor'ları kur (eski API → EventBus otomatik yayın)
      5. Debug logger'ı bağla (wildcard subscriber)
      6. Sync thread'i başlat

    Returns:
        (game_state, event_bus, bridge)
    """
    from compat.bot_bridge import BotBridge
    from core.event_bus import EventBus
    from core.event_types import EVT
    from core.game_state import GameState

    log_fn = getattr(bot, "log", print)

    # 1. EventBus
    event_bus = EventBus(log_fn=log_fn, debug=False)

    # 2. GameState (EventBus bağlı)
    game_state = GameState(event_bus=event_bus)

    # 3. BotBridge
    bridge = BotBridge(bot, game_state, event_bus)

    # 4. Interceptor'lar — eski API çağrıları otomatik event yayınlar
    bridge.install_interceptors()

    # 5. Debug event logger (opsiyonel — sadece kritik event'ler)
    _critical_events = {
        EVT.PHASE_CHANGED,
        EVT.COMBAT_STARTED,
        EVT.COMBAT_FINISHED,
        EVT.BOSS_KILLED,
        EVT.STUCK_DETECTED,
        EVT.MISSION_STARTED,
        EVT.MISSION_ENDED,
    }

    def _event_logger(event):
        if event.event_type in _critical_events:
            log_fn(
                f"[EventBus] {event.event_type} "
                f"from={event.source} "
                f"payload={dict(list(event.payload.items())[:4])}",
            )

    event_bus.subscribe("*", _event_logger, name="debug_logger", priority=999)

    # 6. Periyodik sync thread'i (eski modüllerin attribute değişikliklerini yakalar)
    bridge.start_sync_thread(interval=0.5)

    log_fn("[v6.0] GameState + EventBus + BotBridge aktif.")

    return game_state, event_bus, bridge
