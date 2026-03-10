"""
compat/bot_bridge.py — Backward Compatibility Bridge
======================================================
v5.9 self.bot.* API'si ile v6.0 GameState/EventBus arasında
iki yönlü senkronizasyon sağlar.

Strangler Fig Pattern:
    - Yeni modüller GameState + EventBus kullanır
    - Eski modüller self.bot.* kullanmaya devam eder
    - BotBridge ikisini senkronize tutar
    - Zamanla eski modüller migrate edilir, bridge küçülür

Senkronizasyon Yönleri:
    bot → state : sync_from_bot()   — eski modül state değiştirdiğinde
    state → bot : sync_to_bot()     — yeni modül state değiştirdiğinde
    bot method → event : wrap_*()   — eski metotları event-aware yapar

Kullanım:
    # loabot_main.py (v6.0) init'inde:
    from core.game_state import GameState
    from core.event_bus import EventBus
    from compat.bot_bridge import BotBridge

    self.game_state = GameState()
    self.event_bus = EventBus(log_fn=self.log)
    self.game_state.set_event_bus(self.event_bus)
    self.bridge = BotBridge(self, self.game_state, self.event_bus)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.game_state import GameState


class BotBridge:
    """
    self.bot.* ↔ GameState/EventBus iki yönlü köprü.

    Bu sınıf v5.9 → v6.0 geçiş döneminde her iki tarafı da
    tutarlı tutar. Tüm modüller v6.0'a migrate edildiğinde
    bu sınıf kaldırılabilir.
    """

    # bot.*  →  GameState anahtarı eşlemesi
    _ATTR_MAP: Dict[str, str] = {
        "_global_phase":             "phase",
        "_global_stage":             "stage",
        "_global_mission_reason":    "phase_reason",
        "_global_mission_extra":     "phase_extra",
        "_global_phase_ts":          "phase_ts",
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

        # running Event → bool dönüşümü
        running = getattr(bot, "running", None)
        if running and hasattr(running, "is_set"):
            self.state.set("running", running.is_set(), source="bridge_init")

    # ══════════════════════════════════════════════════════════════
    #  BOT → STATE (eski modül state değiştirdiğinde çağır)
    # ══════════════════════════════════════════════════════════════

    def sync_from_bot(self) -> None:
        """bot.* attribute'lerini GameState'e kopyala."""
        updates = {}
        for bot_attr, state_key in self._ATTR_MAP.items():
            val = getattr(self.bot, bot_attr, None)
            if val is not None:
                updates[state_key] = val

        # running Event özel dönüşüm
        running = getattr(self.bot, "running", None)
        if running and hasattr(running, "is_set"):
            updates["running"] = running.is_set()

        if updates:
            self.state.update(updates, source="bridge_from_bot")

    # ══════════════════════════════════════════════════════════════
    #  STATE → BOT (yeni modül GameState değiştirdiğinde çağır)
    # ══════════════════════════════════════════════════════════════

    def sync_to_bot(self) -> None:
        """GameState'i bot.* attribute'lerine geri yaz."""
        with self._sync_lock:
            for bot_attr, state_key in self._ATTR_MAP.items():
                val = self.state.get(state_key)
                if val is not None and hasattr(self.bot, bot_attr):
                    setattr(self.bot, bot_attr, val)

    # ══════════════════════════════════════════════════════════════
    #  WRAPPED METHODS — Eski metotları event-aware yapar
    # ══════════════════════════════════════════════════════════════

    def set_global_mission_phase(
        self,
        phase: str,
        stage: str,
        reason: str = "",
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Eski set_global_mission_phase() + yeni GameState + EventBus.

        Aynı anda hem eski hem yeni sistemi günceller.
        Eski modüller self.bot.set_global_mission_phase() çağırmaya
        devam edebilir — BotBridge interceptor olarak çalışır.
        """
        # 1. Eski API'yi çağır (log, training_logger vb. hala çalışsın)
        if hasattr(self.bot, "_original_set_global_mission_phase"):
            self.bot._original_set_global_mission_phase(phase, stage, reason, extra)
        else:
            # Doğrudan attribute set et (fallback)
            self.bot._global_phase = phase
            self.bot._global_stage = stage
            self.bot._global_mission_reason = reason
            self.bot._global_mission_extra = dict(extra or {})

        # 2. GameState güncelle (otomatik PHASE_CHANGED yayınlar)
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
        """Eski start_global_mission() + EventBus MISSION_STARTED."""
        from core.event_types import EVT

        # Eski API
        if hasattr(self.bot, "_original_start_global_mission"):
            self.bot._original_start_global_mission(phase, stage, reason, extra)
        else:
            self.set_global_mission_phase(phase, stage, reason, extra)

        # Yeni: MISSION_STARTED event
        self.bus.publish(
            EVT.MISSION_STARTED,
            {"phase": phase, "stage": stage, "reason": reason, "extra": extra or {}},
            source="bridge",
        )

    def stop_global_mission(self, reason: str = "") -> None:
        """Eski stop_global_mission() + EventBus MISSION_ENDED."""
        from core.event_types import EVT

        # Eski API
        if hasattr(self.bot, "_original_stop_global_mission"):
            self.bot._original_stop_global_mission(reason)
        else:
            self.set_global_mission_phase("IDLE_PHASE", "waiting", reason)

        # Yeni: MISSION_ENDED event
        self.bus.publish(
            EVT.MISSION_ENDED,
            {"reason": reason},
            source="bridge",
        )

    # ══════════════════════════════════════════════════════════════
    #  MONKEY-PATCH — Eski API'yi interceptle
    # ══════════════════════════════════════════════════════════════

    def install_interceptors(self) -> None:
        """
        bot.set_global_mission_phase() vb. çağrılarını yakala
        ve BotBridge üzerinden yönlendir.

        Bu sayede eski modüller HIÇBIR DEĞİŞİKLİK yapmadan
        EventBus'a otomatik event yayınlar.

        Çağrı: bridge.install_interceptors() — init sonrası bir kez.
        """
        bot = self.bot

        # Orijinalleri sakla
        if hasattr(bot, "set_global_mission_phase"):
            bot._original_set_global_mission_phase = bot.set_global_mission_phase
        if hasattr(bot, "start_global_mission"):
            bot._original_start_global_mission = bot.start_global_mission
        if hasattr(bot, "stop_global_mission"):
            bot._original_stop_global_mission = bot.stop_global_mission

        # Interceptor'ları yerleştir
        bot.set_global_mission_phase = self.set_global_mission_phase
        bot.start_global_mission = self.start_global_mission
        bot.stop_global_mission = self.stop_global_mission

        if hasattr(bot, "log"):
            bot.log("[BotBridge] Interceptor'lar kuruldu. Eski API -> EventBus köprüsü aktif.")

    def uninstall_interceptors(self) -> None:
        """Interceptor'ları kaldır, orijinal metotları geri yükle."""
        bot = self.bot
        for method in ("set_global_mission_phase", "start_global_mission", "stop_global_mission"):
            original = f"_original_{method}"
            if hasattr(bot, original):
                setattr(bot, method, getattr(bot, original))
                delattr(bot, original)

    # ══════════════════════════════════════════════════════════════
    #  PERİYODİK SYNC (thread'den çağrılır)
    # ══════════════════════════════════════════════════════════════

    def periodic_sync(self, interval: float = 1.0) -> None:
        """
        Arka plan thread'inde periyodik bot ↔ state senkronizasyonu.
        Daemon thread olarak başlatılabilir.
        """
        while True:
            try:
                self.sync_from_bot()
            except Exception:
                pass
            time.sleep(interval)

    def start_sync_thread(self, interval: float = 1.0) -> None:
        """Sync daemon thread'ini başlat."""
        t = threading.Thread(
            target=self.periodic_sync,
            args=(interval,),
            daemon=True,
            name="BotBridge-Sync",
        )
        t.start()
