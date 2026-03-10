"""
event_bus.py — Lightweight Thread-Safe Event Bus + Event Constants
===================================================================
v6.1: event_types.py bu dosyaya birlestirildi.

Modüller arası iletişimi publish/subscribe patterni ile sağlar.
EVT sabitleri de burada tanimlidir.

Kullanım:
    from event_bus import EventBus, EVT, Event

    bus = EventBus()
    bus.subscribe(EVT.VISION_UPDATE, my_handler)
    bus.publish(EVT.VISION_UPDATE, {"objects": [...]})
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ══════════════════════════════════════════════════════════════════════
#  EVENT TIP SABITLERI  (eski event_types.py)
# ══════════════════════════════════════════════════════════════════════

class EVT:
    """Event tip sabitleri — namespace olarak kullanılır, örneklenmez."""

    # ── Vision Pipeline ───────────────────────────────────────────
    VISION_UPDATE       = "VISION_UPDATE"
    ENEMY_DETECTED      = "ENEMY_DETECTED"
    OBJECT_FOUND        = "OBJECT_FOUND"

    # ── Combat Lifecycle ──────────────────────────────────────────
    COMBAT_STARTED      = "COMBAT_STARTED"
    COMBAT_FINISHED     = "COMBAT_FINISHED"
    BOSS_KILLED         = "BOSS_KILLED"
    BOSS_FAILED         = "BOSS_FAILED"

    # ── Navigation ────────────────────────────────────────────────
    NAVIGATION_TARGET   = "NAVIGATION_TARGET"
    NAVIGATION_STARTED  = "NAVIGATION_STARTED"
    NAVIGATION_COMPLETE = "NAVIGATION_COMPLETE"
    LOCATION_CHANGED    = "LOCATION_CHANGED"

    # ── State Machine ─────────────────────────────────────────────
    PHASE_CHANGED       = "PHASE_CHANGED"
    MISSION_STARTED     = "MISSION_STARTED"
    MISSION_ENDED       = "MISSION_ENDED"

    # ── Anomaly Detection ─────────────────────────────────────────
    STUCK_DETECTED      = "STUCK_DETECTED"
    POPUP_DETECTED      = "POPUP_DETECTED"
    PVP_THREAT          = "PVP_THREAT"

    # ── Action System ─────────────────────────────────────────────
    ACTION_REQUEST      = "ACTION_REQUEST"
    ACTION_COMPLETED    = "ACTION_COMPLETED"
    SEQUENCE_REQUEST    = "SEQUENCE_REQUEST"

    # ── Event System ──────────────────────────────────────────────
    TIMED_EVENT_READY   = "TIMED_EVENT_READY"
    EVENT_STARTED       = "EVENT_STARTED"
    EVENT_FINISHED      = "EVENT_FINISHED"

    # ── Loot & Reward ─────────────────────────────────────────────
    LOOT_STARTED        = "LOOT_STARTED"
    LOOT_FINISHED       = "LOOT_FINISHED"
    REWARD_UPDATE       = "REWARD_UPDATE"

    # ── Training / Recording ──────────────────────────────────────
    RECORDING_STARTED   = "RECORDING_STARTED"
    RECORDING_STOPPED   = "RECORDING_STOPPED"

    # ── System ────────────────────────────────────────────────────
    BOT_STARTED         = "BOT_STARTED"
    BOT_STOPPED         = "BOT_STOPPED"
    BOT_PAUSED          = "BOT_PAUSED"
    BOT_RESUMED         = "BOT_RESUMED"
    SHUTDOWN            = "SHUTDOWN"


# ══════════════════════════════════════════════════════════════════════
#  Event Envelope
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Event:
    """Handler'a iletilen standart event zarfı."""
    event_type: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)
    source: str = ""


# ══════════════════════════════════════════════════════════════════════
#  Subscriber Kaydı
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _Subscription:
    handler: Callable[[Event], None]
    name: str = ""
    priority: int = 0


EventHandler = Callable[[Event], None]
_WILDCARD = "*"


# ══════════════════════════════════════════════════════════════════════
#  EVENT BUS
# ══════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Merkezi event dağıtım sistemi.
    Thread-safe, senkron dispatch, hata izolasyonlu.
    """

    def __init__(self, log_fn: Optional[Callable] = None, debug: bool = False):
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[_Subscription]] = defaultdict(list)
        self._log_fn = log_fn
        self._debug = debug
        self._publish_count: int = 0
        self._error_count: int = 0

    # ── SUBSCRIBE ─────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        name: str = "",
        priority: int = 0,
    ) -> None:
        sub = _Subscription(handler=handler, name=name, priority=priority)
        with self._lock:
            subs = self._subscribers[event_type]
            if any(s.handler is handler for s in subs):
                return
            subs.append(sub)
            subs.sort(key=lambda s: s.priority)

        if self._debug:
            self._log(f"[EventBus] +subscribe: {name or '?'} -> {event_type}")

    # ── UNSUBSCRIBE ───────────────────────────────────────────────

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            self._subscribers[event_type] = [
                s for s in subs if s.handler is not handler
            ]

    def unsubscribe_all(self, name: str = "", event_type: str = "") -> int:
        removed = 0
        with self._lock:
            types = [event_type] if event_type else list(self._subscribers.keys())
            for et in types:
                before = len(self._subscribers[et])
                if name:
                    self._subscribers[et] = [
                        s for s in self._subscribers[et] if s.name != name
                    ]
                else:
                    self._subscribers[et] = []
                removed += before - len(self._subscribers[et])
        return removed

    # ── PUBLISH ───────────────────────────────────────────────────

    def publish(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> int:
        event = Event(
            event_type=event_type,
            payload=dict(payload or {}),
            source=source,
        )

        self._publish_count += 1

        if self._debug:
            self._log(
                f"[EventBus] publish: {event_type} "
                f"from={source or '?'} "
                f"keys={list((payload or {}).keys())[:5]}"
            )

        with self._lock:
            specific = list(self._subscribers.get(event_type, []))
            wildcards = list(self._subscribers.get(_WILDCARD, []))

        handlers = specific + wildcards
        called = 0

        for sub in handlers:
            try:
                sub.handler(event)
                called += 1
            except Exception as exc:
                self._error_count += 1
                self._log(
                    f"[EventBus] HATA: {event_type} handler "
                    f"({sub.name or '?'}): {exc}"
                )

        return called

    # ── SORGULAMA ─────────────────────────────────────────────────

    def has_subscribers(self, event_type: str) -> bool:
        with self._lock:
            return bool(self._subscribers.get(event_type))

    def subscriber_count(self, event_type: str = "") -> int:
        with self._lock:
            if event_type:
                return len(self._subscribers.get(event_type, []))
            return sum(len(subs) for subs in self._subscribers.values())

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            topics = {
                et: len(subs) for et, subs in self._subscribers.items() if subs
            }
        return {
            "total_published": self._publish_count,
            "total_errors": self._error_count,
            "active_topics": len(topics),
            "total_subscriptions": sum(topics.values()),
            "topics": topics,
        }

    # ── RESET ─────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._subscribers.clear()
        self._publish_count = 0
        self._error_count = 0

    # ── YARDIMCI ──────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._log_fn:
            try:
                self._log_fn(msg)
            except Exception:
                pass
