"""
core/event_bus.py — Lightweight Thread-Safe Event Bus
======================================================
Modüller arası iletişimi doğrudan çağrı yerine publish/subscribe
patterni ile sağlar. Coupling'i dramatik şekilde azaltır.

Tasarım Kararları:
    - Senkron dispatch: publish() anında tüm handler'ları çağırır
      (oyun botu için <1ms latency kritik, async queue gereksiz overhead)
    - Thread-safe: _lock ile concurrent publish/subscribe korunur
    - Hata izolasyonu: Bir handler patlarsa diğerleri etkilenmez
    - Wildcard: "*" ile TÜM event'lere abone olunabilir (debug/logging)
    - Opsiyonel loglama: _log_fn set edilirse event akışı loglanır

Kullanım:
    from core.event_bus import EventBus
    from core.event_types import EVT

    bus = EventBus()

    # Subscribe
    bus.subscribe(EVT.VISION_UPDATE, my_handler)
    bus.subscribe("*", debug_logger)  # Wildcard: her şeyi logla

    # Publish
    bus.publish(EVT.VISION_UPDATE, {"objects": [...]})

    # Unsubscribe
    bus.unsubscribe(EVT.VISION_UPDATE, my_handler)

    # Named subscribe (modül bazlı temizlik için)
    bus.subscribe(EVT.COMBAT_STARTED, handler, name="combat_mgr")
    bus.unsubscribe_all(name="combat_mgr")
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ══════════════════════════════════════════════════════════════════════
#  Event Envelope
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Event:
    """Handler'a iletilen standart event zarfı."""
    event_type: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)
    source: str = ""  # Publish eden modülün adı (opsiyonel)


# ══════════════════════════════════════════════════════════════════════
#  Subscriber Kaydı
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _Subscription:
    handler: Callable[[Event], None]
    name: str = ""       # Modül bazlı unsubscribe için
    priority: int = 0    # Düşük = önce çalışır (gelecek kullanım)


# Callback tipi
EventHandler = Callable[[Event], None]

# Wildcard sabiti
_WILDCARD = "*"


# ══════════════════════════════════════════════════════════════════════
#  EVENT BUS
# ══════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Merkezi event dağıtım sistemi.

    Thread-safe, senkron dispatch, hata izolasyonlu.
    Tüm modüller aynı EventBus instance'ını paylaşır.
    """

    def __init__(self, log_fn: Optional[Callable] = None, debug: bool = False):
        """
        Args:
            log_fn: Opsiyonel loglama fonksiyonu (bot.log gibi)
            debug:  True ise her publish loglanır
        """
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[_Subscription]] = defaultdict(list)
        self._log_fn = log_fn
        self._debug = debug

        # İstatistikler
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
        """
        Event türüne abone ol.

        Args:
            event_type: Event tipi (EVT.VISION_UPDATE vb.) veya "*" (wildcard)
            handler:    Çağrılacak fonksiyon — imza: handler(event: Event)
            name:       Modül adı (toplu unsubscribe için)
            priority:   Düşük = önce çalışır (varsayılan: 0)
        """
        sub = _Subscription(handler=handler, name=name, priority=priority)
        with self._lock:
            subs = self._subscribers[event_type]
            # Aynı handler'ı iki kez ekleme
            if any(s.handler is handler for s in subs):
                return
            subs.append(sub)
            # Priority sıralaması
            subs.sort(key=lambda s: s.priority)

        if self._debug:
            self._log(f"[EventBus] +subscribe: {name or '?'} -> {event_type}")

    # ── UNSUBSCRIBE ───────────────────────────────────────────────

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Belirli bir handler'ı belirli bir event'ten çıkar."""
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            self._subscribers[event_type] = [
                s for s in subs if s.handler is not handler
            ]

    def unsubscribe_all(self, name: str = "", event_type: str = "") -> int:
        """
        Toplu unsubscribe.

        Args:
            name:       Bu isimle kayıtlı tüm abonelikleri kaldır
            event_type: Sadece bu event'ten kaldır (boşsa tümünden)

        Returns:
            Kaldırılan abonelik sayısı
        """
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
        """
        Event yayınla — tüm abone handler'ları senkron çağırır.

        Args:
            event_type: Event tipi
            payload:    Veri dict'i
            source:     Yayınlayan modül adı

        Returns:
            Çağrılan handler sayısı
        """
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

        # Snapshot: lock altında handler listesini kopyala, sonra lock dışında çağır
        with self._lock:
            # Event'e özel handler'lar + wildcard handler'lar
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
        """Bu event türüne abone var mı?"""
        with self._lock:
            return bool(self._subscribers.get(event_type))

    def subscriber_count(self, event_type: str = "") -> int:
        """Abone sayısı. event_type boşsa toplam."""
        with self._lock:
            if event_type:
                return len(self._subscribers.get(event_type, []))
            return sum(len(subs) for subs in self._subscribers.values())

    def get_stats(self) -> Dict[str, Any]:
        """Event bus istatistikleri."""
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
        """Tüm abonelikleri ve istatistikleri sıfırla (test/shutdown)."""
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
