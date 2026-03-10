"""
core/event_types.py — Merkezi Event Tip Sabitleri
===================================================
Tüm modüllerin ortak kullandığı event isimleri burada tanımlanır.
Serbest string yerine sabit kullanarak typo kaynaklı bug'ları önler.

Kullanım:
    from core.event_types import EVT

    bus.publish(EVT.VISION_UPDATE, {"objects": [...]})
    bus.subscribe(EVT.COMBAT_STARTED, my_handler)
"""

from __future__ import annotations


class _EventTypes:
    """Event tip sabitleri — namespace olarak kullanılır, örneklenmez."""

    # ── Vision Pipeline ───────────────────────────────────────────
    VISION_UPDATE       = "VISION_UPDATE"         # Her frame analizi sonrası
    ENEMY_DETECTED      = "ENEMY_DETECTED"        # Düşman tespit edildi
    OBJECT_FOUND        = "OBJECT_FOUND"          # Herhangi bir görsel bulundu

    # ── Combat Lifecycle ──────────────────────────────────────────
    COMBAT_STARTED      = "COMBAT_STARTED"        # Savaş başladı
    COMBAT_FINISHED     = "COMBAT_FINISHED"       # Savaş bitti (başarılı/başarısız)
    BOSS_KILLED         = "BOSS_KILLED"           # Boss öldürüldü
    BOSS_FAILED         = "BOSS_FAILED"           # Boss saldırısı başarısız

    # ── Navigation ────────────────────────────────────────────────
    NAVIGATION_TARGET   = "NAVIGATION_TARGET"     # Hedef konum belirlendi
    NAVIGATION_STARTED  = "NAVIGATION_STARTED"    # Navigasyon başladı
    NAVIGATION_COMPLETE = "NAVIGATION_COMPLETE"   # Hedefe ulaşıldı
    LOCATION_CHANGED    = "LOCATION_CHANGED"      # Bölge değişti

    # ── State Machine ─────────────────────────────────────────────
    PHASE_CHANGED       = "PHASE_CHANGED"         # _global_phase değişti
    MISSION_STARTED     = "MISSION_STARTED"       # Görev başladı
    MISSION_ENDED       = "MISSION_ENDED"         # Görev bitti

    # ── Anomaly Detection ─────────────────────────────────────────
    STUCK_DETECTED      = "STUCK_DETECTED"        # Bot takıldı
    POPUP_DETECTED      = "POPUP_DETECTED"        # Beklenmeyen popup
    PVP_THREAT          = "PVP_THREAT"            # PvP saldırı tespit

    # ── Action System ─────────────────────────────────────────────
    ACTION_REQUEST      = "ACTION_REQUEST"        # Aksiyon talebi (key/click)
    ACTION_COMPLETED    = "ACTION_COMPLETED"      # Aksiyon tamamlandı
    SEQUENCE_REQUEST    = "SEQUENCE_REQUEST"       # UI sequence çalıştır

    # ── Event System ──────────────────────────────────────────────
    TIMED_EVENT_READY   = "TIMED_EVENT_READY"     # Zamanlanmış etkinlik hazır
    EVENT_STARTED       = "EVENT_STARTED"         # Etkinlik başladı
    EVENT_FINISHED      = "EVENT_FINISHED"        # Etkinlik bitti

    # ── Loot & Reward ─────────────────────────────────────────────
    LOOT_STARTED        = "LOOT_STARTED"          # Ganimet toplama başladı
    LOOT_FINISHED       = "LOOT_FINISHED"         # Ganimet toplama bitti
    REWARD_UPDATE       = "REWARD_UPDATE"         # Ödül metrikleri güncellendi

    # ── Training / Recording ──────────────────────────────────────
    RECORDING_STARTED   = "RECORDING_STARTED"     # Video kaydı başladı
    RECORDING_STOPPED   = "RECORDING_STOPPED"     # Video kaydı bitti

    # ── System ────────────────────────────────────────────────────
    BOT_STARTED         = "BOT_STARTED"           # Bot başlatıldı
    BOT_STOPPED         = "BOT_STOPPED"           # Bot durduruldu
    BOT_PAUSED          = "BOT_PAUSED"            # Bot duraklatıldı
    BOT_RESUMED         = "BOT_RESUMED"           # Bot devam etti
    SHUTDOWN            = "SHUTDOWN"              # Kapanış sinyali


# Modüller bunu import eder: from core.event_types import EVT
EVT = _EventTypes()
