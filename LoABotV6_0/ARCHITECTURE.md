# LoABot v6.0 — GameState + EventBus Mimarisi

## Mimari Diyagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LoABot v6.0                                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     CORE INFRASTRUCTURE                         │   │
│  │                                                                 │   │
│  │   ┌──────────────┐         ┌──────────────┐                    │   │
│  │   │  GameState    │◄───────►│  EventBus    │                    │   │
│  │   │              │  auto   │              │                    │   │
│  │   │ phase        │ publish │ subscribe()  │                    │   │
│  │   │ target       │ ──────► │ publish()    │                    │   │
│  │   │ region       │         │ unsubscribe()│                    │   │
│  │   │ combat_state │         │              │                    │   │
│  │   │ vision_data  │         │ Wildcard: *  │                    │   │
│  │   │ ...          │         │              │                    │   │
│  │   └──────┬───────┘         └──────┬───────┘                    │   │
│  │          │                        │                            │   │
│  │   ┌──────┴────────────────────────┴───────┐                    │   │
│  │   │           BotBridge (compat/)          │                    │   │
│  │   │                                       │                    │   │
│  │   │  self.bot.*  ◄──► GameState           │                    │   │
│  │   │  old methods ──► EventBus publish      │                    │   │
│  │   │  interceptors: monkey-patch eski API   │                    │   │
│  │   └───────────────────────────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      MODULE LAYER                               │   │
│  │                                                                 │   │
│  │  ┌──────────────┐   VISION_UPDATE   ┌──────────────┐          │   │
│  │  │ VisionManager│──────────────────►│TacticalBrain │          │   │
│  │  │              │   ENEMY_DETECTED  │              │          │   │
│  │  │ publish()    │──────────────────►│ subscribe()  │          │   │
│  │  └──────────────┘                   └──────┬───────┘          │   │
│  │                                            │                   │   │
│  │                                     ACTION_REQUEST             │   │
│  │                                            │                   │   │
│  │  ┌──────────────┐   COMBAT_STARTED  ┌──────▼───────┐          │   │
│  │  │CombatManager │◄─────────────────│  Automator   │          │   │
│  │  │              │   BOSS_KILLED     │              │          │   │
│  │  │ subscribe()  │────────────────►│ subscribe()  │          │   │
│  │  │ publish()    │                   │ execute()    │          │   │
│  │  └──────────────┘                   └──────────────┘          │   │
│  │                                                                │   │
│  │  ┌──────────────┐   TIMED_EVENT     ┌──────────────┐          │   │
│  │  │ BossManager  │◄─────────────────│EventManager  │          │   │
│  │  │              │   BOSS_KILLED     │              │          │   │
│  │  │ publish()    │────────────────►│ subscribe()  │          │   │
│  │  └──────────────┘                   └──────────────┘          │   │
│  │                                                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │  PvPManager  │  │PopupManager  │  │RewardEngine  │         │   │
│  │  │  PVP_THREAT  │  │POPUP_DETECTED│  │REWARD_UPDATE │         │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Event Akış Diyagramı

```
Boss Saldırı Döngüsü (Event-Driven):

  BossManager                    EventBus                     CombatManager
  ──────────                    ────────                     ──────────────
       │                            │                              │
       │  publish(NAVIGATION_STARTED)│                              │
       │───────────────────────────►│                              │
       │                            │  NAVIGATION_STARTED ────────►│
       │                            │                              │
       │  state.set_phase(NAV)      │                              │
       │───────► GameState          │  PHASE_CHANGED ────────────►│
       │                            │                              │
       │  publish(COMBAT_STARTED)   │                              │
       │───────────────────────────►│                              │
       │                            │  COMBAT_STARTED ────────────►│
       │                            │              ┌───────────────┘
       │                            │              │ subscribe handler
       │                            │              │ starts monitoring
       │                            │              │
       │                            │◄─────────────┘
       │                            │  publish(BOSS_KILLED)
       │                            │──────────────────────────────►
       │◄───────────────────────────│  BOSS_KILLED
       │  handle: recalculate_times │
       │  state.set_phase(LOOT)     │
       │                            │  PHASE_CHANGED
       │                            │────────► RewardEngine
       │                            │────────► TrainingLogger
       │  publish(LOOT_FINISHED)    │
       │───────────────────────────►│
       │                            │  LOOT_FINISHED
       │  _reset_combat_state()     │────────► EventManager (chain check)
       │                            │
       │  state.set_phase(IDLE)     │
       │───────► GameState          │  PHASE_CHANGED
       │                            │────────► TacticalBrain
       │                            │────────► GUI (status update)
```

## Dosya Yapısı

```
LoABotV6_0/
├── core/                          ★ YENİ — Çekirdek altyapı
│   ├── __init__.py                  GameState, EventBus, EVT export
│   ├── game_state.py                Thread-safe merkezi durum
│   ├── event_bus.py                 Pub/sub event sistemi
│   ├── event_types.py               Event tip sabitleri (EVT.*)
│   └── bootstrap.py                 Tek satırla kurulum
│
├── compat/                        ★ YENİ — Geriye uyumluluk
│   ├── __init__.py
│   └── bot_bridge.py                self.bot.* ↔ GameState köprüsü
│
├── loabot_main.py                 ★ DEĞİŞTİ — 3 satır eklendi
├── vision_manager.py              Aynı (EventBus opt-in)
├── combat_manager.py              Aynı (EventBus opt-in)
├── boss_manager.py                Aynı (EventBus opt-in)
├── tactical_brain.py              Aynı (EventBus opt-in)
├── event_manager.py               Aynı
├── automator.py                   Aynı
├── ... (diğer modüller)           Aynı
│
├── config/                        Aynı
├── prompts/                       Aynı
├── train_full.bat                 Aynı
└── ARCHITECTURE.md                ★ YENİ — Bu dosya
```

## Migrasyon Stratejisi: Strangler Fig Pattern

```
Faz 1 (MEVCUT): Altyapı Kurulumu
  ✓ core/ oluşturuldu (GameState + EventBus + event_types)
  ✓ compat/bot_bridge.py oluşturuldu
  ✓ loabot_main.py'ye bootstrap_v6() eklendi
  ✓ Interceptor'lar eski API'yi otomatik event'e çeviriyor
  → Tüm eski modüller DEĞİŞMEDEN çalışıyor

Faz 2 (SONRAKI): Modül Migrasyon (opt-in)
  Modüller tek tek migrate edilir, eski API çalışmaya devam eder:
  □ VisionManager: find() sonrası → bus.publish(VISION_UPDATE)
  □ CombatManager: is_in_active_combat() → state.is_in_combat
  □ TacticalBrain: self.bot.vision.find() → state.get("detected_objects")
  □ BossManager: self.bot._global_phase → state.phase
  □ EventManager: self.bot.combat.is_in_active_combat() → state.is_in_combat

Faz 3 (GELECEK): Direct Call Elimination
  □ self.bot.combat.* → bus.publish(ACTION_REQUEST, ...)
  □ self.bot.automator.* → bus.publish(SEQUENCE_REQUEST, ...)
  □ self.bot.vision.* → state.get("vision_*")

Faz 4 (UZUN VADE): BotBridge Kaldırma
  □ Tüm modüller GameState + EventBus kullanıyor
  □ BotBridge gereksiz hale geldi → kaldır
  □ self.bot.* erişimi sadece config/settings için
```

## Modül Migrasyon Örnekleri

### VisionManager (Publisher Örneği)

```python
# ESKİ (v5.9):
def find(self, image_name, region, confidence, **kwargs):
    result = self._template_match(image_name, region, confidence)
    return result  # Sadece döndürür, başka modül bilmez

# YENİ (v6.0 — geriye uyumlu):
def find(self, image_name, region, confidence, **kwargs):
    result = self._template_match(image_name, region, confidence)

    # ★ v6.0: Bulunan nesneyi EventBus'a yayınla
    if result and hasattr(self.bot, 'event_bus') and self.bot.event_bus:
        self.bot.event_bus.publish(
            EVT.OBJECT_FOUND,
            {"image": image_name, "region": region, "confidence": confidence},
            source="vision"
        )

    return result  # Eski API aynı çalışıyor
```

### CombatManager (Subscriber + State Reader Örneği)

```python
# ESKİ (v5.9):
def is_in_active_combat(self) -> bool:
    attacking = bool(getattr(self.bot, "attacking_target_aciklama", None))
    phase = str(getattr(self.bot, "_global_phase", "")).strip().upper()
    in_active_phase = phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}
    return attacking or in_active_phase

# YENİ (v6.0 — GameState okuma):
def is_in_active_combat(self) -> bool:
    # ★ v6.0: Önce GameState'ten oku (daha güvenilir)
    state = getattr(self.bot, 'game_state', None)
    if state:
        return state.is_in_combat

    # Fallback: eski yöntem (bridge henüz kurulmamışsa)
    attacking = bool(getattr(self.bot, "attacking_target_aciklama", None))
    phase = str(getattr(self.bot, "_global_phase", "")).strip().upper()
    return attacking or phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}
```

### TacticalBrain (Multi-Event Subscriber Örneği)

```python
# ESKİ (v5.9):
class TacticalBrain:
    def __init__(self, bot):
        self.bot = bot
        # Diğer modülleri doğrudan çağırır:
        #   self.bot.vision.find(...)
        #   self.bot.location_manager.get_region_name()

# YENİ (v6.0 — event-driven):
class TacticalBrain:
    def __init__(self, bot):
        self.bot = bot

        # ★ v6.0: Event'lere abone ol
        bus = getattr(bot, 'event_bus', None)
        if bus:
            bus.subscribe(EVT.VISION_UPDATE, self._on_vision_update, name="brain")
            bus.subscribe(EVT.COMBAT_STARTED, self._on_combat_started, name="brain")
            bus.subscribe(EVT.STUCK_DETECTED, self._on_stuck, name="brain")

    def _on_vision_update(self, event):
        """Vision verisi geldiğinde otomatik analiz başlat."""
        objects = event.payload.get("objects", [])
        # ... analiz mantığı ...

    def _on_combat_started(self, event):
        """Savaş başladığında taktik plan oluştur."""
        target = event.payload.get("target")
        # ... taktik planlama ...
```

## Performans Hedefleri

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| EventBus publish latency | < 0.1ms | Senkron dispatch, no queue |
| GameState read latency | < 0.01ms | RLock + dict lookup |
| GameState write latency | < 0.05ms | RLock + dict update + event |
| Bridge sync overhead | < 1ms/sync | 0.5s interval, 10 attr |
| Memory overhead | < 5MB | State dict + subscriber list |
| Event throughput | > 10K/s | Test: 10K publish in <1s |

## Event Tipi Referansı

| Event | Yayıncı | Abone(ler) | Payload |
|-------|---------|------------|---------|
| VISION_UPDATE | VisionManager | TacticalBrain, CombatManager | objects, frame_ts |
| ENEMY_DETECTED | VisionManager | CombatManager, PvPManager | enemy_type, position |
| COMBAT_STARTED | GameState/Bridge | CombatManager, RewardEngine | target |
| COMBAT_FINISHED | GameState/Bridge | BossManager, EventManager | reason |
| BOSS_KILLED | BossManager | RewardEngine, TrainingLogger | boss_name, kill_time |
| PHASE_CHANGED | GameState | TacticalBrain, GUI, all | new_phase, stage, reason |
| STUCK_DETECTED | GameManager | TacticalBrain, Automator | duration, last_action |
| ACTION_REQUEST | TacticalBrain | Automator | action_type, params |
| NAVIGATION_TARGET | BossManager | Automator, LocationManager | target, protocol |
| LOOT_FINISHED | CombatManager | BossManager, EventManager | success |
| PVP_THREAT | PvPManager | CombatManager, Automator | threat_level |
| POPUP_DETECTED | PopupManager | Automator | popup_type |
