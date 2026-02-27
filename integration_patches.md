# LoABot Agentic YZ — Entegrasyon Yamaları

Yeni modülleri (`user_input_monitor.py`, `reward_engine.py`, `seq_dataset_builder.py`)
mevcut koda bağlamak için gereken minimum değişiklikler.

---

## 1. `loabot_main.py`

### 1.1 Import ekle

```python
# Mevcut importların altına ekle:
from user_input_monitor import UserInputMonitor
from reward_engine import RewardEngine
```

### 1.2 `__init__` içinde modülleri başlat

```python
# self.popup_manager = PopupManager(self)  satırından SONRA:
self.user_monitor = UserInputMonitor(self)
self.reward_engine = RewardEngine(self)
```

### 1.3 `start_global_mission` API düzeltmesi

`TrainingLogger.start_episode()` signature'ı:  `(self, episode_type, context)` iken
`loabot_main.py` bunu `start_episode(mission_type=reason, context=extra)` olarak çağırıyor.
`mission_type` keyword'ü yok → sessizce `episode_type` default değeri kullanılıyor.

**Düzeltme** (`loabot_main.py` → `start_global_mission` metodu):
```python
# ÖNCE (hatalı):
self.training_logger.start_episode(mission_type=reason, context=extra)

# SONRA (doğru):
self.training_logger.start_episode(episode_type=reason, context=extra)
```

### 1.4 `stop_global_mission` — reward özeti ekle

```python
def stop_global_mission(self, reason: str = ""):
    self.set_global_mission_phase("IDLE_PHASE", "waiting", reason, {})
    if hasattr(self, "training_logger"):
        # YENİ: reward özetini ekle
        reward_summary = {}
        if hasattr(self, "reward_engine"):
            reward_summary = self.reward_engine.get_episode_summary()
            self.reward_engine.reset_episode()
        self.training_logger.end_episode(
            status="completed",
            reason=reason,
            metrics=reward_summary,   # TrainingLogger'ın metrics parametresi
        )
```

### 1.5 `_on_close` — user_monitor kapat

```python
def _on_close(self):
    self.log("Uygulama kapatiliyor...")
    try:
        if hasattr(self, "user_monitor"):
            self.user_monitor.shutdown()    # YENİ
    except Exception:
        pass
    try:
        if hasattr(self, "seq_recorder"):
            self.seq_recorder.shutdown()
    except Exception:
        pass
    # ...
```

---

## 2. `automator.py`

### 2.1 `press_key` içinde BOT_IS_PRESSING_KEY_EVENT set et

`UserInputMonitor` kullanıcı klavyesini filtrelerken
`BOT_IS_PRESSING_KEY_EVENT`'in set olmasına bakıyor.
Bunu `press_key()` içine eklememiz gerekiyor:

```python
from user_input_monitor import BOT_IS_PRESSING_KEY_EVENT

def press_key(self, key, label=None):
    decision_payload = {"key": str(key), "label": str(label or "")}
    decision_id = self._capture_action_decision("press_key", decision_payload, stage="key_decision")

    # YENİ: Bot'un kendi tuş basması olarak işaretle
    BOT_IS_PRESSING_KEY_EVENT.set()
    try:
        pyautogui.press(key)
    finally:
        BOT_IS_PRESSING_KEY_EVENT.clear()   # YENİ

    self.bot.log_training_action(
        "press_key",
        {"key": str(key), "label": str(label or ""), "success": True, "decision_id": decision_id},
    )
    self._seal_seq_action(f"key_{key}")
    return True
```

---

## 3. `sequential_recorder.py`

### 3.1 `_start_mouse_listener` devre dışı bırak

`UserInputMonitor` artık fare dinleyicisini de yönetiyor.
Çift kayıt önlemek için SequentialRecorder'daki listener'ı kaldır:

```python
def _start_mouse_listener(self):
    """Artık UserInputMonitor yönetiyor — devre dışı."""
    self.bot.log(
        "SequentialRecorder: Fare/klavye dinleyicisi UserInputMonitor'a devredildi.",
        level="DEBUG",
    )
    # Eski pynput kodu tamamen kaldırıldı.
    pass
```

---

## 4. `combat_manager.py` (veya boss döngüsü neredeyse)

### 4.1 Boss öldürmede reward tetikle

Boss kill tespiti yapılan yerde (victory_image bulunduğunda):

```python
# Boss öldürüldü tespitinden sonra:
kill_time = time.time() - boss_start_time  # boss saldırısı başlangıç zamanı
if hasattr(self.bot, "reward_engine"):
    self.bot.reward_engine.on_boss_killed(
        boss_name=str(boss.get("aciklama", "")),
        kill_time=kill_time,
    )
```

### 4.2 Ölüm / başarısızlık durumunda:

```python
if hasattr(self.bot, "reward_engine"):
    self.bot.reward_engine.on_death(boss_name=str(boss.get("aciklama", "")))
```

---

## 5. `game_manager.py`

### 5.1 Restart'ta reward tetikle

```python
# restart_game() içinde, başarılı restart sonrasında:
if hasattr(self.bot, "reward_engine"):
    self.bot.reward_engine.on_restart(reason="watchdog_restart")
```

---

## 6. `event_manager.py`

### 6.1 Etkinlik sonucunu reward'a bildir

```python
# _run_scheduled_event içinde, action_success sonrası:
if hasattr(self.bot, "reward_engine"):
    self.bot.reward_engine.on_event_entry(
        event_name=event["name"],
        success=action_success,
    )
```

---

## 7. `training_logger.py` — `start_episode` signature düzeltmesi

Mevcut:
```python
def start_episode(self, episode_type: str = "mission", context: dict = None):
```

Bu doğru, sadece `loabot_main.py`'daki çağrı düzeltilmeli (Madde 1.3).

---

## Veri Akışı (Güncellenmiş)

```
Kullanıcı klavye/fare
       ↓
 UserInputMonitor (pynput hook)
   BOT_IS_PRESSING_KEY_EVENT filtresi
       ↓
 SequentialRecorder.seal_action(
     action_label="key_a",
     extra={"trigger_type": "manual_user"}
 )
       ↓
 Sequence_XXXX/metadata.json
   + "reward": {"value": +0.1, ...}  ← RewardEngine
       ↓
 seq_dataset_builder.py
   → seq_train.jsonl + action_to_id.json
       ↓
 train_agentic_seq.py (LSTM katmanı eklenecek)
```

---

## Eksik: LSTM Eğitim Modeli

`PyTorchInferenceEngine` şu an tek kare (single-frame) kullanıyor.
`SequentialRecorder` 10 kare üretiyor ama model bunu kullanamıyor.

Gerekli yeni dosya: `train_agentic_seq.py`

Model değişikliği:
```python
class AgenticNetLSTM(nn.Module):
    def __init__(self, num_phases, num_actions, backbone="resnet18"):
        super().__init__()
        self.backbone = ResNet18(...)       # frame-level feature
        self.lstm = nn.LSTM(
            input_size=512,                 # ResNet18 feat_dim
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.phase_head  = nn.Linear(256, num_phases)
        self.action_head = nn.Linear(256, num_actions)

    def forward(self, frames_batch):
        # frames_batch: [B, T, C, H, W]
        B, T, C, H, W = frames_batch.shape
        feats = self.backbone(frames_batch.view(B*T, C, H, W))  # [B*T, 512]
        feats = feats.view(B, T, -1)                              # [B, T, 512]
        out, _ = self.lstm(feats)                                 # [B, T, 256]
        last = out[:, -1, :]                                      # [B, 256]
        return self.phase_head(last), self.action_head(last)
```

Bu modeli `pytorch_inference.py`'a entegre etmek için
`predict()` metodunu `[1, T, C, H, W]` tensor alacak şekilde güncelle.
```
