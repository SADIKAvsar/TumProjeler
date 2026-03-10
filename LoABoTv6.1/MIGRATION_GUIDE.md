# LoABot v6.0 → v6.1 Migration Guide

## Dosya Değişiklikleri Özeti

### YENİ / DEĞİŞEN DOSYALAR (v6.1 klasöründe)
| Dosya                     | Durum          | Açıklama                                          |
|---------------------------|----------------|---------------------------------------------------|
| `event_bus.py`            | BİRLEŞTİRİLDİ | EVT sabitleri event_types.py'den taşındı          |
| `game_state.py`           | SADELEŞTİ      | Dunder metotlar kaldırıldı, import düzeltildi     |
| `bot_bridge.py`           | BİRLEŞTİRİLDİ | bootstrap_v6() eklendi, ATTR_MAP küçültüldü       |
| `environment_manager.py`  | YENİ           | LocationManager + PopupManager birleştirildi       |
| `pvp_manager.py`          | GÜNCELLENDİ   | Evasion combo (Space/q/Space) eklendi             |
| `boss_manager.py`         | GÜNCELLENDİ   | Evasion combo kaldırıldı, auto recording kaldırıldı |
| `event_manager.py`        | GÜNCELLENDİ   | auto recording çağrıları kaldırıldı               |
| `loabot_main.py`          | GÜNCELLENDİ   | Import yolları, auto_start/stop sadeleştirildi    |
| `AI_MEMORY.md`            | GÜNCELLENDİ   | v6.1 bölümü eklendi                               |

### SİLİNEN DOSYALAR
| Dosya                           | Gerekçe                                  |
|---------------------------------|------------------------------------------|
| `event_types.py`                | event_bus.py içine alındı                |
| `bootstrap.py`                  | bot_bridge.py içine alındı               |
| `location_manager.py`           | environment_manager.py içine alındı      |
| `popup_manager.py`              | environment_manager.py içine alındı      |
| `video_recorder_integration.py` | İçeriği yalnızca eski yama talimatlarıydı|

### DEĞİŞMEYEN DOSYALAR (v6.0'dan aynen kalır)
- `utils.py`, `automator.py`, `combat_manager.py`, `game_manager.py`
- `vision_manager.py`, `gui_manager.py`, `tactical_brain.py`
- `ai_engine.py`, `ai_config.yaml`, `click_knowledge.py`, `click_map.yaml`
- `ollama_client.py`, `pytorch_inference.py`, `memory_manager.py`
- `reward_engine.py`, `training_logger.py`, `user_input_monitor.py`
- `analytics.py`, `video_recorder.py` (not integration)
- `dataset_builder.py`, `seq_dataset_builder.py`, `video_dataset_builder.py`
- `sanitize_video_sessions.py`, `train_agentic.py`, `loabot_dataset.py`
- `GeminiProConfig.yaml`

## Uygulama Adımları

1. v6.1 klasöründeki dosyaları proje köküne kopyalayın
2. Silinen dosyaları (event_types.py, bootstrap.py, location_manager.py, popup_manager.py, video_recorder_integration.py) kaldırın
3. `core/` alt klasörü varsa kaldırın (artık flat import kullanılıyor)
4. GeminiProConfig.yaml'a PvP bölümüne `evasion_cooldown_sn: 3.0` ekleyin
5. Test çalıştırın, log'larda "[v6.1]" etiketini doğrulayın

## Config Eklentisi

```yaml
# GeminiProConfig.yaml → PVP_DEFENSE bölümüne ekleyin:
PVP_DEFENSE:
  enabled: true
  evasion_cooldown_sn: 3.0    # YENİ: Evasion combo tekrar aralığı (saniye)
  # ... mevcut ayarlar aynen kalır ...
```
