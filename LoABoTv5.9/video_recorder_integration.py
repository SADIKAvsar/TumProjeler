# -*- coding: utf-8 -*-
"""
video_recorder_integration.py — LoABoTv5.9 Video Kayıt Entegrasyonu
=====================================================================
Bu dosya, video_recorder.py'nin mevcut bot altyapısına nasıl bağlanacağını
gösteren yama (patch) kodlarını ve entegrasyon talimatlarını içerir.

3 dosyada değişiklik gereklidir:
  1. loabot_main.py  → VideoRecorder'ı başlat
  2. boss_manager.py → start/stop çağrılarını ekle
  3. automator.py    → Bot tıklama/tuş aksiyonlarını logla

Ayrıca opsiyonel:
  4. config/GeminiProConfig.yaml → video ayarları
"""


# ═══════════════════════════════════════════════════════════════════
#  YAMA 1: loabot_main.py → __init__ içinde VideoRecorder başlat
# ═══════════════════════════════════════════════════════════════════
#
# SequentialRecorder'ın başlatıldığı yere yakın eklenecek:
#
#   from video_recorder import VideoRecorder
#   self.video_recorder = VideoRecorder(self)
#
# shutdown() metoduna:
#   if hasattr(self, "video_recorder"):
#       self.video_recorder.shutdown()
#
# auto_start_recording ve auto_stop_recording metodlarına:
#   (mevcut seq_recorder çağrılarının yanına)

LOABOT_MAIN_INIT_PATCH = """
# ── video_recorder.py import ──
# loabot_main.py dosyasının başındaki import bloğuna ekleyin:
from video_recorder import VideoRecorder

# ── __init__ metodu içinde (seq_recorder'dan sonra) ──
# self.seq_recorder = SequentialRecorder(self)  # mevcut satır
self.video_recorder = VideoRecorder(self)       # YENİ
"""

LOABOT_MAIN_SHUTDOWN_PATCH = """
# ── shutdown() metodu içinde ──
# Mevcut cleanup kodlarının yanına:
if hasattr(self, "video_recorder"):
    self.video_recorder.shutdown()
"""

LOABOT_MAIN_AUTO_RECORDING_PATCH = """
# ── auto_start_recording() metodu ──
# Mevcut metodu güncelleyin veya yeni versiyon kullanın:

def auto_start_recording(self, trigger_type: str = "manual", timeout_sec: float = 300):
    \"\"\"Hem sequential hem video kaydını başlat.\"\"\"
    # Mevcut sequential recorder (geriye uyumluluk)
    seq = getattr(self, "seq_recorder", None)
    if seq is not None:
        seq.start(trigger_type)

    # YENİ: Video recorder
    vid = getattr(self, "video_recorder", None)
    if vid is not None:
        vid.start(trigger_type)

def auto_stop_recording(self):
    \"\"\"Hem sequential hem video kaydını durdur.\"\"\"
    seq = getattr(self, "seq_recorder", None)
    if seq is not None:
        seq.stop()

    vid = getattr(self, "video_recorder", None)
    if vid is not None:
        vid.stop()
"""


# ═══════════════════════════════════════════════════════════════════
#  YAMA 2: boss_manager.py → Video kayıt sinyalleri
# ═══════════════════════════════════════════════════════════════════
#
# boss_manager.py automation_thread() içinde zaten
# auto_start_recording ve auto_stop_recording çağrıları var.
# Yukarıdaki auto_start/stop yamaları uygulandığında
# boss_manager otomatik olarak video kaydını da tetikler.
#
# Ek olarak, başarı/başarısızlık durumlarında video'ya sinyal:

BOSS_MANAGER_SUCCESS_PATCH = """
# ── boss_manager.py, automation_thread() içinde ──
# Mevcut signal_success çağrısının yanına:

# Mevcut kod (satır ~641-644):
seq = getattr(self.bot, "seq_recorder", None)
if seq is not None:
    flush_done = bool(seq.signal_success(reason=f"boss_{boss_id}_success"))

# YENİ: Video recorder'a da başarı sinyali
vid = getattr(self.bot, "video_recorder", None)
if vid is not None:
    vid.stop(success=True, reason=f"boss_{boss_id}_success")
"""

BOSS_MANAGER_FAIL_PATCH = """
# ── boss_manager.py, finally bloğu içinde (satır ~656-660) ──
# Mevcut signal_fail çağrısının yanına:

seq = getattr(self.bot, "seq_recorder", None)
if seq is not None:
    seq.signal_fail(reason=f"boss_{boss_id}_fail")

# YENİ: Video recorder'a başarısızlık sinyali
vid = getattr(self.bot, "video_recorder", None)
if vid is not None and vid.is_recording:
    vid.stop(success=False, reason=f"boss_{boss_id}_fail")
"""


# ═══════════════════════════════════════════════════════════════════
#  YAMA 3: automator.py → Bot aksiyonlarını video'ya logla
# ═══════════════════════════════════════════════════════════════════
#
# Bot'un kendi tıklama ve tuş basma fonksiyonlarında
# video_recorder.log_action() çağrısı eklenmeli.

AUTOMATOR_CLICK_PATCH = """
# ── automator.py, safe_click() veya do_click() metodu içinde ──
# Tıklama gerçekleştirildikten HEMEN SONRA:

vid = getattr(self.bot, "video_recorder", None)
if vid is not None and vid.is_recording:
    vid.log_action(
        event_type="bot_click",
        data={"x": int(x), "y": int(y), "target": str(target_name)},
        source="bot",
        action_label=action_label,  # "attack_boss", "select_layer", vb.
    )
"""

AUTOMATOR_KEY_PATCH = """
# ── automator.py, press_key() metodu içinde ──
# Tuş basıldıktan HEMEN SONRA:

vid = getattr(self.bot, "video_recorder", None)
if vid is not None and vid.is_recording:
    vid.log_action(
        event_type="bot_key",
        data={"key": str(key)},
        source="bot",
        action_label=f"bot_key_{key}",
    )
"""


# ═══════════════════════════════════════════════════════════════════
#  YAMA 4: config/GeminiProConfig.yaml → Video ayarları (opsiyonel)
# ═══════════════════════════════════════════════════════════════════

CONFIG_YAML_PATCH = """
# config/GeminiProConfig.yaml dosyasına ekleyin:

recording:
  # Mevcut sequential ayarlar...
  sequential:
    enabled: false          # Artık video kullanıldığı için kapatılabilir
    # ... mevcut ayarlar ...

  # YENİ: Video kayıt ayarları
  video:
    enabled: true
    output_path: "D:\\\\LoABot_Training_Data\\\\videos"
    fps: 10
    input_color_order: "BGR"   # VisionManager çıktısına göre
"""


# ═══════════════════════════════════════════════════════════════════
#  DOĞRULAMA: Entegrasyon testi
# ═══════════════════════════════════════════════════════════════════

def verify_integration(bot) -> dict:
    """
    Bot nesnesinde video_recorder entegrasyonunu doğrula.
    Döndürülen dict'te her kontrol True/False olarak gösterilir.

    Kullanım:
        from video_recorder_integration import verify_integration
        result = verify_integration(bot)
        for check, ok in result.items():
            print(f"  {'✓' if ok else '✗'} {check}")
    """
    checks = {}

    # VideoRecorder var mı?
    vid = getattr(bot, "video_recorder", None)
    checks["video_recorder_exists"] = vid is not None

    if vid is None:
        return checks

    # Temel özellikler
    checks["has_start_method"] = callable(getattr(vid, "start", None))
    checks["has_stop_method"] = callable(getattr(vid, "stop", None))
    checks["has_log_action"] = callable(getattr(vid, "log_action", None))
    checks["has_shutdown"] = callable(getattr(vid, "shutdown", None))
    checks["is_enabled"] = vid._enabled
    checks["capture_thread_alive"] = vid._capture_thread.is_alive()
    checks["action_writer_alive"] = vid._action_writer_thread.is_alive()
    checks["vision_available"] = vid._has_vision()

    # pynput listeners
    checks["mouse_listener_active"] = vid._mouse_listener is not None
    checks["keyboard_listener_active"] = vid._keyboard_listener is not None

    return checks