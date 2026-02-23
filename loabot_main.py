import os
import threading
import time
import tkinter as tk
from datetime import datetime
from queue import Queue
import subprocess
import yaml

from utils import CONFIG_FILE, log_to_file
from vision_manager import VisionManager
from game_manager import GameManager
from combat_manager import CombatManager
from location_manager import LocationManager
from automator import Automator
from event_manager import EventManager
from boss_manager import BossManager
from gui_manager import GUIManager
from pvp_manager import PvPManager
from popup_manager import PopupManager
from tactical_brain import TacticalBrain
from training_logger import TrainingLogger
from sequential_recorder import SequentialRecorder


class LoABot:
    def __init__(self, root):
        self.root = root
        self.gui_queue = Queue()

        self.config = self._load_initial_config()
        if not self.config:
            self.initialized = False
            return

        self.general_cfg = self.config.get("module_general", {})
        self.demon_boss_cfg = self.config.get("module_demon_boss", {})
        self.timed_events_cfg = self.config.get("module_timed_events", {})

        self.pvp_config = self.general_cfg.get("PVP_DEFENSE", {})
        self.popup_config = self.general_cfg.get("POPUP_MONITOR", {})

        self._setup_properties()

        # 1) Modülleri baslat
        self.brain = TacticalBrain(self)
        self.training_logger = TrainingLogger(self)

        self.vision = VisionManager(self)
        self.seq_recorder = SequentialRecorder(self)

        self.game = GameManager(self)
        self.combat = CombatManager(self)
        self.location_manager = LocationManager(self, self.general_cfg)
        self.automator = Automator(self)
        self.event_manager = EventManager(self, self.timed_events_cfg)
        self.boss_manager = BossManager(self)
        self.gui_manager = GUIManager(self)
        self.pvp_manager = PvPManager(self)
        self.popup_manager = PopupManager(self)

        # 2) Threadleri baslat
        self._start_threads()

        self._initialize_boss_data()

        self.initialized = True
        self.log("LoABot v5.9: AgenticNet & RL Altyapisiyla Baslatildi")

    def _load_initial_config(self) -> dict:
        self.log("Config (GeminiProConfig.yaml) yukleniyor...")
        if not os.path.exists(CONFIG_FILE):
            self.log(f"HATA: {CONFIG_FILE} bulunamadi!")
            return {}
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.log(f"Config okuma hatasi: {e}")
            return {}

    def _setup_properties(self):
        self.settings = self.general_cfg.get("settings", {})
        self.process_name = self.general_cfg.get("process_name", "loa-hf-plugin.exe")
        self.game_path = self.general_cfg.get("game_path", "")
        self.ui_regions = self.general_cfg.get("ui_regions", {})

        # Boss-related config: demon_boss_cfg'den (module_demon_boss)
        self.bosslar = self.demon_boss_cfg.get("bosslar", {})
        self.coordinates = self.demon_boss_cfg.get("coordinates", {})
        self.boss_sequence_template = self.demon_boss_cfg.get("sequence_template", [])
        self.victory_check_defaults = self.demon_boss_cfg.get("victory_check_defaults", {})
        if bool(self.settings.get("WALK_TIME_ENABLED", False)):
            self.walk_times = self.demon_boss_cfg.get("walk_times", {})
        else:
            self.walk_times = {}
        self.running = threading.Event()
        self.action_lock = threading.Lock()
        self.paused = False
        self.active_event = None
        self.attacking_target_aciklama = None

        self._global_phase = "IDLE_PHASE"
        self._global_stage = "waiting"
        self._global_mission_reason = ""
        self._global_mission_extra = {}

        self._manual_recording = False
        self._auto_log_active = False
        self._auto_log_trigger = ""
        self._auto_log_timer = None

    def _start_threads(self):
        threading.Thread(target=self.boss_manager.automation_thread, daemon=True).start()
        threading.Thread(target=self.game.watchdog_thread, daemon=True).start()
        threading.Thread(target=self.event_manager.event_monitor_thread, daemon=True).start()
        threading.Thread(target=self.pvp_manager.monitor_thread, daemon=True).start()
        threading.Thread(target=self.popup_manager.monitor_thread, daemon=True).start()
        if hasattr(self.vision, "shadow_mode_thread"):
            threading.Thread(target=self.vision.shadow_mode_thread, daemon=True).start()

    def _initialize_boss_data(self):
        """Boss runtime alanlarını başlat (backup'tan restore edildi)."""
        for name, target in self.bosslar.items():
            target["spawn_time"] = None
            target["aciklama"] = str(name)
            target["_last_attack_ts"] = 0.0
            target.setdefault("_fail_count", 0)
            target.setdefault("_area_check_ok", False)

            # koordinat_ref → koordinat çözümle
            ref = target.get("koordinat_ref")
            if isinstance(ref, str) and ref in self.coordinates:
                target["koordinat"] = self.coordinates[ref]
            elif isinstance(ref, dict):
                target["koordinat"] = ref

            # anchor → gorsel_dogrulama uyumluluğu
            if isinstance(target.get("anchor"), dict) and not isinstance(target.get("gorsel_dogrulama"), dict):
                target["gorsel_dogrulama"] = target["anchor"]

            # victory.image_file → victory_image uyumluluğu
            if not target.get("victory_image"):
                victory_cfg = target.get("victory")
                if isinstance(victory_cfg, dict):
                    img = victory_cfg.get("image_file")
                    if img:
                        target["victory_image"] = img

        self.log(f"Boss data hazir: {len(self.bosslar)} boss yuklendi.")

    def log(self, msg: str, level: str = "INFO"):
        try:
            log_to_file(msg)
            self.gui_queue.put(("log", msg))
        except Exception:
            pass

    def log_training_action(self, action_name: str, payload: dict = None):
        if hasattr(self, "training_logger"):
            self.training_logger.log_action(action_name, payload)

    def log_training_state(self, state_name: str, payload: dict = None):
        if hasattr(self, "training_logger"):
            self.training_logger.log_state(state_name, payload)

    def log_training_outcome(self, outcome_name: str, payload: dict = None):
        if hasattr(self, "training_logger"):
            self.training_logger.log_outcome(outcome_name, payload)

    def set_global_mission_phase(self, phase: str, stage: str, reason: str = "", extra: dict = None):
        with self.action_lock:
            self._global_phase = phase
            self._global_stage = stage
            self._global_mission_reason = reason
            self._global_mission_extra = dict(extra or {})

        self.log_training_state(
            "mission_phase_change",
            {
                "new_phase": phase,
                "new_stage": stage,
                "reason": reason,
                **self._global_mission_extra,
            },
        )
        if hasattr(self, "vision"):
            self.vision.set_mission_namespace(phase, stage)

    def start_global_mission(self, phase: str, stage: str, reason: str = "", extra: dict = None):
        self.set_global_mission_phase(phase, stage, reason, extra)
        if hasattr(self, "training_logger"):
            self.training_logger.start_episode(mission_type=reason, context=extra)

    def stop_global_mission(self, reason: str = ""):
        self.set_global_mission_phase("IDLE_PHASE", "waiting", reason, {})
        if hasattr(self, "training_logger"):
            self.training_logger.end_episode(status="completed", reason=reason)

    def capture_local_decision_frame(
        self, action_name: str, payload: dict = None, phase: str = None, stage: str = None
    ) -> str:
        """SequentialRecorder + AI egitimi icin karar ani karesi (snapshot)."""
        if not hasattr(self, "vision"):
            return ""
        return self.vision.save_ai_decision_snapshot(
            action_name=action_name,
            payload=payload,
            phase=phase or self._global_phase,
            stage=stage or self._global_stage,
        )

    def toggle_recording(self):
        # MANUEL KAYIT TAMAMEN DEVRE DIŞI BIRAKILDI
        self.log("Manuel kayıt butonu devre dışı bırakıldı. Kayıt kontrolü tamamen Otonom Ajan'da.", level="WARNING")
        return

    def start(self):
        """GUI ▶ butonu: Botu baslatir veya duraklamadan cikarir."""
        if self.paused:
            self.paused = False
            self.log("Bot devam ettiriliyor.")
        if not self.running.is_set():
            self.running.set()
            self.log("Bot baslatildi.")

    def stop(self):
        """GUI ⏸ butonu: Botu duraklatir."""
        if not self.paused:
            self.paused = True
            self.log("Bot duraklatildi.")

    def toggle_pause(self):
        self.paused = not self.paused
        durum = "DURAKLATILDI" if self.paused else "DEVAM EDIYOR"
        self.log(f"Sistem: {durum}")

        btn_text = "Devam Et (F6)" if self.paused else "Duraklat (F6)"
        btn_bg = "#FFA500" if self.paused else "#333333"
        self.gui_queue.put(("update_widget", ("pause_btn", {"text": btn_text, "bg": btn_bg})))

    def _interruptible_wait(self, seconds: float) -> bool:
        """Kilitlenme olmadan belirlenen sure kadar bekler."""
        if seconds <= 0:
            return self.running.is_set()
        end_time = time.time() + seconds
        while time.time() < end_time:
            if not self.running.is_set():
                return False
            time.sleep(0.1)
        return True

    def _set_spawn_time(self, boss_adi: str, kalan_saniye: int, source: str = "manual"):
        """GUI zamanlayıcısı: kalan süre (saniye) → mutlak spawn timestamp hesapla."""
        ts = time.time() + int(kalan_saniye)
        self._set_spawn_time_abs(boss_adi, ts, source)

    def _set_spawn_time_abs(self, boss_adi: str, ts: float, source: str = "manual"):
        if boss_adi in self.bosslar:
            self.bosslar[boss_adi]["spawn_time"] = ts
            self.bosslar[boss_adi]["_source"] = source
            st_str = time.strftime("%H:%M:%S", time.localtime(ts))
            self.log(f"{boss_adi} manuel ayarlandi: {st_str}")
            self.gui_queue.put(("set_spawn", (boss_adi, ts, source)))

    def _normalize_action_name(self, action) -> str:
        name = str(action or "").strip().lower()
        aliases = {
            "clcik": "click",
            "katman_secimi": "katman_secimi",
            "katman_seçimi": "katman_secimi",
            "katman_seã§imi": "katman_secimi",
            "boss_secimi": "boss_secimi",
            "boss_seçimi": "boss_secimi",
            "boss_seã§imi": "boss_secimi",
        }
        return aliases.get(name, name)

    def _run_sequence(self, sequence: list, coord_map: dict, context_target: dict = None) -> bool:
        """Belirtilen tiklama sekansini calistirir."""
        if not sequence:
            self.log("UYARI: Bos sekans calistirilmaya calisildi.")
            return False

        for idx, step in enumerate(sequence):
            if not self.running.is_set():
                return False

            action = self._normalize_action_name(step.get("action"))
            label = step.get("label")
            try:
                wait_ms = int(step.get("wait_ms", 100))
            except (TypeError, ValueError):
                wait_ms = 100

            success = False

            if action == "click":
                coord = coord_map.get(label)
                if not coord:
                    self.log(f"Sekans hatasi: '{label}' koordinati bulunamadi.")
                    return False
                success = self.automator.click(label, coord)

            elif action == "press_key":
                key = step.get("key")
                if not key:
                    self.log("Sekans hatasi: press_key adiminda key eksik.")
                    return False
                success = self.automator.press_key(key, label=str(label or key))

            elif action == "katman_secimi":
                if not context_target:
                    self.log("Sekans hatasi: katman_secimi icin context_target eksik.")
                    return False
                katman_label = context_target.get("katman_id")
                coord = coord_map.get(katman_label)
                if not katman_label or not coord:
                    self.log(f"Sekans hatasi: katman koordinati bulunamadi ({katman_label}).")
                    return False
                success = self.automator.click(katman_label, coord, seal_label="seq_katman_secimi")

            elif action == "boss_secimi":
                if not context_target:
                    self.log("Sekans hatasi: boss_secimi icin context_target eksik.")
                    return False
                boss_coord_label = context_target.get("koordinat_ref")
                coord = None
                if isinstance(boss_coord_label, str):
                    coord = coord_map.get(boss_coord_label) or context_target.get("koordinat")
                elif isinstance(boss_coord_label, dict):
                    coord = boss_coord_label
                else:
                    coord = context_target.get("koordinat")
                if not coord:
                    self.log(f"Sekans hatasi: boss koordinati bulunamadi ({boss_coord_label}).")
                    return False
                click_label = boss_coord_label if isinstance(boss_coord_label, str) else context_target.get("aciklama", "boss_coord")
                success = self.automator.click(click_label, coord, seal_label="seq_boss_secimi")

            else:
                self.log(f"Sekans hatasi: bilinmeyen aksiyon '{action}'.")
                return False

            self.log_training_action(
                "sequence_step",
                {"index": idx, "action": action, "label": str(label or ""), "wait_ms": wait_ms, "success": bool(success)},
            )
            if not success:
                self.log(f"Sekans adimi basarisiz: '{action}' -> '{label or ''}'.")
                return False

            if wait_ms > 0 and not self._interruptible_wait(wait_ms / 1000.0):
                return False

        return True

    def auto_start_recording(self, trigger_type: str, timeout_sec: float = 120.0) -> bool:
        """Sadece sistem (boss/event manager) tarafından tetiklenen saf otomatik kayıt."""
        if not hasattr(self, "seq_recorder") or self.seq_recorder is None:
            return False

        # Zaten bir kayıt aktifse, üst üste başlatmayı engelle
        if getattr(self, "_auto_log_active", False):
            return False

        self.log(f"[AUTO-LOG] Kayıt başlatılıyor. Tetik: {trigger_type}", level="DEBUG")
        self._auto_log_active = True
        self._auto_log_trigger = trigger_type

        self.seq_recorder.start(trigger_type=trigger_type)

        # Güvenlik zamanlayıcısı (Kayıt sonsuza kadar açık kalmasın diye)
        if hasattr(self, "_auto_log_timer") and self._auto_log_timer:
            self._auto_log_timer.cancel()
            
        self._auto_log_timer = threading.Timer(timeout_sec, self.auto_stop_recording)
        self._auto_log_timer.start()
        
        return True

    def auto_stop_recording(self):
        """Sistemin kaydı güvenli bir şekilde kapatmasını sağlar."""
        if getattr(self, "_auto_log_active", False):
            self.log("[AUTO-LOG] Kayıt durduruluyor.", level="DEBUG")
            self._auto_log_active = False
            
            if hasattr(self, "_auto_log_timer") and self._auto_log_timer:
                self._auto_log_timer.cancel()
                self._auto_log_timer = None
                
            if hasattr(self, "seq_recorder") and self.seq_recorder:
                self.seq_recorder.stop()

    def restart_game(self):
        if getattr(self, "_is_restarting", False):
            return False

        self._is_restarting = True
        
        # --- GUI DURUM GÜNCELLEMESİ KISMI EKLENDİ ---
        self.gui_queue.put(("status_line1", ("Oyun Yeniden Başlatılıyor...", "red")))
        self.gui_queue.put(("status_line2", ("Lütfen bekleyin...", "gray")))

        self.log("Restart proseduru baslatildi.")
        self.log_training_outcome("restart", {"status": "started"})

        with self.action_lock:
            self.running.clear()
            try:
                # os.system YERİNE subprocess KULLANILDI (Bloklamayı Engellemek İçin)
                subprocess.run(["taskkill", "/f", "/im", self.process_name], 
                               capture_output=True, 
                               creationflags=subprocess.CREATE_NO_WINDOW)
                
                kill_wait = float(self.settings.get("RESTART_AFTER_KILL_WAIT_SN", 1.0))
                time.sleep(max(0.2, kill_wait))

                if not os.path.exists(self.game_path):
                    self.log("HATA: game_path bulunamadi.")
                    return False

                os.startfile(self.game_path)
                # Karakter/sunucu seçim ekranının gelmesi için kısa bekleme
                pre_wait = float(self.settings.get("RESTART_PRE_SERVER_SEARCH_WAIT_SN", 6.0))
                self.log(f"Oyun baslatildi, {int(pre_wait)} sn sunucu ekrani bekleniyor...")
                self.gui_queue.put(("status_line1", (f"Sunucu ekrani bekleniyor ({int(pre_wait)} sn)...", "orange")))
                for _ in range(int(pre_wait)):
                    time.sleep(1.0)

                # Sunucu/karakter seç, harita yükle, etkinlik oku kapat, Z tuşu
                self.log("Login proseduru baslatiliyor...")
                self.gui_queue.put(("status_line1", ("Giris yapiliyor...", "yellow")))
                if hasattr(self, "automator"):
                    self.automator.login_to_game()

                self.running.set()
                self.gui_queue.put(("status_line1", ("Hazır", "green")))
                self.gui_queue.put(("status_line2", ("", "white")))
                self.log("Restart sonrasi running=True yapildi.")
                self.log_training_outcome("restart", {"status": "success"})
                return True
            except Exception as e:
                self.log(f"Restart sirasinda hata: {e}")
                self.log_training_outcome("restart", {"status": "failed", "error": str(e)})
                return False
            finally:
                self._is_restarting = False

    def _on_close(self):
        """Pencere kapatılırken temiz kapatma: seq_recorder ve diğer kaynakları serbest bırakır."""
        self.log("Uygulama kapatiliyor...")
        try:
            if hasattr(self, "seq_recorder"):
                self.seq_recorder.shutdown()
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass

    def run(self):
        self.log("Arayuz yukleniyor...")
        self.gui_manager.create_gui()
        self.root.mainloop()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    root = tk.Tk()
    try:
        app = LoABot(root)
        if app.initialized:
            root.protocol("WM_DELETE_WINDOW", app._on_close)
            app.run()
        else:
            print("Bot baslatilamadi. Config dosyasi kontrol ediniz.")
    except Exception as e:
        print(f"Kritik Hata: {e}")