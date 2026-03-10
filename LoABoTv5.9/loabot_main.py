# -*- coding: utf-8 -*-
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
from video_recorder import VideoRecorder
from user_input_monitor import UserInputMonitor
from reward_engine import RewardEngine


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
        self.video_recorder = VideoRecorder(self)

        self.game = GameManager(self)
        self.combat = CombatManager(self)
        self.location_manager = LocationManager(self, self.general_cfg)
        self.automator = Automator(self)
        self.event_manager = EventManager(self, self.timed_events_cfg)
        self.boss_manager = BossManager(self)
        self.gui_manager = GUIManager(self)
        self.pvp_manager = PvPManager(self)
        self.popup_manager = PopupManager(self)
        self.user_monitor = UserInputMonitor(self)
        self.reward_engine = RewardEngine(self)

        # VideoRecorder'ı UserInputMonitor'a bağla (init sırası gereği)
        if hasattr(self, "video_recorder") and hasattr(self, "user_monitor"):
            self.video_recorder.late_subscribe()

        # v6.0 core entegrasyonu: GameState + EventBus + BotBridge
        try:
            from core.bootstrap import bootstrap_v6
            self.game_state, self.event_bus, self.bridge = bootstrap_v6(self)
        except ImportError:
            self.game_state = None
            self.event_bus = None
            self.bridge = None
            self.log("v6.0 core/ bulunamadi, v5.9 modunda devam ediliyor.", level="WARNING")

        # 2) Threadleri baslat
        self._start_threads()

        self._initialize_boss_data()

        self.initialized = True
        self.log("LoABot v6.0: v5.9 stabil runtime + GameState/EventBus/Bridge aktif")

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
        # Re-entrant lock: nested mission/state updates in same thread must not deadlock.
        self.action_lock = threading.RLock()
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
        """
        Mission state write with lock-timeout guard.
        Never block forever on action_lock; this prevents total bot freeze.
        """
        lock_timeout = float(self.settings.get("MISSION_PHASE_LOCK_TIMEOUT_SN", 2.0))
        acquired = False
        try:
            acquired = self.action_lock.acquire(timeout=max(0.1, lock_timeout))
        except Exception:
            acquired = False

        if not acquired:
            self.log(
                f"[LOCK_GUARD] set_global_mission_phase lock timeout ({lock_timeout:.1f}s). "
                "State unlocked olarak guncellendi.",
                level="WARNING",
            )

        try:
            self._global_phase = phase
            self._global_stage = stage
            self._global_mission_reason = reason
            self._global_mission_extra = dict(extra or {})
        finally:
            if acquired:
                try:
                    self.action_lock.release()
                except Exception:
                    pass

        self.log_training_state(
            "mission_phase_change",
            {
                "new_phase": phase,
                "new_stage": stage,
                "reason": reason,
                **self._global_mission_extra,
            },
        )

    def start_global_mission(self, phase: str, stage: str, reason: str = "", extra: dict = None):
        self.set_global_mission_phase(phase, stage, reason, extra)
        if hasattr(self, "training_logger"):
            self.training_logger.start_episode(episode_type=reason, context=extra)
        if hasattr(self, "reward_engine"):
            self.reward_engine.reset_episode()

    def stop_global_mission(self, reason: str = ""):
        self.set_global_mission_phase("IDLE_PHASE", "waiting", reason, {})
        if hasattr(self, "training_logger"):
            reward_summary = {}
            if hasattr(self, "reward_engine"):
                reward_summary = self.reward_engine.get_episode_summary()
                self.reward_engine.reset_episode()
            self.training_logger.end_episode(status="completed", reason=reason, metrics=reward_summary)

    def _seal_visual_event(self, stage: str, extra: dict = None) -> None:
        """
        Görsel tespit anını video kaydına bildirir.
        area_check / spawn_check / victory gibi vision.find() başarılarında çağrılır.
        Hata durumunda sessizce geçer, main thread bloklanmaz.
        """
        vid = getattr(self, "video_recorder", None)
        if vid is not None and vid.is_recording:
            try:
                vid.log_action(
                    event_type="visual_detect",
                    data=extra or {},
                    source="bot",
                    action_label=f"visual_{stage}",
                )
            except Exception:
                pass

    def capture_local_decision_frame(
        self, action_name: str, payload: dict = None, phase: str = None, stage: str = None
    ) -> str:
        """AI karar anı karesi — shadow mode kaldırıldı, stub olarak kalıyor."""
        # Eski shadow mode kaldırıldı. Bu metod geriye uyumluluk için
        # boş string döner. Gelecekte video frame referansı eklenebilir.
        return ""

    def toggle_recording(self):
<<<<<<< HEAD
        """GUI REC butonu: manuel video kaydini baslatir/durdurur."""
        vid = getattr(self, "video_recorder", None)
        if vid is None:
            self.log("VideoRecorder bulunamadi. Manuel kayit degistirilemedi.", level="WARNING")
            self._manual_recording = False
            return

        # Otomatik kayit aktifken buton "durdur" olarak calissin.
        if getattr(self, "_auto_log_active", False):
            self.log("Manuel komutla otomatik kayit sonlandiriliyor.", level="INFO")
            self.auto_stop_recording()
            self._manual_recording = False
            return

        # Manuel kayit aciksa: durdur
        if getattr(self, "_manual_recording", False):
            if vid.is_recording:
                if vid.stop(success=True, reason="manual_button_stop"):
                    self.log("Manuel kayit durduruldu.", level="INFO")
            else:
                self.log(
                    "Manuel kayit bayragi acikti ama recorder kayitta degildi; durum senkronize edildi.",
                    level="WARNING",
                )
            self._manual_recording = False
            return

        # Bayraklar bozuk ama recorder kayittaysa guvenli senkronizasyon
        if vid.is_recording:
            self.log(
                "Kayit zaten aktifti (bayrak uyumsuzlugu). Buton kaydi guvenli sekilde kapatti.",
                level="WARNING",
            )
            vid.stop(success=True, reason="manual_button_stop_resync")
            self._manual_recording = False
            self._auto_log_active = False
            self._auto_log_trigger = ""
            if hasattr(self, "_auto_log_timer") and self._auto_log_timer:
                self._auto_log_timer.cancel()
                self._auto_log_timer = None
            return

        # Manuel kayit kapaliyken: baslat
        if vid.start(trigger_type="manual_button"):
            self._manual_recording = True
            self.log("Manuel kayit baslatildi.", level="INFO")
        else:
            self._manual_recording = False
            self.log(
                "Manuel kayit baslatilamadi (VideoRecorder devre disi veya hazir degil).",
                level="WARNING",
            )
=======
        """
        GUI kayit butonu ile manuel video kaydini baslat/durdur.

        Not:
        - Manuel kayit acikken otomatik kayit tetikleri dikkate alinmaz.
        - Oto kayit aktifken manuel baslatma engellenir.
        """
        vid = getattr(self, "video_recorder", None)
        if vid is None:
            self.log("Manuel kayit: VideoRecorder hazir degil.", level="WARNING")
            return

        if getattr(self, "_auto_log_active", False):
            self.log("Manuel kayit acilamadi: otomatik kayit aktif.", level="WARNING")
            return

        if getattr(self, "_manual_recording", False):
            stopped = bool(vid.stop(success=True, reason="manual_stop"))
            self._manual_recording = False
            if stopped:
                self.log("Manuel kayit durduruldu.")
            else:
                self.log("Manuel kayit durdurulamadi.", level="WARNING")
            return

        started = bool(vid.start(trigger_type="manual"))
        if started:
            self._manual_recording = True
            self.log("Manuel kayit baslatildi.")
        else:
            self.log("Manuel kayit baslatilamadi.", level="WARNING")
>>>>>>> ceff47febb26aba071d9eff02118217ce2c86780

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

    def is_action_lock_busy(self) -> bool:
        """
        RLock/Lock uyumlu "kilit mesgul mu?" kontrolu.
        """
        lock_obj = getattr(self, "action_lock", None)
        if lock_obj is None:
            return False

        locked_fn = getattr(lock_obj, "locked", None)
        if callable(locked_fn):
            try:
                return bool(locked_fn())
            except Exception:
                pass

        # RLock fallback: non-blocking acquire deneyerek mesguliyet tespiti.
        try:
            acquired = lock_obj.acquire(blocking=False)
        except TypeError:
            acquired = lock_obj.acquire(False)
        except Exception:
            return False

        if acquired:
            try:
                lock_obj.release()
            except Exception:
                pass
            return False
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
        if not hasattr(self, "automator") or self.automator is None:
            self.log("Sekans hatasi: Automator hazir degil.")
            return False
        return self.automator.execute_sequence(
            sequence=sequence,
            coord_map=coord_map,
            context_target=context_target,
        )

    def auto_start_recording(self, trigger_type: str, timeout_sec: float = 120.0) -> bool:
        """Sadece sistem (boss/event manager) tarafindan tetiklenen otomatik kayit."""
        # Varsayilan politika: kayit kontrolu kullanici butonunda.
        manual_only = bool(self.settings.get("MANUAL_VIDEO_CONTROL_ONLY", True))
        if manual_only:
            return False

        # Manuel kayit acikken auto kayit acilmaz.
        if getattr(self, "_manual_recording", False):
            return False

        # Zaten bir kayit aktifse, ust uste baslatmayi engelle
        if getattr(self, "_auto_log_active", False):
            return False

<<<<<<< HEAD
        vid = getattr(self, "video_recorder", None)
        if vid is None:
            self.log("[AUTO-LOG] VideoRecorder bulunamadi, kayit baslatilamadi.", level="WARNING")
            return False

        if getattr(self, "_manual_recording", False):
            self.log("[AUTO-LOG] Manuel kayit aktif oldugu icin otomatik kayit atlandi.", level="DEBUG")
            return False

        if vid.is_recording:
            self.log("[AUTO-LOG] Kayit zaten aktif oldugu icin otomatik kayit atlandi.", level="DEBUG")
            return False

        self.log(f"[AUTO-LOG] Kayıt başlatılıyor. Tetik: {trigger_type}", level="DEBUG")

        # Video kaydını başlat
        if not vid.start(trigger_type=trigger_type):
            self.log("[AUTO-LOG] VideoRecorder baslatma basarisiz, kayit acilmadi.", level="WARNING")
            return False

        self._manual_recording = False
        self._auto_log_active = True
        self._auto_log_trigger = trigger_type
=======
        self.log(f"[AUTO-LOG] Kayit baslatiliyor. Tetik: {trigger_type}", level="DEBUG")
        self._auto_log_active = True
        self._auto_log_trigger = trigger_type

        # Video kaydini baslat
        vid = getattr(self, "video_recorder", None)
        if vid is not None:
            vid.start(trigger_type=trigger_type)
>>>>>>> ceff47febb26aba071d9eff02118217ce2c86780

        # Guvenlik zamanlayicisi (kayit sonsuza kadar acik kalmasin)
        if hasattr(self, "_auto_log_timer") and self._auto_log_timer:
            self._auto_log_timer.cancel()

        self._auto_log_timer = threading.Timer(timeout_sec, self.auto_stop_recording)
        self._auto_log_timer.start()

        return True

    def auto_stop_recording(self):
        """Sistemin kaydı güvenli bir şekilde kapatmasını sağlar."""
        if getattr(self, "_auto_log_active", False):
            trigger = str(getattr(self, "_auto_log_trigger", "")).strip().lower()
            self.log("[AUTO-LOG] Kayıt durduruluyor.", level="DEBUG")
            self._auto_log_active = False

            if hasattr(self, "_auto_log_timer") and self._auto_log_timer:
                self._auto_log_timer.cancel()
                self._auto_log_timer = None

            # Video kaydını durdur
            vid = getattr(self, "video_recorder", None)
            if vid is not None and vid.is_recording:
                vid.stop(success=True, reason=f"auto_stop_{trigger}")

            self._auto_log_trigger = ""

            if trigger == "boss_attack":
                boss_adi = getattr(self, "attacking_target_aciklama", None)
                if boss_adi:
                    # If a boss flow timed out/stuck, push spawn forward to avoid 00:00:00 stale timers.
                    try:
                        boss_obj = self.bosslar.get(str(boss_adi)) if hasattr(self, "bosslar") else None
                        if boss_obj is not None and hasattr(self, "combat"):
                            self.combat.recalculate_times_interrupted(boss_obj)
                    except Exception as exc:
                        self.log(f"AUTO-LOG fallback spawn guncelleme hatasi: {exc}", level="WARNING")

                    self.log(
                        "AUTO-LOG: boss_attack kaydi kapanirken saldiri hedefi temizlendi.",
                        level="DEBUG",
                    )
                    self.attacking_target_aciklama = None

                # Safety: mission phase must not stay in NAV/COMBAT/LOOT after forced auto-stop.
                current_phase = str(getattr(self, "_global_phase", "")).strip().upper()
                if current_phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}:
                    self.stop_global_mission(reason="auto_stop_boss_attack")

    def force_cleanup_combat_state(self, reason: str = "event_interrupt") -> None:
        """
        Aktif saldırı/combat durumunu temiz olarak sonlandırır.
        EventManager etkinlik girişinden önce çağırır; boss akışı kesintiye uğramışsa
        saldırı flagleri sıfırlanır ve spawn süresi fallback ile güncellenir.
        """
        self.log(f"[FORCE_CLEANUP] Combat durumu temizleniyor. Sebep: {reason}")

        # Kesilen boss'u bul (recalculate için)
        interrupted_boss = None
        boss_adi = getattr(self, "attacking_target_aciklama", None)
        if boss_adi and hasattr(self, "bosslar"):
            interrupted_boss = self.bosslar.get(str(boss_adi))

        # Spawn zamanı fallback: boss öldürülemediyse bir sonraki periyot hesapla
        if interrupted_boss is not None and hasattr(self, "combat"):
            try:
                self.combat.recalculate_times_interrupted(interrupted_boss)
            except Exception as exc:
                self.log(
                    f"[FORCE_CLEANUP] recalculate_times_interrupted hatasi: {exc}",
                    level="WARNING",
                )

        # Saldırı flaglerini sıfırla
        self.attacking_target_aciklama = None

        # Global phase'i IDLE'a sıfırla (sadece aktif combat aşamalarında)
        current_phase = str(getattr(self, "_global_phase", "")).strip().upper()
        if current_phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}:
            self.stop_global_mission(reason=f"force_cleanup:{reason}")

        # Manuel kayit aciksa guvenli kapat
        if getattr(self, "_manual_recording", False):
            vid = getattr(self, "video_recorder", None)
            if vid is not None and vid.is_recording:
                try:
                    vid.stop(success=False, reason=f"manual_stop_{reason}")
                except Exception:
                    pass
            self._manual_recording = False
        # Guvenlik: devam eden boss kaydini kapat
        self.auto_stop_recording()

        self.log(f"[FORCE_CLEANUP] Temizlik tamamlandi. Boss: {boss_adi or 'yok'}")

    def restart_game(self):
        if getattr(self, "_is_restarting", False):
            return False

        self._is_restarting = True
        lock_acquired = False
        lock_timeout = float(self.settings.get("RESTART_ACTION_LOCK_TIMEOUT_SN", 3.0))
        
        # --- GUI DURUM GUNCELLEMESI ---
        self.gui_queue.put(("status_line1", ("Oyun Yeniden Baslatiliyor...", "red")))
        self.gui_queue.put(("status_line2", ("Lutfen bekleyin...", "gray")))

        self.log("Restart proseduru baslatildi.")
        self.log_training_outcome("restart", {"status": "started"})

        try:
            lock_acquired = self.action_lock.acquire(timeout=max(0.5, lock_timeout))
            if not lock_acquired:
                self.log(
                    "Restart: action_lock zaman asimi. Zorunlu restart moduna geciliyor.",
                    level="WARNING",
                )

            self.running.clear()
            self.auto_stop_recording()
            if getattr(self, "_manual_recording", False):
                vid = getattr(self, "video_recorder", None)
                if vid is not None and vid.is_recording:
                    try:
                        vid.stop(success=False, reason="manual_stop_restart")
                    except Exception:
                        pass
                self._manual_recording = False
            self.attacking_target_aciklama = None
            self.active_event = None

            # os.system yerine subprocess: bloklamayi azalt.
            subprocess.run(
                ["taskkill", "/f", "/im", self.process_name],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            kill_wait = float(self.settings.get("RESTART_AFTER_KILL_WAIT_SN", 1.0))
            time.sleep(max(0.2, kill_wait))

            if not os.path.exists(self.game_path):
                self.log("HATA: game_path bulunamadi.")
                return False

            os.startfile(self.game_path)

            # Karakter/sunucu secim ekraninin gelmesi icin kisa bekleme.
            pre_wait = float(self.settings.get("RESTART_PRE_SERVER_SEARCH_WAIT_SN", 6.0))
            self.log(f"Oyun baslatildi, {int(pre_wait)} sn sunucu ekrani bekleniyor...")
            self.gui_queue.put(("status_line1", (f"Sunucu ekrani bekleniyor ({int(pre_wait)} sn)...", "orange")))
            for _ in range(int(pre_wait)):
                time.sleep(1.0)

            # Sunucu/karakter sec, harita yukle, etkinlik oku kapat, Z tusu.
            self.log("Login proseduru baslatiliyor...")
            self.gui_queue.put(("status_line1", ("Giris yapiliyor...", "yellow")))
            if hasattr(self, "automator"):
                self.automator.login_to_game()

            self.running.set()
            self.gui_queue.put(("status_line1", ("Hazir", "green")))
            self.gui_queue.put(("status_line2", ("", "white")))
            self.log("Restart sonrasi running=True yapildi.")
            self.log_training_outcome("restart", {"status": "success"})
            return True
        except Exception as e:
            self.log(f"Restart sirasinda hata: {e}")
            self.log_training_outcome("restart", {"status": "failed", "error": str(e)})
            return False
        finally:
            if lock_acquired:
                try:
                    self.action_lock.release()
                except Exception:
                    pass
            self._is_restarting = False

    def _on_close(self):
        """Pencere kapatılırken temiz kapatma: tüm kaynakları serbest bırakır."""
        self.log("Uygulama kapatiliyor...")
        try:
            if hasattr(self, "user_monitor"):
                self.user_monitor.shutdown()
        except Exception:
            pass
        try:
            if hasattr(self, "video_recorder"):
                self.video_recorder.shutdown()
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

