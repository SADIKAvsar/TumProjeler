import os
import subprocess
import time

import mss
import numpy as np
import psutil

from utils import BOT_IS_CLICKING_EVENT


class GameManager:
    def __init__(self, bot):
        self.bot = bot
        self._last_probe = None
        self._freeze_count = 0
        self._last_check_time = 0.0
        self._recovery_attempts = 0
        self._last_recovery_time = 0.0
        self._last_system_capture_time = 0.0

    def is_running(self) -> bool:
        target_name = str(self.bot.process_name or "").lower()
        if not target_name:
            return False

        for proc in psutil.process_iter(["name"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                if name == target_name:
                    return True
            except Exception:
                continue
        return False

    def launch(self) -> bool:
        try:
            game_dir = os.path.dirname(self.bot.game_path)
            subprocess.Popen(self.bot.game_path, cwd=game_dir, creationflags=subprocess.DETACHED_PROCESS)
            return True
        except Exception:
            return False

    def terminate(self):
        target_name = str(self.bot.process_name or "").lower()
        for proc in psutil.process_iter(["name"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                if name == target_name:
                    proc.kill()
            except Exception:
                continue

        # YAMA-7: RL ödül sinyali — oyun yeniden başlatıldı (-0.5)
        if hasattr(self.bot, "reward_engine"):
            self.bot.reward_engine.on_restart(reason="game_terminate")

        time.sleep(2)

    def _capture_probe(self, region: dict):
        monitor = {
            "top": int(region.get("y", 0)),
            "left": int(region.get("x", 0)),
            "width": int(region.get("w", 300)),
            "height": int(region.get("h", 200)),
        }
        if monitor["width"] <= 0 or monitor["height"] <= 0:
            return None

        try:
            with mss.mss() as sct:
                img = np.array(sct.grab(monitor))
        except Exception:
            return None

        # BGRA -> Gray
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.uint8)
        else:
            gray = img.astype(np.uint8)

        # Kucuk boyut: hizli ve stabil fark hesabi
        h, w = gray.shape[:2]
        target_h, target_w = 72, 128
        y_idx = np.linspace(0, h - 1, target_h).astype(int)
        x_idx = np.linspace(0, w - 1, target_w).astype(int)
        return gray[np.ix_(y_idx, x_idx)]

    def monitor_freeze(self) -> bool:
        interval = float(self.bot.settings.get("FREEZE_CHECK_INTERVAL_SN", 15))
        max_freeze_duration = float(self.bot.settings.get("MAX_FREEZE_DURATION_SN", 60))
        motion_eps = float(self.bot.settings.get("FREEZE_MOTION_EPSILON", 1.2))

        now = time.monotonic()
        if now - self._last_check_time < interval:
            return False
        # Kritik: interval bazli sayim icin her check'te guncelle.
        self._last_check_time = now

        if BOT_IS_CLICKING_EVENT.is_set():
            self._freeze_count = 0
            self._last_probe = None
            return False

        region = self.bot.ui_regions.get("freeze_check_area", {"x": 1077, "y": 1033, "w": 309, "h": 904})
        probe = self._capture_probe(region)
        if probe is None:
            return False

        if self._last_probe is None:
            self._last_probe = probe
            self._freeze_count = 0
            return False

        # Ortalama piksel farki (abs diff)
        diff = float(np.mean(np.abs(probe.astype(np.int16) - self._last_probe.astype(np.int16))))
        self._last_probe = probe

        if diff < motion_eps:
            self._freeze_count += 1
        else:
            self._freeze_count = 0
            self._recovery_attempts = 0
            return False

        freeze_seconds = self._freeze_count * interval
        self.bot.log(
            f"Freeze check: diff={diff:.3f}, static={self._freeze_count}, sure={int(freeze_seconds)}s",
            level="DEBUG",
        )
        if freeze_seconds >= max_freeze_duration:
            self._capture_system_recovery_snapshot(
                stage="freeze_check_triggered",
                reason=f"diff={diff:.3f}, static={self._freeze_count}, sure={int(freeze_seconds)}s",
                min_interval_sn=5.0,
            )
            return True
        return False

    def _capture_system_recovery_snapshot(self, stage: str, reason: str = "", min_interval_sn: float = 1.0):
        """No-op: Shadow mode kaldırıldı. Video recorder otomatik olarak kaydeder."""
        pass

    def _handle_freeze_with_ai(self) -> bool:
        """
        AI ile donma teshisi yap.
        Donus:
        - True: sorun cozuldu / bekleme karari verildi
        - False: restart veya fallback gerekli
        """
        ai_engine = getattr(self.bot, "brain", None)
        ai_engine = getattr(ai_engine, "ai_engine", None) if ai_engine else None
        if not ai_engine:
            return False

        duration = self._freeze_count * float(self.bot.settings.get("FREEZE_CHECK_INTERVAL_SN", 15))
        diagnosis = ai_engine.diagnose_freeze(
            duration=duration,
            last_action=getattr(self.bot, "_last_action", "unknown"),
            expected_result=getattr(self.bot, "_expected_result", "unknown"),
        )
        if not diagnosis:
            return False

        action = diagnosis.get("recommended_action", "restart")
        confidence = float(diagnosis.get("confidence", 0))
        freeze_type = diagnosis.get("freeze_type", "unknown")
        self.bot.log(f"AI Freeze Teshis: {freeze_type} -> {action} (guven: {confidence:.0%})")

        if action == "wait":
            wait_time = float(diagnosis.get("wait_time_seconds", 15))
            self.bot.log(f"AI: {int(wait_time)}s bekleniyor.")
            self._freeze_count = 0
            return True

        if action == "close_popup":
            if hasattr(self.bot, "popup") and hasattr(self.bot.popup, "close_all_popups"):
                closed = self.bot.popup.close_all_popups()
                self.bot.log(f"AI: Popup kapatma denendi. closed={closed}", level="DEBUG")
            self._freeze_count = 0
            return True

        if action == "teleport":
            if hasattr(self.bot, "automator"):
                self.bot.automator.return_to_exp_farm(force_restart_if_failed=False)
            self._freeze_count = 0
            return True

        # reconnect/restart veya bilinmeyen: fallback/restart akisi
        return False

    def _safe_freeze_fallback(self) -> bool:
        """
        AI karar veremezse bir kez guvenli toparlama dene.
        Basariliysa True (restart atmadan devam), degilse False.
        """
        cooldown = float(self.bot.settings.get("FREEZE_SAFE_RECOVERY_COOLDOWN_SN", 45))
        max_attempts = int(self.bot.settings.get("FREEZE_MAX_SAFE_RECOVERY_ATTEMPTS", 2))
        now = time.monotonic()

        if self._recovery_attempts >= max_attempts and (now - self._last_recovery_time) < cooldown:
            return False

        self._recovery_attempts += 1
        self._last_recovery_time = now
        self.bot.log(f"Donma fallback denemesi ({self._recovery_attempts}/{max_attempts})", level="WARNING")

        # 1) Popup temizligi
        if hasattr(self.bot, "popup") and hasattr(self.bot.popup, "close_all_popups"):
            try:
                self.bot.popup.close_all_popups()
            except Exception:
                pass

        # 2) Oto savas tekrar
        try:
            if hasattr(self.bot, "automator"):
                self.bot.automator.press_key("z", label="freeze_recovery_auto_mode")
        except Exception:
            pass

        # 3) Kisa bekle ve sayaci sifirla (hemen restart atma)
        self._freeze_count = 0
        return True

    def watchdog_thread(self):
        self.bot.log("GameManager: Watchdog aktif.")
        while True:
            if self.bot.running.is_set():
                if not self.is_running():
                    self._capture_system_recovery_snapshot(
                        stage="process_not_running",
                        reason="watchdog_process_missing",
                        min_interval_sn=5.0,
                    )
                    self.bot.log("Oyun kapanmis, restart atiliyor...")
                    
                    # --- EKLENEN KISIM 1: OYUN ÇÖKTÜĞÜNDE ---
                    self.bot.gui_queue.put(("status_line1", ("Oyun çöktü, yeniden başlatılıyor...", "red")))
                    self.bot.gui_queue.put(("status_line2", ("Lütfen bekleyin...", "gray")))

                    # Reward Engine: Oyun çökmesi = negatif sinyal (-0.5)
                    if hasattr(self.bot, "reward_engine"):
                        self.bot.reward_engine.on_restart(reason="watchdog_process_crash")
                    
                    self.bot.restart_game()
                    self._freeze_count = 0
                    self._last_probe = None
                elif self.monitor_freeze():
                    # 1) AI teshis
                    handled = self._handle_freeze_with_ai()
                    # 2) AI yoksa/kararsizsa fallback
                    if not handled:
                        handled = self._safe_freeze_fallback()
                    # 3) Hala cozulmediyse restart
                    if not handled:
                        self._capture_system_recovery_snapshot(
                            stage="freeze_restart",
                            reason="freeze_unresolved_restart",
                            min_interval_sn=3.0,
                        )
                        self.bot.log("Donma tespit edildi, restart atiliyor...")
                        
                        # --- EKLENEN KISIM 2: OYUN DONDUĞUNDA ---
                        self.bot.gui_queue.put(("status_line1", ("Donma tespit edildi, başlatılıyor...", "red")))
                        self.bot.gui_queue.put(("status_line2", ("Lütfen bekleyin...", "gray")))

                        # Reward Engine: Çözülemeyen freeze = negatif sinyal (-0.5)
                        if hasattr(self.bot, "reward_engine"):
                            self.bot.reward_engine.on_restart(reason="watchdog_freeze_unresolved")
                        
                        self.bot.restart_game()
                        self._freeze_count = 0
                        self._last_probe = None
            time.sleep(1.5)
