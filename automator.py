import time

import pyautogui

from utils import BOT_IS_CLICKING_EVENT


class Automator:
    def __init__(self, bot):
        self.bot = bot

    def _capture_action_decision(self, action_name: str, payload: dict, stage: str) -> str:
        if not hasattr(self.bot, "capture_local_decision_frame"):
            return ""
        try:
            return self.bot.capture_local_decision_frame(
                action_name=action_name,
                payload=dict(payload or {}),
                phase="ACTION_PHASE",
                stage=stage,
            )
        except Exception:
            return ""

    def click(self, label, coord=None, coord_override=None, duration=0.2, seal_label: str = "mouse_click"):
        """Klik islemi. coord_override eski kodlarla uyumluluk icindir.
        seal_label: SequentialRecorder'a gönderilecek aksiyon etiketi.
        """
        target = coord_override if coord_override is not None else coord
        decision_payload = {
            "label": str(label or ""),
            "x": int((target or {}).get("x", 0)) if isinstance(target, dict) else 0,
            "y": int((target or {}).get("y", 0)) if isinstance(target, dict) else 0,
            "duration": float(duration),
        }
        decision_id = self._capture_action_decision("click", decision_payload, stage="click_decision")

        if not target:
            self.bot.log_training_action(
                "click",
                {
                    "label": str(label or ""),
                    "success": False,
                    "reason": "missing_target",
                    "decision_id": decision_id,
                },
            )
            return False

        try:
            BOT_IS_CLICKING_EVENT.set()
            pyautogui.click(target["x"], target["y"], duration=duration)
            self.bot.log_training_action(
                "click",
                {
                    "label": str(label or ""),
                    "x": int(target.get("x", 0)),
                    "y": int(target.get("y", 0)),
                    "duration": float(duration),
                    "success": True,
                    "decision_id": decision_id,
                },
            )
            # ClickKnowledgeBase: başarılı AI tıklamalarını öğren
            # (sadece AI kaynaklı klikler; kural tabanlı sekans klikleri hariç)
            if seal_label == "mouse_click":
                self._record_click_to_kb(target)
            # Konum proxy'si: tıklanan koordinatı "son bilinen konum" olarak güncelle
            if isinstance(target, dict):
                self._update_seq_position(target.get("x", 0), target.get("y", 0))
            # 10 FPS sekans mühürleme
            self._seal_seq_action(seal_label)
            return True
        except Exception as exc:
            self.bot.log_training_action(
                "click",
                {
                    "label": str(label or ""),
                    "x": int(target.get("x", 0)),
                    "y": int(target.get("y", 0)),
                    "duration": float(duration),
                    "success": False,
                    "error": str(exc),
                    "decision_id": decision_id,
                },
            )
            return False
        finally:
            time.sleep(0.1)
            BOT_IS_CLICKING_EVENT.clear()

    def press_key(self, key, label=None):
        decision_payload = {"key": str(key), "label": str(label or "")}
        decision_id = self._capture_action_decision("press_key", decision_payload, stage="key_decision")
        pyautogui.press(key)
        self.bot.log_training_action(
            "press_key",
            {"key": str(key), "label": str(label or ""), "success": True, "decision_id": decision_id},
        )
        # 10 FPS sekans mühürleme
        self._seal_seq_action(f"key_{key}")
        return True

    def _record_click_to_kb(self, coord: dict) -> None:
        """Başarılı AI tıklamasını ClickKnowledgeBase'e bildir (öğrenme)."""
        try:
            tb = getattr(self.bot, "tactical_brain", None)
            if tb is None:
                return
            kb = getattr(tb, "_click_kb", None)
            if kb is None:
                return
            phase = getattr(self.bot, "_global_phase", "UNKNOWN")
            kb.record_click(phase=phase, coord=coord, action="mouse_click", source="bot_observed")
        except Exception:
            pass

    def _seal_seq_action(self, action_label: str):
        """SequentialRecorder'a aksiyon bildirimi gönderir (hata durumunda sessizce geçer)."""
        try:
            if hasattr(self.bot, "seq_recorder"):
                self.bot.seq_recorder.seal_action(action_label)
        except Exception:
            pass

    def _update_seq_position(self, x: float, y: float):
        """SequentialRecorder'a konum proxy'si gönderir (stuck tespiti için)."""
        try:
            if hasattr(self.bot, "seq_recorder"):
                self.bot.seq_recorder.update_position(float(x), float(y))
        except Exception:
            pass

    def login_to_game(self):
        """Restart sonrası tam giriş sekansı:
        1. server_s575.png bul ve tıkla
        2. Harita yüklenme süresi kadar bekle (GAME_UI_READY_DELAY_SN)
        3. Etkinlik okunu kapat (etkinlik_oku_close)
        4. Z tuşu ile otomatik savaş modunu aç
        """
        self.bot.log("Automator: S575 sunucusu araniyor...")
        region_full = self.bot.ui_regions.get("region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440})

        for attempt in range(5):
            match = self.bot.vision.find("server_s575.png", region_full, 0.8)
            if match:
                self.click("server_selection", {"x": match[0], "y": match[1]})

                # Harita yüklenmesi için bekle
                map_wait = float(self.bot.settings.get("GAME_UI_READY_DELAY_SN", 20.0))
                self.bot.log(f"Sunucuya baglandi, harita yukleniyor ({int(map_wait)} sn)...")
                time.sleep(map_wait)

                # Etkinlik okunu kapat — anchor image ile önce varlık kontrol et
                etkinlik_coord = self.bot.coordinates.get("etkinlik_oku_close")
                if etkinlik_coord:
                    region_menu = self.bot.ui_regions.get(
                        "region_ust_menu",
                        self.bot.ui_regions.get("region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440}),
                    )
                    anchor = etkinlik_coord.get("anchor", {})
                    anchor_img = anchor.get("image_file", "etkinlik_oku_close.png")
                    anchor_conf = float(anchor.get("confidence", 0.8))
                    # Görünür olana kadar max 5sn bekle
                    arrow_visible = False
                    for _ in range(5):
                        if self.bot.vision.find(anchor_img, region_menu, anchor_conf):
                            arrow_visible = True
                            break
                        time.sleep(1.0)
                    if arrow_visible:
                        self.bot.log("Etkinlik oku kapatiliyor...")
                        self.click("etkinlik_oku_close", {"x": etkinlik_coord["x"], "y": etkinlik_coord["y"]})
                        time.sleep(1.0)
                    else:
                        self.bot.log("Etkinlik oku gorulmedi, atlanıyor.")

                # Otomatik savaş modunu aç
                self.press_key("z", label="login_auto_mode")
                self.bot.log("Login sekans tamamlandi.")
                return True

            self.bot.log(f"Sunucu bekleniyor... ({attempt + 1}/5)")
            time.sleep(3)

        self.bot.log("HATA: server_s575.png bulunamadi.")
        return False

    def return_to_exp_farm(self, force_restart_if_failed=True):
        """Kullanici talebi: ek exp_farm_sequence olmadan EXP_FARM kabul et."""
        if self.bot.location_manager.get_region_name() == "EXP_FARM":
            # EKLENEN KISIM: Zaten ana haritadaysak da kaydı durdur
            if hasattr(self.bot, "auto_stop_recording"):
                self.bot.auto_stop_recording()
            return True

        self.bot.log("Automator: EXP_FARM'a donus baslatildi.")

        if not self.safe_exit_map():
            if force_restart_if_failed:
                self.bot.log("Automator: Haritadan cikis basarisiz, restart denenecek.")
                return self.bot.restart_game()
            return False

        self.bot.location_manager.set_current_location_by_name("EXP_FARM")
        self.bot.stop_global_mission(reason="exp_farm_return")
        
        # EKLENEN KISIM: Başarılı bir dönüş işlemi yapıldığı anda kaydı kapat
        if hasattr(self.bot, "auto_stop_recording"):
            self.bot.auto_stop_recording()
            
        return True

    def safe_exit_map(self):
        exit_reg = self.bot.ui_regions.get("exit_map_area", {"x": 2000, "y": 0, "w": 560, "h": 300})
        match = self.bot.vision.find("exit_map.png", exit_reg, 0.75)

        if match:
            self.click("exit_map_btn", {"x": match[0], "y": match[1]})
            self.bot._interruptible_wait(3)
            return True

        boss_list_btn = self.bot.coordinates.get("boss_list_ac")
        if boss_list_btn:
            self.click("boss_list_force", boss_list_btn)
            self.bot._interruptible_wait(1)
            self.press_key("v", label="force_close_menu")
            return True

        return False
