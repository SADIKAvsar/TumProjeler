# -*- coding: utf-8 -*-
# LoA Bot v5.9 — GeminiProConfig.YAML hiyerarşik yapısıyla tam uyumlu
# Değişiklikler: anchor/spawn_check/victory per-boss desteği,
#                koordinat_ref dual-format (str | dict),
#                _run_sequence anchor doğrulaması,
#                spawn_check_defaults & victory_check_defaults YAML okuma

import os
import re
import cv2
import mss
import yaml
import json
import time
import psutil
import pyautogui
import numpy as np
import subprocess
import traceback
import threading
import tkinter as tk
from queue import Queue
from datetime import datetime, time as dt_time
from tkinter import messagebox, simpledialog
from pynput import mouse

# =============================================================================
# 0) Genel sabitler / yardımcılar
# =============================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR   = os.path.join(SCRIPT_DIR, "image")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "GeminiProConfig.YAML")
LOG_FILE    = os.path.join(SCRIPT_DIR, "game_monitor_v59.log")
pyautogui.FAILSAFE = False

# Manuel tıklama kilidi ve zamanı
MANUAL_CLICK_LOCK   = threading.Lock()
last_manual_click_time = None
BOT_IS_CLICKING_EVENT  = threading.Event()


def log_to_file(msg: str, log_file: str = LOG_FILE):
    """Mesajı belirtilen log dosyasına zaman damgasıyla yazar."""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{now}] {msg}\n")
    except Exception as e:
        print(f"!!! KRİTİK LOGLAMA HATASI ({log_file}): {e} !!!")
        traceback.print_exc()


def _safe_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def load_image(filename: str):
    """Belirtilen resim dosyasını IMAGE_DIR'dan gri tonlamalı olarak yükler."""
    try:
        path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(path):
            print(f"[UYARI] Resim dosyası bulunamadı: {path}")
            log_to_file(f"Resim dosyası bulunamadı: {path}")
            return None
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[UYARI] Resim dosyası okunamadı/bozuk: {path}")
            log_to_file(f"Resim dosyası okunamadı/bozuk: {path}")
            return None
        return image
    except Exception as e:
        print(f"[HATA] Resim yüklenemedi: {filename} - {e}")
        log_to_file(f"Resim yüklenemedi: {filename} - {e}")
        traceback.print_exc()
        return None


def start_click_listener(bot_instance):
    """
    Global mouse tıklama dinleyicisini başlatır.
    'last_manual_click_time'ı ve 'manual_click_detected' sinyalini ayarlar.
    """
    def on_click(x, y, button, pressed):
        if BOT_IS_CLICKING_EVENT.is_set():
            return
        if pressed:
            with MANUAL_CLICK_LOCK:
                global last_manual_click_time
                last_manual_click_time = time.monotonic()
            if hasattr(bot_instance, 'manual_click_detected'):
                bot_instance.manual_click_detected.set()
    try:
        listener = mouse.Listener(on_click=on_click)
        listener.daemon = True
        listener.start()
        bot_instance.log("Manuel tıklama dinleyici başlatıldı (Sadece algılama, loglama kapalı).")
    except Exception as e:
        bot_instance.log(f"[HATA] Tıklama dinleyici başlatılamadı: {e}")


# =============================================================================
# 1) Config Yükleyici
# =============================================================================

def load_config(filename: str = CONFIG_FILE):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        log_to_file(f"{filename} başarıyla yüklendi.")
        return config_data
    except FileNotFoundError:
        error_msg = f"HATA: {filename} bulunamadı."
        log_to_file(error_msg)
        try:
            messagebox.showerror("Config Hatası", f"{error_msg}\nProgram kapatılacak.")
        except Exception:
            pass
        return None
    except yaml.YAMLError as e:
        error_msg = f"HATA: {filename} dosyası geçersiz YAML formatında: {e}"
        log_to_file(error_msg)
        try:
            messagebox.showerror("Config Hatası", f"{error_msg}\nProgram kapatılacak.")
        except Exception:
            pass
        return None
    except Exception as e:
        error_msg = f"Config yüklenirken beklenmedik hata: {e}"
        log_to_file(error_msg)
        traceback.print_exc()
        try:
            messagebox.showerror("Config Hatası", f"{error_msg}\nProgram kapatılacak.")
        except Exception:
            pass
        return None


# =============================================================================
# 2) LocationManager Sınıfı
# =============================================================================

class LocationManager:
    def __init__(self, bot, general_cfg: dict):
        self.bot         = bot
        self.general_cfg = general_cfg
        self.LOCATION_MAP   = {}
        self.NAME_TO_VALUE  = {}
        self.current_location_value = -1   # UNKNOWN
        self._build_location_maps()
        self.current_location_value = self.NAME_TO_VALUE.get("UNKNOWN", -1)

    # ------------------------------------------------------------------
    def _build_location_maps(self):
        json_locs = self.general_cfg.get("locations", {})
        if not isinstance(json_locs, dict):
            self.bot.log("Config uyarısı: 'module_general.locations' bir sözlük (dict) değil.")
            return
        for name, data in json_locs.items():
            try:
                val_int    = int(data.get("value", -1))
                name_upper = name.upper()
                self.LOCATION_MAP[val_int]      = name_upper
                self.NAME_TO_VALUE[name_upper]  = val_int
            except Exception as e:
                log_to_file(f"Config 'locations' işlenirken hata (konum: {name}): {e}")
        if not self.LOCATION_MAP:
            self.bot.log("UYARI: Hiçbir konum tanımlanamadı.")
            self.LOCATION_MAP[-1]        = "UNKNOWN"
            self.NAME_TO_VALUE["UNKNOWN"] = -1
        self.bot.log(
            f"Konum haritaları oluşturuldu: {len(self.LOCATION_MAP)} adet konum tanımlandı."
        )

    # ------------------------------------------------------------------
    def get_location_name(self, location_value: int) -> str:
        return self.LOCATION_MAP.get(int(location_value), "UNKNOWN")

    def get_region_name(self) -> str:
        return self.get_location_name(self.current_location_value)

    def set_current_location_by_name(self, name: str):
        target_name_upper = name.upper()
        val = self.NAME_TO_VALUE.get(target_name_upper)
        if val is not None:
            old_val = self.current_location_value
            if old_val != val:
                self.current_location_value = val
                self.bot.log(
                    f"KONUM AYARLANDI: '{self.get_location_name(old_val)}' -> '{target_name_upper}'"
                )
        else:
            self.bot.log(
                f"UYARI: set_current_location_by_name('{name}') başarısız. "
                f"Bu isimde bir konum haritada bulunamadı."
            )

    # ------------------------------------------------------------------
    def _get_current_visual_location_value(self) -> int:
        location_definitions = self.general_cfg.get("locations", {})
        if not location_definitions:
            return self.NAME_TO_VALUE.get("UNKNOWN", -1)

        for location_name, data in location_definitions.items():
            anchor_list = []
            single_anchor = data.get("anchor")
            if single_anchor:
                anchor_list.append(single_anchor)
            else:
                multiple_anchors = data.get("anchors")
                if isinstance(multiple_anchors, list):
                    anchor_list.extend(multiple_anchors)
            if not anchor_list:
                continue

            for anchor in anchor_list:
                if not anchor or not isinstance(anchor, dict):
                    continue
                image_file  = anchor.get("image_file")
                confidence  = float(anchor.get("confidence", 0.7))
                region_key  = anchor.get("region_key")
                if not image_file or not region_key:
                    continue
                search_region = self.bot.ui_regions.get(region_key)
                if not search_region:
                    self.bot.log(
                        f"Konum Tespiti Hatası: '{location_name}' için "
                        f"'{region_key}' bölgesi 'ui_regions' içinde bulunamadı."
                    )
                    continue
                # Önbellekten çek — None dönerse atla
                if self.bot.get_cached_image(image_file) is None:
                    continue
                try:
                    match = self.bot.find_image_on_screen(image_file, search_region, confidence)
                    if match:
                        return int(data.get("value", -1))
                except Exception as e:
                    self.bot.log(
                        f"Görsel konum tespiti sırasında OpenCV hatası ({image_file}): {e}"
                    )

        return self.NAME_TO_VALUE.get("UNKNOWN", -1)


# =============================================================================
# 3) Automator Sınıfı
# =============================================================================

class Automator:
    def __init__(self, bot):
        self.bot = bot

    # ------------------------------------------------------------------
    def _get_setting(self, key, default):
        return self.bot.settings.get(key, default)

    # ------------------------------------------------------------------
    def _verify_anchor(self, label: str, coord: dict) -> bool:
        """
        Koordinatın 'anchor' / 'anchors' alanını okur; görsel doğrulama yapar.
        Çapa ekranda bulunamazsa UYARI loglar ama True döner (tıklama engellenmez).
        """
        anchor_list = []
        single_anchor = coord.get("anchor")
        if single_anchor:
            anchor_list.append(single_anchor)
        else:
            multiple_anchors = coord.get("anchors")
            if isinstance(multiple_anchors, list):
                anchor_list.extend(multiple_anchors)

        # Çapa tanımı yoksa — doğrulamayı atla, devam et
        if not anchor_list:
            return True

        found_match = False
        for anchor in anchor_list:
            if not anchor or not isinstance(anchor, dict):
                continue

            image_file = anchor.get("image_file")
            if not image_file:
                continue

            region_key = anchor.get("region_key")
            if not region_key:
                self.bot.log(f"[{label}] Anchor Hatası: 'region_key' eksik. Doğrulama atlandı.")
                continue

            region = self.bot.ui_regions.get(region_key)
            if not region:
                self.bot.log(
                    f"[{label}] Anchor Hatası: 'ui_regions' içinde "
                    f"'{region_key}' bölgesi bulunamadı. Doğrulama atlandı."
                )
                continue

            confidence = float(anchor.get("confidence", 0.8))
            log_msg    = f"Anchor Doğrulama: '{image_file}' aranıyor... ({label})"
            self.bot.log(log_msg)
            self.bot.gui_queue.put(("set_status_line1", log_msg))

            match = self.bot.find_image_on_screen(image_file, region, confidence)
            if match:
                log_msg = f"Anchor Başarılı: '{image_file}' bulundu. ({label})"
                self.bot.log(log_msg)
                self.bot.gui_queue.put(("set_status_line1", log_msg))
                found_match = True
                break

        if not found_match:
            tried = [anc.get("image_file", "N/A") for anc in anchor_list]
            log_msg = (
                f"UYARI: '{label}' için anchor görseli bulunamadı {tried}. "
                f"Yine de tıklanıyor..."
            )
            self.bot.log(log_msg)
            self.bot.gui_queue.put(("set_status_line1", log_msg))

        # Çapa bulunamazsa da tıklamayı engelleme — True döndür
        return True

    # ------------------------------------------------------------------
    def click(self, label: str | None, coord_override: dict | None = None,
              check_game: bool = True) -> bool:
        """Belirtilen etikete veya koordinat override'ına tıklar."""
        if check_game and not self.bot.is_game_running():
            self.bot.log("Oyun çalışmıyor/dondu. Tıklama iptal.")
            return False

        coord_to_use = None
        action_label = "override"

        if coord_override and isinstance(coord_override, dict):
            coord_to_use = coord_override
            if label:
                action_label = label
        elif label:
            action_label = label
            coord_data = self.bot.coordinates.get(label)
            if isinstance(coord_data, dict):
                coord_to_use = coord_data
            else:
                self.bot.log(f"Koordinat bulunamadı veya formatı yanlış: {label}")
                return False
        else:
            self.bot.log("click() fonksiyonuna tıklanacak bir etiket veya koordinat verilmedi.")
            return False

        # Anchor doğrulaması — bulunamazsa uyarı verir ama tıklamayı durdurmaz
        self._verify_anchor(action_label, coord_to_use)

        try:
            x             = coord_to_use["x"]
            y             = coord_to_use["y"]
            move_duration = self._get_setting("MOUSE_MOVE_DURATION_SN", 0.2)
            try:
                BOT_IS_CLICKING_EVENT.set()
                pyautogui.click(x=x, y=y, duration=move_duration)
            finally:
                time.sleep(0.05)
                BOT_IS_CLICKING_EVENT.clear()

            time.sleep(self._get_setting("ACTION_DELAY_SN", 0.15))
            return True

        except KeyError:
            self.bot.log(
                f"Tıklama hatası: Koordinat '{action_label}' için 'x' veya 'y' anahtarı eksik."
            )
            return False
        except Exception as e:
            self.bot.log(f"Tıklama sırasında beklenmedik hata ({action_label}): {e}")
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    def press_key(self, key: str, label: str = "manual_key") -> bool:
        """Klavye tuşuna basar."""
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            self.bot.log(f"Tuş basma hatası ({key}): {e}")
            return False

    # ------------------------------------------------------------------
    def login_to_game(self):
        self.bot.log("Giriş ekranı bekleniyor (6sn)...")
        if not self.bot._interruptible_wait(6):
            return

        if not self.click("server_s575", check_game=False):
            self.bot.log("UYARI: Sunucu koordinatı bulunamadı veya tıklanamadı. Devam ediliyor...")

        self.bot.log("Sunucu seçildi (varsayıldı). Oyunun yüklenmesi bekleniyor (20sn)...")
        if not self.bot._interruptible_wait(20):
            return

        self.bot.log("Etkinlik oku kapatılıyor (varsa) ve otomatik savaş ('Z') başlatılıyor.")
        self.click("etkinlik_oku_close", check_game=False)
        time.sleep(self.bot.settings.get("SLEEP_SHORT", 1))
        self.press_key("z", label="auto_battle_start")
        self.bot.log("Otomatik savaş tuşu ('Z') gönderildi.")

    # ------------------------------------------------------------------
    def return_to_exp_farm(self, force_restart_if_failed: bool = False) -> bool:
        current_region_name = self.bot.location_manager.get_region_name()
        if current_region_name == "EXP_FARM":
            return True

        self.bot.log(f"Haritadan ({current_region_name}) Exp Farm'a dönülüyor...")

        if self.bot.safe_exit_map(confidence=0.8):
            wait_time = self._get_setting("SLEEP_SHORT", 1)
            self.bot.log(f"Haritadan çıkıldı. Exp Farm'a geçiş için {wait_time} sn bekleniyor...")
            if not self.bot._interruptible_wait(wait_time):
                self.bot.log("Exp Farm'a dönüş sırasında bekleme iptal edildi.")
                return False
            self.bot.location_manager.set_current_location_by_name("EXP_FARM")
            return True
        else:
            self.bot.log("Haritadan çıkış işlemi başarısız oldu (buton bulunamadı/tıklanamadı).")
            if force_restart_if_failed:
                self.bot.log("Arayüz donmuş olabilir. Oyun yeniden başlatılıyor.")
                self.bot.restart_game()
            return False


# =============================================================================
# 4) EventManager Sınıfı
# =============================================================================

class EventManager:
    """
    Zamanlanmış oyun içi etkinlikleri yönetir.
    'auto_exits' bayrağını destekler.
    """
    def __init__(self, bot, events_config_data: dict):
        self.bot            = bot
        self.events_config  = events_config_data
        self.SCHEDULED_EVENTS = self._get_scheduled_events()
        self.last_event_run   = {}

    # ------------------------------------------------------------------
    def _get_scheduled_events(self) -> list:
        """Etkinlik takvimini 'module_timed_events' yapısından okur."""
        def t(time_str):
            try:
                h, m = map(int, str(time_str).split(":"))
                return dt_time(hour=h, minute=m)
            except Exception:
                return None

        self.bot.log("Config'den 'module_timed_events' yükleniyor...")
        config_events   = self.events_config.get("events", [])
        processed_events = []

        for event in config_events:
            try:
                start_time = t(event.get("start_time"))
                end_time   = t(event.get("end_time"))

                if start_time is None or end_time is None:
                    self.bot.log(
                        f"UYARI: Etkinlik '{event.get('name')}' için "
                        f"'start_time' veya 'end_time' geçersiz, atlanıyor."
                    )
                    continue

                action_type = event.get("action_type", "static_sequence")

                processed_event = {
                    "name":                  event.get("name", "Bilinmeyen Etkinlik"),
                    "days":                  event.get("days"),
                    "start_time":            start_time,
                    "end_time":              end_time,
                    "action_type":           action_type,
                    "location_name_on_enter": event.get("location_name_on_enter", "UNKNOWN"),
                    "auto_exits":            event.get("auto_exits", False),
                    "action_sequence":       event.get("action_sequence", []),
                    "entry_sequence":        event.get("entry_sequence", []),
                    "hunt_target_list":      event.get("hunt_target_list"),
                }
                processed_events.append(processed_event)

            except Exception as e:
                self.bot.log(
                    f"UYARI: Etkinlik işlenirken hata ({event.get('name', 'Bilinmeyen')}): {e}"
                )
                continue

        valid_events = [
            e for e in processed_events
            if e["start_time"] is not None and e["end_time"] is not None
        ]

        if valid_events:
            self.bot.log(f"Config'den {len(valid_events)} zamanlı etkinlik yüklendi.")
        else:
            self.bot.log("UYARI: Config'den geçerli etkinlik yüklenemedi.")
        return valid_events

    # ------------------------------------------------------------------
    def event_monitor_thread(self):
        """Etkinlik zamanlarını periyodik olarak kontrol eden thread fonksiyonu."""
        self.bot.log("Etkinlik izleyici başlatıldı.")
        interval = self.bot.settings.get("EVENT_MONITOR_SLEEP_SEC", 30)

        while True:
            try:
                today_str = datetime.now().strftime("%Y-%m-%d")
                old_keys  = [k for k in list(self.last_event_run.keys()) if not k.startswith(today_str)]
                for k in old_keys:
                    del self.last_event_run[k]

                if not self.bot.running.is_set() or self.bot.paused:
                    time.sleep(1.0)
                    continue

                now             = datetime.now()
                current_time    = now.time()
                current_weekday = now.weekday()

                event_to_run = None
                for event in self.SCHEDULED_EVENTS:
                    day_match  = event["days"] is None or current_weekday in event["days"]
                    time_match = event["start_time"] <= current_time < event["end_time"]
                    start_time_str = event["start_time"].strftime("%H:%M")
                    event_key      = f"{today_str}_{event['name']}_{start_time_str}"
                    already_run    = self.last_event_run.get(event_key, False)

                    if day_match and time_match and not already_run:
                        event_to_run = event
                        break

                if event_to_run:
                    self.bot.log(f"Etkinlik zamanı geldi: {event_to_run['name']}.")
                    self.bot.active_event = event_to_run
                    start_time_str = event_to_run["start_time"].strftime("%H:%M")
                    event_key      = f"{today_str}_{event_to_run['name']}_{start_time_str}"

                    action_success = False
                    self.bot.log(f"'{event_to_run['name']}' için aksiyon kilidi bekleniyor...")

                    with self.bot.action_lock:
                        self.bot.log(f"'{event_to_run['name']}' için aksiyon kilidi alındı.")
                        if not self.bot.running.is_set():
                            self.bot.log("Etkinlik aksiyonu öncesi bot durduruldu.")
                            self.bot.active_event = None
                            continue

                        try:
                            self.bot.log(f"'{event_to_run['name']}' etkinliğine geçiliyor...")

                            if self.bot.location_manager.get_region_name() != "EXP_FARM":
                                if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=False):
                                    self.bot.log(
                                        f"'{event_to_run['name']}' etkinliğine girmeden önce "
                                        f"Exp Farm'a dönülemedi. İptal."
                                    )
                                    action_success = False
                                else:
                                    if not self.bot._interruptible_wait(
                                        self.bot.settings.get("SLEEP_SHORT", 1)
                                    ):
                                        raise Exception("Bekleme iptal edildi")

                            action_type = event_to_run.get("action_type", "static_sequence")

                            if self.bot.location_manager.get_region_name() == "EXP_FARM":
                                if action_type == "dynamic_hunt":
                                    self.bot.log(
                                        f"UYARI: Dinamik Av kodu mevcut değil "
                                        f"({event_to_run['name']}). 'static_sequence' kullanın."
                                    )
                                    action_success = False

                                elif action_type == "static_sequence":
                                    action_sequence = event_to_run.get("action_sequence", [])
                                    if not action_sequence:
                                        self.bot.log(
                                            f"HATA: '{event_to_run['name']}' için "
                                            f"'action_sequence' bulunamadı."
                                        )
                                        action_success = False
                                    else:
                                        action_success = self.bot._run_sequence(
                                            action_sequence,
                                            self.bot.timed_events_cfg.get("coordinates", {}),
                                        )
                                else:
                                    self.bot.log(
                                        f"HATA: Bilinmeyen action_type: '{action_type}'"
                                    )
                                    action_success = False

                                if action_success:
                                    self.bot.location_manager.set_current_location_by_name(
                                        event_to_run["location_name_on_enter"]
                                    )
                                    self.bot.log(
                                        f"'{event_to_run['name']}' etkinliğine giriş başarılı."
                                    )
                                else:
                                    self.bot.log(
                                        f"'{event_to_run['name']}' etkinlik girişi başarısız."
                                    )

                        except Exception as e:
                            self.bot.log(
                                f"Etkinlik '{event_to_run['name']}' çalıştırılırken "
                                f"hata oluştu: {e}"
                            )
                            traceback.print_exc()
                            action_success = False

                    self.bot.log(f"'{event_to_run['name']}' için aksiyon kilidi bırakıldı.")

                    event_auto_exits = event_to_run.get("auto_exits", False)

                    if action_success and event_auto_exits:
                        self.bot.log(
                            f"'{event_to_run['name']}' etkinliği 'auto_exits: true'. "
                            f"Bitiş bekleniyor..."
                        )
                        end_dt       = datetime.combine(now.date(), event_to_run["end_time"])
                        wait_seconds = max(0, (end_dt - datetime.now()).total_seconds()) + 5
                        if wait_seconds > 0:
                            if not self.bot._interruptible_wait(wait_seconds):
                                self.bot.log("Bekleme kesildi.")
                        self.bot.log(f"'{event_to_run['name']}' bitti. Konum Exp Farm varsayılıyor.")
                        self.bot.location_manager.set_current_location_by_name("EXP_FARM")
                        self.last_event_run[event_key] = True

                    elif action_success and not event_auto_exits:
                        end_dt       = datetime.combine(now.date(), event_to_run["end_time"])
                        wait_seconds = max(0, (end_dt - datetime.now()).total_seconds())
                        if wait_seconds > 0:
                            self.bot.log(
                                f"'{event_to_run['name']}' bitişi için "
                                f"{int(wait_seconds)} sn bekleniyor..."
                            )
                            if not self.bot._interruptible_wait(wait_seconds):
                                self.bot.log("Bekleme kesildi.")
                        self.bot.log(f"'{event_to_run['name']}' bitti, Exp Farm'a dönülüyor...")
                        with self.bot.action_lock:
                            if self.bot.running.is_set():
                                if self.bot.automator.return_to_exp_farm(force_restart_if_failed=True):
                                    self.last_event_run[event_key] = True
                                    self.bot.log("Exp Farm'a dönüldü.")
                                else:
                                    self.bot.log("UYARI: Exp Farm'a dönülemedi.")
                                    self.last_event_run[event_key] = True

                    elif not action_success:
                        self.bot.log(f"'{event_to_run['name']}' girişi BAŞARISIZ.")
                        self.last_event_run[event_key] = True

                    self.bot.active_event = None
                    time.sleep(interval)
                    continue

                now = datetime.now()
                seconds_since_interval = now.second % interval
                sleep_time = interval - seconds_since_interval
                if sleep_time <= 0:
                    sleep_time = interval
                time.sleep(sleep_time)

            except Exception as e:
                self.bot.log(f"Event monitor ana döngüsünde hata: {e}")
                traceback.print_exc()
                time.sleep(interval * 2)


# =============================================================================
# 5) GUIManager Sınıfı
# =============================================================================

class GUIManager:
    def __init__(self, bot, update_interval_ms: int = 1000):
        self.bot  = bot
        self.root = getattr(bot, "root", None)

        if self.root is None:
            print("KRİTİK HATA: GUIManager başlatılamadı, root (Tk ana penceresi) bulunamadı.")
            return

        self.gui_update_interval_ms = int(update_interval_ms)
        self.title_label_line1 = None
        self.title_label_line2 = None
        self.log_text_widget   = None
        self.gui_widgets       = {}
        self.gui_queue         = getattr(bot, "gui_queue", Queue())
        self.boss_targets_dict = getattr(bot, "bosslar", {})
        self._created          = False

        self.status_message_active    = False
        self.last_status_message_time = 0

    # ------------------------------------------------------------------
    def _log(self, text: str):
        try:
            if hasattr(self, "bot") and hasattr(self.bot, "log") and callable(self.bot.log):
                self.bot.log(f"[GUI] {text}")
            else:
                print(f"[GUI-LOG] {text}")
                if self.gui_queue:
                    self.gui_queue.put(("log", f"[GUI-Hata] {text}"))
        except Exception as e:
            print(f"[GUI-Loglama Hatası] {text} - Hata: {e}")

    # ------------------------------------------------------------------
    def _parse_time_to_seconds(self, s: str) -> int | None:
        s = s.strip()
        try:
            parts = list(map(int, s.split(":")))
            if   len(parts) == 3: h, m, sec = parts
            elif len(parts) == 2: h, m, sec = 0, parts[0], parts[1]
            elif len(parts) == 1: h, m, sec = 0, 0, parts[0]
            else: return None
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
                return None
            return h * 3600 + m * 60 + sec
        except ValueError:
            return None
        except Exception as e:
            self._log(f"Zaman ayrıştırma hatası ({s}): {e}")
            return None

    # ------------------------------------------------------------------
    def _create_round_button(self, parent, symbol, color, command, col):
        try:
            canvas  = tk.Canvas(
                parent, width=60, height=60, bg="black", highlightthickness=0, cursor="hand2"
            )
            canvas.grid(row=0, column=col, padx=10, pady=5)
            circle  = canvas.create_oval(
                5, 5, 55, 55,
                fill=color, outline=color,
                activefill=self._adjust_brightness(color, 0.8),
            )
            text_item = canvas.create_text(
                30, 30, text=symbol, fill="white", font=("Consolas", 18, "bold")
            )

            def on_click(event=None):
                try:
                    command()
                except Exception as e:
                    self._log(f"Buton '{symbol}' komutu çalıştırılırken hata: {e}")
                    traceback.print_exc()

            for item in (circle, text_item):
                canvas.tag_bind(item, "<Button-1>", on_click)
        except Exception as e:
            self._log(f"Yuvarlak buton '{symbol}' oluşturulurken hata: {e}")

    # ------------------------------------------------------------------
    def _adjust_brightness(self, color_name, factor):
        try:
            rgb    = self.root.winfo_rgb(color_name)
            r, g, b = [x / 256 for x in rgb]
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return color_name

    # ------------------------------------------------------------------
    def _edit_timer_label(self, event, boss_aciklama: str):
        """Boss zamanlayıcısına tıklandığında manuel süre girişi sağlar."""
        try:
            hedef = self.bot.bosslar.get(boss_aciklama)
            if not hedef:
                self._log(f"Manuel süre düzenleme hatası: Boss '{boss_aciklama}' bulunamadı.")
                return

            input_time = simpledialog.askstring(
                "Zamanı Düzenle",
                f"'{boss_aciklama}' için kalan süreyi girin:\n(HH:MM:SS, MM:SS veya Saniye)",
                parent=self.root,
            )
            if input_time is None:
                return

            kalan_saniye = self._parse_time_to_seconds(input_time)
            if kalan_saniye is None:
                messagebox.showwarning("Geçersiz Format", "Geçersiz zaman formatı!", parent=self.root)
                return

            self.bot._set_spawn_time(boss_aciklama, kalan_saniye, source="manual")
            self._log(
                f"'{boss_aciklama}' için manuel süre ayarlandı: "
                f"{input_time} ({kalan_saniye} saniye)"
            )
        except Exception as e:
            self._log(f"Manuel süre düzenleme sırasında hata ({boss_aciklama}): {e}")
            messagebox.showerror(
                "Hata", f"Süre ayarlanırken bir hata oluştu:\n{e}", parent=self.root
            )

    # ------------------------------------------------------------------
    def _process_gui_queue(self):
        """Diğer thread'lerden gelen mesajları işler."""
        if self.gui_queue is None:
            return

        processed_count      = 0
        max_process_per_cycle = 50

        while processed_count < max_process_per_cycle:
            try:
                task, data = self.gui_queue.get_nowait()
            except Exception:
                break

            processed_count += 1
            try:
                if task == "log":
                    if isinstance(data, str) and self.log_text_widget:
                        now          = datetime.now().strftime("%H:%M:%S")
                        formatted_msg = f"[{now}] {data}"
                        self.log_text_widget.config(state=tk.NORMAL)
                        self.log_text_widget.insert(tk.END, formatted_msg + "\n")
                        self.log_text_widget.see(tk.END)
                        current_lines = int(
                            self.log_text_widget.index(f"{tk.END}-1c").split(".")[0]
                        )
                        if current_lines > 500:
                            lines_to_delete = current_lines - 400
                            self.log_text_widget.delete("1.0", f"{lines_to_delete}.0")
                        self.log_text_widget.config(state=tk.DISABLED)

                elif task == "set_spawn":
                    if isinstance(data, tuple) and len(data) == 3:
                        aciklama, spawn_time, source = data
                        if isinstance(aciklama, str) and isinstance(spawn_time, (int, float)):
                            self._update_spawn_time_in_gui(aciklama, spawn_time, source)
                        else:
                            self._log(
                                f"GUI Queue hatası: 'set_spawn' için geçersiz veri tipi: "
                                f"{type(aciklama)}, {type(spawn_time)}"
                            )
                    else:
                        self._log(
                            f"GUI Queue hatası: 'set_spawn' için geçersiz veri formatı: {data}"
                        )

                elif task == "set_status_line1":
                    if isinstance(data, str) and self.title_label_line1:
                        self.title_label_line1.config(text=data, fg="yellow")
                        self.status_message_active    = True
                        self.last_status_message_time = time.monotonic()

            except Exception as e:
                self._log(f"GUI queue mesajı işlenirken hata (Task: {task}): {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------
    def _update_spawn_time_in_gui(self, aciklama: str, spawn_time: float, source: str):
        """GUI'deki ilgili boss için spawn zamanı entry'sini günceller."""
        if not aciklama:
            return

        widgets = self.gui_widgets.get(aciklama)
        if not widgets or "entry" not in widgets:
            return

        try:
            spawn_str    = datetime.fromtimestamp(spawn_time).strftime("%H:%M:%S") if spawn_time > 0 else "--:--:--"
            entry_widget = widgets["entry"]
            entry_widget.config(state=tk.NORMAL)
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, spawn_str)
            color = "lime" if source == "automation" else "yellow" if source == "manual" else "cyan"
            entry_widget.config(fg=color)
            entry_widget.config(state="readonly")
        except Exception as e:
            self._log(f"GUI spawn zamanı güncellenirken hata ({aciklama}): {e}")

    # ------------------------------------------------------------------
    def update_status_label(self, remaining_targets):
        """GUI'nin en üstündeki İKİ SATIRLI durum başlığını günceller."""
        if self.title_label_line1 is None or self.title_label_line2 is None:
            return
        try:
            is_running   = self.bot.running.is_set()
            is_paused    = self.bot.paused
            active_event = self.bot.active_event
            attacking    = self.bot.attacking_target_aciklama
            region_name  = self.bot.location_manager.get_region_name()

            # --- SATIR 1 ---
            if self.status_message_active:
                if (time.monotonic() - self.last_status_message_time) >= 3.0:
                    self.status_message_active = False

            if not self.status_message_active:
                text_line1  = "Başlatılıyor..."
                color_line1 = "gray"

                if is_paused:
                    text_line1  = "Duraklatıldı"
                    color_line1 = "goldenrod"
                elif not is_running:
                    text_line1  = "Beklemede"
                    color_line1 = "gray"
                elif active_event:
                    try:
                        end_dt      = datetime.combine(datetime.now().date(), active_event["end_time"])
                        kalan_sn    = max(0, (end_dt - datetime.now()).total_seconds())
                        saat, rem   = divmod(int(kalan_sn), 3600)
                        dakika, sn  = divmod(rem, 60)
                        time_str    = f"{saat:02}:{dakika:02}:{sn:02}"
                        event_short = active_event["name"].split("(")[0].strip()
                        text_line1  = f"Etkinlik: {event_short} ({time_str})"
                        color_line1 = "orange"
                    except Exception:
                        text_line1  = f"Etkinlik: {active_event.get('name', 'Bilinmeyen')}"
                        color_line1 = "orange"
                elif attacking:
                    text_line1  = f"Saldırı: {attacking}"
                    color_line1 = "red"
                else:
                    text_line1  = f"Konum: {region_name}"
                    color_line1 = "lime" if region_name == "EXP_FARM" else "gray"

                self.title_label_line1.config(text=text_line1, fg=color_line1)

            # --- SATIR 2 ---
            text_line2  = "Boss hedefi yok"
            color_line2 = "gray"

            if is_paused or not is_running:
                text_line2  = "Boss takibi duraklatıldı"
                color_line2 = "goldenrod"
            elif attacking:
                text_line2  = f"{attacking} saldırılıyor..."
                color_line2 = "red"
            elif remaining_targets:
                kalan, aciklama = remaining_targets[0]
                saat, rem       = divmod(int(kalan), 3600)
                dakika, sn      = divmod(rem, 60)
                time_str        = f"{saat:02}:{dakika:02}:{sn:02}"
                text_line2      = f"Sıradaki: {aciklama} ({time_str})"
                color_line2     = "cyan"

            self.title_label_line2.config(text=text_line2, fg=color_line2)

        except Exception as e:
            self._log(f"GUI durum başlığı güncellenirken hata: {e}")
            try:
                self.title_label_line1.config(text="Durum Hatası", fg="red")
                self.title_label_line2.config(text="Durum Hatası", fg="red")
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _update_gui(self):
        """GUI'yi periyodik olarak güncelleyen ana fonksiyon."""
        try:
            self._process_gui_queue()

            now        = time.monotonic()  # noqa: F841
            remaining  = []
            self.boss_targets_dict = self.bot.bosslar

            for aciklama, hedef in self.boss_targets_dict.items():
                widgets = self.gui_widgets.get(aciklama)
                if not widgets or "timer_label" not in widgets:
                    continue

                spawn_time   = hedef.get("spawn_time")
                timer_label  = widgets["timer_label"]

                if spawn_time and isinstance(spawn_time, (int, float)):
                    kalan       = max(0, spawn_time - time.time())
                    if kalan > 0:
                        remaining.append((kalan, aciklama))
                    saat, rem   = divmod(int(kalan), 3600)
                    dakika, sn  = divmod(rem, 60)
                    time_str    = f"{saat:02}:{dakika:02}:{sn:02}"
                    try:
                        timer_label.config(text=time_str)
                    except Exception as e:
                        self._log(f"Zamanlayıcı etiketi güncellenirken hata ({aciklama}): {e}")
                else:
                    try:
                        timer_label.config(text="--:--:--")
                    except Exception as e:
                        self._log(f"Zamanlayıcı etiketi temizlenirken hata ({aciklama}): {e}")

            remaining.sort(key=lambda x: x[0])
            self.update_status_label(remaining)

        except Exception as e:
            self._log(f"Ana GUI güncelleme döngüsünde hata: {e}")
            traceback.print_exc()
        finally:
            try:
                if self.root and self.root.winfo_exists():
                    self.root.after(self.gui_update_interval_ms, self._update_gui)
            except Exception as e:
                self._log(f"GUI 'after' zamanlaması hatası: {e}")

    # ------------------------------------------------------------------
    def create_gui(self):
        """GUI penceresini ve içindeki widget'ları oluşturur."""
        if self._created:
            return
        if self.root is None:
            self._log("GUI oluşturulamaz: root penceresi yok.")
            return
        try:
            screen_width  = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            width  = min(320, max(280, screen_width  // 6))
            height = min(600, max(500, screen_height // 2 + 50))

            self.root.title("LoA Bot v5.9")
            self.root.wm_attributes("-topmost", True)
            self.root.configure(bg="black")
            self.root.geometry(f"{width}x{height}")
            self.root.resizable(False, False)

            self.title_label_line1 = tk.Label(
                self.root, text="Başlatılıyor...",
                font=("Consolas", 11, "bold"), fg="cyan", bg="black",
            )
            self.title_label_line1.pack(pady=(10, 0), fill=tk.X, padx=10)

            self.title_label_line2 = tk.Label(
                self.root, text="Boss hedefleri bekleniyor...",
                font=("Consolas", 11, "bold"), fg="gray", bg="black",
            )
            self.title_label_line2.pack(pady=(0, 5), fill=tk.X, padx=10)

            boss_frame = tk.LabelFrame(
                self.root, text=" Boss Zamanlayıcıları ",
                bg="black", fg="white", font=("Consolas", 10), relief=tk.GROOVE, bd=2,
            )
            boss_frame.pack(padx=10, pady=5, fill=tk.X)
            boss_frame.grid_columnconfigure(0, weight=1)
            boss_frame.grid_columnconfigure(1, weight=1)
            boss_frame.grid_columnconfigure(2, weight=1)

            self.boss_targets_dict = self.bot.bosslar
            if not self.boss_targets_dict:
                no_target_label = tk.Label(
                    boss_frame,
                    text="Config'de boss hedefi tanımlanmamış.",
                    font=("Consolas", 10), fg="yellow", bg="black",
                )
                no_target_label.grid(row=0, column=0, columnspan=3, pady=10)
            else:
                for i, (aciklama, hedef) in enumerate(self.boss_targets_dict.items()):
                    lbl = tk.Label(
                        boss_frame, text=f"{aciklama}:", anchor="e",
                        font=("Consolas", 10), fg="white", bg="black",
                    )
                    lbl.grid(row=i, column=0, padx=(10, 2), pady=1, sticky="ew")

                    entry = tk.Entry(
                        boss_frame, width=9, font=("Consolas", 10, "bold"),
                        readonlybackground="#202020", relief=tk.FLAT, justify="center",
                    )
                    entry.grid(row=i, column=1, padx=2, pady=1, sticky="ew")
                    entry.config(state="readonly", fg="cyan")

                    timer_label = tk.Label(
                        boss_frame, text="--:--:--", anchor="w",
                        font=("Consolas", 10), fg="white", bg="black", cursor="hand2",
                    )
                    timer_label.grid(row=i, column=2, padx=(2, 10), pady=1, sticky="ew")
                    timer_label.bind(
                        "<Button-1>", lambda event, a=aciklama: self._edit_timer_label(event, a)
                    )

                    self.gui_widgets[aciklama] = {"entry": entry, "timer_label": timer_label}
                    if hedef.get("spawn_time"):
                        self._update_spawn_time_in_gui(
                            aciklama, hedef["spawn_time"], hedef.get("source", "config")
                        )

            button_frame = tk.Frame(self.root, bg="black")
            button_frame.pack(pady=(5, 8))
            self._create_round_button(button_frame, "▶", "darkgreen",  lambda: self.bot.start(), 0)
            self._create_round_button(button_frame, "⏸", "goldenrod",  lambda: self.bot.stop(),  1)
            self._create_round_button(button_frame, "✖", "darkred",    lambda: self.root.quit(), 2)

            self.log_text_widget = tk.Text(
                self.root, height=10, bg="#101010", fg="#DDDDDD",
                font=("Consolas", 9), state=tk.DISABLED,
                relief=tk.SOLID, bd=1, wrap=tk.WORD,
            )
            self.log_text_widget.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

            self.root.after(self.gui_update_interval_ms, self._update_gui)
            self._created = True
            self._log("GUI başarıyla oluşturuldu.")

        except Exception as e:
            self._log(f"GUI oluşturma sırasında kritik hata: {e}")
            traceback.print_exc()
            try:
                messagebox.showerror("GUI Hatası", f"Arayüz oluşturulamadı:\n{e}")
            except Exception:
                pass
            try:
                if self.root:
                    self.root.quit()
            except Exception:
                pass


# =============================================================================
# 6) Ana Bot Sınıfı — LoABot v5.9
# =============================================================================

class LoABot:
    """
    Ana bot sınıfı.
    v5.9 yenilikleri:
      • koordinat_ref → str etiket VEYA doğrudan {x, y} dict
      • Her boss için anchor / spawn_check / victory hiyerarşik YAML alanları
      • _run_sequence click adımlarında anchor doğrulaması
      • _attack_boss: spawn_check bekleme + victory doğrulama
      • spawn_check_defaults & victory_check_defaults YAML'dan okunur
    """

    def __init__(self, root):
        self.root      = root
        self.gui_queue = Queue()

        self.config = load_config(CONFIG_FILE)
        if not self.config:
            log_to_file("KRİTİK: Config yüklenemediği için bot başlatılamıyor.")
            self.initialized = False
            try:
                if self.root:
                    self.root.quit()
            except Exception:
                pass
            return

        self.general_cfg      = self.config.get("module_general",     {})
        self.demon_boss_cfg   = self.config.get("module_demon_boss",  {})
        self.timed_events_cfg = self.config.get("module_timed_events",{})

        self.initialized = True

        # Manuel tıklama sinyali
        self.manual_click_detected = threading.Event()
        start_click_listener(self)

        try:
            self.game_path         = self.general_cfg["game_path"]
            self.process_name      = self.general_cfg["process_name"]
            self.settings          = self.general_cfg.get("settings",       {})
            self.ui_regions        = self.general_cfg.get("ui_regions",     {})
            self.exit_buttons_config = self.general_cfg.get("exit_buttons", {})
            self.popup_config      = self.general_cfg.get("POPUP_MONITOR",  {})
            self.pvp_config        = self.general_cfg.get("PVP_DEFENSE",    {"enabled": False})
        except KeyError as e:
            error_msg = f"Config hatası: 'module_general' içinde zorunlu alan eksik: {e}"
            self._log_to_file_and_message(error_msg)
            self.initialized = False
            try:
                if self.root:
                    self.root.quit()
            except Exception:
                pass
            return

        # -------------------------------------------------------------------
        # YAML'dan spawn_check ve victory varsayılanlarını yükle
        # -------------------------------------------------------------------
        self._spawn_check_defaults = self.demon_boss_cfg.get("spawn_check_defaults", {})
        self._victory_check_defaults = self.demon_boss_cfg.get("victory_check_defaults", {})

        # Koordinat haritaları (demon_boss + timed_events birleştir)
        self.coordinates = self.demon_boss_cfg.get("coordinates", {})
        self.coordinates.update(self.timed_events_cfg.get("coordinates", {}))

        self.bosslar              = self.demon_boss_cfg.get("bosslar",           {})
        self.walk_times           = self.demon_boss_cfg.get("walk_times",         {})
        self.boss_sequence_template = self.demon_boss_cfg.get("sequence_template", [])

        self.settings.setdefault("SLEEP_SHORT",  1)
        self.settings.setdefault("SLEEP_MEDIUM", 2)
        self.settings.setdefault("SLEEP_LONG",   5)

        # Boss verilerini hazırla
        self._initialize_boss_data()

        self.running   = threading.Event()
        self.paused    = True
        self.action_lock = threading.Lock()

        self.active_event               = None
        self.attacking_target_aciklama  = None
        self.loot_collection_ends_at    = 0

        self._pid                      = None
        self._last_screenshot_checksum = None
        self._last_screenshot_time     = 0
        self._freeze_count             = 0

        self.location_manager = LocationManager(self, self.general_cfg)
        self.automator        = Automator(self)
        self.event_manager    = EventManager(self, self.timed_events_cfg)
        self.gui_manager      = GUIManager(self,
                                           self.settings.get("GUI_UPDATE_INTERVAL_MS", 500))

        self.log("Görüntü önbelleği (image cache) yükleniyor...")
        self.image_cache = {}
        self._preload_images_from_config()
        self.log(f"{len(self.image_cache)} adet görüntü hafızaya yüklendi.")

        self._threads_started = False
        self.log("LoABot v5.9 başlatma işlemi tamamlandı.")

    # ==========================================================================
    # Boss Veri Başlatma
    # ==========================================================================

    def _initialize_boss_data(self):
        """
        Boss verilerini başlangıçta hazırlar.

        v5.9: koordinat_ref hem str etiket hem de doğrudan {x, y} dict olabilir.
        """
        if not self.bosslar:
            self.log(
                "UYARI: config.yaml 'module_demon_boss.bosslar' boş. "
                "Sadece etkinlik takibi yapılacak."
            )
            return

        geçersiz_keys = []
        for aciklama, t in list(self.bosslar.items()):
            if not isinstance(t, dict):
                self.log(f"Config hatası: bosslar['{aciklama}'] bir sözlük değil, atlanıyor.")
                geçersiz_keys.append(aciklama)
                continue

            t["spawn_time"] = None
            t["source"]     = "config"
            t["aciklama"]   = str(aciklama)

            coord_ref  = t.get("koordinat_ref")
            katman_ref = t.get("katman_id")

            if not katman_ref:
                self.log(
                    f"Config hatası: Boss '{aciklama}' için 'katman_id' eksik, atlanıyor."
                )
                geçersiz_keys.append(aciklama)
                continue

            # --- YENİ v5.9: koordinat_ref çift format desteği ---
            if isinstance(coord_ref, dict) and "x" in coord_ref and "y" in coord_ref:
                # Doğrudan koordinat dict'i
                t["koordinat"] = coord_ref
            elif isinstance(coord_ref, str):
                # String etiket — coordinates sözlüğünde ara
                resolved = self.coordinates.get(coord_ref)
                if not resolved or not isinstance(resolved, dict):
                    self.log(
                        f"Config hatası: Boss '{aciklama}' için "
                        f"koordinat etiketi ('{coord_ref}') bulunamadı, atlanıyor."
                    )
                    geçersiz_keys.append(aciklama)
                    continue
                t["koordinat"] = resolved
            else:
                self.log(
                    f"Config hatası: Boss '{aciklama}' için 'koordinat_ref' "
                    f"geçersiz veya eksik, atlanıyor."
                )
                geçersiz_keys.append(aciklama)
                continue

        for key in geçersiz_keys:
            self.bosslar.pop(key, None)

        if not self.bosslar:
            self.log("UYARI: 'bosslar' içindeki tüm hedefler geçersiz.")

    # ==========================================================================
    # Görüntü Önbelleği
    # ==========================================================================

    def _preload_images_from_config(self):
        """
        YAML config dosyasını tarar ve 'image_file' içeren tüm görselleri yükler.
        v5.9: boss başına anchor / area_check / spawn_check / victory alanları dahil.
        """
        image_files_to_load = set()

        try:
            # 1. Konum Çapaları (locations)
            for loc_data in self.general_cfg.get("locations", {}).values():
                anchor_list = []
                single_anchor = loc_data.get("anchor")
                if single_anchor:
                    anchor_list.append(single_anchor)
                else:
                    multiple_anchors = loc_data.get("anchors")
                    if isinstance(multiple_anchors, list):
                        anchor_list.extend(multiple_anchors)
                for anchor in anchor_list:
                    if anchor and anchor.get("image_file"):
                        image_files_to_load.add(anchor["image_file"])

            # 2. Popup Monitörü
            for popup in self.popup_config.get("popups_to_close", []):
                if popup.get("image_file"):
                    image_files_to_load.add(popup["image_file"])

            # 3. PVP Monitörü
            if self.pvp_config.get("hp_damaged_template"):
                image_files_to_load.add(self.pvp_config["hp_damaged_template"])

            # 4. Demon Boss Koordinatları (anchor / anchors)
            for coord_data in self.demon_boss_cfg.get("coordinates", {}).values():
                if isinstance(coord_data, dict):
                    for anchor in self._collect_anchors(coord_data):
                        if anchor.get("image_file"):
                            image_files_to_load.add(anchor["image_file"])

            # 5. Demon Boss 'bosslar' — v5.9 hiyerarşik alanlar
            for boss_data in self.demon_boss_cfg.get("bosslar", {}).values():
                if not isinstance(boss_data, dict):
                    continue
                # anchor, area_check, spawn_check, victory alanları
                for field_name in ("anchor", "area_check", "spawn_check", "victory"):
                    field = boss_data.get(field_name)
                    if field and isinstance(field, dict) and field.get("image_file"):
                        image_files_to_load.add(field["image_file"])
                # Eski format geriye dönük uyumluluk
                verify = boss_data.get("gorsel_dogrulama")
                if verify and verify.get("image_file"):
                    image_files_to_load.add(verify["image_file"])

            # 6. Zamanlı Etkinlik Koordinatları
            for coord_data in self.timed_events_cfg.get("coordinates", {}).values():
                if isinstance(coord_data, dict):
                    for anchor in self._collect_anchors(coord_data):
                        if anchor.get("image_file"):
                            image_files_to_load.add(anchor["image_file"])

            # 7. Zamanlı Etkinlik Sekansları (verify_visual)
            for event in self.timed_events_cfg.get("events", []):
                for step in event.get("action_sequence", []):
                    verify = step.get("verify_visual")
                    if verify and verify.get("image_file"):
                        image_files_to_load.add(verify["image_file"])

            # 8. Dinamik Av (Hunt Targets) — geriye dönük
            for hunt_list in self.timed_events_cfg.get("hunt_targets", {}).values():
                for target in hunt_list.get("targets", []):
                    if target.get("live_image"):
                        image_files_to_load.add(target["live_image"])
                    if target.get("dead_image"):
                        image_files_to_load.add(target["dead_image"])

            # 9. Çıkış Butonları
            for exit_list in self.general_cfg.get("exit_buttons", {}).values():
                for button in exit_list:
                    if button.get("image_file"):
                        image_files_to_load.add(button["image_file"])

            # 10. Genel zafer ikonu (fallback)
            image_files_to_load.add("victory_icon.png")

            # Tümünü yükle
            for filename in image_files_to_load:
                if filename:
                    img = load_image(filename)
                    if img is not None:
                        self.image_cache[filename] = img

        except Exception as e:
            self.log(f"Görüntü önbelleği yüklenirken kritik hata: {e}")
            traceback.print_exc()

    @staticmethod
    def _collect_anchors(coord_dict: dict) -> list:
        """Bir koordinat dict'inden anchor listesini toplar."""
        result        = []
        single_anchor = coord_dict.get("anchor")
        if single_anchor and isinstance(single_anchor, dict):
            result.append(single_anchor)
        else:
            multiple_anchors = coord_dict.get("anchors")
            if isinstance(multiple_anchors, list):
                result.extend(multiple_anchors)
        return result

    # ------------------------------------------------------------------
    def get_cached_image(self, filename: str):
        if filename in self.image_cache:
            return self.image_cache[filename]
        self.log(f"UYARI: '{filename}' önbellekte bulunamadı. Diskten okunuyor...")
        img = load_image(filename)
        if img is not None:
            self.image_cache[filename] = img
        return img

    # ==========================================================================
    # Sekans Çalıştırıcı
    # ==========================================================================

    def _run_sequence(
        self,
        sequence: list,
        coord_map: dict,
        context_target: dict = None,
    ) -> bool:
        """
        Verilen bir sekans listesini (config'den) adım adım çalıştırır.

        v5.9 yeniliği — 'click' adımı:
          Koordinatın 'anchor' alanı varsa _verify_anchor() ile doğrulama yapar.
          Çapa bulunamazsa UYARI loglar ama tıklamayı iptal etmez.

        v5.9 yeniliği — 'boss_seçimi' adımı:
          koordinat_ref hem string etiket hem dict desteklenir.
          Görsel doğrulama için 'anchor' alanına bakar (eski 'gorsel_dogrulama' da çalışır).
        """
        if not sequence:
            self.log("UYARI: Boş bir sekans çalıştırılmaya çalışıldı.")
            return False

        for step in sequence:
            if not self.running.is_set():
                self.log("Sekans çalıştırılırken bot durduruldu.")
                return False

            action = step.get("action")
            label  = step.get("label")

            try:
                wait_ms = int(step.get("wait_ms", 100))
            except (ValueError, TypeError):
                self.log(
                    f"UYARI: '{label}' adımı için 'wait_ms' değeri geçersiz "
                    f"('{step.get('wait_ms')}'). Varsayılan 100ms."
                )
                wait_ms = 100

            action_success = False
            try:
                # -------------------------------------------------------
                if action == "click":
                    coord = coord_map.get(label)
                    if not coord:
                        self.log(f"Sekans Hatası: '{label}' için koordinat bulunamadı.")
                        return False
                    # v5.9: Koordinatın anchor alanı varsa önce doğrula
                    self.automator._verify_anchor(label, coord)
                    action_success = self.automator.click(label, coord_override=coord)

                # -------------------------------------------------------
                elif action == "press_key":
                    key = step.get("key")
                    if not key:
                        self.log("Sekans Hatası: 'press_key' için 'key' belirtilmemiş.")
                        return False
                    pyautogui.press(key)
                    action_success = True

                # -------------------------------------------------------
                elif action == "katman_seçimi":
                    if not context_target:
                        self.log("Sekans Hatası: 'katman_seçimi' için 'context_target' sağlanmadı.")
                        return False
                    katman_label = context_target.get("katman_id")
                    coord        = coord_map.get(katman_label)
                    if not katman_label or not coord:
                        self.log(
                            f"Sekans Hatası: Boss '{context_target.get('aciklama')}' için "
                            f"katman '{katman_label}' koordinatı bulunamadı."
                        )
                        return False
                    action_success = self.automator.click(katman_label, coord_override=coord)

                # -------------------------------------------------------
                elif action == "boss_seçimi":
                    if not context_target:
                        self.log("Sekans Hatası: 'boss_seçimi' için 'context_target' sağlanmadı.")
                        return False

                    boss_aciklama  = context_target.get("aciklama", "boss")
                    koordinat_ref  = context_target.get("koordinat_ref")

                    # v5.9: koordinat_ref çift format
                    if isinstance(koordinat_ref, dict) and "x" in koordinat_ref and "y" in koordinat_ref:
                        coord            = koordinat_ref
                        boss_coord_label = boss_aciklama
                    elif isinstance(koordinat_ref, str):
                        boss_coord_label = koordinat_ref
                        coord            = coord_map.get(boss_coord_label)
                        if not coord:
                            self.log(
                                f"Sekans Hatası: Boss '{boss_aciklama}' için "
                                f"koordinat etiketi ('{boss_coord_label}') bulunamadı."
                            )
                            return False
                    else:
                        self.log(
                            f"Sekans Hatası: Boss '{boss_aciklama}' için "
                            f"'koordinat_ref' geçersiz veya eksik."
                        )
                        return False

                    # v5.9: Görsel doğrulama — 'anchor' önce, 'gorsel_dogrulama' fallback
                    boss_verify_cfg = (
                        context_target.get("anchor")
                        or context_target.get("gorsel_dogrulama")
                    )
                    if boss_verify_cfg and isinstance(boss_verify_cfg, dict):
                        image_file  = boss_verify_cfg.get("image_file")
                        region_key  = boss_verify_cfg.get("region_key")
                        confidence  = float(boss_verify_cfg.get("confidence", 0.8))

                        if image_file and region_key:
                            search_region = self.ui_regions.get(region_key)
                            if search_region:
                                log_msg = f"Boss Anchor Doğrulama: '{image_file}' aranıyor..."
                                self.log(log_msg)
                                self.gui_queue.put(("set_status_line1", log_msg))
                                match = self.find_image_on_screen(image_file, search_region, confidence)
                                if match:
                                    log_msg = f"Boss Anchor Başarılı: '{image_file}' görüldü."
                                else:
                                    log_msg = (
                                        f"UYARI: Boss '{boss_aciklama}' anchor görseli "
                                        f"bulunamadı. Yine de tıklanıyor."
                                    )
                                self.log(log_msg)
                                self.gui_queue.put(("set_status_line1", log_msg))
                            else:
                                self.log(
                                    f"Boss Anchor Hatası: '{region_key}' bölgesi "
                                    f"'ui_regions' içinde bulunamadı. Doğrulama atlandı."
                                )
                        else:
                            self.log(
                                f"Boss Anchor UYARISI: '{boss_aciklama}' için "
                                f"doğrulama görseli/bölgesi eksik. Atlandı."
                            )

                    action_success = self.automator.click(boss_coord_label, coord_override=coord)

                # -------------------------------------------------------
                else:
                    self.log(f"Sekans Hatası: Bilinmeyen eylem türü '{action}'.")
                    return False

            except Exception as e:
                self.log(f"Sekans adımı ('{action}') yürütülürken hata: {e}")
                traceback.print_exc()
                return False

            if not action_success:
                self.log(
                    f"Sekans adımı BAŞARISIZ: '{action}' -> '{label or ''}'. "
                    f"Sekans iptal edildi."
                )
                return False

            if wait_ms > 0:
                if not self._interruptible_wait(wait_ms / 1000.0):
                    self.log("Sekans beklemesi sırasında bot durduruldu.")
                    return False

        self.log("Sekans başarıyla tamamlandı.")
        return True

    # ==========================================================================
    # Yardımcı görsel arama — spawn_check / victory için
    # ==========================================================================

    def _wait_for_visual(
        self,
        image_file: str,
        region_key: str,
        confidence: float,
        pre_window_sn: float,
        timeout_sn: float,
        poll_sn: float,
        label: str = "",
    ) -> bool:
        """
        Belirtilen görseli (pre_window_sn + timeout_sn) boyunca poll aralıklarıyla arar.
        Bulursa True, bulamazsa False döner.
        """
        region = self.ui_regions.get(region_key)
        if not region:
            self.log(f"[{label}] Görsel bekleme hatası: '{region_key}' bölgesi bulunamadı.")
            return False

        deadline = time.time() + pre_window_sn + timeout_sn
        self.log(
            f"[{label}] '{image_file}' bekleniyor "
            f"(max {pre_window_sn + timeout_sn:.1f}sn, güven {confidence})..."
        )
        while time.time() < deadline:
            if not self.running.is_set():
                return False
            try:
                if self.find_image_on_screen(image_file, region, confidence):
                    self.log(f"[{label}] '{image_file}' ekranda görüldü.")
                    return True
            except Exception:
                pass
            time.sleep(poll_sn)

        self.log(f"[{label}] '{image_file}' belirlenen sürede görülemedi.")
        return False

    # ==========================================================================
    # Spawn Check (v5.9)
    # ==========================================================================

    def _do_spawn_check(self, hedef: dict) -> bool:
        """
        YAML boss.spawn_check + spawn_check_defaults kullanarak boss doğuşunu bekler.
        Dönen değer: True = devam et, False = saldırıyı iptal et.
        """
        # YAML'dan genel ayarlar
        enabled  = self._spawn_check_defaults.get("enabled", True)
        required = self._spawn_check_defaults.get("required", False)

        # settings üzerinden de override edilebilir
        if "SPAWN_CONFIRM_ENABLED" in self.settings:
            enabled = bool(self.settings["SPAWN_CONFIRM_ENABLED"])

        if not enabled:
            return True

        spawn_check_cfg = hedef.get("spawn_check")
        if not spawn_check_cfg or not isinstance(spawn_check_cfg, dict):
            # Boss için spawn_check tanımlanmamış — devam et
            return True

        image_file  = spawn_check_cfg.get("image_file")
        if not image_file:
            return True

        region_key  = spawn_check_cfg.get(
            "region_key",
            self._spawn_check_defaults.get("region_key", "region_full_screen"),
        )
        confidence  = float(spawn_check_cfg.get(
            "confidence",
            self._spawn_check_defaults.get("confidence", 0.5),
        ))
        pre_window  = float(self._spawn_check_defaults.get("pre_window_sn",  2.0))
        timeout     = float(self._spawn_check_defaults.get("timeout_sn",     1.5))
        poll        = float(self._spawn_check_defaults.get("poll_interval_sn", 0.05))

        # settings override
        if "SPAWN_CONFIRM_PRE_WINDOW_SN" in self.settings:
            pre_window = float(self.settings["SPAWN_CONFIRM_PRE_WINDOW_SN"])
        if "SPAWN_CONFIRM_TIMEOUT_SN"    in self.settings:
            timeout    = float(self.settings["SPAWN_CONFIRM_TIMEOUT_SN"])
        if "SPAWN_CONFIRM_POLL_SN"       in self.settings:
            poll       = float(self.settings["SPAWN_CONFIRM_POLL_SN"])
        if "SPAWN_CONFIRM_REQUIRED"      in self.settings:
            required   = bool(self.settings["SPAWN_CONFIRM_REQUIRED"])

        confirmed = self._wait_for_visual(
            image_file, region_key, confidence, pre_window, timeout, poll,
            label=f"SpawnCheck-{hedef.get('aciklama', '?')}",
        )

        if not confirmed:
            if required:
                self.log(
                    f"SPAWN TEYIT BAŞARISIZ ({hedef['aciklama']}): "
                    f"'required=true' → Saldırı iptal."
                )
                return False
            else:
                self.log(
                    f"Spawn teyit edilemedi ({hedef['aciklama']}), "
                    f"'required=false' → Devam ediliyor."
                )
        return True

    # ==========================================================================
    # Victory Check (v5.9)
    # ==========================================================================

    def verify_boss_kill(self, boss: dict) -> bool:
        """
        v5.9: Önce boss başına 'victory' alanını okur;
        yoksa victory_check_defaults üzerinden varsayılan görseli arar;
        hiçbiri yoksa genel 'victory_icon.png' fallback kullanır.
        """
        boss_aciklama = boss.get("aciklama", "?")

        # --- 1. Boss başına victory alanı ---
        victory_cfg = boss.get("victory")
        if victory_cfg and isinstance(victory_cfg, dict):
            image_file  = victory_cfg.get("image_file")
            region_key  = victory_cfg.get(
                "region_key",
                self._victory_check_defaults.get("region_key", "region_full_screen"),
            )
            confidence  = float(victory_cfg.get(
                "confidence",
                self._victory_check_defaults.get("confidence", 0.4),
            ))
            timeout     = float(self._victory_check_defaults.get("timeout_sn",       1.5))
            poll        = float(self._victory_check_defaults.get("poll_interval_sn", 0.05))

            if image_file:
                found = self._wait_for_visual(
                    image_file, region_key, confidence, 0.0, timeout, poll,
                    label=f"Victory-{boss_aciklama}",
                )
                if found:
                    self.log(f"{boss_aciklama}: Zafer görseli teyit edildi ('{image_file}').")
                    return True
                else:
                    self.log(
                        f"{boss_aciklama}: Boss'a özgü zafer görseli "
                        f"('{image_file}') görülemedi. Fallback'e geçiliyor."
                    )

        # --- 2. Genel fallback: victory_icon.png ---
        try:
            region = self.ui_regions.get(
                "freeze_check_area", {"x": 800, "y": 400, "w": 300, "h": 200}
            )
            if self.find_image_on_screen("victory_icon.png", region, confidence=0.85):
                self.log(f"{boss_aciklama}: Genel zafer ikonu görüldü (fallback).")
                return True
        except Exception:
            pass

        # --- 3. Hiçbiri bulunamazsa varsayılan True ---
        self.log(f"{boss_aciklama}: Zafer görseli doğrulanamadı, varsayılan True döndürülüyor.")
        return True

    # ==========================================================================
    # Saldırı Prosedürü
    # ==========================================================================

    def _attack_boss(self, hedef: dict) -> bool:
        """
        Verilen boss hedefi için 'sequence_template'i çalıştırır.

        v5.9 adımları:
          1. Konum doğrulaması ve katmana geçiş
          2. (aynı katta ise) area_check ile konum teyidi
          3. spawn_timestamp bekleme
          4. _do_spawn_check() — boss doğuşunu görsel olarak teyit et
          5. Saldırı ('z' + 'a')
          6. verify_boss_kill() — zafer görseli ile kesildiğini doğrula
        """
        spawn_timestamp = hedef.get("spawn_time")
        if not spawn_timestamp:
            self.log(f"HATA: {hedef['aciklama']} için spawn_time yok.")
            return False

        self.log(f"Saldırıya hazırlanılıyor: {hedef['aciklama']}")

        target_location_name = (
            "KATMAN_1" if "katman_1" in hedef.get("katman_id", "") else "KATMAN_2"
        )

        # -------- Katman değişimi gerekiyorsa ---------------------------------
        if self.location_manager.get_region_name().upper() != target_location_name:
            self.log(
                f"Konum farklı ({self.location_manager.get_region_name()} "
                f"-> {target_location_name}). Harita değiştiriliyor."
            )
            if not self.automator.return_to_exp_farm(force_restart_if_failed=True):
                self.log("Boss katmanına girmeden önce Exp Farm'a dönülemedi. Saldırı iptal.")
                return False

            sekans_basarili = self._run_sequence(
                self.boss_sequence_template,
                self.demon_boss_cfg.get("coordinates", {}),
                context_target=hedef,
            )
            if not sekans_basarili:
                self.log("Boss katmanına giriş sekansı başarısız.")
                self.automator.return_to_exp_farm()
                return False

            self.location_manager.set_current_location_by_name(target_location_name)

        # -------- Zaten aynı kattayız ----------------------------------------
        else:
            self.log("Konum zaten doğru (aynı kat).")

            # v5.9: area_check — doğru alanda mıyız?
            area_check_cfg = hedef.get("area_check")
            if area_check_cfg and isinstance(area_check_cfg, dict):
                ac_image      = area_check_cfg.get("image_file")
                ac_region_key = area_check_cfg.get("region_key")
                ac_confidence = float(area_check_cfg.get("confidence", 0.8))
                ac_region     = self.ui_regions.get(ac_region_key) if ac_region_key else None

                if ac_image and ac_region:
                    self.log(f"Alan Kontrolü: '{ac_image}' aranıyor...")
                    ac_match = self.find_image_on_screen(ac_image, ac_region, ac_confidence)
                    if ac_match:
                        self.log(f"Alan Kontrolü Başarılı: '{ac_image}' görüldü.")
                    else:
                        self.log(
                            f"UYARI: Alan kontrolü başarısız ('{ac_image}'). "
                            f"Yine de devam ediliyor."
                        )

            # Kısa sekans: sadece boss listesini aç ve boss'a tıkla
            kisa_sekans = [
                {"action": "click",      "label": "boss_list_ac", "wait_ms": 200},
                {"action": "boss_seçimi",                         "wait_ms": 200},
            ]
            sekans_basarili = self._run_sequence(
                kisa_sekans,
                self.demon_boss_cfg.get("coordinates", {}),
                context_target=hedef,
            )
            if not sekans_basarili:
                self.log(f"{hedef['aciklama']} koordinatına tıklanamadı (kısa sekans).")
                return False

        # -------- Otomatik savaş tuşu -----------------------------------------
        try:
            self.automator.press_key("z", label="boss_attack_auto_battle")
        except Exception as e:
            self.log(f"'z' tuşu basma hatası: {e}")
            return False

        # -------- Doğuş zamanını bekle ----------------------------------------
        wait_for_spawn = spawn_timestamp - time.time()
        if wait_for_spawn > 0:
            self.log(f"Doğuş için {wait_for_spawn:.2f} sn bekleniyor...")
            if not self._interruptible_wait(wait_for_spawn):
                self.log("Bekleme sırasında otomasyon durduruldu. Saldırı iptal.")
                return False

        # -------- v5.9: Spawn Check (görsel doğuş teyidi) ---------------------
        if not self._do_spawn_check(hedef):
            return False   # required=true ise saldırıyı iptal et

        # -------- Oyun kapandı mı? --------------------------------------------
        if not self.is_game_running():
            self.log("Saldırı öncesi oyun kapandı.")
            return False

        self.log("Saldırı komutu gönderildi. Yeni spawn hesaplanıyor.")
        time.sleep(self.settings.get("SLEEP_SHORT", 2))
        attack_start_time = time.time()

        try:
            self.automator.press_key("a", label="boss_attack_main")
        except Exception as e:
            self.log(f"'a' tuşu basma hatası: {e}")

        # -------- v5.9: Victory Check -----------------------------------------
        if self.verify_boss_kill(hedef):
            post_attack_wait = self.settings.get("POST_ATTACK_WAIT_SN", 30)
            self.log(f"Ganimet toplamak için {post_attack_wait} sn bekleniyor...")
            self.loot_collection_ends_at = time.time() + post_attack_wait
            self._recalculate_all_spawn_times_after_attack(hedef, attack_start_time)
            self._interruptible_wait(post_attack_wait)
            return True
        else:
            self.log(
                f"{hedef['aciklama']} kesilemedi/doğrulanamadı. "
                f"60 sn sonra tekrar denenecek."
            )
            self._set_spawn_time(hedef["aciklama"], 60, source="retry")
            return False

    # ==========================================================================
    # Spawn zamanı hesaplama
    # ==========================================================================

    def _recalculate_all_spawn_times_after_attack(
        self, killed_boss: dict, attack_start_time: float
    ):
        self.log("--- Tüm Boss Zamanlamaları Güncelleniyor ---")
        periyot_saniye = killed_boss["periyot_saat"] * 3600
        gecikme_gercek = max(0, attack_start_time - killed_boss["spawn_time"])
        yeni_spawn_A   = killed_boss["spawn_time"] + periyot_saniye + gecikme_gercek
        self.log(
            f"Kesilen Boss ({killed_boss['aciklama']}): "
            f"Gecikme {gecikme_gercek:.1f}s. "
            f"Yeni: {datetime.fromtimestamp(yeni_spawn_A).strftime('%H:%M:%S')}"
        )
        self._set_spawn_time(
            killed_boss["aciklama"],
            yeni_spawn_A - time.time(),
            source="automation",
        )

        post_attack_wait  = self.settings.get("POST_ATTACK_WAIT_SN", 30)
        loot_finish_time  = attack_start_time + post_attack_wait

        upcoming_targets = [
            h for aciklama, h in self.bosslar.items()
            if h.get("spawn_time")
            and h["spawn_time"] > loot_finish_time
            and aciklama != killed_boss["aciklama"]
        ]
        upcoming_targets.sort(key=lambda h: (h["spawn_time"], h["boss_id"]))

        for next_boss in upcoming_targets:
            if next_boss["katman_id"] != killed_boss["katman_id"]:
                continue
            yurume_suresi     = self._get_walk_time(
                str(killed_boss["aciklama"]), str(next_boss["aciklama"])
            )
            time_until_spawn  = next_boss["spawn_time"] - loot_finish_time
            if yurume_suresi > time_until_spawn:
                gecikme_kaymasi = yurume_suresi - time_until_spawn
                yeni_spawn_B    = next_boss["spawn_time"] + gecikme_kaymasi
                self.log(
                    f"KAYDIRMA ({next_boss['aciklama']}): "
                    f"Yürüme {yurume_suresi:.1f}s > Kalan {time_until_spawn:.1f}s "
                    f"-> +{gecikme_kaymasi:.1f}s"
                )
                self._set_spawn_time(
                    next_boss["aciklama"],
                    yeni_spawn_B - time.time(),
                    source="automation",
                )
            else:
                self.log(
                    f"Zamanlama OK ({next_boss['aciklama']}): "
                    f"{yurume_suresi:.1f}s <= {time_until_spawn:.1f}s"
                )

    # ------------------------------------------------------------------
    def _get_walk_time(self, boss_a: str, boss_b: str) -> int:
        if boss_a == boss_b:
            return 0
        try:
            t = self.walk_times.get(boss_a, {}).get(boss_b)
            if t is not None:
                return int(t)
            t = self.walk_times.get(boss_b, {}).get(boss_a)
            if t is not None:
                return int(t)
        except (TypeError, ValueError):
            pass
        default = 40
        self.log(
            f"UYARI: Yürüme süresi bulunamadı ({boss_a} <-> {boss_b}). "
            f"Varsayılan {default}sn."
        )
        return default

    # ------------------------------------------------------------------
    def _set_spawn_time(self, boss_aciklama: str, kalan_saniye: float, source: str = "config"):
        hedef = self.bosslar.get(str(boss_aciklama))
        if not hedef:
            self.log(f"UYARI: Spawn zamanı ayarlanamadı, boss bulunamadı: {boss_aciklama}")
            return
        hedef["spawn_time"] = time.time() + max(0.0, kalan_saniye)
        hedef["source"]     = source
        self.gui_queue.put(("set_spawn", (boss_aciklama, hedef["spawn_time"], source)))

    # ==========================================================================
    # Yardımcı / Ortak Metodlar
    # ==========================================================================

    def _log_to_file_and_message(self, msg: str):
        if hasattr(self, "gui_queue") and self.gui_queue:
            self.gui_queue.put(("log", msg))
        log_to_file(msg)

    def log(self, msg: str):
        self._log_to_file_and_message(msg)

    # ------------------------------------------------------------------
    def is_game_running(self) -> bool:
        if self._pid and psutil.pid_exists(self._pid):
            try:
                p = psutil.Process(self._pid)
                return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._pid = None
        try:
            for proc in psutil.process_iter(["pid", "name"]):
                if proc.info["name"].lower() == self.process_name.lower():
                    self._pid = proc.pid
                    return True
        except Exception as e:
            self.log(f"Oyun prosesi aranırken hata: {e}")
        return False

    # ------------------------------------------------------------------
    def find_image_on_screen(
        self, template_name: str, region: dict, confidence: float = 0.7
    ) -> tuple | None:
        template_gray = self.get_cached_image(template_name)
        if template_gray is None:
            return None

        monitor = {
            "top":    _safe_int(region.get("y", 0),    0),
            "left":   _safe_int(region.get("x", 0),    0),
            "width":  _safe_int(region.get("w", 1920), 1920),
            "height": _safe_int(region.get("h", 1080), 1080),
        }

        th, tw = template_gray.shape[:2]
        mh, mw = monitor["height"], monitor["width"]
        if th <= 0 or tw <= 0 or mh <= 0 or mw <= 0 or th > mh or tw > mw:
            return None

        try:
            with mss.mss() as sct:
                screenshot_bgra = sct.grab(monitor)
                screenshot_gray = cv2.cvtColor(
                    np.array(screenshot_bgra), cv2.COLOR_BGRA2GRAY
                )
            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= confidence:
                center_x = monitor["left"] + max_loc[0] + tw // 2
                center_y = monitor["top"]  + max_loc[1] + th // 2
                return (center_x, center_y, max_val)
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    def safe_exit_map(self, confidence: float = 0.8) -> bool:
        """
        Mevcut konuma göre tanımlanmış çıkış butonlarını arar ve tıklar.
        """
        current_location_name = self.location_manager.get_region_name().upper()
        exit_buttons_list     = self.exit_buttons_config.get(current_location_name, [])

        if not exit_buttons_list:
            self.log(
                f"({current_location_name}) için özel çıkış tanımı yok. "
                f"'DEFAULT' çıkış listesi aranıyor..."
            )
            exit_buttons_list = self.exit_buttons_config.get("DEFAULT", [])

        if not exit_buttons_list:
            self.log(
                f"({current_location_name}) için özel VEYA 'DEFAULT' çıkış "
                f"butonu tanımı bulunamadı."
            )
            return False

        default_region = self.ui_regions.get("exit_map_area")
        match_result   = None
        found_button_config = None

        def _try_buttons(button_list):
            nonlocal match_result, found_button_config
            for button_config in button_list:
                img_file = button_config.get("image_file")
                if not img_file:
                    continue
                region_key    = button_config.get("region_key")
                search_region = (
                    self.ui_regions.get(region_key)
                    if region_key
                    else default_region
                )
                if not search_region:
                    self.log(f"safe_exit_map Hatası: '{img_file}' için bölge bulunamadı.")
                    continue
                sc = float(button_config.get("confidence") or confidence)
                match = self.find_image_on_screen(img_file, search_region, sc)
                if match:
                    match_result        = match
                    found_button_config = button_config
                    self.log(f"Çıkış butonu '{img_file}' bulundu.")
                    return True
            return False

        _try_buttons(exit_buttons_list)

        # Fallback: KATMAN_1/2'de boss listesi açık değilse
        needs_fallback = (
            not match_result
            and current_location_name in ("KATMAN_1", "KATMAN_2")
        )
        if needs_fallback:
            self.log("Çıkış butonu görünür değil, boss listesine tıklanıyor (fallback)...")
            if self.automator.click("boss_list_ac", check_game=True):
                time.sleep(0.5)
                _try_buttons(exit_buttons_list)

        if match_result and found_button_config:
            click_x, click_y, max_val = match_result
            button_label = found_button_config.get("image_file", "unknown_exit_button")

            manual_override = False
            with MANUAL_CLICK_LOCK:
                if last_manual_click_time is not None:
                    if (time.monotonic() - last_manual_click_time) <= 3.0:
                        manual_override = True

            if manual_override:
                self.log(
                    f"Kullanıcı müdahalesi algılandı ({button_label}). "
                    f"Bot tıklamıyor. (güven: {max_val:.2f})"
                )
                return True
            else:
                try:
                    self.automator.click(
                        label=f"exit_{current_location_name}_{button_label}",
                        coord_override={"x": click_x, "y": click_y},
                        check_game=True,
                    )
                    self.log(f"Harita çıkışı '{button_label}' tıklandı (güven: {max_val:.2f})")
                    return True
                except Exception as e:
                    self.log(f"Harita çıkışına ({button_label}) tıklanırken hata: {e}")
                    return False
        else:
            manual_override_found = False
            with MANUAL_CLICK_LOCK:
                if last_manual_click_time is not None:
                    if (time.monotonic() - last_manual_click_time) <= 3.0:
                        manual_override_found = True
            if manual_override_found:
                self.log(
                    f"({current_location_name}) Çıkış butonu bulunamadı "
                    f"ama manuel tıklama algılandı -> Başarılı varsayılıyor."
                )
                return True
            else:
                tried = [b.get("image_file", "N/A") for b in exit_buttons_list]
                self.log(
                    f"({current_location_name}) için uygun çıkış butonu "
                    f"bulunamadı (Denenenler: {tried})."
                )
                return False

    # ------------------------------------------------------------------
    def _interruptible_wait(self, seconds: float, step: float = 0.5) -> bool:
        """self.running set olduğu sürece bekler; kesilirse False döner."""
        end_time = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end_time:
            if not self.running.is_set():
                return False
            remaining      = end_time - time.monotonic()
            sleep_duration = min(step, max(0.05, remaining))
            time.sleep(sleep_duration)
        return self.running.is_set()

    # ==========================================================================
    # Thread'ler
    # ==========================================================================

    def _boss_automation_thread(self):
        self.log("Boss otomasyon döngüsü başlatıldı.")

        while True:
            try:
                if not self.running.is_set() or self.paused:
                    time.sleep(0.5)
                    continue

                if self.active_event is not None:
                    time.sleep(2)
                    continue

                # Konum senkronizasyonu
                if (
                    not self.action_lock.locked()
                    and self.attacking_target_aciklama is None
                    and self.active_event is None
                ):
                    is_in_exp_farm         = self.location_manager.get_region_name() == "EXP_FARM"
                    manual_click_happened  = self.manual_click_detected.is_set()

                    if (not is_in_exp_farm) or (is_in_exp_farm and manual_click_happened):
                        if manual_click_happened:
                            self.manual_click_detected.clear()
                        visual_val = self.location_manager._get_current_visual_location_value()
                        if (
                            visual_val != self.location_manager.NAME_TO_VALUE.get("UNKNOWN", -1)
                            and self.location_manager.current_location_value != visual_val
                        ):
                            self.log(
                                f"KONUM SENKRON: "
                                f"'{self.location_manager.get_region_name()}' "
                                f"-> '{self.location_manager.get_location_name(visual_val)}'"
                            )
                            self.location_manager.current_location_value = visual_val

                now           = time.time()
                ready_targets = []
                for aciklama, hedef in self.bosslar.items():
                    spawn_time  = hedef.get("spawn_time")
                    head_start  = hedef.get("head_start_saniye", 0)
                    if spawn_time and (now >= spawn_time - head_start):
                        ready_targets.append(hedef)

                if not ready_targets:
                    time.sleep(self.settings.get("MAIN_LOOP_SLEEP_SN", 1.0))
                    continue

                current_target = sorted(ready_targets, key=lambda h: h["spawn_time"])[0]
                current_target_aciklama = current_target["aciklama"]

                with self.action_lock:
                    self.attacking_target_aciklama = current_target_aciklama
                    self.loot_collection_ends_at   = 0
                    self.log(
                        f"Sıradaki hedef: {current_target_aciklama}. "
                        f"Saldırı prosedürü başlıyor."
                    )
                    action_success = self._attack_boss(current_target)

                if not action_success:
                    self.log(
                        f"Saldırı başarısız: {current_target_aciklama}. "
                        f"Exp Farm'a dönülüyor."
                    )
                    with self.action_lock:
                        self.automator.return_to_exp_farm(force_restart_if_failed=True)
                    self.attacking_target_aciklama = None
                    continue

                now_after_attack = time.time()
                upcoming_targets = sorted(
                    [
                        h for h in self.bosslar.values()
                        if h.get("spawn_time") and h["spawn_time"] > now_after_attack
                    ],
                    key=lambda h: (h["spawn_time"], h["boss_id"]),
                )
                next_target = upcoming_targets[0] if upcoming_targets else None

                should_stay = False
                if next_target:
                    is_same_map      = next_target["katman_id"] == current_target["katman_id"]
                    post_attack_wait = self.settings.get("POST_ATTACK_WAIT_SN", 30)
                    loot_finish_time = now_after_attack + post_attack_wait
                    time_to_next     = next_target["spawn_time"] - loot_finish_time
                    threshold_sn     = self.settings.get("BOSS_SWITCH_THRESHOLD_SN", 91)
                    is_soon          = time_to_next < threshold_sn

                    if is_same_map and is_soon:
                        should_stay = True
                        self.log(
                            f"Stratejik bekleme: Sonraki boss ({next_target['aciklama']}) "
                            f"aynı katta ve {int(time_to_next)}sn içinde."
                        )
                    else:
                        if not is_same_map:
                            self.log("Sonraki boss farklı haritada. Çıkış.")
                        elif not is_soon:
                            self.log(f"Sonraki boss'a çok var ({int(time_to_next)}sn). Çıkış.")

                if not should_stay:
                    self.log("Bekleme kriterleri karşılanmadı. Exp Farm'a dönülüyor.")
                    with self.action_lock:
                        self.automator.return_to_exp_farm(force_restart_if_failed=True)

                self.attacking_target_aciklama = None
                self.loot_collection_ends_at   = 0

            except Exception as e:
                self.log(f"Boss otomasyon thread hatası: {e}")
                traceback.print_exc()
                self.attacking_target_aciklama = None
                time.sleep(5)

    # ------------------------------------------------------------------
    def _game_monitor_thread(self):
        """Oyunun çalışıp çalışmadığını ve donup donmadığını izler."""
        self.log("Oyun Monitör thread'i başlatıldı.")
        freeze_check_interval   = self.settings.get("FREEZE_CHECK_INTERVAL_SN", 15)
        max_freeze_duration     = self.settings.get("MAX_FREEZE_DURATION_SN", 60)
        freeze_check_region_cfg = self.ui_regions.get("freeze_check_area", {})
        if not freeze_check_region_cfg:
            self.log("UYARI: 'freeze_check_area' tanımlı değil. Donma tespiti pasif.")

        time.sleep(5)

        while True:
            if not self.running.is_set():
                time.sleep(1.5)
                continue

            try:
                if not self.is_game_running():
                    self.log("Oyun prosesi bulunamadı, yeniden başlatma tetikleniyor...")
                    with self.action_lock:
                        if not self.is_game_running() and self.running.is_set():
                            self._handle_game_restart()
                    time.sleep(15)
                    continue

                now      = time.monotonic()
                is_busy  = (
                    not self.paused
                    and (
                        self.active_event is not None
                        or self.attacking_target_aciklama is not None
                    )
                )

                if (
                    is_busy
                    and freeze_check_region_cfg
                    and (now - self._last_screenshot_time > freeze_check_interval)
                ):
                    try:
                        r = freeze_check_region_cfg
                        monitor = {
                            "top":    _safe_int(r.get("y", 400), 400),
                            "left":   _safe_int(r.get("x", 800), 800),
                            "width":  _safe_int(r.get("w", 300), 300),
                            "height": _safe_int(r.get("h", 200), 200),
                        }
                        with mss.mss() as sct:
                            shot = sct.grab(monitor)
                            arr  = np.array(shot, dtype=np.uint8)
                            current_checksum = int(np.sum(arr) % 1000003)

                        self._last_screenshot_time = now

                        if (
                            self._last_screenshot_checksum is not None
                            and current_checksum == self._last_screenshot_checksum
                        ):
                            self._freeze_count += 1
                            freeze_duration = self._freeze_count * freeze_check_interval
                            self.log(
                                f"Donma şüphesi: Ekran {freeze_duration:.1f} saniyedir değişmedi."
                            )
                            if freeze_duration >= max_freeze_duration:
                                self.log(
                                    f"MAKSİMUM DONMA SÜRESİ ({max_freeze_duration}sn) AŞILDI! "
                                    f"Oyun yeniden başlatılıyor."
                                )
                                self.restart_game()
                                self._freeze_count             = 0
                                self._last_screenshot_checksum = None
                                time.sleep(10)
                        else:
                            self._freeze_count             = 0
                            self._last_screenshot_checksum = current_checksum

                    except Exception as e:
                        self.log(f"Donma kontrolü sırasında hata: {e}")
                        self._last_screenshot_checksum = None
                        self._freeze_count             = 0

                time.sleep(self.settings.get("MAIN_LOOP_SLEEP_SN", 1.0))

            except Exception as e:
                self.log(f"Oyun monitör thread'inde K R İ T İ K hata: {e}")
                traceback.print_exc()
                time.sleep(15)

    # ------------------------------------------------------------------
    def _handle_game_restart(self):
        self.log("Oyun yeniden başlatma prosedürü başlıyor...")
        self._freeze_count             = 0
        self._last_screenshot_checksum = None
        self._last_screenshot_time     = time.monotonic()

        if not self._start_game():
            self.log("Oyun başlatılamadığı için yeniden başlatma durdu.")
            return

        self.log("Oyun başlatıldı/login denendi. Konum tespiti için 5sn bekleniyor...")
        if not self._interruptible_wait(5):
            return

        self.location_manager.current_location_value = (
            self.location_manager._get_current_visual_location_value()
        )
        self.log(
            f"Yeniden başlatma sonrası konum tespiti: "
            f"{self.location_manager.get_region_name()}"
        )

        if self.active_event:
            event_name  = self.active_event.get("name", "Bilinmeyen Etkinlik")
            now         = datetime.now()
            event       = self.active_event
            is_still_due = (
                (event["days"] is None or now.weekday() in event["days"])
                and (event["start_time"] <= now.time() < event["end_time"])
            )

            if is_still_due:
                self.log(f"Etkinlik '{event_name}' hala devam ediyor. Yeniden girilecek.")
                try:
                    action_type = event.get("action_type", "static_sequence")
                    success     = False
                    if action_type == "static_sequence":
                        action_sequence = event.get("action_sequence", [])
                        success = self._run_sequence(
                            action_sequence,
                            self.timed_events_cfg.get("coordinates", {}),
                        )
                    if success:
                        self.log(f"Etkinliğe '{event_name}' başarıyla yeniden girildi.")
                        self.location_manager.set_current_location_by_name(
                            event["location_name_on_enter"]
                        )
                    else:
                        self.log(f"Etkinliğe '{event_name}' yeniden girilemedi.")
                        self.active_event = None
                        self.automator.return_to_exp_farm(force_restart_if_failed=False)
                except Exception as e:
                    self.log(f"Etkinlik '{event_name}' kurtarma sırasında hata: {e}")
                    self.active_event = None
                    self.automator.return_to_exp_farm(force_restart_if_failed=False)
            else:
                self.log(f"Etkinlik '{event_name}' süresi dolmuş. Exp Farm'a dönülüyor.")
                self.active_event = None
                if self.location_manager.get_region_name() != "EXP_FARM":
                    self.automator.return_to_exp_farm(force_restart_if_failed=False)
        else:
            if self.location_manager.get_region_name() != "EXP_FARM":
                self.log("Mevcut konum Exp Farm değil, geri dönülüyor...")
                self.automator.return_to_exp_farm(force_restart_if_failed=False)

        self.location_manager.set_current_location_by_name("EXP_FARM")
        self.log("Yeniden başlatma prosedürü tamamlandı.")
        self._freeze_count             = 0
        self._last_screenshot_checksum = None
        self._last_screenshot_time     = time.monotonic()

    # ------------------------------------------------------------------
    def _start_game(self) -> bool:
        if self.is_game_running():
            self.log("Oyun zaten çalışıyor, tekrar başlatılmıyor.")
            return True
        try:
            game_dir         = os.path.dirname(self.game_path)
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            proc = subprocess.Popen(
                self.game_path,
                cwd=game_dir,
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
                close_fds=True,
                shell=False,
            )
            self._pid = proc.pid
            self.log(f"Oyun başlatıldı (Bağımsız Mod): {self.game_path}")
            if not self._interruptible_wait(5):
                return False
            self.automator.login_to_game()
            return True
        except Exception as e:
            self.log(f"Oyun başlatılırken hata: {e}")
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    def restart_game(self):
        self.log("Oyun yeniden başlatma komutu işleniyor...")
        time.sleep(1)
        pid_to_kill = self._pid
        if not pid_to_kill or not psutil.pid_exists(pid_to_kill):
            try:
                for proc in psutil.process_iter(["pid", "name"]):
                    if proc.info["name"].lower() == self.process_name.lower():
                        pid_to_kill = proc.pid
                        break
            except Exception:
                pass

        if pid_to_kill and psutil.pid_exists(pid_to_kill):
            try:
                p = psutil.Process(pid_to_kill)
                p.kill()
                self.log(f"Oyun prosesi (PID: {pid_to_kill}) sonlandırıldı.")
                self._pid = None
                time.sleep(2)
            except Exception as e:
                self.log(f"Oyun prosesi (PID: {pid_to_kill}) sonlandırılamadı: {e}")
                self._pid = None
        else:
            self.log("Aktif oyun prosesi bulunamadı, sonlandırma atlandı.")
            self._pid = None

    # ------------------------------------------------------------------
    def _popup_monitor_thread(self):
        self.log("Pop-up izleyici thread'i başlatıldı.")

        popup_configs       = self.popup_config.get("popups_to_close", [])
        region_key          = self.popup_config.get("region_key")
        search_region_config = self.ui_regions.get(region_key) if region_key else None
        check_interval      = self.popup_config.get("check_interval_sn", 20)

        if not search_region_config or not all(
            k in search_region_config for k in ("x", "y", "w", "h")
        ):
            self.log(
                f"Pop-up UYARI: 'region_key' ({region_key}) config'de eksik "
                f"veya 'ui_regions' içinde bulunamadı. İzleyici pasif."
            )
            return
        if not popup_configs:
            self.log("Kapatılacak pop-up tanımı bulunamadı. İzleyici pasif.")
            return

        loaded_templates = []
        for cfg in popup_configs:
            img_file = cfg.get("image_file")
            if not img_file:
                continue
            img = self.get_cached_image(img_file)
            if img is not None:
                loaded_templates.append({
                    "template":   img,
                    "confidence": float(cfg.get("confidence", 0.7)),
                    "name":       img_file,
                })

        if not loaded_templates:
            self.log("Popup monitor için hiçbir resim yüklenemedi, thread durduruldu.")
            return

        while True:
            if not self.running.is_set() or self.paused or self.action_lock.locked():
                time.sleep(check_interval / 2)
                continue

            try:
                found_and_clicked = False
                monitor_for_mss   = {
                    "top":    search_region_config.get("y", 0),
                    "left":   search_region_config.get("x", 0),
                    "width":  search_region_config.get("w", 1200),
                    "height": search_region_config.get("h", 600),
                }

                with mss.mss() as sct:
                    screenshot_color = sct.grab(monitor_for_mss)
                    screenshot_gray  = cv2.cvtColor(
                        np.array(screenshot_color, dtype=np.uint8), cv2.COLOR_BGRA2GRAY
                    )

                for item in loaded_templates:
                    template_gray = item["template"]
                    confidence    = item["confidence"]
                    template_name = item["name"]

                    th, tw = template_gray.shape[:2]
                    sh, sw = screenshot_gray.shape[:2]
                    if th > sh or tw > sw:
                        continue

                    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val >= confidence:
                        click_x = search_region_config.get("x", 0) + max_loc[0] + tw // 2
                        click_y = search_region_config.get("y", 0) + max_loc[1] + th // 2
                        self.log(
                            f"Beklenmedik pencere/buton tespit edildi ('{template_name}'). "
                            f"Kapatılıyor..."
                        )
                        try:
                            self.automator.click(
                                label=f"popup_close_{template_name}",
                                coord_override={"x": click_x, "y": click_y},
                                check_game=True,
                            )
                            found_and_clicked = True
                            time.sleep(2.0)
                            break
                        except Exception as click_err:
                            self.log(f"Pop-up kapatma tıklaması sırasında hata: {click_err}")
                            found_and_clicked = True
                            time.sleep(1.0)
                            break

                time.sleep(1.0 if found_and_clicked else check_interval)

            except Exception as e:
                self.log(f"Pop-up izleyici thread'inde hata: {e}")
                time.sleep(check_interval * 2)

    # ------------------------------------------------------------------
    def _pvp_defense_monitor_thread(self):
        if not self.pvp_config.get("enabled", False):
            return

        self.log("PVP Savunma Monitörü başlatılıyor...")
        region_config       = self.pvp_config.get("hp_orb_region")
        template_name       = self.pvp_config.get("hp_damaged_template")
        confidence          = float(self.pvp_config.get("hp_damage_confidence", 0.85))
        skill_rotation      = self.pvp_config.get("skill_rotation", [])
        check_interval      = float(self.pvp_config.get("check_interval_sn", 0.7))
        cooldown_after_attack = float(self.pvp_config.get("cooldown_after_attack_sn", 3.0))

        if not isinstance(region_config, dict) or not all(
            k in region_config for k in ("x", "y", "w", "h")
        ):
            self.log("PVP Savunma UYARI: 'hp_orb_region' config'de eksik. Thread pasif.")
            return
        if not template_name or not skill_rotation:
            self.log(
                "PVP Savunma UYARI: 'hp_damaged_template' veya 'skill_rotation' eksik. "
                "Thread pasif."
            )
            return

        damage_template = self.get_cached_image(template_name)
        if damage_template is None:
            self.log(
                f"PVP Savunma HATA: Hasar şablonu '{template_name}' "
                f"önbellekte bulunamadı. Thread pasif."
            )
            return

        self.log(f"PVP Savunma aktif. Skiller: {skill_rotation}, Aralık: {check_interval}sn")
        was_under_attack = False

        while True:
            if not self.running.is_set() or self.paused:
                time.sleep(check_interval * 2)
                was_under_attack = False
                continue

            try:
                monitor_for_mss = {
                    "top":    region_config.get("y", 0),
                    "left":   region_config.get("x", 0),
                    "width":  region_config.get("w", 100),
                    "height": region_config.get("h", 100),
                }
                with mss.mss() as sct:
                    screenshot_color = sct.grab(monitor_for_mss)
                    screenshot_gray  = cv2.cvtColor(
                        np.array(screenshot_color, dtype=np.uint8), cv2.COLOR_BGRA2GRAY
                    )

                result = cv2.matchTemplate(screenshot_gray, damage_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                is_under_attack  = (max_val >= confidence)

                if is_under_attack:
                    if not was_under_attack:
                        self.log(f"SALDIRI ALTINDA! (Güven: {max_val:.2f}). Karşı saldırı...")
                        was_under_attack = True
                    try:
                        for skill in skill_rotation:
                            pyautogui.press(skill)
                            self.automator.press_key(skill, label=f"pvp_defense_{skill}")
                            time.sleep(0.1)
                        time.sleep(cooldown_after_attack)
                    except Exception as skill_err:
                        self.log(f"PVP skill kullanılırken hata: {skill_err}")
                else:
                    if was_under_attack:
                        self.log("PVP Savunma: Tehlike geçti.")
                        was_under_attack = False
                    time.sleep(check_interval)

            except Exception as e:
                self.log(f"PVP Savunma thread'inde hata: {e}")
                was_under_attack = False
                time.sleep(check_interval * 10)

    # ==========================================================================
    # Başlat / Durdur / Çalıştır
    # ==========================================================================

    def start(self):
        """Otomasyonu başlatır veya duraklatılmışsa devam ettirir."""
        if self.running.is_set() and not self.paused:
            self.log("Otomasyon zaten çalışıyor.")
            return

        if self.bosslar and not any(h.get("spawn_time") for h in self.bosslar.values()):
            try:
                if self.root and self.root.winfo_exists():
                    if not messagebox.askyesno(
                        "Zaman Yok",
                        "Hiçbir boss için başlangıç zamanı girilmedi.\nBaşlatılsın mı?",
                        parent=self.root,
                    ):
                        self.log("Başlatma iptal edildi (kullanıcı onayı alınamadı).")
                        return
            except Exception:
                pass

        if not self._threads_started:
            self.log("Arka plan thread'leri başlatılıyor...")
            try:
                threading.Thread(
                    target=self._game_monitor_thread,
                    daemon=True, name="GameMonitorThread",
                ).start()
                threading.Thread(
                    target=self._boss_automation_thread,
                    daemon=True, name="BossAutomationThread",
                ).start()
                threading.Thread(
                    target=self.event_manager.event_monitor_thread,
                    daemon=True, name="EventMonitorThread",
                ).start()
                threading.Thread(
                    target=self._popup_monitor_thread,
                    daemon=True, name="PopupMonitorThread",
                ).start()
                if self.pvp_config.get("enabled", False):
                    threading.Thread(
                        target=self._pvp_defense_monitor_thread,
                        daemon=True, name="PvPDefenseThread",
                    ).start()
                self._threads_started = True
            except Exception as e:
                self.log(f"KRİTİK HATA: Arka plan thread'leri başlatılamadı: {e}")
                traceback.print_exc()
                self.initialized = False
                try:
                    if self.root:
                        self.root.quit()
                except Exception:
                    pass
                return

        self.running.set()
        self.paused = False
        self.log("Otomasyon başlatıldı/devam ettirildi.")

    # ------------------------------------------------------------------
    def stop(self):
        """Otomasyonu duraklatır."""
        if self.paused or not self.running.is_set():
            self.log("Otomasyon zaten duraklatılmış veya çalışmıyor.")
            return
        self.paused = True
        self.log("Otomasyon duraklatıldı.")

    # ------------------------------------------------------------------
    def run(self):
        """Ana program döngüsünü (GUI) başlatır."""
        if not self.initialized:
            self.log("Bot düzgün başlatılamadığı için GUI çalıştırılamıyor.")
            try:
                messagebox.showerror("Başlatma Hatası", "Bot başlatılamadı.\nLog dosyasını kontrol edin.")
            except Exception:
                pass
            try:
                if self.root:
                    self.root.destroy()
            except Exception:
                pass
            return

        try:
            if hasattr(self, "gui_manager") and self.gui_manager:
                self.gui_manager.create_gui()
                self.log("GUI mainloop başlatılıyor...")
                self.root.mainloop()
                self.log("GUI mainloop sona erdi.")
        except Exception as e:
            self.log(f"Ana 'run' metodu sırasında hata: {e}")
            traceback.print_exc()
        finally:
            self.log("Bot kapatılıyor...")
            self.running.clear()


# =============================================================================
# 7) Program Başlangıcı
# =============================================================================

if __name__ == "__main__":
    root = None
    try:
        log_to_file("Program başlatılıyor (v5.9)...")
        root = tk.Tk()

        app = LoABot(root)

        if app.initialized:
            app.run()
        else:
            log_to_file("Bot başlatılamadığı için program sonlandırılıyor.")
            if root:
                try:
                    root.destroy()
                except Exception:
                    pass

        log_to_file("Program normal şekilde sonlandırıldı.")

    except Exception as e:
        error_msg = f"Program çalıştırılırken K R İ T İ K bir hata oluştu: {e}"
        print(error_msg)
        log_to_file(error_msg)
        traceback.print_exc(file=open(LOG_FILE, "a"))
        try:
            messagebox.showerror(
                "Kritik Hata",
                f"Program beklenmedik bir hatayla karşılaştı:\n{e}\n"
                f"Log dosyasını ({LOG_FILE}) kontrol edin.",
            )
        except Exception:
            pass
        finally:
            if root:
                try:
                    root.destroy()
                except Exception:
                    pass