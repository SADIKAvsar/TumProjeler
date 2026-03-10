# -*- coding: utf-8 -*-
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "GeminiProConfig.YAML")
LOG_FILE = os.path.join(SCRIPT_DIR, "game_monitor_v58.log")
pyautogui.FAILSAFE = False

# Manuel tıklama kilidi ve zamanı
MANUAL_CLICK_LOCK = threading.Lock()
last_manual_click_time = None
BOT_IS_CLICKING_EVENT = threading.Event()

def log_to_file(msg, log_file=LOG_FILE):
    """Mesajı belirtilen log dosyasına zaman damgasıyla yazar."""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{now}] {msg}\n")
    except Exception as e:
        print(f"!!! KRİTİK LOGLAMA HATASI ({log_file}): {e} !!!")
        traceback.print_exc()

def _safe_int(v, default):
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

# --- GÜNCELLENDİ: Artık 'manual_click_detected' sinyalini de tetikliyor ---
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

def load_config(filename=CONFIG_FILE): 
    try:
        with open(filename, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) 
        log_to_file(f"{filename} başarıyla yüklendi.")
        return config_data
    except FileNotFoundError:
        error_msg = f"HATA: {filename} bulunamadı."
        log_to_file(error_msg)
        try: messagebox.showerror("Config Hatası", f"{error_msg}\nProgram kapatılacak.")
        except Exception: pass
        return None
    except yaml.YAMLError as e: 
        error_msg = f"HATA: {filename} dosyası geçersiz YAML formatında: {e}"
        log_to_file(error_msg)
        try: messagebox.showerror("Config Hatası", f"{error_msg}\nProgram kapatılacak.")
        except Exception: pass
        return None
    except Exception as e:
        error_msg = f"Config yüklenirken beklenmedik hata: {e}"
        log_to_file(error_msg)
        traceback.print_exc()
        try: messagebox.showerror("Config Hatası", f"{error_msg}\nProgram kapatılacak.")
        except Exception: pass
        return None

# =============================================================================
# 2) YENİ: LocationManager Sınıfı
# =============================================================================

class LocationManager:
    def __init__(self, bot, general_cfg: dict):
        self.bot = bot
        self.general_cfg = general_cfg       
        self.LOCATION_MAP = {} 
        self.NAME_TO_VALUE = {} 
        self.current_location_value = -1 # UNKNOWN
        self._build_location_maps()
        self.current_location_value = self.NAME_TO_VALUE.get("UNKNOWN", -1)
    def _build_location_maps(self):
        json_locs = self.general_cfg.get("locations", {})
        if not isinstance(json_locs, dict):
            self.bot.log("Config uyarısı: 'module_general.locations' bir sözlük (dict) değil.")
            return
        for name, data in json_locs.items():
            try:
                val_int = int(data.get("value", -1))
                name_upper = name.upper()
                self.LOCATION_MAP[val_int] = name_upper
                self.NAME_TO_VALUE[name_upper] = val_int
            except Exception as e:
                 log_to_file(f"Config 'locations' işlenirken hata (konum: {name}): {e}")
        if not self.LOCATION_MAP:
             self.bot.log("UYARI: Hiçbir konum tanımlanamadı.")
             self.LOCATION_MAP[-1] = "UNKNOWN"
             self.NAME_TO_VALUE["UNKNOWN"] = -1
        self.bot.log(f"Konum haritaları oluşturuldu: {len(self.LOCATION_MAP)} adet konum tanımlandı.")
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
                self.bot.log(f"KONUM AYARLANDI: '{self.get_location_name(old_val)}' -> '{target_name_upper}'")
        else:
            self.bot.log(f"UYARI: set_current_location_by_name('{name}') başarısız. Bu isimde bir konum haritada bulunamadı.")
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
                image_file = anchor.get("image_file")
                confidence = float(anchor.get("confidence", 0.7))
                region_key = anchor.get("region_key")
                if not image_file or not region_key:
                    continue 
                search_region = self.bot.ui_regions.get(region_key)
                if not search_region:
                    self.bot.log(f"Konum Tespiti Hatası: '{location_name}' için '{region_key}' bölgesi 'ui_regions' içinde bulunamadı.")
                    continue

                template_gray = self.bot.get_cached_image(image_file)
                if template_gray is None:
                    continue 
                try:
                    match = self.bot.find_image_on_screen(image_file, search_region, confidence)
                    if match:
                        return int(data.get("value", -1))
                except Exception as e:
                     self.bot.log(f"Görsel konum tespiti sırasında OpenCV hatası ({image_file}): {e}")
        return self.NAME_TO_VALUE.get("UNKNOWN", -1)

# =============================================================================
# 3) Güncellenmiş Automator Sınıfı
# =============================================================================

class Automator:
    def __init__(self, bot):
        self.bot = bot 
    
    def _get_setting(self, key, default):
        return self.bot.settings.get(key, default)

    def _verify_anchor(self, label: str, coord: dict) -> bool:
        anchor_list = []
        single_anchor = coord.get("anchor")
        if single_anchor:
            anchor_list.append(single_anchor)
        else:
            multiple_anchors = coord.get("anchors")
            if isinstance(multiple_anchors, list):
                anchor_list.extend(multiple_anchors)
        
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
                self.bot.log(f"[{label}] Anchor Hatası: 'ui_regions' içinde '{region_key}' bölgesi bulunamadı. Doğrulama atlandı.")
                continue 
                
            confidence = float(anchor.get("confidence", 0.8))

            log_msg = f"Doğrulama (Statik): '{image_file}' aranıyor..."
            self.bot.log(log_msg)
            self.bot.gui_queue.put(("set_status_line1", log_msg))
            
            match = self.bot.find_image_on_screen(image_file, region, confidence)

            if match:
                log_msg = f"Başarılı: Statik '{image_file}' bulundu."
                self.bot.log(log_msg)
                self.bot.gui_queue.put(("set_status_line1", log_msg))
                found_match = True
                break 
            
        if not found_match:
            tried_images = [anc.get('image_file', 'N/A') for anc in anchor_list]
            log_msg = f"UYARI: '{label}' için beklenen görsel bulunamadı ({tried_images}). Yine de tıklanıyor..."
            self.bot.log(log_msg)
            self.bot.gui_queue.put(("set_status_line1", log_msg))

        return True 

    def click(self, label: str | None, coord_override: dict | None = None, check_game=True) -> bool:
        """
        Belirtilen etikete tıklar.
        """
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

        if not self._verify_anchor(action_label, coord_to_use):
             pass 

        try:
            x = coord_to_use["x"]
            y = coord_to_use["y"]
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
             self.bot.log(f"Tıklama hatası: Koordinat '{action_label}' için 'x' veya 'y' anahtarı eksik.")
             return False
        except Exception as e:
            self.bot.log(f"Tıklama sırasında beklenmedik hata ({action_label}): {e}")
            traceback.print_exc()
            return False
    
    def press_key(self, key: str, label: str = "manual_key"):
        """Klavye tuşuna basar."""
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            self.bot.log(f"Tuş basma hatası ({key}): {e}")
            return False

    def login_to_game(self):
        self.bot.log("Giriş ekranı bekleniyor (6sn)...") 
        if not self.bot._interruptible_wait(6): return

        if not self.click("server_s575", check_game=False):
            self.bot.log("UYARI: Sunucu koordinatı bulunamadı veya tıklanamadı. Yine de devam ediliyor...")

        self.bot.log("Sunucu seçildi (varsayıldı). Oyunun yüklenmesi bekleniyor (20sn)...")
        if not self.bot._interruptible_wait(20): return

        self.bot.log("Etkinlik oku kapatılıyor (varsa) ve otomatik savaş ('Z') başlatılıyor.")
        
        self.click("etkinlik_oku_close", check_game=False) 
        time.sleep(self.bot.settings.get("SLEEP_SHORT", 1))

        # 'Z' tuşuna bas ve kaydet
        self.press_key('z', label="auto_battle_start")
        self.bot.log("Otomatik savaş tuşu ('Z') gönderildi ve kaydedildi.")

    def return_to_exp_farm(self, force_restart_if_failed=False) -> bool:
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
    (Artık 'auto_exits' bayrağını destekliyor)
    """
    def __init__(self, bot, events_config_data: dict):
        self.bot = bot
        self.events_config = events_config_data 
        self.SCHEDULED_EVENTS = self._get_scheduled_events()
        self.last_event_run = {}

    def _get_scheduled_events(self):
        """Etkinlik takvimini 'module_timed_events' yapısından okur."""
        def t(time_str):
            try:
                h, m = map(int, str(time_str).split(":"))
                return dt_time(hour=h, minute=m)
            except Exception as e:
                # Loglama gerekirse buraya eklenebilir
                return None
        
        self.bot.log("Config'den 'module_timed_events' yükleniyor...")
        config_events = self.events_config.get("events", [])
        processed_events = []
        
        for event in config_events:
            try:
                start_time = t(event.get("start_time"))
                end_time = t(event.get("end_time"))
                
                if start_time is None or end_time is None:
                    self.bot.log(f"UYARI: Etkinlik '{event.get('name')}' için 'start_time' veya 'end_time' geçersiz, atlanıyor.")
                    continue
                    
                action_type = event.get("action_type", "static_sequence")
                
                processed_event = {
                    "name": event.get("name", "Bilinmeyen Etkinlik"),
                    "days": event.get("days"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "action_type": action_type,
                    "location_name_on_enter": event.get("location_name_on_enter", "UNKNOWN"),
                    "auto_exits": event.get("auto_exits", False), 
                    "action_sequence": event.get("action_sequence", []),
                    "entry_sequence": event.get("entry_sequence", []),
                    "hunt_target_list": event.get("hunt_target_list")
                }
                processed_events.append(processed_event)
                
            except Exception as e:
                self.bot.log(f"UYARI: Etkinlik işlenirken hata ({event.get('name', 'Bilinmeyen')}): {e}")
                continue
        
        valid_events = [e for e in processed_events if e["start_time"] is not None and e["end_time"] is not None]
        
        if valid_events:
            self.bot.log(f"Config'den {len(valid_events)} zamanlı etkinlik yüklendi.")
            return valid_events
        else:
            self.bot.log("UYARI: Config'den geçerli etkinlik yüklenemedi.")
            return []

    # DİKKAT: Bu fonksiyon Class hizasının içinde olmalı (bir tab içeride)
    def event_monitor_thread(self):
        """Etkinlik zamanlarını periyodik olarak kontrol eden thread fonksiyonu."""
        self.bot.log("Etkinlik izleyici başlatıldı.")
        interval = self.bot.settings.get("EVENT_MONITOR_SLEEP_SEC", 30)

        while True:
            try:
                today_str = datetime.now().strftime('%Y-%m-%d')
                old_keys = [k for k in list(self.last_event_run.keys()) if not k.startswith(today_str)]
                for k in old_keys: del self.last_event_run[k]

                if not self.bot.running.is_set() or self.bot.paused:
                    time.sleep(1.0) 
                    continue

                now = datetime.now()
                current_time = now.time()
                current_weekday = now.weekday() 

                event_to_run = None
                for event in self.SCHEDULED_EVENTS:
                    day_match = (event["days"] is None or current_weekday in event["days"])
                    time_match = (event["start_time"] <= current_time < event["end_time"])
                    
                    start_time_str = event['start_time'].strftime('%H:%M') 
                    event_key = f"{today_str}_{event['name']}_{start_time_str}"
                    
                    already_run = self.last_event_run.get(event_key, False)

                    if day_match and time_match and not already_run:
                        event_to_run = event
                        break 

                if event_to_run:
                    self.bot.log(f"Etkinlik zamanı geldi: {event_to_run['name']}.")
                    self.bot.active_event = event_to_run
                    start_time_str = event_to_run['start_time'].strftime('%H:%M') 
                    event_key = f"{today_str}_{event_to_run['name']}_{start_time_str}"

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
                            
                            # Konum kontrolü
                            if self.bot.location_manager.get_region_name() != "EXP_FARM":
                                if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=False):
                                    self.bot.log(f"'{event_to_run['name']}' etkinliğine girmeden önce Exp Farm'a dönülemedi. İptal.")
                                    action_success = False
                                else:
                                    if not self.bot._interruptible_wait(self.bot.settings.get("SLEEP_SHORT", 1)):
                                        raise Exception("Bekleme iptal edildi")
                            
                            action_type = event_to_run.get("action_type", "static_sequence")
                            
                            # EXP_FARM'daysak veya dönüş yaptıysak
                            if self.bot.location_manager.get_region_name() == "EXP_FARM":
                                if action_type == "dynamic_hunt":
                                    # Fonksiyon silindiği için uyarı veriyoruz
                                    self.bot.log(f"UYARI: Dinamik Av kodu silindi ({event_to_run['name']}). Lütfen 'static_sequence' kullanın.")
                                    action_success = False
                                
                                elif action_type == "static_sequence":
                                    action_sequence = event_to_run.get("action_sequence", [])
                                    if not action_sequence:
                                        self.bot.log(f"HATA: '{event_to_run['name']}' için 'action_sequence' bulunamadı.")
                                        action_success = False
                                    else:
                                        action_success = self.bot._run_sequence(
                                            action_sequence,
                                            self.bot.timed_events_cfg.get("coordinates", {})
                                        )
                                else:
                                    self.bot.log(f"HATA: Bilinmeyen action_type: '{action_type}'")
                                    action_success = False

                                if action_success:
                                    self.bot.location_manager.set_current_location_by_name(event_to_run["location_name_on_enter"])
                                    self.bot.log(f"'{event_to_run['name']}' etkinliğine giriş başarılı.")
                                else:
                                    self.bot.log(f"'{event_to_run['name']}' etkinlik girişi başarısız.")
                            
                        except Exception as e:
                            self.bot.log(f"Etkinlik '{event_to_run['name']}' çalıştırılırken hata oluştu: {e}")
                            traceback.print_exc()
                            action_success = False
                            
                    self.bot.log(f"'{event_to_run['name']}' için aksiyon kilidi bırakıldı.")

                    event_auto_exits = event_to_run.get("auto_exits", False)
                    
                    if action_success and event_auto_exits:
                        self.bot.log(f"'{event_to_run['name']}' etkinliği 'auto_exits: true'. Bitiş bekleniyor...")
                        end_dt = datetime.combine(now.date(), event_to_run["end_time"])
                        wait_seconds = max(0, (end_dt - datetime.now()).total_seconds()) + 5 
                        
                        if wait_seconds > 0:
                            if not self.bot._interruptible_wait(wait_seconds):
                                self.bot.log("Bekleme kesildi.")
                        
                        self.bot.log(f"'{event_to_run['name']}' bitti. Konum Exp Farm varsayılıyor.")
                        self.bot.location_manager.set_current_location_by_name("EXP_FARM")
                        self.last_event_run[event_key] = True
                        
                    elif action_success and not event_auto_exits:
                        end_dt = datetime.combine(now.date(), event_to_run["end_time"])
                        wait_seconds = max(0, (end_dt - datetime.now()).total_seconds())

                        if wait_seconds > 0:
                            self.bot.log(f"'{event_to_run['name']}' bitişi için {int(wait_seconds)} sn bekleniyor...")
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
        self.bot = bot
        self.root = getattr(bot, "root", None) 

        if self.root is None:
             print("KRİTİK HATA: GUIManager başlatılamadı, root (Tk ana penceresi) bulunamadı.")
             return 

        self.gui_update_interval_ms = int(update_interval_ms)
        self.title_label_line1 = None 
        self.title_label_line2 = None 
        self.log_text_widget = None
        self.gui_widgets = {}
        self.gui_queue = getattr(bot, "gui_queue", Queue()) 
        self.boss_targets_dict = getattr(bot, "bosslar", {}) 
        self._created = False
        
        self.status_message_active = False
        self.last_status_message_time = 0

    def _log(self, text: str):
        try:
            if hasattr(self, "bot") and hasattr(self.bot, "log") and callable(self.bot.log):
                self.bot.log(f"[GUI] {text}")
            else:
                 print(f"[GUI-LOG] {text}")
                 if self.gui_queue: self.gui_queue.put(("log", f"[GUI-Hata] {text}"))
        except Exception as e:
            print(f"[GUI-Loglama Hatası] {text} - Hata: {e}")

    def _parse_time_to_seconds(self, s: str) -> int | None:
        s = s.strip()
        try:
            parts = list(map(int, s.split(':')))
            if len(parts) == 3: h, m, sec = parts
            elif len(parts) == 2: h, m, sec = 0, parts[0], parts[1]
            elif len(parts) == 1: h, m, sec = 0, 0, parts[0]
            else: return None
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
                return None
            return h * 3600 + m * 60 + sec
        except ValueError: return None
        except Exception as e:
            self._log(f"Zaman ayrıştırma hatası ({s}): {e}")
            return None

    def _create_round_button(self, parent, symbol, color, command, col):
        try:
            canvas = tk.Canvas(parent, width=60, height=60, bg="black", highlightthickness=0, cursor="hand2")
            canvas.grid(row=0, column=col, padx=10, pady=5)
            circle = canvas.create_oval(5, 5, 55, 55, fill=color, outline=color, activefill=self._adjust_brightness(color, 0.8))
            text_item = canvas.create_text(30, 30, text=symbol, fill="white", font=("Consolas", 18, "bold"))
            def on_click(event=None):
                try: command()
                except Exception as e:
                    self._log(f"Buton '{symbol}' komutu çalıştırılırken hata: {e}")
                    traceback.print_exc()
            for item in (circle, text_item):
                canvas.tag_bind(item, "<Button-1>", on_click)
        except Exception as e:
            self._log(f"Yuvarlak buton '{symbol}' oluşturulurken hata: {e}")

    def _adjust_brightness(self, color_name, factor):
        try:
            rgb = self.root.winfo_rgb(color_name)
            r, g, b = [x / 256 for x in rgb]
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return color_name

    def _edit_timer_label(self, event, boss_aciklama: str):
        """Boss zamanlayıcısına tıklandığında manuel süre girişi sağlar."""
        try:
            hedef = self.bot.bosslar.get(boss_aciklama) 
            if not hedef:
                self._log(f"Manuel süre düzenleme hatası: Boss '{boss_aciklama}' bulunamadı.")
                return

            input_time = simpledialog.askstring("Zamanı Düzenle",
                                                f"'{boss_aciklama}' için kalan süreyi girin:\n(HH:MM:SS, MM:SS veya Saniye)",
                                                parent=self.root)
            if input_time is None: return

            kalan_saniye = self._parse_time_to_seconds(input_time)

            if kalan_saniye is None:
                messagebox.showwarning("Geçersiz Format",
                                       "Geçersiz zaman formatı!",
                                       parent=self.root)
                return

            self.bot._set_spawn_time(boss_aciklama, kalan_saniye, source="manual")
            self._log(f"'{boss_aciklama}' için manuel süre ayarlandı: {input_time} ({kalan_saniye} saniye)")

        except Exception as e:
            self._log(f"Manuel süre düzenleme sırasında hata ({boss_aciklama}): {e}")
            messagebox.showerror("Hata", f"Süre ayarlanırken bir hata oluştu:\n{e}", parent=self.root)

    def _process_gui_queue(self):
        """
        Diğer thread'lerden gelen mesajları işler.
        """
        if self.gui_queue is None: return 

        processed_count = 0
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
                        now = datetime.now().strftime("%H:%M:%S")
                        formatted_msg = f"[{now}] {data}"
                        self.log_text_widget.config(state=tk.NORMAL)
                        self.log_text_widget.insert(tk.END, formatted_msg + "\n")
                        self.log_text_widget.see(tk.END)
                        current_lines = int(self.log_text_widget.index(f"{tk.END}-1c").split('.')[0])
                        if current_lines > 500:
                             lines_to_delete = current_lines - 400
                             self.log_text_widget.delete('1.0', f'{lines_to_delete}.0')
                        self.log_text_widget.config(state=tk.DISABLED)
                
                elif task == "set_spawn":
                    if isinstance(data, tuple) and len(data) == 3:
                        aciklama, spawn_time, source = data
                        if isinstance(aciklama, str) and isinstance(spawn_time, (int, float)):
                            self._update_spawn_time_in_gui(aciklama, spawn_time, source)
                        else:
                             self._log(f"GUI Queue hatası: 'set_spawn' için geçersiz veri tipi: {type(aciklama)}, {type(spawn_time)}")
                    else:
                        self._log(f"GUI Queue hatası: 'set_spawn' için geçersiz veri formatı: {data}")

                elif task == "set_status_line1":
                    if isinstance(data, str) and self.title_label_line1:
                        self.title_label_line1.config(text=data, fg="yellow")
                        self.status_message_active = True
                        self.last_status_message_time = time.monotonic()

            except Exception as e:
                 self._log(f"GUI queue mesajı işlenirken hata (Task: {task}): {e}")
                 traceback.print_exc()

    def _update_spawn_time_in_gui(self, aciklama: str, spawn_time: float, source: str):
        """GUI'deki ilgili boss için spawn zamanı entry'sini günceller."""
        if not aciklama: return

        widgets = self.gui_widgets.get(aciklama)
        if not widgets or "entry" not in widgets: return 

        try:
            if spawn_time > 0:
                spawn_str = datetime.fromtimestamp(spawn_time).strftime("%H:%M:%S")
            else: 
                spawn_str = "--:--:--"

            entry_widget = widgets["entry"]
            entry_widget.config(state=tk.NORMAL)
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, spawn_str)

            color = "lime" if source == "automation" else "yellow" if source == "manual" else "cyan" 
            entry_widget.config(fg=color)

            entry_widget.config(state="readonly")
        except Exception as e:
            self._log(f"GUI spawn zamanı güncellenirken hata ({aciklama}): {e}")

    def update_status_label(self, remaining_targets):
        """GUI'nin en üstündeki İKİ SATIRLI durum başlığını günceller."""
        if self.title_label_line1 is None or self.title_label_line2 is None:
            return 

        try:
            is_running = self.bot.running.is_set()
            is_paused = self.bot.paused
            active_event = self.bot.active_event
            attacking_target_aciklama = self.bot.attacking_target_aciklama
            region_name = self.bot.location_manager.get_region_name()

            # --- SATIR 1 GÜNCELLEMESİ (Durum / Etkinlik / Konum) ---
            
            if self.status_message_active:
                if (time.monotonic() - self.last_status_message_time) < 3.0:
                    pass 
                else:
                    self.status_message_active = False 
            
            if not self.status_message_active:
                text_line1 = "Başlatılıyor..."
                color_line1 = "gray"

                if is_paused:
                    text_line1 = "Duraklatıldı"
                    color_line1 = "goldenrod"
                elif not is_running:
                     text_line1 = "Beklemede"
                     color_line1 = "gray"
                elif active_event:
                    try:
                        end_dt = datetime.combine(datetime.now().date(), active_event["end_time"])
                        kalan_saniye = max(0, (end_dt - datetime.now()).total_seconds())
                        saat, rem = divmod(int(kalan_saniye), 3600)
                        dakika, saniye = divmod(rem, 60)
                        time_str = f"{saat:02}:{dakika:02}:{saniye:02}"
                        event_name_short = active_event['name'].split('(')[0].strip()
                        text_line1 = f"Etkinlik: {event_name_short} ({time_str})"
                        color_line1 = "orange"
                    except Exception:
                        text_line1 = f"Etkinlik: {active_event.get('name', 'Bilinmeyen')}"
                        color_line1 = "orange"
                elif attacking_target_aciklama:
                    text_line1 = f"Saldırı: {attacking_target_aciklama}"
                    color_line1 = "red"
                else:
                    text_line1 = f"Konum: {region_name}"
                    color_line1 = "lime" if region_name == "EXP_FARM" else "gray"
                
                self.title_label_line1.config(text=text_line1, fg=color_line1)

            # --- SATIR 2 GÜNCELLEMESİ (Sadece Boss Geri Sayımı) ---
            
            text_line2 = "Boss hedefi yok"
            color_line2 = "gray"
            
            if is_paused or not is_running:
                 text_line2 = "Boss takibi duraklatıldı"
                 color_line2 = "goldenrod"
            elif attacking_target_aciklama:
                text_line2 = f"{attacking_target_aciklama} saldırılıyor..."
                color_line2 = "red"
            elif remaining_targets:
                kalan, aciklama = remaining_targets[0] 
                saat, rem = divmod(int(kalan), 3600)
                dakika, saniye = divmod(rem, 60)
                time_str = f"{saat:02}:{dakika:02}:{saniye:02}"
                text_line2 = f"Sıradaki: {aciklama} ({time_str})"
                color_line2 = "cyan"
            
            self.title_label_line2.config(text=text_line2, fg=color_line2)

        except Exception as e:
            self._log(f"GUI durum başlığı güncellenirken hata: {e}")
            try: 
                self.title_label_line1.config(text="Durum Hatası", fg="red")
                self.title_label_line2.config(text="Durum Hatası", fg="red")
            except Exception: pass


    def _update_gui(self):
        """GUI'yi periyodik olarak güncelleyen ana fonksiyon."""
        try:
            self._process_gui_queue()

            now = time.monotonic() 
            remaining = []
            self.boss_targets_dict = self.bot.bosslar 

            for aciklama, hedef in self.boss_targets_dict.items():
                widgets = self.gui_widgets.get(aciklama)
                if not widgets or "timer_label" not in widgets: continue 

                spawn_time = hedef.get("spawn_time") 
                timer_label = widgets["timer_label"]

                if spawn_time and isinstance(spawn_time, (int, float)):
                    kalan = max(0, spawn_time - time.time())
                    if kalan > 0:
                        remaining.append((kalan, aciklama))
                    
                    saat, rem = divmod(int(kalan), 3600)
                    dakika, saniye = divmod(rem, 60)
                    time_str = f"{saat:02}:{dakika:02}:{saniye:02}"
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


    def create_gui(self):
        """GUI penceresini ve içindeki widget'ları oluşturur."""
        if self._created: return 
        if self.root is None:
             self._log("GUI oluşturulamaz: root penceresi yok.")
             return 
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            width = min(320, max(280, screen_width // 6))
            height = min(600, max(500, screen_height // 2 + 50))
            self.root.title("LoA Bot v5.8")
            self.root.wm_attributes("-topmost", True)
            self.root.configure(bg="black")
            self.root.geometry(f"{width}x{height}")
            self.root.resizable(False, False)

            self.title_label_line1 = tk.Label(self.root, text="Başlatılıyor...", font=("Consolas", 11, "bold"), fg="cyan", bg="black")
            self.title_label_line1.pack(pady=(10, 0), fill=tk.X, padx=10) 
            self.title_label_line2 = tk.Label(self.root, text="Boss hedefleri bekleniyor...", font=("Consolas", 11, "bold"), fg="gray", bg="black")
            self.title_label_line2.pack(pady=(0, 5), fill=tk.X, padx=10)

            boss_frame = tk.LabelFrame(self.root, text=" Boss Zamanlayıcıları ", bg="black", fg="white", font=("Consolas", 10), relief=tk.GROOVE, bd=2)
            boss_frame.pack(padx=10, pady=5, fill=tk.X)
            boss_frame.grid_columnconfigure(0, weight=1)
            boss_frame.grid_columnconfigure(1, weight=1)
            boss_frame.grid_columnconfigure(2, weight=1)

            self.boss_targets_dict = self.bot.bosslar
            if not self.boss_targets_dict:
                 no_target_label = tk.Label(boss_frame, text="Config'de boss hedefi tanımlanmamış.", font=("Consolas", 10), fg="yellow", bg="black")
                 no_target_label.grid(row=0, column=0, columnspan=3, pady=10)
            else:
                for i, (aciklama, hedef) in enumerate(self.boss_targets_dict.items()):
                    lbl = tk.Label(boss_frame, text=f"{aciklama}:", anchor="e", font=("Consolas", 10), fg="white", bg="black")
                    lbl.grid(row=i, column=0, padx=(10, 2), pady=1, sticky="ew")
                    
                    entry = tk.Entry(boss_frame, width=9, font=("Consolas", 10, "bold"), readonlybackground="#202020", relief=tk.FLAT, justify='center')
                    entry.grid(row=i, column=1, padx=2, pady=1, sticky="ew")
                    entry.config(state="readonly", fg="cyan")
                    
                    timer_label = tk.Label(boss_frame, text="--:--:--", anchor="w", font=("Consolas", 10), fg="white", bg="black", cursor="hand2")
                    timer_label.grid(row=i, column=2, padx=(2, 10), pady=1, sticky="ew")
                    
                    timer_label.bind("<Button-1>", lambda event, a=aciklama: self._edit_timer_label(event, a))

                    self.gui_widgets[aciklama] = {"entry": entry, "timer_label": timer_label}
                    if hedef.get("spawn_time"):
                        self._update_spawn_time_in_gui(aciklama, hedef["spawn_time"], hedef.get("source", "config"))

            button_frame = tk.Frame(self.root, bg="black")
            button_frame.pack(pady=(5, 8))
            self._create_round_button(button_frame, "▶", "darkgreen", lambda: self.bot.start(), 0)
            self._create_round_button(button_frame, "⏸", "goldenrod", lambda: self.bot.stop(), 1)
            self._create_round_button(button_frame, "✖", "darkred", lambda: self.root.quit(), 2)

            self.log_text_widget = tk.Text(self.root, height=10, bg="#101010", fg="#DDDDDD", font=("Consolas", 9), state=tk.DISABLED, relief=tk.SOLID, bd=1, wrap=tk.WORD)
            self.log_text_widget.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

            self.root.after(self.gui_update_interval_ms, self._update_gui)
            self._created = True
            self._log("GUI başarıyla oluşturuldu.")

        except Exception as e:
            self._log(f"GUI oluşturma sırasında kritik hata: {e}")
            traceback.print_exc()
            try: messagebox.showerror("GUI Hatası", f"Arayüz oluşturulamadı:\n{e}")
            except Exception: pass
            try:
                 if self.root: self.root.quit()
            except Exception: pass

# =============================================================================
# 6) Ana Bot Sınıfı
# =============================================================================

class LoABot:
    """
    Ana bot sınıfı. 'LocationManager' kullanıyor.
    """
    def __init__(self, root):
        self.root = root
        self.gui_queue = Queue()

        self.config = load_config(CONFIG_FILE)
        if not self.config:
            log_to_file("KRİTİK: Config yüklenemediği için bot başlatılamıyor.")
            self.initialized = False
            try:
                 if self.root: self.root.quit()
            except Exception: pass
            return
        
        self.general_cfg = self.config.get("module_general", {})
        self.demon_boss_cfg = self.config.get("module_demon_boss", {})
        self.timed_events_cfg = self.config.get("module_timed_events", {})

        self.initialized = True
        
        # --- YENİ: Manuel tıklama sinyalini başlat ---
        self.manual_click_detected = threading.Event()
        
        start_click_listener(self) 

        try:
            self.game_path = self.general_cfg["game_path"]
            self.process_name = self.general_cfg["process_name"]
            self.settings = self.general_cfg.get("settings", {})
            self.ui_regions = self.general_cfg.get("ui_regions", {})
            self.exit_buttons_config = self.general_cfg.get("exit_buttons", {})
            self.popup_config = self.general_cfg.get("POPUP_MONITOR", {})
            self.pvp_config = self.general_cfg.get("PVP_DEFENSE", {"enabled": False})

        except KeyError as e:
             error_msg = f"Config hatası: 'module_general' içinde zorunlu alan eksik: {e}"
             self._log_to_file_and_message(error_msg)
             self.initialized = False
             try:
                  if self.root: self.root.quit()
             except Exception: pass
             return

        self.coordinates = self.demon_boss_cfg.get("coordinates", {})
        self.coordinates.update(self.timed_events_cfg.get("coordinates", {}))
        
        self.bosslar = self.demon_boss_cfg.get("bosslar", {}) 
        self.walk_times = self.demon_boss_cfg.get("walk_times", {})
        self.boss_sequence_template = self.demon_boss_cfg.get("sequence_template", [])

        self.settings.setdefault("SLEEP_SHORT", 1)
        self.settings.setdefault("SLEEP_MEDIUM", 2)
        self.settings.setdefault("SLEEP_LONG", 5)
        
        # Boss verilerini başlangıçta hazırla
        self._initialize_boss_data()
        
        self.running = threading.Event()
        self.paused = True 
        self.action_lock = threading.Lock() 
        
        self.active_event = None 
        self.attacking_target_aciklama = None 
        self.loot_collection_ends_at = 0 

        self._pid = None 
        self._last_screenshot_checksum = None 
        self._last_screenshot_time = 0 
        self._freeze_count = 0 

        self.location_manager = LocationManager(self, self.general_cfg)
        self.automator = Automator(self)
        self.event_manager = EventManager(self, self.timed_events_cfg)
        self.gui_manager = GUIManager(self) 

        self.log("Görüntü önbelleği (image cache) yükleniyor...")
        self.image_cache = {} 
        self._preload_images_from_config()
        self.log(f"{len(self.image_cache)} adet görüntü hafızaya yüklendi.")

        self._threads_started = False 

        self.log("LoABot başlatma işlemi tamamlandı.")

    def _initialize_boss_data(self):
        """Boss verilerini başlangıçta hazırlar."""
        if not self.bosslar:
            self.log("UYARI: config.yaml 'module_demon_boss.bosslar' boş. Sadece etkinlik takibi yapılacak.")
            return
        
        valid_boss_keys = list(self.bosslar.keys())
        for aciklama in valid_boss_keys:
            t = self.bosslar[aciklama]
            if not isinstance(t, dict):
                self.log(f"Config hatası: bosslar['{aciklama}'] bir sözlük değil, atlanıyor.")
                del self.bosslar[aciklama]
                continue
            
            t['spawn_time'] = None 
            t['source'] = 'config' 
            t['aciklama'] = str(aciklama)
            
            coord_ref = t.get("koordinat_ref")
            katman_ref = t.get("katman_id")

            if not coord_ref or not katman_ref:
                 self.log(f"Config hatası: Boss '{aciklama}' için 'koordinat_ref' veya 'katman_id' eksik, hedef atlanıyor.")
                 del self.bosslar[aciklama]
                 continue

            if coord_ref not in self.coordinates:
                self.log(f"Config hatası: Boss '{aciklama}' için koordinat ('{coord_ref}') bulunamadı, hedef atlanıyor.")
                del self.bosslar[aciklama]
                continue
            
            t['koordinat'] = self.coordinates[coord_ref]
        
        if not self.bosslar:
             self.log("UYARI: 'bosslar' içindeki tüm hedefler geçersiz.")

    def _preload_images_from_config(self):
        """YAML config dosyasını tarar ve 'image_file' içeren tüm görselleri yükler."""
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

            # 4. Demon Boss Koordinatları (Anchor'lar ve Anchors Listeleri)
            for coord_data in self.demon_boss_cfg.get("coordinates", {}).values():
                if isinstance(coord_data, dict):
                    anchor_list = []
                    single_anchor = coord_data.get("anchor")
                    if single_anchor:
                        anchor_list.append(single_anchor)
                    else:
                        multiple_anchors = coord_data.get("anchors")
                        if isinstance(multiple_anchors, list):
                            anchor_list.extend(multiple_anchors)
                    
                    for anchor in anchor_list:
                        if anchor and anchor.get("image_file"):
                            image_files_to_load.add(anchor["image_file"])
            
            # 5. Demon Boss 'bosslar' (Dinamik Doğrulama)
            for boss_data in self.demon_boss_cfg.get("bosslar", {}).values():
                verify = boss_data.get("gorsel_dogrulama")
                if verify and verify.get("image_file"):
                    image_files_to_load.add(verify["image_file"])

            # 6. Zamanlı Etkinlik Koordinatları (Anchor'lar ve Anchors Listeleri)
            for coord_data in self.timed_events_cfg.get("coordinates", {}).values():
                if isinstance(coord_data, dict):
                    anchor_list = []
                    single_anchor = coord_data.get("anchor")
                    if single_anchor:
                        anchor_list.append(single_anchor)
                    else:
                        multiple_anchors = coord_data.get("anchors")
                        if isinstance(multiple_anchors, list):
                            anchor_list.extend(multiple_anchors)
                    
                    for anchor in anchor_list:
                        if anchor and anchor.get("image_file"):
                            image_files_to_load.add(anchor["image_file"])
            
            # 7. Zamanlı Etkinlik Sekansları (Statik Doğrulama)
            for event in self.timed_events_cfg.get("events", []):
                for step in event.get("action_sequence", []):
                    verify = step.get("verify_visual")
                    if verify and verify.get("image_file"):
                        image_files_to_load.add(verify["image_file"])

            # 8. Dinamik Av (Hunt Targets)
            for hunt_list in self.timed_events_cfg.get("hunt_targets", {}).values():
                for target in hunt_list.get("targets", []):
                    if target.get("live_image"):
                        image_files_to_load.add(target["live_image"])
                    if target.get("dead_image"):
                        image_files_to_load.add(target["dead_image"])
            
            # 9. Çıkış Butonları (Exit Buttons)
            for exit_list in self.general_cfg.get("exit_buttons", {}).values():
                for button in exit_list:
                    if button.get("image_file"):
                        image_files_to_load.add(button["image_file"])

            # 10. Zafer İkonu (verify_boss_kill)
            image_files_to_load.add("victory_icon.png") # Bunu manuel ekliyoruz

            # Şimdi tüm toplanan resimleri yükle
            for filename in image_files_to_load:
                if filename:
                    image_data = load_image(filename)
                    if image_data is not None:
                        self.image_cache[filename] = image_data
                
        except Exception as e:
            self.log(f"Görüntü önbelleği yüklenirken kritik hata: {e}")
            traceback.print_exc()

    def get_cached_image(self, filename: str):
        """
        Bir görüntüyü önbellekten (RAM) döndürür. 
        """
        if filename in self.image_cache:
            return self.image_cache[filename]
        else:
            self.log(f"UYARI: '{filename}' önbellekte bulunamadı. Disketen okunuyor...")
            image_data = load_image(filename)
            if image_data is not None:
                self.image_cache[filename] = image_data
            return image_data

    def _run_sequence(self, sequence: list, coord_map: dict, context_target: dict = None) -> bool:
        """
        Verilen bir sekans listesini (config'den) adım adım çalıştırır.
        """
        if not sequence:
            self.log("UYARI: Boş bir sekans çalıştırılmaya çalışıldı.")
            return False
            
        for step in sequence:
            if not self.running.is_set():
                self.log("Sekans çalıştırılırken bot durduruldu.")
                return False

            action = step.get("action")
            label = step.get("label")
            
            try:
                wait_ms = int(step.get("wait_ms", 100))
            except (ValueError, TypeError):
                self.log(f"UYARI: '{label}' adımı için 'wait_ms' değeri geçersiz ('{step.get('wait_ms')}'). Varsayılan 100ms kullanılıyor.")
                wait_ms = 100
            
            action_success = False
            try:
                if action == "click":
                    coord = coord_map.get(label)
                    if not coord:
                        self.log(f"Sekans Hatası: '{label}' için koordinat bulunamadı.")
                        return False
                    action_success = self.automator.click(label, coord_override=coord)
                
                elif action == "press_key":
                    key = step.get("key")
                    if not key:
                         self.log(f"Sekans Hatası: 'press_key' için 'key' belirtilmemiş.")
                         return False
                    pyautogui.press(key)
                    action_success = True

                elif action == "katman_seçimi":
                    if not context_target:
                         self.log("Sekans Hatası: 'katman_seçimi' için 'context_target' (boss verisi) sağlanmadı.")
                         return False
                    katman_label = context_target.get("katman_id") 
                    coord = coord_map.get(katman_label)
                    if not katman_label or not coord:
                        self.log(f"Sekans Hatası: Boss '{context_target.get('aciklama')}' için katman '{katman_label}' koordinatı bulunamadı.")
                        return False
                    action_success = self.automator.click(katman_label, coord_override=coord)

                elif action == "boss_seçimi":
                    if not context_target:
                         self.log("Sekans Hatası: 'boss_seçimi' için 'context_target' (boss verisi) sağlanmadı.")
                         return False
                    
                    boss_coord_label = context_target.get("koordinat_ref") 
                    coord = coord_map.get(boss_coord_label)
                    if not boss_coord_label or not coord:
                        self.log(f"Sekans Hatası: Boss '{context_target.get('aciklama')}' için boss koordinatı ('{boss_coord_label}') bulunamadı.")
                        return False
                        
                    boss_verify_config = context_target.get("gorsel_dogrulama")
                    if boss_verify_config and isinstance(boss_verify_config, dict):
                        image_file = boss_verify_config.get("image_file")
                        region_key = boss_verify_config.get("region_key")
                        
                        if image_file and region_key:
                            search_region = self.ui_regions.get(region_key)
                            if search_region:
                                log_msg = f"Doğrulama (Dinamik): '{image_file}' aranıyor..." 
                                self.log(log_msg)
                                self.gui_queue.put(("set_status_line1", log_msg)) 
                                
                                confidence = float(boss_verify_config.get("confidence", 0.8))
                                match = self.find_image_on_screen(image_file, search_region, confidence)
                                if not match:
                                    log_msg = f"UYARI: Boss '{image_file}' bulunamadı. Tıklanıyor..." 
                                    self.log(log_msg)
                                    self.gui_queue.put(("set_status_line1", log_msg)) 
                                else:
                                    log_msg = f"Başarılı: Boss '{image_file}' bulundu." 
                                    self.log(log_msg)
                                    self.gui_queue.put(("set_status_line1", log_msg)) 
                            else:
                                self.log(f"Sekans Hatası: Boss '{context_target.get('aciklama')}' için '{region_key}' bölgesi bulunamadı. Doğrulama atlandı.")
                        else:
                            self.log(f"Sekans UYARISI: Boss '{context_target.get('aciklama')}' için 'gorsel_dogrulama' tanımı eksik. Doğrulama atlandı.")

                    action_success = self.automator.click(boss_coord_label, coord_override=coord)
                
                else:
                    self.log(f"Sekans Hatası: Bilinmeyen eylem türü '{action}'.")
                    return False

            except Exception as e:
                self.log(f"Sekans adımı ('{action}') yürütülürken hata: {e}")
                traceback.print_exc()
                return False

            if not action_success:
                self.log(f"Sekans adımı BAŞARISIZ: '{action}' -> '{label or ''}'. Sekans iptal edildi.")
                return False
            
            if wait_ms > 0:
                if not self._interruptible_wait(wait_ms / 1000.0):
                    self.log("Sekans beklemesi sırasında bot durduruldu.")
                    return False
                    
        self.log("Sekans başarıyla tamamlandı.")
        return True

    def _interruptible_wait(self, seconds: float, step: float = 0.5) -> bool:
        """
        self.running event'i set olduğu sürece belirtilen saniye kadar bekler.
        """
        end_time = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end_time:
            if not self.running.is_set(): 
                return False
            remaining = end_time - time.monotonic()
            sleep_duration = min(step, max(0.05, remaining)) 
            time.sleep(sleep_duration)
        return self.running.is_set()

    def _boss_automation_thread(self):
        self.log("Boss otomasyon döngüsü başlatıldı.")
        
        while True:
            try:
                if not self.running.is_set() or self.paused:
                    time.sleep(0.5)
                    continue

                current_loc = self.location_manager.get_region_name()
                is_preparing = self.attacking_target_aciklama is not None
                is_event = self.active_event is not None

                if self.active_event is not None:
                    time.sleep(2)
                    continue

                # --- YENİ: Optimize Edilmiş Konum Senkronizasyonu (DÜZELTİLDİ) ---
                # Bot boştaysa (saldırı/event yoksa)
                if (not self.action_lock.locked() and 
                    self.attacking_target_aciklama is None and 
                    self.active_event is None):
                    
                    is_in_exp_farm = (self.location_manager.get_region_name() == "EXP_FARM")
                    manual_click_happened = self.manual_click_detected.is_set()

                    # Konum taramasını NE ZAMAN tetiklemeli?
                    # 1. Bot EXP_FARM'da DEĞİLSE (kaybolmuş olabilir, her döngüde kontrol etmeli)
                    # 2. VEYA Bot EXP_FARM'dayken KULLANICI tıkladıysa
                    if (not is_in_exp_farm) or (is_in_exp_farm and manual_click_happened):
                        
                        if manual_click_happened:
                            self.manual_click_detected.clear() # Sinyali temizle
                        else:
                            # Botun EXP_FARM'da olmadığını düşündüğü için (örn: KATMAN_1'de) 
                            # sürekli tarama yapıyor (bu normaldir)
                            pass 
                        
                        visual_val = self.location_manager._get_current_visual_location_value()
                        if visual_val != self.location_manager.NAME_TO_VALUE.get("UNKNOWN", -1) and self.location_manager.current_location_value != visual_val:
                            self.log(f"KONUM SENKRON: '{self.location_manager.get_region_name()}' -> '{self.location_manager.get_location_name(visual_val)}'")
                            self.location_manager.current_location_value = visual_val
                # --- YENİ BÖLÜM SONU ---

                now = time.time()
                ready_targets = []
                for aciklama, hedef in self.bosslar.items():
                    spawn_time = hedef.get('spawn_time')
                    head_start = hedef.get('head_start_saniye', 0)
                    if spawn_time and (now >= spawn_time - head_start):
                        ready_targets.append(hedef)

                if not ready_targets:
                    time.sleep(self.settings.get("MAIN_LOOP_SLEEP_SN", 1.0))
                    continue

                current_target = sorted(ready_targets, key=lambda h: h['spawn_time'])[0]
                current_target_aciklama = current_target['aciklama']

                with self.action_lock:
                    self.attacking_target_aciklama = current_target_aciklama
                    self.loot_collection_ends_at = 0
                    self.log(f"Sıradaki hedef: {current_target_aciklama}. Saldırı prosedürü başlıyor.")
                    
                    action_success = self._attack_boss(current_target)

                if not action_success:
                    self.log(f"Saldırı başarısız: {current_target_aciklama}. Exp Farm'a dönülüyor.")
                    with self.action_lock:
                        self.automator.return_to_exp_farm(force_restart_if_failed=True)
                    self.attacking_target_aciklama = None
                    continue

                now_after_attack = time.time()
                
                upcoming_targets = []
                for aciklama, hedef in self.bosslar.items():
                    spawn_time = hedef.get('spawn_time')
                    if spawn_time and spawn_time > now_after_attack:
                        upcoming_targets.append(hedef)
                
                upcoming_targets.sort(key=lambda h: (h['spawn_time'], h['boss_id']))
                next_target = upcoming_targets[0] if upcoming_targets else None

                should_stay = False
                if next_target:
                    is_same_map = next_target['katman_id'] == current_target['katman_id']
                    post_attack_wait = self.settings.get("POST_ATTACK_WAIT_SN", 30)
                    loot_finish_time = now_after_attack + post_attack_wait
                    time_to_next_spawn = next_target['spawn_time'] - loot_finish_time
                    threshold_sn = self.settings.get("BOSS_SWITCH_THRESHOLD_SN", 91)
                    is_soon = time_to_next_spawn < threshold_sn

                    if is_same_map and is_soon:
                        should_stay = True
                        self.log(f"Stratejik bekleme: Sonraki boss ({next_target['aciklama']}) aynı katta ve {int(time_to_next_spawn)} sn içinde. Haritada bekleniyor.")
                    else:
                        if not is_same_map: self.log(f"Sonraki boss farklı haritada. Çıkış.")
                        elif not is_soon: self.log(f"Sonraki boss'a çok var ({int(time_to_next_spawn)} sn). Çıkış.")

                if not should_stay:
                    self.log("Bekleme kriterleri karşılanmadı. Exp Farm'a dönülüyor.")
                    with self.action_lock:
                        self.automator.return_to_exp_farm(force_restart_if_failed=True)

                self.attacking_target_aciklama = None
                self.loot_collection_ends_at = 0
            except Exception as e:
                self.log(f"Boss otomasyon thread hatası: {e}")
                traceback.print_exc()
                self.attacking_target_aciklama = None
                time.sleep(5)

    def _get_walk_time(self, boss_a_aciklama: str, boss_b_aciklama: str) -> int:
        """İki boss arasındaki yürüme süresini 'walk_times' sözlüğünden alır."""
        if boss_a_aciklama == boss_b_aciklama: return 0

        try:
            boss_a_aciklama_str = str(boss_a_aciklama)
            boss_b_aciklama_str = str(boss_b_aciklama)
            
            t = self.walk_times.get(boss_a_aciklama_str, {}).get(boss_b_aciklama_str)
            if t is not None:
                return int(t)
            t = self.walk_times.get(boss_b_aciklama_str, {}).get(boss_a_aciklama_str)
            if t is not None:
                return int(t)
        except (TypeError, ValueError):
            pass

        default_walk_time = 40
        self.log(f"UYARI: Yürüme süresi bulunamadı ({boss_a_aciklama} <-> {boss_b_aciklama}). Varsayılan {default_walk_time} sn kullanılıyor.")
        return default_walk_time

    def _attack_boss(self, hedef: dict) -> bool:
        """
        Verilen boss hedefi için 'sequence_template'i çalıştırır.
        """
        spawn_timestamp = hedef.get('spawn_time')
        if not spawn_timestamp:
            self.log(f"HATA: {hedef['aciklama']} için spawn_time yok.")
            return False

        self.log(f"Saldırıya hazırlanılıyor: {hedef['aciklama']}")
        
        target_location_name = "KATMAN_1" if "katman_1" in hedef['katman_id'] else "KATMAN_2"

        if self.location_manager.get_region_name().upper() != target_location_name:
            self.log(f"Konum farklı ({self.location_manager.get_region_name()} -> {target_location_name}). Harita değiştiriliyor.")
            if not self.automator.return_to_exp_farm(force_restart_if_failed=True):
                self.log("Boss katmanına girmeden önce Exp Farm'a dönülemedi. Saldırı iptal.")
                return False
            
            sekans_basarili = self._run_sequence(
                self.boss_sequence_template, 
                self.demon_boss_cfg.get("coordinates", {}),
                context_target=hedef
            )
            
            if not sekans_basarili:
                self.log("Boss katmanına giriş sekansı başarısız.")
                self.automator.return_to_exp_farm()
                return False
            
            self.location_manager.set_current_location_by_name(target_location_name)
        
        else:
            self.log("Konum zaten doğru (aynı kat). Sadece boss'a tıklanıyor.")
            kisa_sekans = [
                     {"action": "click", "label": "boss_list_ac", "wait_ms": 200},
                     {"action": "boss_seçimi", "wait_ms": 200}
                 ] 
                     
            sekans_basarili = self._run_sequence(
                kisa_sekans,
                self.demon_boss_cfg.get("coordinates", {}),
                context_target=hedef
            )
            if not sekans_basarili:
                self.log(f"{hedef['aciklama']} koordinatına tıklanamadı (kısa sekans).")
                return False

        try:
            # 'Z' tuşunu da kaydet
            self.automator.press_key('z', label="boss_attack_auto_battle")
        except Exception as e:
            self.log(f"'z' tuşu basma hatası: {e}")
            return False

        wait_for_spawn = spawn_timestamp - time.time()
        if wait_for_spawn > 0:
            self.log(f"Doğuş için {wait_for_spawn:.2f} sn bekleniyor...")
            if not self._interruptible_wait(wait_for_spawn):
                self.log("Bekleme sırasında otomasyon durduruldu. Saldırı iptal.")
                return False

        if not self.is_game_running():
            self.log("Saldırı öncesi oyun kapandı.")
            return False

        self.log("Saldırı komutu gönderildi. Yeni spawn hesaplanıyor.")
        time.sleep(self.settings.get("SLEEP_SHORT", 2))
        attack_start_time = time.time()

        try:
            # 'A' tuşunu da kaydet
            self.automator.press_key('a', label="boss_attack_main")
        except Exception as e:
            self.log(f"'a' tuşu basma hatası: {e}")

        if self.verify_boss_kill(hedef):
            post_attack_wait = self.settings.get("POST_ATTACK_WAIT_SN", 30)
            self.log(f"Ganimet toplamak için {post_attack_wait} sn bekleniyor...")
            self.loot_collection_ends_at = time.time() + post_attack_wait
            self._recalculate_all_spawn_times_after_attack(hedef, attack_start_time)
            self._interruptible_wait(post_attack_wait)
            return True
        else:
            self.log(f"{hedef['aciklama']} kesilemedi/doğrulanamadı. 60 sn sonra tekrar denenecek.")
            self._set_spawn_time(hedef['aciklama'], 60, source='retry')
            return False

    def verify_boss_kill(self, boss) -> bool:
        try:
            region = self.ui_regions.get("freeze_check_area", {"x": 800, "y": 400, "w": 300, "h": 200})
            match = self.find_image_on_screen("victory_icon.png", region, confidence=0.85)
            if match:
                self.log(f"{boss['aciklama']} için zafer ikonu görüldü.")
                return True
        except Exception:
            pass
        self.log(f"{boss['aciklama']} için görsel doğrulama atlandı (varsayılan True).")
        return True

    def _recalculate_all_spawn_times_after_attack(self, killed_boss: dict, attack_start_time: float):
        self.log("--- Tüm Boss Zamanlamaları Güncelleniyor ---")
        periyot_saniye = killed_boss["periyot_saat"] * 3600
        gecikme_gercek = max(0, attack_start_time - killed_boss['spawn_time'])
        yeni_spawn_A = killed_boss['spawn_time'] + periyot_saniye + gecikme_gercek
        self.log(f"Kesilen Boss ({killed_boss['aciklama']}): Gecikme {gecikme_gercek:.1f}s. Yeni: {datetime.fromtimestamp(yeni_spawn_A).strftime('%H:%M:%S')}")
        self._set_spawn_time(killed_boss['aciklama'], yeni_spawn_A - time.time(), source='automation')

        post_attack_wait = self.settings.get("POST_ATTACK_WAIT_SN", 30)
        loot_finish_time = attack_start_time + post_attack_wait

        upcoming_targets = []
        for aciklama, hedef in self.bosslar.items():
            spawn_time = hedef.get('spawn_time')
            if spawn_time and spawn_time > loot_finish_time and aciklama != killed_boss['aciklama']:
                upcoming_targets.append(hedef)
        upcoming_targets.sort(key=lambda h: (h["spawn_time"], h['boss_id']))

        for next_boss in upcoming_targets:
            if next_boss['katman_id'] != killed_boss['katman_id']:
                continue
            yurume_suresi_tahmini = self._get_walk_time(str(killed_boss["aciklama"]), str(next_boss["aciklama"]))
            time_until_next_spawn = next_boss['spawn_time'] - loot_finish_time
            if yurume_suresi_tahmini > time_until_next_spawn:
                gecikme_kaymasi = yurume_suresi_tahmini - time_until_next_spawn
                yeni_spawn_B = next_boss['spawn_time'] + gecikme_kaymasi
                self.log(f"KAYDIRMA ({next_boss['aciklama']}): Yürüme {yurume_suresi_tahmini:.1f}s > Kalan {time_until_next_spawn:.1f}s -> +{gecikme_kaymasi:.1f}s")
                self._set_spawn_time(next_boss['aciklama'], yeni_spawn_B - time.time(), source='automation')
            else:
                self.log(f"Zamanlama OK ({next_boss['aciklama']}): {yurume_suresi_tahmini:.1f}s <= {time_until_next_spawn:.1f}s")

    def _set_spawn_time(self, boss_aciklama: str, kalan_saniye: float, source: str = 'config'):
        hedef = self.bosslar.get(str(boss_aciklama))
        if not hedef:
            self.log(f"UYARI: Spawn zamanı ayarlanamadı, boss bulunamadı: {boss_aciklama}")
            return
            
        hedef["spawn_time"] = time.time() + max(0.0, kalan_saniye)
        hedef["source"] = source
        self.gui_queue.put(("set_spawn", (boss_aciklama, hedef["spawn_time"], source)))

    def _log_to_file_and_message(self, msg):
        if hasattr(self, 'gui_queue') and self.gui_queue:
            self.gui_queue.put(("log", msg))
        log_to_file(msg)

    def log(self, msg: str):
        self._log_to_file_and_message(msg)

    def is_game_running(self) -> bool:
        if self._pid and psutil.pid_exists(self._pid):
            try:
                p = psutil.Process(self._pid)
                return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._pid = None
        try:
             for proc in psutil.process_iter(['pid', 'name']):
                 if proc.info['name'].lower() == self.process_name.lower():
                     self._pid = proc.pid
                     return True
        except Exception as e:
             self.log(f"Oyun prosesi aranırken hata: {e}")
        return False

    def _start_game(self):
        if self.is_game_running():
            self.log("Oyun zaten çalışıyor, tekrar başlatılmıyor.")
            return True
        try:
            game_dir = os.path.dirname(self.game_path)
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            proc = subprocess.Popen(
                self.game_path, 
                cwd=game_dir, 
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
                close_fds=True,
                shell=False
            )
            self._pid = proc.pid

            self.log(f"Oyun başlatıldı (Bağımsız Mod): {self.game_path}")
            if not self._interruptible_wait(5): return False
            self.automator.login_to_game()
            return True
        except Exception as e:
            self.log(f"Oyun başlatılırken hata: {e}")
            traceback.print_exc()
            return False

    def restart_game(self):
        self.log("Oyun yeniden başlatma komutu işleniyor...")
        time.sleep(1)
        pid_to_kill = self._pid 
        if not pid_to_kill or not psutil.pid_exists(pid_to_kill):
             try:
                 for proc in psutil.process_iter(['pid', 'name']):
                     if proc.info['name'].lower() == self.process_name.lower():
                         pid_to_kill = proc.pid
                         break
             except Exception: pass
        
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
            self.log("Aktif oyun prosesi bulunamadı, sonlandırma işlemi atlandı.")
            self._pid = None

    def find_image_on_screen(self, template_name: str, region: dict, confidence: float = 0.7) -> tuple | None:
        template_gray = self.get_cached_image(template_name)
        
        if template_gray is None:
            return None

        monitor = {
            "top": _safe_int(region.get("y", 0), 0),
            "left": _safe_int(region.get("x", 0), 0),
            "width": _safe_int(region.get("w", 1920), 1920),
            "height": _safe_int(region.get("h", 1080), 1080),
        }
        
        th, tw = template_gray.shape[:2]
        mh, mw = monitor["height"], monitor["width"]

        if th <= 0 or tw <= 0 or mh <= 0 or mw <= 0 or th > mh or tw > mw:
             return None

        try:
            with mss.mss() as sct:
                screenshot_bgra = sct.grab(monitor)
                screenshot_gray = cv2.cvtColor(np.array(screenshot_bgra), cv2.COLOR_BGRA2GRAY)

            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= confidence:
                center_x = monitor["left"] + max_loc[0] + tw // 2
                center_y = monitor["top"] + max_loc[1] + th // 2
                return (center_x, center_y, max_val)
            else:
                return None
        except Exception as e:
            return None

    def safe_exit_map(self, confidence=0.8) -> bool:
        """
        Mevcut konuma göre tanımlanmış çıkış butonlarını arar ve tıklar.
        """
        current_location_name = self.location_manager.get_region_name().upper() 
        
        exit_buttons_list = self.exit_buttons_config.get(current_location_name, [])
        
        if not exit_buttons_list:
            self.log(f"({current_location_name}) için özel çıkış tanımı yok. 'DEFAULT' çıkış listesi aranıyor...")
            exit_buttons_list = self.exit_buttons_config.get("DEFAULT", [])

        if not exit_buttons_list:
            self.log(f"({current_location_name}) için özel VEYA 'DEFAULT' çıkış butonu tanımı bulunamadı.")
            return False

        default_region = self.ui_regions.get("exit_map_area") 

        match_result = None
        found_button_config = None 

        for button_config in exit_buttons_list:
            img_file = button_config.get("image_file")
            if not img_file: continue
            
            region_key = button_config.get("region_key")
            search_region = None
            if region_key:
                search_region = self.ui_regions.get(region_key)
                if not search_region:
                    self.log(f"safe_exit_map Hatası: '{region_key}' ui_regions'da bulunamadı. Atlanıyor.")
                    continue
            else:
                search_region = default_region 
                
            if not search_region:
                 self.log(f"safe_exit_map Hatası: '{img_file}' için aranacak bir bölge bulunamadı.")
                 continue
                 
            search_confidence = float(button_config.get("confidence") or confidence)

            match = self.find_image_on_screen(img_file, search_region, search_confidence)
            if match:
                match_result = match
                found_button_config = button_config
                self.log(f"Çıkış butonu '{img_file}' bulundu.")
                break 

        needs_boss_list_fallback = current_location_name in ["KATMAN_1", "KATMAN_2"]
        if not match_result and needs_boss_list_fallback:
            self.log("Çıkış butonu görünür değil, boss listesine tıklanıyor (fallback)...")
            clicked_boss_list = self.automator.click("boss_list_ac", check_game=True) 
            if clicked_boss_list:
                time.sleep(0.5) 
                for button_config in exit_buttons_list:
                    img_file = button_config.get("image_file")
                    if not img_file: continue
                    
                    region_key = button_config.get("region_key")
                    search_region = None
                    if region_key:
                        search_region = self.ui_regions.get(region_key)
                        if not search_region: continue
                    else:
                        search_region = default_region
                    if not search_region: continue
                    
                    search_confidence = float(button_config.get("confidence") or confidence)
                    match = self.find_image_on_screen(img_file, search_region, search_confidence)
                    if match:
                        match_result = match
                        found_button_config = button_config
                        self.log(f"Çıkış butonu '{img_file}' bulundu (fallback sonrası).")
                        break

        if match_result and found_button_config:
            click_x, click_y, max_val = match_result
            button_label = found_button_config.get("image_file", "unknown_exit_button")

            manual_override = False
            with MANUAL_CLICK_LOCK:
                if last_manual_click_time is not None:
                    if (time.monotonic() - last_manual_click_time) <= 3.0:
                        manual_override = True
            
            if manual_override:
                self.log(f"Kullanıcı müdahalesi algılandı ({button_label}). Bot tıklamıyor. (güven: {max_val:.2f})")
                return True
            else:
                try:
                    self.automator.click(
                        label=f"exit_{current_location_name}_{button_label}", 
                        coord_override={"x": click_x, "y": click_y}, 
                        check_game=True
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
                self.log(f"({current_location_name}) Çıkış butonu bulunamadı ama manuel tıklama algılandı -> Başarılı varsayılıyor.")
                return True
            else:
                tried_images = [b.get('image_file', 'N/A') for b in exit_buttons_list]
                self.log(f"({current_location_name}) için uygun çıkış butonu bulunamadı (Denenenler: {tried_images}).")
                return False
                
    def start(self):
        """Otomasyonu başlatır veya duraklatılmışsa devam ettirir."""
        if self.running.is_set() and not self.paused:
            self.log("Otomasyon zaten çalışıyor.")
            return

        if self.bosslar and not any(h.get('spawn_time') for h in self.bosslar.values()):
            try:
                if self.root and self.root.winfo_exists():
                    if not messagebox.askyesno("Zaman Yok",
                                               "Hiçbir boss için başlangıç zamanı girilmedi.\nBaşlatılsın mı?",
                                               parent=self.root):
                        self.log("Başlatma iptal edildi (kullanıcı onayı alınamadı).")
                        return
            except Exception: pass

        if not self._threads_started:
            self.log("Arka plan thread'leri başlatılıyor...")
            try:
                threading.Thread(target=self._game_monitor_thread, daemon=True, name="GameMonitorThread").start()
                threading.Thread(target=self._boss_automation_thread, daemon=True, name="BossAutomationThread").start()
                threading.Thread(target=self.event_manager.event_monitor_thread, daemon=True, name="EventMonitorThread").start()
                threading.Thread(target=self._popup_monitor_thread, daemon=True, name="PopupMonitorThread").start()

                if self.pvp_config.get("enabled", False): 
                    threading.Thread(target=self._pvp_defense_monitor_thread, daemon=True, name="PvPDefenseThread").start()
                self._threads_started = True
            except Exception as e:
                 self.log(f"KRİTİK HATA: Arka plan thread'leri başlatılamadı: {e}")
                 traceback.print_exc()
                 self.initialized = False
                 try:
                      if self.root: self.root.quit()
                 except Exception: pass
                 return 

        self.running.set()
        self.paused = False
        self.log("Otomasyon başlatıldı/devam ettirildi.")

    def stop(self):
        """Otomasyonu duraklatır."""
        if self.paused or not self.running.is_set():
            self.log("Otomasyon zaten duraklatılmış veya çalışmıyor.")
            return
        self.paused = True
        self.log("Otomasyon duraklatıldı.")

    def run(self):
        """Ana program döngüsünü (GUI) başlatır."""
        if not self.initialized:
             self.log("Bot düzgün başlatılamadığı için GUI çalıştırılamıyor.")
             try: messagebox.showerror("Başlatma Hatası", "Bot başlatılamadı.\nLog dosyasını kontrol edin.")
             except Exception: pass 
             try:
                 if self.root: self.root.destroy()
             except Exception: pass
             return 

        try:
            if hasattr(self, 'gui_manager') and self.gui_manager:
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
  
    def _game_monitor_thread(self):
        """Oyunun çalışıp çalışmadığını ve donup donmadığını izler."""
        self.log("Oyun Monitör thread'i başlatıldı.")
        freeze_check_interval = self.settings.get("FREEZE_CHECK_INTERVAL_SN", 15)
        max_freeze_duration = self.settings.get("MAX_FREEZE_DURATION_SN", 60)
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
                    self.log("Oyun prosesi bulunamadı, yeniden başlatma prosedürü tetikleniyor...")
                    with self.action_lock:
                        if not self.is_game_running() and self.running.is_set():
                            self._handle_game_restart() 
                    time.sleep(15) 
                    continue 

                now = time.monotonic()
                is_bot_busy = not self.paused and (self.active_event is not None or self.attacking_target_aciklama is not None)

                if is_bot_busy and freeze_check_region_cfg and (now - self._last_screenshot_time > freeze_check_interval):
                    try:
                        region_coords = (
                            _safe_int(freeze_check_region_cfg.get('x', 800), 800),
                            _safe_int(freeze_check_region_cfg.get('y', 400), 400),
                            _safe_int(freeze_check_region_cfg.get('w', 300), 300),
                            _safe_int(freeze_check_region_cfg.get('h', 200), 200)
                        )
                        with mss.mss() as sct:
                             monitor = {"top": region_coords[1], "left": region_coords[0], "width": region_coords[2], "height": region_coords[3]}
                             shot = sct.grab(monitor)
                             arr = np.array(shot, dtype=np.uint8)
                             current_checksum = int(np.sum(arr) % 1000003)

                        self._last_screenshot_time = now 

                        if self._last_screenshot_checksum is not None and current_checksum == self._last_screenshot_checksum:
                            self._freeze_count += 1
                            freeze_duration = self._freeze_count * freeze_check_interval
                            self.log(f"Donma şüphesi: Ekran {freeze_duration:.1f} saniyedir değişmedi.")
                            if freeze_duration >= max_freeze_duration:
                                self.log(f"MAKSİMUM DONMA SÜRESİ ({max_freeze_duration}sn) AŞILDI! Oyun yeniden başlatılıyor.")
                                self.restart_game()
                                self._freeze_count = 0
                                self._last_screenshot_checksum = None
                                time.sleep(10)
                        else:
                            self._freeze_count = 0
                            self._last_screenshot_checksum = current_checksum
                    except Exception as e:
                        self.log(f"Donma kontrolü sırasında hata: {e}")
                        self._last_screenshot_checksum = None
                        self._freeze_count = 0

                time.sleep(self.settings.get("MAIN_LOOP_SLEEP_SN", 1.0))
            except Exception as e:
                self.log(f"Oyun monitör thread'inde K R İ T İ K hata: {e}")
                traceback.print_exc()
                time.sleep(15)

    def _handle_game_restart(self):
        self.log("Oyun yeniden başlatma prosedürü başlıyor...")
        self._freeze_count = 0
        self._last_screenshot_checksum = None
        self._last_screenshot_time = time.monotonic()
        if not self._start_game():
            self.log("Oyun başlatılamadığı için yeniden başlatma prosedürü durduruldu.")
            return 

        self.log("Oyun başlatıldı/login denendi. Konum tespiti için 5sn bekleniyor...")
        if not self._interruptible_wait(5): return 

        self.location_manager.current_location_value = self.location_manager._get_current_visual_location_value()
        self.log(f"Yeniden başlatma sonrası görsel konum tespiti: {self.location_manager.get_region_name()}")

        if self.active_event:
            event_name = self.active_event.get("name", "Bilinmeyen Etkinlik")
            self.log(f"Yeniden başlatma sonrası aktif etkinlik kontrolü: {event_name}")
            now = datetime.now()
            event = self.active_event
            is_still_due = (event["days"] is None or now.weekday() in event["days"]) and \
                            (event["start_time"] <= now.time() < event["end_time"])

            if is_still_due:
                self.log(f"Etkinlik '{event_name}' hala devam ediyor. Yeniden girilmeye çalışılacak.")
                try:
                    action_type = event.get("action_type", "static_sequence")
                    success = False
                    if action_type == "dynamic_hunt":
                        success = self._run_dynamic_hunt(event)
                    elif action_type == "static_sequence":
                        action_sequence = event.get("action_sequence", [])
                        success = self._run_sequence(
                            action_sequence,
                            self.timed_events_cfg.get("coordinates", {})
                        )
                    
                    if success:
                        self.log(f"Etkinliğe '{event_name}' başarıyla yeniden girildi.")
                        self.location_manager.set_current_location_by_name(event["location_name_on_enter"])
                    else:
                        self.log(f"Etkinliğe '{event_name}' yeniden girilemedi. Exp Farm'a dönülüyor.")
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
        self._freeze_count = 0
        self._last_screenshot_checksum = None
        self._last_screenshot_time = time.monotonic()

    def _popup_monitor_thread(self):
        self.log("Pop-up izleyici thread'i başlatıldı.")
        
        popup_configs = self.popup_config.get("popups_to_close", [])
        region_key = self.popup_config.get("region_key")
        search_region_config = self.ui_regions.get(region_key) if region_key else None
        check_interval = self.popup_config.get("check_interval_sn", 20)

        if not search_region_config or not all(k in search_region_config for k in ('x', 'y', 'w', 'h')):
            self.log(f"Pop-up UYARI: 'region_key' ({region_key}) config'de eksik veya 'ui_regions' içinde bulunamadı. İzleyici pasif.")
            return
        if not popup_configs:
            self.log("Kapatılacak pop-up tanımı bulunamadı. İzleyici pasif.")
            return

        loaded_templates = []
        for cfg in popup_configs:
            img_file = cfg.get("image_file")
            if not img_file: continue
            
            img = self.get_cached_image(img_file) 
            
            if img is not None:
                loaded_templates.append({"template": img, "confidence": float(cfg.get("confidence", 0.7)), "name": img_file})
        
        if not loaded_templates: 
            self.log("Popup monitor için önbellekten hiçbir resim yüklenemedi, thread durduruldu.")
            return

        while True:
            if not self.running.is_set() or self.paused or self.action_lock.locked():
                time.sleep(check_interval / 2)
                continue

            try:
                found_and_clicked = False
                monitor_for_mss = {
                    "top": search_region_config.get("y", 0),
                    "left": search_region_config.get("x", 0),
                    "width": search_region_config.get("w", 1200),
                    "height": search_region_config.get("h", 600)
                }

                with mss.mss() as sct:
                    screenshot_color = sct.grab(monitor_for_mss)
                    screenshot_gray = cv2.cvtColor(np.array(screenshot_color, dtype=np.uint8), cv2.COLOR_BGRA2GRAY)

                for item in loaded_templates:
                    template_gray = item["template"]
                    confidence = item["confidence"]
                    template_name = item["name"]

                    th, tw = template_gray.shape[:2]
                    sh, sw = screenshot_gray.shape[:2]
                    if th > sh or tw > sw: continue

                    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    if max_val >= confidence:
                        click_x = search_region_config.get("x", 0) + max_loc[0] + tw // 2
                        click_y = search_region_config.get("y", 0) + max_loc[1] + th // 2
                        self.log(f"Beklenmedik pencere/buton tespit edildi ('{template_name}'). Kapatılıyor...")

                        try:
                            self.automator.click(
                                label=f"popup_close_{template_name}",
                                coord_override={"x": click_x, "y": click_y},
                                check_game=True
                            )
                            found_and_clicked = True
                            time.sleep(2.0)
                            break 
                        except Exception as click_err:
                             self.log(f"Pop-up kapatma tıklaması sırasında hata: {click_err}")
                             found_and_clicked = True
                             time.sleep(1.0)
                             break 

                sleep_time = 1.0 if found_and_clicked else check_interval
                time.sleep(sleep_time)
            except Exception as e:
                self.log(f"Pop-up izleyici thread'inde hata: {e}")
                time.sleep(check_interval * 2)

    def _pvp_defense_monitor_thread(self):
        if not self.pvp_config.get("enabled", False):
            return

        self.log("PVP Savunma Monitörü başlatılıyor...")
        region_config = self.pvp_config.get("hp_orb_region")
        template_name = self.pvp_config.get("hp_damaged_template")
        confidence = float(self.pvp_config.get("hp_damage_confidence", 0.85))
        skill_rotation = self.pvp_config.get("skill_rotation", [])
        check_interval = float(self.pvp_config.get("check_interval_sn", 0.7))
        cooldown_after_attack = float(self.pvp_config.get("cooldown_after_attack_sn", 3.0))

        if not isinstance(region_config, dict) or not all(k in region_config for k in ('x', 'y', 'w', 'h')):
            self.log("PVP Savunma UYARI: 'hp_orb_region' config'de eksik. Thread pasif.")
            return
        if not template_name or not skill_rotation:
             self.log("PVP Savunma UYARI: 'hp_damaged_template' veya 'skill_rotation' eksik. Thread pasif.")
             return

        damage_template = self.get_cached_image(template_name)
        if damage_template is None:
            self.log(f"PVP Savunma HATA: Hasar şablonu '{template_name}' önbellekte bulunamadı/yüklenemedi. Thread pasif.")
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
                    "top": region_config.get("y", 0),
                    "left": region_config.get("x", 0),
                    "width": region_config.get("w", 100),
                    "height": region_config.get("h", 100)
                }
                with mss.mss() as sct:
                     screenshot_color = sct.grab(monitor_for_mss)
                     screenshot_gray = cv2.cvtColor(np.array(screenshot_color, dtype=np.uint8), cv2.COLOR_BGRA2GRAY)

                result = cv2.matchTemplate(screenshot_gray, damage_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                is_under_attack = (max_val >= confidence)

                if is_under_attack:
                    if not was_under_attack:
                        self.log(f"SALDIRI ALTINDA! (Güven: {max_val:.2f}). Karşı saldırı skilleri kullanılıyor...")
                        was_under_attack = True
                    try:
                        for skill in skill_rotation:
                            pyautogui.press(skill)
                            # Skill tuşlarını kaydet
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
# =============================================================================
# 7) Program Başlangıcı (Ana Çalıştırma Bloğu)
# =============================================================================

if __name__ == "__main__":
    root = None 
    try:
        log_to_file("Program başlatılıyor (v5.8)...")
        root = tk.Tk()
        # root.withdraw() 

        app = LoABot(root)

        if app.initialized:
            app.run()
        else:
             log_to_file("Bot başlatılamadığı için program sonlandırılıyor.")
             if root:
                  try: root.destroy()
                  except: pass

        log_to_file("Program normal şekilde sonlandırıldı.")

    except Exception as e:
        error_msg = f"Program çalıştırılırken K R İ T İ K bir hata oluştu: {e}"
        print(error_msg)
        log_to_file(error_msg)
        traceback.print_exc(file=open(LOG_FILE, "a")) 

        try:
            messagebox.showerror("Kritik Hata", f"Program beklenmedik bir hatayla karşılaştı:\n{e}\nLog dosyasını ({LOG_FILE}) kontrol edin.")
        except Exception:
            pass
        finally:
             if root:
                  try: root.destroy()
                  except: pass