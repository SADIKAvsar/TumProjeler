# -*- coding: utf-8 -*-
"""
environment_manager.py — Konum ve Popup Yönetimi
===================================================
v6.1: location_manager.py + popup_manager.py birleştirildi.
"""

import time
from utils import log_to_file


# ══════════════════════════════════════════════════════════════════════
#  LOCATION MANAGER
# ══════════════════════════════════════════════════════════════════════

class LocationManager:
    def __init__(self, bot, general_cfg: dict):
        self.bot = bot
        self.general_cfg = general_cfg
        self.LOCATION_MAP = {}
        self.NAME_TO_VALUE = {}
        self._build_location_maps()
        self.current_location_value = self.NAME_TO_VALUE.get("UNKNOWN", -1)

    def _build_location_maps(self):
        locations = self.general_cfg.get("locations", {})
        for name, data in locations.items():
            try:
                val_int = int(data.get("value", -1))
            except (TypeError, ValueError):
                val_int = -1
            self.LOCATION_MAP[val_int] = str(name).upper()
            self.NAME_TO_VALUE[str(name).upper()] = val_int

    def get_location_name(self, location_value: int) -> str:
        return self.LOCATION_MAP.get(int(location_value), "UNKNOWN")

    def get_region_name(self) -> str:
        return self.get_location_name(self.current_location_value)

    def set_current_location_by_name(self, name: str):
        key = str(name).upper()
        val = self.NAME_TO_VALUE.get(key)
        if val is None:
            self.bot.log(f"UYARI: set_current_location_by_name('{name}') basarisiz.")
            return

        self.current_location_value = val
        self.bot.log(f"KONUM SENKRONIZE: {key}")

    def set_location_by_name(self, name: str):
        """Eski cagrilarla uyum icin alias."""
        self.set_current_location_by_name(name)

    def _get_current_visual_location_value(self) -> int:
        for _, data in self.general_cfg.get("locations", {}).items():
            anchors = data.get("anchors") if isinstance(data.get("anchors"), list) else [data.get("anchor")]
            for anchor in anchors:
                if not anchor or not isinstance(anchor, dict):
                    continue

                img = anchor.get("image_file")
                reg = self.bot.ui_regions.get(anchor.get("region_key"))
                conf = float(anchor.get("confidence", 0.7))

                if img and reg and self.bot.vision.find(img, reg, conf):
                    try:
                        return int(data.get("value", -1))
                    except (TypeError, ValueError):
                        return self.NAME_TO_VALUE.get("UNKNOWN", -1)

        return self.NAME_TO_VALUE.get("UNKNOWN", -1)

    def update_visual_location(self):
        """Ekrandan gorup konumu gunceller."""
        new_val = self._get_current_visual_location_value()
        unknown_val = self.NAME_TO_VALUE.get("UNKNOWN", -1)

        if new_val != unknown_val and new_val != self.current_location_value:
            old_name = self.get_region_name()
            self.current_location_value = new_val
            self.bot.log(f"KONUM SENKRON: '{old_name}' -> '{self.get_region_name()}'")

        return self.current_location_value


# ══════════════════════════════════════════════════════════════════════
#  POPUP MANAGER
# ══════════════════════════════════════════════════════════════════════

class PopupManager:
    def __init__(self, bot):
        self.bot = bot
        self.cfg = bot.popup_config
        self.popups = self.cfg.get("popups_to_close", [])

    def monitor_thread(self):
        """Ekrana gelen gereksiz pencereleri periyodik olarak kontrol eder."""
        if not self.popups:
            self.bot.log("PopupManager: Kapatilacak pop-up listesi bos, izleme pasif.")
            return

        self.bot.log("PopupManager: Temizlikci gorev basinda.")
        reg_key = self.cfg.get("region_key", "popup_search_area")
        reg = self.bot.ui_regions.get(reg_key, {"x": 500, "y": 200, "w": 1500, "h": 1000})
        interval = self.cfg.get("check_interval_sn", 20)

        while True:
            if not self.bot.running.is_set() or self.bot.paused or self.bot.is_action_lock_busy():
                time.sleep(5)
                continue

            for p in self.popups:
                image_name = p.get("image_file")
                confidence = float(p.get("confidence", 0.7))

                match = self.bot.vision.find(image_name, reg, confidence)

                if match:
                    self.bot.log(f"PopupManager: '{image_name}' tespit edildi, kapatiliyor...")
                    self.bot.automator.click(f"popup_{image_name}", {"x": match[0], "y": match[1]})
                    time.sleep(1)

            time.sleep(interval)

    def close_all_popups(self):
        """
        Tek seferlik popup temizligi.
        Donus: En az bir popup kapatildiysa True.
        """
        if not self.popups:
            return False

        reg_key = self.cfg.get("region_key", "popup_search_area")
        reg = self.bot.ui_regions.get(reg_key, {"x": 500, "y": 200, "w": 1500, "h": 1000})
        closed_any = False

        for p in self.popups:
            image_name = p.get("image_file")
            confidence = float(p.get("confidence", 0.7))
            if not image_name:
                continue

            match = self.bot.vision.find(image_name, reg, confidence)
            if match:
                self.bot.automator.click(f"popup_{image_name}", {"x": match[0], "y": match[1]})
                closed_any = True
                time.sleep(0.2)

        return closed_any
