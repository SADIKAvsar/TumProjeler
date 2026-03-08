# modules/popup_manager.py
import time
from utils import log_to_file

class PopupManager:
    def __init__(self, bot):
        self.bot = bot
        self.cfg = bot.popup_config # module_general içindeki POPUP_MONITOR bloğu
        self.popups = self.cfg.get("popups_to_close", []) # Kapatılacak resim listesi

    def monitor_thread(self):
        """Ekrana gelen gereksiz pencereleri periyodik olarak kontrol eder."""
        if not self.popups:
            self.bot.log("PopupManager: Kapatılacak pop-up listesi boş, izleme pasif.")
            return

        self.bot.log("PopupManager: Temizlikçi görev başında.")
        reg_key = self.cfg.get("region_key", "popup_search_area")
        reg = self.bot.ui_regions.get(reg_key, {"x": 500, "y": 200, "w": 1500, "h": 1000})
        interval = self.cfg.get("check_interval_sn", 20)

        while True:
            # Bot duraklatılmışsa veya başka bir kritik aksiyon varsa bekle
            if not self.bot.running.is_set() or self.bot.paused or self.bot.action_lock.locked():
                time.sleep(5)
                continue

            for p in self.popups:
                image_name = p.get("image_file")
                confidence = float(p.get("confidence", 0.7))

                match = self.bot.vision.find(image_name, reg, confidence)

                if match:
                    self.bot.log(f"PopupManager: '{image_name}' tespit edildi, kapatılıyor...")
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
