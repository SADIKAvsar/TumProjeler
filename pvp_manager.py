import time



class PvPManager:
    def __init__(self, bot):
        self.bot = bot
        self.cfg = bot.pvp_config

    def monitor_thread(self):
        """Hasar sarti saglandiginda skill rotasyonunu donguye sokar."""
        if not self.cfg.get("enabled", False):
            self.bot.log("PvP monitor pasif (enabled=false).")
            return

        region = self.cfg.get("hp_orb_region")
        image_name = self.cfg.get("hp_damaged_template")
        confidence = float(self.cfg.get("hp_damage_confidence", 0.85))
        skill_rotation = self.cfg.get("skill_rotation", ["q", "w", "e", "r"])
        check_interval = float(self.cfg.get("check_interval_sn", 0.7))
        cooldown_after_attack = float(self.cfg.get("cooldown_after_attack_sn", 3.0))

        if not isinstance(region, dict) or not image_name or not skill_rotation:
            self.bot.log("PvP monitor config eksik, thread durduruldu.")
            return

        self.bot.log("PvP savunma monitoru aktif.")
        was_under_attack = False

        while True:
            if not self.bot.running.is_set() or self.bot.paused:
                was_under_attack = False
                time.sleep(max(0.5, check_interval))
                continue

            match = self.bot.vision.find(image_name, region, confidence)
            is_under_attack = bool(match)

            if is_under_attack:
                if not was_under_attack:
                    self.bot.log("PvP: saldiri algilandi, skill dongusu baslatildi.")
                    was_under_attack = True

                for key in skill_rotation:
                    self.bot.automator.press_key(key, label="pvp_rotation")
                    time.sleep(0.05)

                time.sleep(cooldown_after_attack)
                continue

            if was_under_attack:
                self.bot.log("PvP: tehlike gecti.")
                was_under_attack = False

            time.sleep(check_interval)
