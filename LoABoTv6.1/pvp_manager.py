# -*- coding: utf-8 -*-
"""
pvp_manager.py — PvP Savunma + Kaçış Yönetimi
================================================
v6.1: Space/q/Space evasion combo boss_manager navigasyonundan
      buraya transfer edildi. Artık yalnızca saldırı altındayken tetiklenir.
"""

import time


class PvPManager:
    def __init__(self, bot):
        self.bot = bot
        self.cfg = bot.pvp_config

    # ── EVASION COMBO ─────────────────────────────────────────────

    def _run_evasion_combo(self):
        """
        Saldırı altındayken kaçış kombosu: Space → q → Space
        (v5.9'da boss_manager._run_navigation_evasion_combo idi)
        """
        self.bot.log(
            "[PVP_EVASION] Saldiri algilandi, kacis kombosu: Space -> q -> Space",
            level="WARNING",
        )
        self.bot.automator.press_key("space", label="pvp_evasion_space_1")
        time.sleep(0.04)
        self.bot.automator.press_key("q", label="pvp_evasion_Q")
        time.sleep(0.04)
        self.bot.automator.press_key("space", label="pvp_evasion_space_2")

    # ── ANA MONITOR THREAD ────────────────────────────────────────

    def monitor_thread(self):
        """
        Hasar algılandığında:
          1. Evasion combo (cooldown ile) — kaçış öncelikli
          2. Skill rotation — saldırı becerileri
        """
        if not self.cfg.get("enabled", False):
            self.bot.log("PvP monitor pasif (enabled=false).")
            return

        region = self.cfg.get("hp_orb_region")
        image_name = self.cfg.get("hp_damaged_template")
        confidence = float(self.cfg.get("hp_damage_confidence", 0.85))
        skill_rotation = self.cfg.get("skill_rotation", ["q", "w", "e", "r"])
        check_interval = float(self.cfg.get("check_interval_sn", 0.7))
        cooldown_after_attack = float(self.cfg.get("cooldown_after_attack_sn", 1.0))
        evasion_cooldown = float(self.cfg.get("evasion_cooldown_sn", 3.0))

        if not isinstance(region, dict) or not image_name or not skill_rotation:
            self.bot.log("PvP monitor config eksik, thread durduruldu.")
            return

        self.bot.log("PvP savunma monitoru aktif (evasion combo dahil).")
        was_under_attack = False
        last_evasion_ts = 0.0

        while True:
            if not self.bot.running.is_set() or self.bot.paused:
                was_under_attack = False
                time.sleep(max(0.5, check_interval))
                continue

            match = self.bot.vision.find(image_name, region, confidence)
            is_under_attack = bool(match)

            if is_under_attack:
                now = time.time()

                if not was_under_attack:
                    self.bot.log("PvP: saldiri algilandi.")
                    was_under_attack = True

                # 1) Evasion combo (cooldown ile)
                if (now - last_evasion_ts) >= evasion_cooldown:
                    self._run_evasion_combo()
                    last_evasion_ts = now
                    time.sleep(0.1)

                # 2) Skill rotation
                for key in skill_rotation:
                    self.bot.automator.press_key(key, label="pvp_rotation")
                    time.sleep(0.05)

                time.sleep(cooldown_after_attack)
                continue

            if was_under_attack:
                self.bot.log("PvP: tehlike gecti.")
                was_under_attack = False

            time.sleep(check_interval)
