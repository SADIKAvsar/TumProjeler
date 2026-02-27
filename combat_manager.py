import time


class CombatManager:
    def __init__(self, bot):
        self.bot = bot

    # =========================================================
    # DURUM SORGULAMA
    # =========================================================

    def is_in_active_combat(self) -> bool:
        """
        Bot'un o an fiziksel olarak bir boss'a saldirip saldirmadığını döner.

        Sadece 'attacking_target_aciklama' flagini değil; navigasyon,
        dövüş VE loot fazlarını birlikte kontrol eder.
        Etkinlik girişi öncesi bu metod 'False' döndürene kadar beklemek
        gereklidir (State Lock'u önler).
        """
        attacking = bool(getattr(self.bot, "attacking_target_aciklama", None))
        phase = str(getattr(self.bot, "_global_phase", "")).strip().upper()
        # NAV + COMBAT + LOOT → hepsi aktif dövüş zincirine dahil
        in_active_phase = phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}
        return attacking or in_active_phase

    # =========================================================
    # KESILMIŞ DÖVÜŞ FALLBACK
    # =========================================================

    def recalculate_times_interrupted(self, interrupted_boss: dict) -> None:
        """
        Boss dövüşü etkinlik veya timeout nedeniyle kesildiğinde çağrılır.
        Boss öldürülmedi; ancak spawn_time stale (geçmiş) kalmamalı.

        Kural:
        - Spawn_time geçmişte kaldıysa → now + periyot ile shift et.
        - Spawn_time henüz gelmemişse → dokunma (doğal zamanını beklesin).
        """
        periyot = float(interrupted_boss.get("periyot_saat", 0)) * 3600.0
        if periyot <= 0:
            return

        now = time.time()
        current_spawn = interrupted_boss.get("spawn_time")

        # Spawn zamanı zaten geçmişse (boss spawn etmişti ama biz kesilmek zorunda kaldık)
        if not isinstance(current_spawn, (int, float)) or current_spawn <= now:
            yeni_spawn = now + periyot
            with self.bot.action_lock:
                self.bot._set_spawn_time_abs(
                    interrupted_boss["aciklama"], yeni_spawn, source="interrupted_fallback"
                )
            self.bot.log(
                f"[INTERRUPT] Spawn fallback: {interrupted_boss['aciklama']} -> "
                f"{time.strftime('%H:%M:%S', time.localtime(yeni_spawn))} "
                f"(dovus kesildi, bir sonraki periyot hesaplandi)"
            )
        else:
            # Spawn zamanı ileride → dokunma, bot zamanında tekrar dener
            self.bot.log(
                f"[INTERRUPT] Spawn korundu: {interrupted_boss['aciklama']} "
                f"(spawn henüz gelmemişti, aynı zaman geçerli)",
                level="DEBUG",
            )

    def _is_next_boss_urgent(self, current_boss=None) -> bool:
        # DRY: Birincil mantik BossManager'da tutulur; oraya delege et.
        bm = getattr(self.bot, "boss_manager", None)
        if bm is not None:
            return bm._is_next_boss_urgent(current_boss=current_boss)

        # Fallback: boss_manager henuz hazir degilse (erken init) yerel hesapla.
        now = time.time()
        urgent_window = float(self.bot.settings.get("URGENT_NEXT_BOSS_THRESHOLD_SN", 15.0))
        candidates = []
        for b in self.bot.bosslar.values():
            if current_boss and b.get("aciklama") == current_boss.get("aciklama"):
                continue
            spawn_ts = b.get("spawn_time")
            if not isinstance(spawn_ts, (int, float)):
                continue
            head_start = float(b.get("head_start_saniye", 0))
            candidates.append((spawn_ts - head_start) - now)
        if not candidates:
            return False
        return min(candidates) <= urgent_window

    def _should_skip_optional_check(
        self,
        required: bool,
        current_boss,
        check_name: str,
        check_type: str = "",
    ) -> bool:
        if required:
            return False

        # Yeni kural:
        # area_check OK ise spawn/victory kontrolu asla skip edilmez.
        if check_type in {"spawn_check", "victory"} and bool((current_boss or {}).get("_area_check_ok", False)):
            return False

        if self._is_next_boss_urgent(current_boss=current_boss):
            self.bot.log(f"Zaman dar, opsiyonel kontrol atlandi: {check_name}", level="WARNING")
            return True
        return False

    def _normalize_image_files(self, cfg: dict) -> dict:
        image_files = cfg.get("image_files")
        if isinstance(image_files, str):
            cfg["image_files"] = [image_files]
        elif not isinstance(image_files, list):
            image_file = cfg.get("image_file")
            cfg["image_files"] = [image_file] if isinstance(image_file, str) else []

        cfg["image_files"] = [f for f in cfg.get("image_files", []) if f]
        return cfg

    def _resolve_victory_check(self, boss) -> dict:
        """YAML'dan zafer gorseli ayarlarini guvenli bicimde cozer."""
        cfg = {}
        defaults = getattr(self.bot, "victory_check_defaults", {}) or {}
        if isinstance(defaults, dict):
            cfg.update(defaults)

        # Yeni yapi: victory: {image_file, region_key, confidence, ...}
        victory_data = boss.get("victory", {})
        if isinstance(victory_data, dict):
            image = victory_data.get("image_file")
            if image:
                cfg["image_files"] = [image]
            cfg["confidence"] = float(victory_data.get("confidence", cfg.get("confidence", 0.4)))
            cfg["region_key"] = victory_data.get("region_key", cfg.get("region_key", "region_full_screen"))
            if "timeout_sn" in victory_data:
                cfg["timeout_sn"] = float(victory_data.get("timeout_sn", 6.0))
            if "poll_interval_sn" in victory_data:
                cfg["poll_interval_sn"] = float(victory_data.get("poll_interval_sn", 0.4))
            if "required" in victory_data:
                cfg["required"] = bool(victory_data.get("required", False))

        boss_cfg = boss.get("victory_check")
        if isinstance(boss_cfg, dict):
            cfg.update(boss_cfg)

        # Kisa kullanim: victory_image: "victory_800.png"
        if boss.get("victory_image"):
            cfg["image_files"] = [boss.get("victory_image")]

        return self._normalize_image_files(cfg)

    def _resolve_spawn_check(self, boss) -> dict:
        """Spawn ayarlari: defaults + boss spawn_check."""
        cfg = {}
        defaults = getattr(self.bot, "demon_boss_cfg", {}).get("spawn_check_defaults", {}) or {}
        if isinstance(defaults, dict):
            cfg.update(defaults)

        boss_cfg = boss.get("spawn_check")
        if isinstance(boss_cfg, dict):
            cfg.update(boss_cfg)

        return self._normalize_image_files(cfg)

    def _confirm_spawn_ready(self, boss) -> bool:
        spawn_ts = boss.get("spawn_time")
        if not isinstance(spawn_ts, (int, float)):
            return False

        cfg = self._resolve_spawn_check(boss)
        enabled = bool(cfg.get("enabled", self.bot.settings.get("SPAWN_CONFIRM_ENABLED", True)))
        image_files = cfg.get("image_files", [])
        required = bool(cfg.get("required", self.bot.settings.get("SPAWN_CONFIRM_REQUIRED", False)))
        pre_window_sn = float(cfg.get("pre_window_sn", self.bot.settings.get("SPAWN_CONFIRM_PRE_WINDOW_SN", 4.0)))
        # 1.5 saniye kurali (sabit)
        timeout_sn = 1.5
        # 1.5 saniyede olabildigince fazla kare icin hizli poll
        poll_interval_sn = min(float(cfg.get("poll_interval_sn", 0.25)), 0.05)

        # Spawn dogrulamasi kapaliysa klasik zaman bazli bekle.
        if not enabled or not image_files:
            return self.bot._interruptible_wait(max(0, spawn_ts - time.time()))

        wait_before_probe = max(0, (spawn_ts - pre_window_sn) - time.time())
        if wait_before_probe > 0 and not self.bot._interruptible_wait(wait_before_probe):
            return False

        region_key = cfg.get("region_key", "region_full_screen")
        region = self.bot.ui_regions.get(region_key) or self.bot.ui_regions.get(
            "region_full_screen",
            {"x": 0, "y": 0, "w": 2560, "h": 1440},
        )
        confidence = float(cfg.get("confidence", 0.70))

        # T=0 anina kadar bekle, sonra sabit 1.5sn spawn penceresi calistir.
        wait_to_spawn = max(0, spawn_ts - time.time())
        if wait_to_spawn > 0 and not self.bot._interruptible_wait(wait_to_spawn):
            return False
        probe_duration = timeout_sn
        end_t = time.monotonic() + timeout_sn

        self.bot.log(
            f"Spawn kontrol ({boss['aciklama']}): {', '.join(image_files)} "
            f"| conf={confidence:.2f} timeout={probe_duration:.1f}s",
            level="DEBUG",
        )

        found_any = False
        while time.monotonic() < end_t:
            if not self.bot.running.is_set():
                return False

            for image_name in image_files:
                try:
                    if self.bot.vision.find(
                        image_name,
                        region,
                        confidence,
                        target_data=boss,
                        stage="spawn_check",
                        shadow_force_capture=True,
                    ):
                        found_any = True
                        self.bot.log(f"Spawn bulundu: {image_name}", level="DEBUG")
                except Exception as exc:
                    self.bot.log(f"Spawn arama hatasi ({image_name}): {exc}", level="WARNING")

            if self._should_skip_optional_check(
                required=required,
                current_boss=boss,
                check_name=f"Spawn kontrol ({boss['aciklama']})",
                check_type="spawn_check",
            ):
                return True

            remaining = end_t - time.monotonic()
            if remaining > 0 and not self.bot._interruptible_wait(min(max(0.01, poll_interval_sn), remaining)):
                return False

        if found_any:
            return True

        self.bot.log(f"Spawn bulunamadi ({boss['aciklama']}). required={required}", level="WARNING")
        return not required

    def execute_boss_attack(self, target) -> bool:
        # Legacy akis area_check yapmadigi icin stale bayrak kalmasin.
        target["_area_check_ok"] = False
        self.bot.start_global_mission(
            phase="NAV_PHASE",
            stage="anchor_click",
            reason=f"legacy_boss_{target.get('aciklama')}",
            extra={"boss_id": str(target.get("aciklama", "unknown"))},
        )
        self.bot.set_global_mission_phase(
            phase="NAV_PHASE",
            stage="anchor_click",
            reason=f"legacy_boss_{target.get('aciklama')}",
            extra={"boss_id": str(target.get("aciklama", "unknown"))},
        )

        current_region = self.bot.location_manager.get_region_name()
        target_loc = "KATMAN_1" if "katman_1" in str(target.get("katman_id", "")).lower() else "KATMAN_2"

        if current_region == target_loc:
            sequence = [
                {"action": "click", "label": "boss_list_ac", "wait_ms": 200},
                {"action": "boss_secimi", "wait_ms": 200},
            ]
            success = self.bot._run_sequence(sequence, self.bot.coordinates, context_target=target)
        else:
            if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=True):
                return False
            success = self.bot._run_sequence(self.bot.boss_sequence_template, self.bot.coordinates, context_target=target)
            if success:
                self.bot.location_manager.set_current_location_by_name(target_loc)

        if not success:
            self.bot.log_training_outcome(
                "boss_attack",
                {"boss_id": str(target.get("aciklama", "unknown")), "success": False, "reason": "navigation_failed"},
            )
            return False

        self.bot.automator.press_key("z", label="combat_auto_mode")
        if not self._confirm_spawn_ready(target):
            self.bot.log_training_outcome(
                "boss_attack",
                {"boss_id": str(target.get("aciklama", "unknown")), "success": False, "reason": "spawn_not_confirmed"},
            )
            return False

        self.bot.set_global_mission_phase(
            phase="COMBAT_PHASE",
            stage="attack_start",
            reason=f"combat_{target.get('aciklama')}",
            extra={"boss_id": str(target.get("aciklama", "unknown"))},
        )
        self.bot.automator.press_key("a", label="combat_attack_key")
        self.bot.log(f"Boss dogrulama basladi: {target['aciklama']}")
        kill_ok = self.verify_kill(target)
        self.bot.log(f"Boss dogrulama sonucu ({target['aciklama']}): {'OK' if kill_ok else 'FAIL'}")
        self.bot.log_training_outcome(
            "boss_attack",
            {"boss_id": str(target.get("aciklama", "unknown")), "success": bool(kill_ok), "reason": "verify_kill"},
        )
        return bool(kill_ok)

    def verify_kill(self, boss) -> bool:
        self.bot.log(f"Ganimet araniyor: {boss['aciklama']}")
        cfg = self._resolve_victory_check(boss)
        image_files = cfg.get("image_files", [])

        if image_files:
            region_key = cfg.get("region_key", "region_full_screen")
            region = self.bot.ui_regions.get(region_key) or self.bot.ui_regions.get(
                "region_full_screen",
                {"x": 0, "y": 0, "w": 2560, "h": 1440},
            )
            confidence = float(cfg.get("confidence", 0.75))
            # 1.5 saniye kurali (sabit)
            timeout_sn = 1.5
            # 1.5 saniyede daha fazla kare yakalamak icin hizli poll
            poll_interval_sn = min(float(cfg.get("poll_interval_sn", 0.4)), 0.05)
            required = bool(cfg.get("required", False))

            end_t = time.monotonic() + max(0.5, timeout_sn)
            self.bot.log(
                f"Victory kontrol ({boss['aciklama']}): {', '.join(image_files)} "
                f"| conf={confidence:.2f} timeout={timeout_sn:.1f}s",
                level="DEBUG",
            )

            found_any = False
            while time.monotonic() < end_t:
                if not self.bot.running.is_set():
                    return False

                for image_name in image_files:
                    try:
                        if self.bot.vision.find(
                            image_name,
                            region,
                            confidence,
                            target_data=boss,
                            stage="victory",
                            shadow_force_capture=True,
                        ):
                            found_any = True
                            self.bot.log(f"Victory bulundu: {image_name}", level="DEBUG")
                    except Exception as exc:
                        self.bot.log(f"Victory arama hatasi ({image_name}): {exc}", level="WARNING")

                if self._should_skip_optional_check(
                    required=required,
                    current_boss=boss,
                    check_name=f"Victory kontrol ({boss['aciklama']})",
                    check_type="victory",
                ):
                    return True

                remaining = end_t - time.monotonic()
                if remaining > 0 and not self.bot._interruptible_wait(min(max(0.05, poll_interval_sn), remaining)):
                    return False

            if found_any:
                return True

            self.bot.log(f"Victory bulunamadi ({boss['aciklama']}). required={required}", level="WARNING")
            return not required

        # Victory gorseli tanimli degilse fallback: kisa bekleme.
        return self.bot._interruptible_wait(4.0)

    def run_sequence(self, sequence, target=None) -> bool:
        return self.bot._run_sequence(sequence, self.bot.coordinates, context_target=target)

    def wait_post_attack_loot(self) -> bool:
        loot_wait = float(self.bot.settings.get("POST_ATTACK_WAIT_SN", 30))
        self.bot.set_global_mission_phase(
            phase="LOOT_PHASE",
            stage="loot_wait",
            reason=f"loot_wait_{int(loot_wait)}s",
        )
        self.bot.log(f"Ganimet toplama bekleme: {int(loot_wait)} sn")
        return self.bot._interruptible_wait(loot_wait)

    def recalculate_times(self, killed_boss, attack_start):
        periyot = float(killed_boss.get("periyot_saat", 0)) * 3600.0
        if periyot <= 0:
            return

        spawn_mode = str(self.bot.settings.get("SPAWN_RECALC_MODE", "fixed_cycle")).strip().lower()
        planned_spawn = killed_boss.get("spawn_time")
        now = time.time()

        # --- DRIFT-PROOF (KAYMA KORUMALI) ZAMAN HESAPLAMA ---
        if spawn_mode == "fixed_cycle" and isinstance(planned_spawn, (int, float)) and planned_spawn > 0:
            # Sıkı referans: Kesinlikle bir önceki planlı zamanın üzerine ekle
            yeni_spawn = planned_spawn + periyot
            
            # Catch-up (Yetişme) Mantığı: Eğer bot kapalı kalmışsa veya bir boss atlandıysa,
            # şimdiki zamanı (now) geçene kadar periyot ekleyerek oyun saatiyle tam senkron ol.
            while yeni_spawn <= now:
                yeni_spawn += periyot
        else:
            # Eğer kill_time modu seçiliyse veya ilk kesimse mecburen şimdiki anı referans al
            yeni_spawn = now + periyot

        # Thread-safe yazma: automation_thread ayni anda bot.bosslar okuyabilir.
        with self.bot.action_lock:
            self.bot._set_spawn_time_abs(killed_boss["aciklama"], yeni_spawn, source="automation")
        self.bot.log(
            f"Spawn guncellendi: {killed_boss['aciklama']} -> "
            f"{time.strftime('%H:%M:%S', time.localtime(yeni_spawn))}"
        )

        # Isteyenler icin eski davranis (diger boss timerlarini loot/yurume ile kaydirma):
        # Varsayilan kapali, cunku gercek spawn zamaninda drift biriktirebilir.
        if bool(self.bot.settings.get("ADJUST_OTHER_BOSS_TIMERS_FOR_LOOT", False)):
            loot_finish = attack_start + float(self.bot.settings.get("POST_ATTACK_WAIT_SN", 30))
            with self.bot.action_lock:
                for next_b in self.bot.bosslar.values():
                    if next_b["katman_id"] == killed_boss["katman_id"] and next_b["aciklama"] != killed_boss["aciklama"]:
                        next_spawn = next_b.get("spawn_time")
                        if not isinstance(next_spawn, (int, float)):
                            continue
                        walk_t = self.get_walk_time(killed_boss["aciklama"], next_b["aciklama"])
                        if walk_t > (next_spawn - loot_finish):
                            drift = walk_t - (next_spawn - loot_finish)
                            self.bot._set_spawn_time_abs(next_b["aciklama"], next_spawn + drift, source="automation")
                            self.bot.log(
                                f"Spawn kaydirma uygulandi: {next_b['aciklama']} (+{int(drift)}s)",
                                level="DEBUG",
                            )

    def get_walk_time(self, a, b):
        if not bool(self.bot.settings.get("WALK_TIME_ENABLED", False)):
            return int(self.bot.settings.get("WALK_TIME_DISABLED_VALUE_SN", 0))

        t = self.bot.walk_times.get(str(a), {}).get(str(b)) or self.bot.walk_times.get(str(b), {}).get(str(a))
        return int(t) if t is not None else int(self.bot.settings.get("WALK_TIME_DEFAULT_SN", 40))

    def check_strategic_wait(self, current):
        now = time.time()
        upc = sorted(
            [h for h in self.bot.bosslar.values() if h["spawn_time"] and h["spawn_time"] > now],
            key=lambda x: x["spawn_time"],
        )

        if not upc:
            self.bot.automator.return_to_exp_farm()
            return

        next_boss = upc[0]
        time_until_next = next_boss["spawn_time"] - now

        ai_engine = getattr(self.bot, "brain", None)
        ai_engine = getattr(ai_engine, "ai_engine", None) if ai_engine else None

        if ai_engine:
            try:
                decision = ai_engine.evaluate_strategic_wait(
                    current_boss=current,
                    next_boss=next_boss,
                    time_until_next=time_until_next,
                )
            except Exception as exc:
                self.bot.log(f"Stratejik bekleme AI hatasi: {exc}")
                decision = None

            if decision:
                action = decision.get("decision", "return_to_farm")
                self.bot.log(
                    f"AI Stratejik Karar: {action} ({next_boss['aciklama']}, {int(time_until_next)}s, "
                    f"guven: {decision.get('confidence', 0):.0%})"
                )

                if action == "wait":
                    self.bot.log(f"AI: {next_boss['aciklama']} icin haritada kaliniyor.")
                    return

                self.bot.automator.return_to_exp_farm()
                return

        threshold = float(self.bot.settings.get("BOSS_SWITCH_THRESHOLD_SN", 91))
        if next_boss["katman_id"] == current["katman_id"] and time_until_next < threshold:
            self.bot.log(f"Stratejik Bekleme: {next_boss['aciklama']} icin haritada kaliniyor.")
            return

        self.bot.automator.return_to_exp_farm()
