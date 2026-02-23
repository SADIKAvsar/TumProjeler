import time


class BossManager:
    def __init__(self, bot):
        self.bot = bot
        self.bosslar = bot.bosslar
        self.settings = bot.settings

    def _resolve_target_coordinates(self, target):
        """
        Hedef boss koordinatini yeni/eski yapiya gore cozer.
        - koordinat_ref: {x, y}
        - koordinat_ref: "boss_800"
        - koordinat: {x, y}
        """
        ref = target.get("koordinat_ref")
        if isinstance(ref, dict):
            return ref
        if isinstance(ref, str):
            coord = self.bot.coordinates.get(ref)
            if isinstance(coord, dict):
                return coord
        coord = target.get("koordinat")
        return coord if isinstance(coord, dict) else {}

    def is_next_boss_urgent(self, current_target=None) -> bool:
        now = time.time()
        urgent_window = float(self.settings.get("URGENT_NEXT_BOSS_THRESHOLD_SN", 15.0))
        candidates = []

        for b in self.bosslar.values():
            if current_target and b.get("aciklama") == current_target.get("aciklama"):
                continue
            spawn_ts = b.get("spawn_time")
            if not isinstance(spawn_ts, (int, float)):
                continue
            head_start = float(b.get("head_start_saniye", 0))
            ready_at = spawn_ts - head_start
            candidates.append(ready_at - now)

        if not candidates:
            return False

        return min(candidates) <= urgent_window

    def smart_check(self, check_cfg, current_target=None, check_name="check", stage="generic_check") -> bool:
        """
        Gorsel kontrol:
        - Buldugu an cikar.
        - Opsiyonel kontrolde zaman darsa erken atlar.
        """
        image_file = check_cfg.get("image_file")
        if not image_file or image_file == "default.png":
            return True

        region = self.bot.ui_regions.get(
            check_cfg.get("region_key", "region_full_screen"),
            self.bot.ui_regions.get("region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440}),
        )
        confidence = float(check_cfg.get("confidence", 0.70))
        timeout = float(check_cfg.get("timeout_sn", 3.0))
        poll = float(check_cfg.get("poll_interval_sn", 0.5))
        required = bool(check_cfg.get("required", True))

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if not self.bot.running.is_set():
                return False

            if self.bot.vision.find(
                image_file,
                region,
                confidence,
                target_data=current_target,
                stage=stage,
            ):
                self.bot.log(f"{check_name}: bulundu ({image_file})", level="DEBUG")
                return True

            if not required and self.is_next_boss_urgent(current_target=current_target):
                self.bot.log(f"Zaman dar, opsiyonel kontrol atlandi: {check_name}", level="WARNING")
                break

            if not self.bot._interruptible_wait(min(poll, max(0.05, timeout))):
                return False

        self.bot.log(f"{check_name}: bulunamadi ({image_file})", level="DEBUG")
        return False

    def _smart_check_images(self, cfg, current_target=None, check_name="check", stage="generic_check") -> bool:
        """
        image_files listesi icin smart_check mantigi.
        """
        image_files = cfg.get("image_files", [])
        if not image_files:
            return True

        required = bool(cfg.get("required", False))
        timeout = float(cfg.get("timeout_sn", 2.0))
        poll = float(cfg.get("poll_interval_sn", 0.2))
        confidence = float(cfg.get("confidence", 0.70))
        region = self.bot.ui_regions.get(
            cfg.get("region_key", "region_full_screen"),
            self.bot.ui_regions.get("region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440}),
        )

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if not self.bot.running.is_set():
                return False

            for image_name in image_files:
                if self.bot.vision.find(
                    image_name,
                    region,
                    confidence,
                    target_data=current_target,
                    stage=stage,
                ):
                    self.bot.log(f"{check_name}: bulundu ({image_name})", level="DEBUG")
                    return True

            if not required and self.is_next_boss_urgent(current_target=current_target):
                self.bot.log(f"Zaman dar, opsiyonel kontrol atlandi: {check_name}", level="WARNING")
                return True

            if not self.bot._interruptible_wait(min(poll, max(0.05, timeout))):
                return False

        self.bot.log(f"{check_name}: bulunamadi", level="DEBUG")
        return not required

    def _start_navigation(self, target) -> bool:
        """
        Boss listesi ve hedef boss tiklamasini baslatir.
        """
        current_region = self.bot.location_manager.get_region_name()
        target_loc = "KATMAN_1" if "katman_1" in str(target.get("katman_id", "")).lower() else "KATMAN_2"

        if current_region == target_loc:
            seq = [
                {"action": "click", "label": "boss_list_ac", "wait_ms": 200},
                {"action": "boss_secimi", "wait_ms": 200},
            ]
            ok = self.bot._run_sequence(seq, self.bot.coordinates, context_target=target)
            if ok:
                self.bot.location_manager.set_current_location_by_name(target_loc)
            return ok

        if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=True):
            return False

        ok = self.bot._run_sequence(self.bot.boss_sequence_template, self.bot.coordinates, context_target=target)
        if ok:
            self.bot.location_manager.set_current_location_by_name(target_loc)
        return ok

    def _re_anchor(self, target) -> bool:
        """
        Area MISS durumunda hedef tiklamasini tazeler (kisa sekans).
        """
        seq = [
            {"action": "click", "label": "boss_list_ac", "wait_ms": 120},
            {"action": "boss_secimi", "wait_ms": 120},
        ]
        return self.bot._run_sequence(seq, self.bot.coordinates, context_target=target)

    def execute_precise_boss_flow(self, target) -> bool:
        """
        Esnek zaman pencereli hassas akis:
        - T-PRE: Tarama penceresi acilir (SCAN_WINDOW_PRE_SN, varsayilan 5sn)
        - T-PRE..T=0 : Area tarama (bir kez bulununca yeniden aramaz)
        - T=0..T+POST: Spawn + saldiri + victory (SCAN_WINDOW_POST_SN, varsayilan 5sn)
        - Victory: Erken cikis — 10sn pencereyi bekleme
        """
        if not hasattr(self.bot, "combat"):
            return False

        coords = self._resolve_target_coordinates(target)
        tx, ty = coords.get("x"), coords.get("y")
        if tx is not None and ty is not None:
            self.bot.log(f"{target['aciklama']} yoluna cikildi. Hedef: {tx}, {ty}")

        self.bot.start_global_mission(
            phase="NAV_PHASE",
            stage="anchor_click",
            reason=f"boss_{target.get('aciklama')}_start",
            extra={"boss_id": str(target.get("aciklama", "unknown"))},
        )
        self.bot.set_global_mission_phase(
            phase="NAV_PHASE",
            stage="anchor_click",
            reason=f"boss_{target.get('aciklama')}_start",
            extra={"boss_id": str(target.get("aciklama", "unknown"))},
        )

        if not self._start_navigation(target):
            self.bot.log_training_outcome(
                "boss_flow",
                {"boss_id": str(target.get("aciklama", "unknown")), "success": False, "reason": "navigation_failed"},
            )
            return False

        self.bot.automator.press_key("z")

        spawn_ts = target.get("spawn_time")
        if not isinstance(spawn_ts, (int, float)):
            return False

        # Planlı spawn zamanı = saldırı referans noktası (recalculate_times için)
        attack_start = float(spawn_ts)

        kill_ok = self._unified_spawn_sequence(target, spawn_ts)
        self.bot.log(f"Boss sonucu ({target['aciklama']}): {'OK' if kill_ok else 'FAIL'}")
        self.bot.log_training_outcome(
            "boss_flow",
            {"boss_id": str(target.get("aciklama", "unknown")), "success": bool(kill_ok), "reason": "flex_scan"},
        )

        if kill_ok:
            self.bot.combat.recalculate_times(target, attack_start)
        return bool(kill_ok)

    def _unified_spawn_sequence(self, target, spawn_ts: float) -> bool:
        """
        T-PRE / T+POST esnek pencere tarayıcısı.

        Hiyerarşik durum makinesi:
          area → spawn → saldırı → victory

        Kurallar:
          - Bir görsel bulununca o durum True kalır, yeniden aranmaz.
          - Victory yakalanınca döngü anında kırılır (erken çıkış).
          - Aksiyon bitince tüm durumlar otomatik sıfırlanır (yerel değişkenler).
          - 200ms poll → saniyede 5 kontrol, UI/sistem kasma yok.

        Args:
            target   : Boss hedef dict'i
            spawn_ts : Beklenen doğma Unix zamanı

        Returns:
            True  → kill onaylandı (ya da onay gerektirmeyen config)
            False → zaman aşımı, bot durumu veya kritik hata
        """
        POLL = 0.1  # 100ms — non-blocking hassasiyet
        PRE  = float(self.settings.get("SCAN_WINDOW_PRE_SN",  5.0))  # T-5
        POST = float(self.settings.get("SCAN_WINDOW_POST_SN", 5.0))  # T+5

        window_open_ts  = spawn_ts - PRE
        window_close_ts = spawn_ts + POST

        # ── Görsel konfigürasyonlarını hazırla ──────────────────────────
        try:
            area_data = self.bot.vision.get_boss_visual_data(target, "area_check") or {}
        except Exception:
            area_data = {}

        spawn_cfg   = self.bot.combat._resolve_spawn_check(target)
        victory_cfg = self.bot.combat._resolve_victory_check(target)

        area_image   = area_data.get("image_file", "")
        area_conf    = float(area_data.get("confidence", 0.70))

        spawn_images  = spawn_cfg.get("image_files", [])
        spawn_conf    = float(spawn_cfg.get("confidence", 0.60))
        spawn_enabled = bool(spawn_cfg.get(
            "enabled", self.bot.settings.get("SPAWN_CONFIRM_ENABLED", True)
        ))

        victory_images   = victory_cfg.get("image_files", [])
        victory_conf     = float(victory_cfg.get("confidence", 0.40))
        victory_required = bool(victory_cfg.get("required", False))

        region = self.bot.ui_regions.get(
            "region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440}
        )

        # ── Durum bayrakları (yerel → her çağrıda sıfırlanır) ───────────
        area_found    = False
        spawn_found   = False
        attack_done   = False
        victory_found = False
        re_anchor_done = False

        target["_area_check_ok"] = False

        # ── T-PRE anına kadar bekle ──────────────────────────────────────
        wait_pre = window_open_ts - time.time()
        if wait_pre > 0 and not self.bot._interruptible_wait(wait_pre):
            return False

        self.bot.log(
            f"[FlexScan] Pencere ACIK | {target['aciklama']} | "
            f"T-{PRE:.0f}..T+{POST:.0f} | "
            f"area={bool(area_image and area_image != 'default.png')} "
            f"spawn={len(spawn_images)} victory={len(victory_images)}"
        )

        # ── Ana tarama döngüsü ───────────────────────────────────────────
        while self.bot.running.is_set():
            now = time.time()
            if now >= window_close_ts:
                break

            # ── 1. AREA ─────────────────────────────────────────────────
            # T-PRE'dan itibaren taranır; bir kez bulununca tekrar aranmaz.
            if not area_found and area_image and area_image != "default.png":
                if self.bot.vision.find(
                    area_image, region, area_conf,
                    target_data=target, stage="area_check",
                ):
                    area_found = True
                    target["_area_check_ok"] = True
                    self.bot.log(f"[FlexScan] AREA bulundu: {area_image}")

            # ── 2. SPAWN ────────────────────────────────────────────────
            # T=0'dan itibaren taranır; bir kez bulununca tekrar aranmaz.
            if not spawn_found and spawn_enabled and spawn_images and now >= spawn_ts:
                for img in spawn_images:
                    if self.bot.vision.find(
                        img, region, spawn_conf,
                        target_data=target, stage="spawn_check",
                        shadow_force_capture=True,
                    ):
                        spawn_found = True
                        self.bot.log(f"[FlexScan] SPAWN bulundu: {img}")
                        break

            # ── 3. SALDIRI ──────────────────────────────────────────────
            # Spawn doğrulandığında VEYA T=0 geçtiğinde (spawn onayı opsiyonel).
            if not attack_done and (spawn_found or now >= spawn_ts):
                self.bot.set_global_mission_phase(
                    phase="COMBAT_PHASE",
                    stage="attack_start",
                    reason=f"combat_{target.get('aciklama')}",
                    extra={"boss_id": str(target.get("aciklama", "unknown"))},
                )
                self.bot.automator.press_key("a")
                attack_done = True
                self.bot.log(f"[FlexScan] SALDIRI: {target['aciklama']}")

            # ── 4. VICTORY ──────────────────────────────────────────────
            # Saldırı başladıktan sonra taranır; bulununca ERKEN ÇIKIŞ.
            if attack_done and not victory_found and victory_images:
                for img in victory_images:
                    if self.bot.vision.find(
                        img, region, victory_conf,
                        target_data=target, stage="victory",
                        shadow_force_capture=True,
                    ):
                        victory_found = True
                        self.bot.log(f"[FlexScan] VICTORY — erken cikis: {img}")
                        break
                if victory_found:
                    break  # ← ERKEN ÇIKIŞ: 10sn pencereyi bekleme

            # ── 100ms bekleme (non-blocking) ────────────────────────────
            time.sleep(POLL)

        self.bot.log(
            f"[FlexScan] Pencere KAPANDI | "
            f"area={area_found} spawn={spawn_found} "
            f"attack={attack_done} victory={victory_found}"
        )

        # ── Sonuç değerlendirmesi ────────────────────────────────────────
        if not victory_images:
            # Victory görseli yapılandırılmamış → saldırı olduysa başarı say
            if not attack_done:
                # Hiç saldırı olmadıysa fallback bekleme (legacy davranış)
                return self.bot._interruptible_wait(4.0)
            return attack_done

        if victory_found:
            return True

        # Victory bulunamadı; "required=False" ise saldırı olduysa başarı say
        if not victory_required:
            return attack_done

        return False

    def _find_fast_chain_candidate(self, current):
        """
        90s kurali:
        - ayni katman
        - kalan sure < 90
        - yurume suresi < 90
        """
        now = time.time()
        threshold = float(self.settings.get("FAST_CHAIN_THRESHOLD_SN", 90.0))

        same_map = [
            b
            for b in self.bosslar.values()
            if b.get("aciklama") != current.get("aciklama")
            and b.get("katman_id") == current.get("katman_id")
            and isinstance(b.get("spawn_time"), (int, float))
            and b["spawn_time"] > now
        ]
        if not same_map:
            return None

        next_boss = sorted(same_map, key=lambda x: x["spawn_time"])[0]
        remaining = next_boss["spawn_time"] - now
        walk_time_enabled = bool(self.settings.get("WALK_TIME_ENABLED", False))
        walk_time = self.bot.combat.get_walk_time(current["aciklama"], next_boss["aciklama"]) if walk_time_enabled else 0

        if remaining < threshold and (not walk_time_enabled or walk_time < threshold):
            return next_boss
        return None

    def automation_thread(self):
        self.bot.log("BossManager aktif.")
        while True:
            if not self.bot.running.is_set() or self.bot.paused or self.bot.active_event:
                time.sleep(1.0)
                continue

            try:
                self.bot.location_manager.update_visual_location()
                now = time.time()

                # ANTI-LOOP: Son saldırının üzerinden en az BOSS_ATTACK_COOLDOWN_SN
                # saniye geçmemiş bosslar ready listesine alınmaz.
                cooldown = float(self.settings.get("BOSS_ATTACK_COOLDOWN_SN", 15.0))
                ready = [
                    h
                    for h in self.bosslar.values()
                    if h.get("spawn_time")
                    and now >= (h["spawn_time"] - h.get("head_start_saniye", 0))
                    and (now - h.get("_last_attack_ts", 0)) >= cooldown
                ]
                if not ready:
                    time.sleep(float(self.settings.get("MAIN_LOOP_SLEEP_SN", 1.0)))
                    continue

                ai_target = self.bot.brain.decide_next_target(ready) if hasattr(self.bot, "brain") else None
                target = ai_target if ai_target else sorted(ready, key=lambda h: h["spawn_time"])[0]
                try:
                    with self.bot.action_lock:
                        self.bot.attacking_target_aciklama = target["aciklama"]
                        self.bot.auto_start_recording("boss_attack", timeout_sec=300)

                        try:
                            success = self.execute_precise_boss_flow(target)
                        finally:
                            self.bot.attacking_target_aciklama = None

                    if success:
                        self._handle_post_attack_logic(target)
                    else:
                        self.bot.automator.return_to_exp_farm(force_restart_if_failed=True)
                finally:
                    # Her saldırı girişimi sonrası zaman damgası güncelle (başarılı/başarısız)
                    target["_last_attack_ts"] = time.time()
                    self.bot.auto_stop_recording()

            except Exception as exc:
                self.bot.log(f"BossManager hatasi: {exc}")
                time.sleep(2.0)

    def _verify_kill(self, boss):
        if hasattr(self.bot, "combat") and hasattr(self.bot.combat, "verify_kill"):
            return bool(self.bot.combat.verify_kill(boss))
        return True

    def _handle_post_attack_logic(self, current):
        if not hasattr(self.bot, "combat"):
            self.bot.automator.return_to_exp_farm()
            return

        loot_done = self.bot.combat.wait_post_attack_loot()
        if not loot_done or not self.bot.running.is_set():
            return

        current_target = current

        # 90s hizli intikal zinciri (AI'yi beklemeden)
        while self.bot.running.is_set():
            next_boss = self._find_fast_chain_candidate(current_target)
            if not next_boss:
                break

            self.bot.log(
                f"90s Kurali: bekleme yapilmadan intikal -> {next_boss['aciklama']}",
                level="WARNING",
            )
            self.bot.set_global_mission_phase(
                phase="NAV_PHASE",
                stage="Transition",
                reason=f"fast_chain_{current_target.get('aciklama')}_to_{next_boss.get('aciklama')}",
                extra={
                    "from_boss": str(current_target.get("aciklama", "unknown")),
                    "to_boss": str(next_boss.get("aciklama", "unknown")),
                    "transition": "90s_rule",
                },
            )

            with self.bot.action_lock:
                self.bot.attacking_target_aciklama = next_boss["aciklama"]
                try:
                    chained_ok = self.execute_precise_boss_flow(next_boss)
                finally:
                    self.bot.attacking_target_aciklama = None

            if not chained_ok:
                self.bot.automator.return_to_exp_farm(force_restart_if_failed=True)
                return

            loot_done = self.bot.combat.wait_post_attack_loot()
            if not loot_done or not self.bot.running.is_set():
                return

            current_target = next_boss

        # Hizli zincir yoksa / bittiyse normal strateji
        self.bot.combat.check_strategic_wait(current_target)
