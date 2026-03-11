# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np

import os


class BossManager:
    def __init__(self, bot):
        self.bot = bot
        self.bosslar = bot.bosslar
        self.settings = bot.settings
        self._last_nav_motion_ts = time.time()
        self._last_nav_reanchor_ts = 0.0

    def _reset_combat_state(self, reason: str = "cycle_end"):
        """
        Boss saldÄ±rÄ± dÃ¶ngÃ¼sÃ¼ tamamlandÄ±ÄŸÄ±nda (baÅŸarÄ±lÄ±/baÅŸarÄ±sÄ±z/hata)
        tÃ¼m durum bayraklarÄ±nÄ± temizler.

        Bu metot Ã§aÄŸrÄ±lmadan bir sonraki dÃ¶ngÃ¼ iterasyonu
        is_in_active_combat() == True gÃ¶rerek yeni saldÄ±rÄ±yÄ± BLOKLAR.

        KÃ¶k neden: _global_phase loot/combat'ta kalÄ±yordu â†’ State Lock.
        """
        # 1. SaldÄ±rÄ± hedefini temizle (zaten finally'de yapÄ±lÄ±yor ama gÃ¼venlik)
        self.bot.attacking_target_aciklama = None

        # 2. _global_phase'i IDLE'a dÃ¶ndÃ¼r â€” KRÄ°TÄ°K FIX
        #    stop_global_mission() bunu GARANTÄ° ETMÄ°YOR.
        if hasattr(self.bot, "_global_phase"):
            old_phase = str(getattr(self.bot, "_global_phase", ""))
            self.bot._global_phase = "IDLE"
            # Staleness guard iÃ§in timestamp'i de sÄ±fÄ±rla
            self.bot._global_phase_ts = time.time()
            if old_phase and old_phase.upper() != "IDLE":
                self.bot.log(
                    f"[STATE_RESET] _global_phase: {old_phase} â†’ IDLE (neden: {reason})",
                    level="DEBUG",
                )

        # 3. stop_global_mission'Ä± da Ã§aÄŸÄ±r (varsa) â€” log/metric tutarlÄ±lÄ±ÄŸÄ±
        if hasattr(self.bot, "stop_global_mission"):
            try:
                self.bot.stop_global_mission(reason=reason)
            except Exception:
                pass  # Ana thread'i asla bloklama

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

    def _capture_navigation_gray(self) -> np.ndarray | None:
        """
        Nabiz olcumu icin ekrani griye cevirir ve karakterin merkez bolgesini maskeler.
        """
        try:
            frame = self.bot.vision.capture_full_screen()
            if frame is None:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

            h, w = gray.shape
            x1, x2 = int(w * 0.35), int(w * 0.65)
            y1, y2 = int(h * 0.30), int(h * 0.78)
            gray[y1:y2, x1:x2] = 0
            return gray
        except Exception:
            return None

    def _run_navigation_reanchor_combo(self, target) -> None:
        self.bot.log("[NAV_PULSE] Takilma suruyor, Re-Anchor tetikleniyor.", level="WARNING")
        self._re_anchor(target)
        self.bot.automator.press_key("z", label="nav_pulse_reanchor_z")

    def _check_navigation_pulse(self, target) -> bool:
        """
        1 saniyelik nabiz kontrolu:
        - Iki gri frame arasi fark cok dusukse re-anchor tetiklenir.
        """
        if not self.bot.running.is_set() or self.bot.paused:
            return False

        first = self._capture_navigation_gray()
        if first is None:
            return True

        if not self.bot._interruptible_wait(1.0):
            return False

        second = self._capture_navigation_gray()
        if second is None:
            return True

        diff = cv2.absdiff(first, second)
        changed_ratio = float(np.count_nonzero(diff > 14)) / float(diff.size)
        move_ratio_threshold = float(self.settings.get("NAV_PULSE_MOVE_RATIO_THRESHOLD", 0.015))
        now = time.time()

        if changed_ratio >= move_ratio_threshold:
            self._last_nav_motion_ts = now
            return True

        self.bot.log(
            f"[NAV_PULSE] Dusuk hareket algisi: ratio={changed_ratio:.4f} (<{move_ratio_threshold:.4f})",
            level="DEBUG",
        )

        # v6.1: Evasion combo PvPManager'a transfer edildi.
        # Dusuk hareket → dogrudan re-anchor denemesi.
        reanchor_after = float(self.settings.get("NAV_PULSE_REANCHOR_AFTER_SN", 3.0))
        reanchor_cd = float(self.settings.get("NAV_PULSE_REANCHOR_COOLDOWN_SN", 2.0))
        if (now - self._last_nav_motion_ts) >= max(1.0, reanchor_after):
            if (now - self._last_nav_reanchor_ts) >= max(0.5, reanchor_cd):
                self._last_nav_reanchor_ts = now
                self._run_navigation_reanchor_combo(target)
                self._last_nav_motion_ts = time.time()
        return True

    def _start_navigation(self, target, ui_protocol: str = None) -> bool:
        """
        Boss listesi ve hedef boss tiklamasini AI ui_protocol kararina gore baslatir.
        """
        target_loc = "KATMAN_1" if "katman_1" in str(target.get("katman_id", "")).lower() else "KATMAN_2"
        current_region = str(self.bot.location_manager.get_region_name() or "UNKNOWN").upper()

        protocol = str(ui_protocol or "").strip().upper()
        valid = {"FULL_MENU_SEQUENCE", "SHORT_LIST_SEQUENCE", "DIRECT_BOSS_SELECTION"}
        if protocol not in valid:
            protocol = "FULL_MENU_SEQUENCE" if current_region != target_loc else "SHORT_LIST_SEQUENCE"

        # Guvenlik: katman yanlissa short/direct zorlanamaz.
        if protocol in {"SHORT_LIST_SEQUENCE", "DIRECT_BOSS_SELECTION"} and current_region != target_loc:
            protocol = "FULL_MENU_SEQUENCE"

        self.bot.log(
            f"[NAV] Protokol=[{protocol}] current_region={current_region} target_region={target_loc}",
            level="DEBUG",
        )

        if protocol == "FULL_MENU_SEQUENCE":
            if current_region != "EXP_FARM":
                if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=True):
                    return False

            full_seq = list(getattr(self.bot, "boss_sequence_template", []) or [])
            if not full_seq:
                self.bot.log("Navigasyon hatasi: boss_sequence_template bos.", level="ERROR")
                return False

            self.bot.log(f"[NAV] FULL_MENU_SEQUENCE basladi (adim={len(full_seq)})", level="DEBUG")
            ok = self.bot._run_sequence(full_seq, self.bot.coordinates, context_target=target)
            if ok:
                self.bot.location_manager.set_current_location_by_name(target_loc)
            return ok

        if protocol == "DIRECT_BOSS_SELECTION":
            self.bot.log("[NAV] DIRECT_BOSS_SELECTION basladi (adim=1)", level="DEBUG")
            direct_seq = [{"action": "boss_secimi", "wait_ms": 200}]
            return self.bot._run_sequence(direct_seq, self.bot.coordinates, context_target=target)

        self.bot.log("[NAV] SHORT_LIST_SEQUENCE basladi (adim=2)", level="DEBUG")
        short_seq = [
            {"action": "click", "label": "boss_list_ac", "wait_ms": 200},
            {"action": "boss_secimi", "wait_ms": 200},
        ]
        ok = self.bot._run_sequence(short_seq, self.bot.coordinates, context_target=target)
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

    def _refresh_interrupted_spawn(self, target) -> None:
        """
        DÃ¶vÃ¼ÅŸ tam baÅŸlayamadan kesilen akÄ±ÅŸlarda spawn_time'Ä±n stale kalmasÄ±nÄ± Ã¶nler.
        """
        combat = getattr(self.bot, "combat", None)
        if combat is None or not hasattr(combat, "recalculate_times_interrupted"):
            return
        try:
            combat.recalculate_times_interrupted(target)
        except Exception as exc:
            self.bot.log(f"[INTERRUPT] Spawn refresh hatasi: {exc}", level="WARNING")

    def execute_precise_boss_flow(self, target, ui_protocol: str = None) -> bool:
        """
        Esnek zaman pencereli hassas akis:
        - T-PRE: Tarama penceresi acilir (SCAN_WINDOW_PRE_SN, varsayilan 5sn)
        - T-PRE..T=0 : Area tarama (bir kez bulununca yeniden aramaz)
        - T=0..T+POST: Spawn + saldiri + victory (SCAN_WINDOW_POST_SN, varsayilan 5sn)
        - Victory: Erken cikis AI 10sn pencereyi bekleme
        """
        if not hasattr(self.bot, "combat"):
            return False

        spawn_ts = target.get("spawn_time")
        if not isinstance(spawn_ts, (int, float)):
            return False

        protocol_name = str(ui_protocol or "").strip().upper() or self._derive_safe_protocol(target)
        if not self._wait_for_navigation_window(target, protocol_name):
            self._refresh_interrupted_spawn(target)
            return False

        self.bot.log(
            f"[BOSS_FLOW] basladi boss={target.get('aciklama')} protocol={protocol_name}",
            level="DEBUG",
        )

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
        # â˜… Staleness guard: phase baÅŸlangÄ±Ã§ zamanÄ±nÄ± kaydet
        self.bot._global_phase_ts = time.time()

        nav_ok = self._start_navigation(target, ui_protocol=protocol_name)
        self.bot.log(f"Navigasyon fazi sonucu: boss={target.get('aciklama')} success={nav_ok}", level="DEBUG")
        if not nav_ok:
            self.bot.log_training_outcome(
                "boss_flow",
                {"boss_id": str(target.get("aciklama", "unknown")), "success": False, "reason": "navigation_failed"},
            )
            if hasattr(self.bot, "reward_engine"):
                self.bot.reward_engine.on_death(boss_name=str(target.get("aciklama", "")))
            self._refresh_interrupted_spawn(target)
            return False

        self.bot.automator.press_key("z")

        # PlanlI spawn zamanI = saldÄ±rÄ± referans noktasÄ± (recalculate_times iÃ§in)
        attack_start = float(spawn_ts)
        _combat_start_ts = time.time()

        kill_ok = self._unified_spawn_sequence(target, spawn_ts)
        kill_time = time.time() - _combat_start_ts
        self.bot.log(f"Boss sonucu ({target['aciklama']}): {'OK' if kill_ok else 'FAIL'}")
        self.bot.log_training_outcome(
            "boss_flow",
            {"boss_id": str(target.get("aciklama", "unknown")), "success": bool(kill_ok), "reason": "flex_scan"},
        )

        # ★ FIX (Baş Mühendis): Boss'un yeni doğuş süresi (Respawn), 
        # doğduğu (spawn) anına göre değil, ÖLDÜĞÜ (Victory) anına göre hesaplanmalıdır.
        if kill_ok:
            # Boss başarıyla kesildi! Süreyi tam şu ana (Ölüm anı) göre hesapla.
            self.bot.combat.recalculate_times(target, time.time())
        else:
            # Başarısız olduysa, sonsuz döngüden kurtulmak için eski teorik zamana göre ileri sar.
            self.bot.combat.recalculate_times(target, attack_start)

        ai_engine = getattr(getattr(self.bot, "brain", None), "ai_engine", None)
        memory = getattr(ai_engine, "memory", None)

        if kill_ok:
            # GUI "Boss: X kill" degeri AIEngine -> MemoryManager metadata.total_bosses_killed
            # alanindan okunuyor. update_boss_performance(success=True) bu sayaci artirir.
            if memory is not None and hasattr(memory, "update_boss_performance"):
                try:
                    memory.update_boss_performance(
                        boss_name=str(target.get("aciklama", "")),
                        kill_time=float(round(kill_time, 2)),
                        success=True,
                    )
                except Exception as exc:
                    self.bot.log(f"[AI_MEMORY] boss_kill guncelleme hatasi: {exc}", level="WARNING")
            if hasattr(self.bot, "reward_engine"):
                self.bot.reward_engine.on_boss_killed(
                    boss_name=str(target.get("aciklama", "")),
                    kill_time=round(kill_time, 2),
                )
        else:
            if memory is not None and hasattr(memory, "update_boss_performance"):
                try:
                    memory.update_boss_performance(
                        boss_name=str(target.get("aciklama", "")),
                        kill_time=float(round(kill_time, 2)),
                        success=False,
                    )
                except Exception as exc:
                    self.bot.log(f"[AI_MEMORY] boss_fail guncelleme hatasi: {exc}", level="WARNING")
            if hasattr(self.bot, "reward_engine"):
                self.bot.reward_engine.on_death(boss_name=str(target.get("aciklama", "")))
        return bool(kill_ok)

    def _unified_spawn_sequence(self, target, spawn_ts: float) -> bool:
     
        POLL = 0.1  # 100ms  non-blocking hassasiyet
        PRE  = float(self.settings.get("SCAN_WINDOW_PRE_SN",  5.0))  # T-5
        POST = float(self.settings.get("SCAN_WINDOW_POST_SN", 5.0))  # T+5

        window_open_ts  = spawn_ts - PRE
        window_close_ts = spawn_ts + POST

        # GÃ¶rsel konfigÃ¼rasyonlarÄ±nÄ± hazÄ±rla
        try:
            area_data = self.bot.vision.get_boss_visual_data(target, "area_check") or {}
        except Exception:
            area_data = {}

        spawn_cfg   = self.bot.combat._resolve_spawn_check(target)
        victory_cfg = self.bot.combat._resolve_victory_check(target)

        area_image   = area_data.get("image_file", "")
        # v6.1: UI kapanma halüsinasyonunu önlemek için minimum %85 eşik zorla
        area_conf    = max(float(area_data.get("confidence", 0.70)), 0.85)

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

        # Durum bayraklar (yerel AI her ÅŸartda sÄ±fÄ±rlanÄ±r)
        area_found    = False
        spawn_found   = False
        attack_done   = False
        victory_found = False
        re_anchor_done = False

        target["_area_check_ok"] = False

        # ── v6.1: Kör Uçuş (Grace Period) ──────────────────────────
        # Navigasyon başladıktan sonra UI menüleri (boss listesi, katman
        # seçimi vb.) kapanması ~2-3 saniye sürer. Bu süre içinde ekran
        # taraması yapmak UI yazısını area olarak algılama riski taşır
        # (False Positive). Bot ilk 3 saniye ekrana bakmaz.
        _grace_sn = float(self.settings.get("NAV_AREA_GRACE_PERIOD_SN", 3.0))
        self.bot.log(
            f"[FlexScan] Kor ucus: {_grace_sn:.1f}s (UI kapanma bekleme)",
            level="DEBUG",
        )
        time.sleep(_grace_sn)

        # T-PRE anina kadar bekle 
        wait_pre = window_open_ts - time.time()
        if wait_pre > 0:
            self.bot.log(f"[NAV_PULSE] Spawn oncesi bekleme nabiz kontrolune alindi: {wait_pre:.1f}s", level="DEBUG")
            while self.bot.running.is_set():
                remaining = window_open_ts - time.time()
                if remaining <= 0:
                    break

                # Hedefe erken ulasilmissa beklemeyi kes.
                if not area_found and area_image and area_image != "default.png":
                    if self.bot.vision.find(
                        area_image,
                        region,
                        area_conf,
                        target_data=target,
                        stage="area_check",
                    ):
                        area_found = True
                        target["_area_check_ok"] = True
                        self.bot.log(f"[NAV_PULSE] AREA erken bulundu: {area_image}", level="DEBUG")
                        self.bot._seal_visual_event(
                            "area_check",
                            extra={"boss_id": str(target.get("aciklama", "")), "image": area_image},
                        )
                        break

                if not self._check_navigation_pulse(target):
                    return False

        self.bot.log(
            f"[FlexScan] Pencere ACIK | {target['aciklama']} | "
            f"T-{PRE:.0f}..T+{POST:.0f} | "
            f"area={bool(area_image and area_image != 'default.png')} "
            f"spawn={len(spawn_images)} victory={len(victory_images)}"
        )

        # Ana tarama dÃ¶ngÃ¼sÃ¼
        while self.bot.running.is_set():
            now = time.time()
            if now >= window_close_ts:
                break

            # Spawn'a kadar navigasyonda nabiz kontrolu devam eder.
            # AREA bulunduysa karakterin durmasi normaldir; pulse tetikleme.
            if not area_found and not attack_done and now < spawn_ts:
                if not self._check_navigation_pulse(target):
                    return False

            # 1. AREA 
            # T-PRE'dan itibaren taranÄ±r; bir kez bulununca tekrar aranmaz.
            if not area_found and area_image and area_image != "default.png":
                if self.bot.vision.find(
                    area_image, region, area_conf,
                    target_data=target, stage="area_check",
                ):
                    area_found = True
                    target["_area_check_ok"] = True
                    self.bot.log(f"[FlexScan] AREA bulundu: {area_image}")
                    self.bot._seal_visual_event(
                        "area_check",
                        extra={"boss_id": str(target.get("aciklama", "")), "image": area_image},
                    )

            # 2. SPAWN 
            # T=0'dan itibaren taranÄ±r; bir kez bulununca tekrar aranmaz.
            if not spawn_found and spawn_enabled and spawn_images and now >= spawn_ts:
                for img in spawn_images:
                    if self.bot.vision.find(
                        img, region, spawn_conf,
                        target_data=target, stage="spawn_check",
                        shadow_force_capture=True,
                    ):
                        spawn_found = True
                        self.bot.log(f"[FlexScan] SPAWN bulundu: {img}")
                        self.bot._seal_visual_event(
                            "spawn_check",
                            extra={"boss_id": str(target.get("aciklama", "")), "image": img},
                        )
                        break

            # 3. SALDIRI 
            # Spawn doÄŸrulandÄ±ÄŸÄ±nda VEYA T=0 geÃ§tiÄŸinde (spawn onayÄ± opsiyonel).
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

            # 4. VICTORY 
            # SaldÄ±rÄ± baÅŸladÄ±ktan sonra taranÄ±r; bulununca ERKEN Ã‡IKIÅ.
            if attack_done and not victory_found and victory_images:
                for img in victory_images:
                    if self.bot.vision.find(
                        img, region, victory_conf,
                        target_data=target, stage="victory",
                        shadow_force_capture=True,
                    ):
                        victory_found = True
                        self.bot._seal_visual_event(
                            "victory",
                            extra={"boss_id": str(target.get("aciklama", "")), "image": img},
                        )
                        self.bot.log(f"[FlexScan] VICTORY - erken cikis: {img}")
                        break
                if victory_found:
                    break  # ERKEN Ã‡IKIÅ: 10sn pencereyi bekleme

            # 100ms bekleme (non-blocking) 
            time.sleep(POLL)

        self.bot.log(
            f"[FlexScan] Pencere KAPANDI | "
            f"area={area_found} spawn={spawn_found} "
            f"attack={attack_done} victory={victory_found}"
        )

        # SonuÃ§ deÄŸerlendirmesi
        if not victory_images:
            if not attack_done:
                # HiÃ§ saldÄ±rÄ± olmadÄ±ysa fallback bekleme (legacy davranÄ±ÅŸ)
                return self.bot._interruptible_wait(4.0)
            return attack_done

        if victory_found:
            return True

        # Victory bulunamadÄ±; "required=False" ise saldÄ±rÄ± olduysa baÅŸarÄ± say
        if not victory_required:
            return attack_done

        return False

    def _find_fast_chain_candidate(self, current):
        """
        90s kurali:
        - ayni katman
        - head_start dikkate alinmis kalan sure < 90
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
        head_start = float(next_boss.get("head_start_saniye", 0.0))
        # Fast-chain karari spawn anina gore degil "navigasyon baslama" anina gore verilir.
        ready_at = float(next_boss["spawn_time"]) - max(0.0, head_start)
        remaining = ready_at - now
        walk_time_enabled = bool(self.settings.get("WALK_TIME_ENABLED", False))
        walk_time = self.bot.combat.get_walk_time(current["aciklama"], next_boss["aciklama"]) if walk_time_enabled else 0

        if remaining < threshold and (not walk_time_enabled or walk_time < threshold):
            return next_boss
        return None

    def _collect_ready_targets(self, now_ts: float) -> list:
        """
        Cooldown + head_start + lookahead kurallarina gore saldiri adayi toplar.
        Not:
        - Sadece "tam hazir" bosslari degil, head_start penceresine cok yakin
          bosslari da dahil eder.
        - Secim asamasinda "en yakin dogus" politikasi uygulanir.
        """
        cooldown = float(self.settings.get("BOSS_ATTACK_COOLDOWN_SN", 15.0))
        lookahead = float(self.settings.get("BOSS_SELECTION_LOOKAHEAD_SN", 15.0))
        ready = []

        for boss in self.bosslar.values():
            spawn_ts = boss.get("spawn_time")
            if not isinstance(spawn_ts, (int, float)):
                continue

            head_start = float(boss.get("head_start_saniye", 0))
            ready_at_ts = float(spawn_ts) - head_start
            last_attack_ts = float(boss.get("_last_attack_ts", 0))
            cooldown_ok = (now_ts - last_attack_ts) >= cooldown
            near_ready = now_ts + max(0.0, lookahead) >= ready_at_ts
            if near_ready and cooldown_ok:
                boss["_ready_at_ts"] = ready_at_ts
                boss["_spawn_delta_sn"] = max(0.0, float(spawn_ts) - now_ts)
                ready.append(boss)

        return ready

    def _derive_safe_protocol(self, target: dict) -> str:
        """
        AI protokol vermezse veya gecersiz verirse v5.8 disiplini fallback.
        """
        current_region = "UNKNOWN"
        try:
            current_region = str(self.bot.location_manager.get_region_name() or "UNKNOWN").upper()
        except Exception:
            pass

        target_loc = "KATMAN_1" if "katman_1" in str(target.get("katman_id", "")).lower() else "KATMAN_2"
        if current_region != target_loc:
            return "FULL_MENU_SEQUENCE"
        return "SHORT_LIST_SEQUENCE"

    def _select_target_with_protocol(self, ready_targets: list):
        """
        Aday listeden hedef secer ve uygulanacak navigasyon protokolunu kesinler.
        Politika:
        - Varsayilan olarak en yakin dogus (en kucuk spawn_time) zorunlu.
        - AI secimi farkliysa loglanir ve override edilir.
        """
        force_earliest_spawn = bool(self.settings.get("FORCE_EARLIEST_SPAWN_SELECTION", True))
        earliest_spawn_target = sorted(
            ready_targets,
            key=lambda h: (
                float(h.get("spawn_time", 10**12)),
                float(h.get("_ready_at_ts", 10**12)),
                str(h.get("aciklama", "")),
            ),
        )[0]

        target = None
        if hasattr(self.bot, "brain"):
            try:
                target = self.bot.brain.decide_next_target(ready_targets)
            except Exception as exc:
                self.bot.log(f"TacticalBrain hedef secim hatasi: {exc}", level="WARNING")

        if not target:
            target = earliest_spawn_target

        if force_earliest_spawn and target is not earliest_spawn_target:
            self.bot.log(
                f"[BOSS_LOOP] AI secimi override edildi: "
                f"ai={target.get('aciklama')} -> earliest_spawn={earliest_spawn_target.get('aciklama')}",
                level="WARNING",
            )
            target = earliest_spawn_target

        protocol = str((target or {}).get("_ui_protocol", "")).strip().upper()
        if not protocol:
            protocol = self._derive_safe_protocol(target)

        return target, protocol

    def _execute_single_attack_attempt(self, target: dict, protocol: str) -> bool:
        boss_id = str(target.get("aciklama", "unknown"))
        protocol_name = str(protocol or "").strip().upper() or self._derive_safe_protocol(target)

        self.bot.log(f"[BOSS_LOOP] hedef={boss_id} protocol={protocol_name}", level="DEBUG")
        self.bot.attacking_target_aciklama = boss_id
        try:
            return bool(self.execute_precise_boss_flow(target, ui_protocol=protocol_name))
        finally:
            self.bot.attacking_target_aciklama = None

    def _wait_for_navigation_window(self, target: dict, protocol: str) -> bool:
        """
        Navigasyonu boss'un head_start_saniye kuralina gore zamanlar.
        Baslangic noktasi: spawn_time - head_start_saniye
        """
        spawn_ts = target.get("spawn_time")
        if not isinstance(spawn_ts, (int, float)):
            return False

        proto = str(protocol or "").strip().upper()
        head_start = float((target or {}).get("head_start_saniye", 0.0))
        ready_at = float(spawn_ts) - max(0.0, head_start)
        wait_sn = ready_at - time.time()
        if wait_sn <= 0:
            return True

        self.bot.log(
            f"[NAV] HeadStart bekleme: protocol={proto} head_start={head_start:.1f}s "
            f"spawn_oncesi_bekleme={wait_sn:.1f}s",
            level="DEBUG",
        )
        return bool(self.bot._interruptible_wait(wait_sn))

    def automation_thread(self):
        self.bot.log("BossManager aktif.")
        self.bot.log(f"BossManager source aktif: {os.path.abspath(__file__)}", level="DEBUG")
        loop_sleep = float(self.settings.get("MAIN_LOOP_SLEEP_SN", 1.0))
        while True:
            if not self.bot.running.is_set() or self.bot.paused or self.bot.active_event:
                time.sleep(1.0)
                continue

            # Guard: Navigasyon sablonu eksikse boss dongusu baslatamaz;
            # sonsuz deneme yerine 10s bekleyip yeniden kontrol et.
            if not getattr(self.bot, "boss_sequence_template", None):
                self.bot.log(
                    "BossManager: boss_sequence_template bos veya yuklenmemis, bekleniyor.",
                    level="WARNING",
                )
                time.sleep(10.0)
                continue

            try:
                self.bot.location_manager.update_visual_location()
                now = time.time()
                ready = self._collect_ready_targets(now)
                if not ready:
                    time.sleep(loop_sleep)
                    continue

                target, selected_protocol = self._select_target_with_protocol(ready)
                success = False

                # v6.1: Video kaydi kullanici istegiyle baslatilir.
                time.sleep(1.0)  # Buffer warmup

                try:
                    success = self._execute_single_attack_attempt(target, selected_protocol)

                    if success:
                        # Ganimet, zincir ve stratejik bekleme BÄ°TTÄ°KTEN SONRA flush:
                        self._handle_post_attack_logic(target)
                        # Video: baÅŸarÄ±lÄ± oturumu etiketle
                        vid = getattr(self.bot, "video_recorder", None)
                        if vid is not None and vid.is_recording:
                            boss_id = str(target.get("aciklama", "unknown"))
                            vid.stop(success=True, reason=f"boss_{boss_id}_success")
                        # â˜… FIX: State Lock Ã¶nleyici â€” phase'i IDLE'a Ã§ek
                        self._reset_combat_state(
                            reason=f"boss_{target.get('aciklama')}_success"
                        )
                    else:
                        if hasattr(self.bot, "reward_engine"):
                            self.bot.reward_engine.on_restart(reason="boss_flow_failed")
                        self.bot.automator.return_to_exp_farm(force_restart_if_failed=True)
                        # â˜… FIX: State Lock Ã¶nleyici â€” fail'de de phase temizle
                        self._reset_combat_state(
                            reason=f"boss_{target.get('aciklama')}_failed"
                        )
                finally:
                    # Her saldÄ±rÄ± giriÅŸimi sonrasÄ± zaman damgasÄ± gÃ¼ncelle
                    target["_last_attack_ts"] = time.time()
                    # Video: baÅŸarÄ±sÄ±zsa burada durdur (baÅŸarÄ±lÄ±ysa yukarÄ±da durdu)
                    vid = getattr(self.bot, "video_recorder", None)
                    if vid is not None and vid.is_recording and not success:
                        boss_id = str(target.get("aciklama", "unknown"))
                        vid.stop(success=False, reason=f"boss_{boss_id}_fail")
                    # â˜… FIX: SON SAVUNMA HATTI â€” ne olursa olsun phase'i temizle.
                    #   YukarÄ±daki success/fail bloklarÄ±ndan biri zaten Ã§aÄŸÄ±rmÄ±ÅŸ olsa bile
                    #   idempotent olduÄŸu iÃ§in tekrar Ã§aÄŸÄ±rmak gÃ¼venlidir.
                    self._reset_combat_state(reason="finally_guard")

            except Exception as exc:
                self.bot.log(f"BossManager hatasi: {exc}")
                if "target" in locals() and isinstance(target, dict):
                    self._refresh_interrupted_spawn(target)
                # â˜… FIX: Hata durumunda da tÃ¼m state'i temizle
                self._reset_combat_state(reason=f"exception_{type(exc).__name__}")
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

            self.bot.attacking_target_aciklama = next_boss["aciklama"]
            try:
                chained_ok = self.execute_precise_boss_flow(next_boss, ui_protocol="SHORT_LIST_SEQUENCE")
            finally:
                self.bot.attacking_target_aciklama = None

            if not chained_ok:
                self.bot.automator.return_to_exp_farm(force_restart_if_failed=True)
                return

            loot_done = self.bot.combat.wait_post_attack_loot()
            if not loot_done or not self.bot.running.is_set():
                return

            current_target = next_boss

        # â˜… FIX: Zincir bitti, stratejik bekleme Ã¶ncesi LOOT fazÄ±nÄ± temizle.
        #   check_strategic_wait() _global_phase'i DEÄÄ°ÅTÄ°RMEZ;
        #   burada IDLE'a Ã§ekmezsen is_in_active_combat() True kalÄ±r.
        if hasattr(self.bot, "_global_phase"):
            self.bot._global_phase = "IDLE"
            self.bot.log(
                "[STATE_RESET] Post-attack: _global_phase â†’ IDLE (zincir sonu)",
                level="DEBUG",
            )

        # Hizli zincir yoksa / bittiyse normal strateji
        self.bot.combat.check_strategic_wait(current_target)
