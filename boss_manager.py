import time

import cv2
import numpy as np

import os


class BossManager:
    def __init__(self, bot):
        self.bot = bot
        self.bosslar = bot.bosslar
        self.settings = bot.settings
        self._last_nav_evasion_ts = 0.0
        self._last_nav_motion_ts = time.time()
        self._last_nav_reanchor_ts = 0.0

    # ═══════════════════════════════════════════════════════════════════
    #  STATE RESET — State Lock önleyici merkezi temizlik
    # ═══════════════════════════════════════════════════════════════════

    def _reset_combat_state(self, reason: str = "cycle_end"):
        """
        Boss saldırı döngüsü tamamlandığında (başarılı/başarısız/hata)
        tüm durum bayraklarını temizler.

        Bu metot çağrılmadan bir sonraki döngü iterasyonu
        is_in_active_combat() == True görerek yeni saldırıyı BLOKLAR.

        Kök neden: _global_phase loot/combat'ta kalıyordu → State Lock.
        """
        # 1. Saldırı hedefini temizle (zaten finally'de yapılıyor ama güvenlik)
        self.bot.attacking_target_aciklama = None

        # 2. _global_phase'i IDLE'a döndür — KRİTİK FIX
        #    stop_global_mission() bunu GARANTİ ETMİYOR.
        if hasattr(self.bot, "_global_phase"):
            old_phase = str(getattr(self.bot, "_global_phase", ""))
            self.bot._global_phase = "IDLE"
            # Staleness guard için timestamp'i de sıfırla
            self.bot._global_phase_ts = time.time()
            if old_phase and old_phase.upper() != "IDLE":
                self.bot.log(
                    f"[STATE_RESET] _global_phase: {old_phase} → IDLE (neden: {reason})",
                    level="DEBUG",
                )

        # 3. stop_global_mission'ı da çağır (varsa) — log/metric tutarlılığı
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

    def _run_navigation_evasion_combo(self) -> None:
        """
        Navigasyonda pusu/stagger durumunda kisa kacis refleksi.
        """
        self.bot.log(
            "[NAV_EVASION] Dusuk hareket algisi, kacis kombosu: Space -> q -> Space",
            level="WARNING",
        )
        self.bot.automator.press_key("space", label="nav_evasion_space_1")
        time.sleep(0.04)
        self.bot.automator.press_key("q", label="nav_evasion_q")
        time.sleep(0.04)
        self.bot.automator.press_key("space", label="nav_evasion_space_2")

    def _capture_navigation_gray(self) -> np.ndarray | None:
        """
        Nabiz kontrolu icin ekrani kucultulmus gri forma cevirir.
        Merkez bolge maskelenerek karakter idle animasyonlarinin etkisi azaltilir.
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

    def _probe_area_during_navigation(
        self,
        target: dict,
        area_image: str,
        area_conf: float,
        region: dict,
    ) -> bool:
        """
        Pulse kontrolu icinde area tespiti yapar.
        Area bulunduysa hedefe varildi kabul edilir ve pulse aksiyonlari durur.
        """
        if not area_image or area_image == "default.png":
            return False
        if bool(target.get("_area_check_ok")):
            return True

        try:
            matched = self.bot.vision.find(
                area_image,
                region,
                float(area_conf),
                target_data=target,
                stage="area_check",
            )
        except Exception:
            matched = False

        if matched:
            target["_area_check_ok"] = True
            self._last_nav_motion_ts = time.time()
            self.bot.log(f"[NAV_PULSE] AREA tespit edildi: {area_image}", level="DEBUG")
            self.bot._seal_visual_event(
                "area_check",
                extra={
                    "boss_id": str(target.get("aciklama", "")),
                    "image": area_image,
                    "source": "navigation_pulse",
                },
            )
            return True

        return False

    def _check_navigation_pulse(
        self,
        target: dict,
        area_image: str = "",
        area_conf: float = 0.70,
        region: dict | None = None,
    ) -> bool:
        """
        1 saniyelik nabiz kontrolu:
        - Hareket var mi? (2 gri kare farki)
        - Pulse icinde area'ya ulasildi mi?
        - Hareket yoksa once evasion, devam ederse re-anchor.
        """
        if not self.bot.running.is_set() or self.bot.paused:
            return False

        region = region or self.bot.ui_regions.get(
            "region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440}
        )

        if self._probe_area_during_navigation(target, area_image, area_conf, region):
            return True

        first = self._capture_navigation_gray()
        if first is None:
            return True

        if not self.bot._interruptible_wait(1.0):
            return False

        if self._probe_area_during_navigation(target, area_image, area_conf, region):
            return True

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
            f"[NAV_PULSE] Dusuk hareket: ratio={changed_ratio:.4f} (<{move_ratio_threshold:.4f})",
            level="DEBUG",
        )

        evasion_cd = float(self.settings.get("NAV_EVASION_COOLDOWN_SN", 1.0))
        if (now - self._last_nav_evasion_ts) >= max(0.2, evasion_cd):
            self._last_nav_evasion_ts = now
            self._run_navigation_evasion_combo()

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

    def execute_precise_boss_flow(self, target, ui_protocol: str = None) -> bool:
        """
        Esnek zaman pencereli hassas akis:
        - T-PRE: Tarama penceresi acilir (SCAN_WINDOW_PRE_SN, varsayilan 5sn)
        - T-PRE..T=0 : Area tarama (bir kez bulununca yeniden aramaz)
        - T=0..T+POST: Spawn + saldiri + victory (SCAN_WINDOW_POST_SN, varsayilan 5sn)
        - Victory: Erken cikis â€” 10sn pencereyi bekleme
        """
        if not hasattr(self.bot, "combat"):
            return False

        spawn_ts = target.get("spawn_time")
        if not isinstance(spawn_ts, (int, float)):
            return False

        protocol_name = str(ui_protocol or "").strip().upper() or self._derive_safe_protocol(target)
        if not self._wait_for_navigation_window(target, protocol_name):
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
        # ★ Staleness guard: phase başlangıç zamanını kaydet
        self.bot._global_phase_ts = time.time()
        self._last_nav_motion_ts = time.time()
        self._last_nav_evasion_ts = 0.0
        self._last_nav_reanchor_ts = 0.0

        nav_ok = self._start_navigation(target, ui_protocol=protocol_name)
        self.bot.log(f"Navigasyon fazi sonucu: boss={target.get('aciklama')} success={nav_ok}", level="DEBUG")
        if not nav_ok:
            self.bot.log_training_outcome(
                "boss_flow",
                {"boss_id": str(target.get("aciklama", "unknown")), "success": False, "reason": "navigation_failed"},
            )
            if hasattr(self.bot, "reward_engine"):
                self.bot.reward_engine.on_death(boss_name=str(target.get("aciklama", "")))
            return False

        self.bot.automator.press_key("z")

        # PlanlÄ± spawn zamanÄ± = saldÄ±rÄ± referans noktasÄ± (recalculate_times iÃ§in)
        attack_start = float(spawn_ts)
        _combat_start_ts = time.time()

        kill_ok = self._unified_spawn_sequence(target, spawn_ts)
        kill_time = time.time() - _combat_start_ts
        self.bot.log(f"Boss sonucu ({target['aciklama']}): {'OK' if kill_ok else 'FAIL'}")
        self.bot.log_training_outcome(
            "boss_flow",
            {"boss_id": str(target.get("aciklama", "unknown")), "success": bool(kill_ok), "reason": "flex_scan"},
        )

        # ★ FIX (Codex): Boss sabit periyotla respawn eder — başarılı/başarısız fark etmez.
        # recalculate_times her zaman çağrılmalı; catch-up mantığı stale
        # spawn_time'ı da otomatik olarak ileriye taşır.
        self.bot.combat.recalculate_times(target, attack_start)

        ai_engine = getattr(getattr(self.bot, "brain", None), "ai_engine", None)
        memory = getattr(ai_engine, "memory", None)

        if kill_ok:
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
        """
        T-PRE / T+POST esnek pencere tarayÄ±cÄ±sÄ±.

        HiyerarÅŸik durum makinesi:
          area â†’ spawn â†’ saldÄ±rÄ± â†’ victory

        Kurallar:
          - Bir gÃ¶rsel bulununca o durum True kalÄ±r, yeniden aranmaz.
          - Victory yakalanÄ±nca dÃ¶ngÃ¼ anÄ±nda kÄ±rÄ±lÄ±r (erken Ã§Ä±kÄ±ÅŸ).
          - Aksiyon bitince tÃ¼m durumlar otomatik sÄ±fÄ±rlanÄ±r (yerel deÄŸiÅŸkenler).
          - 200ms poll â†’ saniyede 5 kontrol, UI/sistem kasma yok.

        Args:
            target   : Boss hedef dict'i
            spawn_ts : Beklenen doÄŸma Unix zamanÄ±

        Returns:
            True  â†’ kill onaylandÄ± (ya da onay gerektirmeyen config)
            False â†’ zaman aÅŸÄ±mÄ±, bot durumu veya kritik hata
        """
        POLL = 0.1  # 100ms â€” non-blocking hassasiyet
        PRE  = float(self.settings.get("SCAN_WINDOW_PRE_SN",  5.0))  # T-5
        POST = float(self.settings.get("SCAN_WINDOW_POST_SN", 5.0))  # T+5

        window_open_ts  = spawn_ts - PRE
        window_close_ts = spawn_ts + POST

        # â”€â”€ GÃ¶rsel konfigÃ¼rasyonlarÄ±nÄ± hazÄ±rla â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ Durum bayraklarÄ± (yerel â†’ her Ã§aÄŸrÄ±da sÄ±fÄ±rlanÄ±r) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        area_found    = False
        spawn_found   = False
        attack_done   = False
        victory_found = False
        re_anchor_done = False

        target["_area_check_ok"] = False

        # â”€â”€ T-PRE anÄ±na kadar bekle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        wait_pre = window_open_ts - time.time()
        if wait_pre > 0:
            self.bot.log(
                f"[NAV_PULSE] Spawn oncesi bekleme pulse modunda: {wait_pre:.1f}s",
                level="DEBUG",
            )
            while self.bot.running.is_set():
                remaining = window_open_ts - time.time()
                if remaining <= 0:
                    break

                if not self._check_navigation_pulse(
                    target=target,
                    area_image=area_image,
                    area_conf=area_conf,
                    region=region,
                ):
                    return False

                if bool(target.get("_area_check_ok")):
                    area_found = True
                    break

        self.bot.log(
            f"[FlexScan] Pencere ACIK | {target['aciklama']} | "
            f"T-{PRE:.0f}..T+{POST:.0f} | "
            f"area={bool(area_image and area_image != 'default.png')} "
            f"spawn={len(spawn_images)} victory={len(victory_images)}"
        )

        # â”€â”€ Ana tarama dÃ¶ngÃ¼sÃ¼ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        while self.bot.running.is_set():
            now = time.time()
            if now >= window_close_ts:
                break

            # Spawn'a kadar navigasyon nabzini izle.
            # Area bulunduysa karakterin durmasi normal kabul edilir.
            if not area_found and not attack_done and now < spawn_ts:
                if not self._check_navigation_pulse(
                    target=target,
                    area_image=area_image,
                    area_conf=area_conf,
                    region=region,
                ):
                    return False
                if bool(target.get("_area_check_ok")):
                    area_found = True

            # â”€â”€ 1. AREA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ 2. SPAWN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ 3. SALDIRI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ 4. VICTORY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                        self.bot.log(f"[FlexScan] VICTORY â€” erken cikis: {img}")
                        break
                if victory_found:
                    break  # â† ERKEN Ã‡IKIÅ: 10sn pencereyi bekleme

            # â”€â”€ 100ms bekleme (non-blocking) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            time.sleep(POLL)

        self.bot.log(
            f"[FlexScan] Pencere KAPANDI | "
            f"area={area_found} spawn={spawn_found} "
            f"attack={attack_done} victory={victory_found}"
        )

        # â”€â”€ SonuÃ§ deÄŸerlendirmesi â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not victory_images:
            # Victory gÃ¶rseli yapÄ±landÄ±rÄ±lmamÄ±ÅŸ â†’ saldÄ±rÄ± olduysa baÅŸarÄ± say
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
            # auto_start_recording automation_thread'de cagrilir (EXP_FARM'dayken + warmup)
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

                # Episode başlangıcı: EXP_FARM'dayken kaydı başlat.
                # 1.0 saniyelik warmup: rolling buffer (WINDOW_SIZE=10 frame, 10 FPS)
                # dolmadan ilk navigasyon seal'i boş pencere yakalar.
                self.bot.auto_start_recording("boss_attack", timeout_sec=300)
                time.sleep(1.0)  # Buffer warmup: 10 frame / 10 FPS = 1.0s

                try:
                    success = self._execute_single_attack_attempt(target, selected_protocol)

                    if success:
                        # Ganimet, zincir ve stratejik bekleme BİTTİKTEN SONRA flush:
                        self._handle_post_attack_logic(target)
                        # Video: başarılı oturumu etiketle
                        vid = getattr(self.bot, "video_recorder", None)
                        if vid is not None and vid.is_recording:
                            boss_id = str(target.get("aciklama", "unknown"))
                            vid.stop(success=True, reason=f"boss_{boss_id}_success")
                        # ★ FIX: State Lock önleyici — phase'i IDLE'a çek
                        self._reset_combat_state(
                            reason=f"boss_{target.get('aciklama')}_success"
                        )
                    else:
                        if hasattr(self.bot, "reward_engine"):
                            self.bot.reward_engine.on_restart(reason="boss_flow_failed")
                        self.bot.automator.return_to_exp_farm(force_restart_if_failed=True)
                        # ★ FIX: State Lock önleyici — fail'de de phase temizle
                        self._reset_combat_state(
                            reason=f"boss_{target.get('aciklama')}_failed"
                        )
                finally:
                    # Her saldırı girişimi sonrası zaman damgası güncelle
                    target["_last_attack_ts"] = time.time()
                    # Video: başarısızsa burada durdur (başarılıysa yukarıda durdu)
                    vid = getattr(self.bot, "video_recorder", None)
                    if vid is not None and vid.is_recording and not success:
                        boss_id = str(target.get("aciklama", "unknown"))
                        vid.stop(success=False, reason=f"boss_{boss_id}_fail")
                    self.bot.auto_stop_recording()
                    # ★ FIX: SON SAVUNMA HATTI — ne olursa olsun phase'i temizle.
                    #   Yukarıdaki success/fail bloklarından biri zaten çağırmış olsa bile
                    #   idempotent olduğu için tekrar çağırmak güvenlidir.
                    self._reset_combat_state(reason="finally_guard")

            except Exception as exc:
                self.bot.log(f"BossManager hatasi: {exc}")
                # ★ FIX: Hata durumunda da tüm state'i temizle
                self._reset_combat_state(reason=f"exception_{type(exc).__name__}")
                if hasattr(self.bot, "auto_stop_recording"):
                    self.bot.auto_stop_recording()
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

        # ★ FIX: Zincir bitti, stratejik bekleme öncesi LOOT fazını temizle.
        #   check_strategic_wait() _global_phase'i DEĞİŞTİRMEZ;
        #   burada IDLE'a çekmezsen is_in_active_combat() True kalır.
        if hasattr(self.bot, "_global_phase"):
            self.bot._global_phase = "IDLE"
            self.bot.log(
                "[STATE_RESET] Post-attack: _global_phase → IDLE (zincir sonu)",
                level="DEBUG",
            )

        # Hizli zincir yoksa / bittiyse normal strateji
        self.bot.combat.check_strategic_wait(current_target)
