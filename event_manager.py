import time
from datetime import datetime, time as dt_time


class EventManager:
    """Zamanlanmis oyun ici etkinlikleri yonetir."""

    # WildLands: canli boss anchor anahtarlari (GeminiProConfig koordinatlariyla eslesen)
    _WILD_LANDS_ALIVE_KEYS = [
        "wildboss_1_mage",
        "wildboss_2_blood",
        "wildboss_3_overlord",
        "wildboss_4_dragon",
    ]

    def __init__(self, bot, events_config: dict):
        self.bot = bot
        self.events_config = events_config
        self.SCHEDULED_EVENTS = self._get_scheduled_events()
        self.last_event_run = {}
        self._defer_log_ts = {}

    def _get_scheduled_events(self):
        def parse_time(time_str):
            try:
                h, m = map(int, str(time_str).split(":"))
                return dt_time(hour=h, minute=m)
            except Exception:
                return None

        self.bot.log("Etkinlik takvimi yukleniyor...")
        processed_events = []

        for event in self.events_config.get("events", []):
            start = parse_time(event.get("start_time"))
            end = parse_time(event.get("end_time"))
            if not start or not end:
                continue

            processed_events.append(
                {
                    "name": event.get("name", "Bilinmeyen Etkinlik"),
                    "days": event.get("days"),
                    "start_time": start,
                    "end_time": end,
                    "location_name_on_enter": event.get("location_name_on_enter", "UNKNOWN"),
                    "auto_exits": bool(event.get("auto_exits", False)),
                    "action_sequence": event.get("action_sequence", []),
                }
            )

        self.bot.log(f"{len(processed_events)} adet etkinlik yuklendi.")
        return processed_events

    def event_monitor_thread(self):
        self.bot.log("Etkinlik izleyici aktif.")
        poll_interval = float(self.bot.settings.get("EVENT_MONITOR_POLL_SEC", 5))

        while True:
            try:
                if not self.bot.running.is_set() or self.bot.paused:
                    time.sleep(1.0)
                    continue

                now = datetime.now()
                current_time = now.time()
                current_weekday = now.weekday()
                today_str = now.strftime("%Y-%m-%d")

                old_keys = [k for k in self.last_event_run if not k.startswith(today_str)]
                for key in old_keys:
                    del self.last_event_run[key]
                old_defer_keys = [k for k in self._defer_log_ts if not k.startswith(today_str)]
                for key in old_defer_keys:
                    del self._defer_log_ts[key]

                for event in self.SCHEDULED_EVENTS:
                    day_match = event["days"] is None or current_weekday in event["days"]
                    time_match = event["start_time"] <= current_time < event["end_time"]
                    event_key = f"{today_str}_{event['name']}_{event['start_time'].strftime('%H:%M')}"

                    if day_match and time_match and not self.last_event_run.get(event_key):
                        if self._should_defer_event(event, event_key):
                            continue
                        self._run_scheduled_event(event, event_key)
                        break

                time.sleep(poll_interval)
            except Exception as exc:
                self.bot.log(f"Etkinlik dongusu hatasi: {exc}")
                time.sleep(5)

    def _is_boss_flow_active(self) -> bool:
        attacking = bool(getattr(self.bot, "attacking_target_aciklama", None))
        auto_active = bool(getattr(self.bot, "_auto_log_active", False))
        auto_trigger = str(getattr(self.bot, "_auto_log_trigger", "")).strip().lower()
        phase = str(getattr(self.bot, "_global_phase", "")).strip().upper()
        # LOOT_PHASE dahil: boss öldükten sonra yağma aşamasında da etkinlik ateşlenmemeli
        in_boss_phase = phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE"}
        return attacking or (auto_active and auto_trigger == "boss_attack") or in_boss_phase

    def _wild_lands_all_dead(self, coord_map: dict) -> bool:
        """
        WildLands boss listesindeki tum canli sablonlari arar.
        Hicbiri bulunamazsa (hepsi olu / yarin yenilenir) True doner.
        Canlı en az bir boss varsa False doner.
        """
        region = self.bot.ui_regions.get(
            "region_wild_lands_boss_list",
            {"x": 594, "y": 356, "w": 1374, "h": 786},
        )
        for key in self._WILD_LANDS_ALIVE_KEYS:
            coord = coord_map.get(key, {})
            anchor = coord.get("anchor", {}) if isinstance(coord, dict) else {}
            img = anchor.get("image_file", "")
            conf = float(anchor.get("confidence", 0.75))
            if img:
                try:
                    if self.bot.vision.find(img, region, conf):
                        return False  # Canli boss bulundu
                except Exception:
                    pass
        return True  # Hicbir canli boss gorsel eslesmesi yok

    def _should_defer_event(self, event: dict, event_key: str) -> bool:
        if not self._is_boss_flow_active():
            return False

        now = time.monotonic()
        last_log = float(self._defer_log_ts.get(event_key, 0.0))
        if now - last_log >= 15.0:
            self.bot.log(
                f"Etkinlik ertelendi: {event.get('name', 'unknown')} "
                "(aktif boss akisi tamamlanmadi).",
                level="DEBUG",
            )
            self._defer_log_ts[event_key] = now
        return True

    def _wait_for_combat_to_finish(self, max_wait_sn: float = 60.0) -> bool:
        """
        Aktif boss akışı (NAV/COMBAT/LOOT_PHASE) tamamlanana kadar bekler.
        max_wait_sn saniye içinde bitmezse False döner ve arayana temizlik işlemi
        yapması gerektiğini bildirir. Bot durdurulursa veya duraklatılırsa
        erken çıkış yapar.

        Returns:
            True  → combat düzgün bitti (etkinliğe normal girebiliriz).
            False → timeout; force_cleanup_combat_state() çağrısı gerekiyor.
        """
        if not self._is_boss_flow_active():
            return True  # Zaten aktif combat yok

        self.bot.log(
            f"[EVENT_WAIT] Aktif boss akisi bekleniyor (maks {int(max_wait_sn)} sn)...",
            level="DEBUG",
        )
        deadline = time.monotonic() + max_wait_sn
        poll_interval = 2.0

        while time.monotonic() < deadline:
            if not self.bot.running.is_set() or self.bot.paused:
                return False  # Bot durduruldu/duraklatıldı
            if not self._is_boss_flow_active():
                self.bot.log("[EVENT_WAIT] Boss akisi tamamlandi, etkinlige geciliyor.", level="DEBUG")
                return True
            time.sleep(poll_interval)

        self.bot.log(
            f"[EVENT_WAIT] Boss akisi {int(max_wait_sn)} sn icinde bitmedi; zorla temizlenecek.",
            level="WARNING",
        )
        return False

    def _run_scheduled_event(self, event, event_key):
        self.bot.log(f"Etkinlik zamani geldi: {event['name']}")

        # Aktif boss akışı (NAV/COMBAT/LOOT) bitmeden EVENT_PHASE'e geçme.
        # İkinci savunma hattı: _should_defer_event LOOT_PHASE'i kaçırsa bile bu blok yakalar.
        max_combat_wait = float(self.bot.settings.get("EVENT_COMBAT_WAIT_SN", 60.0))
        combat_done = self._wait_for_combat_to_finish(max_wait_sn=max_combat_wait)
        if not combat_done:
            # 60 saniye bitti, bot sıkışmış → zorla temizle
            self.bot.log(
                f"[{event['name']}] Boss akisi timeout; combat durumu zorla temizleniyor.",
                level="WARNING",
            )
            if hasattr(self.bot, "force_cleanup_combat_state"):
                self.bot.force_cleanup_combat_state(reason="event_preempt")

        self.bot.start_global_mission(
            phase="EVENT_PHASE",
            stage="Event_Start",
            reason=event["name"],
            extra={"event_flag": "Event_Start", "event_name": event["name"]},
        )
        self.bot.set_global_mission_phase(
            phase="EVENT_PHASE",
            stage="Event_Start",
            reason=event["name"],
            extra={"event_flag": "Event_Start", "event_name": event["name"]},
        )
        self.bot.active_event = event
        action_success = False
        try:
            lock_timeout = float(self.bot.settings.get("EVENT_ACTION_LOCK_TIMEOUT_SN", 0.5))
            lock_acquired = self.bot.action_lock.acquire(timeout=max(0.1, lock_timeout))
            if not lock_acquired:
                self.bot.log(
                    f"Etkinlik ertelendi: {event['name']} (action_lock mesgul).",
                    level="DEBUG",
                )
                return
            try:
                if not self.bot.running.is_set() or self.bot.paused:
                    return

                if self.bot.location_manager.get_region_name() != "EXP_FARM":
                    if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=False):
                        self.bot.log("Etkinlik oncesi EXP_FARM'a donulemedi.")
                        return

                # Dinamik timeout: etkinlik kalan suresi + 60s tampon (min 120s, max 600s).
                # Sabit 3600s yerine, etkinlik bitince kayit da durur → disk/RAM guvenligi.
                _end_dt = datetime.combine(datetime.now().date(), event["end_time"])
                _remaining_sn = max(0, (_end_dt - datetime.now()).total_seconds())
                _rec_timeout = min(max(int(_remaining_sn) + 60, 120), 600)
                self.bot.auto_start_recording("event_entry", timeout_sec=_rec_timeout)

                action_sequence = event.get("action_sequence", [])
                if not action_sequence:
                    self.bot.log(f"HATA: '{event['name']}' icin action_sequence bos.")
                else:
                    action_success = self.bot._run_sequence(
                        action_sequence,
                        self.bot.timed_events_cfg.get("coordinates", {}),
                    )

                if action_success:
                    self.bot.set_global_mission_phase(
                        phase="EVENT_PHASE",
                        stage="Event_Action",
                        reason=event["name"],
                        extra={"event_name": event["name"]},
                    )
                    self.bot.location_manager.set_current_location_by_name(event["location_name_on_enter"])
                    self.bot.log(f"'{event['name']}' etkinligine giris basarili.")
                    self.bot.log_training_outcome(
                        "event_entry",
                        {"event_name": event["name"], "success": True},
                    )
                    if hasattr(self.bot, "reward_engine"):
                        self.bot.reward_engine.on_event_entry(event_name=event["name"], success=True)
                else:
                    self.bot.log(f"'{event['name']}' giris basarisiz.")
                    self.bot.log_training_outcome(
                        "event_entry",
                        {"event_name": event["name"], "success": False},
                    )
                    if hasattr(self.bot, "reward_engine"):
                        self.bot.reward_engine.on_event_entry(event_name=event["name"], success=False)

                seq = getattr(self.bot, "seq_recorder", None)
                if seq is not None:
                    if action_success:
                        seq.signal_success(reason=f"event_{event['name']}_success")
                    else:
                        seq.signal_fail(reason=f"event_{event['name']}_fail")
            finally:
                self.bot.action_lock.release()

            if action_success and event.get("auto_exits", False):
                end_dt = datetime.combine(datetime.now().date(), event["end_time"])
                wait_seconds = max(0, (end_dt - datetime.now()).total_seconds()) + 5
                if wait_seconds > 0:
                    self.bot.set_global_mission_phase(
                        phase="EVENT_PHASE",
                        stage="Event_Wait",
                        reason=event["name"],
                        extra={"event_name": event["name"]},
                    )
                    # --- DEĞİŞTİRİLEN KISIM: Akıllı Bekleme ---
                    wild_coords = self.bot.timed_events_cfg.get("coordinates", {})
                    _is_wild = "Wild Lands" in event.get("name", "")

                    end_wait = time.monotonic() + wait_seconds
                    while time.monotonic() < end_wait and self.bot.running.is_set():
                        # WildLands: canli boss kalmadiginda aninda cik
                        if _is_wild and self._wild_lands_all_dead(wild_coords):
                            self.bot.log(
                                f"[{event['name']}] Tum bosslar olu (gorsel esleme), "
                                "etkinlik erken bitiriliyor."
                            )
                            break
                        self.bot.location_manager.update_visual_location()
                        if self.bot.location_manager.get_region_name() == "EXP_FARM":
                            self.bot.log(f"[{event['name']}] Erken bittiği tespit edildi (EXP_FARM'a dönüldü).")
                            break
                        if not self.bot._interruptible_wait(5.0): # 5 saniyede bir uyan ve kontrol et
                            break
                    # ------------------------------------------

                self.bot.location_manager.set_current_location_by_name("EXP_FARM")
                self.bot.stop_global_mission(reason=f"event_completed:{event['name']}")
                self.bot.auto_stop_recording() # <--- KAYDI ANINDA KAPAT

            elif action_success and not event.get("auto_exits", False):
                end_dt = datetime.combine(datetime.now().date(), event["end_time"])
                wait_seconds = max(0, (end_dt - datetime.now()).total_seconds())
                if wait_seconds > 0:
                    self.bot.set_global_mission_phase(
                        phase="EVENT_PHASE",
                        stage="Event_Wait",
                        reason=event["name"],
                        extra={"event_name": event["name"]},
                    )
                    # --- DEĞİŞTİRİLEN KISIM: Akıllı Bekleme ---
                    end_wait = time.monotonic() + wait_seconds
                    while time.monotonic() < end_wait and self.bot.running.is_set():
                        # Etkinlik bittiğinde "exit_map" butonu çıkıyorsa erken çıkış yap
                        if self.bot.automator.safe_exit_map():
                            self.bot.log(f"[{event['name']}] Çıkış butonu bulundu, etkinlik erken bitiriliyor.")
                            break
                        if not self.bot._interruptible_wait(5.0): # 5 saniyede bir kontrol et
                            break
                    # ------------------------------------------

                with self.bot.action_lock:
                    if self.bot.running.is_set():
                        self.bot.automator.return_to_exp_farm(force_restart_if_failed=True)
                
                self.bot.auto_stop_recording() # <--- DÖNÜŞ YAPILDIKTAN SONRA ANINDA KAPAT
            self.last_event_run[event_key] = True
        finally:
            self.bot.active_event = None
            self.bot.location_manager.update_visual_location()
            if self.bot.location_manager.get_region_name() == "EXP_FARM":
                self.bot.stop_global_mission(reason=f"event_finalize:{event['name']}")
            self.bot.auto_stop_recording()
