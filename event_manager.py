import time
from datetime import datetime, time as dt_time


class EventManager:
    """Zamanlanmis oyun ici etkinlikleri yonetir."""

    def __init__(self, bot, events_config: dict):
        self.bot = bot
        self.events_config = events_config
        self.SCHEDULED_EVENTS = self._get_scheduled_events()
        self.last_event_run = {}

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

                for event in self.SCHEDULED_EVENTS:
                    day_match = event["days"] is None or current_weekday in event["days"]
                    time_match = event["start_time"] <= current_time < event["end_time"]
                    event_key = f"{today_str}_{event['name']}_{event['start_time'].strftime('%H:%M')}"

                    if day_match and time_match and not self.last_event_run.get(event_key):
                        self._run_scheduled_event(event, event_key)
                        break

                time.sleep(poll_interval)
            except Exception as exc:
                self.bot.log(f"Etkinlik dongusu hatasi: {exc}")
                time.sleep(5)

    def _run_scheduled_event(self, event, event_key):
        self.bot.log(f"Etkinlik zamani geldi: {event['name']}")
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
            with self.bot.action_lock:
                if not self.bot.running.is_set() or self.bot.paused:
                    self.last_event_run[event_key] = True
                    return

                if self.bot.location_manager.get_region_name() != "EXP_FARM":
                    if not self.bot.automator.return_to_exp_farm(force_restart_if_failed=False):
                        self.bot.log("Etkinlik oncesi EXP_FARM'a donulemedi.")
                        self.last_event_run[event_key] = True
                        return

                self.bot.auto_start_recording("event_entry", timeout_sec=3600)

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
                else:
                    self.bot.log(f"'{event['name']}' giris basarisiz.")
                    self.bot.log_training_outcome(
                        "event_entry",
                        {"event_name": event["name"], "success": False},
                    )

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
                    end_wait = time.monotonic() + wait_seconds
                    while time.monotonic() < end_wait and self.bot.running.is_set():
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
