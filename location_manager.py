class LocationManager:
    def __init__(self, bot, general_cfg: dict):
        self.bot = bot
        self.general_cfg = general_cfg
        self.LOCATION_MAP = {}
        self.NAME_TO_VALUE = {}
        self._build_location_maps()
        self.current_location_value = self.NAME_TO_VALUE.get("UNKNOWN", -1)
        self._unknown_streak = 0

    _NAME_ALIASES = {
        # Config typo compatibility: event config may use EXTREMISCASTLE
        "EXTREMISCASTLE": "EXREMISCASTLE",
    }

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

    def _sync_game_state_region(self, region_name: str) -> None:
        state = getattr(self.bot, "game_state", None)
        if state is None:
            return
        try:
            state.update(
                {
                    "current_region": str(region_name).upper(),
                    "current_location": str(region_name).upper(),
                },
                source="location_manager",
            )
        except Exception:
            pass

    def set_current_location_by_name(self, name: str):
        key = str(name).upper()
        key = self._NAME_ALIASES.get(key, key)
        val = self.NAME_TO_VALUE.get(key)
        if val is None:
            self.bot.log(f"UYARI: set_current_location_by_name('{name}') basarisiz.")
            return

        self.current_location_value = val
        self._unknown_streak = 0
        self._sync_game_state_region(key)
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
                region_key = anchor.get("region_key")
                reg = self.bot.ui_regions.get(region_key) or self.bot.ui_regions.get(
                    "region_full_screen",
                    {"x": 0, "y": 0, "w": 2560, "h": 1440},
                )
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
        old_name = self.get_region_name()
        phase = str(getattr(self.bot, "_global_phase", "")).strip().upper()
        in_event_phase = phase == "EVENT_PHASE" or bool(getattr(self.bot, "active_event", None))

        if new_val == unknown_val:
            self._unknown_streak += 1
            # If we are in event flow, keeping stale KATMAN state is worse than UNKNOWN.
            if in_event_phase and old_name.startswith("KATMAN_"):
                self.current_location_value = unknown_val
                self._sync_game_state_region("UNKNOWN")
                self.bot.log(f"KONUM SENKRON: '{old_name}' -> 'UNKNOWN' (event fazi, anchor yok)")
            return self.current_location_value

        self._unknown_streak = 0
        if new_val != self.current_location_value:
            self.current_location_value = new_val
            self._sync_game_state_region(self.get_region_name())
            self.bot.log(f"KONUM SENKRON: '{old_name}' -> '{self.get_region_name()}'")

        return self.current_location_value
