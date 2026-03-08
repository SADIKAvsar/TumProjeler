# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional

from ai_engine import AIEngine
from click_knowledge import ClickKnowledgeBase
from pytorch_inference import ACTION_COMMAND_MAP, PyTorchInferenceEngine

_ACTION_CONF_THRESHOLD: float = 0.60
_PHASE_CONF_THRESHOLD: float = 0.55

_NOOP_ACTIONS = frozenset({"noop", "unknown", "key_unknown"})


class TacticalBrain:

    def __init__(self, bot):
        self.bot = bot
        self.ai_engine = AIEngine(bot=self.bot)
        self._inference: Optional[PyTorchInferenceEngine] = None
        self._ollama = None
        self._click_kb: Optional[ClickKnowledgeBase] = None
        self._init_inference_engine()
        self._init_ollama()
        self._init_click_knowledge()
        self.bot.log("TacticalBrain: karar motoru hazir.")

    def _init_inference_engine(self):
        try:
            self._inference = PyTorchInferenceEngine(logger=self.bot)
            if self._inference.is_loaded:
                self.bot.log(
                    "TacticalBrain: Alamet modeli CUDA'da aktif "
                    f"(device={self._inference.get_statistics()['device']})."
                )
            else:
                self.bot.log(
                    "TacticalBrain: Alamet modeli yuklenemedi - "
                    "kural tabanli mod aktif.",
                    level="WARNING",
                )
        except Exception as exc:
            self.bot.log(
                f"TacticalBrain: Inference engine baslatma hatasi: {exc}",
                level="WARNING",
            )
            self._inference = None

    def _init_click_knowledge(self):
        try:
            cfg = getattr(self.bot, "general_cfg", {})
            w = int(cfg.get("screen_w", 2560))
            h = int(cfg.get("screen_h", 1440))
            self._click_kb = ClickKnowledgeBase(screen_w=w, screen_h=h, logger=self.bot.log)
            summary = self._click_kb.summary()
            self.bot.log(
                f"TacticalBrain: ClickKnowledgeBase hazir - "
                f"statik={summary['static_entries']}, "
                f"dinamik={summary['dynamic_entries']}"
            )
        except Exception as exc:
            self._click_kb = None
            self.bot.log(
                f"TacticalBrain: ClickKnowledgeBase baslatma hatasi: {exc}",
                level="WARNING",
            )

    def _resolve_click_coord(self, phase: str) -> Optional[Dict]:
        if self._click_kb is None:
            return None
        return self._click_kb.get_coord(phase, "mouse_click")
    def _derive_navigation_protocol(self, boss: Dict) -> str:
        current_region = "UNKNOWN"
        try:
            if hasattr(self.bot, "location_manager"):
                current_region = str(self.bot.location_manager.get_region_name() or "UNKNOWN").upper()
        except Exception:
            current_region = "UNKNOWN"

        katman = str((boss or {}).get("katman_id", "")).lower()
        target_loc = "KATMAN_1" if "katman_1" in katman else "KATMAN_2"
        if current_region != target_loc:
            return "FULL_MENU_SEQUENCE"
        return "SHORT_LIST_SEQUENCE"

    def _init_ollama(self):
        settings = getattr(self.bot, "settings", {})
        if not settings.get("OLLAMA_ENABLED", True):
            self.bot.log("TacticalBrain: Ollama devre disi (OLLAMA_ENABLED=false).")
            return
        try:
            from ollama_client import OllamaClient
            model    = settings.get("OLLAMA_MODEL", "qwen3-vl:4b")
            base_url = settings.get("OLLAMA_BASE_URL", "http://localhost:11434")
            timeout  = float(settings.get("OLLAMA_TIMEOUT_SEC", 60.0))
            min_conf = float(settings.get("OLLAMA_MIN_CONFIDENCE", 0.50))
            num_gpu        = int(settings.get("OLLAMA_NUM_GPU", 99))
            vision_enabled = bool(settings.get("OLLAMA_VISION_ENABLED", True))
            self._ollama = OllamaClient(
                logger=self.bot,
                model=model,
                base_url=base_url,
                timeout_sec=timeout,
                min_confidence=min_conf,
                num_gpu=num_gpu,
                vision_enabled=vision_enabled,
            )
            self.bot.log(f"TacticalBrain: Ollama istemcisi hazir (model={model}).")
        except Exception as exc:
            self._ollama = None
            self.bot.log(
                f"TacticalBrain: Ollama baslatma hatasi: {exc}",
                level="WARNING",
            )

    def decide_next_target(self, ready_bosses: List[Dict]) -> Optional[Dict]:
        """AIEngine kural motoru ile bir sonraki boss'u secer."""
        if not self.ai_engine or not ready_bosses:
            return None

        ai_decision = self.ai_engine.decide_next_boss(ready_bosses)
        if not ai_decision:
            return None

        boss_name = None
        ui_protocol = None
        if isinstance(ai_decision, dict):
            boss_name = (
                ai_decision.get("decision")
                or ai_decision.get("boss_name")
                or ai_decision.get("boss")
            )
            raw_protocol = str(ai_decision.get("ui_protocol", "")).strip().upper()
            if raw_protocol:
                ui_protocol = raw_protocol
        else:
            boss_name = ai_decision

        for boss in ready_bosses:
            if str(boss.get("aciklama", "")) == str(boss_name):
                protocol = ui_protocol or self._derive_navigation_protocol(boss)
                boss["_ui_protocol"] = str(protocol).strip().upper()
                self.bot.log(
                    f"TacticalBrain: ui_protocol koprusu aktif -> {boss.get('_ui_protocol')} "
                    f"(boss={boss.get('aciklama')})",
                    level="DEBUG",
                )
                return boss
        return None

    def analyze_combat(self) -> Optional[Dict]:
        result = self._inference_predict()
        if result is None:
            return None

        if result["phase_confidence"] < _PHASE_CONF_THRESHOLD:
            self.bot.log(
                f"TacticalBrain: Faz guven dusuk ({result['phase_confidence']:.0%}) - "
                "sonuc geri cekildi.",
                level="DEBUG",
            )
            return None

        return {
            "phase": result["phase_name"],
            "phase_confidence": result["phase_confidence"],
            "recommended_action": result["action_name"],
            "action_confidence": result["action_confidence"],
            "inference_ms": result["inference_ms"],
        }

    def get_next_move(self):

        result = self._inference_predict()
        if result is None:
            return "CONTINUE_FARM"

        action_conf = result["action_confidence"]
        phase_conf = result["phase_confidence"]

        if action_conf < _ACTION_CONF_THRESHOLD or phase_conf < _PHASE_CONF_THRESHOLD:
            self.bot.log(
                f"TacticalBrain: Guven esigi altinda "
                f"(aksiyon={action_conf:.0%}, faz={phase_conf:.0%}) - "
                "Ollama deneniyor...",
                level="DEBUG",
            )
            ollama_move = self._ollama_ask()
            if ollama_move is not None:
                return ollama_move
            return "CONTINUE_FARM"

        action_name = result["action_name"]

        if action_name in _NOOP_ACTIONS:
            return "CONTINUE_FARM"

        command = ACTION_COMMAND_MAP.get(action_name, {"type": "noop"})
        if command["type"] == "noop":
            return "CONTINUE_FARM"

        if command["type"] == "click":
            coord = self._resolve_click_coord(result["phase_name"])
            if coord is None:
                self.bot.log(
                    f"TacticalBrain: mouse_click koordinati bulunamadi "
                    f"(faz={result['phase_name']}) - CONTINUE_FARM'a dusuluyor.",
                    level="WARNING",
                )
                return "CONTINUE_FARM"
            command = {**command, "coord": coord}

        return {
            "source": "alamet_ai",
            "action": action_name,
            "command": command,
            "phase": result["phase_name"],
            "action_confidence": action_conf,
            "phase_confidence": phase_conf,
            "inference_ms": result["inference_ms"],
        }

    def get_ai_statistics(self) -> Dict:
        stats: Dict = {}
        if self.ai_engine:
            try:
                stats["rule_engine"] = self.ai_engine.get_statistics()
            except Exception:
                stats["rule_engine"] = {"error": "istatistik_alinamadi"}
        if self._inference:
            stats["alamet_model"] = self._inference.get_statistics()
        else:
            stats["alamet_model"] = {"loaded": False, "error": "engine_baslatilamadi"}
        if self._ollama:
            stats["ollama"] = self._ollama.get_statistics()
        else:
            stats["ollama"] = {"available": False}
        return stats

    # -- Kapatma -----------------------------------------------------------------

    def shutdown(self):
        """Bot kapanirken VRAM monitor thread'ini temizler."""
        if self._inference:
            self._inference.shutdown()

    # -- Dahili Yardimci ----------------------------------------------------------

    def _ollama_ask(self) -> Optional[Dict]:
        """Ollama VL modelinden ekran yorumu ve aksiyon karari alir."""
        if self._ollama is None:
            return None
        if not hasattr(self.bot, "vision") or self.bot.vision is None:
            return None

        try:
            frame = self.bot.vision.capture_full_screen()
            if frame is None:
                return None

            bot_state = {
                "faz"           : getattr(self.bot, "_global_phase", "UNKNOWN"),
                "konum"         : (
                    self.bot.location_manager.get_region_name()
                    if hasattr(self.bot, "location_manager") else "UNKNOWN"
                ),
                "saldiri_hedefi": getattr(self.bot, "attacking_target_aciklama", None) or "yok",
            }

            result = self._ollama.ask(frame, bot_state)
            if result is None:
                return None

            action  = result["action"]
            command = ACTION_COMMAND_MAP.get(action, {"type": "noop"})
            if command["type"] == "noop":
                return None

            # mouse_click -> koordinat coz
            current_phase = getattr(self.bot, "_global_phase", "UNKNOWN")
            if command["type"] == "click":
                coord = self._resolve_click_coord(current_phase)
                if coord is None:
                    return None
                command = {**command, "coord": coord}

            return {
                "source"           : "ollama_ai",
                "action"           : action,
                "command"          : command,
                "phase"            : current_phase,
                "action_confidence": result["confidence"],
                "phase_confidence" : 1.0,
                "inference_ms"     : result.get("latency_ms", 0.0),
            }

        except Exception as exc:
            self.bot.log(
                f"TacticalBrain: _ollama_ask hatasi: {exc}", level="WARNING"
            )
            return None

    def _inference_predict(self) -> Optional[Dict]:
        """Inference engine hazir ve vision aktifse ekrandan tahmin yapar."""
        if (
            self._inference is None
            or not self._inference.is_loaded
            or not hasattr(self.bot, "vision")
            or self.bot.vision is None
        ):
            return None
        return self._inference.predict_from_screen(self.bot.vision)
