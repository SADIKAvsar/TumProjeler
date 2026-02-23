"""
tactical_brain.py — Karar Orkestrasyon Katmanı
================================================
Kural tabanlı AIEngine (Tier-1) + Alamet PyTorch Modeli (Tier-2) +
Ollama VL Model (Tier-2b) birlikte çalışır.

Karar hiyerarşisi:
  1. AIEngine    → boss seçimi, freeze teşhisi (kural tabanlı, her zaman aktif)
  2. Alamet AI   → ekran görüntüsünden faz + aksiyon tahmini (CUDA, güven eşiği kontrolü)
  2b. Ollama VL  → PyTorch güveni düşükse devreye girer (yerel VL model, isteğe bağlı)
  3. Fallback    → ikisi de başarısızsa "CONTINUE_FARM" (mevcut kural tabanlı akış)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ai_engine import AIEngine
from click_knowledge import ClickKnowledgeBase
from pytorch_inference import ACTION_COMMAND_MAP, PyTorchInferenceEngine

# ── Güven Eşikleri ───────────────────────────────────────────────────────
# Modelin %90 aksiyon / %86 faz doğruluğu göz önüne alındığında güvenli eşikler.
_ACTION_CONF_THRESHOLD: float = 0.60
_PHASE_CONF_THRESHOLD: float = 0.55

# "Eylem yok" olarak değerlendirilen aksiyon isimleri → CONTINUE_FARM'a düşülür
_NOOP_ACTIONS = frozenset({"noop", "unknown", "key_unknown"})


class TacticalBrain:
    """
    Bot karar merkezi.

    Dışa açık metodlar:
      decide_next_target(ready_bosses) → seçilen boss dict veya None
      analyze_combat()                 → faz analizi dict veya None
      get_next_move()                  → AI komut dict veya "CONTINUE_FARM"
      get_ai_statistics()              → birleşik istatistik dict
      shutdown()                       → temiz kapatma
    """

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

    # ── Başlatma ──────────────────────────────────────────────────────────

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
                    "TacticalBrain: Alamet modeli yuklenemedi — "
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
        """ClickKnowledgeBase'i başlatır — YAML statik + öğrenilmiş dinamik koordinatlar."""
        try:
            cfg = getattr(self.bot, "general_cfg", {})
            w = int(cfg.get("screen_w", 2560))
            h = int(cfg.get("screen_h", 1440))
            self._click_kb = ClickKnowledgeBase(screen_w=w, screen_h=h, logger=self.bot.log)
            summary = self._click_kb.summary()
            self.bot.log(
                f"TacticalBrain: ClickKnowledgeBase hazir — "
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
        """
        mouse_click kararı için koordinat çözer.
        1. ClickKnowledgeBase (dinamik > statik)
        2. Bulunamazsa None döner (caller fallback'e geçer)
        """
        if self._click_kb is None:
            return None
        return self._click_kb.get_coord(phase, "mouse_click")

    def _init_ollama(self):
        """Ollama VL istemcisini ayarlardan başlatır (isteğe bağlı)."""
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

    # ── Kural Tabanlı: Boss Seçimi (Tier-1) ──────────────────────────────

    def decide_next_target(self, ready_bosses: List[Dict]) -> Optional[Dict]:
        """AIEngine kural motoru ile bir sonraki boss'u seçer."""
        if not self.ai_engine or not ready_bosses:
            return None

        boss_name = self.ai_engine.decide_next_boss(ready_bosses)
        if not boss_name:
            return None

        for boss in ready_bosses:
            if str(boss.get("aciklama", "")) == str(boss_name):
                return boss
        return None

    # ── Alamet AI: Savaş Faz Analizi (Tier-2) ────────────────────────────

    def analyze_combat(self) -> Optional[Dict]:
        """
        Mevcut ekran görüntüsünden savaş fazını analiz eder.

        Returns:
            {
                "phase": str,
                "phase_confidence": float,
                "recommended_action": str,
                "action_confidence": float,
                "inference_ms": float,
            }
            veya None (model yüklü değil / güven düşük)
        """
        result = self._inference_predict()
        if result is None:
            return None

        if result["phase_confidence"] < _PHASE_CONF_THRESHOLD:
            self.bot.log(
                f"TacticalBrain: Faz guven dusuk ({result['phase_confidence']:.0%}) — "
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

    # ── Alamet AI: Sonraki Hareket (Tier-2) ──────────────────────────────

    def get_next_move(self):
        """
        Ekran görüntüsünden sonraki hareketi belirler.

        Returns:
            dict  → AI'nın yüksek güvenle belirlediği komut:
                    {
                        "source": "alamet_ai",
                        "action": str,          # "key_a", "mouse_click" vb.
                        "command": dict,        # {"type": "key", "key": "a"} vb.
                        "phase": str,
                        "action_confidence": float,
                        "phase_confidence": float,
                        "inference_ms": float,
                    }

            "CONTINUE_FARM"  → AI yoksa / güven düşükse / noop aksiyonsa
        """
        result = self._inference_predict()
        if result is None:
            return "CONTINUE_FARM"

        action_conf = result["action_confidence"]
        phase_conf = result["phase_confidence"]

        if action_conf < _ACTION_CONF_THRESHOLD or phase_conf < _PHASE_CONF_THRESHOLD:
            self.bot.log(
                f"TacticalBrain: Guven esigi altinda "
                f"(aksiyon={action_conf:.0%}, faz={phase_conf:.0%}) — "
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

        # mouse_click → koordinat çöz (ClickKnowledgeBase: dinamik > YAML statik)
        if command["type"] == "click":
            coord = self._resolve_click_coord(result["phase_name"])
            if coord is None:
                self.bot.log(
                    f"TacticalBrain: mouse_click koordinati bulunamadi "
                    f"(faz={result['phase_name']}) — CONTINUE_FARM'a dusuluyor.",
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

    # ── İstatistikler ─────────────────────────────────────────────────────

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

    # ── Kapatma ───────────────────────────────────────────────────────────

    def shutdown(self):
        """Bot kapanırken VRAM monitör thread'ini temizler."""
        if self._inference:
            self._inference.shutdown()

    # ── Dahili Yardımcı ───────────────────────────────────────────────────

    def _ollama_ask(self) -> Optional[Dict]:
        """Ollama VL modelinden ekran yorumu ve aksiyon kararı alır."""
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

            # mouse_click → koordinat çöz
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
        """Inference engine hazır ve vision aktifse ekrandan tahmin yapar."""
        if (
            self._inference is None
            or not self._inference.is_loaded
            or not hasattr(self.bot, "vision")
            or self.bot.vision is None
        ):
            return None
        return self._inference.predict_from_screen(self.bot.vision)
