# -*- coding: utf-8 -*-
"""
ollama_client.py — Yerel Ollama Vision-Language Model İstemcisi
================================================================
qwen3-vl veya uyumlu Ollama VL modelini kullanarak ekran görüntüsü
üzerinden oyun aksiyonu kararı alır.

Entegrasyon noktası: TacticalBrain → OllamaClient
  PyTorchInferenceEngine güveni düşük kaldığında devreye girer.

Beklenen JSON çıktı formatı:
  {
    "action"     : "key_a",   // key_a | key_q | key_v | key_z | mouse_click | noop
    "confidence" : 0.85,      // 0.0 – 1.0
    "reasoning"  : "..."      // kısa açıklama (opsiyonel)
  }
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── Sabitler ──────────────────────────────────────────────────────────────

_VALID_ACTIONS = frozenset({
    "key_a", "key_q", "key_v", "key_z",
    "mouse_click", "noop",
})

_SYSTEM_PROMPT = (
    "You are a game bot decision module for League of Angels.\n"
    "Pick ONE action from: key_a, key_q, key_v, key_z, mouse_click, noop\n"
    "  key_a=attack  key_q=skill  key_v=close_menu  key_z=auto_move  mouse_click=click  noop=wait\n"
    "Reply with ONLY this JSON (no other text):\n"
    '{"action":"key_a","confidence":0.85,"reasoning":"boss health low attack now"}'
)


# ══════════════════════════════════════════════════════════════════════════
#  Ana İstemci
# ══════════════════════════════════════════════════════════════════════════

class OllamaClient:
    def __init__(
        self,
        logger,
        model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
        timeout_sec: float = 60.0,
        min_confidence: float = 0.50,
        max_width: int = 1280,
        jpeg_quality: int = 80,
        num_gpu: int = 99,
        vision_enabled: bool = True,
    ):
        self._log = logger
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_sec
        self._min_conf = min_confidence
        self._max_width = max_width
        self._jpeg_quality = jpeg_quality
        self._num_gpu = num_gpu  # GPU katmanları (99 = tümü GPU'da)
        # vision_enabled=False → görüntü göndermez, sadece metin (llama3 gibi modeller için)
        self._vision_enabled = vision_enabled

        # Sunucu durumu: None = henüz kontrol edilmedi
        self._available: Optional[bool] = None

        # İstatistikler
        self._call_count: int = 0
        self._fail_count: int = 0
        self._total_ms: float = 0.0

    # ── Dışa Açık API ─────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Ollama sunucusunu ilk çağrıda test eder, sonucu önbelleğe alır."""
        if self._available is None:
            self._available = self._check_server()
        return self._available

    def ask(
        self,
        frame_bgr: np.ndarray,
        bot_state: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if not _HAS_REQUESTS:
            self._log_msg("'requests' paketi eksik — pip install requests", "WARNING")
            return None
        if not self.is_available:
            return None
        if frame_bgr is None and self._vision_enabled:
            return None

        t0 = time.monotonic()
        try:
            image_b64 = self._encode_frame(frame_bgr) if self._vision_enabled and frame_bgr is not None else None
            prompt    = self._build_prompt(bot_state)
            raw       = self._send_request(prompt, image_b64)
            result    = self._parse_response(raw)

            if result is None:
                self._fail_count += 1
                return None

            if result["confidence"] < self._min_conf:
                self._log_msg(
                    f"Guven dusuk ({result['confidence']:.0%}) — karar reddedildi.",
                    "DEBUG",
                )
                return None

            latency_ms = (time.monotonic() - t0) * 1000
            self._call_count += 1
            self._total_ms += latency_ms
            result["latency_ms"] = round(latency_ms, 1)

            self._log_msg(
                f"Karar: {result['action']} "
                f"(guven={result['confidence']:.0%}, {latency_ms:.0f}ms) "
                f"| {result.get('reasoning', '')}",
                "DEBUG",
            )
            return result

        except Exception as exc:
            self._fail_count += 1
            self._log_msg(f"Hata: {exc}", "WARNING")
            return None

    def get_statistics(self) -> Dict:
        avg_ms = self._total_ms / self._call_count if self._call_count else 0.0
        return {
            "model"          : self._model,
            "available"      : self._available,
            "calls"          : self._call_count,
            "failures"       : self._fail_count,
            "avg_latency_ms" : round(avg_ms, 1),
        }

    def reset_availability(self):
        """Sunucu bağlantısını bir sonraki çağrıda yeniden test etmeye zorlar."""
        self._available = None

    # ── Dahili: Sunucu Kontrolü ────────────────────────────────────────────

    def _check_server(self) -> bool:
        if not _HAS_REQUESTS:
            return False
        try:
            resp = _requests.get(f"{self._base_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False

            models = [m.get("name", "") for m in resp.json().get("models", [])]
            base_name = self._model.split(":")[0].lower()
            match = any(base_name in m.lower() for m in models)

            if match:
                self._log_msg(f"Hazir. Model: {self._model}")
            else:
                self._log_msg(
                    f"Sunucu calisiyor ama '{self._model}' listede yok. "
                    f"Mevcut modeller: {models}",
                    "WARNING",
                )
                # Yine de dene; model adı kısmen farklı olabilir
            return True

        except Exception:
            self._log_msg(
                "localhost:11434 erisilemez — Ollama calisiyor mu?",
                "WARNING",
            )
            return False

    # ── Dahili: Görüntü İşleme ─────────────────────────────────────────────

    def _encode_frame(self, frame_bgr: np.ndarray) -> str:
        """BGR kareyi yeniden boyutlandırıp JPEG → base64 yapar."""
        h, w = frame_bgr.shape[:2]
        if w > self._max_width:
            scale = self._max_width / w
            frame_bgr = cv2.resize(
                frame_bgr,
                (self._max_width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encode basarisiz")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ── Dahili: Prompt ─────────────────────────────────────────────────────

    def _build_prompt(self, bot_state: Optional[Dict]) -> str:
        if not bot_state:
            return "Ekrani analiz et, en uygun aksiyonu sec."

        lines = ["Mevcut bot durumu:"]
        for key, val in bot_state.items():
            if val:
                lines.append(f"  {key}: {val}")
        lines.append("\nBu durumda hangi aksiyon uygun?")
        return "\n".join(lines)

    # ── Dahili: API İsteği ─────────────────────────────────────────────────

    def _send_request(self, prompt: str, image_b64: Optional[str]) -> str:
        options = {
            "temperature": 0.1,
            "num_predict": 128,   # Kısa yanıt yeterli (JSON tek satır)
            "num_gpu"    : self._num_gpu,
        }

        if image_b64 is not None:
            # Görüntü modu → /api/chat (VL modeller için)
            payload = {
                "model"  : self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt, "images": [image_b64]},
                ],
                "stream" : False,
                "think"  : False,
                "options": {**options, "num_predict": 512},  # Thinking bütçesi
            }
            resp = _requests.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
            msg     = resp.json()["message"]
            content = msg.get("content", "")
            if not content and msg.get("thinking"):
                t = msg["thinking"]
                s, e = t.rfind("{"), t.rfind("}") + 1
                if s >= 0 and e > s:
                    content = t[s:e]
            return content
        else:
            # Metin modu → /api/generate (daha hızlı, görüntüsüz)
            full_prompt = f"{_SYSTEM_PROMPT}\n\n{prompt}"
            payload = {
                "model"  : self._model,
                "prompt" : full_prompt,
                "stream" : False,
                "options": options,
            }
            resp = _requests.post(
                f"{self._base_url}/api/generate", json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    # ── Dahili: Yanıt Ayrıştırma ──────────────────────────────────────────

    def _parse_response(self, raw: str) -> Optional[Dict]:
        """Model çıktısından JSON action bloğunu çıkarır."""
        text = raw.strip()

        # Markdown code block temizle
        for marker in ("```json", "```"):
            if marker in text:
                text = text.split(marker)[-1].split("```")[0].strip()
                break

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Son çare: ilk { ... } bloğunu bul
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except Exception:
                    pass

        if data is None:
            self._log_msg(f"JSON ayristirilamadi: {text[:120]}", "WARNING")
            return None

        action = str(data.get("action", "noop")).strip().lower()
        if action not in _VALID_ACTIONS:
            self._log_msg(
                f"Gecersiz aksiyon '{action}' — noop'a donuldu.", "WARNING"
            )
            action = "noop"

        return {
            "action"    : action,
            "confidence": float(data.get("confidence", 0.0)),
            "reasoning" : str(data.get("reasoning", "")),
        }

    # ── Dahili: Loglama ───────────────────────────────────────────────────

    def _log_msg(self, msg: str, level: str = "INFO"):
        try:
            self._log.log(f"[Ollama] {msg}", level=level)
        except Exception:
            pass
