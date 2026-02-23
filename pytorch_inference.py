"""
pytorch_inference.py — Alamet Model Inference Engine
=====================================================
AgenticNet (ResNet18) modelini CUDA üzerinde yükler ve oyun karesi
üzerinden faz + aksiyon tahmini yapar.

Özellikler:
 - CUDA zorunlu: CPU'ya düşme engeli + açıklayıcı hata mesajı
 - torch.no_grad() → düşük latency inference döngüsü
 - Her inference için terminale latency logu
 - Her 5 dakikada bir VRAM kullanım raporu (daemon thread)
 - Eğitimle birebir preprocessing pipeline (BGR→RGB, 224×224, /255.0)
 - best.pt yoksa veya bozuksa açıklayıcı hata, bot çökmez
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# ── torchvision isteğe bağlı ──────────────────────────────────────────────
try:
    from torchvision.models import (
        EfficientNet_B0_Weights,
        ResNet18_Weights,
        efficientnet_b0,
        resnet18,
    )

    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# ── Sabitler ──────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent / "models" / "active_model"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "best.pt"
_IMAGE_SIZE = 224
_VRAM_LOG_INTERVAL_SEC = 300  # 5 dakika


# ══════════════════════════════════════════════════════════════════════════
#  Model Mimarisi (train_agentic.py ile birebir aynı — state_dict uyumluluğu)
# ══════════════════════════════════════════════════════════════════════════

class _TinyBackbone(nn.Module):
    """Torchvision yüklü değilse kullanılan küçük yedek backbone."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)


class _AgenticNet(nn.Module):
    """
    Çift kafali sınıflandırıcı: faz tahmini + aksiyon tahmini.
    train_agentic.py::AgenticNet ile birebir aynı mimari.
    """

    def __init__(
        self,
        num_phases: int,
        num_actions: int,
        backbone: str = "resnet18",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim = self._build_backbone(backbone)
        self.dropout = nn.Dropout(float(dropout))
        self.phase_head = nn.Linear(feat_dim, int(num_phases))
        self.action_head = nn.Linear(feat_dim, int(num_actions))

    def _build_backbone(self, backbone: str):
        b = str(backbone).strip().lower()
        if _HAS_TORCHVISION and b == "resnet18":
            model = resnet18(weights=None)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feat_dim
        if _HAS_TORCHVISION and b == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
            feat_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feat_dim
        tiny = _TinyBackbone(out_dim=256)
        return tiny, tiny.out_dim

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.phase_head(feat), self.action_head(feat)


# ══════════════════════════════════════════════════════════════════════════
#  Ana Inference Engine
# ══════════════════════════════════════════════════════════════════════════

class PyTorchInferenceEngine:
    """
    Alamet modelini CUDA üzerinde yükler ve tek-frame inference yapar.

    Kullanım:
        engine = PyTorchInferenceEngine(logger=bot)
        if engine.is_loaded:
            result = engine.predict(frame_bgr)
            # result = {action_id, action_name, phase_id, phase_name,
            #           action_confidence, phase_confidence, inference_ms}
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        logger: Any = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._device: Optional[torch.device] = None
        self._model: Optional[_AgenticNet] = None
        self._id_to_action: Dict[int, str] = {}
        self._id_to_phase: Dict[int, str] = {}
        self._num_actions: int = 11
        self._num_phases: int = 7
        self._backbone: str = "resnet18"
        self._loaded: bool = False
        self._lock = threading.Lock()

        # İstatistik
        self._inference_count: int = 0
        self._total_inference_ms: float = 0.0

        # VRAM monitor
        self._stop_event = threading.Event()
        self._vram_thread: Optional[threading.Thread] = None

        self._load_model()
        if self._loaded:
            self._start_vram_monitor()

    # ── Yükleme ───────────────────────────────────────────────────────────

    def _require_cuda(self) -> torch.device:
        """CUDA mevcut değilse açıklayıcı hata fırlatır — CPU'ya sessizce düşmez."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[Alamet] CUDA bulunamadi!\n"
                "  Beklenen: NVIDIA RTX 4060 Ti (CUDA 12.4)\n"
                "  Kontrol: python -c \"import torch; print(torch.cuda.is_available())\"\n"
                "  Cozum  : pip install torch torchvision "
                "--extra-index-url https://download.pytorch.org/whl/cu124"
            )
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        self._log(
            f"[Alamet] GPU: {props.name} | "
            f"VRAM: {props.total_memory / 1024**3:.1f} GB | "
            f"CUDA: {torch.version.cuda} | "
            f"PyTorch: {torch.__version__}"
        )
        return device

    def _load_model(self):
        """best.pt'yi yükler, modeli CUDA'ya taşır ve eval moduna alır."""

        # 1. Dosya varlık kontrolü
        if not self._model_path.exists():
            self._log(
                f"[Alamet] HATA: Model bulunamadi: {self._model_path.absolute()}\n"
                "  best.pt dosyasini models/active_model/ klasorune kopyalayin.",
                level="ERROR",
            )
            return

        # 2. CUDA zorunluluğu
        try:
            self._device = self._require_cuda()
        except RuntimeError as exc:
            self._log(str(exc), level="ERROR")
            return

        # 3. Checkpoint yükle
        self._log(f"[Alamet] Model yukleniyor: {self._model_path} ({self._model_path.stat().st_size / 1024**2:.1f} MB) ...")
        try:
            ckpt = torch.load(
                str(self._model_path),
                map_location=self._device,
                weights_only=False,
            )
        except Exception as exc:
            self._log(f"[Alamet] torch.load hatasi: {exc}", level="ERROR")
            return

        # 4. Checkpoint'ten model konfigürasyonu
        args = ckpt.get("args", {})
        self._backbone = str(args.get("backbone", "resnet18"))
        self._num_phases = int(args.get("num_phases", 7))
        self._num_actions = int(args.get("num_actions", 11))

        # 5. Aksiyon / Faz eşlemelerini yükle
        self._load_mappings(
            ckpt_action_to_id=ckpt.get("action_to_id", {}),
            ckpt_phase_map=ckpt.get("phase_id_to_name", {}),
        )

        # 6. Modeli oluştur ve ağırlıkları yükle
        try:
            self._model = _AgenticNet(
                num_phases=self._num_phases,
                num_actions=self._num_actions,
                backbone=self._backbone,
                dropout=0.2,  # Eğitimle aynı — eval() zaten kapatır
            ).to(self._device)

            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()
            torch.backends.cudnn.benchmark = True

        except Exception as exc:
            self._log(f"[Alamet] state_dict yukleme hatasi: {exc}", level="ERROR")
            self._model = None
            return

        self._loaded = True
        epoch = ckpt.get("epoch", "?")
        m = ckpt.get("metrics", {})
        self._log(
            f"[Alamet] Model hazir! "
            f"epoch={epoch} | "
            f"val_action_acc={m.get('val_action_acc', '?')} | "
            f"val_phase_acc={m.get('val_phase_acc', '?')} | "
            f"backbone={self._backbone} | "
            f"device={self._device}"
        )

    def _load_mappings(self, ckpt_action_to_id: dict, ckpt_phase_map: dict):
        """JSON dosyalarından (veya checkpoint'ten) id↔isim eşlemelerini yükler."""

        # Aksiyon eşlemesi: {"action_name": id} → {id: "action_name"}
        action_json = _MODEL_DIR / "action_to_id.json"
        if action_json.exists():
            try:
                raw = json.loads(action_json.read_text(encoding="utf-8"))
                self._id_to_action = {int(v): str(k) for k, v in raw.items()}
            except Exception:
                self._id_to_action = {int(v): str(k) for k, v in ckpt_action_to_id.items()}
        elif ckpt_action_to_id:
            self._id_to_action = {int(v): str(k) for k, v in ckpt_action_to_id.items()}

        # Faz eşlemesi: {"phase_id": "phase_name"}
        phase_json = _MODEL_DIR / "phase_id_to_name.json"
        if phase_json.exists():
            try:
                raw = json.loads(phase_json.read_text(encoding="utf-8"))
                self._id_to_phase = {int(k): str(v) for k, v in raw.items()}
            except Exception:
                self._id_to_phase = {int(k): str(v) for k, v in ckpt_phase_map.items()}
        elif ckpt_phase_map:
            self._id_to_phase = {int(k): str(v) for k, v in ckpt_phase_map.items()}

        self._log(
            f"[Alamet] Esleme yuklendi: "
            f"{len(self._id_to_action)} aksiyon, "
            f"{len(self._id_to_phase)} faz"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────

    @staticmethod
    def preprocess_frame(frame: np.ndarray, image_size: int = _IMAGE_SIZE) -> torch.Tensor:
        """
        BGR frame → CUDA tensor (eğitimle birebir aynı pipeline).

        train_agentic.py::AgenticManifestDataset._preprocess() ile aynı adımlar:
          1. BGR → RGB
          2. 224×224 INTER_AREA resize
          3. float32 / 255.0  (ImageNet normalizasyonu YOK — eğitimde de kullanılmadı)
          4. HWC → CHW
          5. Batch boyutu ekle [1, C, H, W]
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        return torch.from_numpy(img).unsqueeze(0).contiguous()  # [1,C,H,W]

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Tek bir BGR frame için inference yapar.

        Returns:
            {
                "action_id": int,
                "action_name": str,
                "phase_id": int,
                "phase_name": str,
                "action_confidence": float,   # 0.0 – 1.0
                "phase_confidence": float,    # 0.0 – 1.0
                "inference_ms": float,
            }
            veya None (model yüklü değilse / hata durumunda)
        """
        if not self._loaded or self._model is None or frame is None:
            return None

        t0 = time.perf_counter()

        with self._lock:
            try:
                tensor = (
                    self.preprocess_frame(frame)
                    .to(self._device, non_blocking=True)
                )

                with torch.no_grad():
                    phase_logits, action_logits = self._model(tensor)

                phase_probs = torch.softmax(phase_logits, dim=1)[0]
                action_probs = torch.softmax(action_logits, dim=1)[0]

                phase_id = int(torch.argmax(phase_probs).item())
                action_id = int(torch.argmax(action_probs).item())
                phase_conf = float(phase_probs[phase_id].item())
                action_conf = float(action_probs[action_id].item())

            except Exception as exc:
                self._log(f"[Alamet] Inference hatasi: {exc}", level="ERROR")
                return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_count += 1
        self._total_inference_ms += elapsed_ms

        action_name = self._id_to_action.get(action_id, f"action_{action_id}")
        phase_name = self._id_to_phase.get(phase_id, f"phase_{phase_id}")

        self._log(
            f"[Alamet #{self._inference_count}] "
            f"faz={phase_name}({phase_conf:.0%}) | "
            f"aksiyon={action_name}({action_conf:.0%}) | "
            f"{elapsed_ms:.1f}ms"
        )

        return {
            "action_id": action_id,
            "action_name": action_name,
            "phase_id": phase_id,
            "phase_name": phase_name,
            "action_confidence": action_conf,
            "phase_confidence": phase_conf,
            "inference_ms": elapsed_ms,
        }

    def predict_from_screen(self, vision_manager) -> Optional[Dict]:
        """VisionManager üzerinden tam ekran yakalar ve inference yapar."""
        if vision_manager is None:
            return None
        frame = vision_manager.capture_full_screen()
        if frame is None:
            return None
        return self.predict(frame)

    # ── VRAM Monitörü ─────────────────────────────────────────────────────

    def _start_vram_monitor(self):
        self._vram_thread = threading.Thread(
            target=self._vram_monitor_loop,
            daemon=True,
            name="VRAMMonitor-Alamet",
        )
        self._vram_thread.start()

    def _vram_monitor_loop(self):
        # İlk raporu hemen ver, sonra her 5 dakikada bir
        self._log_vram()
        while not self._stop_event.wait(timeout=_VRAM_LOG_INTERVAL_SEC):
            self._log_vram()

    def _log_vram(self):
        if not torch.cuda.is_available():
            return
        try:
            dev = self._device or torch.device("cuda")
            allocated_mb = torch.cuda.memory_allocated(dev) / 1024**2
            reserved_mb = torch.cuda.memory_reserved(dev) / 1024**2
            total_mb = torch.cuda.get_device_properties(dev).total_memory / 1024**2
            avg_ms = self._total_inference_ms / max(1, self._inference_count)
            self._log(
                f"[Alamet VRAM] "
                f"Kullanilan={allocated_mb:.0f}MB | "
                f"Ayrilmis={reserved_mb:.0f}MB | "
                f"Toplam={total_mb:.0f}MB | "
                f"Inference={self._inference_count} | "
                f"OrtLatency={avg_ms:.1f}ms"
            )
        except Exception:
            pass

    # ── İstatistik ve Yaşam Döngüsü ───────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_statistics(self) -> Dict:
        avg_ms = self._total_inference_ms / max(1, self._inference_count)
        vram_mb: float = 0.0
        if torch.cuda.is_available() and self._device:
            try:
                vram_mb = torch.cuda.memory_allocated(self._device) / 1024**2
            except Exception:
                pass
        return {
            "loaded": self._loaded,
            "device": str(self._device) if self._device else "none",
            "backbone": self._backbone,
            "num_actions": self._num_actions,
            "num_phases": self._num_phases,
            "inference_count": self._inference_count,
            "avg_inference_ms": round(avg_ms, 2),
            "vram_allocated_mb": round(vram_mb, 1),
            "model_path": str(self._model_path),
        }

    def shutdown(self):
        """Bot kapanırken çağrılır; VRAM monitor thread'ini durdurur."""
        self._stop_event.set()

    # ── Loglama ───────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        """LoABot logger (bot.log) veya standart logging.Logger ile uyumlu."""
        import logging as _logging

        if isinstance(self._logger, _logging.Logger):
            # standart Python logger: log(level_int, msg)
            level_int = getattr(_logging, level.upper(), _logging.INFO)
            self._logger.log(level_int, msg)
        elif hasattr(self._logger, "log"):
            # LoABot bot.log(msg, level="INFO")
            self._logger.log(msg, level=level)
        else:
            print(msg)


# ── Aksiyon Eşleme Yardımcısı ─────────────────────────────────────────────

# action_to_id.json'daki aksiyon isimlerini bot komutlarına map eder.
# Bu sözlük, tactical_brain.py tarafından kullanılır.
ACTION_COMMAND_MAP: Dict[str, Dict] = {
    "key_a":              {"type": "key",      "key": "a"},
    "key_q":              {"type": "key",      "key": "q"},
    "key_v":              {"type": "key",      "key": "v"},
    "key_z":              {"type": "key",      "key": "z"},
    "mouse_click":        {"type": "click"},
    "noop":               {"type": "noop"},
    "event_sequence":     {"type": "sequence", "name": "event"},
    "seq_boss_secimi":    {"type": "sequence", "name": "boss_secimi"},
    "seq_katman_secimi":  {"type": "sequence", "name": "katman_secimi"},
    "key_unknown":        {"type": "noop"},
    "unknown":            {"type": "noop"},
}
