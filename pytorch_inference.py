# -*- coding: utf-8 -*-
"""
pytorch_inference.py — Temporal 9-Kanal Inference Engine (v2.0)
================================================================
TemporalAgenticNet (ResNet18/EfficientNet-B0, 9ch girdi) modelini
CUDA üzerinde yükler ve kayan pencere (ring buffer) üzerinden
koordinat regresyon tahmini yapar.

Mimari Değişiklik (v1 → v2):
  v1: Tek frame (3ch) → phase_head + action_head (sınıflandırma)
  v2: 3 frame [T-2, T-1, T] → 9ch stacked → coord_head (regresyon)

Frame Buffer Mantığı:
  Eğitimde [T-1, T, T+1] kullanıldı; canlıda geleceği göremeyiz.
  Bu yüzden [T-2, T-1, T] kayan pencere kullanılır.
  Buffer 3 frame dolmadan tahmin yapılmaz → None (Tier-1 fallback).

Çıktı:
  predict() → {"x": int, "y": int, "norm_x": float, "norm_y": float,
                "inference_ms": float, "buffer_size": int}
  veya None (buffer eksik / model yüklü değil / hata)

Entegrasyon:
  # VisionManager veya capture thread'inden:
  engine.update_buffer(frame_bgr)

  # Karar anında:
  result = engine.predict()
  if result is None:
      # Tier-1 kural tabanlı sisteme dön
      ...
"""

from __future__ import annotations

import collections
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

# ── torchvision isteğe bağlı ──────────────────────────────────────────
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

# ── Sabitler ────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent / "models" / "active_model"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "best.pt"
_IMAGE_SIZE = 224
_BUFFER_SIZE = 3               # T-2, T-1, T (kayan pencere)
_VRAM_LOG_INTERVAL_SEC = 300   # 5 dakika


# ══════════════════════════════════════════════════════════════════════════
#  Model Mimarisi (train_agentic.py ile birebir aynı — state_dict uyumluluğu)
# ══════════════════════════════════════════════════════════════════════════

class _TinyBackbone(nn.Module):
    """Torchvision yüklü değilse kullanılan küçük yedek backbone."""

    def __init__(self, in_channels: int = 9, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
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


class _TemporalAgenticNet(nn.Module):
    """
    Temporal Stacked 2D CNN — Koordinat Regresyon Modeli.

    Girdi : (B, 9, H, W)  — 3 frame kanal birleşik
    Çıktı : (B, 2)        — sigmoid normalize (x, y) [0, 1]

    train_agentic.py::TemporalAgenticNet ile birebir aynı mimari.
    state_dict uyumluluğu: aynı katman isimleri, aynı boyutlar.
    """

    def __init__(self, backbone: str = "resnet18", dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        # Inference: pretrained ağırlıklar gereksiz (checkpoint'ten yüklenir)
        self.backbone, feat_dim = self._build_backbone(backbone)
        self.dropout = nn.Dropout(float(dropout))
        self.coord_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def _build_backbone(self, backbone: str):
        b = str(backbone).strip().lower()

        if _HAS_TORCHVISION and b == "resnet18":
            model = resnet18(weights=None)
            feat_dim = model.fc.in_features
            # İlk conv: 3ch → 9ch genişlet (ağırlıklar checkpoint'ten gelecek)
            old_conv = model.conv1
            model.conv1 = self._expand_first_conv(old_conv, 9)
            model.fc = nn.Identity()
            return model, feat_dim

        if _HAS_TORCHVISION and b == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
            feat_dim = model.classifier[1].in_features
            old_conv = model.features[0][0]
            model.features[0][0] = self._expand_first_conv(old_conv, 9)
            model.classifier = nn.Identity()
            return model, feat_dim

        tiny = _TinyBackbone(in_channels=9, out_dim=256)
        return tiny, tiny.out_dim

    @staticmethod
    def _expand_first_conv(old_conv: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
        """Conv2d'yi 9ch kabul edecek şekilde genişletir (ağırlıklar checkpoint'ten yüklenir)."""
        return nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.coord_head(feat)  # (B, 2) — [0, 1] sigmoid


# ══════════════════════════════════════════════════════════════════════════
#  Ana Inference Engine
# ══════════════════════════════════════════════════════════════════════════

class PyTorchInferenceEngine:
    """
    Temporal 9-kanal inference engine — Ring Buffer + Koordinat Regresyon.

    Kullanım:
        engine = PyTorchInferenceEngine(bot=bot)

        # Her yeni ekran karesinde (VisionManager/capture loop):
        engine.update_buffer(frame_bgr)

        # Karar anında:
        result = engine.predict()
        if result is not None:
            px, py = result["x"], result["y"]
        else:
            # Tier-1 fallback
    """

    def __init__(
        self,
        bot=None,
        model_path: Optional[Path] = None,
        logger: Any = None,
    ):
        self._bot = bot
        self._logger = logger or (bot if bot else logging.getLogger(__name__))
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._device: Optional[torch.device] = None
        self._model: Optional[_TemporalAgenticNet] = None
        self._backbone: str = "resnet18"
        self._loaded: bool = False
        self._inference_lock = threading.Lock()

        # ── Ekran çözünürlüğü (koordinat dönüşümü) ───────────────
        self._screen_width: int = 2560
        self._screen_height: int = 1440
        if bot is not None:
            self._screen_width = int(getattr(bot, "settings", {}).get("SCREEN_WIDTH", 2560))
            self._screen_height = int(getattr(bot, "settings", {}).get("SCREEN_HEIGHT", 1440))

        # ── Ring Buffer (thread-safe kayan pencere) ───────────────
        # Son 3 frame: [T-2, T-1, T] — her biri (3, H, W) CHW tensor
        self._frame_buffer: collections.deque = collections.deque(maxlen=_BUFFER_SIZE)
        self._buffer_lock = threading.Lock()

        # ── İstatistik ────────────────────────────────────────────
        self._inference_count: int = 0
        self._total_inference_ms: float = 0.0
        self._buffer_miss_count: int = 0  # buffer eksikliğinden kaçırılan tahminler

        # ── VRAM monitor ──────────────────────────────────────────
        self._stop_event = threading.Event()
        self._vram_thread: Optional[threading.Thread] = None

        self._load_model()
        if self._loaded:
            self._start_vram_monitor()

    # ══════════════════════════════════════════════════════════════════
    #  MODEL YÜKLEME
    # ══════════════════════════════════════════════════════════════════

    def _require_cuda(self) -> torch.device:
        """CUDA yoksa açıklayıcı hata — CPU'ya sessizce düşmez."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[Temporal] CUDA bulunamadi!\n"
                "  Kontrol: python -c \"import torch; print(torch.cuda.is_available())\"\n"
                "  Cozum  : pip install torch torchvision "
                "--extra-index-url https://download.pytorch.org/whl/cu124"
            )
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        self._log(
            f"[Temporal] GPU: {props.name} | "
            f"VRAM: {props.total_memory / 1024**3:.1f} GB | "
            f"PyTorch: {torch.__version__}"
        )
        return device

    def _load_model(self):
        """best.pt checkpoint'ını yükler, modeli CUDA'ya taşır ve eval moduna alır."""

        # 1. Dosya varlık kontrolü
        if not self._model_path.exists():
            self._log(
                f"[Temporal] HATA: Model bulunamadi: {self._model_path.absolute()}\n"
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
        size_mb = self._model_path.stat().st_size / 1024**2
        self._log(f"[Temporal] Model yukleniyor: {self._model_path} ({size_mb:.1f} MB) ...")
        try:
            ckpt = torch.load(
                str(self._model_path),
                map_location=self._device,
                weights_only=False,
            )
        except Exception as exc:
            self._log(f"[Temporal] torch.load hatasi: {exc}", level="ERROR")
            return

        # 4. Checkpoint'ten konfigürasyon
        args = ckpt.get("args", {})
        self._backbone = str(args.get("backbone", "resnet18"))
        input_channels = int(ckpt.get("input_channels", args.get("input_channels", 9)))

        if input_channels != 9:
            self._log(
                f"[Temporal] UYARI: Checkpoint input_channels={input_channels}, "
                f"beklenen=9. Eski (3ch) model kullanılamaz.",
                level="ERROR",
            )
            return

        # 5. Modeli oluştur ve ağırlıkları yükle
        dropout = float(args.get("dropout", 0.3))
        try:
            self._model = _TemporalAgenticNet(
                backbone=self._backbone,
                dropout=dropout,
            ).to(self._device)

            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()
            torch.backends.cudnn.benchmark = True

        except Exception as exc:
            self._log(f"[Temporal] state_dict yukleme hatasi: {exc}", level="ERROR")
            self._model = None
            return

        self._loaded = True
        epoch = ckpt.get("epoch", "?")
        m = ckpt.get("metrics", {})
        self._log(
            f"[Temporal] Model hazir! "
            f"epoch={epoch} | "
            f"val_mse={m.get('val_mse_loss', '?')} | "
            f"val_dist={m.get('val_mean_dist', '?')} | "
            f"backbone={self._backbone} | "
            f"device={self._device}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  FRAME BUFFER (Ring Buffer)
    # ══════════════════════════════════════════════════════════════════

    def update_buffer(self, frame_bgr: np.ndarray) -> None:
        """
        VisionManager veya capture thread'inden gelen BGR frame'i
        ön-işleyip ring buffer'a ekler.

        Bu metot HER YENI FRAME'de çağrılmalıdır (tipik: 10-30 FPS).
        Sadece preprocessed tensörü (3, H, W) tutar — orijinal frame RAM'de kalmaz.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR ekran karesi
        """
        if frame_bgr is None:
            return

        try:
            tensor = self._preprocess_single_frame(frame_bgr)
            with self._buffer_lock:
                self._frame_buffer.append(tensor)
        except Exception as exc:
            self._log(f"[Temporal] Buffer guncelleme hatasi: {exc}", level="WARNING")

    def clear_buffer(self) -> None:
        """Buffer'ı temizler (sahne değişimi, menü geçişi sonrası)."""
        with self._buffer_lock:
            self._frame_buffer.clear()

    @property
    def buffer_ready(self) -> bool:
        """Buffer 3 frame ile dolu mu?"""
        with self._buffer_lock:
            return len(self._frame_buffer) >= _BUFFER_SIZE

    # ══════════════════════════════════════════════════════════════════
    #  PREPROCESSING
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _preprocess_single_frame(
        frame_bgr: np.ndarray,
        image_size: int = _IMAGE_SIZE,
    ) -> torch.Tensor:
        """
        Tek BGR frame → (3, H, W) float32 tensor [0, 1].

        train_agentic.py ile birebir aynı pipeline:
          1. BGR → RGB
          2. 224×224 INTER_AREA resize
          3. float32 / 255.0
          4. HWC → CHW
        """
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return torch.from_numpy(img).contiguous()  # (3, H, W)

    # ══════════════════════════════════════════════════════════════════
    #  INFERENCE
    # ══════════════════════════════════════════════════════════════════

    def predict(self) -> Optional[Dict]:
        """
        Ring buffer'daki 3 frame'den 9-kanallı tensör oluşturur,
        modelden normalize koordinat alır ve piksel değerine çevirir.

        Returns:
            {
                "x": int,               # piksel X [0, SCREEN_WIDTH]
                "y": int,               # piksel Y [0, SCREEN_HEIGHT]
                "norm_x": float,        # normalize X [0.0, 1.0]
                "norm_y": float,        # normalize Y [0.0, 1.0]
                "inference_ms": float,  # çıkarım süresi
                "buffer_size": int,     # mevcut buffer doluluğu
            }
            veya None (model yüklü değil / buffer eksik / hata)
        """
        if not self._loaded or self._model is None:
            return None

        # ── Buffer kontrolü ───────────────────────────────────────
        with self._buffer_lock:
            buf_len = len(self._frame_buffer)
            if buf_len < _BUFFER_SIZE:
                self._buffer_miss_count += 1
                return None  # Tier-1 fallback
            # Snapshot: deque'yu kopyala (lock altında hızlı)
            frames = list(self._frame_buffer)  # [T-2, T-1, T] — her biri (3,H,W)

        t0 = time.perf_counter()

        with self._inference_lock:
            try:
                # ── 9-kanal tensör oluştur ────────────────────────
                # (3,H,W) * 3 → cat dim=0 → (9,H,W) → unsqueeze → (1,9,H,W)
                stacked = torch.cat(frames, dim=0).unsqueeze(0)  # (1, 9, H, W)
                stacked = stacked.to(self._device, non_blocking=True)

                # ── Model çıkarımı ───────────────────────────────
                with torch.no_grad():
                    pred_coords = self._model(stacked)  # (1, 2) — sigmoid [0, 1]

                # ── Normalize → piksel dönüşümü ──────────────────
                norm_x = float(pred_coords[0, 0].item())
                norm_y = float(pred_coords[0, 1].item())

                # Clamp güvenliği (sigmoid zaten [0,1] ama float hassasiyet)
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                pixel_x = int(round(norm_x * self._screen_width))
                pixel_y = int(round(norm_y * self._screen_height))

                # Ekran sınırları içinde tut
                pixel_x = max(0, min(self._screen_width - 1, pixel_x))
                pixel_y = max(0, min(self._screen_height - 1, pixel_y))

            except Exception as exc:
                self._log(f"[Temporal] Inference hatasi: {exc}", level="ERROR")
                return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_count += 1
        self._total_inference_ms += elapsed_ms

        self._log(
            f"[Temporal #{self._inference_count}] "
            f"coord=({pixel_x},{pixel_y}) "
            f"norm=({norm_x:.3f},{norm_y:.3f}) | "
            f"{elapsed_ms:.1f}ms",
            level="DEBUG",
        )

        return {
            "x": pixel_x,
            "y": pixel_y,
            "norm_x": round(norm_x, 5),
            "norm_y": round(norm_y, 5),
            "inference_ms": round(elapsed_ms, 2),
            "buffer_size": buf_len,
        }

    def predict_from_screen(self, vision_manager) -> Optional[Dict]:
        """
        VisionManager'dan tam ekran yakalar, buffer'a ekler ve tahmin yapar.
        Uyumluluk katmanı — tercih edilen yol: ayrı update_buffer() + predict().
        """
        if vision_manager is None:
            return None
        frame = vision_manager.capture_full_screen()
        if frame is None:
            return None
        self.update_buffer(frame)
        return self.predict()

    # ══════════════════════════════════════════════════════════════════
    #  VRAM MONİTÖRÜ
    # ══════════════════════════════════════════════════════════════════

    def _start_vram_monitor(self):
        self._vram_thread = threading.Thread(
            target=self._vram_monitor_loop,
            daemon=True,
            name="VRAMMonitor-Temporal",
        )
        self._vram_thread.start()

    def _vram_monitor_loop(self):
        self._log_vram()
        while not self._stop_event.wait(timeout=_VRAM_LOG_INTERVAL_SEC):
            self._log_vram()

    def _log_vram(self):
        if not torch.cuda.is_available():
            return
        try:
            dev = self._device or torch.device("cuda")
            alloc_mb = torch.cuda.memory_allocated(dev) / 1024**2
            reserved_mb = torch.cuda.memory_reserved(dev) / 1024**2
            total_mb = torch.cuda.get_device_properties(dev).total_memory / 1024**2
            avg_ms = self._total_inference_ms / max(1, self._inference_count)
            self._log(
                f"[Temporal VRAM] "
                f"Kullanilan={alloc_mb:.0f}MB | "
                f"Ayrilmis={reserved_mb:.0f}MB | "
                f"Toplam={total_mb:.0f}MB | "
                f"Inference={self._inference_count} | "
                f"BufferMiss={self._buffer_miss_count} | "
                f"OrtLatency={avg_ms:.1f}ms"
            )
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    #  İSTATİSTİK ve YAŞAM DÖNGÜSÜ
    # ══════════════════════════════════════════════════════════════════

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
        with self._buffer_lock:
            buf_size = len(self._frame_buffer)
        return {
            "loaded": self._loaded,
            "device": str(self._device) if self._device else "none",
            "backbone": self._backbone,
            "input_channels": 9,
            "architecture": "temporal_stacked_2d_cnn",
            "screen_resolution": f"{self._screen_width}x{self._screen_height}",
            "inference_count": self._inference_count,
            "buffer_miss_count": self._buffer_miss_count,
            "buffer_size": buf_size,
            "buffer_ready": buf_size >= _BUFFER_SIZE,
            "avg_inference_ms": round(avg_ms, 2),
            "vram_allocated_mb": round(vram_mb, 1),
            "model_path": str(self._model_path),
        }

    def shutdown(self):
        """Bot kapanırken çağrılır; VRAM monitor'u durdurur, buffer'ı temizler."""
        self._stop_event.set()
        self.clear_buffer()
        self._log("[Temporal] Inference engine kapatildi.")

    # ══════════════════════════════════════════════════════════════════
    #  LOGLAMA
    # ══════════════════════════════════════════════════════════════════

    def _log(self, msg: str, level: str = "INFO"):
        import logging as _logging
        if isinstance(self._logger, _logging.Logger):
            level_int = getattr(_logging, level.upper(), _logging.INFO)
            self._logger.log(level_int, msg)
        elif hasattr(self._logger, "log"):
            self._logger.log(msg, level=level)
        else:
            print(msg)


# ── Aksiyon Eşleme Yardımcısı ───────────────────────────────────────────
# tactical_brain.py tarafından kullanılır.
ACTION_COMMAND_MAP: Dict[str, Dict] = {
    "key_a":              {"type": "key",      "key": "a"},
    "key_q":              {"type": "key",      "key": "q"},
    "key_e":              {"type": "key",      "key": "e"},
    "key_r":              {"type": "key",      "key": "r"},
    "key_w":              {"type": "key",      "key": "w"},
    "key_space":          {"type": "key",      "key": "space"},
    "key_z":              {"type": "key",      "key": "z"},
    "key_1":              {"type": "key",      "key": "1"},
    "key_2":              {"type": "key",      "key": "2"},
    "key_3":              {"type": "key",      "key": "3"},
    "key_4":              {"type": "key",      "key": "4"},
    "mouse_click":        {"type": "click"},
    "noop":               {"type": "noop"},
    "event_sequence":     {"type": "sequence", "name": "event"},
    "seq_boss_secimi":    {"type": "sequence", "name": "boss_secimi"},
    "seq_katman_secimi":  {"type": "sequence", "name": "katman_secimi"},
    "key_unknown":        {"type": "noop"},
    "unknown":            {"type": "noop"},
}
