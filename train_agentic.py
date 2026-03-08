"""
train_agentic.py — LoABot v5.9 Agentic YZ Eğitim Script'i (v2.0 — Temporal)
==============================================================================
Eski statik JPEG pipeline tamamen kaldırıldı.
Yeni video tabanlı pipeline (video_dataset_builder.py) kullanılır.

Mimari: Stacked 2D CNN
  - T-1, T, T+1 frame'leri kanal boyutunda birleştirilir → (B, 9, H, W)
  - ResNet18 veya EfficientNet-B0 backbone (ilk conv: 3ch → 9ch modifiye)
  - Regression Head: Sigmoid → (B, 2) normalize koordinat [0, 1]
  - Loss: MSELoss — sadece mouse_click event'lerinde hesaplanır

Çıktılar:
  - best.pt / last.pt checkpoint'ları
  - metrics.csv (epoch bazlı)
  - TensorBoard logları
  - summary.json

Kullanım:
  python train_agentic.py --video-root D:/LoABot_Training_Data/videos
  python train_agentic.py --video-root D:/LoABot_Training_Data/videos --backbone efficientnet_b0 --epochs 30
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchvision.models import (
        EfficientNet_B0_Weights,
        ResNet18_Weights,
        efficientnet_b0,
        resnet18,
    )
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

# ── Video dataset pipeline ────────────────────────────────────────────
from video_dataset_builder import build_dataloaders


# ══════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════

def create_summary_writer(log_dir: Path):
    """TensorBoard SummaryWriter oluşturma (import hatalarını yakalar)."""
    try:
        from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter
        return TBSummaryWriter(log_dir=str(log_dir)), True
    except Exception:
        return None, False


def seed_everything(seed: int = 42):
    """Tüm rastgelelik kaynaklarını sabitler (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: Dict):
    """CSV dosyasına tek satır ekler (header otomatik)."""
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def auto_num_workers() -> int:
    """CPU çekirdeğine göre optimal worker sayısı."""
    c = int(os.cpu_count() or 1)
    return max(2, min(16, c - 2))


def auto_batch_size(device: torch.device, backbone: str) -> int:
    """VRAM'e göre güvenli batch_size tahmini (9ch = ~3x bellek)."""
    if device.type != "cuda":
        return 8  # CPU: 9ch temporal daha ağır
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    b = str(backbone).lower()
    # 9 kanallı girdi normal 3ch'ye göre ~2-3x daha fazla bellek kullanır
    if "efficientnet" in b:
        if vram_gb >= 15.0:
            return 48
        if vram_gb >= 11.0:
            return 32
        return 16
    # resnet18 — daha hafif
    if vram_gb >= 15.0:
        return 64
    if vram_gb >= 11.0:
        return 48
    return 24


# ══════════════════════════════════════════════════════════════════════
#  TINY BACKBONE (torchvision yoksa fallback)
# ══════════════════════════════════════════════════════════════════════

class TinyBackbone(nn.Module):
    """Hafif CNN backbone — torchvision kurulu olmadığında fallback."""
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

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════════
#  TEMPORAL AGENTIC NET
# ══════════════════════════════════════════════════════════════════════

class TemporalAgenticNet(nn.Module):
    """
    Temporal Stacked 2D CNN — Koordinat Regresyon Modeli.

    Girdi : (B, 9, H, W)  — T-1, T, T+1 frame'leri kanal birleşik
    Çıktı : (B, 2)        — sigmoid normalize (x, y) koordinatları [0, 1]

    İlk conv katmanı 3ch → 9ch olarak genişletilir:
      - Pretrained ise: orijinal 3ch ağırlıkları 3 kez kopyalanır / 3'e bölünür
      - Scratch ise: Kaiming init
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim = self._build_backbone(backbone, pretrained)
        self.dropout = nn.Dropout(float(dropout))

        # Koordinat regresyon kafası: (B, feat_dim) → (B, 2)
        self.coord_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid(),  # çıktı [0, 1] aralığında — normalize koordinat
        )

    def _build_backbone(self, backbone: str, pretrained: bool):
        """Backbone oluştur, ilk conv'u 9 kanal yapacak şekilde modifiye et."""
        b = str(backbone).strip().lower()

        # ── ResNet18 ──────────────────────────────────────────────
        if HAS_TORCHVISION and b == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            feat_dim = model.fc.in_features

            # İlk conv: 3ch → 9ch genişlet
            old_conv = model.conv1  # Conv2d(3, 64, 7, stride=2, padding=3)
            model.conv1 = self._expand_first_conv(old_conv, 9, pretrained)

            # Classifier kafasını kaldır (backbone sadece feature extractor)
            model.fc = nn.Identity()
            return model, feat_dim

        # ── EfficientNet-B0 ──────────────────────────────────────
        if HAS_TORCHVISION and b == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = efficientnet_b0(weights=weights)
            feat_dim = model.classifier[1].in_features

            # İlk conv: features[0][0] → Conv2d(3, 32, 3, stride=2, padding=1)
            old_conv = model.features[0][0]
            model.features[0][0] = self._expand_first_conv(old_conv, 9, pretrained)

            # Classifier kafasını kaldır
            model.classifier = nn.Identity()
            return model, feat_dim

        # ── Tiny Fallback ────────────────────────────────────────
        tiny = TinyBackbone(in_channels=9, out_dim=256)
        return tiny, tiny.out_dim

    @staticmethod
    def _expand_first_conv(
        old_conv: nn.Conv2d,
        new_in_channels: int,
        copy_weights: bool,
    ) -> nn.Conv2d:
        """
        Mevcut Conv2d'yi new_in_channels kabul edecek şekilde genişletir.

        Pretrained ağırlık aktarımı:
          - Orijinal (out, 3, kH, kW) → (out, 9, kH, kW)
          - 3 kopya / 3.0'a bölerek ortalama korunur (Xavier benzeri)
        """
        new_conv = nn.Conv2d(
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

        if copy_weights and old_conv.weight is not None:
            with torch.no_grad():
                # Orijinal ağırlık: (out_ch, 3, kH, kW)
                old_w = old_conv.weight.data
                # 3 kez tekrarla → (out_ch, 9, kH, kW), ortalaması korunsun
                repeat_count = new_in_channels // old_w.shape[1]
                remainder = new_in_channels % old_w.shape[1]

                parts = [old_w] * repeat_count
                if remainder > 0:
                    parts.append(old_w[:, :remainder, :, :])

                new_w = torch.cat(parts, dim=1) / float(repeat_count + (1 if remainder else 0))
                new_conv.weight.data.copy_(new_w)

                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.data.copy_(old_conv.bias.data)
        else:
            # Kaiming init (scratch eğitim)
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        return new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, H, W) — birleştirilmiş temporal frame'ler
        Returns:
            coords: (B, 2) — sigmoid normalize [0, 1] koordinatlar
        """
        feat = self.backbone(x)
        feat = self.dropout(feat)
        coords = self.coord_head(feat)
        return coords


# ══════════════════════════════════════════════════════════════════════
#  METRİK HESAPLAMA
# ══════════════════════════════════════════════════════════════════════

def compute_mean_euclidean_distance(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Masked örnekler üzerinden ortalama Öklid mesafesi hesaplar.

    Args:
        pred_coords: (N, 2) model çıktısı [0, 1]
        target_coords: (N, 2) gerçek koordinatlar [0, 1]
        mask: (N,) bool — True olan indeksler dahil edilir

    Returns:
        Ortalama mesafe (float). Mask boşsa 0.0.
    """
    if not mask.any():
        return 0.0
    p = pred_coords[mask]
    t = target_coords[mask]
    # Öklid mesafesi: sqrt((x1-x2)^2 + (y1-y2)^2)
    dist = torch.sqrt(((p - t) ** 2).sum(dim=1) + 1e-8)
    return float(dist.mean().item())


# ══════════════════════════════════════════════════════════════════════
#  BATCH İŞLEME
# ══════════════════════════════════════════════════════════════════════

def prepare_batch(batch: Dict, device: torch.device):
    """
    DataLoader batch'ini eğitim döngüsü için hazırlar.

    Döndürür:
        stacked: (B, 9, H, W) — birleştirilmiş frame'ler
        coords:  (B, 2) — normalize koordinatlar
        click_mask: (B,) — sadece mouse_click event'leri True
    """
    f_before = batch["frame_before"]   # (B, 3, H, W)
    f_action = batch["frame_action"]   # (B, 3, H, W)
    f_after = batch["frame_after"]     # (B, 3, H, W)

    # Kanal boyutunda birleştir: (B, 9, H, W)
    stacked = torch.cat([f_before, f_action, f_after], dim=1)
    stacked = stacked.to(device, non_blocking=True)

    coords = batch["coords"].to(device, non_blocking=True)  # (B, 2)

    # Mouse click maskesi: sadece tıklama olaylarında koordinat loss'u hesapla
    event_types = batch["event_type"]  # list[str]
    click_mask = torch.tensor(
        [et == "mouse_click" for et in event_types],
        dtype=torch.bool,
        device=device,
    )

    return stacked, coords, click_mask


# ══════════════════════════════════════════════════════════════════════
#  EĞİTİM ADIMI
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    """
    Tek epoch eğitim.

    Returns:
        {
            "mse_loss": float,
            "mean_dist": float,  — Öklid mesafesi (sadece click örnekleri)
            "click_ratio": float — batch'lerdeki click oranı
        }
    """
    model.train()
    loss_sum = 0.0
    dist_sum = 0.0
    click_count = 0
    total_count = 0

    for batch in loader:
        stacked, coords_gt, click_mask = prepare_batch(batch, device)
        bs = stacked.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred_coords = model(stacked)  # (B, 2)

            # Loss: sadece mouse_click örnekleri üzerinden
            if click_mask.any():
                loss = loss_fn(pred_coords[click_mask], coords_gt[click_mask])
            else:
                # Batch'te hiç click yoksa → küçük dummy loss (gradient akışı korunsun)
                loss = loss_fn(pred_coords[:1], pred_coords[:1].detach()) * 0.0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrik biriktir
        n_clicks = int(click_mask.sum().item())
        total_count += bs
        click_count += n_clicks

        if click_mask.any():
            loss_sum += float(loss.item()) * n_clicks
            with torch.no_grad():
                dist_sum += compute_mean_euclidean_distance(
                    pred_coords, coords_gt, click_mask
                ) * n_clicks

    safe_clicks = max(1, click_count)
    return {
        "mse_loss": loss_sum / safe_clicks,
        "mean_dist": dist_sum / safe_clicks,
        "click_ratio": click_count / max(1, total_count),
    }


# ══════════════════════════════════════════════════════════════════════
#  VALİDASYON
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    """
    Validasyon/test değerlendirmesi.

    Returns:
        {
            "mse_loss": float,
            "mean_dist": float,
            "click_ratio": float,
        }
    """
    model.eval()
    loss_sum = 0.0
    dist_sum = 0.0
    click_count = 0
    total_count = 0

    for batch in loader:
        stacked, coords_gt, click_mask = prepare_batch(batch, device)
        bs = stacked.size(0)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred_coords = model(stacked)

            if click_mask.any():
                loss = loss_fn(pred_coords[click_mask], coords_gt[click_mask])
            else:
                loss = torch.tensor(0.0, device=device)

        n_clicks = int(click_mask.sum().item())
        total_count += bs
        click_count += n_clicks

        if click_mask.any():
            loss_sum += float(loss.item()) * n_clicks
            dist_sum += compute_mean_euclidean_distance(
                pred_coords, coords_gt, click_mask
            ) * n_clicks

    safe_clicks = max(1, click_count)
    return {
        "mse_loss": loss_sum / safe_clicks,
        "mean_dist": dist_sum / safe_clicks,
        "click_ratio": click_count / max(1, total_count),
    }


# ══════════════════════════════════════════════════════════════════════
#  ANA EĞİTİM FONKSİYONU
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoABot v5.9 Agentic Trainer — Temporal Video Pipeline"
    )

    # ── Veri ──────────────────────────────────────────────────────
    parser.add_argument(
        "--video-root",
        default=r"D:\LoABot_Training_Data\videos",
        help="VID_* oturum klasörlerinin kök dizini.",
    )
    parser.add_argument(
        "--log-root",
        default=r"D:\LoABot_Training_Data\runtime_data\training_logs",
        help="Eğitim çıktı dizini (checkpoint, log, TB).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Fine-tuning icin egitilmis .pt checkpoint yolu",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)

    # ── Model ─────────────────────────────────────────────────────
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "efficientnet_b0", "tiny"],
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)

    # ── Eğitim hiperparametreleri ─────────────────────────────────
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss-fn", default="mse", choices=["mse", "l1", "smooth_l1"],
        help="Koordinat regresyon loss fonksiyonu.",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Frame resize boyutu (H=W kare).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="0 → otomatik (VRAM'e göre).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=-1,
        help="-1 → otomatik (CPU çekirdeğine göre).",
    )
    parser.add_argument("--seed", type=int, default=42)

    # ── Early stopping ────────────────────────────────────────────
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--disable-early-stop", action="store_true")

    # ── Loglama ───────────────────────────────────────────────────
    parser.add_argument("--no-tensorboard", action="store_true")

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────
    #  SETUP
    # ──────────────────────────────────────────────────────────────

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    # CUDA tanılama
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda} | CUDA available: {cuda_ok}")
    if cuda_ok:
        _dev0 = torch.cuda.get_device_properties(0)
        print(f"GPU: {_dev0.name} | VRAM: {_dev0.total_memory / 1024**3:.1f} GB")
    else:
        print("UYARI: CUDA bulunamadi — egitim CPU uzerinde calisacak (cok yavas).")
        print("  Cozum: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124")

    device = torch.device("cuda" if cuda_ok else "cpu")

    # Worker ve batch boyutu
    if args.num_workers < 0:
        num_workers = auto_num_workers()
    else:
        num_workers = max(0, int(args.num_workers))

    if args.batch_size <= 0:
        batch_size = auto_batch_size(device, args.backbone)
    else:
        batch_size = int(args.batch_size)

    # CPU koruması
    if device.type != "cuda":
        _cpu_max_batch = 8
        if batch_size > _cpu_max_batch:
            print(f"UYARI: CPU modunda batch_size={batch_size} → {_cpu_max_batch} olarak düşürüldü.")
            batch_size = _cpu_max_batch
        _cpu_max_workers = max(2, (os.cpu_count() or 4) // 2)
        if num_workers > _cpu_max_workers:
            print(f"UYARI: CPU modunda num_workers → {_cpu_max_workers} olarak ayarlandi.")
            num_workers = _cpu_max_workers

    # ──────────────────────────────────────────────────────────────
    #  DATA LOADER (video_dataset_builder entegrasyonu)
    # ──────────────────────────────────────────────────────────────

    image_size = int(args.image_size)
    resize = (image_size, image_size)

    print(f"\nDataset taranıyor: {args.video_root}")
    print(f"  resize={resize} | batch_size={batch_size} | num_workers={num_workers}")

    train_loader, test_loader = build_dataloaders(
        video_root=args.video_root,
        train_ratio=args.train_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        resize=resize,
        seed=args.seed,
    )

    # Boş dataset kontrolü
    train_len = len(train_loader.dataset)
    test_len = len(test_loader.dataset)
    if train_len == 0:
        raise SystemExit("HATA: Train dataset bos. Video dosyalarini kontrol edin.")
    print(f"  Train: {train_len} örnek | Test: {test_len} örnek\n")

    # ──────────────────────────────────────────────────────────────
    #  MODEL
    # ──────────────────────────────────────────────────────────────

    model = TemporalAgenticNet(
        backbone=args.backbone,
        pretrained=(not args.no_pretrained),
        dropout=args.dropout,
    ).to(device)
    if args.weights and os.path.isfile(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Pre-trained agirliklar yuklendi: {args.weights}")

    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: TemporalAgenticNet ({args.backbone})")
    print(f"  Toplam parametre : {total_params:,}")
    print(f"  Eğitilebilir     : {trainable_params:,}")

    # ──────────────────────────────────────────────────────────────
    #  LOSS & OPTİMİZER
    # ──────────────────────────────────────────────────────────────

    # Loss fonksiyonu seçimi
    loss_name = str(args.loss_fn).lower()
    if loss_name == "l1":
        loss_fn = nn.L1Loss()
    elif loss_name == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()
    print(f"  Loss fonksiyonu  : {loss_fn.__class__.__name__}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"

    # ──────────────────────────────────────────────────────────────
    #  ÇIKTI DİZİNİ
    # ──────────────────────────────────────────────────────────────

    run_root = Path(args.log_root)
    run_dir = run_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Eğitim konfigürasyonu kaydet
    train_config = {
        "video_root": str(args.video_root),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "epochs": args.epochs,
        "backbone": args.backbone,
        "pretrained": not args.no_pretrained,
        "dropout": args.dropout,
        "loss_fn": loss_fn.__class__.__name__,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_enabled": not args.disable_early_stop,
        "tensorboard_enabled": not args.no_tensorboard,
        "train_samples": train_len,
        "test_samples": test_len,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_channels": 9,
        "architecture": "stacked_2d_cnn_temporal",
    }
    (run_dir / "train_config.json").write_text(
        json.dumps(train_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = run_dir / "metrics.csv"

    # TensorBoard
    tb_dir = run_dir / "tensorboard"
    tb_writer = None
    if not args.no_tensorboard:
        tb_writer, tb_ok = create_summary_writer(tb_dir)
        if not tb_ok:
            print("UYARI: TensorBoard import edilemedi.")

    # ──────────────────────────────────────────────────────────────
    #  EARLY STOPPING STATE
    # ──────────────────────────────────────────────────────────────

    best_metric = float("inf")   # loss minimize edilir → küçük = iyi
    best_epoch = 0
    early_stopped = False
    stopped_epoch = args.epochs
    early_stop_enabled = not args.disable_early_stop
    patience = max(1, int(args.early_stop_patience))
    min_delta = max(0.0, float(args.early_stop_min_delta))
    no_improve_epochs = 0

    print(f"\nDevice: {device} | batch_size={batch_size} | num_workers={num_workers}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Early stop: {early_stop_enabled} (patience={patience})")
    print(f"Run dir: {run_dir}\n")
    print("=" * 100)

    # ──────────────────────────────────────────────────────────────
    #  EĞİTİM DÖNGÜSÜ
    # ──────────────────────────────────────────────────────────────

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────
        val_metrics = evaluate(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
        )

        epoch_time = time.time() - t0

        # ── Metrik kaydı ──────────────────────────────────────────
        row = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "train_mse_loss": round(train_metrics["mse_loss"], 6),
            "train_mean_dist": round(train_metrics["mean_dist"], 6),
            "train_click_ratio": round(train_metrics["click_ratio"], 4),
            "val_mse_loss": round(val_metrics["mse_loss"], 6),
            "val_mean_dist": round(val_metrics["mean_dist"], 6),
            "val_click_ratio": round(val_metrics["click_ratio"], 4),
            "epoch_sec": round(epoch_time, 2),
        }
        append_csv(csv_path, row)

        # ── TensorBoard ──────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train_mse", train_metrics["mse_loss"], epoch)
            tb_writer.add_scalar("Loss/val_mse", val_metrics["mse_loss"], epoch)
            tb_writer.add_scalar("Distance/train_mean", train_metrics["mean_dist"], epoch)
            tb_writer.add_scalar("Distance/val_mean", val_metrics["mean_dist"], epoch)
            tb_writer.add_scalar("Data/train_click_ratio", train_metrics["click_ratio"], epoch)
            tb_writer.add_scalar("Data/val_click_ratio", val_metrics["click_ratio"], epoch)
            tb_writer.add_scalar("LR/current", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        # ── Checkpoint kaydet ─────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "backbone": args.backbone,
            "input_channels": 9,
            "args": vars(args),
            "metrics": row,
        }
        torch.save(ckpt, run_dir / "last.pt")

        # Best model: val_mse_loss minimize
        val_loss = val_metrics["mse_loss"]
        improved = val_loss < (best_metric - min_delta)
        if improved:
            best_metric = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(ckpt, run_dir / "best.pt")
            marker = " ★ best"
        else:
            no_improve_epochs += 1
            marker = ""

        # ── Konsol çıktısı ────────────────────────────────────────
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={row['train_mse_loss']:.5f} "
            f"val_loss={row['val_mse_loss']:.5f} "
            f"val_dist={row['val_mean_dist']:.4f} "
            f"click%={row['val_click_ratio']:.2f} "
            f"lr={row['lr']:.2e} "
            f"time={row['epoch_sec']:.1f}s"
            f"{marker}"
        )

        # ── Early stopping kontrolü ──────────────────────────────
        if early_stop_enabled and no_improve_epochs >= patience:
            early_stopped = True
            stopped_epoch = epoch
            print(
                f"\nEarly stopping: {no_improve_epochs} epoch boyunca iyilesme yok "
                f"(patience={patience}, min_delta={min_delta})."
            )
            break

    # ──────────────────────────────────────────────────────────────
    #  SONUÇ ÖZETİ
    # ──────────────────────────────────────────────────────────────

    print("=" * 100)

    summary = {
        "best_val_mse_loss": round(best_metric, 6),
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "total_epochs_run": min(stopped_epoch, args.epochs),
        "run_dir": str(run_dir),
        "best_ckpt": str(run_dir / "best.pt"),
        "last_ckpt": str(run_dir / "last.pt"),
        "metrics_csv": str(csv_path),
        "tensorboard_enabled": tb_writer is not None,
        "tensorboard_dir": str(tb_dir) if tb_writer is not None else "",
    }

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nEgitim tamamlandi.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
