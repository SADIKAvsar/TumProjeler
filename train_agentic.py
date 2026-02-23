import argparse
import bisect
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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


def create_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter

        return TBSummaryWriter(log_dir=str(log_dir)), True
    except Exception:
        return None, False


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def parse_image_ts_from_name(image_path: str) -> Optional[float]:
    """
    Dosya adindan timestamp parse eder:
    20260218_063205_265252_...
    """
    try:
        stem = Path(image_path).stem
        parts = stem.split("_")
        if len(parts) < 3:
            return None
        dt_part = "_".join(parts[0:3])
        dt = datetime.strptime(dt_part, "%Y%m%d_%H%M%S_%f")
        return dt.timestamp()
    except Exception:
        return None


def normalize_action_label(name: str, payload: Dict) -> str:
    name = str(name or "").strip().lower()
    payload = payload or {}
    if name == "press_key":
        key = str(payload.get("key", "")).strip().lower()
        if key:
            return f"key_{key}"
        return "key_unknown"
    if name == "click":
        return "mouse_click"
    if name == "sequence_step":
        action = str(payload.get("action", "")).strip().lower()
        if action == "press_key":
            key = str(payload.get("key", "")).strip().lower()
            if key:
                return f"key_{key}"
            return "key_unknown"
        if action == "click":
            return "mouse_click"
        return f"seq_{action or 'unknown'}"
    return f"act_{name or 'unknown'}"


def infer_action_label_from_stage(stage: str, phase_label: str) -> str:
    s = str(stage or "").strip().lower()
    p = str(phase_label or "").strip().upper()

    if "attack_start" in s:
        return "key_a"
    if "anchor" in s or "spawn" in s or "area" in s or "victory" in s:
        return "mouse_click"
    if "transition" in s:
        return "mouse_click"
    if "loot" in s or "wait" in s or "mission_loop" in s:
        return "noop"
    if "event_action" in s:
        return "event_sequence"
    if "event_start" in s:
        return "mouse_click"
    if "event_wait" in s:
        return "noop"

    if p == "COMBAT_PHASE":
        return "key_a"
    if p == "NAV_PHASE":
        return "mouse_click"
    if p == "LOOT_PHASE":
        return "noop"
    if p == "EVENT_PHASE":
        return "event_sequence"
    return "unknown"


def load_action_events(action_log_root: Path) -> Tuple[List[float], List[str]]:
    if not action_log_root.exists():
        return [], []

    action_times = []
    action_labels = []

    for day_dir in action_log_root.glob("*"):
        if not day_dir.is_dir():
            continue
        for file in day_dir.glob("*.jsonl"):
            rows = read_jsonl(file)
            for row in rows:
                if str(row.get("event_type")) != "action":
                    continue
                ts = row.get("ts_unix")
                if not isinstance(ts, (int, float)):
                    continue
                payload = row.get("payload") or {}
                action_name = payload.get("name", "")
                label = normalize_action_label(action_name, payload)
                action_times.append(float(ts))
                action_labels.append(label)

    if not action_times:
        return [], []

    zipped = sorted(zip(action_times, action_labels), key=lambda x: x[0])
    return [x[0] for x in zipped], [x[1] for x in zipped]


def nearest_action_label(
    image_ts: Optional[float],
    action_times: List[float],
    action_labels: List[str],
    window_sec: float = 0.75,
) -> Optional[str]:
    if image_ts is None or not action_times:
        return None

    idx = bisect.bisect_left(action_times, image_ts)
    candidates = []
    if idx < len(action_times):
        candidates.append(idx)
    if idx - 1 >= 0:
        candidates.append(idx - 1)

    best = None
    best_dist = None
    for ci in candidates:
        dist = abs(action_times[ci] - image_ts)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = ci

    if best is None or best_dist is None or best_dist > window_sec:
        return None
    return action_labels[best]


class AgenticManifestDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        image_size: int = 224,
        train: bool = True,
        action_to_id: Optional[Dict[str, int]] = None,
        action_times: Optional[List[float]] = None,
        action_labels: Optional[List[str]] = None,
        action_window_sec: float = 0.75,
    ):
        self.rows = rows
        self.image_size = int(image_size)
        self.train = bool(train)
        self.action_to_id = action_to_id or {}
        self.action_times = action_times or []
        self.action_labels = action_labels or []
        self.action_window_sec = float(action_window_sec)

    def __len__(self):
        return len(self.rows)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not self.train:
            return img
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.2:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = self._augment(img)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).contiguous()

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img_path = row["image_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Goruntu bulunamadi: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self._preprocess(img)

        phase_id = int(row.get("phase_id", -1))
        stage = str(row.get("stage", ""))
        phase_label = str(row.get("phase_label", "UNKNOWN_PHASE"))

        action_label = row.get("action_label")
        if not action_label:
            image_ts = parse_image_ts_from_name(img_path)
            action_label = nearest_action_label(
                image_ts=image_ts,
                action_times=self.action_times,
                action_labels=self.action_labels,
                window_sec=self.action_window_sec,
            )
        if not action_label:
            action_label = infer_action_label_from_stage(stage=stage, phase_label=phase_label)

        action_id = int(self.action_to_id.get(action_label, self.action_to_id.get("unknown", -100)))
        return x, torch.tensor(phase_id, dtype=torch.long), torch.tensor(action_id, dtype=torch.long)


class TinyBackbone(nn.Module):
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

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


class AgenticNet(nn.Module):
    def __init__(
        self,
        num_phases: int,
        num_actions: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        with_click_head: bool = False,   # True → koordinat regresyon kafası ekle
    ):
        super().__init__()
        self.backbone_name  = backbone
        self.with_click_head = with_click_head
        self.backbone, feat_dim = self._build_backbone(backbone, pretrained)
        self.dropout     = nn.Dropout(float(dropout))
        self.phase_head  = nn.Linear(feat_dim, int(num_phases))
        self.action_head = nn.Linear(feat_dim, int(num_actions))
        if with_click_head:
            # (norm_x, norm_y) ∈ [0,1]  → ekran çözünürlüğünden bağımsız
            self.click_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Sigmoid(),   # 0–1 arası normalize koordinat
            )

    def _build_backbone(self, backbone: str, pretrained: bool):
        b = str(backbone).strip().lower()
        if HAS_TORCHVISION and b == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_features
        if HAS_TORCHVISION and b == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, in_features
        tiny = TinyBackbone(out_dim=256)
        return tiny, tiny.out_dim

    def forward(self, x):
        feat         = self.backbone(x)
        feat         = self.dropout(feat)
        phase_logits = self.phase_head(feat)
        action_logits = self.action_head(feat)
        if self.with_click_head:
            click_coords = self.click_head(feat)   # shape: [B, 2]  (x_norm, y_norm)
            return phase_logits, action_logits, click_coords
        return phase_logits, action_logits


def accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    if target.numel() == 0:
        return 0.0
    if ignore_index >= -1:
        valid = target != ignore_index
        if valid.sum().item() == 0:
            return 0.0
        logits = logits[valid]
        target = target[valid]
    pred = torch.argmax(logits, dim=1)
    return float((pred == target).float().mean().item())


def build_weights(rows: List[Dict], key: str, n_classes: int) -> torch.Tensor:
    counts = Counter(int(r.get(key, -1)) for r in rows if int(r.get(key, -1)) >= 0)
    if not counts:
        return torch.ones(n_classes, dtype=torch.float32)
    arr = np.ones(n_classes, dtype=np.float32)
    for i in range(n_classes):
        c = counts.get(i, 1)
        arr[i] = 1.0 / float(c)
    arr = arr / max(arr.mean(), 1e-6)
    return torch.tensor(arr, dtype=torch.float32)


def auto_num_workers() -> int:
    c = int(os.cpu_count() or 1)
    return max(2, min(16, c - 2))


def auto_batch_size(device: torch.device, backbone: str) -> int:
    if device.type != "cuda":
        return 16
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    b = str(backbone).lower()
    if "efficientnet" in b:
        if vram_gb >= 15.0:
            return 96
        if vram_gb >= 11.0:
            return 64
        return 32
    # resnet18
    if vram_gb >= 15.0:
        return 128
    if vram_gb >= 11.0:
        return 96
    return 48


def ensure_action_vocab(rows_train: List[Dict], rows_test: List[Dict]) -> Dict[str, int]:
    labels = set()
    for r in rows_train + rows_test:
        if r.get("action_label"):
            labels.add(str(r["action_label"]))
    if "unknown" not in labels:
        labels.add("unknown")
    ordered = sorted(labels)
    return {name: idx for idx, name in enumerate(ordered)}


def append_csv(path: Path, row: Dict):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, loader, device, phase_loss_fn, action_loss_fn, amp_enabled):
    model.eval()
    loss_sum = 0.0
    phase_acc_sum = 0.0
    action_acc_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, phase_t, action_t in loader:
            images = images.to(device, non_blocking=True)
            phase_t = phase_t.to(device, non_blocking=True)
            action_t = action_t.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                phase_logits, action_logits = model(images)
                p_loss = phase_loss_fn(phase_logits, phase_t)
                a_loss = action_loss_fn(action_logits, action_t)
                loss = p_loss + a_loss

            bs = images.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs
            phase_acc_sum += accuracy(phase_logits, phase_t, ignore_index=-100) * bs
            action_acc_sum += accuracy(action_logits, action_t, ignore_index=-100) * bs

    return {
        "loss": loss_sum / max(1, n),
        "phase_acc": phase_acc_sum / max(1, n),
        "action_acc": action_acc_sum / max(1, n),
    }


def main():
    parser = argparse.ArgumentParser(description="LoABot Agentic Trainer")
    parser.add_argument(
        "--train-manifest",
        default=r"E:\LoABot_Training_Data\datasets\vision\vision_train.jsonl",
    )
    parser.add_argument(
        "--test-manifest",
        default=r"E:\LoABot_Training_Data\datasets\vision\vision_test.jsonl",
    )
    parser.add_argument(
        "--action-log-root",
        default=r"E:\LoABot_Training_Data\AGENTIC_LOGS",
        help="Eger varsa gercek action loglari buradan alinip action etiketleri zenginlestirilir.",
    )
    parser.add_argument(
        "--log-root",
        default=r"E:\LoABot_Training_Data\runtime_data\training_logs",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0", "tiny"])
    parser.add_argument("--batch-size", type=int, default=0, help="0 ise otomatik secilir")
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 ise otomatik secilir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-loss-weight", type=float, default=0.8)
    parser.add_argument("--action-window-sec", type=float, default=0.75)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--with-click-head",
        action="store_true",
        help="Koordinat regresyon kafasini etkinlestir (click_knowledge.json gerektirir).",
    )
    parser.add_argument(
        "--click-loss-weight",
        type=float,
        default=0.3,
        help="Koordinat kayip agirligi (varsayilan 0.3).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    # --- CUDA Tanilama ---
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda} | CUDA available: {cuda_ok}")
    if cuda_ok:
        _dev0 = torch.cuda.get_device_properties(0)
        print(f"GPU: {_dev0.name} | VRAM: {_dev0.total_memory / 1024**3:.1f} GB")
    else:
        print("UYARI: CUDA bulunamadi — egitim CPU uzerinde calisacak (cok yavas).")
        print("  Cozum: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124")

    train_manifest = Path(args.train_manifest)
    test_manifest = Path(args.test_manifest)
    if not train_manifest.exists() or not test_manifest.exists():
        raise SystemExit("Manifest bulunamadi. Once dataset_builder.py calistirin.")

    rows_train = read_jsonl(train_manifest)
    rows_test = read_jsonl(test_manifest)
    if not rows_train or not rows_test:
        raise SystemExit("Train/Test manifest bos.")

    action_times, action_labels = load_action_events(Path(args.action_log_root))

    # Action label enrich (manifestte yoksa stage/nearest event'ten doldur)
    for r in rows_train + rows_test:
        if not r.get("action_label"):
            img_ts = parse_image_ts_from_name(r.get("image_path", ""))
            inferred = nearest_action_label(
                image_ts=img_ts,
                action_times=action_times,
                action_labels=action_labels,
                window_sec=args.action_window_sec,
            )
            if not inferred:
                inferred = infer_action_label_from_stage(r.get("stage", ""), r.get("phase_label", ""))
            r["action_label"] = inferred

    phase_id_to_name = {}
    for r in rows_train + rows_test:
        pid = int(r.get("phase_id", -1))
        if pid >= 0:
            phase_id_to_name[pid] = str(r.get("phase_label", f"phase_{pid}"))

    action_to_id = ensure_action_vocab(rows_train, rows_test)
    for r in rows_train + rows_test:
        r["action_id"] = int(action_to_id.get(str(r.get("action_label", "unknown")), action_to_id["unknown"]))

    num_phases = max(phase_id_to_name.keys()) + 1
    num_actions = len(action_to_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_workers < 0:
        num_workers = auto_num_workers()
    else:
        num_workers = max(0, int(args.num_workers))

    if args.batch_size <= 0:
        batch_size = auto_batch_size(device, args.backbone)
    else:
        batch_size = int(args.batch_size)

    # CPU korumasi: elle girilen degerler CPU icin fazla agir olabilir.
    if device.type != "cuda":
        _cpu_max_batch = 32
        if batch_size > _cpu_max_batch:
            print(f"UYARI: CPU modunda batch_size={batch_size} cok yuksek; {_cpu_max_batch} olarak dusuruldu.")
            batch_size = _cpu_max_batch
        _cpu_max_workers = max(2, (os.cpu_count() or 4) // 2)
        if num_workers > _cpu_max_workers:
            print(f"UYARI: CPU modunda num_workers={num_workers} azaltildi; {_cpu_max_workers} olarak ayarlandi.")
            num_workers = _cpu_max_workers

    train_ds = AgenticManifestDataset(
        rows=rows_train,
        image_size=args.image_size,
        train=True,
        action_to_id=action_to_id,
        action_times=action_times,
        action_labels=action_labels,
        action_window_sec=args.action_window_sec,
    )
    test_ds = AgenticManifestDataset(
        rows=rows_test,
        image_size=args.image_size,
        train=False,
        action_to_id=action_to_id,
        action_times=action_times,
        action_labels=action_labels,
        action_window_sec=args.action_window_sec,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model = AgenticNet(
        num_phases=num_phases,
        num_actions=num_actions,
        backbone=args.backbone,
        pretrained=(not args.no_pretrained),
        with_click_head=args.with_click_head,
    ).to(device)

    phase_weights = build_weights(rows_train, key="phase_id", n_classes=num_phases).to(device)
    action_weights = build_weights(rows_train, key="action_id", n_classes=num_actions).to(device)

    phase_loss_fn = nn.CrossEntropyLoss(weight=phase_weights)
    action_loss_fn = nn.CrossEntropyLoss(weight=action_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_root = Path(args.log_root)
    run_dir = run_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "phase_id_to_name.json").write_text(
        json.dumps({str(k): v for k, v in sorted(phase_id_to_name.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "action_to_id.json").write_text(
        json.dumps(action_to_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "train_manifest": str(train_manifest),
                "test_manifest": str(test_manifest),
                "action_log_root": str(args.action_log_root),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "epochs": args.epochs,
                "backbone": args.backbone,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "image_size": args.image_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "action_loss_weight": args.action_loss_weight,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "early_stop_enabled": not args.disable_early_stop,
                "tensorboard_enabled": not args.no_tensorboard,
                "num_phases": num_phases,
                "num_actions": num_actions,
                "train_samples": len(rows_train),
                "test_samples": len(rows_test),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = run_dir / "metrics.csv"
    tb_dir = run_dir / "tensorboard"
    tb_writer = None
    if not args.no_tensorboard:
        tb_writer, tb_ok = create_summary_writer(tb_dir)
        if not tb_ok:
            print("UYARI: TensorBoard import edilemedi, --no-tensorboard gibi davranilacak.")

    best_metric = -1.0
    best_epoch = 0
    early_stopped = False
    stopped_epoch = args.epochs
    early_stop_enabled = not args.disable_early_stop
    patience = max(1, int(args.early_stop_patience))
    min_delta = max(0.0, float(args.early_stop_min_delta))
    no_improve_epochs = 0
    amp_enabled = device.type == "cuda"

    print(f"Device: {device} | batch_size={batch_size} | num_workers={num_workers}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_phase_acc = 0.0
        train_action_acc = 0.0
        n = 0

        for images, phase_t, action_t in train_loader:
            images = images.to(device, non_blocking=True)
            phase_t = phase_t.to(device, non_blocking=True)
            action_t = action_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(images)
                if args.with_click_head:
                    phase_logits, action_logits, click_coords = out
                else:
                    phase_logits, action_logits = out
                p_loss = phase_loss_fn(phase_logits, phase_t)
                a_loss = action_loss_fn(action_logits, action_t)
                loss = p_loss + (args.action_loss_weight * a_loss)
                # Koordinat kaybı: sadece mouse_click örüntülerinde hesapla
                if args.with_click_head and "click_coords_t" in dir():
                    mask = (action_t == action_to_id.get("mouse_click", -1))
                    if mask.any():
                        c_loss = nn.functional.mse_loss(
                            click_coords[mask], click_coords_t[mask]
                        )
                        loss = loss + (args.click_loss_weight * c_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            n += bs
            train_loss += float(loss.item()) * bs
            train_phase_acc += accuracy(phase_logits, phase_t, ignore_index=-100) * bs
            train_action_acc += accuracy(action_logits, action_t, ignore_index=-100) * bs

        scheduler.step()

        train_metrics = {
            "loss": train_loss / max(1, n),
            "phase_acc": train_phase_acc / max(1, n),
            "action_acc": train_action_acc / max(1, n),
        }
        val_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            phase_loss_fn=phase_loss_fn,
            action_loss_fn=action_loss_fn,
            amp_enabled=amp_enabled,
        )

        composite = (val_metrics["phase_acc"] * 0.7) + (val_metrics["action_acc"] * 0.3)
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_metrics["loss"], 6),
            "train_phase_acc": round(train_metrics["phase_acc"], 6),
            "train_action_acc": round(train_metrics["action_acc"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_phase_acc": round(val_metrics["phase_acc"], 6),
            "val_action_acc": round(val_metrics["action_acc"], 6),
            "composite": round(composite, 6),
            "epoch_sec": round(epoch_time, 3),
        }
        append_csv(csv_path, row)

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            tb_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            tb_writer.add_scalar("Accuracy/train_phase", train_metrics["phase_acc"], epoch)
            tb_writer.add_scalar("Accuracy/val_phase", val_metrics["phase_acc"], epoch)
            tb_writer.add_scalar("Accuracy/train_action", train_metrics["action_acc"], epoch)
            tb_writer.add_scalar("Accuracy/val_action", val_metrics["action_acc"], epoch)
            tb_writer.add_scalar("Metric/composite_val", composite, epoch)
            tb_writer.add_scalar("LR/current", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()  # Her epoch sonunda diske yaz; TensorBoard aninda guncellenir.

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "phase_id_to_name": phase_id_to_name,
            "action_to_id": action_to_id,
            "args": vars(args),
            "metrics": row,
        }
        torch.save(ckpt, run_dir / "last.pt")
        improved = composite > (best_metric + min_delta)
        if improved:
            best_metric = composite
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            no_improve_epochs += 1

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_phase_acc={row['val_phase_acc']:.4f} "
            f"val_action_acc={row['val_action_acc']:.4f} "
            f"time={row['epoch_sec']:.1f}s"
        )

        if early_stop_enabled and no_improve_epochs >= patience:
            early_stopped = True
            stopped_epoch = epoch
            print(
                f"Early stopping tetiklendi: {no_improve_epochs} epoch boyunca iyilesme yok "
                f"(patience={patience}, min_delta={min_delta})."
            )
            break

    summary = {
        "best_composite": best_metric,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
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
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Egitim tamamlandi.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
