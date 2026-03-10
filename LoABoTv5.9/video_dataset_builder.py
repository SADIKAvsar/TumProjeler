# -*- coding: utf-8 -*-
"""
video_dataset_builder.py — PyTorch DataLoader Entegrasyonu (v2.0)
=================================================================
LoABoTv5.9 video kayıtlarından (video.mp4 + actions.jsonl) Agentic YZ
eğitimi için PyTorch Dataset/DataLoader oluşturur.

Özellikler:
  - decord ile lazy frame okuma → videonun tamamı RAM'e alınmaz
  - Her aksiyon için frame üçlüsü (T-1, T, T+1) döndürür
  - Tıklama koordinatları (x, y) ekran çözünürlüğüne göre normalize edilir
  - Oturum bazlı stratified train/test split
  - Opsiyonel frame augmentation (transform parametresi)

Kullanım:
    from video_dataset_builder import LoAVideoDataset, build_dataloaders

    train_loader, test_loader = build_dataloaders(
        video_root="D:/LoABot_Training_Data/videos",
        train_ratio=0.8,
        batch_size=16,
    )

    for batch in train_loader:
        frames_before = batch["frame_before"]   # (B, C, H, W)
        frames_action = batch["frame_action"]   # (B, C, H, W)
        frames_after  = batch["frame_after"]    # (B, C, H, W)
        coords        = batch["coords"]         # (B, 2)  normalized
        event_types   = batch["event_type"]     # list[str]
        sources       = batch["source"]         # list[str]
        labels        = batch["action_label"]   # list[str]

Gereksinimler:
    pip install decord torch torchvision
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError("PyTorch gerekli: pip install torch torchvision")

try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:
    raise ImportError("decord gerekli: pip install decord")

try:
    import cv2

    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False


# ══════════════════════════════════════════════════════════════════════
#  VERI YAPILARI
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ActionSample:
    """Tek bir eğitim örneğinin metadata'sı."""
    video_path: str
    frame_before: int       # T-1 frame index
    frame_action: int       # T frame index
    frame_after: int        # T+1 frame index
    event_type: str         # "mouse_click" | "key_press" | "bot_click" | ...
    source: str             # "bot" | "user"
    action_label: str
    phase: str
    coord_x_norm: float     # normalize (0-1), fare olayı yoksa -1.0
    coord_y_norm: float
    success: bool           # oturum başarılı mı?
    session_id: str


# ══════════════════════════════════════════════════════════════════════
#  SESSION PARSER
# ══════════════════════════════════════════════════════════════════════

def _parse_session(session_dir: Path) -> List[ActionSample]:
    """
    Tek bir oturum klasöründen ActionSample listesi üretir.

    Beklenen yapı:
        session_dir/
        ├── video.mp4
        ├── actions.jsonl
        └── session_meta.json
    """
    video_path = session_dir / "video.mp4"
    actions_path = session_dir / "actions.jsonl"
    meta_path = session_dir / "session_meta.json"

    if not video_path.exists() or not actions_path.exists():
        return []

    # Video frame sayısını al (decord ile hızlı)
    try:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames >= 3:
            # Erken bozuk-video elemesi: rastgele frame decode probe
            probe_indices = sorted({0, total_frames // 2, total_frames - 1})
            _ = vr.get_batch(probe_indices)
        del vr  # Handle'ı hemen serbest bırak
    except Exception:
        return []

    if total_frames < 3:
        return []

    # Oturum metadata'sı
    success = True
    screen_w, screen_h = 2560, 1440  # varsayılan
    session_id = session_dir.name

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            success = bool(meta.get("success", True))
            res = meta.get("resolution", {})
            screen_w = int(res.get("width", screen_w))
            screen_h = int(res.get("height", screen_h))
        except Exception:
            pass

    # Aksiyonları parse et
    samples = []
    try:
        with open(actions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    action = json.loads(line)
                except json.JSONDecodeError:
                    continue

                frame_idx = int(action.get("frame_idx", 0))
                # Sınır kontrolü — video dışına taşma engeli
                frame_before = max(0, frame_idx - 1)
                frame_after = min(total_frames - 1, frame_idx + 1)
                frame_action = min(max(0, frame_idx), total_frames - 1)

                # Koordinat normalizasyonu
                data = action.get("data", {})
                raw_x = data.get("x", -1)
                raw_y = data.get("y", -1)

                if isinstance(raw_x, (int, float)) and raw_x >= 0 and screen_w > 0:
                    coord_x = float(raw_x) / float(screen_w)
                else:
                    coord_x = -1.0

                if isinstance(raw_y, (int, float)) and raw_y >= 0 and screen_h > 0:
                    coord_y = float(raw_y) / float(screen_h)
                else:
                    coord_y = -1.0

                # Clamp to [0, 1]
                coord_x = max(0.0, min(1.0, coord_x)) if coord_x >= 0 else -1.0
                coord_y = max(0.0, min(1.0, coord_y)) if coord_y >= 0 else -1.0

                samples.append(ActionSample(
                    video_path=str(video_path),
                    frame_before=frame_before,
                    frame_action=frame_action,
                    frame_after=frame_after,
                    event_type=str(action.get("event_type", "unknown")),
                    source=str(action.get("source", "unknown")),
                    action_label=str(action.get("action_label", "")),
                    phase=str(action.get("phase", "UNKNOWN")),
                    coord_x_norm=coord_x,
                    coord_y_norm=coord_y,
                    success=success,
                    session_id=session_id,
                ))
    except Exception:
        pass

    return samples


# ══════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════

class LoAVideoDataset(Dataset):
    """
    PyTorch Dataset: Video frame üçlüsü + aksiyon metadata.

    Her __getitem__ çağrısında sadece 3 frame okunur (decord seek).
    Tüm video RAM'e yüklenmez.

    Döndürdüğü dict:
        frame_before : (3, H, W) float32 tensor [0-1 arası]
        frame_action : (3, H, W) float32 tensor
        frame_after  : (3, H, W) float32 tensor
        coords       : (2,) float32 tensor [norm x, norm y] (-1 = yok)
        event_type   : str
        source       : str
        action_label : str
        phase        : str
        success      : bool
    """

    def __init__(
        self,
        samples: List[ActionSample],
        transform: Optional[Callable] = None,
        resize: Optional[Tuple[int, int]] = None,
        max_open_readers: int = 2,
    ):
        """
        Args:
            samples: ActionSample listesi
            transform: Opsiyonel torchvision transform (frame tensörüne uygulanır)
            resize: Opsiyonel (H, W) — frame'leri bu boyuta yeniden ölçekle
        """
        self.samples = samples
        self.transform = transform
        self.resize = resize  # (H, W) tuple

        # Video reader cache — aynı videoyu tekrar açmamak için
        # {video_path: VideoReader}
        self.max_open_readers = max(1, int(max_open_readers))
        self._vr_cache: "OrderedDict[str, VideoReader]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        indices = [sample.frame_before, sample.frame_action, sample.frame_after]
        frames: Optional[np.ndarray] = None

        # Primary path: decord
        try:
            vr = self._get_video_reader(sample.video_path)
            try:
                frames = vr.get_batch(indices).asnumpy()  # (3, H, W, C) uint8 RGB
            except Exception:
                frames = np.stack([self._safe_read_frame(vr, i) for i in indices])
        except Exception:
            frames = None

        # Secondary path: OpenCV fallback
        if frames is None and HAS_OPENCV:
            frames = self._read_triplet_with_opencv(sample.video_path, indices)

        # Last-resort blank triplet (prevents hard crash on rare decode glitches)
        if frames is None:
            frames = np.zeros((3, 240, 320, 3), dtype=np.uint8)

        f_before = self._process_frame(frames[0])
        f_action = self._process_frame(frames[1])
        f_after = self._process_frame(frames[2])

        coords = torch.tensor(
            [sample.coord_x_norm, sample.coord_y_norm],
            dtype=torch.float32,
        )

        return {
            "frame_before": f_before,
            "frame_action": f_action,
            "frame_after": f_after,
            "coords": coords,
            "event_type": sample.event_type,
            "source": sample.source,
            "action_label": sample.action_label,
            "phase": sample.phase,
            "success": sample.success,
        }

    def _get_video_reader(self, video_path: str) -> VideoReader:
        """Video reader'i cache'den al veya olustur (LRU)."""
        if video_path in self._vr_cache:
            vr = self._vr_cache.pop(video_path)
            self._vr_cache[video_path] = vr
            return vr

        vr = VideoReader(video_path, ctx=decord_cpu(0), num_threads=1)
        self._vr_cache[video_path] = vr

        while len(self._vr_cache) > self.max_open_readers:
            _, old_vr = self._vr_cache.popitem(last=False)
            try:
                del old_vr
            except Exception:
                pass
        return vr

    @staticmethod
    def _safe_read_frame(vr: VideoReader, idx: int) -> np.ndarray:
        """Tek frame guvenli okuma."""
        try:
            idx = max(0, min(len(vr) - 1, idx))
            return vr[idx].asnumpy()
        except Exception:
            return np.zeros((240, 320, 3), dtype=np.uint8)

    @staticmethod
    def _read_triplet_with_opencv(video_path: str, indices: List[int]) -> Optional[np.ndarray]:
        """OpenCV fallback for sessions that intermittently fail in decord."""
        if not HAS_OPENCV:
            return None

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frames: List[np.ndarray] = []
            for idx in indices:
                safe_idx = max(0, idx)
                if total > 0:
                    safe_idx = min(total - 1, safe_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    return None
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            return np.stack(frames)
        except Exception:
            return None
        finally:
            if cap is not None:
                cap.release()


    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Frame'i PyTorch tensörüne çevir:
        (H, W, C) uint8 → (C, H, W) float32 [0-1]
        """
        # Resize (opsiyonel)
        if self.resize is not None:
            import cv2
            frame = cv2.resize(
                frame,
                (self.resize[1], self.resize[0]),  # cv2: (W, H)
                interpolation=cv2.INTER_LINEAR,
            )

        # HWC → CHW, uint8 → float32 [0, 1]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Opsiyonel transform (torchvision Normalize vb.)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def close(self):
        """Video reader cache'ini temizle."""
        self._vr_cache.clear()


# ══════════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════

def scan_sessions(video_root: str) -> List[ActionSample]:
    """Tüm VID_* oturumlarını tarayıp ActionSample listesi oluşturur."""
    root = Path(video_root)
    if not root.exists():
        print(f"[WARN] Video root bulunamadı: {root}")
        return []

    all_samples = []
    session_dirs = sorted(root.glob("VID_*"))
    print(f"[INFO] {len(session_dirs)} oturum klasörü bulundu.")

    for sdir in session_dirs:
        samples = _parse_session(sdir)
        all_samples.extend(samples)

    print(f"[INFO] Toplam {len(all_samples)} aksiyon örneği parse edildi.")
    return all_samples


def stratified_split(
    samples: List[ActionSample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[ActionSample], List[ActionSample]]:
    """
    Oturum bazlı stratified split.
    Aynı oturumdaki tüm aksiyonlar aynı split'te kalır.
    Başarılı/başarısız oturum oranı korunur.
    """
    # Oturumlara grupla
    sessions: Dict[str, List[ActionSample]] = {}
    for s in samples:
        sessions.setdefault(s.session_id, []).append(s)

    # Başarılı / başarısız ayrımı
    success_sessions = []
    fail_sessions = []
    for sid, session_samples in sessions.items():
        if session_samples[0].success:
            success_sessions.append(sid)
        else:
            fail_sessions.append(sid)

    rng = random.Random(seed)

    def _split_list(ids: List[str]) -> Tuple[List[str], List[str]]:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * train_ratio))
        return shuffled[:cut], shuffled[cut:]

    train_success, test_success = _split_list(success_sessions)
    train_fail, test_fail = _split_list(fail_sessions)

    train_ids = set(train_success + train_fail)
    test_ids = set(test_success + test_fail)

    train_samples = [s for s in samples if s.session_id in train_ids]
    test_samples = [s for s in samples if s.session_id in test_ids]

    print(
        f"[INFO] Split: train={len(train_samples)} "
        f"({len(train_ids)} oturum), "
        f"test={len(test_samples)} ({len(test_ids)} oturum)"
    )
    return train_samples, test_samples


def collate_action_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Windows multiprocessing-safe collate function.
    Must stay at module scope so DataLoader workers can pickle it.
    """
    if not batch:
        return {}

    result: Dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def build_dataloaders(
    video_root: str,
    train_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 0,
    resize: Optional[Tuple[int, int]] = None,
    transform: Optional[Callable] = None,
    seed: int = 42,
    max_open_readers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    End-to-end: tarama → split → Dataset → DataLoader.

    Args:
        video_root: VID_* klasörlerinin bulunduğu kök dizin
        train_ratio: eğitim seti oranı (0-1)
        batch_size: batch boyutu
        num_workers: DataLoader worker sayısı (0 = ana thread)
        resize: opsiyonel (H, W) — frame boyutu
        transform: opsiyonel torchvision transform
        seed: rastgelelik seed'i

    Returns:
        (train_loader, test_loader)
    """
    all_samples = scan_sessions(video_root)
    if not all_samples:
        print("[WARN] Hiç aksiyon örneği bulunamadı — boş DataLoader döndürülüyor.")
        empty_ds = LoAVideoDataset([], transform=transform, resize=resize)
        return (
            DataLoader(empty_ds, batch_size=1),
            DataLoader(empty_ds, batch_size=1),
        )

    train_samples, test_samples = stratified_split(
        all_samples, train_ratio=train_ratio, seed=seed
    )

    train_ds = LoAVideoDataset(train_samples, transform=transform, resize=resize, max_open_readers=max_open_readers)
    test_ds = LoAVideoDataset(test_samples, transform=transform, resize=resize, max_open_readers=max_open_readers)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_action_batch,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        # ffmpeg/decord memory spikes on weak CPUs: keep worker prefetch minimal.
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = False

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="LoABot Video Dataset Builder")
    parser.add_argument(
        "--video-root",
        type=str,
        default=r"D:\LoABot_Training_Data\videos",
        help="VID_* oturum klasörlerinin kök dizini",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--resize", type=str, default=None,
        help="Frame boyutu: 'HxW' formatı (örn: 360x640)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Sadece istatistik göster")
    args = parser.parse_args()

    resize = None
    if args.resize:
        parts = args.resize.lower().split("x")
        if len(parts) == 2:
            resize = (int(parts[0]), int(parts[1]))

    all_samples = scan_sessions(args.video_root)

    if not all_samples:
        print("Hiç aksiyon örneği bulunamadı.")
        sys.exit(1)

    # İstatistikler
    sessions = set(s.session_id for s in all_samples)
    success_count = sum(1 for s in all_samples if s.success)
    event_types = {}
    sources = {}
    phases = {}
    for s in all_samples:
        event_types[s.event_type] = event_types.get(s.event_type, 0) + 1
        sources[s.source] = sources.get(s.source, 0) + 1
        phases[s.phase] = phases.get(s.phase, 0) + 1

    print("\n" + "=" * 60)
    print("DATASET İSTATİSTİKLERİ")
    print("=" * 60)
    print(f"  Toplam oturum     : {len(sessions)}")
    print(f"  Toplam aksiyon    : {len(all_samples)}")
    print(f"  Başarılı aksiyonlar: {success_count} ({100*success_count/len(all_samples):.1f}%)")
    print(f"\n  Event türleri: {json.dumps(event_types, indent=4)}")
    print(f"\n  Kaynaklar    : {json.dumps(sources, indent=4)}")
    print(f"\n  Fazlar       : {json.dumps(phases, indent=4)}")

    if args.dry_run:
        print("\n[DRY-RUN] DataLoader oluşturma atlandı.")
        sys.exit(0)

    train_loader, test_loader = build_dataloaders(
        video_root=args.video_root,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        resize=resize,
    )

    # Bir batch test et
    print(f"\n[TEST] İlk batch okunuyor (batch_size={args.batch_size})...")
    try:
        batch = next(iter(train_loader))
        print(f"  frame_action shape: {batch['frame_action'].shape}")
        print(f"  coords shape      : {batch['coords'].shape}")
        print(f"  event_types       : {batch['event_type'][:3]}...")
        print(f"  sources           : {batch['source'][:3]}...")
        print("\n✓ DataLoader başarıyla çalışıyor!")
    except StopIteration:
        print("[WARN] Train loader boş.")
    except Exception as exc:
        print(f"[HATA] Batch okuma hatası: {exc}")
