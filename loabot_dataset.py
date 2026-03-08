import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def _image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    arr = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class LoABotVisionDataset(Dataset):
    """
    Manifest tabanli dataset.
    mode='classification' -> (image, phase_id)
    mode='detection'     -> (image, target_dict)
    """

    def __init__(
        self,
        manifest_path: str,
        mode: str = "classification",
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        return_meta: bool = False,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.mode = str(mode).strip().lower()
        self.transform = transform
        self.return_meta = bool(return_meta)
        self.samples = _load_jsonl(self.manifest_path)

        if self.mode not in {"classification", "detection"}:
            raise ValueError(f"Desteklenmeyen mode: {self.mode}")

        if not self.samples:
            raise ValueError(f"Manifest bos veya okunamadi: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Goruntu okunamadi: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img = self._read_image(row["image_path"])

        if self.transform is not None:
            out = self.transform(img)
            if isinstance(out, torch.Tensor):
                image_tensor = out
            elif isinstance(out, np.ndarray):
                image_tensor = _image_to_tensor(out)
            elif isinstance(out, dict) and "image" in out:
                image_tensor = out["image"]
                if isinstance(image_tensor, np.ndarray):
                    image_tensor = _image_to_tensor(image_tensor)
            else:
                image_tensor = _image_to_tensor(img)
        else:
            image_tensor = _image_to_tensor(img)

        phase_id = int(row.get("phase_id", -1))

        if self.mode == "classification":
            target = torch.tensor(phase_id, dtype=torch.long)
            if self.return_meta:
                return image_tensor, target, row
            return image_tensor, target

        # detection mode
        boxes = torch.tensor(row.get("boxes_norm", []), dtype=torch.float32)
        labels = torch.tensor(row.get("labels", []), dtype=torch.long)
        target = {
            "boxes_norm": boxes,  # [N,4], normalized x_center,y_center,w,h
            "labels": labels,     # [N]
            "phase_id": torch.tensor(phase_id, dtype=torch.long),
            "has_bbox": torch.tensor(bool(row.get("has_bbox", False)), dtype=torch.bool),
        }
        if self.return_meta:
            return image_tensor, target, row
        return image_tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_dataloader(
    manifest_path: str,
    mode: str = "classification",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    transform: Optional[Callable[[np.ndarray], Any]] = None,
) -> Tuple[LoABotVisionDataset, DataLoader]:
    ds = LoABotVisionDataset(
        manifest_path=manifest_path,
        mode=mode,
        transform=transform,
        return_meta=False,
    )
    workers = int(num_workers) if num_workers is not None else max(0, min(8, ((os.cpu_count() or 1) - 1)))
    collate = detection_collate if mode == "detection" else None
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(pin_memory),
        collate_fn=collate,
        persistent_workers=bool(workers > 0),
    )
    return ds, dl


def os_cpu_count() -> int:
    import os

    return int(os.cpu_count() or 1)
