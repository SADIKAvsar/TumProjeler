"""
seq_dataset_builder.py — Sekans Veri Seti Oluşturucu
=====================================================
SequentialRecorder'ın ürettiği Sequence_XXXX klasör yapısını okur ve
LSTM / TimeSformer eğitimine hazır manifest JSONL dosyaları üretir.

SequentialRecorder çıktı yapısı:
  sequences/
  └── SESSION_YYYYMMDD_HHMMSS/
      ├── Sequence_0001/
      │   ├── frame_00.jpg ... frame_09.jpg
      │   └── metadata.json
      ...

metadata.json alanları okunur:
  phase, action_label, trigger_type, stuck, idle, reward, char_position

Çıktı (manifest):
  datasets/sequences/
  ├── seq_all.jsonl        → tüm sekanslar
  ├── seq_train.jsonl      → %80 train
  ├── seq_test.jsonl       → %20 test
  ├── action_to_id.json    → {"key_a": 0, "key_q": 1, ...}
  └── phase_to_id.json     → {"COMBAT_PHASE": 0, ...}

Her satır:
  {
    "sequence_id"  : "SESSION_20260223_143022/Sequence_0001",
    "frames"       : ["...frame_00.jpg", ...],   # mutlak yollar
    "phase"        : "COMBAT_PHASE",
    "phase_id"     : 0,
    "action_label" : "key_a",
    "action_id"    : 0,
    "trigger_type" : "auto_rule_engine",   # manual_user | auto_rule_engine
    "stuck"        : false,
    "idle"         : false,
    "reward"       : 0.1,                  # reward yoksa 0.0
    "num_frames"   : 10,
    "split"        : "train"
  }

Kullanım:
    python seq_dataset_builder.py --seq-root E:\\LoABot_Training_Data\\sequences
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Sabitler ──────────────────────────────────────────────────────────────

# Eğer reward yoksa bu değeri kullan
_DEFAULT_REWARD = 0.0

# Minimum kare sayısı — daha azı olan sekansları atla
_MIN_FRAMES = 5


# ── Okuma ─────────────────────────────────────────────────────────────────

def _load_metadata(meta_path: Path) -> Optional[Dict]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_sequences(seq_root: Path) -> List[Dict]:
    """
    seq_root altındaki tüm SESSION_*/Sequence_*/ klasörlerini tara.
    Her geçerli sekans için ham kayıt döndür.
    """
    records = []

    for session_dir in sorted(seq_root.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith("SESSION_"):
            continue

        for seq_dir in sorted(session_dir.iterdir()):
            if not seq_dir.is_dir() or not seq_dir.name.startswith("Sequence_"):
                continue

            meta_path = seq_dir / "metadata.json"
            if not meta_path.exists():
                continue

            meta = _load_metadata(meta_path)
            if meta is None:
                continue

            # Kare dosyalarını sıraya diz
            frames = []
            for frame_info in meta.get("frames", []):
                fname = frame_info.get("filename", "")
                fpath = seq_dir / fname
                if fpath.exists():
                    frames.append(str(fpath))

            if len(frames) < _MIN_FRAMES:
                continue  # Yarım sekans — atla

            # Reward: direkt value veya iç içe dict
            reward_raw = meta.get("reward", _DEFAULT_REWARD)
            if isinstance(reward_raw, dict):
                reward = float(reward_raw.get("value", _DEFAULT_REWARD))
            else:
                reward = float(reward_raw if reward_raw is not None else _DEFAULT_REWARD)

            records.append({
                "sequence_id"  : f"{session_dir.name}/{seq_dir.name}",
                "frames"       : frames,
                "phase"        : str(meta.get("phase", "UNKNOWN_PHASE")),
                "action_label" : str(meta.get("action_label", "unknown")),
                "trigger_type" : str(meta.get("trigger_type", "auto_rule_engine")),
                "stuck"        : bool(meta.get("stuck", False)),
                "idle"         : bool(meta.get("idle", False)),
                "reward"       : reward,
                "num_frames"   : len(frames),
            })

    return records


# ── Etiket Eşleme ─────────────────────────────────────────────────────────

def _build_label_maps(records: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    action_label → action_id
    phase        → phase_id
    """
    actions = sorted({r["action_label"] for r in records})
    phases  = sorted({r["phase"]        for r in records})
    action_to_id = {a: i for i, a in enumerate(actions)}
    phase_to_id  = {p: i for i, p in enumerate(phases)}
    return action_to_id, phase_to_id


# ── Train/Test Split (stratified by action_label) ─────────────────────────

def _split_stratified(
    records: List[Dict],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        grouped[r["action_label"]].append(r)

    train, test = [], []
    for items in grouped.values():
        rng.shuffle(items)
        n = len(items)
        cut = int(n * train_ratio)
        cut = max(1, min(n - 1, cut)) if n >= 2 else n
        train.extend(items[:cut])
        test.extend(items[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ── Yazma ─────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Ana ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SequentialRecorder çıktısından LSTM/TimeSformer manifest üretir."
    )
    parser.add_argument(
        "--seq-root",
        default=r"E:\LoABot_Training_Data\sequences",
        help="SequentialRecorder kayıt kök klasörü",
    )
    parser.add_argument(
        "--output-root",
        default=r"E:\LoABot_Training_Data\datasets\sequences",
        help="Manifest çıktı klasörü",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seq_root  = Path(args.seq_root).resolve()
    out_root  = Path(args.output_root).resolve()
    train_ratio = min(0.95, max(0.05, float(args.train_ratio)))

    if not seq_root.exists():
        raise SystemExit(f"seq-root bulunamadi: {seq_root}")

    print(f"[1/4] Sekanslar taranıyor: {seq_root}")
    records = _collect_sequences(seq_root)
    if not records:
        raise SystemExit("Hiç geçerli sekans bulunamadı.")
    print(f"       Bulunan sekans: {len(records)}")

    print("[2/4] Etiket haritaları oluşturuluyor...")
    action_to_id, phase_to_id = _build_label_maps(records)
    for r in records:
        r["action_id"] = action_to_id[r["action_label"]]
        r["phase_id"]  = phase_to_id[r["phase"]]

    print(f"[3/4] Train/Test split (%{int(train_ratio*100)} / %{int((1-train_ratio)*100)})")
    train_rows, test_rows = _split_stratified(records, train_ratio, args.seed)
    for r in train_rows:
        r["split"] = "train"
    for r in test_rows:
        r["split"] = "test"
    all_rows = train_rows + test_rows

    print("[4/4] Dosyalar yazılıyor...")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_root / "seq_all.jsonl",   all_rows)
    _write_jsonl(out_root / "seq_train.jsonl", train_rows)
    _write_jsonl(out_root / "seq_test.jsonl",  test_rows)
    (out_root / "action_to_id.json").write_text(
        json.dumps(action_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "phase_to_id.json").write_text(
        json.dumps(phase_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "seq_root"           : str(seq_root),
        "total_sequences"    : len(records),
        "train_sequences"    : len(train_rows),
        "test_sequences"     : len(test_rows),
        "train_ratio"        : train_ratio,
        "action_distribution": dict(Counter(r["action_label"] for r in records)),
        "phase_distribution" : dict(Counter(r["phase"]        for r in records)),
        "trigger_distribution": dict(Counter(r["trigger_type"] for r in records)),
        "stuck_count"        : sum(1 for r in records if r["stuck"]),
        "reward_mean"        : round(sum(r["reward"] for r in records) / max(1, len(records)), 4),
        "num_actions"        : len(action_to_id),
        "num_phases"         : len(phase_to_id),
    }
    (out_root / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTamamlandı.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
