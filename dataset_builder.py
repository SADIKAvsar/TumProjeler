import argparse
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


PHASE_FOLDERS = {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE", "EVENT_PHASE"}
STAGE_HINTS = [
    "Transition",
    "anchor_click",
    "attack_start",
    "loot_wait",
    "Event_Start",
    "Event_Action",
    "Event_Wait",
    "mission_loop",
    "spawn_check",
    "area_check",
    "victory",
    "popup_detected",
    "dc_popup_detected",
    "freeze_check_triggered",
    "restart_triggered",
    "server_selection_screen",
    "character_selection_screen",
    "play_button_screen",
]


def _scan_dir_for_ext(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    found = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.lower().endswith(exts):
                found.append(str(Path(root) / name))
    return found


def _collect_files_parallel(root: Path, exts: Tuple[str, ...], workers: int) -> List[str]:
    root = root.resolve()
    top_dirs = [p for p in root.iterdir() if p.is_dir()]
    files_at_root = [str(p) for p in root.iterdir() if p.is_file() and p.name.lower().endswith(exts)]

    if not top_dirs:
        return files_at_root + _scan_dir_for_ext(str(root), exts)

    out = list(files_at_root)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_scan_dir_for_ext, str(d), exts) for d in top_dirs]
        for fut in as_completed(futures):
            out.extend(fut.result())
    return out


def _infer_stage_from_stem(stem: str) -> str:
    s = stem.lower()
    for hint in STAGE_HINTS:
        if hint.lower() in s:
            return hint
    return "mission_loop"


def _infer_namespace_stage_phase(rel_path: Path) -> Tuple[str, str, str]:
    parts = rel_path.parts
    if not parts:
        return "UNKNOWN", "unknown_stage", "UNKNOWN_PHASE"

    namespace = str(parts[0])
    stem = rel_path.stem

    if len(parts) >= 3:
        stage = str(parts[1])
    else:
        stage = _infer_stage_from_stem(stem)

    ns_upper = namespace.upper()
    st_lower = stage.lower()

    if ns_upper in PHASE_FOLDERS:
        phase = ns_upper
    elif "transition" in st_lower:
        phase = "NAV_PHASE"
    elif ns_upper in {"SYSTEM_RECOVERY", "STARTUP_SYNC"}:
        phase = ns_upper
    elif namespace.isdigit() or ns_upper.startswith("BOSS_"):
        if "victory" in st_lower:
            phase = "COMBAT_PHASE"
        elif "loot" in st_lower:
            phase = "LOOT_PHASE"
        elif any(k in st_lower for k in ("spawn", "area", "anchor")):
            phase = "NAV_PHASE"
        else:
            phase = "BOSS_PHASE"
    else:
        phase = "UNKNOWN_PHASE"

    return namespace, stage, phase


def _read_yolo_txt(txt_path: Path) -> Tuple[List[int], List[List[float]]]:
    if not txt_path.exists():
        return [], []

    labels = []
    boxes = []
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return [], []

    if not content:
        return [], []

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except Exception:
            continue
        labels.append(cls)
        boxes.append([x, y, w, h])

    return labels, boxes


def _process_image_chunk(paths: List[str], root_path: str) -> List[Dict]:
    out = []
    root = Path(root_path)

    for p in paths:
        img_path = Path(p)
        try:
            rel = img_path.relative_to(root)
        except Exception:
            rel = img_path

        namespace, stage, phase = _infer_namespace_stage_phase(rel)
        txt_path = img_path.with_suffix(".txt")
        yolo_meta_path = img_path.with_suffix(".yolo.json")
        labels, boxes = _read_yolo_txt(txt_path)

        rec = {
            "image_path": str(img_path),
            "rel_image_path": str(rel),
            "label_path": str(txt_path) if txt_path.exists() else "",
            "yolo_meta_path": str(yolo_meta_path) if yolo_meta_path.exists() else "",
            "namespace": namespace,
            "stage": stage,
            "phase_label": phase,
            "has_bbox": bool(len(boxes) > 0),
            "labels": labels,
            "boxes_norm": boxes,
        }
        out.append(rec)

    return out


def _chunk_list(items: List[str], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _build_vision_index(root: Path, workers: int) -> List[Dict]:
    image_paths = _collect_files_parallel(root=root, exts=(".jpg",), workers=workers)
    if not image_paths:
        return []

    chunk_size = max(256, len(image_paths) // max(1, workers * 8))
    chunks = list(_chunk_list(image_paths, chunk_size))
    rows = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_image_chunk, chunk, str(root)) for chunk in chunks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
    return rows


def _split_stratified(rows: List[Dict], train_ratio: float, seed: int):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["phase_label"]].append(r)

    train_rows, test_rows = [], []
    for phase, items in grouped.items():
        rng.shuffle(items)
        n = len(items)
        cut = int(n * train_ratio)
        if n >= 2:
            cut = max(1, min(n - 1, cut))
        else:
            cut = n
        train_rows.extend(items[:cut])
        test_rows.extend(items[cut:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def _write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summarize_episode_file(path_str: str) -> Dict:
    p = Path(path_str)
    counts = Counter()
    episode_id = p.stem
    first_ts = None
    last_ts = None
    final_status = ""
    total = 0

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                total += 1
                evt = str(obj.get("event_type", "unknown"))
                counts[evt] += 1
                if not first_ts:
                    first_ts = obj.get("ts_iso")
                last_ts = obj.get("ts_iso")
                episode_id = str(obj.get("episode_id", episode_id))
                if evt == "episode_end":
                    final_status = str((obj.get("payload") or {}).get("status", ""))
    except Exception:
        return {}

    return {
        "episode_id": episode_id,
        "file_path": str(p),
        "event_count": total,
        "event_type_counts": dict(counts),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "final_status": final_status,
    }


def _build_episode_index(data_root: Path, workers: int) -> List[Dict]:
    agentic_root = data_root / "AGENTIC_LOGS"
    if not agentic_root.exists():
        return []

    files = _collect_files_parallel(root=agentic_root, exts=(".jsonl",), workers=workers)
    if not files:
        return []

    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_summarize_episode_file, p) for p in files]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec:
                out.append(rec)
    return out


def _split_simple(rows: List[Dict], train_ratio: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    cut = int(len(rows) * train_ratio)
    if len(rows) >= 2:
        cut = max(1, min(len(rows) - 1, cut))
    return rows[:cut], rows[cut:]


def main():
    parser = argparse.ArgumentParser(description="LoABot veri seti olusturucu (parallel scan + train/test split)")
    parser.add_argument("--data-root", default=r"E:\LoABot_Training_Data", help="Toplanan ham veri kok klasoru")
    parser.add_argument(
        "--output-root",
        default=r"E:\LoABot_Training_Data\datasets",
        help="Manifest cikti klasoru",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train orani (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohum degeri")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)), help="Paralel worker sayisi")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_root).resolve()
    workers = max(1, int(args.workers))
    train_ratio = min(0.95, max(0.05, float(args.train_ratio)))

    if not data_root.exists():
        raise SystemExit(f"Data root bulunamadi: {data_root}")

    print(f"[1/4] Vision index olusturuluyor... workers={workers}")
    vision_rows = _build_vision_index(root=data_root, workers=workers)
    if not vision_rows:
        raise SystemExit("Hic .jpg bulunamadi.")

    phase_labels = sorted({r["phase_label"] for r in vision_rows})
    phase_map = {name: idx for idx, name in enumerate(phase_labels)}
    for r in vision_rows:
        r["phase_id"] = phase_map[r["phase_label"]]

    print(f"[2/4] Train/Test split (%{int(train_ratio*100)} / %{int((1-train_ratio)*100)})")
    vision_train, vision_test = _split_stratified(vision_rows, train_ratio=train_ratio, seed=args.seed)

    for r in vision_train:
        r["split"] = "train"
    for r in vision_test:
        r["split"] = "test"

    vision_dir = out_root / "vision"
    _write_jsonl(vision_dir / "vision_all.jsonl", vision_rows)
    _write_jsonl(vision_dir / "vision_train.jsonl", vision_train)
    _write_jsonl(vision_dir / "vision_test.jsonl", vision_test)
    (vision_dir / "phase_map.json").write_text(json.dumps(phase_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[3/4] Episode index olusturuluyor...")
    episodes = _build_episode_index(data_root=data_root, workers=workers)
    episodes_dir = out_root / "episodes"
    if episodes:
        ep_train, ep_test = _split_simple(episodes, train_ratio=train_ratio, seed=args.seed)
        for r in ep_train:
            r["split"] = "train"
        for r in ep_test:
            r["split"] = "test"
        _write_jsonl(episodes_dir / "episodes_all.jsonl", episodes)
        _write_jsonl(episodes_dir / "episodes_train.jsonl", ep_train)
        _write_jsonl(episodes_dir / "episodes_test.jsonl", ep_test)

    print("[4/4] Istatistikler yaziliyor...")
    stats = {
        "data_root": str(data_root),
        "workers": workers,
        "train_ratio": train_ratio,
        "vision_total": len(vision_rows),
        "vision_train": len(vision_train),
        "vision_test": len(vision_test),
        "phase_distribution": dict(Counter(r["phase_label"] for r in vision_rows)),
        "episodes_total": len(episodes),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Tamamlandi.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
