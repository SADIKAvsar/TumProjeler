# -*- coding: utf-8 -*-
"""
sanitize_video_sessions.py

Small utility to validate and quarantine broken video training sessions.

Session structure expected:
    VID_YYYYMMDD_HHMMSS/
      - video.mp4
      - actions.jsonl
      - session_meta.json (optional)

Default behavior:
    - scan sessions under --video-root
    - validate video.mp4 + actions.jsonl
    - move invalid sessions to quarantine folder

Examples:
    python sanitize_video_sessions.py
    python sanitize_video_sessions.py --dry-run
    python sanitize_video_sessions.py --mode delete
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from decord import VideoReader, cpu as decord_cpu

    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import cv2

    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False


@dataclass
class SessionCheck:
    session_id: str
    path: str
    valid: bool
    frame_count: int
    action_count: int
    video_backend: str
    reasons: List[str]
    action_taken: str = "none"
    action_target: str = ""


def _check_video_with_decord(video_path: Path, min_frames: int) -> Tuple[bool, int, List[str]]:
    reasons: List[str] = []
    try:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        frame_count = int(len(vr))
        if frame_count < min_frames:
            reasons.append(f"video_too_short: {frame_count} < {min_frames}")
            return False, frame_count, reasons

        probe_indices = sorted({0, frame_count // 2, frame_count - 1})
        _ = vr.get_batch(probe_indices)
        return True, frame_count, reasons
    except Exception as exc:
        reasons.append(f"decord_open_failed: {exc}")
        return False, 0, reasons


def _check_video_with_opencv(video_path: Path, min_frames: int) -> Tuple[bool, int, List[str]]:
    reasons: List[str] = []
    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            reasons.append("opencv_open_failed")
            return False, 0, reasons

        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            reasons.append("opencv_read_failed")
            return False, 0, reasons

        frame_count = raw_count
        if frame_count <= 0:
            # Conservative fallback counting if container metadata is missing.
            frame_count = 1
            while frame_count < min_frames:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1

        if frame_count < min_frames:
            reasons.append(f"video_too_short: {frame_count} < {min_frames}")
            return False, frame_count, reasons

        return True, frame_count, reasons
    except Exception as exc:
        reasons.append(f"opencv_open_failed: {exc}")
        return False, 0, reasons
    finally:
        if cap is not None:
            cap.release()


def _check_video(video_path: Path, min_frames: int, prefer_decord: bool) -> Tuple[bool, int, str, List[str]]:
    all_reasons: List[str] = []

    if prefer_decord and HAS_DECORD:
        ok, frames, reasons = _check_video_with_decord(video_path, min_frames)
        if ok:
            return True, frames, "decord", []
        all_reasons.extend(reasons)

    if HAS_OPENCV:
        ok, frames, reasons = _check_video_with_opencv(video_path, min_frames)
        if ok:
            return True, frames, "opencv", []
        all_reasons.extend(reasons)

    if not HAS_DECORD and not HAS_OPENCV:
        all_reasons.append("no_video_backend_available(decord/opencv)")

    return False, 0, "none", all_reasons


def _check_actions(actions_path: Path, require_nonempty: bool) -> Tuple[bool, int, List[str]]:
    reasons: List[str] = []
    valid_lines = 0

    try:
        with open(actions_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                row = raw.strip()
                if not row:
                    continue
                try:
                    _ = json.loads(row)
                    valid_lines += 1
                except Exception:
                    reasons.append(f"actions_json_decode_failed@line={line_no}")
                    # Keep scanning for better diagnostics.
    except Exception as exc:
        reasons.append(f"actions_open_failed: {exc}")
        return False, 0, reasons

    if require_nonempty and valid_lines == 0:
        reasons.append("actions_empty_or_invalid")
        return False, 0, reasons

    if reasons:
        return False, valid_lines, reasons
    return True, valid_lines, reasons


def _validate_session(session_dir: Path, min_frames: int, prefer_decord: bool) -> SessionCheck:
    session_id = session_dir.name
    reasons: List[str] = []
    frame_count = 0
    action_count = 0
    backend = "none"

    video_path = session_dir / "video.mp4"
    actions_path = session_dir / "actions.jsonl"

    if not video_path.exists():
        reasons.append("missing_video.mp4")
    if not actions_path.exists():
        reasons.append("missing_actions.jsonl")

    if reasons:
        return SessionCheck(
            session_id=session_id,
            path=str(session_dir),
            valid=False,
            frame_count=frame_count,
            action_count=action_count,
            video_backend=backend,
            reasons=reasons,
        )

    video_ok, frame_count, backend, video_reasons = _check_video(video_path, min_frames, prefer_decord)
    reasons.extend(video_reasons)

    actions_ok, action_count, action_reasons = _check_actions(actions_path, require_nonempty=True)
    reasons.extend(action_reasons)

    is_valid = video_ok and actions_ok and not reasons
    return SessionCheck(
        session_id=session_id,
        path=str(session_dir),
        valid=is_valid,
        frame_count=frame_count,
        action_count=action_count,
        video_backend=backend,
        reasons=reasons,
    )


def _safe_move(src: Path, dst_root: Path) -> Path:
    dst_root.mkdir(parents=True, exist_ok=True)
    target = dst_root / src.name
    if not target.exists():
        shutil.move(str(src), str(target))
        return target

    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = dst_root / f"{src.name}_{stamp}"
    shutil.move(str(src), str(target))
    return target


def _act_on_invalid(session: SessionCheck, mode: str, quarantine_root: Path, dry_run: bool) -> SessionCheck:
    src = Path(session.path)
    if mode == "report" or dry_run:
        session.action_taken = "none"
        session.action_target = ""
        return session

    if mode == "move":
        dst = _safe_move(src, quarantine_root)
        session.action_taken = "moved"
        session.action_target = str(dst)
        return session

    if mode == "delete":
        shutil.rmtree(src, ignore_errors=False)
        session.action_taken = "deleted"
        session.action_target = ""
        return session

    session.action_taken = "none"
    return session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and quarantine broken VID_* video sessions.")
    parser.add_argument(
        "--video-root",
        default=r"D:\LoABot_Training_Data\videos",
        help="Root directory containing VID_* session folders.",
    )
    parser.add_argument(
        "--quarantine-root",
        default="",
        help="Invalid sessions are moved here in --mode move. Default: <video-root>/_quarantine_bad_sessions",
    )
    parser.add_argument(
        "--mode",
        choices=["report", "move", "delete"],
        default="move",
        help="Action for invalid sessions.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Minimum acceptable frame count.",
    )
    parser.add_argument(
        "--skip-decord",
        action="store_true",
        help="Do not use decord first for video validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only. No move/delete actions.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional JSON report output path. Default: <video-root>/sanitize_report.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    video_root = Path(args.video_root).resolve()
    if not video_root.exists():
        raise SystemExit(f"video_root not found: {video_root}")

    quarantine_root = (
        Path(args.quarantine_root).resolve()
        if args.quarantine_root
        else (video_root / "_quarantine_bad_sessions")
    )
    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else (video_root / "sanitize_report.json")
    )

    session_dirs = sorted([p for p in video_root.glob("VID_*") if p.is_dir()])
    print(f"[INFO] found_sessions={len(session_dirs)} root={video_root}")
    print(
        f"[INFO] mode={args.mode} dry_run={args.dry_run} min_frames={args.min_frames} "
        f"decord={HAS_DECORD} opencv={HAS_OPENCV}"
    )

    checked: List[SessionCheck] = []
    valid_count = 0
    invalid_count = 0

    prefer_decord = not bool(args.skip_decord)
    for session_dir in session_dirs:
        result = _validate_session(
            session_dir=session_dir,
            min_frames=max(1, int(args.min_frames)),
            prefer_decord=prefer_decord,
        )

        if result.valid:
            valid_count += 1
            checked.append(result)
            continue

        invalid_count += 1
        result = _act_on_invalid(
            session=result,
            mode=args.mode,
            quarantine_root=quarantine_root,
            dry_run=bool(args.dry_run),
        )
        checked.append(result)

        reason_preview = "; ".join(result.reasons[:2])
        print(
            f"[INVALID] {result.session_id} backend={result.video_backend} "
            f"action={result.action_taken} reasons={reason_preview}"
        )

    report = {
        "video_root": str(video_root),
        "quarantine_root": str(quarantine_root),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "min_frames": int(args.min_frames),
        "prefer_decord": prefer_decord,
        "decord_available": HAS_DECORD,
        "opencv_available": HAS_OPENCV,
        "total_sessions": len(session_dirs),
        "valid_sessions": valid_count,
        "invalid_sessions": invalid_count,
        "results": [asdict(item) for item in checked],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[DONE] total={len(session_dirs)} valid={valid_count} invalid={invalid_count} "
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
