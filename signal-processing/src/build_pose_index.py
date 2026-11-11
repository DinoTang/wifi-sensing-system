"""Build the session index that pairs each CSI capture with its video.

Layout produced by the recorder (csi-data/):

    csi-data/labeled/<trial>.../          raw.csi, recorder.log, metadata.txt,
                                          start-time.txt, end-time.txt, wifi.txt,
                                          ping.log, screen-clock.csv,
                                          video-session-notes.txt
    csi-data/video-sessions/<session>/    *.mov, session-notes.txt, screen-clock.csv,
                                          csi-folder.txt, capture-console.log

The CSI recorder always writes ``label=test`` and only knows the trial id; the
real activity comes from ``video-session-notes.txt``. This script joins the two
sides by trial id, verifies that every required file exists and that the video
decodes, then writes ``data/pose/index.csv``.

Run from anywhere::

    python signal-processing/src/build_pose_index.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_intel_csi import GROUPS, parse_recorder_log, read_key_values  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CSI_DATA = PROJECT / "csi-data"
DEFAULT_OUTPUT = PROJECT / "data" / "pose" / "index.csv"

SIX_ACTIVITIES = ["empty", "standing", "left_arm", "right_arm", "walking", "mixed"]

CSI_FILES = [
    "raw.csi",
    "recorder.log",
    "metadata.txt",
    "start-time.txt",
    "end-time.txt",
    "wifi.txt",
    "ping.log",
    "video-session-notes.txt",
]
ACTIVITY_CSI_FILES = [
    "raw.csi",
    "recorder.log",
    "metadata.txt",
    "start-time.txt",
    "end-time.txt",
    "wifi.txt",
    "ping.log",
    "video-session-notes.txt",
]

VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair CSI sessions with their videos")
    parser.add_argument("--csi-data", type=Path, default=DEFAULT_CSI_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group", choices=sorted(GROUPS), default="vht")
    parser.add_argument(
        "--skip-video", action="store_true", help="Đừng giải mã video (nhanh hơn)"
    )
    parser.add_argument(
        "--activity-only",
        action="store_true",
        help=(
            "Build an activity-classification index without requiring video or "
            "screen-clock files. This cannot produce pose-training labels."
        ),
    )
    return parser.parse_args()


def clock_summary(path: Path) -> dict:
    info = {"clock_rows": "", "clock_span_s": "", "clock_gaps": "", "clock_start_ns": ""}
    if not path.exists():
        return info
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return info
    host = [int(row["host_unix_ns"]) for row in rows]
    deltas = [(b - a) / 1e6 for a, b in zip(host, host[1:])]
    info["clock_rows"] = len(rows)
    info["clock_span_s"] = round((host[-1] - host[0]) / 1e9, 2)
    info["clock_start_ns"] = host[0]
    if deltas:
        median = statistics.median(deltas)
        info["clock_gaps"] = sum(1 for value in deltas if value > 3 * median)
    return info


def video_summary(path: Path | None, skip: bool) -> dict:
    info = {
        "video_name": path.name if path else "",
        "video_fps": "",
        "video_frames": "",
        "video_duration_s": "",
        "video_size": "",
        "video_decodes": "",
    }
    if not path or skip:
        return info
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        info["video_decodes"] = "NO"
        return info
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = capture.read()
    capture.release()
    info["video_decodes"] = "yes" if ok else "NO"
    info["video_fps"] = round(fps, 2) if fps else ""
    info["video_frames"] = frames
    info["video_duration_s"] = round(frames / fps, 2) if fps else ""
    return info


def find_video(folder: Path) -> Path | None:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    return None


def capture_session_fields(csi_meta: dict, notes: dict) -> tuple[str, str, str]:
    capture_id = csi_meta.get("trial", "")
    trial = notes.get("trial", "") or capture_id
    repeat = notes.get("repeat", "") or trial
    return capture_id, trial, repeat


def capture_session_id(notes: dict, capture_id: str, trial: str, activity: str) -> str:
    if notes.get("session_id"):
        return notes["session_id"]
    participant = notes.get("participant") or notes.get("participant_id") or "unknown"
    return f"{participant}_trial{trial}_{activity}_{capture_id or 'unknown'}"


def main() -> None:
    args = parse_args()
    csi_data = Path(args.csi_data)
    labeled = csi_data / "labeled"
    videos_root = csi_data / "video-sessions"
    if not labeled.is_dir() or not videos_root.is_dir():
        raise SystemExit(f"Không thấy {labeled} hoặc {videos_root}")

    # --- video side, keyed by trial id -----------------------------------
    video_sessions: dict[str, Path] = {}
    for folder in sorted(p for p in videos_root.iterdir() if p.is_dir()):
        notes = read_key_values(folder / "session-notes.txt")
        trial = notes.get("trial", "")
        if not trial or (
            not args.activity_only and find_video(folder) is None
        ):
            continue  # aggregate export folder, not a session
        video_sessions[trial] = folder

    spec = GROUPS[args.group]
    rows = []
    problems = []

    for csi_dir in sorted(p for p in labeled.iterdir() if p.is_dir()):
        csi_meta = read_key_values(csi_dir / "metadata.txt")
        notes = read_key_values(csi_dir / "video-session-notes.txt")
        capture_id, trial, repeat = capture_session_fields(csi_meta, notes)
        activity = notes.get("activity", "")
        participant = notes.get("participant") or notes.get("participant_id", "")

        required_csi_files = (
            ACTIVITY_CSI_FILES if args.activity_only else CSI_FILES
        )
        missing_csi = [
            name for name in required_csi_files if not (csi_dir / name).exists()
        ]
        video_folder = video_sessions.get(trial)
        video = find_video(video_folder) if video_folder else None

        csi_start = csi_end = frames_in_group = ""
        ts_first = ts_last = ""
        if (csi_dir / "recorder.log").exists():
            frames = parse_recorder_log(csi_dir / "recorder.log")
            kept = [
                frame
                for frame in frames
                if frame["rate"].startswith(spec["rate_prefix"])
                and (frame["nrx"], frame["ntx"], frame["ntone"])
                == (spec["nrx"], spec["ntx"], spec["ntone"])
            ]
            frames_in_group = len(kept)
            if kept:
                ts_first = kept[0]["ts"]
                ts_last = kept[-1]["ts"]
        if (csi_dir / "start-time.txt").exists():
            csi_start = (csi_dir / "start-time.txt").read_text().strip()
        if (csi_dir / "end-time.txt").exists():
            csi_end = (csi_dir / "end-time.txt").read_text().strip()

        row = {
            "session_id": capture_session_id(notes, capture_id, trial, activity),
            "recording_session_id": video_folder.name if video_folder else "",
            "activity": activity,
            "participant": participant,
            "repeat": repeat,
            "trial": trial,
            "capture_id": capture_id,
            "sync_status": notes.get("synchronization_status", ""),
            "csi_dir": str(csi_dir.resolve()),
            "video_file": str(video.resolve()) if video else "",
            "csi_start": csi_start,
            "csi_end": csi_end,
            "csi_frames_vht": frames_in_group,
            "csi_first_ts_ns": ts_first,
            "csi_last_ts_ns": ts_last,
            "csi_span_s": round((ts_last - ts_first) / 1e9, 2) if ts_first else "",
            **clock_summary(csi_dir / "screen-clock.csv"),
            **video_summary(video, args.skip_video),
            "missing_csi_files": ";".join(missing_csi),
        }
        rows.append(row)

        reasons = []
        if missing_csi:
            reasons.append(f"thiếu {', '.join(missing_csi)}")
        if video is None and not args.activity_only:
            reasons.append("không có video/không khớp trial")
        if activity not in SIX_ACTIVITIES:
            reasons.append(f"activity='{activity}' ngoài 6 hoạt động")
        if row["video_decodes"] == "NO":
            reasons.append("video không giải mã được")
        if not row["csi_frames_vht"]:
            reasons.append("không có frame VHT")
        if reasons:
            problems.append((video_folder.name if video_folder else csi_dir.name, reasons))

    if not rows:
        raise SystemExit("Không tìm thấy phiên nào.")

    print("=" * 96)
    print(
        "LẬP DANH SÁCH PHIÊN HOẠT ĐỘNG"
        if args.activity_only
        else "GHÉP PHIÊN CSI ↔ VIDEO"
    )
    print("=" * 96)
    header = (
        f"{'activity':<11}{'rep':<4}{'CSI':>6}{'video':>7}{'fps':>6}"
        f"{'CSI s':>7}{'vid s':>7}{'clock':>7}  session"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['activity']:<11}{row['repeat']:<4}"
            f"{row['csi_frames_vht']:>6}{row['video_decodes']:>7}"
            f"{str(row['video_fps']):>6}{str(row['csi_span_s']):>7}"
            f"{str(row['video_duration_s']):>7}{str(row['clock_rows']):>7}  "
            f"{row['session_id']}"
        )

    print()
    print("--- phủ 6 hoạt động ---")
    for activity in SIX_ACTIVITIES:
        count = sum(1 for row in rows if row["activity"] == activity)
        print(f"{activity:<12}{count} phiên")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"index: {output}")

    if problems:
        print()
        print("VẤN ĐỀ:")
        for name, reasons in problems:
            print(f"  - {name}: {'; '.join(reasons)}")
        raise SystemExit(1)
    if args.activity_only:
        print(
            "OK: các phiên CSI có metadata hoạt động hợp lệ. "
            "Video/nhãn khớp không được kiểm tra ở chế độ này."
        )
    else:
        print("OK: mọi phiên đủ file CSI, có video giải mã được, nhãn thuộc 6 hoạt động.")


if __name__ == "__main__":
    main()
