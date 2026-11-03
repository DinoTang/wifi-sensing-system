"""Audit the video+CSI session bundles of the pose work package.

A bundle is one folder such as::

    P01_left_arm_01_20261103-025544_ttVs/
        session-notes.txt      session_id, participant, activity, repeat, sync status
        screen-clock.csv       host clock log of the CSI machine
        csi-folder.txt         where the CSI folder lived on the capturing machine
        *.mov / *.mp4          the recording
        csi/                   optional: the copied CSI folder (raw.csi, recorder.log, ...)

The script answers three questions without guessing:

1. Which bundles exist and which activity/repeat each one is?
2. Is the CSI of that session actually **in the bundle** (not just a path to
   another machine)?
3. Are the sync inputs usable (video decodable, clock log clean, status)?

Run from anywhere::

    python signal-processing/src/check_pose_sessions.py
    python signal-processing/src/check_pose_sessions.py --skip-video
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_ROOT = PROJECT
DEFAULT_OUTPUT = PROJECT / "data" / "pose_manifest.csv"

SIX_ACTIVITIES = ["empty", "standing", "left_arm", "right_arm", "walking", "mixed"]

VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".m4v"}

# Files the CSI side must provide for one session.
CSI_FILES = [
    "raw.csi",
    "recorder.log",
    "wifi.txt",
    "metadata.txt",
    "start-time.txt",
    "end-time.txt",
]

# Files the video side must provide for one session.
VIDEO_FILES = ["session-notes.txt", "screen-clock.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check video+CSI session bundles before pose processing"
    )
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Do not decode the video (faster, but duration/fps are left empty)",
    )
    parser.add_argument(
        "--aggregate-prefix",
        default="P01_all_cases",
        help="Folders starting with this prefix are aggregate exports, not sessions",
    )
    return parser.parse_args()


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def find_video(bundle: Path) -> Path | None:
    for path in sorted(bundle.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    return None


def clock_stats(path: Path) -> dict:
    info = {
        "clock_rows": "",
        "clock_span_s": "",
        "clock_interval_ms": "",
        "clock_gaps": "",
        "clock_backwards": "",
    }
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
    if deltas:
        median = statistics.median(deltas)
        info["clock_interval_ms"] = round(median, 2)
        info["clock_gaps"] = sum(1 for value in deltas if value > 3 * median)
        info["clock_backwards"] = sum(1 for value in deltas if value <= 0)
    return info


def video_stats(path: Path | None, skip: bool) -> dict:
    info = {
        "video_file": path.name if path else "",
        "video_mb": round(path.stat().st_size / 1e6, 1) if path else "",
        "video_fps": "",
        "video_frames": "",
        "video_duration_s": "",
        "video_size": "",
        "video_decodes": "",
    }
    if not path or skip:
        return info

    import cv2  # imported lazily so --skip-video works without OpenCV

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        info["video_decodes"] = "NO"
        return info
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ok, _ = capture.read()
    capture.release()

    info["video_decodes"] = "yes" if ok else "NO"
    info["video_fps"] = round(fps, 2) if fps else ""
    info["video_frames"] = frames
    info["video_duration_s"] = round(frames / fps, 2) if fps else ""
    info["video_size"] = f"{width:.0f}x{height:.0f}"
    return info


def inspect_bundle(bundle: Path, skip_video: bool) -> dict:
    notes = read_key_values(bundle / "session-notes.txt")
    pointer = ""
    if (bundle / "csi-folder.txt").exists():
        pointer = (bundle / "csi-folder.txt").read_text(
            encoding="utf-8", errors="replace"
        ).strip()

    # The CSI folder counts as available only if it is inside the bundle.
    csi_dir = bundle / "csi"
    if not csi_dir.is_dir() and (bundle / "raw.csi").exists():
        csi_dir = bundle
    csi_present = {name: (csi_dir / name).exists() for name in CSI_FILES}
    missing_csi = [name for name in CSI_FILES if not csi_present[name]]

    video_present = {
        name: (bundle / name).exists() for name in VIDEO_FILES
    }
    video = find_video(bundle)

    row = {
        "bundle": bundle.name,
        "activity": notes.get("activity", ""),
        "participant": notes.get("participant_id", ""),
        "repeat": notes.get("repeat", ""),
        "declared_duration_s": notes.get("duration_seconds", ""),
        "sync_status": notes.get("synchronization_status", "chưa ghi"),
        "csi_local": "yes" if csi_dir.is_dir() else "NO",
        "csi_source_pointer": pointer,
        "missing_csi_files": ";".join(missing_csi),
        "has_notes": "yes" if (bundle / "session-notes.txt").exists() else "NO",
        "has_screen_clock": "yes"
        if video_present["screen-clock.csv"]
        else "NO",
        **clock_stats(bundle / "screen-clock.csv"),
        **video_stats(video, skip_video),
    }
    return row


def main() -> None:
    args = parse_args()
    root = Path(args.bundle_root)
    if not root.is_dir():
        raise SystemExit(f"Bundle root not found: {root}")

    bundles = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"src", "data", "docs", "tests", "results"}
        and not path.name.startswith(args.aggregate_prefix)
        and (path / "session-notes.txt").exists()
    ]
    if not bundles:
        raise SystemExit(
            f"No session bundle (folder with session-notes.txt) inside {root}"
        )

    rows = [inspect_bundle(bundle, args.skip_video) for bundle in bundles]

    print("=" * 92)
    print("POSE SESSION BUNDLES")
    print("=" * 92)
    header = (
        f"{'activity':<11}{'rep':<4}{'csi':<5}{'video':<9}"
        f"{'fps':>6}{'dur':>7}{'clock':>7}{'gaps':>5}  sync"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['activity']:<11}{row['repeat']:<4}{row['csi_local']:<5}"
            f"{row['video_decodes']:<9}"
            f"{str(row['video_fps']):>6}{str(row['video_duration_s']):>7}"
            f"{str(row['clock_rows']):>7}{str(row['clock_gaps']):>5}  "
            f"{row['sync_status']}"
        )

    print()
    print("--- CSI của từng phiên ---")
    no_csi = [row for row in rows if row["csi_local"] == "NO"]
    if no_csi:
        print(
            f"{len(no_csi)}/{len(rows)} phiên KHÔNG có thư mục CSI trong bundle."
        )
        for row in no_csi:
            print(f"  - {row['bundle']}")
            if row["csi_source_pointer"]:
                print(f"      cần copy từ: {row['csi_source_pointer']}")
    else:
        missing = [row for row in rows if row["missing_csi_files"]]
        if missing:
            for row in missing:
                print(f"  - {row['bundle']}: thiếu {row['missing_csi_files']}")
        else:
            print("tất cả phiên đều có đủ file CSI trong bundle")

    print()
    print("--- Phủ 6 hoạt động và số lần lặp ---")
    print(f"{'activity':<12}{'repeats':>8}{'bundles':>9}")
    for activity in SIX_ACTIVITIES:
        matching = [row for row in rows if row["activity"] == activity]
        print(f"{activity:<12}{len(matching):>8}{len(matching):>9}")
    unknown = [row for row in rows if row["activity"] not in SIX_ACTIVITIES]
    if unknown:
        print(f"nhãn ngoài 6 hoạt động: {', '.join(r['activity'] for r in unknown)}")

    print()
    print("--- Điều kiện đồng bộ thời gian ---")
    unverified = [row for row in rows if row["sync_status"] != "verified"]
    print(
        "synchronization_status=verified: "
        f"{len(rows) - len(unverified)}/{len(rows)} phiên"
    )
    bad_clock = [
        row
        for row in rows
        if row["has_screen_clock"] != "yes"
        or (isinstance(row["clock_gaps"], int) and row["clock_gaps"])
        or (isinstance(row["clock_backwards"], int) and row["clock_backwards"])
    ]
    if bad_clock:
        print("phiên có log đồng hồ thiếu/lỗi: " + ", ".join(r["bundle"] for r in bad_clock))
    else:
        print("log đồng hồ: không phiên nào có khoảng trống hay timestamp lùi")
    print(
        "lưu ý: script này KHÔNG kiểm tra được đồng hồ có hiện trong khung hình hay "
        "không — phải xem video; thiếu đồng hồ trong hình thì không đồng bộ được."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"manifest: {output}")

    # Verdict: a session is usable only with CSI locally + video + clock.
    broken = [
        row
        for row in rows
        if row["csi_local"] == "NO"
        or row["missing_csi_files"]
        or row["has_screen_clock"] != "yes"
        or row["video_decodes"] == "NO"
    ]
    print()
    if broken:
        print(
            f"CHƯA ĐỦ: {len(broken)}/{len(rows)} phiên thiếu dữ liệu "
            "(CSI local, video hoặc log đồng hồ)."
        )
        raise SystemExit(1)
    print("ĐỦ DỮ LIỆU: mọi phiên có CSI local + video + log đồng hồ.")


if __name__ == "__main__":
    main()
