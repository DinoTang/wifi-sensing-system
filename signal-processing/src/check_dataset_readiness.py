"""Audit a capture folder before it is parsed and handed to Người 3.

Implements the first two steps of the task sheet:

    Bước 1 — nhận và sắp xếp dữ liệu
    Bước 2 — đọc và kiểm tra CSI

For each session folder it checks

* which capture files exist (and which are missing),
* how many records ``raw.csi`` holds versus how many ``recorder.log`` describes,
* which CSI configuration(s) appear inside one session (rate / antenna /
  subcarrier / payload length) — mixed configurations must be split, never
  concatenated into one series,
* the actual sampling interval and how many timestamps go backwards,
* whether the session is one of the six activities of the task sheet, and
  whether the video-side inputs exist.

Nothing is modified: this script only reads and writes one manifest CSV.

Run from anywhere:

    python signal-processing/src/check_dataset_readiness.py
    python signal-processing/src/check_dataset_readiness.py --strict
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_intel_csi import (  # noqa: E402  (path setup above)
    GROUPS,
    parse_raw_payloads,
    parse_recorder_log,
    read_key_values,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "hardware" / "csi_extracted" / "labeled"
DEFAULT_OUTPUT = ROOT / "signal-processing" / "data" / "intel_raw"

# Files every capture session must contain (delivered by Người 1).
REQUIRED_FILES = [
    "raw.csi",
    "recorder.log",
    "metadata.txt",
    "start-time.txt",
    "end-time.txt",
    "wifi.txt",
    "ping.log",
]

# Extra files needed by the pose/video work package (steps 3, 5, 6).
POSE_FILES = ["screen-clock.csv"]

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

# The six activities named in the task sheet. ``seated`` is deliberately not
# mapped to ``standing``: sitting and standing are different postures and the
# recording notes say the person was seated.
SIX_ACTIVITIES = ["empty", "standing", "left_arm", "right_arm", "walking", "mixed"]

LABEL_TO_ACTIVITY = {
    "empty": "empty",
    "standing": "standing",
    "left_arm": "left_arm",
    "right_arm": "right_arm",
    "walking": "walking",
    "mixed": "mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check capture sessions for missing files and CSI configuration"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--group",
        choices=sorted(GROUPS),
        default="vht",
        help="CSI group used to count usable frames (vht = 2x2x56/896B)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when a required file or a required activity is missing",
    )
    return parser.parse_args()


def find_video(session_dir: Path, input_dir: Path) -> Path | None:
    """Look for the session video in the session folder, then in ../videos/."""
    for folder in (session_dir, input_dir.parent / "videos"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                return path
    return None


def find_notes(session_dir: Path) -> Path | None:
    for path in sorted(session_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "note" in name or "ghi-chu" in name or "ghichu" in name:
            return path
    return None


def inspect_session(session_dir: Path, input_dir: Path, group: str) -> dict:
    spec = GROUPS[group]
    expected_shape = (spec["nrx"], spec["ntx"], spec["ntone"])
    expected_len = spec["csi_len"]

    present = {name: (session_dir / name).exists() for name in REQUIRED_FILES}
    pose_present = {name: (session_dir / name).exists() for name in POSE_FILES}
    video = find_video(session_dir, input_dir)
    notes = find_notes(session_dir)

    meta = read_key_values(session_dir / "metadata.txt")
    label = meta.get("label", "")
    activity = LABEL_TO_ACTIVITY.get(label, "")

    records_raw = ""
    records_log = ""
    frames_in_group = 0
    interval_ms = ""
    fps = ""
    ts_backwards = ""
    gap_count = ""
    gap_max_ms = ""
    config_signatures: list[str] = []
    groups_seen: list[str] = []

    if present["raw.csi"]:
        payloads = parse_raw_payloads(session_dir / "raw.csi")
        records_raw = len(payloads)

    if present["recorder.log"]:
        frames = parse_recorder_log(session_dir / "recorder.log")
        records_log = len(frames)

        for frame in frames:
            prefix = frame["rate"].split()[0]
            if prefix not in groups_seen:
                groups_seen.append(prefix)

        kept = [
            frame
            for frame in frames
            if frame["rate"].startswith(spec["rate_prefix"])
            and (frame["nrx"], frame["ntx"], frame["ntone"]) == expected_shape
            and frame["csi_len"] == expected_len
        ]
        frames_in_group = len(kept)
        timestamps = [frame["ts"] for frame in kept]

        # Configuration variants *inside the selected group*: those are the
        # ones that would corrupt a series if they were glued together.
        for frame in kept:
            signature = (
                f"{frame['nrx']}x{frame['ntx']}x{frame['ntone']}|"
                f"{frame['csi_len']}B|"
                f"{frame['rate']}"
            )
            if signature not in config_signatures:
                config_signatures.append(signature)

        if len(timestamps) > 1:
            deltas = [
                (b - a) / 1_000_000
                for a, b in zip(timestamps, timestamps[1:])
                if b > a
            ]
            if deltas:
                median = statistics.median(deltas)
                interval_ms = round(median, 2)
                fps = round(1000 / median, 2) if median else ""
                gaps = [gap for gap in deltas if gap > 3 * median]
                gap_count = len(gaps)
                gap_max_ms = round(max(gaps), 1) if gaps else 0.0
        ts_backwards = sum(
            1 for a, b in zip(timestamps, timestamps[1:]) if b <= a
        )

    missing_required = [name for name in REQUIRED_FILES if not present[name]]
    missing_pose = [name for name in POSE_FILES if not pose_present[name]]
    if video is None:
        missing_pose.append("video")
    if notes is None:
        missing_pose.append("notes")

    return {
        "session_dir": session_dir.name,
        "activity": activity,
        "activity_in_task_sheet": "yes" if activity in SIX_ACTIVITIES else "no",
        "label_raw": label,
        "trial": meta.get("trial", ""),
        "declared_duration_s": meta.get("duration_seconds", ""),
        "records_raw": records_raw,
        "records_log": records_log,
        "records_match": (
            ""
            if records_raw == "" or records_log == ""
            else "yes" if records_raw == records_log else "NO"
        ),
        "frames_in_group": frames_in_group,
        "sampling_interval_ms": interval_ms,
        "sampling_fps": fps,
        "timestamp_backwards": ts_backwards,
        "gap_count": gap_count,
        "gap_max_ms": gap_max_ms,
        "config_count": len(config_signatures),
        "config_signatures": " ; ".join(config_signatures),
        "groups_seen": ";".join(groups_seen),
        **{f"has_{name.replace('.', '_')}": "yes" if ok else "no" for name, ok in present.items()},
        "has_video": "yes" if video else "no",
        "video_file": video.name if video else "",
        "has_notes": "yes" if notes else "no",
        "missing_required": ";".join(missing_required),
        "missing_pose_inputs": ";".join(missing_pose),
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'session':<34}{'label':<9}{'act':<10}"
        f"{'rec/rec':>9}{'frame':>7}{'ms':>7}{'fps':>6}{'gap':>5}{'cfg':>4}  miss"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['session_dir']:<34}{row['label_raw']:<9}{row['activity'] or '-':<10}"
            f"{str(row['records_raw']) + '/' + str(row['records_log']):>9}"
            f"{row['frames_in_group']:>7}"
            f"{str(row['sampling_interval_ms']):>7}"
            f"{str(row['sampling_fps']):>6}"
            f"{str(row['gap_count']):>5}"
            f"{row['config_count']:>4}  {row['missing_required'] or '-'}"
        )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    session_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not session_dirs:
        raise SystemExit(f"No session folder inside {input_dir}")

    rows = [inspect_session(path, input_dir, args.group) for path in session_dirs]

    print("=" * 78)
    print("DATASET READINESS CHECK")
    print("=" * 78)
    print(f"input : {input_dir}")
    print(f"group : {args.group} ({GROUPS[args.group]['nrx']}x"
          f"{GROUPS[args.group]['ntx']}x{GROUPS[args.group]['ntone']}, "
          f"{GROUPS[args.group]['csi_len']} bytes)")
    print(f"sessions: {len(rows)}")
    print()
    print_table(rows)

    # ---- CSI consistency checks (Bước 2) -------------------------------
    print()
    print("--- KIỂM TRA CSI ---")
    mismatched = [r["session_dir"] for r in rows if r["records_match"] == "NO"]
    print(
        f"raw.csi vs recorder.log: "
        + ("khớp ở mọi phiên" if not mismatched else f"LỆCH: {', '.join(mismatched)}")
    )

    mixed = [r for r in rows if r["config_count"] > 1]
    if mixed:
        print("phiên có >1 cấu hình CSI trong cùng nhóm (phải tách trước khi xử lý):")
        for row in mixed:
            print(f"  - {row['session_dir']}: {row['config_signatures']}")
    else:
        print(
            f"trong nhóm {args.group}: mỗi phiên chỉ có 1 cấu hình CSI "
            "(an toàn để ghép chuỗi)"
        )

    group_sets = sorted({r["groups_seen"] for r in rows if r["groups_seen"]})
    print(f"nhóm rate xuất hiện trong các phiên: {' | '.join(group_sets)}")
    print(
        "  -> các nhóm khác nhau bị parser tách bằng --group; "
        "bản ghi của nhóm không chọn nằm trong _rejected_records.csv"
    )

    backwards = [r["session_dir"] for r in rows if r["timestamp_backwards"]]
    print(
        "timestamp không tăng: "
        + ("không phiên nào" if not backwards else ", ".join(backwards))
    )

    gapped = [r for r in rows if isinstance(r["gap_count"], int) and r["gap_count"]]
    if gapped:
        print(f"phiên có khoảng thiếu dữ liệu (>3x nhịp trung vị): {len(gapped)}")
        for row in gapped:
            print(
                f"  - {row['session_dir']}: {row['gap_count']} khoảng, "
                f"dài nhất {row['gap_max_ms']} ms"
            )
    else:
        print("không có khoảng thiếu dữ liệu bất thường (>3x nhịp trung vị)")

    empty_group = [r["session_dir"] for r in rows if r["frames_in_group"] == 0]
    if empty_group:
        print(f"phiên không có frame thuộc nhóm {args.group}: {', '.join(empty_group)}")

    # ---- Coverage of the six activities --------------------------------
    print()
    print("--- PHỦ 6 HOẠT ĐỘNG (yêu cầu của gói việc) ---")
    print(f"{'activity':<12}{'sessions':>9}{'frames':>9}  trạng thái")
    covered = 0
    for activity in SIX_ACTIVITIES:
        matching = [r for r in rows if r["activity"] == activity]
        frames = sum(r["frames_in_group"] for r in matching)
        if matching:
            covered += 1
        status = "có" if matching else "THIẾU"
        print(f"{activity:<12}{len(matching):>9}{frames:>9}  {status}")

    other = [r["session_dir"] for r in rows if not r["activity"]]
    if other:
        labels = sorted({r["label_raw"] for r in rows if not r["activity"]})
        print(
            f"\nNgoài 6 hoạt động: {len(other)} phiên (nhãn {', '.join(labels)}) — "
            "giữ được cho mục tiêu 3 lớp, KHÔNG dùng thay cho hoạt động còn thiếu."
        )

    # ---- Video-side inputs ---------------------------------------------
    print()
    print("--- ĐẦU VÀO PHÍA VIDEO (bước 3, 5, 6) ---")
    with_video = [r for r in rows if r["has_video"] == "yes"]
    clock = [
        r["session_dir"]
        for r in rows
        if "screen-clock.csv" not in r["missing_pose_inputs"]
    ]
    print(f"phiên có video          : {len(with_video)}/{len(rows)}")
    print(f"phiên có screen-clock   : {len(clock)}/{len(rows)}")
    print(
        "phiên có ghi chú riêng  : "
        f"{sum(1 for r in rows if r['has_notes'] == 'yes')}/{len(rows)}"
    )

    # ---- Write the manifest --------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "_session_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"manifest: {manifest_path}")

    # ---- Verdict -------------------------------------------------------
    missing_files = sum(1 for r in rows if r["missing_required"])
    problems = []
    if missing_files:
        problems.append(f"{missing_files} phiên thiếu file bắt buộc")
    if covered < len(SIX_ACTIVITIES):
        problems.append(
            f"{len(SIX_ACTIVITIES) - covered}/{len(SIX_ACTIVITIES)} hoạt động chưa có dữ liệu "
            f"({', '.join(a for a in SIX_ACTIVITIES if not any(r['activity'] == a for r in rows))})"
        )
    if not with_video:
        problems.append("chưa có video nên bước 3/5/6 chưa chạy được")

    print()
    if problems:
        print("CHƯA SẴN SÀNG: " + "; ".join(problems))
    else:
        print("SẴN SÀNG: đủ file, đủ 6 hoạt động, có video và screen-clock.")

    if args.strict and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
