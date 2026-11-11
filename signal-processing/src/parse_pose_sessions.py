"""Parse the CSI of every video session into a labelled CSV.

Uses the session index written by ``build_pose_index.py`` (``data/pose/index.csv``).
The activity label comes from ``video-session-notes.txt`` because the CSI
recorder writes ``label=test`` for all of these captures.

Output:
    data/pose/raw_csv/<activity>_<repeat>.csv   9 columns, same schema as the
                                                rest of the project
    data/pose/_pose_metadata.csv               one row per session

Run from anywhere::

    python signal-processing/src/parse_pose_sessions.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_intel_csi import (  # noqa: E402
    GROUPS,
    convert_session,
    parse_wifi,
    ping_stats,
    write_session_csv,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT / "data" / "pose" / "index.csv"
DEFAULT_OUTPUT = PROJECT / "data" / "pose"


def resolve_indexed_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse pose-package CSI sessions")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group", choices=sorted(GROUPS), default="vht")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(
            f"Không thấy {index_path}. Chạy build_pose_index.py trước."
        )

    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))

    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "raw_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PARSE CSI CHO GÓI TƯ THẾ")
    print("=" * 78)

    metadata = []
    skipped_sessions = []
    for row in index_rows:
        csi_dir = resolve_indexed_path(row["csi_dir"])
        activity = row["activity"]
        repeat = row["repeat"]

        session, skip = convert_session(
            csi_dir,
            args.group,
            include_test=True,
            label_override=activity,
            trial_override=repeat,
        )
        if skip is not None:
            print(f"[SKIP] {skip['session']}: {skip['reason']}")
            skipped_sessions.append(
                f"{skip['session']}: {skip['reason']}"
            )
            continue

        session["file_name"] = f"{activity}_{repeat}.csv"
        session["scenario"] = activity
        rows_written = write_session_csv(csv_dir, session)

        wifi = parse_wifi(csi_dir / "wifi.txt")
        ping = ping_stats(csi_dir / "ping.log")
        metadata.append(
            {
                "session_id": row["session_id"],
                "activity": activity,
                "participant": row["participant"],
                "repeat": repeat,
                "trial": row["trial"],
                "csv_file": session["file_name"],
                "csi_dir": row["csi_dir"],
                "video_file": row["video_file"],
                "video_duration_s": row["video_duration_s"],
                "video_fps": row["video_fps"],
                "csi_start": row["csi_start"],
                "csi_end": row["csi_end"],
                "csi_span_s": row["csi_span_s"],
                "csi_first_ts_ns": row["csi_first_ts_ns"],
                "csi_last_ts_ns": row["csi_last_ts_ns"],
                "sync_status": row["sync_status"],
                "frames_kept": session["frames_kept"],
                "frames_total": session["frames_total"],
                "frames_filtered_out": (
                    session["frames_total"] - session["frames_kept"]
                ),
                "rows_written": rows_written,
                "timestamp_backwards": session["timestamp_backwards"],
                "ap_ssid": wifi.get("ssid", ""),
                "channel": wifi.get("channel", ""),
                "width_mhz": wifi.get("width_mhz", ""),
                "signal_dbm": wifi.get("signal_dbm", ""),
                "ping_replies": ping["ping_replies"],
                "ping_rtt_avg_ms": ping["ping_rtt_avg_ms"],
            }
        )
        print(
            f"[OK] {session['file_name']:<20} frames {session['frames_kept']:>4}"
            f"/{session['frames_total']:<5} rows {rows_written:>6} "
            f"ts_backwards={session['timestamp_backwards']}"
        )

    if skipped_sessions:
        details = "\n  - ".join(skipped_sessions)
        raise SystemExit(
            "Không hoàn tất parse: có phiên CSI bị bỏ qua. "
            "Dừng để tránh huấn luyện trên feature cũ hoặc thiếu phiên:\n"
            f"  - {details}"
        )
    if not metadata:
        raise SystemExit("Không parse được phiên nào.")

    meta_path = output_dir / "_pose_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)

    print()
    print(f"sessions: {len(metadata)}")
    print(f"rows    : {sum(row['rows_written'] for row in metadata):,}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
