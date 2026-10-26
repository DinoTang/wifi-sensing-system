"""Convert the Intel/`iwlwifi` CSI capture sessions into labeled CSV files.

Input  : hardware/csi_extracted/labeled/<session>/
           - raw.csi        binary records
           - recorder.log   per frame metadata (csi_len, rssi, seq, ts, mac, rate_bw)
           - metadata.txt   label, trial, duration_seconds, recorder_exit_code
           - start-time.txt / end-time.txt / wifi.txt / ping.log

Output : signal-processing/data/intel_raw/
           - <scenario>_<trial>.csv     9 columns required by csi_preprocessing_pipeline
           - _sessions_metadata.csv     one row per session (metadata deliverable)

Binary record layout (verified against recorder.log on every session):
    [uint32 big-endian: 276 + csi_len] [276 byte header] [4 byte] [csi_len byte payload]

Payload -> complex int16 little-endian.
    VHT  csi_len=896 -> 224 points = 4 blocks x 56 tones  (nrx,ntx,ntone)=(2,2,56)
    NOHT csi_len=208 ->  52 points                        (nrx,ntx,ntone)=(1,1,52)

Inside every 56 tone block the positions 7, 21, 34, 48 are always exactly zero in
all frames (the four pilot tones are not reported), so effective usable tones = 52.

Run from anywhere:
    python signal-processing/src/parse_intel_csi.py
    python signal-processing/src/parse_intel_csi.py --group noht
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
PROJECT = Path(__file__).resolve().parents[1]       # signal-processing/
DEFAULT_INPUT = ROOT / "hardware" / "csi_extracted" / "labeled"
DEFAULT_OUTPUT = PROJECT / "data" / "intel_raw"

CSV_COLUMNS = [
    "frame_id",
    "timestamp_ms",
    "rssi",
    "len",
    "first_word",
    "sample_index",
    "i",
    "q",
    "amplitude",
]

# metadata.txt value -> scenario name used for the output file name.
SCENARIO_FILE_STEM = {
    "empty": "room_empty",
    "seated": "person_static",
    "walking": "person_moving",
    "multiple": "multiple_people",
}

# Expected payload length for each capture group.
GROUPS = {
    "vht": {"csi_len": 896, "rate_prefix": "VHT", "nrx": 2, "ntx": 2, "ntone": 56},
    "noht": {"csi_len": 208, "rate_prefix": "NOHT", "nrx": 1, "ntx": 1, "ntone": 52},
}

# Columns that can never be derived from the capture files; they are
# typed in by the person who recorded the session.
HAND_FIELDS = ["distance_m", "position", "environment", "notes"]

PLACEHOLDER = "chưa ghi nhận"

META_RE = re.compile(
    r"csi_len\((?P<csi_len>\d+)\)[^\n]*\(nrx,ntx,ntone\)="
    r"\((?P<nrx>\d+),(?P<ntx>\d+),(?P<ntone>\d+)\)\s*[\r\n]+"
    r"rssi\((?P<rssi>[^)]*)\)\s+seq\((?P<seq>\d+)\)\s+us\((?P<us>\d+)\)"
    r"[^\n]*ts\((?P<ts>\d+)\)\s*[\r\n]+"
    r"mac\((?P<mac>[0-9a-fA-F:]+)\)\s*[\r\n]+"
    r"rate_bw\((?P<rate>[^\n)]*)\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Intel raw.csi sessions to labeled CSI CSV files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder containing one sub folder per capture session",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Folder for the generated CSV files",
    )
    parser.add_argument(
        "--group",
        choices=sorted(GROUPS),
        default="vht",
        help="Keep only this PHY group (vht = 2x2x56/896B, noht = 1x1x52/208B)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also convert sessions whose metadata label is 'test'",
    )
    return parser.parse_args()


def parse_raw_payloads(path: Path) -> list[bytes]:
    """Split raw.csi into one CSI payload per record."""
    blob = path.read_bytes()
    payloads: list[bytes] = []
    offset = 0
    while offset < len(blob):
        if offset + 4 > len(blob):
            raise ValueError(f"{path.name}: truncated record length at byte {offset}")
        (record_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4
        if record_len < 276 or offset + record_len > len(blob):
            raise ValueError(
                f"{path.name}: invalid record length {record_len} at byte {offset - 4}"
            )
        offset += 276
        data_block_len = record_len - 276
        payload = blob[offset + 4:offset + data_block_len]
        offset += data_block_len
        payloads.append(payload)
    if offset != len(blob):
        raise ValueError(f"{path.name}: {len(blob) - offset} trailing bytes")
    return payloads


def parse_recorder_log(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    frames = []
    for match in META_RE.finditer(text):
        frame = match.groupdict()
        frame["csi_len"] = int(frame["csi_len"])
        frame["nrx"] = int(frame["nrx"])
        frame["ntx"] = int(frame["ntx"])
        frame["ntone"] = int(frame["ntone"])
        frame["seq"] = int(frame["seq"])
        frame["us"] = int(frame["us"])
        frame["ts"] = int(frame["ts"])
        frame["rssi_first"] = int(frame["rssi"].split(",")[0])
        frame["rate"] = frame["rate"].strip()
        frames.append(frame)
    return frames


def read_key_values(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_wifi(path: Path) -> dict:
    info = {}
    if not path.exists():
        return info
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern, key in (
        (r"SSID:\s*(\S+)", "ssid"),
        (r"freq:\s*(\d+)", "freq_mhz"),
        (r"signal:\s*(-?\d+)", "signal_dbm"),
        (r"channel\s+(\d+)\s+\(\d+ MHz\)", "channel"),
        (r"width:\s+(\d+) MHz", "width_mhz"),
        (r"rx bitrate:\s*([^\n]+)", "rx_bitrate"),
    ):
        found = re.search(pattern, text)
        if found:
            info[key] = found.group(1).strip() if found.groups() else found.group(0)
    return info


def ping_stats(path: Path) -> dict:
    info = {"ping_replies": 0, "ping_rtt_avg_ms": ""}
    if not path.exists():
        return info
    text = path.read_text(encoding="utf-8", errors="replace")
    info["ping_replies"] = text.count("64 bytes from")
    found = re.search(r"min/avg/max/mdev = [\d.]+/([\d.]+)/", text)
    if found:
        info["ping_rtt_avg_ms"] = found.group(1)
    return info


def decode_points(payload: bytes) -> list[tuple[int, int]]:
    values = struct.unpack(f"<{len(payload) // 2}h", payload)
    return list(zip(values[0::2], values[1::2]))


def convert_session(session_dir: Path, group: str, include_test: bool):
    spec = GROUPS[group]
    meta_info = read_key_values(session_dir / "metadata.txt")
    label = meta_info.get("label", "")
    trial = meta_info.get("trial", "")
    if not label or not trial:
        raise ValueError(f"{session_dir.name}: metadata.txt missing label/trial")
    if label == "test" and not include_test:
        return None, {"session": session_dir.name, "reason": "label=test, skipped"}

    payload_records = parse_raw_payloads(session_dir / "raw.csi")
    frames = parse_recorder_log(session_dir / "recorder.log")
    if len(payload_records) != len(frames):
        raise ValueError(
            f"{session_dir.name}: raw.csi has {len(payload_records)} records but "
            f"recorder.log describes {len(frames)}"
        )

    timestamp_backwards = 0
    kept = []
    for record_index, (payload, frame) in enumerate(
        zip(payload_records, frames), start=1
    ):
        if len(payload) != frame["csi_len"]:
            raise ValueError(
                f"{session_dir.name}: record {record_index} payload "
                f"{len(payload)} != csi_len {frame['csi_len']}"
            )
        if not frame["rate"].startswith(spec["rate_prefix"]):
            continue
        if (
            frame["nrx"],
            frame["ntx"],
            frame["ntone"],
        ) != (spec["nrx"], spec["ntx"], spec["ntone"]):
            raise ValueError(
                f"{session_dir.name}: record {record_index} unexpected shape "
                f"{(frame['nrx'], frame['ntx'], frame['ntone'])}"
            )
        kept.append((record_index, frame, decode_points(payload)))

    for (_, frame_a, _), (_, frame_b, _) in zip(kept, kept[1:]):
        if frame_b["ts"] <= frame_a["ts"]:
            timestamp_backwards += 1

    scenario = SCENARIO_FILE_STEM.get(label, label)
    file_name = f"{scenario}_{trial}.csv"

    points_per_frame = spec["nrx"] * spec["ntx"] * spec["ntone"]
    return {
        "file_name": file_name,
        "scenario": scenario,
        "label_raw": label,
        "trial": trial,
        "session_dir": session_dir.name,
        "meta_info": meta_info,
        "kept": kept,
        "frames_total": len(frames),
        "frames_kept": len(kept),
        "points_per_frame": points_per_frame,
        "timestamp_backwards": timestamp_backwards,
        "group": group,
        "spec": spec,
        "wifi": parse_wifi(session_dir / "wifi.txt"),
        "ping": ping_stats(session_dir / "ping.log"),
    }, None


def write_session_csv(output_dir: Path, session: dict) -> int:
    out_path = output_dir / session["file_name"]
    count = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for record_index, frame, points in session["kept"]:
            for sample_index, (real, imag) in enumerate(points):
                amplitude = (real * real + imag * imag) ** 0.5
                writer.writerow(
                    [
                        record_index,
                        frame["ts"] // 1_000_000,
                        frame["rssi_first"],
                        frame["csi_len"],
                        0,
                        sample_index,
                        real,
                        imag,
                        f"{amplitude:.4f}",
                    ]
                )
                count += 1
    return count


def load_previous_metadata(path: Path) -> dict[str, dict[str, str]]:
    """
    Read the hand-filled metadata columns of an existing metadata file.

    Rerunning the parser must not wipe what the recording person typed
    in. Only HAND_FIELDS are carried over, keyed by csv_file, and the
    placeholder value is ignored.
    """

    if not path.exists():
        return {}

    carried: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            file_name = (row.get("csv_file") or "").strip()
            if not file_name:
                continue
            values = {
                field: (row.get(field) or "").strip()
                for field in HAND_FIELDS
                if (row.get(field) or "").strip()
                and (row.get(field) or "").strip() != PLACEHOLDER
            }
            if values:
                carried[file_name] = values
    return carried


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = GROUPS[args.group]
    print("=" * 70)
    print("INTEL CSI -> CSV")
    print("=" * 70)
    print(f"input   : {input_dir}")
    print(f"output  : {output_dir}")
    print(f"group   : {args.group}  ({spec['nrx']}x{spec['ntx']}x{spec['ntone']}, "
          f"{spec['csi_len']} bytes)")
    print(f"points per frame: {spec['nrx'] * spec['ntx'] * spec['ntone']}")
    print()

    metadata_rows = []
    skipped = []
    for session_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        session, skip = convert_session(session_dir, args.group, args.include_test)
        if skip is not None:
            skipped.append(skip)
            print(f"[SKIP] {skip['session']}: {skip['reason']}")
            continue

        session["rows_written"] = write_session_csv(output_dir, session)
        duration_ms = 0
        if session["kept"]:
            duration_ms = (
                session["kept"][-1][1]["ts"] - session["kept"][0][1]["ts"]
            ) // 1_000_000

        start_text = (session_dir / "start-time.txt").read_text().strip()
        end_text = (session_dir / "end-time.txt").read_text().strip()
        wifi = session["wifi"]
        ping = session["ping"]
        meta_info = session["meta_info"]

        metadata_rows.append(
            {
                "scenario": session["scenario"],
                "trial": session["trial"],
                "label_raw": session["label_raw"],
                "csv_file": session["file_name"],
                "source_session_dir": session["session_dir"],
                "start_time": start_text,
                "end_time": end_text,
                "duration_seconds_declared": meta_info.get("duration_seconds", ""),
                "duration_seconds_from_csi": round(duration_ms / 1000, 1),
                "recorder_exit_code": meta_info.get("recorder_exit_code", ""),
                "num_people": 0 if session["scenario"] == "room_empty" else 1,
                "capture_group": session["group"],
                "phy_rate": args.group.upper(),
                "nrx": session["spec"]["nrx"],
                "ntx": session["spec"]["ntx"],
                "ntone": session["spec"]["ntone"],
                "csi_len_bytes": session["spec"]["csi_len"],
                "sample_count_per_frame": session["points_per_frame"],
                "ap_ssid": wifi.get("ssid", ""),
                "freq_mhz": wifi.get("freq_mhz", ""),
                "channel": wifi.get("channel", ""),
                "width_mhz": wifi.get("width_mhz", ""),
                "signal_dbm": wifi.get("signal_dbm", ""),
                "ping_replies": ping["ping_replies"],
                "ping_rtt_avg_ms": ping["ping_rtt_avg_ms"],
                "frames_total_in_recorder": session["frames_total"],
                "frames_kept": session["frames_kept"],
                "frames_filtered_out": (
                    session["frames_total"] - session["frames_kept"]
                ),
                "rows_written": session["rows_written"],
                "timestamp_backwards": session["timestamp_backwards"],
                "null_tone_indices": "7;21;34;48",
                # These four fields cannot be derived from the capture
                # files. They are filled in by the person who recorded
                # the session; leave an explicit marker instead of
                # blank. Values already typed in are carried over on
                # rerun (see load_previous_metadata).
                "distance_m": "chưa ghi nhận",
                "position": "chưa ghi nhận",
                "environment": "chưa ghi nhận",
                "notes": "chưa ghi nhận",
            }
        )
        print(
            f"[OK]   {session['file_name']:<28} "
            f"frames {session['frames_kept']:>4}/{session['frames_total']:<4} "
            f"rows {session['rows_written']:>6} "
            f"ts_backwards={session['timestamp_backwards']}"
        )

    if not metadata_rows:
        raise SystemExit("No session was converted.")

    meta_path = output_dir / "_sessions_metadata.csv"

    # Keep whatever the recording person already typed in.
    previous = load_previous_metadata(meta_path)
    carried_over = 0
    for row in metadata_rows:
        values = previous.get(row["csv_file"])
        if not values:
            continue
        for field, value in values.items():
            row[field] = value
        carried_over += 1

    if carried_over:
        print()
        print(
            f"[METADATA] carried over hand-filled fields for "
            f"{carried_over} session(s) from {meta_path.name}."
        )

    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    total_rows = sum(row["rows_written"] for row in metadata_rows)
    print()
    print(f"sessions : {len(metadata_rows)}")
    print(f"rows     : {total_rows:,}")
    print(f"metadata : {meta_path}")
    print(f"skipped  : {len(skipped)}")
    print()
    remaining = sum(
        1
        for row in metadata_rows
        for field in HAND_FIELDS
        if row[field] == PLACEHOLDER
    )
    if remaining:
        print(
            f"[METADATA] {remaining} field(s) still '{PLACEHOLDER}' — "
            "fill distance_m, position, environment, notes by hand."
        )
    else:
        print("[METADATA] all hand-filled fields are complete.")


if __name__ == "__main__":
    main()
