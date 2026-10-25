"""Cross-reference the 45 incomplete frames with the 5 raw serial logs.

Run from signal-processing/:  python analyze_45_frames.py
Read-only: does not modify any raw data.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("data/raw")
LOGS = Path("../hardware")

CSV_TO_LOG = {
    "nguoi_ngoi_im.csv": "csi_capture_20260925_202440.log",
    "phong_trong.csv": "csi_capture_20260925_203950.log",
    "nguoi_di_ra.csv": "csi_capture_20260925_212313.log",
    "nguoi_di_vao.csv": "csi_capture_20260925_213320.log",
}
ORPHAN_LOG = "csi_capture_20260925_204849.log"

LOG_LINE_RE = re.compile(
    r'^CSI_DATA,(\d+),(\d+),(-?\d+),(\d+),(\d+),"([0-9A-Fa-f]*)"\s*$'
)


def read_csv_frames(path: Path):
    frames = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = int(row["frame_id"])
            f = frames.setdefault(
                fid,
                {
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "rssi": int(row["rssi"]),
                    "len": int(row["len"]),
                    "first_word": row["first_word"],
                    "samples": {},
                },
            )
            si = int(row["sample_index"])
            f["samples"][si] = (int(row["i"]), int(row["q"]))
    return frames


def read_log(path: Path):
    """Return (frames, counters, malformed_lines, raw_line_by_frame)."""
    frames = {}
    counters = {}
    malformed = []
    stats = Counter()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            if line.startswith("CAPTURE,complete"):
                for kv in line.split(",")[1:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        counters[k] = v
                continue
            if not line.startswith("CSI_DATA,"):
                continue
            if line.startswith("CSI_DATA,frame_id"):
                continue
            m = LOG_LINE_RE.match(line)
            if not m:
                # header present but structure differs -> keep the raw text
                malformed.append((lineno, line[:120]))
                stats["malformed"] += 1
                continue
            fid, ts, rssi, ln, fw, hexdata = m.groups()
            fid = int(fid)
            stats["parsed"] += 1
            odd = len(hexdata) % 2
            frames[fid] = {
                "timestamp_ms": int(ts),
                "rssi": int(rssi),
                "len_field": int(ln),
                "first_word": int(fw),
                "hex_chars": len(hexdata),
                "hex_bytes": len(hexdata) // 2,
                "odd_hex": odd,
                "declared_ok": len(hexdata) == int(ln) * 2,
                "line_no": lineno,
                "lineno": lineno,
            }
    return frames, counters, malformed, stats


def runs(ids):
    ids = sorted(ids)
    out, start, prev = [], None, None
    for x in ids:
        if start is None:
            start = prev = x
            continue
        if x == prev + 1:
            prev = x
            continue
        out.append((start, prev))
        start = prev = x
    if start is not None:
        out.append((start, prev))
    return out


def fmt_runs(ids):
    return ", ".join(
        f"{a}" if a == b else (f"{a}-{b}" if b - a > 1 else f"{a},{b}")
        for a, b in runs(ids)
    )


def main():
    print("=" * 78)
    print("PHAN TICH 45 FRAME THIEU MAU - DOI CHIEU VOI 5 LOG GOC")
    print("=" * 78)

    all_csv = {}
    for name in sorted(CSV_TO_LOG):
        frames = read_csv_frames(RAW / name)
        all_csv[name] = frames
        ids = sorted(frames)
        lens = Counter(f["len"] for f in frames.values())
        counts = Counter(len(f["samples"]) for f in frames.values())
        print(f"\n[CSV] {name}")
        print(f"  frames={len(frames)}  range={ids[0]}..{ids[-1]}")
        print(f"  len values: {dict(lens)}")
        print(f"  sample-count distribution: {dict(counts)}")

    print("\n" + "-" * 78)
    print("LOG FILES")
    print("-" * 78)
    all_logs = {}
    for log_name in list(CSV_TO_LOG.values()) + [ORPHAN_LOG]:
        frames, counters, malformed, stats = read_log(LOGS / log_name)
        all_logs[log_name] = (frames, counters, malformed, stats)
        ids = sorted(frames)
        lens = Counter(f["len_field"] for f in frames.values())
        baddecl = [i for i, f in frames.items() if not f["declared_ok"]]
        print(f"\n[LOG] {log_name}")
        print(f"  parsed CSI lines : {stats['parsed']}")
        print(f"  malformed lines  : {stats['malformed']} {malformed[:3]}")
        print(f"  frame range      : {ids[0]}..{ids[-1]}" if ids else "  no frames")
        print(f"  len fields       : {dict(lens)}")
        print(f"  hex!=len*2 lines : {baddecl[:10]}{' ...' if len(baddecl) > 10 else ''}")
        print(f"  CAPTURE complete : {counters}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BUOC 2-3: DOI CHIEU 45 FRAME LOI")
    print("=" * 78)

    header = (
        f"{'CSV':<18} {'frame':>6} {'n_csv':>5} {'len_csv':>7} "
        f"{'in_log':>6} {'len_log':>7} {'hexB':>5} {'IQ_log':>6} {'nhan_dinh'}"
    )
    print(header)
    print("-" * len(header))

    incomplete_all = {}
    for name, frames in all_csv.items():
        bad = [fid for fid, f in frames.items() if len(f["samples"]) != 192]
        incomplete_all[name] = sorted(bad)

    case_counter = Counter()
    rows = []
    for name in sorted(all_csv):
        log_name = CSV_TO_LOG[name]
        log_frames = all_logs[log_name][0]
        for fid in incomplete_all[name]:
            cf = all_csv[name][fid]
            n_csv = len(cf["samples"])
            lf = log_frames.get(fid)
            if lf is None:
                case = "D: khong co trong log"
                note = "frame khong xuat hien trong log"
                case_counter["D"] += 1
                row = (name, fid, n_csv, cf["len"], "KHONG", "-", "-", "-", note)
            else:
                iq_log = lf["hex_bytes"] // 2
                if lf["len_field"] == 128 and lf["hex_bytes"] == 128:
                    case = "A: log len=128/128B"
                    note = "buffer ngan tu firmware"
                    case_counter["A"] += 1
                elif lf["len_field"] == 384 and lf["hex_bytes"] == 384 and n_csv == 64:
                    case = "B: log day du, CSV thieu"
                    note = "mat o buoc Python/CSV"
                    case_counter["B"] += 1
                elif lf["len_field"] == 384 and lf["hex_bytes"] < 384:
                    case = "C: log len=384 nhung hex thieu"
                    note = "hex ngan hon khai bao"
                    case_counter["C"] += 1
                else:
                    case = "E: khac"
                    note = f"len_log={lf['len_field']} hex={lf['hex_bytes']} n_csv={n_csv}"
                    case_counter["E"] += 1
                row = (
                    name, fid, n_csv, cf["len"], "CO",
                    lf["len_field"], lf["hex_bytes"], iq_log,
                    f"{case} | {note}",
                )
            rows.append(row)
            print(
                f"{row[0]:<18} {row[1]:>6} {row[2]:>5} {row[3]:>7} "
                f"{row[4]:>6} {str(row[5]):>7} {str(row[6]):>5} {str(row[7]):>6}  {row[8]}"
            )

    print(f"\nTong frame loi: {len(rows)}  |  phan loai: {dict(case_counter)}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BUOC 4: FRAME BINH THUONG XUNG QUANH FRAME LOI")
    print("=" * 78)
    for name in sorted(incomplete_all):
        log_name = CSV_TO_LOG[name]
        log_frames = all_logs[log_name][0]
        ids = sorted(all_csv[name])
        idset = set(ids)
        print(f"\n--- {name} (log {log_name}) ---")
        for bad in incomplete_all[name]:
            probes = [bad - 1, bad + 1, bad - 2, bad + 2]
            for p in probes:
                if p not in idset:
                    continue
                cf = all_csv[name][p]
                lf = log_frames.get(p)
                tag = "LOI" if p in incomplete_all[name] else "ok "
                linfo = (
                    f"len={lf['len_field']} hex={lf['hex_bytes']}"
                    if lf
                    else "KHONG CO"
                )
                print(
                    f"  [{tag}] frame {p}: csv_n={len(cf['samples']):>3} "
                    f"csv_len={cf['len']} ts={cf['timestamp_ms']} | log: {linfo}"
                )
            lf = log_frames.get(bad)
            cf = all_csv[name][bad]
            linfo = (
                f"len={lf['len_field']} hex={lf['hex_bytes']} "
                f"ts={lf['timestamp_ms']} rssi={lf['rssi']} line={lf['line_no']}"
                if lf
                else "KHONG CO"
            )
            print(
                f"  [LOI] frame {bad}: csv_n={len(cf['samples']):>3} "
                f"csv_len={cf['len']} ts={cf['timestamp_ms']} | log: {linfo}"
            )

    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("KIEM TRA CHAT LUONG TONG THE")
    print("=" * 78)
    for name, frames in sorted(all_csv.items()):
        ids = sorted(frames)
        dup_ts = Counter(f["timestamp_ms"] for f in frames.values())
        dup_ts = [t for t, c in dup_ts.items() if c > 1]
        deltas = [b - a for a, b in zip([frames[i]["timestamp_ms"] for i in ids[:-1]],
                                        [frames[i]["timestamp_ms"] for i in ids[1:]])]
        neg = sum(1 for d in deltas if d < 0)
        zero = sum(1 for d in deltas if d == 0)
        big = [d for d in deltas if d > 10000]
        # sample index continuity
        bad_idx = []
        zero_frames = 0
        for fid, f in frames.items():
            idx = sorted(f["samples"])
            if idx != list(range(len(idx))):
                bad_idx.append(fid)
            amps = [abs(complex(i, q)) for i, q in f["samples"].values()]
            if amps and max(amps) == 0:
                zero_frames += 1
        # len vs sample count consistency
        len_mismatch = [
            fid for fid, f in frames.items()
            if f["len"] != 192 or len(f["samples"]) * 2 != f["len"]
        ]
        print(f"\n[QUALITY] {name}")
        print(f"  frames={len(frames)}  sample-count dist={dict(Counter(len(f['samples']) for f in frames.values()))}")
        print(f"  sample_index khong lien tuc: {len(bad_idx)} -> {bad_idx[:10]}")
        print(f"  timestamp trung: {len(dup_ts)} {dup_ts[:10]}")
        print(f"  timestamp giam: {neg} | bang 0: {zero} | nhay >10s: {len(big)} {big[:5]}")
        print(f"  frame toan 0: {zero_frames}")
        print(f"  len!=192 hoac 2*n!=len: {len(len_mismatch)} -> {len_mismatch[:10]}")
        # classify len vs n
        combos = Counter((f["len"], len(f["samples"])) for f in frames.values())
        print(f"  (len, n_samples) combos: {dict(combos)}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("KET QUA GHEP NOI LOG KHONG CO CSV")
    print("=" * 78)
    frames, counters, malformed, stats = all_logs[ORPHAN_LOG]
    ids = sorted(frames)
    print(f"{ORPHAN_LOG}: frames {ids[0]}..{ids[-1]}, {len(ids)} frames, "
          f"len fields {dict(Counter(f['len_field'] for f in frames.values()))}")
    print(f"  khong co CSV tuong ung trong data/raw")


if __name__ == "__main__":
    main()
