"""End-to-end verification: decode hex from raw log -> compare with raw CSV rows.

For every frame of every CSV we re-derive (i, q, amplitude) from the hex payload
found in the corresponding serial log and compare cell by cell.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")
LOGS = Path("../hardware")

CSV_TO_LOG = {
    "nguoi_ngoi_im.csv": "csi_capture_20260925_202440.log",
    "phong_trong.csv": "csi_capture_20260925_203950.log",
    "nguoi_di_ra.csv": "csi_capture_20260925_212313.log",
    "nguoi_di_vao.csv": "csi_capture_20260925_213320.log",
}

CSI_RE = re.compile(r'^CSI_DATA,(\d+),(\d+),(-?\d+),(\d+),(\d+),"([0-9A-Fa-f]*)"\r?$')


def s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def load_log(path: Path):
    frames = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        m = CSI_RE.match(line.rstrip("\r\n"))
        if not m:
            continue
        fid, ts, rssi, ln, fw, hx = m.groups()
        frames[int(fid)] = dict(ts=int(ts), rssi=int(rssi), ln=int(ln), hx=hx)
    return frames


def load_csv(path: Path):
    frames = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            f = frames.setdefault(int(row["frame_id"]), {
                "ts": int(row["timestamp_ms"]), "rssi": int(row["rssi"]),
                "len": int(row["len"]), "samples": {},
            })
            f["samples"][int(row["sample_index"])] = (
                int(row["i"]), int(row["q"]), float(row["amplitude"])
            )
    return frames


total_frames = 0
total_cells = 0
mismatch_frames = []
order_stats = Counter()
problems = Counter()

for csv_name, log_name in CSV_TO_LOG.items():
    logs = load_log(LOGS / log_name)
    csvs = load_csv(RAW / csv_name)
    assert set(csvs) == set(logs), (
        f"{csv_name}: frame id sets differ: "
        f"csv-only={sorted(set(csvs) - set(logs))[:5]} "
        f"log-only={sorted(set(logs) - set(csvs))[:5]}"
    )
    for fid in sorted(csvs):
        c = csvs[fid]
        l = logs[fid]
        total_frames += 1
        if c["ts"] != l["ts"]:
            problems["timestamp_mismatch"] += 1
        if c["rssi"] != l["rssi"]:
            problems["rssi_mismatch"] += 1
        if c["len"] != l["ln"]:
            problems["len_mismatch"] += 1
        if c["len"] != len(l["hx"]) // 2:
            problems["log_hex_len_mismatch"] += 1

        # decode hex -> samples
        bytes_ = [int(l["hx"][i:i + 2], 16) for i in range(0, len(l["hx"]), 2)]
        samples_iq = list(zip(bytes_[0::2], bytes_[1::2]))

        # determine byte order by comparing with the first CSV sample
        first = c["samples"].get(0)
        if first is None:
            problems["csv_missing_sample_0"] += 1
            continue
        ci, cq, _ = first
        b0, b1 = samples_iq[0]
        if s8(b0) == ci and s8(b1) == cq:
            order = "iq"
        elif s8(b1) == ci and s8(b0) == cq:
            order = "qi"
        else:
            order = "unknown"
        order_stats[order] += 1
        if order == "unknown":
            problems["cannot_match_byte_order"] += 1
            continue

        bad = 0
        for idx, (hi, hq) in enumerate(samples_iq):
            row = c["samples"].get(idx)
            if row is None:
                bad += 1
                continue
            ri, rq, ra = row
            if order == "iq":
                ei, eq = s8(hi), s8(hq)
            else:
                ei, eq = s8(hq), s8(hi)
            amp_ok = abs(ra - math.sqrt(ei * ei + eq * eq)) <= 0.0001
            total_cells += 1
            if ri != ei or rq != eq or not amp_ok:
                bad += 1
        extra = set(c["samples"]) - set(range(len(samples_iq)))
        bad += len(extra)
        if bad:
            mismatch_frames.append((csv_name, fid, bad))
            problems["frame_cell_mismatch"] += 1

        # CSV n samples vs log bytes
        if len(c["samples"]) * 2 != l["ln"]:
            problems["csv_sample_count_vs_len"] += 1

print(f"frames compared      : {total_frames}")
print(f"sample cells compared: {total_cells}")
print(f"byte-order decision  : {dict(order_stats)}")
print(f"problems             : {dict(problems)}")
print(f"frames with mismatch : {len(mismatch_frames)} {mismatch_frames[:10]}")
