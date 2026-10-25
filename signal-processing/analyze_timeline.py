"""Deep dive: timeline, stats lines and payload structure of short CSI frames."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

LOGS = Path("../hardware")

LOG_FILES = [
    "csi_capture_20260925_202440.log",
    "csi_capture_20260925_203950.log",
    "csi_capture_20260925_204849.log",
    "csi_capture_20260925_212313.log",
    "csi_capture_20260925_213320.log",
]

CSI_RE = re.compile(r'^CSI_DATA,(\d+),(\d+),(-?\d+),(\d+),(\d+),"([0-9A-Fa-f]*)"\r?$')


def parse(path: Path):
    frames, stats, other = [], [], []
    for lineno, line in enumerate(path.open(encoding="utf-8", errors="replace"), 1):
        line = line.rstrip("\r\n")
        if line.startswith("CSI_DATA,") and not line.startswith("CSI_DATA,frame_id"):
            m = CSI_RE.match(line)
            if not m:
                other.append((lineno, line[:80]))
                continue
            fid, ts, rssi, ln, fw, hx = m.groups()
            frames.append(
                dict(fid=int(fid), ts=int(ts), rssi=int(rssi), ln=int(ln),
                     n=len(hx) // 2, hx=hx, line=lineno)
            )
        elif line.startswith("stats,"):
            d = {}
            for kv in line.split(",")[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = v
            stats.append(d)
    return frames, stats, other


def zero_ratio(hx: str) -> float:
    vals = [int(hx[i:i + 4], 16) for i in range(0, len(hx) - 3, 4)]
    return sum(1 for v in vals if v == 0) / len(vals) if vals else 0.0


for name in LOG_FILES:
    frames, stats, other = parse(LOGS / name)
    print("=" * 90)
    print(name)
    print(f"  frames={len(frames)}  stats_lines={len(stats)}  unparseable={len(other)}")
    print(f"  len distribution: {dict(Counter(f['ln'] for f in frames))}")
    print(f"  csi_length in stats: {dict(Counter(s.get('csi_length') for s in stats))}")
    print(f"  csi_status: {dict(Counter(s.get('csi_status') for s in stats))}")
    print(f"  csi_rate_fps: {dict(Counter(s.get('csi_rate_fps') for s in stats))}")
    print(f"  channel: {dict(Counter(s.get('channel') for s in stats))}")
    print(f"  wifi: {dict(Counter(s.get('wifi') for s in stats))}")
    print(f"  udp_received: {dict(Counter(s.get('udp_received') for s in stats))}")
    print(f"  udp_lost: {dict(Counter(s.get('udp_lost') for s in stats))}")
    # consecutive deltas grouped by the length of the *next* frame
    deltas = []
    for a, b in zip(frames, frames[1:]):
        deltas.append((b["ts"] - a["ts"], b["ln"]))
    for target in (128, 384):
        ds = [d for d, ln in deltas if ln == target]
        if not ds:
            continue
        c = Counter()
        for d in ds:
            if d < 150:
                c["<150ms"] += 1
            elif d < 400:
                c["150-400"] += 1
            elif d < 700:
                c["400-700"] += 1
            elif d < 1500:
                c["700-1500"] += 1
            else:
                c[">1500"] += 1
        print(f"  gap before len={target} frame: {dict(c)} "
              f"median={sorted(ds)[len(ds)//2]}")

    # show runs of short frames
    short_ids = [f["fid"] for f in frames if f["ln"] == 128]
    print(f"  short frames: {short_ids}")

    # zero ratio comparison
    for ln in (128, 384):
        zs = [zero_ratio(f["hx"]) for f in frames if f["ln"] == ln]
        if zs:
            print(f"  mean zero-sample ratio for len={ln}: "
                  f"{sum(zs)/len(zs):.3f} (min={min(zs):.3f}, max={max(zs):.3f})")

    # per-frame time deltas across the whole capture
    all_d = [b["ts"] - a["ts"] for a, b in zip(frames, frames[1:])]
    if all_d:
        print(f"  ts deltas: min={min(all_d)} max={max(all_d)} "
              f"median={sorted(all_d)[len(all_d)//2]}")

print()
print("=" * 90)
print("CHI TIET CAC CUM FRAME NGAN (phong_trong - log 203950)")
print("=" * 90)
frames, stats, _ = parse(LOGS / "csi_capture_20260925_203950.log")
by_id = {f["fid"]: i for i, f in enumerate(frames)}
for start, end in [(815, 835), (855, 870), (895, 950), (955, 1000)]:
    print(f"\n--- frames {start}..{end} ---")
    prev = None
    for f in frames:
        if not (start <= f["fid"] <= end):
            prev = f
            continue
        d = f["ts"] - prev["ts"] if prev else 0
        flag = "SHORT" if f["ln"] == 128 else "     "
        print(f"  {flag} {f['fid']:>4} ts={f['ts']:>7} d={d:>6} "
              f"len={f['ln']} n={f['n']} rssi={f['rssi']} line={f['line']}")
        prev = f
