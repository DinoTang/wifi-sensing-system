"""Locate exact zero positions inside short (128B) and normal (384B) CSI payloads."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

LOGS = Path("../hardware")
CSI_RE = re.compile(r'^CSI_DATA,(\d+),(\d+),(-?\d+),(\d+),(\d+),"([0-9A-Fa-f]*)"\r?$')


def load(path):
    out = []
    for line in Path(path).open(encoding="utf-8", errors="replace"):
        m = CSI_RE.match(line.rstrip("\r\n"))
        if m:
            fid, ts, rssi, ln, fw, hx = m.groups()
            out.append(dict(fid=int(fid), ln=int(ln), fw=fw, hx=hx))
    return out


logs = sorted(LOGS.glob("*.log"))
all_short, all_long = [], []
for lg in logs:
    for f in load(lg):
        vals = [int(f["hx"][i:i + 4], 16) for i in range(0, len(f["hx"]), 4)]
        (all_short if f["ln"] == 128 else all_long).append(
            (lg.name, f["fid"], vals, f["fw"])
        )

print(f"short frames: {len(all_short)}   normal frames: {len(all_long)}")
print(f"first_word values (short): {Counter(w for _, _, _, w in all_short)}")
print(f"first_word values (normal): {Counter(w for _, _, _, w in all_long)}")

# zero positions in short frames
zero_pos_sets = Counter()
for name, fid, vals, _ in all_short:
    pos = tuple(i for i, v in enumerate(vals) if v == 0)
    zero_pos_sets[pos] += 1
print(f"\nDistinct zero-position patterns in SHORT frames: {len(zero_pos_sets)}")
for pos, cnt in zero_pos_sets.most_common(3):
    print(f"  count={cnt} n_zero_values={len(pos)} positions={list(pos)}")

# per-sample (I,Q both zero) in short frames
samp_zero_sets = Counter()
for name, fid, vals, _ in all_short:
    pairs = list(zip(vals[0::2], vals[1::2]))
    pos = tuple(i for i, (a, b) in enumerate(pairs) if a == 0 and b == 0)
    samp_zero_sets[pos] += 1
print(f"\nDistinct fully-zero SAMPLE positions in SHORT frames: {len(samp_zero_sets)}")
for pos, cnt in samp_zero_sets.most_common(3):
    print(f"  count={cnt} n_zero_samples={len(pos)} positions={list(pos)}")

# same for normal frames (use a few)
samp_zero_long = Counter()
for name, fid, vals, _ in all_long:
    pairs = list(zip(vals[0::2], vals[1::2]))
    pos = tuple(i for i, (a, b) in enumerate(pairs) if a == 0 and b == 0)
    samp_zero_long[pos] += 1
print(f"\nDistinct fully-zero SAMPLE positions in NORMAL frames: {len(samp_zero_long)}")
for pos, cnt in samp_zero_long.most_common(3):
    print(f"  count={cnt} n_zero_samples={len(pos)} positions={list(pos)}")

# count of zero values per normal frame
zc = Counter(sum(1 for v in vals if v == 0) for _, _, vals, _ in all_long)
print(f"\nzero-value count distribution in NORMAL frames: {dict(sorted(zc.items()))}")
zc_s = Counter(sum(1 for v in vals if v == 0) for _, _, vals, _ in all_short)
print(f"zero-value count distribution in SHORT frames  : {dict(sorted(zc_s.items()))}")

# Is a short payload ever a prefix / suffix / segment of a normal payload?
sample_short = all_short[0][2]
sh_hex = "".join(f"{v:04X}" for v in sample_short)
hits_prefix = 0
for name, fid, vals, _ in all_long[:50]:
    h = "".join(f"{v:04X}" for v in vals)
    if h.startswith(sh_hex) or h.endswith(sh_hex) or sh_hex in h:
        hits_prefix += 1
print(f"\nshort payload found inside first 50 normal payloads: {hits_prefix}")
