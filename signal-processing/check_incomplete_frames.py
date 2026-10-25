
import pandas as pd
from pathlib import Path

base = Path("data/raw")
files = [
    "nguoi_di_ra.csv",
    "nguoi_di_vao.csv",
    "nguoi_ngoi_im.csv",
    "phong_trong.csv",
]

for filename in files:
    path = base / filename
    df = pd.read_csv(
        path,
        usecols=[
            "frame_id",
            "timestamp_ms",
            "sample_index",
            "i",
            "q",
            "amplitude",
        ],
    )

    counts = df.groupby("frame_id").size()
    incomplete_ids = counts[counts < 192].index

    print(f"\n=== {filename} ===")
    print(f"Total incomplete frames: {len(incomplete_ids)}")

    for frame_id in incomplete_ids:
        frame = df[df["frame_id"] == frame_id]
        indices = sorted(frame["sample_index"].unique())

        print(
            f"frame={frame_id}, "
            f"count={len(frame)}, "
            f"timestamp={frame['timestamp_ms'].iloc[0]}, "
            f"index_range=({min(indices)}, {max(indices)}), "
            f"unique_indices={len(indices)}, "
            f"first_iq={list(zip(frame['i'].head(3), frame['q'].head(3)))}"
        )
