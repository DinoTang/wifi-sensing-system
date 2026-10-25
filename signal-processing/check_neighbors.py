import pandas as pd
from pathlib import Path

files = [
    "nguoi_di_ra.csv",
    "nguoi_di_vao.csv",
    "nguoi_ngoi_im.csv",
    "phong_trong.csv",
]

for filename in files:
    df = pd.read_csv(
        Path("data/raw") / filename,
        usecols=["frame_id", "timestamp_ms", "sample_index"]
    )

    stats = (
        df.groupby("frame_id")
        .agg(
            timestamp_ms=("timestamp_ms", "first"),
            sample_count=("sample_index", "size"),
            unique_samples=("sample_index", "nunique"),
            min_index=("sample_index", "min"),
            max_index=("sample_index", "max"),
        )
        .sort_index()
    )

    bad_ids = stats.index[stats["sample_count"] < 192]

    print(f"\n===== {filename} =====")

    for fid in bad_ids:
        print(f"\n--- Frame lỗi {fid} và các frame lân cận ---")
        print(stats.loc[
            (stats.index >= fid - 1) & (stats.index <= fid + 1)
        ].to_string())
