import pandas as pd
from pathlib import Path

for path in sorted(Path("data/raw").glob("*.csv")):
    df = pd.read_csv(
        path,
        usecols=["frame_id", "timestamp_ms", "sample_index"]
    )

    stats = df.groupby("frame_id").agg(
        timestamp_ms=("timestamp_ms", "first"),
        sample_count=("sample_index", "size"),
    ).sort_index()

    stats["timestamp_diff"] = stats["timestamp_ms"].diff()
    stats["incomplete"] = stats["sample_count"] < 192

    print(f"\n===== {path.name} =====")
    print("Timestamp trùng giữa các frame:")
    print(
        stats[stats["timestamp_diff"] == 0][
            ["timestamp_ms", "sample_count", "incomplete"]
        ].head(20).to_string()
    )

    print("\nPhân bố khoảng cách timestamp (ms):")
    print(stats["timestamp_diff"].describe().to_string())

    print("\nTimestamp trùng ở frame đầy đủ:")
    full_duplicates = stats[
        (stats["timestamp_diff"] == 0) & (~stats["incomplete"])
    ]
    print(full_duplicates.head(10).to_string())
