"""Kiểm tra "nghi vấn thứ tự thu dồn theo lớp" trên 9 phiên Intel.

Bối cảnh: 9 phiên được thu theo thứ tự room_empty -> person_static ->
person_moving (xem data/intel_raw/_sessions_metadata.csv). Nếu trong lúc thu
môi trường trôi (dịch chuyển thiết bị, đổi nền phản xạ...), độ trôi sẽ trùng
với nhãn và ta không phân biệt được "khác do trạng thái người" hay "khác do
thời gian".

Script này chỉ kiểm tra mức ĐỦ để nhận định ban đầu (chưa đóng được nghi vấn):
1. Bảng chỉ số từng phiên theo đúng thứ tự thời gian thu.
2. Ma trận tương quan hồ sơ amplitude theo từng tone (56 tone) giữa các phiên.
3. So sánh "khác biệt giữa các lớp" với "khác biệt giữa các phiên cùng lớp":
   nếu hai khái niệm này cùng độ lớn thì thứ tự thu chưa thể tách khỏi nhãn.

Chạy:
    python analyze_session_drift.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_DIR = Path("data/intel_raw")
METADATA = RAW_DIR / "_sessions_metadata.csv"
OUT_DIR = Path("data/intel/results")


def tone_profile(df: pd.DataFrame) -> pd.Series:
    """Trung bình amplitude theo tone, tính trên toàn bộ frame của phiên."""
    return df.groupby("sample_index")["amplitude"].mean()


def session_metrics(session_id: str) -> tuple[dict | None, pd.Series | None]:
    path = RAW_DIR / f"{session_id}.csv"
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    if df.empty:
        return None, None

    profile = tone_profile(df)
    frame_rssi = df.groupby("frame_id")["rssi"].first()
    frame_mean = df.groupby("frame_id")["amplitude"].mean()

    metrics = {
        "session_id": session_id,
        "frames": int(df["frame_id"].nunique()),
        "rssi_median_dbm": float(frame_rssi.median()),
        "rssi_mean_dbm": float(frame_rssi.mean()),
        "mean_amplitude": float(df["amplitude"].mean()),
        "median_amplitude": float(df["amplitude"].median()),
        "frame_mean_std": float(frame_mean.std(ddof=0)),
        "tone_profile_mean": float(profile.mean()),
        "tone_profile_std": float(profile.std(ddof=0)),
    }
    return metrics, profile


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA)

    # _sessions_metadata.csv dùng csv_file (vd room_empty_01.csv) -> bỏ đuôi.
    meta["session_id"] = meta["csv_file"].str.replace(r"\.csv$", "", regex=True)
    order = meta.sort_values("start_time")["session_id"].tolist()

    rows = []
    profiles = {}
    for session_id in order:
        metrics, profile = session_metrics(session_id)
        if metrics is None:
            print(f"[SKIP] {session_id}")
            continue
        metrics["capture_order"] = len(rows) + 1
        rows.append(metrics)
        profiles[session_id] = profile

    if not rows:
        print("No session analysed.")
        return

    report = pd.DataFrame(rows)
    start_map = dict(zip(meta["session_id"], meta["start_time"]))
    report["start_time"] = report["session_id"].map(start_map)

    label_map = dict(zip(meta["session_id"], meta["scenario"]))
    report["scenario_label"] = report["session_id"].map(label_map)

    cols = [
        "capture_order",
        "start_time",
        "scenario_label",
        "session_id",
        "frames",
        "rssi_median_dbm",
        "rssi_mean_dbm",
        "mean_amplitude",
        "median_amplitude",
        "frame_mean_std",
        "tone_profile_mean",
        "tone_profile_std",
    ]
    report = report[cols]

    out_csv = OUT_DIR / "session_drift_check.csv"
    report.round(3).to_csv(out_csv, index=False)

    print()
    print("[SESSION DRIFT CHECK — theo thứ tự thời gian thu]")
    print()
    print(report.round(3).to_string(index=False))

    # --------------------------------------------------------
    # 1. Khác biệt giữa lớp vs khác biệt trong lớp.
    # --------------------------------------------------------
    work = report.copy()
    metric_cols = [
        "rssi_median_dbm",
        "mean_amplitude",
        "tone_profile_mean",
        "frame_mean_std",
    ]

    within = work.groupby("scenario_label")[metric_cols].agg(lambda s: s.max() - s.min())
    between = work.groupby("scenario_label")[metric_cols].mean()
    grand_mean = work[metric_cols].mean()

    print()
    print("[BIÊN ĐỘ TRONG LỚP (max-min của 3 phiên cùng lớp)]")
    print(within.round(3).to_string())

    print()
    print("[CHÊNH LỆCH GIỮA LỚP (trung bình lớp - tổng trung bình)]")
    deviation = between - grand_mean
    print(deviation.round(3).to_string())

    print()
    print("[TỶ SỐ |chênh lệch giữa lớp| / biên độ trong lớp]")
    ratio = (deviation.abs() / within.replace(0, np.nan)).round(2)
    print(ratio.to_string())
    print(
        "\nĐọc: tỷ số <= 1 nghĩa là chênh lệch giữa các lớp cùng độ lớn với "
        "phần dao động bên trong lớp,\n"
        "tức là dữ liệu hiện có KHÔNG đủ để tách 'khác do trạng thái người' "
        "khỏi 'khác do thời gian thu'."
    )

    # --------------------------------------------------------
    # 2. Ma trận tương quan hồ sơ tone.
    # --------------------------------------------------------
    profile_df = pd.DataFrame(profiles)
    corr = profile_df.corr()

    print()
    print("[MA TRẬN TƯƠNG QUAN HỒ SƠ AMPLITUDE THEO TONE (9x9)]")
    print(corr.round(3).to_string())

    # --------------------------------------------------------
    # 3. Hình: chuỗi thời gian các chỉ số + ma trận tương quan.
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    x = np.arange(len(work))
    labels = [f"{i}\n{s}" for i, s in zip(work["capture_order"], work["session_id"])]

    ax = axes[0][0]
    ax.plot(x, work["rssi_median_dbm"], "o-", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("RSSI median (dBm)")
    ax.set_title("RSSI theo thứ tự thu", fontsize=10)

    ax = axes[0][1]
    ax.plot(x, work["mean_amplitude"], "o-", color="#d62728", label="mean amplitude")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(x, work["frame_mean_std"], "s--", color="#ff7f0e", label="frame-to-frame std")
    ax.set_ylabel("mean amplitude")
    ax2.set_ylabel("frame std")
    ax.set_title("Biên độ nền và độ dao động giữa các frame", fontsize=10)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7)

    ax = axes[1][0]
    for label, group in work.groupby("scenario_label"):
        ax.plot(group["capture_order"], group["mean_amplitude"], "o-", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean amplitude")
    ax.legend(fontsize=7)
    ax.set_title("Mỗi lớp một màu — chú ý thứ tự thu bị dồn theo lớp", fontsize=10)

    ax = axes[1][1]
    im = ax.imshow(corr.values, vmin=0.9, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Tương quan hồ sơ tone giữa các phiên", fontsize=10)

    fig.tight_layout()
    out_png = OUT_DIR / "session_drift_check.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    print()
    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()
