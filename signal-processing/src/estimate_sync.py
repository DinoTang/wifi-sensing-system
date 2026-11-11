"""Estimate the CSI ↔ video time offset when visual clock anchors are unavailable.

The preferred method is to fit visible clock readings from the video. This
motion-correlation estimator is a fallback and reports weak or ambiguous peaks
instead of treating them as verified synchronization.

Method
------
1. CSI side: mean amplitude per frame → ``|Δ|`` between consecutive frames
   (the same idea as ``motion_score`` in the preprocessing pipeline).
2. Video side: mean absolute difference between consecutive grayscale frames.
3. Both series are resampled to 10 Hz and z-scored.
4. For every candidate lag in the shared-data range the Pearson correlation of
   the overlapping part is computed.

The search covers every lag with at least ``MIN_OVERLAP_S`` of shared data.

Output per session: best offset, correlation at that offset, the runner-up
offset and the peak ratio. A weak peak means "cannot be established", and the
script says so instead of inventing a number.

Run from anywhere::

    python signal-processing/src/estimate_sync.py
    python signal-processing/src/estimate_sync.py --plot
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT / "data" / "pose" / "index.csv"
DEFAULT_OUTPUT = PROJECT / "data" / "pose" / "sync_report.csv"

GRID_HZ = 10.0
LAG_STEP_S = 0.05
MIN_OVERLAP_S = 20.0

# Thresholds used only to *label* confidence, never to hide a weak result.
R_STRONG = 0.45
RATIO_STRONG = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate CSI↔video offset")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plot", action="store_true", help="Vẽ đường cong tương quan ra PNG"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="Làm mượt cả hai chuỗi trước khi tương quan (giây)",
    )
    return parser.parse_args()


def lag_bounds(csi_duration_s: float, video_duration_s: float) -> tuple[float, float]:
    """Return all lags that leave enough overlap for correlation."""
    return (
        -(csi_duration_s - MIN_OVERLAP_S),
        video_duration_s - MIN_OVERLAP_S,
    )


def lag_to_offset_s(lag_s: float) -> float:
    """Convert video-time-at-first-CSI lag to the offset used by windowing."""
    return -lag_s


def csi_motion(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamp seconds relative to first frame, motion)."""
    frame = pd.read_csv(csv_path, usecols=["frame_id", "timestamp_ms", "amplitude"])
    per_frame = (
        frame.groupby("frame_id", sort=True)
        .agg(ts_ms=("timestamp_ms", "first"), mean_amp=("amplitude", "mean"))
        .reset_index()
    )
    ts = per_frame["ts_ms"].to_numpy(dtype=float) / 1000.0
    amp = per_frame["mean_amp"].to_numpy(dtype=float)
    motion = np.abs(np.diff(amp, prepend=amp[0]))
    return ts - ts[0], motion


def video_motion(path: Path) -> tuple[np.ndarray, float]:
    """Return (motion per frame, fps)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Không mở được video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    values = []
    previous = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.cvtColor(
            cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        if previous is not None:
            values.append(float(np.abs(small - previous).mean()))
        previous = small
    capture.release()
    return np.asarray(values, dtype=float), float(fps)


def resample(values: np.ndarray, fps: float, duration_s: float, grid_hz: float):
    """Average-resample a per-frame series onto a fixed-rate grid."""
    grid = np.arange(0.0, duration_s, 1.0 / grid_hz)
    times = np.arange(len(values)) / fps
    return np.interp(grid, times, values), grid


def zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return values - values.mean()
    return (values - values.mean()) / std


def smooth(values: np.ndarray, seconds: float) -> np.ndarray:
    """Centered moving average (offline)."""
    width = max(1, int(round(seconds * GRID_HZ)))
    if width < 2:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def correlate(csi: np.ndarray, video: np.ndarray, lags_s: np.ndarray) -> np.ndarray:
    """Pearson r for each lag; positive lag maps CSI t=0 to later video time."""
    shift = np.round(lags_s * GRID_HZ).astype(int)
    scores = np.full(len(lags_s), np.nan)
    for index, offset in enumerate(shift):
        if offset >= 0:
            moved = video[offset:]
            base = csi[: len(moved)]
        else:
            moved = video[: len(video) + offset]
            base = csi[-offset:]
        overlap = min(len(moved), len(base))
        if overlap < MIN_OVERLAP_S * GRID_HZ:
            continue
        moved = moved[:overlap]
        base = base[:overlap]
        a = base - base.mean()
        b = moved - moved.mean()
        denominator = np.sqrt((a * a).sum() * (b * b).sum())
        if denominator == 0:
            continue
        scores[index] = float((a * b).sum() / denominator)
    return scores


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Không thấy {index_path}. Chạy build_pose_index.py trước.")

    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))

    rows = []
    curves = {}

    print("=" * 84)
    print("ƯỚC LƯỢNG ĐỘ LỆCH THỜI GIAN CSI ↔ VIDEO (tương quan chuyển động)")
    print("=" * 84)

    for row in index_rows:
        activity = row["activity"]
        repeat = row["repeat"]
        csv_path = PROJECT / "data" / "pose" / "raw_csv" / f"{activity}_{repeat}.csv"
        video_path = PROJECT / row["video_file"]
        if not csv_path.exists() or not video_path.exists():
            print(f"[SKIP] {activity}_{repeat}: thiếu CSV hoặc video")
            continue

        csi_t, csi_m = csi_motion(csv_path)
        video_m, fps = video_motion(video_path)

        csi_grid, _ = resample(csi_m, 1.0 / np.median(np.diff(csi_t)), csi_t[-1], GRID_HZ)
        video_duration = len(video_m) / fps
        video_grid, _ = resample(video_m, fps, video_duration, GRID_HZ)

        smooth_s = args.smooth
        csi_series = smooth(zscore(csi_grid), smooth_s)
        video_series = smooth(zscore(video_grid), smooth_s)
        lag_min, lag_max = lag_bounds(csi_t[-1], video_duration)
        if lag_min > lag_max:
            print(f"[WARN] {activity}_{repeat}: CSI/video overlap is under 20 seconds")
            continue
        lags = np.arange(lag_min, lag_max + LAG_STEP_S / 2, LAG_STEP_S)
        scores = correlate(csi_series, video_series, lags)
        curves[(activity, repeat)] = (lags.copy(), scores.copy())

        valid = np.isfinite(scores)
        if not valid.any():
            print(f"[WARN] {activity}_{repeat}: không đủ vùng chồng lấn")
            continue

        best = int(np.nanargmax(scores))
        best_lag = float(lags[best])
        best_r = float(scores[best])

        # Runner-up: best correlation at least 1 s away from the main peak.
        far = np.abs(lags - best_lag) > 1.0
        runner_idx = int(np.nanargmax(scores[far])) if far.any() else -1
        runner = float(scores[far][runner_idx]) if runner_idx >= 0 else float("nan")
        ratio = best_r / abs(runner) if runner and not np.isnan(runner) else float("nan")
        best_offset = lag_to_offset_s(best_lag)
        runner_lag = float(lags[far][runner_idx]) if runner_idx >= 0 else float("nan")

        if best_r >= R_STRONG and ratio >= RATIO_STRONG:
            verdict = "usable"
        elif best_r >= 0.25:
            verdict = "weak"
        else:
            verdict = "unreliable"

        rows.append(
            {
                "session": row["session_id"],
                "activity": activity,
                "repeat": repeat,
                "best_lag_s": round(best_lag, 2),
                "best_offset_s": round(best_offset, 2),
                "correlation_r": round(best_r, 3),
                "runner_up_offset_s": round(lag_to_offset_s(runner_lag), 2)
                if runner_idx >= 0
                else "",
                "runner_up_r": round(runner, 3) if not np.isnan(runner) else "",
                "peak_ratio": round(ratio, 2) if not np.isnan(ratio) else "",
                "verdict": verdict,
            }
        )
        print(
            f"{activity:<11}{repeat:<3} offset={best_offset:>7.2f} s  r={best_r:>5.3f}"
            f"  runner={runner:>5.3f}  ratio={ratio:>4.2f}  -> {verdict}"
        )

    if not rows:
        raise SystemExit("Không ước lượng được phiên nào.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"report: {output}")

    usable = [row for row in rows if row["verdict"] == "usable"]
    print()
    if usable:
        offsets = sorted(row["best_offset_s"] for row in usable)
        median = offsets[len(offsets) // 2]
        spread = max(offsets) - min(offsets)
        print(f"phiên đủ tin cậy: {len(usable)}/{len(rows)}")
        print(f"offset trung vị: {median:.2f} s (dao động {spread:.2f} s)")
    else:
        print("KHÔNG có phiên nào đủ tin cậy để chốt offset từ chuyển động.")
        print("→ phải quay lại có đồng hồ trong khung hình, hoặc chấp nhận giả định.")

    if args.plot and curves:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            len(curves), 1, figsize=(8, 1.6 * len(curves)), sharex=True
        )
        for axis, ((activity, repeat), (lag, score)) in zip(axes, curves.items()):
            axis.plot(lag, score, lw=0.9)
            axis.axvline(0, color="gray", ls="--", lw=0.8)
            axis.axhline(0, color="gray", lw=0.5)
            axis.set_ylabel(f"{activity}_{repeat}", fontsize=8)
            axis.tick_params(labelsize=7)
        axes[-1].set_xlabel("lag (s) — video time aligned with first CSI frame")
        figure.tight_layout()
        plot_path = PROJECT / "data" / "pose" / "results" / "sync_correlation.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_path, dpi=130)
        print(f"plot   : {plot_path}")


if __name__ == "__main__":
    main()
