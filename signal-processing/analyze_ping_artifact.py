"""Kiểm chứng giả thuyết "cú nhọn amplitude định kỳ do lưu lượng ping".

Câu hỏi cần trả lời:
    Các cú nhọn amplitude ở phiên phòng trống có chu kỳ ~100 ms (10 Hz,
    bằng lưu lượng ping) hay không, và có đường phổ tại tần số alias của
    ping (10 Hz khi lấy mẫu ở 9,9 Hz -> 0,099 Hz) hay không?

Phương pháp (chỉ numpy/pandas/matplotlib, không có scipy):
1. Chuỗi tín hiệu phân tích là **mean amplitude theo frame** — đúng chuỗi
   có đường nền ~131 và các cú nhọn được báo cáo trong báo cáo nhóm.
2. Phát hiện cú nhọn: điểm đỉnh cục bộ (đỉnh nội bộ) có z >= 2 so với
   toàn chuỗi; thống kê khoảng cách giữa các cú nhọn theo mili-giây.
3. ACF (autocorrelation) trên chuỗi chưa khử xu hướng, lấy đỉnh cục bộ
   trong 0,3 - 5 s và so với ngưỡng 1.96/sqrt(N).
4. Periodogram (FFT có cửa sổ Hann); tại tần số alias của ping so công
   suất của bin đó với **median của lân cận ±1 octave** — cách này không
   bị lệch do phổ có dạng 1/f.
5. Làm cả 3 lớp làm nhóm đối chứng: nếu cú nhọn/ping là hiện tượng chung
   thì phải thấy ở cả 3 lớp, không chỉ ở phòng trống.

Kết luận phải dựa trên bảng số, không suy diễn: một chuỗi 60 s (~560
frame) chỉ chứa ~6 chu kỳ của tần số 0,099 Hz nên khả năng phát hiện rất yếu.

Chạy:
    python analyze_ping_artifact.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_DIR = Path("data/intel_raw")
OUT_DIR = Path("data/intel/results")

SESSION_GROUPS = {
    "room_empty": ["room_empty_01", "room_empty_02", "room_empty_03"],
    "person_static": ["person_static_01", "person_static_02", "person_static_03"],
    "person_moving": ["person_moving_01", "person_moving_02", "person_moving_03"],
}

PING_HZ = 10.0          # tần số lưu lượng ping khai báo
PEAK_Z = 2.0           # ngưỡng z để coi một đỉnh cục bộ là cú nhọn
MIN_LAG_S = 0.3        # ACF chỉ quan tâm độ trễ >= 0,3 s
MAX_LAG_S = 12.0       # ... và tới 12 s (phải phủ cả chu kỳ alias 10,1 s)
FMIN, FMAX = 0.02, 5.0  # dải tần số của periodogram (Hz)


def load_frame_series(session_id: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{session_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(
        path,
        usecols=["frame_id", "timestamp_ms", "amplitude"],
    )
    if df.empty:
        return None
    frames = df.groupby("frame_id", sort=True).agg(
        timestamp_ms=("timestamp_ms", "first"),
        mean_amplitude=("amplitude", "mean"),
        max_amplitude=("amplitude", "max"),
    )
    return frames.sort_values("timestamp_ms").reset_index(drop=True)


def local_peaks(values: np.ndarray, z_threshold: float) -> np.ndarray:
    """Chỉ số các đỉnh cục bộ có z >= ngưỡng."""
    z = (values - values.mean()) / values.std(ddof=0)
    peaks = [
        i
        for i in range(1, len(values) - 1)
        if values[i] >= values[i - 1]
        and values[i] > values[i + 1]
        and z[i] >= z_threshold
    ]
    return np.array(peaks, dtype=int)


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    y = values - values.mean()
    denom = np.dot(y, y)
    if denom <= 0:
        return np.zeros(max_lag + 1)
    full = np.correlate(y, y, mode="full")[len(y) - 1:]
    return full[: max_lag + 1] / denom


def periodogram(values: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    y = values - values.mean()
    windowed = y * np.hanning(len(y))
    power = (np.abs(np.fft.rfft(windowed)) ** 2) / len(y)
    freqs = np.fft.rfftfreq(len(y), d=dt_s)
    return freqs, power


def alias_ratio(freqs: np.ndarray, power: np.ndarray, alias_hz: float) -> tuple[int, float, float, float]:
    """(bin, công suất tại alias, median lân cận ±1 octave, tỷ lệ)."""
    idx = int(np.argmin(np.abs(freqs - alias_hz)))
    low = int(np.argmin(np.abs(freqs - alias_hz / 2)))
    high = int(np.argmin(np.abs(freqs - alias_hz * 2)))
    local = power[low: high + 1]
    local_median = float(np.median(local)) if len(local) else float("nan")
    ratio = float(power[idx] / local_median) if local_median > 0 else float("nan")
    return idx, float(power[idx]), local_median, ratio


def analyse(session_id: str, scenario_label: str) -> tuple[dict | None, dict | None]:
    frames = load_frame_series(session_id)
    if frames is None or len(frames) < 50:
        return None, None

    timestamps = frames["timestamp_ms"].to_numpy()
    dt_ms = float(np.median(np.diff(timestamps)))
    dt_s = dt_ms / 1000.0
    fs = 1000.0 / dt_ms

    series = frames["mean_amplitude"].to_numpy(dtype=float)

    # --- cú nhọn ---
    peaks = local_peaks(series, PEAK_Z)
    if len(peaks) > 1:
        gaps = np.diff(timestamps[peaks])
        gap_median = float(np.median(gaps))
        gap_p25 = float(np.percentile(gaps, 25))
        gap_p75 = float(np.percentile(gaps, 75))
    else:
        gap_median = gap_p25 = gap_p75 = float("nan")

    # --- ACF ---
    max_lag = min(len(series) - 2, int(MAX_LAG_S * fs))
    min_lag = max(3, int(MIN_LAG_S * fs))
    acf = autocorrelation(series, max_lag)
    threshold = 1.96 / np.sqrt(len(series))
    candidates = [
        (lag, float(acf[lag]))
        for lag in range(min_lag, max_lag)
        if acf[lag] >= acf[lag - 1] and acf[lag] >= acf[lag + 1]
    ]
    significant = [item for item in candidates if item[1] > threshold]
    significant.sort(key=lambda item: -item[1])

    # --- ACF tại đúng chu kỳ alias của ping (1/|fs-ping| giây) ---
    alias_hz = abs(fs - PING_HZ)
    alias_lag = int(round(fs / alias_hz)) if alias_hz > 0 else 0
    acf_at_alias = float(acf[alias_lag]) if alias_lag < len(acf) else float("nan")

    # --- periodogram + tỷ lệ công suất tại alias của ping ---
    freqs, power = periodogram(series, dt_s)
    _, alias_power, local_median, ratio = alias_ratio(freqs, power, alias_hz)

    band = (freqs >= FMIN) & (freqs <= FMAX)
    band_freqs = freqs[band]
    band_power = power[band]
    top = band_freqs[int(np.argmax(band_power))] if len(band_freqs) else float("nan")

    summary = {
        "session_id": session_id,
        "scenario_label": scenario_label,
        "frames": len(frames),
        "median_dt_ms": round(dt_ms, 1),
        "sampling_hz": round(fs, 3),
        "mean_amp_std": round(float(series.std(ddof=0)), 2),
        "peaks_count": int(len(peaks)),
        "peaks_per_min": round(len(peaks) / (len(series) * dt_s / 60.0), 1),
        "peak_gap_median_ms": round(gap_median, 0) if np.isfinite(gap_median) else float("nan"),
        "peak_gap_p25_ms": round(gap_p25, 0) if np.isfinite(gap_p25) else float("nan"),
        "peak_gap_p75_ms": round(gap_p75, 0) if np.isfinite(gap_p75) else float("nan"),
        "acf_threshold": round(float(threshold), 3),
        "acf_significant_count": len(significant),
        "acf_top_lag_ms": round(significant[0][0] * dt_ms, 0) if significant else float("nan"),
        "acf_top_value": round(significant[0][1], 3) if significant else float("nan"),
        "alias_hz": round(alias_hz, 3),
        "alias_period_s": round(1.0 / alias_hz, 2) if alias_hz > 0 else float("nan"),
        "acf_at_alias_lag": round(acf_at_alias, 3),
        "alias_power": round(alias_power, 2),
        "alias_local_median": round(local_median, 2),
        "alias_ratio": round(ratio, 2),
        "top_freq_hz": round(float(top), 3),
    }

    detail = {
        "session_id": session_id,
        "series": series,
        "dt_s": dt_s,
        "fs": fs,
        "peaks": peaks,
        "acf": acf,
        "threshold": threshold,
        "freqs": band_freqs,
        "power": band_power,
        "alias_hz": alias_hz,
    }
    return summary, detail


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    details: dict[str, dict] = {}
    for label, sessions in SESSION_GROUPS.items():
        for session_id in sessions:
            summary, detail = analyse(session_id, label)
            if summary is None:
                print(f"[SKIP] {session_id}: not found or too short")
                continue
            rows.append(summary)
            details[session_id] = detail

    if not rows:
        print("No session analysed.")
        return

    report = pd.DataFrame(rows)
    out_csv = OUT_DIR / "ping_periodicity_check.csv"
    report.to_csv(out_csv, index=False)

    print()
    print("[PING PERIODICITY CHECK]")
    print()
    print(report.to_string(index=False))

    empty = report.loc[report["scenario_label"] == "room_empty"]

    print()
    print("[READING]")
    print(
        f"- Chu kỳ 100 ms (10 Hz ping) -> kỳ vọng: gap_median ~100 ms và "
        f"ACF đỉnh tại 101 ms."
    )
    print(
        f"- Thực tế room_empty: gap_median = "
        f"{sorted(empty['peak_gap_median_ms'].dropna().astype(int))} ms, "
        f"acf_significant_count = {sorted(empty['acf_significant_count'])}, "
        f"acf_top_lag = {sorted(empty['acf_top_lag_ms'].dropna().astype(int))} ms."
    )
    print(
        f"- Tần số alias của ping = {empty['alias_hz'].iloc[0]:.3f} Hz "
        f"(chu kỳ {empty['alias_period_s'].iloc[0]:.1f} s, ~100 frame); "
        f"ACF tại đúng độ trễ đó = {sorted(empty['acf_at_alias_lag'])}; "
        f"tỷ lệ công suất so với lân cận = {sorted(empty['alias_ratio'])} "
        f"(>= ~2 mới đáng ngờ, và phải lặp lại ở cả 3 phiên)."
    )
    print(
        "- Nếu cả ba mục trên không cho giá trị ổn định thì giả thuyết "
        "ping-khơi-mắc (beat) KHÔNG được dữ liệu hỗ trợ."
    )

    # --- figure ---
    ids = list(details)
    fig, axes = plt.subplots(len(ids), 3, figsize=(16, 3.0 * len(ids)))
    if len(ids) == 1:
        axes = np.array([axes])

    for row, session_id in enumerate(ids):
        d = details[session_id]
        time_s = np.arange(len(d["series"])) * d["dt_s"]

        ax = axes[row][0]
        ax.plot(time_s, d["series"], lw=0.7, color="#1f77b4")
        if len(d["peaks"]):
            ax.plot(time_s[d["peaks"]], d["series"][d["peaks"]], "o",
                    ms=3, color="#d62728")
        ax.axhline(d["series"].mean(), color="gray", lw=0.7, ls="--")
        ax.set_title(f"{session_id} · mean amplitude (red = peak)", fontsize=9)
        ax.set_xlabel("time (s)")

        ax = axes[row][1]
        lag_s = np.arange(len(d["acf"])) * d["dt_s"]
        ax.plot(lag_s, d["acf"], lw=0.9, color="#d62728")
        ax.axhline(d["threshold"], color="gray", ls="--", lw=0.8, label="1.96/sqrt(N)")
        ax.axhline(-d["threshold"], color="gray", ls="--", lw=0.8)
        ax.axvline(0.101, color="orange", ls=":", lw=1.2, label="100 ms (ping)")
        if d["alias_hz"] > 0:
            ax.axvline(1.0 / d["alias_hz"], color="purple", ls="--", lw=1.2,
                       label=f"alias period {1.0 / d['alias_hz']:.1f} s")
        ax.set_title("autocorrelation (raw mean amplitude)", fontsize=9)
        ax.set_xlabel("lag (s)")
        ax.legend(fontsize=7)

        ax = axes[row][2]
        ax.semilogy(d["freqs"], d["power"], lw=0.9, color="#2ca02c")
        ax.axvline(d["alias_hz"], color="purple", ls="--", lw=1.2,
                   label=f"alias {d['alias_hz']:.3f} Hz")
        ax.axvline(PING_HZ, color="orange", ls=":", lw=1.2, label="ping 10 Hz")
        ax.set_title("periodogram (0.02-5 Hz)", fontsize=9)
        ax.set_xlabel("frequency (Hz)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_png = OUT_DIR / "ping_periodicity_check.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    print()
    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()
