"""Build CSI windows from a saved capture session and run pose inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIGNAL_PROCESSING_SRC = ROOT / "signal-processing" / "src"
sys.path.insert(0, str(SIGNAL_PROCESSING_SRC))

from parse_intel_csi import convert_session, read_key_values  # noqa: E402

DEFAULT_MODEL = (
    ROOT
    / "machine-learning"
    / "results"
    / "pose-sync-check"
    / "final-csi-rebuild"
    / "retrained-verified-lan2-lan3-20261110"
    / "pose_baseline.joblib"
)
DEFAULT_OUTPUT_DIR = ROOT / "machine-learning" / "results" / "raw-session-inference"
CSI_FEATURES = 224
MAX_GAP_MS = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict person presence and pose from a saved raw CSI session."
    )
    parser.add_argument("--input-session", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions.csv",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=0.2)
    return parser.parse_args()


def build_csi_windows(
    amplitudes: np.ndarray,
    timestamps_ms: np.ndarray,
    window_frames: int,
    step_frames: int,
    max_gap_ms: float = MAX_GAP_MS,
) -> tuple[np.ndarray, np.ndarray, int]:
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != CSI_FEATURES:
        raise ValueError(
            f"CSI amplitudes must have shape [frames, {CSI_FEATURES}]."
        )
    if timestamps_ms.shape != (len(amplitudes),):
        raise ValueError("timestamps_ms must contain one value per CSI frame.")
    if not np.isfinite(amplitudes).all() or not np.isfinite(timestamps_ms).all():
        raise ValueError("CSI amplitudes and timestamps must be finite.")
    if np.any(np.diff(timestamps_ms) <= 0):
        raise ValueError("CSI timestamps must be strictly increasing.")
    if window_frames < 2 or step_frames < 1:
        raise ValueError("window_frames must be >= 2 and step_frames must be >= 1.")
    if max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive.")
    if len(amplitudes) < window_frames:
        return (
            np.empty((0, window_frames, CSI_FEATURES), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            0,
        )

    windows = []
    end_timestamps = []
    rejected_gaps = 0
    for end in range(window_frames - 1, len(amplitudes), step_frames):
        start = end - window_frames + 1
        if np.any(np.diff(timestamps_ms[start : end + 1]) > max_gap_ms):
            rejected_gaps += 1
            continue
        windows.append(amplitudes[start : end + 1])
        end_timestamps.append(timestamps_ms[end])

    if not windows:
        return (
            np.empty((0, window_frames, CSI_FEATURES), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            rejected_gaps,
        )
    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(end_timestamps, dtype=np.float64),
        rejected_gaps,
    )


def windows_from_session(
    session_dir: Path,
    window_seconds: float = 2.0,
    step_seconds: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Window and step durations must be positive.")
    for name in ("raw.csi", "recorder.log", "metadata.txt"):
        if not (session_dir / name).is_file():
            raise FileNotFoundError(f"CSI session is missing {name}: {session_dir}")

    metadata = read_key_values(session_dir / "metadata.txt")
    parsed, skipped = convert_session(
        session_dir,
        "vht",
        include_test=True,
        label_override="inference",
        trial_override=metadata.get("trial") or session_dir.name,
    )
    if skipped is not None:
        raise ValueError(f"Cannot parse CSI session: {skipped['reason']}")
    if not parsed["kept"]:
        raise ValueError(f"No VHT CSI frames found in {session_dir}.")

    amplitudes = np.asarray(
        [
            np.round(
                np.hypot(
                    np.asarray(points, dtype=np.float32)[:, 0],
                    np.asarray(points, dtype=np.float32)[:, 1],
                ),
                4,
            )
            for _, _, points in parsed["kept"]
        ],
        dtype=np.float32,
    )
    timestamps_ms = np.asarray(
        [frame["ts"] // 1_000_000 for _, frame, _ in parsed["kept"]],
        dtype=np.float64,
    )
    intervals_s = np.diff(timestamps_ms) / 1000.0
    positive_intervals = intervals_s[intervals_s > 0]
    if not len(positive_intervals):
        raise ValueError(f"Could not determine CSI sampling interval in {session_dir}.")
    sampling_interval_s = float(np.median(positive_intervals))
    window_frames = max(2, int(round(window_seconds / sampling_interval_s)))
    step_frames = max(1, int(round(step_seconds / sampling_interval_s)))
    windows, end_timestamps, rejected_gaps = build_csi_windows(
        amplitudes,
        timestamps_ms,
        window_frames,
        step_frames,
    )
    if not len(windows):
        raise ValueError(
            f"No CSI windows survived the {MAX_GAP_MS:g} ms gap check "
            f"for {session_dir}."
        )
    return windows, end_timestamps, rejected_gaps, sampling_interval_s * 1000.0


def run(
    session_dir: Path,
    model_path: Path,
    output_path: Path,
    window_path: Path | None = None,
    window_seconds: float = 2.0,
    step_seconds: float = 0.2,
) -> int:
    if not session_dir.is_dir():
        raise FileNotFoundError(f"CSI session directory not found: {session_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")
    from predict_pose import predict

    started = time.perf_counter()
    parse_started = time.perf_counter()
    windows, timestamps_ms, rejected_gaps, interval_ms = windows_from_session(
        session_dir,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    parse_window_seconds = time.perf_counter() - parse_started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if window_path is None:
        window_path = output_path.with_name(f"{output_path.stem}_windows.npz")
    window_path.parent.mkdir(parents=True, exist_ok=True)
    archive_started = time.perf_counter()
    np.savez_compressed(
        window_path,
        csi=windows,
        timestamp_ms=timestamps_ms,
        session=np.asarray(session_dir.name),
        window_seconds=np.asarray(window_seconds),
        step_seconds=np.asarray(step_seconds),
        sampling_interval_ms=np.asarray(interval_ms),
    )
    archive_seconds = time.perf_counter() - archive_started
    inference_started = time.perf_counter()
    count = predict(window_path, model_path, output_path)
    inference_seconds = time.perf_counter() - inference_started
    total_seconds = time.perf_counter() - started
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "input_session": str(session_dir.resolve()),
                "model": str(model_path.resolve()),
                "window_count": count,
                "rejected_gap_windows": rejected_gaps,
                "sampling_interval_ms": interval_ms,
                "window_seconds": window_seconds,
                "step_seconds": step_seconds,
                "parse_and_window_seconds": parse_window_seconds,
                "window_archive_seconds": archive_seconds,
                "model_load_inference_and_csv_seconds": inference_seconds,
                "end_to_end_seconds": total_seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"CSI frames -> windows -> predictions: "
        f"{len(windows)} windows; rejected_gap_windows={rejected_gaps}; "
        f"sampling_interval_ms={interval_ms:.3f}"
    )
    print(
        f"timing_s: parse+window={parse_window_seconds:.3f}, "
        f"archive={archive_seconds:.3f}, "
        f"model+CSV={inference_seconds:.3f}, end-to-end={total_seconds:.3f}"
    )
    print(f"Window archive: {window_path}")
    print(f"Timing report: {metrics_path}")
    return count


def main() -> None:
    args = parse_args()
    run(
        session_dir=args.input_session,
        model_path=args.model,
        output_path=args.output,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )


if __name__ == "__main__":
    main()
