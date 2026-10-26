"""Smoke tests for the CSI preprocessing CLI using a tiny synthetic CSV."""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "csi_preprocessing_pipeline.py"


def test_pipeline_generates_expected_outputs(tmp_path):
    raw = tmp_path / "raw"
    project = tmp_path / "project"
    raw.mkdir()
    project.mkdir()
    source = raw / "phong_trong_test.csv"
    fields = ["frame_id", "timestamp_ms", "rssi", "len", "first_word", "sample_index", "i", "q", "amplitude"]
    rows = []
    # Two complete frames with four samples each; override expected-samples=4.
    for frame_id, timestamp in [(1, 1000), (2, 2000)]:
        for sample_index in range(4):
            i = float(frame_id + sample_index)
            q = float(sample_index)
            rows.append({
                "frame_id": frame_id, "timestamp_ms": timestamp, "rssi": -40,
                "len": 4, "first_word": 0, "sample_index": sample_index,
                "i": i, "q": q, "amplitude": (i * i + q * q) ** 0.5,
            })
    with source.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-dir", str(raw), "--output-dir", str(project), "--expected-samples", "4"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert (project / "data/processed/csi_dataset_with_session_labels.csv").exists()
    assert (project / "data/processed/frame_quality_report.csv").exists()
    assert (project / "data/processed/session_quality_summary.csv").exists()
    assert (project / "data/processed/preliminary_frame_features.csv").exists()
    assert (project / "results/heatmap_phong_trong_test.png").exists()
    assert (project / "results/mean_amplitude_by_session.png").exists()
    assert (project / "results/preliminary_motion_score.png").exists()


def _write_session(path, frame_ids, amplitude_base):
    """Write one 4-sample-per-frame session CSV with known amplitudes."""
    fields = ["frame_id", "timestamp_ms", "rssi", "len", "first_word", "sample_index", "i", "q", "amplitude"]
    rows = []
    for frame_id in frame_ids:
        for sample_index in range(4):
            amplitude = amplitude_base + 10 * frame_id + sample_index
            rows.append({
                "frame_id": frame_id, "timestamp_ms": 1000 * frame_id, "rssi": -40,
                "len": 4, "first_word": 0, "sample_index": sample_index,
                "i": amplitude, "q": 0.0, "amplitude": float(amplitude),
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_normalization_fits_only_on_train_sessions(tmp_path):
    """Scaler statistics must come from --train-sessions, not all rows."""
    import numpy as np
    import pandas as pd

    raw = tmp_path / "raw"
    project = tmp_path / "project"
    raw.mkdir()
    project.mkdir()

    # Train session: frame means 11.5, 21.5, 31.5 -> mean 21.5.
    _write_session(raw / "phong_trong_train.csv", [1, 2, 3], amplitude_base=0)
    # Held-out session with a very different level.
    _write_session(raw / "nguoi_di_ra_holdout.csv", [1, 2, 3], amplitude_base=1000)

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--input-dir", str(raw),
            "--output-dir", str(project),
            "--expected-samples", "4",
            "--normalization", "zscore",
            "--train-sessions", "phong_trong_train",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "--train-sessions not set" not in result.stdout

    processed = project / "data/processed"
    params = pd.read_csv(processed / "normalization_params.csv")

    row = params.loc[
        (params["dataset"] == "preliminary_frame_features")
        & (params["column"] == "mean_amplitude")
    ].iloc[0]

    # Expected statistics from the train session only (recomputed offline).
    train_rows = []
    for frame_id in [1, 2, 3]:
        train_rows.append(np.mean([10 * frame_id + s for s in range(4)]))
    expected_center = float(np.mean(train_rows))
    expected_scale = float(np.std(train_rows, ddof=0))

    assert abs(row["center"] - expected_center) < 1e-9
    assert abs(row["scale"] - expected_scale) < 1e-9
    assert row["note"] != "ALL 6 rows (no split)"

    # Train rows are centered: mean of scaled values must be ~0.
    train_scaled = pd.read_csv(processed / "preliminary_frame_features_normalized.csv")
    train_only = train_scaled.loc[
        train_scaled["session_id"] == "phong_trong_train", "mean_amplitude"
    ]
    assert abs(train_only.mean()) < 1e-9
    assert abs(train_only.std(ddof=0) - 1.0) < 1e-9

    # Held-out rows must NOT be centered around zero (it was never fitted).
    holdout_only = train_scaled.loc[
        train_scaled["session_id"] == "nguoi_di_ra_holdout", "mean_amplitude"
    ]
    assert holdout_only.mean() > 100


def test_normalization_rejects_unknown_session(tmp_path):
    """A typo in --train-sessions must fail loudly, not silently fit all rows."""
    raw = tmp_path / "raw"
    project = tmp_path / "project"
    raw.mkdir()
    project.mkdir()
    _write_session(raw / "phong_trong_test.csv", [1, 2, 3], amplitude_base=0)

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--input-dir", str(raw),
            "--output-dir", str(project),
            "--expected-samples", "4",
            "--normalization", "minmax",
            "--train-sessions", "does_not_exist",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "does_not_exist" in (result.stdout + result.stderr)
