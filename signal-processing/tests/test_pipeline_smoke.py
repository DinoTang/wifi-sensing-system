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
