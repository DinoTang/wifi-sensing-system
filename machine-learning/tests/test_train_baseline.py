import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_baseline import evaluate, load_features, save_charts


def _write_features(directory: Path, sessions_per_class: int = 2) -> None:
    directory.mkdir()
    frame_rows = []
    temporal_rows = []
    for label, base in (
        ("room_empty", 0.0),
        ("person_static", 10.0),
        ("person_moving", 20.0),
    ):
        for session_number in range(sessions_per_class):
            session_id = f"{label}_{session_number + 1}"
            for frame_id in range(3):
                frame_rows.append(
                    {
                        "session_id": session_id,
                        "scenario_label": label,
                        "frame_id": frame_id,
                        "frame_complete": True,
                        "mean_amplitude": base + frame_id,
                        "std_amplitude": base + frame_id + 1,
                        "max_amplitude": base + frame_id + 2,
                        "mean_rssi": -50 + base,
                        "motion_score": base + frame_id + 3,
                    }
                )
                temporal_rows.append(
                    {
                        "session_id": session_id,
                        "scenario_label": label,
                        "frame_id": frame_id,
                        "temporal_std_amplitude": base + frame_id + 4,
                        "temporal_std_amplitude_max": base + frame_id + 5,
                    }
                )
    pd.DataFrame(frame_rows).to_csv(
        directory / "preliminary_frame_features.csv", index=False
    )
    pd.DataFrame(temporal_rows).to_csv(
        directory / "temporal_features.csv", index=False
    )


def test_load_features_joins_frames_and_keeps_complete_frames(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    frames_path = processed / "preliminary_frame_features.csv"
    frames = pd.read_csv(frames_path)
    frames.loc[0, "frame_complete"] = False
    frames.to_csv(frames_path, index=False)

    merged = load_features(processed)

    assert len(merged) == 17
    assert "temporal_std_amplitude" in merged.columns
    assert "frame_complete" not in merged.columns


def test_evaluate_holds_out_whole_sessions(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    data = load_features(processed)

    summary, predictions, session_scores, _ = evaluate(
        data=data,
        task_name="three_class",
        model_name="logistic_regression",
        feature_set="temporal_only",
        feature_columns=["temporal_std_amplitude", "temporal_std_amplitude_max"],
        labels=["room_empty", "person_static", "person_moving"],
    )

    assert summary["sessions"] == 6
    assert summary["frames"] == 18
    assert len(session_scores) == 6
    assert len(predictions) == 18
    assert {row["session_id"] for row in predictions} == set(data["session_id"])


def test_load_features_rejects_duplicate_session_frame(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    frames_path = processed / "preliminary_frame_features.csv"
    frames = pd.read_csv(frames_path)
    frames = pd.concat([frames, frames.iloc[[0]]], ignore_index=True)
    frames.to_csv(frames_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_features(processed)


def test_save_charts_creates_confusion_and_session_plots(tmp_path):
    predictions = []
    session_scores = []
    for task, labels in (
        ("three_class", ["room_empty", "person_static", "person_moving"]),
        ("presence", ["room_empty", "person_present"]),
    ):
        for session_index, label in enumerate(labels):
            session_id = f"{label}_{session_index + 1}"
            predictions.append(
                {
                    "task": task,
                    "model": "random_forest",
                    "feature_set": "frame_plus_temporal",
                    "session_id": session_id,
                    "frame_id": 1,
                    "label_true": label,
                    "label_predicted": label,
                }
            )
            session_scores.append(
                {
                    "task": task,
                    "model": "random_forest",
                    "feature_set": "frame_plus_temporal",
                    "session_id": session_id,
                    "held_out_label": label,
                    "accuracy": 1.0,
                }
            )

    charts = save_charts(predictions, session_scores, tmp_path)

    assert [path.name for path in charts] == [
        "random_forest_confusion_matrices.png",
        "random_forest_session_accuracy.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in charts)
