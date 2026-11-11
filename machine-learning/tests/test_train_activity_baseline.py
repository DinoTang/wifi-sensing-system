import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_activity_baseline import (
    FEATURE_SETS,
    LABELS,
    blocked_time_splits,
    confusion_rows,
    evaluate,
    load_features,
    make_model,
    save_confusion_chart,
    session_held_out_splits,
)
from predict_activity import load_inference_features, predict_activity


def _write_features(
    directory: Path,
    frames_per_session: int = 60,
    sessions_per_class: int = 1,
) -> None:
    directory.mkdir()
    frame_rows = []
    temporal_rows = []
    for label_index, label in enumerate(LABELS):
        for session_number in range(1, sessions_per_class + 1):
            session_id = f"{label}_{session_number:02}"
            for frame_id in range(frames_per_session):
                value = (
                    label_index * 10
                    + session_number * 0.1
                    + frame_id / frames_per_session
                )
                frame_rows.append(
                    {
                        "session_id": session_id,
                        "scenario_label": label,
                        "frame_id": frame_id,
                        "frame_complete": True,
                        "mean_amplitude": value,
                        "std_amplitude": value + 1,
                        "max_amplitude": value + 2,
                        "mean_rssi": -50 + label_index,
                        "motion_score": value + 3,
                    }
                )
                temporal_rows.append(
                    {
                        "session_id": session_id,
                        "scenario_label": label,
                        "frame_id": frame_id,
                        "temporal_std_amplitude": value + 4,
                        "temporal_std_amplitude_max": value + 5,
                    }
                )
    pd.DataFrame(frame_rows).to_csv(
        directory / "preliminary_frame_features.csv", index=False
    )
    pd.DataFrame(temporal_rows).to_csv(
        directory / "temporal_features.csv", index=False
    )


def test_load_features_joins_and_keeps_complete_frames(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    frame_path = processed / "preliminary_frame_features.csv"
    frames = pd.read_csv(frame_path)
    frames.loc[0, "frame_complete"] = False
    frames.to_csv(frame_path, index=False)

    data = load_features(processed)

    assert len(data) == len(LABELS) * 60 - 1
    assert "temporal_std_amplitude" in data.columns
    assert "frame_complete" not in data.columns


def test_blocked_splits_are_disjoint_and_guard_temporal_neighbors(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    data = load_features(processed)
    guard = 3

    splits = blocked_time_splits(data, folds=3, guard_frames=guard)

    test_indices = np.concatenate([test for _, test in splits])
    assert sorted(test_indices.tolist()) == list(range(len(data)))
    for train, test in splits:
        assert not np.intersect1d(train, test).size
        for session_id in data["session_id"].unique():
            train_positions = np.flatnonzero(
                data.iloc[train]["session_id"].to_numpy() == session_id
            )
            test_positions = np.flatnonzero(
                data.iloc[test]["session_id"].to_numpy() == session_id
            )
            train_frames = data.iloc[train].iloc[train_positions]["frame_id"].to_numpy()
            test_frames = data.iloc[test].iloc[test_positions]["frame_id"].to_numpy()
            if len(train_frames) and len(test_frames):
                assert min(
                    abs(int(train_frame) - int(test_frame))
                    for train_frame in train_frames
                    for test_frame in test_frames
                ) > guard


def test_session_held_out_splits_never_share_sessions(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed, sessions_per_class=2)
    data = load_features(processed)

    session_splits = session_held_out_splits(data)

    assert len(session_splits) == 12
    for train, test in session_splits:
        train_sessions = set(data.iloc[train]["session_id"])
        test_sessions = set(data.iloc[test]["session_id"])
        assert len(test_sessions) == 1
        assert not train_sessions.intersection(test_sessions)
        assert set(data.iloc[train]["scenario_label"]) == set(LABELS)


def test_auto_evaluation_uses_session_holdout_when_repeats_exist(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed, sessions_per_class=2)
    data = load_features(processed)

    summaries, predictions, _, _ = evaluate(data, evaluation_mode="auto")

    assert {row["evaluation_mode"] for row in summaries} == {"session_held_out"}
    for row in predictions:
        assert row["session_id"].endswith(("01", "02"))


def test_session_evaluation_refuses_single_session_per_activity(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    data = load_features(processed)

    with pytest.raises(ValueError, match="at least two sessions per activity"):
        evaluate(data, evaluation_mode="session")


def test_evaluate_returns_oof_predictions_for_all_configurations(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    data = load_features(processed)

    summaries, predictions, session_scores, importances = evaluate(
        data, folds=3, guard_frames=3
    )

    assert len(summaries) == 6
    assert len(predictions) == len(data) * 6
    assert len(session_scores) == len(LABELS) * 6
    assert importances
    assert {row["label_true"] for row in predictions} == set(LABELS)
    assert {row["evaluation_mode"] for row in summaries} == {
        "within_session_time_block_diagnostic"
    }


def test_save_confusion_chart_creates_image(tmp_path):
    predictions = []
    for label in LABELS:
        predictions.append(
            {
                "model": "random_forest",
                "feature_set": "temporal_only",
                "label_true": label,
                "label_predicted": label,
            }
        )
        predictions.append(
            {
                "model": "random_forest",
                "feature_set": "frame_plus_temporal",
                "label_true": label,
                "label_predicted": label,
            }
        )

    chart = save_confusion_chart(
        predictions, tmp_path, evaluation_mode="session_held_out"
    )

    assert chart.is_file()
    assert chart.stat().st_size > 0


def test_confusion_rows_preserve_actual_and_predicted_counts():
    predictions = [
        {
            "model": "random_forest",
            "feature_set": "temporal_only",
            "label_true": "empty",
            "label_predicted": "empty",
        },
        {
            "model": "random_forest",
            "feature_set": "temporal_only",
            "label_true": "empty",
            "label_predicted": "standing",
        },
    ]

    rows = confusion_rows(predictions)
    empty_as_empty = next(
        row
        for row in rows
        if row["actual_label"] == "empty" and row["predicted_label"] == "empty"
    )
    empty_as_standing = next(
        row
        for row in rows
        if row["actual_label"] == "empty"
        and row["predicted_label"] == "standing"
    )

    assert empty_as_empty["count"] == 1
    assert empty_as_standing["count"] == 1


def test_load_features_rejects_missing_activity(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    temporal_path = processed / "temporal_features.csv"
    temporal = pd.read_csv(temporal_path)
    temporal = temporal.loc[temporal["scenario_label"] != "mixed"]
    temporal.to_csv(temporal_path, index=False)

    with pytest.raises(ValueError, match="Expected all six activity labels"):
        load_features(processed)


def test_predict_activity_works_without_ground_truth_labels(tmp_path):
    processed = tmp_path / "processed"
    _write_features(processed)
    training = load_features(processed)
    feature_columns = FEATURE_SETS["frame_plus_temporal"]
    model = make_model("random_forest")
    model.fit(training[feature_columns], training["scenario_label"])

    frame_path = processed / "preliminary_frame_features.csv"
    temporal_path = processed / "temporal_features.csv"
    frames = pd.read_csv(frame_path).drop(columns=["scenario_label"])
    temporal = pd.read_csv(temporal_path).drop(columns=["scenario_label"])
    frames.to_csv(frame_path, index=False)
    temporal.to_csv(temporal_path, index=False)

    inference_rows = load_inference_features(processed, feature_columns)
    predictions = predict_activity(
        inference_rows,
        {
            "model": model,
            "feature_columns": feature_columns,
            "labels": LABELS,
            "evaluation_mode": "session_held_out",
            "model_version": "test-version",
        },
    )

    assert len(predictions) == len(training)
    assert "predicted_activity" in predictions.columns
    assert "scenario_label" not in predictions.columns
    assert set(predictions["model_validation_status"]) == {"session_held_out"}
    assert set(predictions["model_version"]) == {"test-version"}
