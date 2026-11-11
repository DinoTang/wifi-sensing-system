import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_pose import predict
from train_pose_baseline import (
    evaluate_leave_one_session_out,
    load_dataset,
    run,
)


LANDMARKS = np.asarray(
    [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
)


def _write_session(
    directory: Path,
    session_id: str,
    sync_ok: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sum(ord(char) for char in session_id))
    csi = rng.normal(size=(4, 2, 224)).astype(np.float32)
    presence = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    mask = np.repeat(presence[:, None], len(LANDMARKS), axis=1)
    coordinates = np.full((4, len(LANDMARKS), 2), np.nan, dtype=np.float32)
    coordinates[1, :, :] = 0.4
    coordinates[2, :, :] = 0.6

    np.savez_compressed(
        directory / f"{session_id}.npz",
        csi=csi,
        keypoints_normalized=coordinates,
        mask=mask,
        person_present=presence,
        sync_ok=np.asarray(sync_ok, dtype=np.uint8),
        session=np.asarray(session_id),
        activity=np.asarray("standing"),
        landmarks=LANDMARKS,
    )


def test_loader_refuses_unverified_windows(tmp_path):
    windows = tmp_path / "windows"
    _write_session(windows, "standing_01", sync_ok=False)

    with pytest.raises(ValueError, match="explicitly pass"):
        load_dataset(windows)


def test_loader_accepts_delivered_legacy_schema_only_for_diagnostic(tmp_path):
    windows = tmp_path / "windows"
    windows.mkdir()
    landmarks = np.asarray(
        [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
    )
    coordinates = np.full((4, 33, 2), [960.0, 540.0], dtype=np.float32)
    np.savez_compressed(
        windows / "standing_01.npz",
        csi=np.ones((4, 20, 224), dtype=np.float32),
        keypoints=coordinates,
        mask=np.ones((4, 33), dtype=np.uint8),
        sync_ok=np.asarray(0, dtype=np.uint8),
        session=np.asarray("standing_01"),
        activity=np.asarray("standing"),
        landmarks=landmarks,
    )
    index_path = tmp_path / "windows_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "session",
                "activity",
                "repeat",
                "window_index",
                "timestamp_ms",
                "person_present",
                "sync_ok",
            ),
        )
        writer.writeheader()
        for window_index in range(4):
            writer.writerow(
                {
                    "session": "P01_standing_01_20261103-test",
                    "activity": "standing",
                    "repeat": "01",
                    "window_index": window_index,
                    "timestamp_ms": 1000 + window_index * 200,
                    "person_present": 1,
                    "sync_ok": 0,
                }
            )

    data = load_dataset(windows, allow_unverified_sync=True)

    assert data.features.shape == (4, 20 * 224)
    assert data.coordinates[0, 0].tolist() == [0.5, 0.5]
    assert data.person_present.tolist() == [1, 1, 1, 1]
    assert data.timestamps_ms.tolist() == [1000, 1200, 1400, 1600]
    assert data.unverified_sessions_used == ["standing_01"]


def test_loader_keeps_only_verified_windows_and_validates_shapes(tmp_path):
    windows = tmp_path / "windows"
    _write_session(windows, "standing_01", sync_ok=True)
    _write_session(windows, "standing_02", sync_ok=False)

    data = load_dataset(windows)

    assert data.features.shape == (4, 448)
    assert set(data.session_ids) == {"standing_01"}
    assert data.rejected_unverified == ["standing_02.npz"]
    assert np.isnan(data.coordinates[0]).all()
    assert data.mask[0].sum() == 0


def test_session_held_out_metrics_and_prediction_pipeline(tmp_path):
    windows = tmp_path / "windows"
    for session_id in ("standing_01", "standing_02", "walking_01"):
        _write_session(windows, session_id)

    data = load_dataset(windows)
    summary, session_metrics, predictions = evaluate_leave_one_session_out(data)

    assert {row["session_id"] for row in session_metrics} == {
        "standing_01",
        "standing_02",
        "walking_01",
    }
    assert {row["model"] for row in summary} == {
        "training_mean",
        "ridge",
        "logistic_or_constant",
    }
    assert len(predictions) == 2 * len(data.features)

    output_dir = tmp_path / "output"
    run(windows, output_dir)
    model_path = output_dir / "pose_baseline.joblib"
    model = joblib.load(model_path)
    assert model["coordinate_system"] == "normalized_image_xy_[0,1]"
    assert model["sync_verified"] is True
    assert model["diagnostic_only"] is False

    inference_input = tmp_path / "inference.npz"
    np.savez_compressed(
        inference_input,
        csi=np.ones((1, 2, 224), dtype=np.float32),
        session=np.asarray("new_session"),
        timestamp_ms=np.asarray([1234.5]),
    )
    prediction_path = tmp_path / "predictions.csv"
    assert predict(inference_input, model_path, prediction_path) == 1
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["session_id"] == "new_session"
    assert rows[0]["timestamp_ms"] == "1234.5"
    assert rows[0]["person_present"] in {"0", "1"}
    assert rows[0]["sync_verified"] == "1"
    assert rows[0]["diagnostic_only"] == "0"


def test_strict_session_evaluation_requires_two_sessions(tmp_path):
    windows = tmp_path / "windows"
    _write_session(windows, "standing_01")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="at least two usable sessions"):
        run(windows, output_dir, evaluation_mode="session")
