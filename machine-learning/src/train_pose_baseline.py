"""Train a masked CSI-to-2D-pose baseline from pose windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOW_DIR = (
    ROOT
    / "signal-processing"
    / "data"
    / "pose"
    / "data"
    / "pose_windows_dataset"
    / "windows"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "machine-learning" / "results" / "pose-baseline-exploratory"
)
CSI_FEATURES = 224
RIDGE_ALPHA = 10.0
PCK_DIAGONAL_THRESHOLD = 0.05
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
POSE_LANDMARKS = [
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


@dataclass
class PoseDataset:
    features: np.ndarray
    coordinates: np.ndarray
    mask: np.ndarray
    person_present: np.ndarray
    session_ids: np.ndarray
    activities: np.ndarray
    window_indices: np.ndarray
    landmarks: list[str]
    window_shape: tuple[int, int]
    rejected_unverified: list[str]
    unverified_sessions_used: list[str]
    rejected_invalid_windows: int
    image_size: tuple[int, int]
    timestamps_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CSI-to-pose baseline. Unverified synchronization is "
            "rejected unless explicitly enabled for a diagnostic run."
        )
    )
    parser.add_argument("--window-dir", type=Path, default=DEFAULT_WINDOW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-unverified-sync",
        action="store_true",
        help=(
            "Allow windows marked sync_ok=0 for an exploratory diagnostic only. "
            "The resulting metrics are not valid pose-accuracy estimates."
        ),
    )
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--evaluation-mode",
        choices=("auto", "session", "none"),
        default="auto",
        help=(
            "auto uses leave-one-session-out CV when there are at least two "
            "sessions; session requires it; none skips evaluation."
        ),
    )
    return parser.parse_args()


def _scalar_text(value: np.ndarray) -> str:
    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def _sync_flags(value: np.ndarray, count: int, path: Path) -> np.ndarray:
    flags = np.asarray(value, dtype=bool)
    if flags.ndim == 0:
        return np.full(count, bool(flags), dtype=bool)
    if flags.shape != (count,):
        raise ValueError(f"{path.name}: sync_ok must be scalar or one value per window.")
    return flags


def _session_index_rows(
    index: pd.DataFrame | None,
    session: str,
    activity: str,
    path: Path,
    count: int,
) -> pd.DataFrame | None:
    if index is None:
        return None

    rows = index[index["session"].astype(str) == session]
    if rows.empty and {"activity", "repeat"}.issubset(index.columns):
        prefix = f"{activity}_"
        repeat = path.stem[len(prefix) :] if path.stem.startswith(prefix) else ""
        if repeat:
            indexed_repeats = index["repeat"].astype(str).str.lstrip("0")
            indexed_repeats = indexed_repeats.mask(indexed_repeats == "", "0")
            repeat_key = repeat.lstrip("0") or "0"
            rows = index[
                (index["activity"].astype(str) == activity)
                & (indexed_repeats == repeat_key)
            ]
    if rows.empty:
        raise ValueError(
            f"{path.name}: no matching session was found in windows_index.csv."
        )

    rows = rows.sort_values("window_index").reset_index(drop=True)
    if len(rows) != count:
        raise ValueError(
            f"{path.name}: windows_index.csv has {len(rows)} rows for this "
            f"session, but the NPZ has {count} windows."
        )
    if not (rows["activity"].astype(str) == activity).all():
        raise ValueError(f"{path.name}: activity labels conflict with windows_index.csv.")
    return rows


def load_dataset(
    window_dir: Path,
    allow_unverified_sync: bool = False,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
) -> PoseDataset:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    files = sorted(Path(window_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No pose window .npz files found in {window_dir}")

    feature_rows = []
    coordinate_rows = []
    mask_rows = []
    presence_rows = []
    session_rows = []
    activity_rows = []
    window_rows = []
    timestamp_rows = []
    landmarks: list[str] | None = None
    window_shape: tuple[int, int] | None = None
    rejected_unverified = []
    unverified_sessions_used = []
    rejected_invalid_windows = 0
    index_path = Path(window_dir).parent / "windows_index.csv"
    window_index = pd.read_csv(index_path) if index_path.is_file() else None
    if window_index is not None:
        required_index_columns = {"session", "activity", "window_index"}
        missing_index_columns = required_index_columns - set(window_index.columns)
        if missing_index_columns:
            raise ValueError(
                f"{index_path.name} is missing columns "
                f"{sorted(missing_index_columns)}."
            )

    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "csi",
                "mask",
                "sync_ok",
                "session",
                "activity",
                "landmarks",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(
                    f"{path.name} is missing {missing}; expected the delivered "
                    "CSI/pose window schema."
                )
            if not ({"keypoints", "keypoints_normalized"} & set(archive.files)):
                raise ValueError(
                    f"{path.name} must contain keypoints or keypoints_normalized."
                )

            csi = np.asarray(archive["csi"], dtype=np.float32)
            if "keypoints_normalized" in archive.files:
                coordinates = np.asarray(
                    archive["keypoints_normalized"], dtype=np.float32
                )
            else:
                coordinates = np.asarray(archive["keypoints"], dtype=np.float32)
                coordinates = coordinates / np.asarray(
                    [image_width, image_height], dtype=np.float32
                )
            mask = np.asarray(archive["mask"], dtype=bool)
            session = _scalar_text(archive["session"])
            activity = _scalar_text(archive["activity"])
            if "person_present" in archive.files:
                presence = np.asarray(archive["person_present"], dtype=np.uint8)
            else:
                presence = np.full(
                    len(csi), 0 if activity == "empty" else 1, dtype=np.uint8
                )
            source_landmarks = [
                _scalar_text(name) for name in np.asarray(archive["landmarks"])
            ]
            missing_landmarks = sorted(set(POSE_LANDMARKS) - set(source_landmarks))
            if missing_landmarks:
                raise ValueError(
                    f"{path.name}: pose labels are missing COCO-17 landmarks "
                    f"{missing_landmarks}."
                )
            landmark_indices = [
                source_landmarks.index(name) for name in POSE_LANDMARKS
            ]
            coordinates = coordinates[:, landmark_indices, :]
            mask = mask[:, landmark_indices]
            sync_ok = _sync_flags(archive["sync_ok"], len(csi), path)

        metadata_rows = _session_index_rows(
            window_index, session, activity, path, len(csi)
        )
        if metadata_rows is not None:
            if "person_present" in metadata_rows.columns:
                indexed_presence = metadata_rows["person_present"].to_numpy(
                    dtype=np.uint8
                )
                if not np.isin(indexed_presence, (0, 1)).all():
                    raise ValueError(
                        f"{path.name}: windows_index.csv person_present must "
                        "contain only 0 or 1."
                    )
                presence = indexed_presence
            if "sync_ok" in metadata_rows.columns:
                indexed_sync = metadata_rows["sync_ok"].to_numpy(dtype=np.uint8)
                if not np.isin(indexed_sync, (0, 1)).all():
                    raise ValueError(
                        f"{path.name}: windows_index.csv sync_ok must contain "
                        "only 0 or 1."
                    )
                sync_ok &= indexed_sync.astype(bool)

        if not sync_ok.all():
            rejected_unverified.append(path.name)
            if not allow_unverified_sync:
                continue
            unverified_sessions_used.append(session)
        if csi.ndim != 3 or csi.shape[2] != CSI_FEATURES:
            raise ValueError(
                f"{path.name}: csi must have shape [windows, time, {CSI_FEATURES}], "
                f"got {csi.shape}."
            )
        if coordinates.ndim != 3 or coordinates.shape[2] != 2:
            raise ValueError(
                f"{path.name}: keypoints_normalized must have shape "
                f"[windows, joints, 2], got {coordinates.shape}."
            )
        if mask.shape != coordinates.shape[:2]:
            raise ValueError(f"{path.name}: mask shape does not match keypoints.")
        if presence.shape != (len(csi),):
            raise ValueError(f"{path.name}: person_present must have one value per window.")
        if not np.isin(presence, (0, 1)).all():
            raise ValueError(f"{path.name}: person_present must contain only 0 or 1.")
        if window_shape is None:
            window_shape = (int(csi.shape[1]), int(csi.shape[2]))
            landmarks = list(POSE_LANDMARKS)
        elif csi.shape[1:] != window_shape:
            raise ValueError(
                f"{path.name}: CSI window shape differs from other sessions."
            )

        accepted_sync = np.ones_like(sync_ok) if allow_unverified_sync else sync_ok
        valid = accepted_sync & np.isfinite(csi).all(axis=(1, 2))
        invalid_targets = (mask & ~np.isfinite(coordinates).all(axis=2)).any(axis=1)
        valid &= ~invalid_targets
        rejected_invalid_windows += int((accepted_sync & ~valid).sum())
        if not valid.any():
            continue
        if np.any(mask[valid] & (presence[valid, None] == 0)):
            raise ValueError(
                f"{path.name}: absent-person windows must not have valid keypoints."
            )

        selected_coordinates = coordinates[valid].copy()
        selected_mask = mask[valid].copy()
        selected_coordinates[~selected_mask] = np.nan
        selected_csi = csi[valid]
        count = len(selected_csi)
        feature_rows.append(selected_csi.reshape(count, -1))
        coordinate_rows.append(selected_coordinates)
        mask_rows.append(selected_mask)
        presence_rows.append(presence[valid])
        session_rows.extend([session] * count)
        activity_rows.extend([activity] * count)
        selected_window_indices = np.flatnonzero(valid)
        window_rows.extend(selected_window_indices.tolist())
        if metadata_rows is not None and "timestamp_ms" in metadata_rows.columns:
            timestamp_rows.extend(
                metadata_rows.loc[selected_window_indices, "timestamp_ms"]
                .to_numpy(dtype=np.float64)
                .tolist()
            )
        else:
            timestamp_rows.extend([np.nan] * count)

    if not feature_rows:
        skipped = ", ".join(rejected_unverified) or "none"
        raise ValueError(
            "No synchronized, valid pose windows are available. "
            f"Rejected unverified sessions: {skipped}. "
            "For a diagnostic-only run, explicitly pass "
            "--allow-unverified-sync; do not report those scores as valid "
            "pose accuracy."
        )

    timestamps_ms = np.asarray(timestamp_rows, dtype=np.float64)

    session_activities: dict[str, set[str]] = {}
    for session, activity in zip(session_rows, activity_rows):
        session_activities.setdefault(session, set()).add(activity)
    conflicting = {
        session: sorted(activities)
        for session, activities in session_activities.items()
        if len(activities) != 1
    }
    if conflicting:
        raise ValueError(
            f"Sessions have conflicting activity labels: {conflicting}"
        )

    return PoseDataset(
        features=np.concatenate(feature_rows),
        coordinates=np.concatenate(coordinate_rows),
        mask=np.concatenate(mask_rows),
        person_present=np.concatenate(presence_rows),
        session_ids=np.asarray(session_rows, dtype=str),
        activities=np.asarray(activity_rows, dtype=str),
        window_indices=np.asarray(window_rows, dtype=np.int64),
        landmarks=landmarks or [],
        window_shape=window_shape or (0, 0),
        rejected_unverified=rejected_unverified,
        unverified_sessions_used=sorted(set(unverified_sessions_used)),
        rejected_invalid_windows=rejected_invalid_windows,
        image_size=(image_width, image_height),
        timestamps_ms=timestamps_ms,
    )


def _fit_models(
    features: np.ndarray,
    coordinates: np.ndarray,
    mask: np.ndarray,
    person_present: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    joint_count = coordinates.shape[1]
    coordinate_means = np.full((joint_count, 2), np.nan, dtype=np.float32)
    regressors: dict[int, Ridge] = {}

    for joint in range(joint_count):
        valid = (
            (person_present == 1)
            & mask[:, joint]
            & np.isfinite(coordinates[:, joint]).all(axis=1)
        )
        if not valid.any():
            continue
        coordinate_means[joint] = coordinates[valid, joint].mean(axis=0)
        if int(valid.sum()) >= 2:
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(scaled[valid], coordinates[valid, joint])
            regressors[joint] = model

    presence_model: Any
    if np.unique(person_present).size > 1:
        presence_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    else:
        presence_model = DummyClassifier(strategy="most_frequent")
    presence_model.fit(scaled, person_present)

    if not regressors:
        raise ValueError(
            "Training split has fewer than two valid person-present examples "
            "for every joint; cannot fit pose regressors."
        )
    return {
        "scaler": scaler,
        "coordinate_means": coordinate_means,
        "coordinate_models": regressors,
        "presence_model": presence_model,
    }


def _coordinate_predictions(
    model: dict[str, Any],
    features: np.ndarray,
    model_name: str = "ridge",
) -> np.ndarray:
    means = model["coordinate_means"]
    if model_name == "training_mean":
        return np.repeat(means[None, :, :], len(features), axis=0)
    if model_name != "ridge":
        raise ValueError(f"Unknown coordinate model: {model_name}")
    scaled = model["scaler"].transform(features)
    predictions = np.repeat(means[None, :, :], len(features), axis=0)
    for joint, regressor in model["coordinate_models"].items():
        predictions[:, joint, :] = regressor.predict(scaled)
    return predictions


def _presence_predictions(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    scaled = model["scaler"].transform(features)
    return np.asarray(model["presence_model"].predict(scaled), dtype=np.uint8)


def _pose_metrics(
    coordinates: np.ndarray,
    mask: np.ndarray,
    presence: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    valid = (
        mask
        & (presence[:, None] == 1)
        & np.isfinite(coordinates).all(axis=2)
        & np.isfinite(predictions).all(axis=2)
    )
    errors = np.linalg.norm(predictions - coordinates, axis=2) / np.sqrt(2.0)
    selected = errors[valid]
    return {
        "valid_joint_count": int(valid.sum()),
        "mean_error_image_diagonal": float(selected.mean()) if selected.size else float("nan"),
        "pck_at_005_image_diagonal": (
            float(np.mean(selected <= PCK_DIAGONAL_THRESHOLD))
            if selected.size
            else float("nan")
        ),
    }


def _presence_metrics(
    truth: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "presence_accuracy": float(accuracy_score(truth, predictions)),
        "presence_balanced_accuracy": (
            float(balanced_accuracy_score(truth, predictions))
            if np.unique(truth).size > 1
            else float("nan")
        ),
        "presence_f1": float(f1_score(truth, predictions, zero_division=0)),
    }


def _prediction_rows(
    data: PoseDataset,
    indices: np.ndarray,
    model_name: str,
    coordinates: np.ndarray,
    presence: np.ndarray,
    fold: str,
) -> list[dict[str, Any]]:
    rows = []
    for local_index, source_index in enumerate(indices):
        row: dict[str, Any] = {
            "evaluation": fold,
            "coordinate_model": model_name,
            "session_id": data.session_ids[source_index],
            "activity": data.activities[source_index],
            "window_index": int(data.window_indices[source_index]),
            "timestamp_ms": (
                float(data.timestamps_ms[source_index])
                if np.isfinite(data.timestamps_ms[source_index])
                else ""
            ),
            "person_present_true": int(data.person_present[source_index]),
            "person_present_predicted": int(presence[local_index]),
        }
        for joint, landmark in enumerate(data.landmarks):
            for axis, suffix in enumerate(("x", "y")):
                truth = data.coordinates[source_index, joint, axis]
                prediction = coordinates[local_index, joint, axis]
                row[f"{landmark}_{suffix}_true"] = (
                    float(truth) if np.isfinite(truth) else ""
                )
                row[f"{landmark}_{suffix}_predicted"] = (
                    float(prediction) if np.isfinite(prediction) else ""
                )
            row[f"{landmark}_valid"] = int(data.mask[source_index, joint])
        rows.append(row)
    return rows


def evaluate_leave_one_session_out(
    data: PoseDataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = np.unique(data.session_ids)
    if len(groups) < 2:
        raise ValueError("Session-held-out evaluation requires at least two sessions.")

    splits = LeaveOneGroupOut()
    prediction_rows: list[dict[str, Any]] = []
    metric_parts: dict[str, dict[str, list[Any]]] = {
        name: {"coordinates": [], "mask": [], "presence": [], "prediction": []}
        for name in ("training_mean", "ridge")
    }
    session_metrics = []
    split_rows = []

    for fold_number, (train, test) in enumerate(
        splits.split(data.features, groups=data.session_ids), start=1
    ):
        held_session = str(data.session_ids[test[0]])
        model = _fit_models(
            data.features[train],
            data.coordinates[train],
            data.mask[train],
            data.person_present[train],
        )
        ridge_predictions = _coordinate_predictions(model, data.features[test])
        mean_predictions = np.repeat(
            model["coordinate_means"][None, :, :], len(test), axis=0
        )
        present_predictions = _presence_predictions(model, data.features[test])

        split_rows.append(
            {
                "fold": fold_number,
                "session_id": held_session,
                "samples_test": len(test),
                "sessions_train": len(np.unique(data.session_ids[train])),
            }
        )
        for model_name, coordinate_predictions in (
            ("training_mean", mean_predictions),
            ("ridge", ridge_predictions),
        ):
            prediction_rows.extend(
                _prediction_rows(
                    data,
                    test,
                    model_name,
                    coordinate_predictions,
                    present_predictions,
                    f"held_out_session_{fold_number}",
                )
            )
            parts = metric_parts[model_name]
            parts["coordinates"].append(data.coordinates[test])
            parts["mask"].append(data.mask[test])
            parts["presence"].append(data.person_present[test])
            parts["prediction"].append(coordinate_predictions)

            pose_metrics = _pose_metrics(
                data.coordinates[test],
                data.mask[test],
                data.person_present[test],
                coordinate_predictions,
            )
            session_metrics.append(
                {
                    "session_id": held_session,
                    "coordinate_model": model_name,
                    "samples": len(test),
                    **pose_metrics,
                    **_presence_metrics(data.person_present[test], present_predictions),
                }
            )

    summary = []
    for model_name, parts in metric_parts.items():
        coordinates = np.concatenate(parts["coordinates"])
        masks = np.concatenate(parts["mask"])
        presence = np.concatenate(parts["presence"])
        predictions = np.concatenate(parts["prediction"])
        summary.append(
            {
                "metric_group": "pose",
                "model": model_name,
                "sessions": len(groups),
                "samples": len(data.features),
                **_pose_metrics(coordinates, masks, presence, predictions),
            }
        )

    presence_rows = [
        row for row in prediction_rows if row["coordinate_model"] == "ridge"
    ]
    truth = np.asarray(
        [row["person_present_true"] for row in presence_rows], dtype=np.uint8
    )
    predicted = np.asarray(
        [row["person_present_predicted"] for row in presence_rows], dtype=np.uint8
    )
    summary.append(
        {
            "metric_group": "presence",
            "model": "logistic_or_constant",
            "sessions": len(groups),
            "samples": len(truth),
            **_presence_metrics(truth, predicted),
        }
    )
    return summary, session_metrics, prediction_rows


def _fit_final_model(
    data: PoseDataset,
    selected_coordinate_model: str,
) -> dict[str, Any]:
    fitted = _fit_models(
        data.features,
        data.coordinates,
        data.mask,
        data.person_present,
    )
    return {
        **fitted,
        "landmarks": data.landmarks,
        "window_shape": data.window_shape,
        "image_size": data.image_size,
        "coordinate_system": "normalized_image_xy_[0,1]",
        "model_type": "standard_scaler_ridge_per_joint_logistic_presence",
        "selected_coordinate_model": selected_coordinate_model,
        "sync_verified": not bool(data.unverified_sessions_used),
        "diagnostic_only": bool(data.unverified_sessions_used),
        "unverified_sessions": data.unverified_sessions_used,
    }


def run(
    window_dir: Path,
    output_dir: Path,
    evaluation_mode: str = "auto",
    allow_unverified_sync: bool = False,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
) -> Path:
    data = load_dataset(
        window_dir,
        allow_unverified_sync=allow_unverified_sync,
        image_width=image_width,
        image_height=image_height,
    )
    session_count = len(np.unique(data.session_ids))
    should_evaluate = evaluation_mode == "session" or (
        evaluation_mode == "auto" and session_count >= 2
    )
    if evaluation_mode == "session" and session_count < 2:
        raise ValueError(
            "Session-held-out evaluation requires at least two usable sessions."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    if should_evaluate:
        summary, session_metrics, predictions = evaluate_leave_one_session_out(data)
        pd.DataFrame(summary).to_csv(run_dir / "summary_metrics.csv", index=False)
        pd.DataFrame(session_metrics).to_csv(
            run_dir / "per_session_metrics.csv", index=False
        )
        if predictions:
            pd.DataFrame(predictions).to_csv(
                run_dir / "out_of_fold_predictions.csv", index=False
            )
        pd.DataFrame(
            [
                {
                    "session_id": session,
                    "evaluation_role": "held_out_once",
                }
                for session in sorted(np.unique(data.session_ids))
            ]
        ).to_csv(run_dir / "session_splits.csv", index=False)
        evaluation = (
            "leave_one_session_out_diagnostic_unverified_sync"
            if data.unverified_sessions_used
            else "leave_one_session_out_cross_validation"
        )
        pose_scores = {
            row["model"]: row["mean_error_image_diagonal"]
            for row in summary
            if row["metric_group"] == "pose"
        }
        mean_score = pose_scores.get("training_mean", float("nan"))
        ridge_score = pose_scores.get("ridge", float("nan"))
        selected_coordinate_model = (
            "ridge"
            if np.isfinite(ridge_score)
            and (not np.isfinite(mean_score) or ridge_score <= mean_score)
            else "training_mean"
        )
    else:
        pd.DataFrame(
            [
                {
                    "metric_group": "not_evaluated",
                    "model": "",
                    "sessions": session_count,
                    "samples": len(data.features),
                    "reason": "Fewer than two usable sessions are available.",
                }
            ]
        ).to_csv(run_dir / "summary_metrics.csv", index=False)
        pd.DataFrame(
            columns=["session_id", "evaluation_role"]
        ).to_csv(run_dir / "session_splits.csv", index=False)
        pd.DataFrame(
            columns=[
                "evaluation",
                "coordinate_model",
                "session_id",
                "activity",
                "window_index",
                "person_present_true",
                "person_present_predicted",
            ]
        ).to_csv(run_dir / "out_of_fold_predictions.csv", index=False)
        pd.DataFrame(
            columns=[
                "session_id",
                "coordinate_model",
                "samples",
                "valid_joint_count",
                "mean_error_image_diagonal",
                "pck_at_005_image_diagonal",
                "presence_accuracy",
                "presence_balanced_accuracy",
                "presence_f1",
            ]
        ).to_csv(run_dir / "per_session_metrics.csv", index=False)
        evaluation = "not_evaluated"
        selected_coordinate_model = "ridge"

    model_package = _fit_final_model(data, selected_coordinate_model)
    model_package["model_version"] = run_id
    joblib.dump(model_package, run_dir / "pose_baseline.joblib")

    metadata = {
        "model_version": run_id,
        "target": "single-person 2D pose and person presence",
        "sessions": sorted(np.unique(data.session_ids).tolist()),
        "activities": sorted(np.unique(data.activities).tolist()),
        "session_count": session_count,
        "sample_count": len(data.features),
        "window_shape": list(data.window_shape),
        "image_size": {
            "width": data.image_size[0],
            "height": data.image_size[1],
            "source": (
                "signal-processing/docs/report_nguoi2.md"
                if data.image_size
                == (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
                else "CLI arguments"
            ),
        },
        "feature_count": int(data.features.shape[1]),
        "landmarks": data.landmarks,
        "coordinate_system": "normalized_image_xy_[0,1]",
        "selected_coordinate_model": selected_coordinate_model,
        "selection_rule": (
            "Choose the lower leave-one-session-out mean keypoint error; "
            "if evaluation is unavailable, retain Ridge as the baseline."
            if should_evaluate
            else "Evaluation unavailable; retain Ridge as the baseline."
        ),
        "pose_metric": (
            "mean Euclidean keypoint error divided by image diagonal; "
            "PCK threshold is 0.05 of image diagonal"
        ),
        "evaluation": evaluation,
        "sync_policy": (
            "Unverified sync_ok=0 windows were explicitly allowed for this "
            "diagnostic run; metrics are not valid pose-accuracy estimates."
            if data.unverified_sessions_used
            else "Only samples from NPZ files with sync_ok=1 are loaded."
        ),
        "unverified_window_files": data.rejected_unverified,
        "rejected_unverified_files": (
            [] if data.unverified_sessions_used else data.rejected_unverified
        ),
        "unverified_sessions_used": data.unverified_sessions_used,
        "rejected_invalid_windows": data.rejected_invalid_windows,
        "limitations": [
            "Leave-one-session-out cross-validation is not a separate final test set.",
            "Results apply only to the recorded participant and fixed capture setup.",
            "Pose coordinates are pseudo-labels from the video pose detector.",
            "The model is a baseline and does not establish real-world pose accuracy.",
            *(
                [
                    "All included windows have unverified synchronization; "
                    "reported scores are diagnostic only."
                ]
                if data.unverified_sessions_used
                else []
            ),
        ],
    }
    (run_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name in (
        "summary_metrics.csv",
        "per_session_metrics.csv",
        "out_of_fold_predictions.csv",
        "session_splits.csv",
        "pose_baseline.joblib",
        "model_metadata.json",
    ):
        source = run_dir / name
        if source.exists():
            target = output_dir / name
            target.write_bytes(source.read_bytes())

    print(f"Sessions: {session_count}; usable windows: {len(data.features)}")
    print(f"Evaluation: {evaluation}")
    print(f"Model: {output_dir / 'pose_baseline.joblib'}")
    print(f"Run snapshot: {run_dir}")
    if data.rejected_unverified:
        print(
            (
                "Included unverified sessions for diagnostic only: "
                + ", ".join(data.unverified_sessions_used)
                if data.unverified_sessions_used
                else "Excluded unverified sessions: "
                + ", ".join(data.rejected_unverified)
            )
        )
    if data.unverified_sessions_used:
        print(
            "WARNING: all pose metrics are diagnostic only because "
            "synchronization is unverified."
        )
    if not should_evaluate:
        print("Warning: model fitted for technical use; no independent evaluation.")
    return run_dir


def main() -> None:
    args = parse_args()
    run(
        args.window_dir,
        args.output_dir,
        args.evaluation_mode,
        allow_unverified_sync=args.allow_unverified_sync,
        image_width=args.image_width,
        image_height=args.image_height,
    )


if __name__ == "__main__":
    main()
