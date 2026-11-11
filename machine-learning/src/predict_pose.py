"""Predict person presence and 2D keypoints from preprocessed CSI windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from train_pose_baseline import (
    DEFAULT_OUTPUT_DIR,
    _coordinate_predictions,
    _presence_predictions,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "pose_baseline.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict pose from an NPZ containing CSI windows [N,T,224]."
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions.csv",
    )
    return parser.parse_args()


def predict(input_path: Path, model_path: Path, output_path: Path) -> int:
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")
    model = joblib.load(model_path)
    required_model = {
        "scaler",
        "coordinate_means",
        "coordinate_models",
        "presence_model",
        "landmarks",
        "window_shape",
    }
    missing_model = sorted(required_model - set(model))
    if missing_model:
        raise ValueError(f"Model package is missing fields: {missing_model}")

    with np.load(input_path, allow_pickle=False) as archive:
        if "csi" not in archive:
            raise ValueError(f"{input_path.name} must contain a 'csi' array.")
        csi = np.asarray(archive["csi"], dtype=np.float32)
        if csi.ndim == 2:
            csi = csi[None, ...]
        session_id = ""
        if "session" in archive:
            value = np.asarray(archive["session"]).item()
            session_id = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        timestamps = (
            np.asarray(archive["timestamp_ms"], dtype=np.float64)
            if "timestamp_ms" in archive
            else np.full(len(csi), np.nan)
        )

    if csi.ndim != 3 or tuple(csi.shape[1:]) != tuple(model["window_shape"]):
        raise ValueError(
            "CSI windows must match the model window shape "
            f"{tuple(model['window_shape'])}; got {csi.shape}."
        )
    if not np.isfinite(csi).all():
        raise ValueError("CSI input contains NaN or infinite values.")
    if timestamps.shape != (len(csi),):
        raise ValueError("timestamp_ms must contain one value per CSI window.")

    features = csi.reshape(len(csi), -1)
    coordinates = _coordinate_predictions(
        model,
        features,
        model.get("selected_coordinate_model", "ridge"),
    )
    presence = _presence_predictions(model, features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "session_id",
        "window_index",
        "timestamp_ms",
        "person_present",
        "model_version",
        "sync_verified",
        "diagnostic_only",
    ]
    for landmark in model["landmarks"]:
        fieldnames.extend([f"{landmark}_x", f"{landmark}_y"])

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, is_present in enumerate(presence):
            row = {
                "session_id": session_id,
                "window_index": index,
                "timestamp_ms": (
                    float(timestamps[index]) if np.isfinite(timestamps[index]) else ""
                ),
                "person_present": int(is_present),
                "model_version": model.get("model_version", "unknown"),
                "sync_verified": int(model.get("sync_verified", False)),
                "diagnostic_only": int(model.get("diagnostic_only", True)),
            }
            for joint, landmark in enumerate(model["landmarks"]):
                for axis, suffix in enumerate(("x", "y")):
                    value = coordinates[index, joint, axis]
                    row[f"{landmark}_{suffix}"] = (
                        float(value)
                        if is_present and np.isfinite(value)
                        else ""
                    )
            writer.writerow(row)

    print(f"Predicted windows: {len(csi)}")
    print(f"Model version: {model.get('model_version', 'unknown')}")
    print(f"Output: {output_path}")
    return len(csi)


def main() -> None:
    args = parse_args()
    predict(args.input_npz, args.model, args.output)


if __name__ == "__main__":
    main()
