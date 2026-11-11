"""Predict activity labels from processed CSI feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_activity_baseline import (
    DEFAULT_OUTPUT_DIR,
    FRAME_FEATURES,
    KEY_COLUMNS,
    TEMPORAL_FEATURES,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = (
    ROOT / "signal-processing" / "data" / "pose" / "data" / "processed"
)
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "activity_classifier_prototype.joblib"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-frame activity labels from processed CSI features."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing frame and temporal feature CSV files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Trained activity_classifier_prototype.joblib file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="CSV file to write frame-level predictions to.",
    )
    return parser.parse_args()


def load_inference_features(processed_dir: Path, feature_columns: list[str]) -> pd.DataFrame:
    frame_path = processed_dir / "preliminary_frame_features.csv"
    temporal_path = processed_dir / "temporal_features.csv"
    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame feature CSV not found: {frame_path}")
    if not temporal_path.is_file():
        raise FileNotFoundError(f"Temporal feature CSV not found: {temporal_path}")

    frames = pd.read_csv(frame_path)
    temporal = pd.read_csv(temporal_path)
    required_frame = {*KEY_COLUMNS, "frame_complete", *FRAME_FEATURES}
    required_temporal = {*KEY_COLUMNS, *TEMPORAL_FEATURES}
    missing_frame = sorted(required_frame - set(frames.columns))
    missing_temporal = sorted(required_temporal - set(temporal.columns))
    if missing_frame:
        raise ValueError(f"{frame_path.name} is missing columns: {missing_frame}")
    if missing_temporal:
        raise ValueError(
            f"{temporal_path.name} is missing columns: {missing_temporal}"
        )

    complete_values = frames["frame_complete"].astype(str).str.strip().str.lower()
    complete_map = {"true": True, "false": False, "1": True, "0": False}
    frames["frame_complete"] = complete_values.map(complete_map)
    if frames["frame_complete"].isna().any():
        raise ValueError("frame_complete contains values other than true/false.")
    frames = frames.loc[frames["frame_complete"]].copy()

    for name, table in (("frame features", frames), ("temporal features", temporal)):
        if table[KEY_COLUMNS].isna().any().any():
            raise ValueError(f"{name} contains missing session or frame keys.")
        if table.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} has duplicate (session_id, frame_id) rows.")

    merged = frames.merge(
        temporal,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_frame", "_temporal"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching complete frames across the feature tables.")

    for column in feature_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = merged[column].replace([np.inf, -np.inf], np.nan)

    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def predict_activity(
    feature_rows: pd.DataFrame,
    model_package: dict,
) -> pd.DataFrame:
    """Return one CSI-derived activity prediction for every complete feature row."""
    required_package = {"model", "feature_columns", "labels"}
    missing_package = required_package - set(model_package)
    if missing_package:
        raise ValueError(
            f"Model package is missing fields: {sorted(missing_package)}"
        )

    feature_columns = model_package["feature_columns"]
    missing_features = sorted(set(feature_columns) - set(feature_rows.columns))
    if missing_features:
        raise ValueError(f"Input is missing model features: {missing_features}")
    if feature_rows.empty:
        raise ValueError("No complete feature rows are available for prediction.")

    predictions = model_package["model"].predict(feature_rows[feature_columns])
    output = feature_rows[KEY_COLUMNS].copy()
    output["predicted_activity"] = predictions
    output["model_validation_status"] = model_package.get(
        "evaluation_mode", "not_evaluated"
    )
    output["model_version"] = model_package.get("model_version", "unknown")
    return output


def run(processed_dir: Path, model_path: Path, output_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model_package = joblib.load(model_path)
    feature_rows = load_inference_features(
        processed_dir, model_package.get("feature_columns", [])
    )
    predictions = predict_activity(feature_rows, model_package)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predicted frames: {len(predictions)}")
    print(f"Model version: {model_package.get('model_version', 'unknown')}")
    print(
        "Evaluation mode: "
        f"{model_package.get('evaluation_mode', 'not recorded')}"
    )
    print(f"Output: {output_path}")
    if model_package.get("evaluation_mode") != "session_held_out":
        print(
            "Warning: evaluation used time blocks from sessions also used in "
            "training; predictions are exploratory."
        )


def main() -> None:
    args = parse_args()
    run(args.processed_dir, args.model, args.output)


if __name__ == "__main__":
    main()
