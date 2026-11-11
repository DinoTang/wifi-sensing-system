"""Train an exploratory CSI activity classifier on the pose-package captures."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = (
    ROOT / "signal-processing" / "data" / "pose" / "data" / "processed"
)
DEFAULT_OUTPUT_DIR = ROOT / "machine-learning" / "results" / "activity-baseline"

FRAME_FEATURES = [
    "mean_amplitude",
    "std_amplitude",
    "max_amplitude",
    "mean_rssi",
    "motion_score",
]
TEMPORAL_FEATURES = [
    "temporal_std_amplitude",
    "temporal_std_amplitude_max",
]
KEY_COLUMNS = ["session_id", "frame_id"]
LABELS = ["empty", "standing", "left_arm", "right_arm", "walking", "mixed"]
FEATURE_SETS = {
    "temporal_only": TEMPORAL_FEATURES,
    "frame_plus_temporal": FRAME_FEATURES + TEMPORAL_FEATURES,
}
MODELS = ["dummy_most_frequent", "logistic_regression", "random_forest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exploratory activity classifiers using guarded, "
            "within-session time-block evaluation."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing frame and temporal feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, plots, and the prototype model.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of contiguous held-out time blocks per session.",
    )
    parser.add_argument(
        "--guard-frames",
        type=int,
        default=20,
        help="Training frames excluded around each test block.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["auto", "session", "time-block"],
        default="auto",
        help=(
            "auto uses session-held-out validation when every class has at "
            "least two sessions; otherwise it runs a within-session diagnostic."
        ),
    )
    return parser.parse_args()


def load_features(processed_dir: Path) -> pd.DataFrame:
    frame_path = processed_dir / "preliminary_frame_features.csv"
    temporal_path = processed_dir / "temporal_features.csv"
    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame feature CSV not found: {frame_path}")
    if not temporal_path.is_file():
        raise FileNotFoundError(f"Temporal feature CSV not found: {temporal_path}")

    frames = pd.read_csv(frame_path)
    temporal = pd.read_csv(temporal_path)
    required_frame = {
        *KEY_COLUMNS,
        "scenario_label",
        "frame_complete",
        *FRAME_FEATURES,
    }
    required_temporal = {
        *KEY_COLUMNS,
        "scenario_label",
        *TEMPORAL_FEATURES,
    }
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
        if table.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} has duplicate (session_id, frame_id) rows.")
        if table[KEY_COLUMNS + ["scenario_label"]].isna().any().any():
            raise ValueError(f"{name} contains missing session, frame, or label.")
        unexpected = sorted(set(table["scenario_label"]) - set(LABELS))
        if unexpected:
            raise ValueError(f"{name} has unsupported labels: {unexpected}")
        labels_per_session = table.groupby("session_id")["scenario_label"].nunique()
        if (labels_per_session != 1).any():
            raise ValueError(f"{name} assigns multiple labels to one session.")

    merged = frames.merge(
        temporal,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_frame", "_temporal"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching complete frames across the feature tables.")
    if not merged["scenario_label_frame"].equals(
        merged["scenario_label_temporal"]
    ):
        raise ValueError("Frame and temporal feature labels do not match.")
    merged = merged.rename(columns={"scenario_label_frame": "scenario_label"})
    merged = merged.drop(columns=["scenario_label_temporal", "frame_complete"])

    present_labels = set(merged["scenario_label"])
    if present_labels != set(LABELS):
        raise ValueError(
            f"Expected all six activity labels; found {sorted(present_labels)}."
        )

    for column in FRAME_FEATURES + TEMPORAL_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = merged[column].replace([np.inf, -np.inf], np.nan)

    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def blocked_time_splits(
    data: pd.DataFrame, folds: int, guard_frames: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split contiguous blocks within each session, removing a temporal guard."""
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    if guard_frames < 0:
        raise ValueError("--guard-frames cannot be negative.")

    session_positions = []
    data = data.reset_index(drop=True)
    for _, group in data.groupby("session_id", sort=True):
        positions = group.sort_values("frame_id").index.to_numpy()
        if len(positions) < folds * (guard_frames * 2 + 2):
            raise ValueError(
                "Each session needs enough frames for the requested folds and "
                f"guard; got {len(positions)} frames."
            )
        session_positions.append((positions, np.array_split(positions, folds)))

    splits = []
    for fold in range(folds):
        test_parts = [blocks[fold] for _, blocks in session_positions]
        test_indices = np.sort(np.concatenate(test_parts))
        test_mask = np.zeros(len(data), dtype=bool)
        test_mask[test_indices] = True
        train_mask = ~test_mask

        for positions, blocks in session_positions:
            block = blocks[fold]
            start = int(np.searchsorted(positions, block[0]))
            end = int(np.searchsorted(positions, block[-1]))
            guarded = positions[
                max(0, start - guard_frames) : min(
                    len(positions), end + guard_frames + 1
                )
            ]
            train_mask[guarded] = False

        train_indices = np.flatnonzero(train_mask)
        if set(data.iloc[train_indices]["scenario_label"]) != set(LABELS):
            raise ValueError(
                f"Fold {fold} training split does not contain all six labels."
            )
        if np.intersect1d(train_indices, test_indices).size:
            raise AssertionError("Training and test rows overlap.")
        splits.append((train_indices, test_indices))
    return splits


def session_held_out_splits(
    data: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out each entire recording session once."""
    data = data.reset_index(drop=True)
    session_counts = data.groupby("scenario_label")["session_id"].nunique()
    insufficient = {
        label: int(session_counts.get(label, 0))
        for label in LABELS
        if session_counts.get(label, 0) < 2
    }
    if insufficient:
        raise ValueError(
            "Session-held-out evaluation needs at least two sessions per "
            f"activity; found {insufficient}."
        )

    splits = []
    for session_id in sorted(data["session_id"].unique()):
        test_mask = data["session_id"].eq(session_id).to_numpy()
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)
        train_labels = set(data.iloc[train_indices]["scenario_label"])
        if train_labels != set(LABELS):
            raise ValueError(
                f"Training data without session {session_id} is missing "
                f"activities: {sorted(set(LABELS) - train_labels)}."
            )
        splits.append((train_indices, test_indices))
    return splits


def choose_splits(
    data: pd.DataFrame,
    evaluation_mode: str,
    folds: int,
    guard_frames: int,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    session_counts = data.groupby("scenario_label")["session_id"].nunique()
    can_hold_out_sessions = all(
        session_counts.get(label, 0) >= 2 for label in LABELS
    )
    if evaluation_mode == "session" or (
        evaluation_mode == "auto" and can_hold_out_sessions
    ):
        return "session_held_out", session_held_out_splits(data)
    return "within_session_time_block_diagnostic", blocked_time_splits(
        data, folds, guard_frames
    )


def make_model(model_name: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "dummy_most_frequent":
        steps.append(("classifier", DummyClassifier(strategy="prior")))
    elif model_name == "logistic_regression":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
    elif model_name == "random_forest":
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(steps)


def evaluate(
    data: pd.DataFrame,
    folds: int = 5,
    guard_frames: int = 20,
    evaluation_mode: str = "auto",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    actual_evaluation_mode, splits = choose_splits(
        data, evaluation_mode, folds, guard_frames
    )
    summaries = []
    predictions = []
    session_scores = []
    importances = []

    for model_name in MODELS:
        for feature_set, feature_columns in FEATURE_SETS.items():
            model_predictions = []
            for fold_number, (train_indices, test_indices) in enumerate(splits, 1):
                train = data.iloc[train_indices]
                test = data.iloc[test_indices]
                model = make_model(model_name)
                model.fit(train[feature_columns], train["scenario_label"])
                predicted = model.predict(test[feature_columns])

                for row_index, label_true, label_predicted in zip(
                    test_indices, test["scenario_label"], predicted
                ):
                    row = data.iloc[row_index]
                    prediction = {
                        "model": model_name,
                        "feature_set": feature_set,
                        "fold": fold_number,
                        "evaluation_mode": actual_evaluation_mode,
                        "session_id": row["session_id"],
                        "frame_id": int(row["frame_id"]),
                        "label_true": label_true,
                        "label_predicted": label_predicted,
                    }
                    predictions.append(prediction)
                    model_predictions.append(prediction)

                classifier = model.named_steps["classifier"]
                if model_name == "random_forest":
                    importances.extend(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "fold": fold_number,
                            "feature": feature,
                            "importance": float(importance),
                        }
                        for feature, importance in zip(
                            feature_columns, classifier.feature_importances_
                        )
                    )

            y_true = [row["label_true"] for row in model_predictions]
            y_pred = [row["label_predicted"] for row in model_predictions]
            summaries.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "evaluation_mode": actual_evaluation_mode,
                    "sessions": data["session_id"].nunique(),
                    "folds": len(splits),
                    "guard_frames": (
                        guard_frames
                        if actual_evaluation_mode
                        == "within_session_time_block_diagnostic"
                        else 0
                    ),
                    "frames": len(model_predictions),
                    "pooled_accuracy": accuracy_score(y_true, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(
                        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
                    ),
                }
            )

            for session_id in sorted(data["session_id"].unique()):
                session_rows = [
                    row for row in model_predictions if row["session_id"] == session_id
                ]
                session_true = [row["label_true"] for row in session_rows]
                session_pred = [row["label_predicted"] for row in session_rows]
                session_scores.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "evaluation_mode": actual_evaluation_mode,
                        "session_id": session_id,
                        "label": session_true[0],
                        "frames": len(session_rows),
                        "accuracy": accuracy_score(session_true, session_pred),
                    }
                )

    return summaries, predictions, session_scores, importances


def confusion_rows(predictions: list[dict]) -> list[dict]:
    rows = []
    configurations = sorted(
        {(row["model"], row["feature_set"]) for row in predictions}
    )
    for model_name, feature_set in configurations:
        selected = [
            row
            for row in predictions
            if row["model"] == model_name and row["feature_set"] == feature_set
        ]
        matrix = confusion_matrix(
            [row["label_true"] for row in selected],
            [row["label_predicted"] for row in selected],
            labels=LABELS,
        )
        for actual_index, actual_label in enumerate(LABELS):
            for predicted_index, predicted_label in enumerate(LABELS):
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[actual_index, predicted_index]),
                    }
                )
    return rows


def save_confusion_chart(
    predictions: list[dict], output_dir: Path, evaluation_mode: str
) -> Path:
    path = output_dir / "random_forest_confusion_matrices.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    selected = [
        row for row in predictions if row["model"] == "random_forest"
    ]
    for axis, feature_set in zip(axes, FEATURE_SETS):
        rows = [row for row in selected if row["feature_set"] == feature_set]
        matrix = confusion_matrix(
            [row["label_true"] for row in rows],
            [row["label_predicted"] for row in rows],
            labels=LABELS,
        )
        totals = matrix.sum(axis=1, keepdims=True)
        shares = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix, dtype=float),
            where=totals != 0,
        )
        image = axis.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set(
            xticks=np.arange(len(LABELS)),
            yticks=np.arange(len(LABELS)),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted activity",
            ylabel="Actual activity",
            title=feature_set.replace("_", " ").title(),
        )
        axis.tick_params(axis="x", labelrotation=35)
        for row_index in range(len(LABELS)):
            for column_index in range(len(LABELS)):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]}\n"
                    f"({shares[row_index, column_index]:.1%})",
                    ha="center",
                    va="center",
                    color="white" if shares[row_index, column_index] > 0.5 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=axes, label="Share of actual activity")
    fig.suptitle(
        "Random Forest · " + evaluation_mode.replace("_", " ") + "\n"
        "Activity labels from recording-session notes"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(
    processed_dir: Path,
    output_dir: Path,
    folds: int = 5,
    guard_frames: int = 20,
    evaluation_mode: str = "auto",
) -> None:
    data = load_features(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, predictions, session_scores, importances = evaluate(
        data,
        folds=folds,
        guard_frames=guard_frames,
        evaluation_mode=evaluation_mode,
    )
    actual_evaluation_mode = summaries[0]["evaluation_mode"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_dir = output_dir / "runs"
    run_dir = runs_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{run_id}-{suffix}"
        suffix += 1

    pd.DataFrame(summaries).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        output_dir / "out_of_fold_predictions.csv", index=False
    )
    pd.DataFrame(session_scores).to_csv(
        output_dir / "per_session_metrics.csv", index=False
    )
    pd.DataFrame(confusion_rows(predictions)).to_csv(
        output_dir / "confusion_matrices.csv", index=False
    )
    pd.DataFrame(importances).to_csv(
        output_dir / "random_forest_feature_importance.csv", index=False
    )
    save_confusion_chart(predictions, output_dir, actual_evaluation_mode)

    prototype = make_model("random_forest")
    feature_columns = FEATURE_SETS["frame_plus_temporal"]
    prototype.fit(data[feature_columns], data["scenario_label"])
    sessions_per_activity = {
        str(label): int(count)
        for label, count in data.groupby("scenario_label")["session_id"]
        .nunique()
        .items()
    }
    joblib.dump(
        {
            "model": prototype,
            "feature_columns": feature_columns,
            "labels": LABELS,
            "training_sessions": sorted(data["session_id"].unique().tolist()),
            "sessions_per_activity": sessions_per_activity,
            "evaluation_mode": actual_evaluation_mode,
            "model_version": run_id,
            "warning": (
                "Activity classifier; not a pose estimator. "
                "Review evaluation metadata before interpreting generalization."
            ),
        },
        output_dir / "activity_classifier_prototype.joblib",
    )

    metadata = {
        "model_version": run_id,
        "target": "six session-level activity labels",
        "labels": LABELS,
        "feature_columns": feature_columns,
        "training_sessions": sorted(data["session_id"].unique().tolist()),
        "sessions_per_activity": sessions_per_activity,
        "evaluation": actual_evaluation_mode,
        "folds": len({row["fold"] for row in predictions}),
        "guard_frames": (
            guard_frames
            if actual_evaluation_mode == "within_session_time_block_diagnostic"
            else 0
        ),
        "limitations": [],
    }
    if actual_evaluation_mode == "within_session_time_block_diagnostic":
        metadata["limitations"].extend(
            [
                "Evaluation blocks come from sessions also represented in training.",
                "Scores are diagnostic only and do not estimate new-session performance.",
            ]
        )
    metadata["limitations"].extend(
        [
            "Session-level activity labels are copied to each frame.",
            "The mixed activity has no event-level transition labels.",
            "This model classifies activities and does not predict joint coordinates.",
        ]
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    run_dir.mkdir(parents=True)
    artifact_names = [
        "summary_metrics.csv",
        "out_of_fold_predictions.csv",
        "per_session_metrics.csv",
        "confusion_matrices.csv",
        "random_forest_feature_importance.csv",
        "random_forest_confusion_matrices.png",
        "activity_classifier_prototype.joblib",
        "model_metadata.json",
    ]
    for artifact_name in artifact_names:
        shutil.copy2(output_dir / artifact_name, run_dir / artifact_name)
    history_path = output_dir / "run_history.csv"
    history_row = {
        "run_id": run_dir.name,
        "evaluation_mode": actual_evaluation_mode,
        "sessions": data["session_id"].nunique(),
        "frames": len(data),
        "random_forest_frame_plus_temporal_balanced_accuracy": next(
            row["balanced_accuracy"]
            for row in summaries
            if row["model"] == "random_forest"
            and row["feature_set"] == "frame_plus_temporal"
        ),
        "model_snapshot": str(
            (run_dir / "activity_classifier_prototype.joblib").relative_to(
                output_dir
            )
        ),
    }
    history = (
        pd.read_csv(history_path)
        if history_path.exists()
        else pd.DataFrame(columns=history_row.keys())
    )
    history = pd.concat([history, pd.DataFrame([history_row])], ignore_index=True)
    history.to_csv(history_path, index=False)

    print(pd.DataFrame(summaries).to_string(index=False, float_format="%.3f"))
    print(f"\nEvaluation mode: {actual_evaluation_mode}")
    print(f"Latest outputs: {output_dir}")
    print(f"Immutable run snapshot: {run_dir}")
    if actual_evaluation_mode == "within_session_time_block_diagnostic":
        print(
            "Caution: these scores use held-out time blocks from sessions also "
            "used in training; they do not measure new-session performance."
        )


def main() -> None:
    args = parse_args()
    run(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        folds=args.folds,
        guard_frames=args.guard_frames,
        evaluation_mode=args.evaluation_mode,
    )


if __name__ == "__main__":
    main()
