"""Evaluate exploratory CSI classification baselines with session-held-out CV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = (
    ROOT / "signal-processing" / "data" / "intel" / "data" / "processed"
)
DEFAULT_OUTPUT_DIR = ROOT / "machine-learning" / "results" / "baseline"

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
SUPPORTED_LABELS = {"room_empty", "person_static", "person_moving"}

TASKS = {
    "three_class": ["room_empty", "person_static", "person_moving"],
    "presence": ["room_empty", "person_present"],
}
CHART_MODEL = "random_forest"
CHART_FEATURE_SET = "frame_plus_temporal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Logistic Regression and Random Forest baselines using "
            "leave-one-session-out evaluation."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing pipeline feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for out-of-fold predictions and evaluation CSVs.",
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
    required_frame_columns = {
        *KEY_COLUMNS,
        "scenario_label",
        "frame_complete",
        *FRAME_FEATURES,
    }
    required_temporal_columns = {
        *KEY_COLUMNS,
        "scenario_label",
        *TEMPORAL_FEATURES,
    }
    missing_frame = sorted(required_frame_columns - set(frames.columns))
    missing_temporal = sorted(required_temporal_columns - set(temporal.columns))
    if missing_frame:
        raise ValueError(f"{frame_path.name} is missing columns: {missing_frame}")
    if missing_temporal:
        raise ValueError(
            f"{temporal_path.name} is missing columns: {missing_temporal}"
        )

    complete_values = (
        frames["frame_complete"].astype(str).str.strip().str.lower()
    )
    complete_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    frames["frame_complete"] = complete_values.map(complete_map)
    if frames["frame_complete"].isna().any():
        raise ValueError("frame_complete contains values other than true/false.")
    frames = frames.loc[frames["frame_complete"]].copy()

    for name, table in (("frame features", frames), ("temporal features", temporal)):
        if table.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} has duplicate (session_id, frame_id) rows.")
        if table[KEY_COLUMNS + ["scenario_label"]].isna().any().any():
            raise ValueError(f"{name} contains missing session, frame, or label.")
        unexpected = sorted(set(table["scenario_label"]) - SUPPORTED_LABELS)
        if unexpected:
            raise ValueError(f"{name} has unsupported labels: {unexpected}")
        labels_per_session = table.groupby("session_id")[
            "scenario_label"
        ].nunique()
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
        raise ValueError("No matching complete frames across the two feature tables.")
    if not merged["scenario_label_frame"].equals(
        merged["scenario_label_temporal"]
    ):
        raise ValueError("Frame and temporal feature labels do not match.")
    merged = merged.rename(columns={"scenario_label_frame": "scenario_label"})
    merged = merged.drop(columns=["scenario_label_temporal", "frame_complete"])

    session_labels = merged.groupby("session_id")["scenario_label"].nunique()
    if (session_labels != 1).any():
        raise ValueError("Merged features assign multiple labels to one session.")

    available_sessions = merged.groupby("scenario_label")["session_id"].nunique()
    missing_repeats = {
        label: int(available_sessions.get(label, 0))
        for label in sorted(SUPPORTED_LABELS)
        if available_sessions.get(label, 0) < 2
    }
    if missing_repeats:
        raise ValueError(
            "Each class needs at least two independent sessions for "
            f"leave-one-session-out evaluation; found {missing_repeats}."
        )

    feature_columns = FRAME_FEATURES + TEMPORAL_FEATURES
    for column in feature_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = merged[column].replace([np.inf, -np.inf], np.nan)

    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def make_model(model_name: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if model_name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            )
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
    task_name: str,
    model_name: str,
    feature_set: str,
    feature_columns: list[str],
    labels: list[str],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    working = data.copy()
    if task_name == "presence":
        working["target"] = working["scenario_label"].replace(
            {
                "room_empty": "room_empty",
                "person_static": "person_present",
                "person_moving": "person_present",
            }
        )
    else:
        working["target"] = working["scenario_label"]

    x = working[feature_columns]
    y = working["target"]
    groups = working["session_id"]
    logo = LeaveOneGroupOut()
    predictions: list[dict] = []
    session_scores: list[dict] = []
    confusion_rows: list[dict] = []
    seen_test_sessions: set[str] = set()

    for train_indices, test_indices in logo.split(x, y, groups):
        x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        train_sessions = set(groups.iloc[train_indices])
        test_sessions = set(groups.iloc[test_indices])
        if train_sessions.intersection(test_sessions):
            raise AssertionError("Session leakage detected in cross-validation.")
        if set(labels) - set(y_train):
            raise ValueError(
                "A training fold is missing a target class; collect more "
                "independent sessions before evaluating this task."
            )

        model = make_model(model_name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        test_session = str(next(iter(test_sessions)))
        seen_test_sessions.add(test_session)

        for row_index, truth, prediction in zip(test_indices, y_test, y_pred):
            predictions.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "feature_set": feature_set,
                    "session_id": str(working.iloc[row_index]["session_id"]),
                    "frame_id": int(working.iloc[row_index]["frame_id"]),
                    "label_true": truth,
                    "label_predicted": prediction,
                }
            )

        session_scores.append(
            {
                "task": task_name,
                "model": model_name,
                "feature_set": feature_set,
                "session_id": test_session,
                "held_out_label": str(y_test.iloc[0]),
                "frames": len(test_indices),
                "accuracy": accuracy_score(y_test, y_pred),
                "held_out_class_recall": accuracy_score(y_test, y_pred),
            }
        )

        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        for actual_index, actual_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "task": task_name,
                        "model": model_name,
                        "feature_set": feature_set,
                        "session_id": test_session,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[actual_index, predicted_index]),
                    }
                )

    if seen_test_sessions != set(groups.astype(str)):
        raise AssertionError("Not every session received exactly one held-out fold.")

    y_true = [row["label_true"] for row in predictions]
    y_predicted = [row["label_predicted"] for row in predictions]
    summary = {
        "task": task_name,
        "model": model_name,
        "feature_set": feature_set,
        "sessions": len(seen_test_sessions),
        "frames": len(predictions),
        "pooled_accuracy": accuracy_score(y_true, y_predicted),
        "pooled_balanced_accuracy": balanced_accuracy_score(
            y_true, y_predicted
        ),
        "pooled_macro_f1": f1_score(
            y_true, y_predicted, labels=labels, average="macro", zero_division=0
        ),
        "mean_session_accuracy": float(
            np.mean([row["accuracy"] for row in session_scores])
        ),
    }
    return summary, predictions, session_scores, confusion_rows


def save_charts(
    predictions: list[dict],
    session_scores: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Save pooled confusion matrices and held-out accuracy by session."""
    selected_predictions = [
        row
        for row in predictions
        if row["model"] == CHART_MODEL
        and row["feature_set"] == CHART_FEATURE_SET
    ]
    selected_scores = [
        row
        for row in session_scores
        if row["model"] == CHART_MODEL
        and row["feature_set"] == CHART_FEATURE_SET
    ]
    available_tasks = {row["task"] for row in selected_predictions}
    if available_tasks != set(TASKS):
        raise ValueError(
            "Chart input must include Random Forest frame_plus_temporal "
            f"predictions for {sorted(TASKS)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = output_dir / "random_forest_confusion_matrices.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, task_name in zip(axes, TASKS):
        labels = TASKS[task_name]
        task_rows = [
            row for row in selected_predictions if row["task"] == task_name
        ]
        matrix = confusion_matrix(
            [row["label_true"] for row in task_rows],
            [row["label_predicted"] for row in task_rows],
            labels=labels,
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        image = axis.imshow(percentages, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted label",
            ylabel="Actual label",
            title=task_name.replace("_", " ").title(),
        )
        axis.tick_params(axis="x", labelrotation=25)
        threshold = percentages.max() / 2 if percentages.size else 0
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]}\n"
                    f"({percentages[row_index, column_index]:.1%})",
                    ha="center",
                    va="center",
                    color="white"
                    if percentages[row_index, column_index] > threshold
                    else "black",
                )
    fig.colorbar(image, ax=axes, label="Share of actual class")
    fig.suptitle(
        "Random Forest · frame + temporal features\n"
        "Pooled held-out predictions by recording session"
    )
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    session_path = output_dir / "random_forest_session_accuracy.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, task_name in zip(axes, TASKS):
        task_rows = sorted(
            (row for row in selected_scores if row["task"] == task_name),
            key=lambda row: (row["held_out_label"], row["session_id"]),
        )
        if not task_rows:
            raise ValueError(f"No session scores available for task {task_name}.")
        labels = [row["session_id"] for row in task_rows]
        scores = [row["accuracy"] for row in task_rows]
        bars = axis.barh(labels, scores, color="#4472C4")
        axis.set(
            xlim=(0.0, 1.0),
            xlabel="Accuracy on the held-out session",
            title=task_name.replace("_", " ").title(),
        )
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        for bar, score in zip(bars, scores):
            axis.text(
                min(score + 0.015, 0.91),
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=8,
            )
    fig.suptitle(
        "Random Forest · frame + temporal features\n"
        "Each bar is one whole held-out recording session"
    )
    fig.savefig(session_path, dpi=160)
    plt.close(fig)
    return [confusion_path, session_path]


def run(processed_dir: Path, output_dir: Path) -> None:
    data = load_features(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = {
        "temporal_only": TEMPORAL_FEATURES,
        "frame_plus_temporal": FRAME_FEATURES + TEMPORAL_FEATURES,
    }
    summaries = []
    all_predictions = []
    all_session_scores = []
    all_confusion_rows = []

    for task_name, labels in TASKS.items():
        for model_name in ("logistic_regression", "random_forest"):
            for feature_set, columns in feature_sets.items():
                summary, predictions, session_scores, confusion_rows = evaluate(
                    data=data,
                    task_name=task_name,
                    model_name=model_name,
                    feature_set=feature_set,
                    feature_columns=columns,
                    labels=labels,
                )
                summaries.append(summary)
                all_predictions.extend(predictions)
                all_session_scores.extend(session_scores)
                all_confusion_rows.extend(confusion_rows)

    pd.DataFrame(summaries).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(
        output_dir / "out_of_fold_predictions.csv", index=False
    )
    pd.DataFrame(all_session_scores).to_csv(
        output_dir / "per_session_metrics.csv", index=False
    )
    pd.DataFrame(all_confusion_rows).to_csv(
        output_dir / "confusion_matrices.csv", index=False
    )
    chart_paths = save_charts(
        predictions=all_predictions,
        session_scores=all_session_scores,
        output_dir=output_dir,
    )

    print(f"[SAVE] {output_dir / 'summary_metrics.csv'}")
    print(f"[SAVE] {output_dir / 'out_of_fold_predictions.csv'}")
    print(f"[SAVE] {output_dir / 'per_session_metrics.csv'}")
    print(f"[SAVE] {output_dir / 'confusion_matrices.csv'}")
    for chart_path in chart_paths:
        print(f"[SAVE] {chart_path}")
    print()
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print(
        "\nExploratory result only: frames in a held-out session are not "
        "independent samples, and the number of sessions is small."
    )


def main() -> None:
    args = parse_args()
    run(args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
