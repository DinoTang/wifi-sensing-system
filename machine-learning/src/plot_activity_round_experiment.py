"""Plot the leave-one-round-out CSI activity-classification results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    ROOT
    / "machine-learning"
    / "results"
    / "activity-round-experiment"
    / "round-heldout-comparison"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "activity_round_results.png"
ROUNDS = ("Lan1", "Lan2", "Lan3")
ACTIVITIES = ("empty", "standing", "left_arm", "right_arm", "walking", "mixed")
MODELS = ("dummy_most_frequent", "logistic_regression", "random_forest")
MODEL_LABELS = {
    "dummy_most_frequent": "Dummy baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
MODEL_COLORS = {
    "dummy_most_frequent": "#A0A0A0",
    "logistic_regression": "#4C78A8",
    "random_forest": "#F58518",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the saved leave-one-round-out activity results."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_round = pd.read_csv(results_dir / "per_round_metrics.csv")
    per_class = pd.read_csv(results_dir / "per_class_recall.csv")
    predictions = pd.read_csv(results_dir / "round_heldout_predictions.csv")
    pooled = pd.read_csv(results_dir / "round_heldout_metrics.csv")

    required_per_round = {
        "signal", "model", "features", "held_out_round", "balanced_accuracy",
        "macro_f1",
    }
    required_per_class = {
        "signal", "model", "features", "held_out_round", "label", "recall",
    }
    required_predictions = {
        "signal", "model", "features", "held_out_round", "actual", "predicted",
    }
    for table, required, name in (
        (per_round, required_per_round, "per_round_metrics.csv"),
        (per_class, required_per_class, "per_class_recall.csv"),
        (predictions, required_predictions, "round_heldout_predictions.csv"),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    return per_round, per_class, predictions, pooled


def create_chart(results_dir: Path, output_path: Path) -> None:
    per_round, per_class, predictions, pooled = _read_results(results_dir)
    selected = per_round.loc[
        (per_round["signal"] == "raw")
        & (per_round["features"] == "frame_plus_temporal")
        & per_round["model"].isin(MODELS)
        & per_round["held_out_round"].isin(ROUNDS)
    ]
    if len(selected) != len(MODELS) * len(ROUNDS):
        raise ValueError("Expected raw frame-plus-temporal results for all models and rounds.")

    recall = per_class.loc[
        (per_class["signal"] == "raw")
        & (per_class["model"] == "random_forest")
        & (per_class["features"] == "frame_plus_temporal")
        & per_class["held_out_round"].isin(ROUNDS)
        & per_class["label"].isin(ACTIVITIES)
    ]
    if len(recall) != len(ROUNDS) * len(ACTIVITIES):
        raise ValueError("Expected Random Forest recall for all activities and rounds.")

    selected_predictions = predictions.loc[
        (predictions["signal"] == "raw")
        & (predictions["model"] == "random_forest")
        & (predictions["features"] == "frame_plus_temporal")
        & predictions["held_out_round"].isin(ROUNDS)
    ]
    if selected_predictions.empty:
        raise ValueError("No matching held-out predictions are available.")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle(
        "CSI activity classification — leave-one-round-out evaluation",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    metric_axis = axes[0, 0]
    x = np.arange(len(ROUNDS))
    width = 0.24
    for offset, model in enumerate(MODELS):
        rows = selected.loc[selected["model"] == model].set_index("held_out_round")
        values = [float(rows.loc[round_name, "balanced_accuracy"]) for round_name in ROUNDS]
        bars = metric_axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        metric_axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    metric_axis.axhline(
        1 / len(ACTIVITIES),
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Dummy / 6-class reference (16.7%)",
    )
    metric_axis.set_xticks(x, ROUNDS)
    metric_axis.set_ylim(0, 0.72)
    metric_axis.set_ylabel("Balanced accuracy")
    metric_axis.set_title("Performance on each held-out recording round")
    metric_axis.grid(axis="y", alpha=0.25)
    metric_axis.set_axisbelow(True)
    metric_axis.legend(fontsize=8, loc="upper right")

    heat_axis = axes[0, 1]
    heat = (
        recall.pivot(index="label", columns="held_out_round", values="recall")
        .reindex(index=ACTIVITIES, columns=ROUNDS)
        .to_numpy(dtype=float)
    )
    image = heat_axis.imshow(heat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    heat_axis.set_xticks(range(len(ROUNDS)), ROUNDS)
    heat_axis.set_yticks(range(len(ACTIVITIES)), ACTIVITIES)
    heat_axis.set_title("Random Forest recall by activity and held-out round")
    for row in range(len(ACTIVITIES)):
        for column in range(len(ROUNDS)):
            color = "white" if heat[row, column] >= 0.6 else "#222222"
            heat_axis.text(
                column,
                row,
                f"{heat[row, column]:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04, label="Recall")

    matrix_axis = axes[1, 0]
    actual = selected_predictions["actual"].astype(str)
    predicted = selected_predictions["predicted"].astype(str)
    matrix = confusion_matrix(actual, predicted, labels=ACTIVITIES, normalize="true")
    matrix_image = matrix_axis.imshow(
        matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto"
    )
    matrix_axis.set_xticks(range(len(ACTIVITIES)), ACTIVITIES, rotation=35, ha="right")
    matrix_axis.set_yticks(range(len(ACTIVITIES)), ACTIVITIES)
    matrix_axis.set_xlabel("Predicted activity")
    matrix_axis.set_ylabel("Actual activity")
    matrix_axis.set_title("Random Forest confusion matrix (row-normalized)")
    for row in range(len(ACTIVITIES)):
        for column in range(len(ACTIVITIES)):
            color = "white" if matrix[row, column] >= 0.55 else "#222222"
            matrix_axis.text(
                column,
                row,
                f"{matrix[row, column]:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    figure.colorbar(matrix_image, ax=matrix_axis, fraction=0.046, pad=0.04)

    filter_axis = axes[1, 1]
    forest = per_round.loc[
        (per_round["model"] == "random_forest")
        & (per_round["features"] == "frame_plus_temporal")
        & per_round["held_out_round"].isin(ROUNDS)
    ]
    if set(forest["signal"]) != {"raw", "moving_average_5"}:
        raise ValueError("Expected raw and moving_average_5 Random Forest results.")
    filter_width = 0.34
    for offset, signal_name, label, color in (
        (-0.5, "raw", "Raw CSI features", "#4C78A8"),
        (0.5, "moving_average_5", "5-frame moving average", "#E45756"),
    ):
        rows = forest.loc[forest["signal"] == signal_name].set_index("held_out_round")
        values = [float(rows.loc[round_name, "balanced_accuracy"]) for round_name in ROUNDS]
        bars = filter_axis.bar(
            x + offset * filter_width,
            values,
            filter_width,
            label=label,
            color=color,
        )
        filter_axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    filter_axis.set_xticks(x, ROUNDS)
    filter_axis.set_ylim(0, 0.72)
    filter_axis.set_ylabel("Balanced accuracy")
    filter_axis.set_title("Filtering did not improve cross-round results")
    filter_axis.grid(axis="y", alpha=0.25)
    filter_axis.set_axisbelow(True)
    filter_axis.legend(fontsize=8)

    pooled_rf = pooled.loc[
        (pooled["signal"] == "raw")
        & (pooled["model"] == "random_forest")
        & (pooled["features"] == "frame_plus_temporal")
    ]
    if len(pooled_rf) != 1:
        raise ValueError("Expected exactly one pooled Random Forest summary row.")
    summary = pooled_rf.iloc[0]
    figure.text(
        0.5,
        0.025,
        (
            f"18 sessions, 3 held-out rounds, {int(summary['frames']):,} CSI frames; "
            f"pooled RF balanced accuracy={summary['balanced_accuracy']:.1%}, "
            f"macro-F1={summary['macro_f1']:.1%}. Session labels are repeated on frames; "
            "frames within a session are correlated. Lan1 activity labels are used, "
            "but Lan1 CSI/video pose alignment remains unverified."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
        wrap=True,
    )
    figure.tight_layout(rect=(0.03, 0.07, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    create_chart(args.results_dir, args.output)
    print(f"Saved activity classification chart: {args.output}")


if __name__ == "__main__":
    main()
