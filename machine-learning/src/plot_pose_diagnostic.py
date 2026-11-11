"""Plot the saved CSI-to-pose diagnostic metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    ROOT
    / "machine-learning"
    / "results"
    / "pose-sync-check"
    / "combined-all-rounds"
    / "training"
)
ACTIVITIES = ("empty", "standing", "left_arm", "right_arm", "walking", "mixed")
MODEL_LABELS = {
    "training_mean": "Mean-pose baseline",
    "ridge": "Ridge",
}
MODEL_COLORS = {
    "training_mean": "#4C78A8",
    "ridge": "#F58518",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a chart from the saved pose-baseline metrics."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: <results-dir>/pose_baseline_diagnostic.png).",
    )
    return parser.parse_args()


def _activity_label(session_id: str) -> str:
    for activity in ACTIVITIES:
        if f"_{activity}_" in session_id or session_id.endswith(f"_{activity}"):
            return activity
    return session_id


def _bar_values(axis: plt.Axes, values: list[float], labels: list[str]) -> None:
    bars = axis.bar(labels, values, color=["#4C78A8", "#F58518"])
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.margins(y=0.2)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def create_chart(results_dir: Path, output_path: Path) -> None:
    summary = pd.read_csv(results_dir / "summary_metrics.csv")
    per_session = pd.read_csv(results_dir / "per_session_metrics.csv")
    metadata = json.loads((results_dir / "model_metadata.json").read_text(encoding="utf-8"))

    pose = summary[summary["metric_group"] == "pose"].set_index("model")
    expected_models = list(MODEL_LABELS)
    missing_models = set(expected_models) - set(pose.index)
    if missing_models:
        raise ValueError(f"Missing pose model metrics: {', '.join(sorted(missing_models))}")

    diagnostic_only = (
        "unverified" in str(metadata.get("evaluation", "")).lower()
        or metadata.get("diagnostic_only", False)
        or not metadata.get("sync_verified", True)
    )
    sync_note = (
        "DIAGNOSTIC ONLY — CSI/video synchronization is unverified (sync_ok=0)"
        if diagnostic_only
        else "Pose baseline evaluation"
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 1.25]}
    )
    figure.suptitle(
        "CSI → 2D pose baseline (Diagnostic)" if diagnostic_only else "CSI → 2D pose baseline",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.945,
        sync_note,
        ha="center",
        color="#B22222" if diagnostic_only else "#333333",
        fontsize=11,
        fontweight="bold",
    )

    model_names = [MODEL_LABELS[name] for name in expected_models]
    errors = [float(pose.loc[name, "mean_error_image_diagonal"]) for name in expected_models]
    _bar_values(axes[0, 0], errors, model_names)
    axes[0, 0].set_title("Mean joint error (lower is better)")
    axes[0, 0].set_ylabel("Fraction of image diagonal")

    pck = [float(pose.loc[name, "pck_at_005_image_diagonal"]) for name in expected_models]
    _bar_values(axes[0, 1], pck, model_names)
    axes[0, 1].set_title("PCK@0.05 (higher is better)")
    axes[0, 1].set_ylabel("Fraction of valid joints")
    axes[0, 1].set_ylim(0, 1)

    presence = summary[summary["metric_group"] == "presence"]
    presence_axis = axes[1, 0]
    if presence.empty:
        presence_axis.text(0.5, 0.5, "Presence metrics unavailable", ha="center", va="center")
        presence_axis.set_axis_off()
    else:
        row = presence.iloc[0]
        presence_labels = ["Accuracy", "Balanced\naccuracy", "F1"]
        presence_values = [
            float(row["presence_accuracy"]),
            float(row["presence_balanced_accuracy"]),
            float(row["presence_f1"]),
        ]
        bars = presence_axis.bar(
            presence_labels,
            presence_values,
            color=["#72B7B2", "#E45756", "#54A24B"],
        )
        presence_axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        presence_axis.set_ylim(0, 1)
        presence_axis.set_title("Person-presence metrics")
        presence_axis.set_ylabel("Score")
        presence_axis.grid(axis="y", alpha=0.25)
        presence_axis.set_axisbelow(True)

    session_axis = axes[1, 1]
    session_rows = per_session[per_session["coordinate_model"].isin(expected_models)].copy()
    session_rows["activity"] = session_rows["session_id"].map(_activity_label)
    activities = list(dict.fromkeys(session_rows["activity"]))
    y_positions = list(range(len(activities)))
    bar_height = 0.36
    for offset, model in enumerate(expected_models):
        values_by_activity = session_rows[
            session_rows["coordinate_model"] == model
        ].groupby("activity")["mean_error_image_diagonal"].mean()
        values = [float(values_by_activity.get(activity, float("nan"))) for activity in activities]
        positions = [position + (offset - 0.5) * bar_height for position in y_positions]
        bars = session_axis.barh(
            positions,
            values,
            height=bar_height,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                session_axis.annotate(
                    f"{value:.3f}",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
    session_axis.set_yticks(y_positions, activities)
    session_axis.invert_yaxis()
    session_axis.set_title("Mean joint error by activity")
    session_axis.set_xlabel("Fraction of image diagonal")
    largest_error = session_rows["mean_error_image_diagonal"].max()
    if pd.notna(largest_error):
        session_axis.set_xlim(0, largest_error * 1.25)
    session_axis.grid(axis="x", alpha=0.25)
    session_axis.set_axisbelow(True)
    session_axis.legend(fontsize=8)

    session_count = per_session["session_id"].nunique()
    figure.text(
        0.5,
        0.025,
        f"{session_count} recording session(s); empty has no labeled joints. "
        "Presence scores are sensitive to class imbalance. "
        "All scores are exploratory diagnostics; CSI/video synchronization is unverified.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.03, 0.06, 0.98, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_path = args.output or args.results_dir / "pose_baseline_diagnostic.png"
    create_chart(args.results_dir, output_path)
    print(f"Saved diagnostic chart: {output_path}")


if __name__ == "__main__":
    main()
