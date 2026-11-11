"""Plot verified CSI-to-pose metrics and held-out pose examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "machine-learning" / "results" / "pose-sync-check" / "final-csi-rebuild"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "combined-verified" / "training"
ACTIVITIES = ("empty", "standing", "left_arm", "right_arm", "walking", "mixed")
MODELS = ("training_mean", "ridge")
MODEL_LABELS = {
    "training_mean": "Training-mean pose",
    "ridge": "CSI Ridge",
}
MODEL_COLORS = {
    "training_mean": "#4C78A8",
    "ridge": "#F58518",
}
LANDMARKS = (
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
)
POSE_CONNECTIONS = (
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)
IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the verified Lan2/Lan3 pose metrics and a held-out skeleton comparison."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _activity_from_session(session_id: str) -> str | None:
    for activity in ACTIVITIES:
        if f"_{activity}_" in session_id or session_id.endswith(f"_{activity}"):
            return activity
    return None


def _read_required_csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required results file not found: {path}")
    table = pd.read_csv(path)
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return table


def _weighted_activity_errors(per_session: pd.DataFrame) -> pd.DataFrame:
    pose_rows = per_session.loc[
        per_session["coordinate_model"].isin(MODELS)
        & (per_session["valid_joint_count"] > 0)
    ].copy()
    pose_rows["activity"] = pose_rows["session_id"].map(_activity_from_session)
    pose_rows = pose_rows.loc[pose_rows["activity"].notna()]
    if pose_rows.empty:
        raise ValueError("No valid pose sessions found in per_session_metrics.csv.")

    def weighted_mean(group: pd.DataFrame, column: str) -> float:
        weights = group["valid_joint_count"].to_numpy(dtype=float)
        values = group[column].to_numpy(dtype=float)
        return float(np.average(values, weights=weights))

    rows = []
    for (activity, model), group in pose_rows.groupby(
        ["activity", "coordinate_model"], sort=False
    ):
        rows.append(
            {
                "activity": activity,
                "model": model,
                "mean_error_image_diagonal": weighted_mean(
                    group, "mean_error_image_diagonal"
                ),
                "pck_at_005_image_diagonal": weighted_mean(
                    group, "pck_at_005_image_diagonal"
                ),
                "sessions": int(group["session_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_dashboard(
    results_dir: Path,
    output_path: Path,
) -> None:
    summary = _read_required_csv(
        results_dir / "combined-verified" / "training" / "summary_metrics.csv",
        {
            "metric_group",
            "model",
            "sessions",
            "samples",
            "mean_error_image_diagonal",
            "pck_at_005_image_diagonal",
            "presence_accuracy",
            "presence_balanced_accuracy",
            "presence_f1",
        },
    )
    per_session = _read_required_csv(
        results_dir / "combined-verified" / "training" / "per_session_metrics.csv",
        {
            "session_id",
            "coordinate_model",
            "valid_joint_count",
            "mean_error_image_diagonal",
            "pck_at_005_image_diagonal",
        },
    )
    cross_l2_l3 = _read_required_csv(
        results_dir / "cross-round-Lan2-to-Lan3" / "metrics.csv",
        {
            "train_round",
            "test_round",
            "coordinate_model",
            "samples",
            "mean_error_image_diagonal",
            "pck_at_005_image_diagonal",
        },
    )
    cross_l3_l2 = _read_required_csv(
        results_dir / "cross-round-Lan3-to-Lan2" / "metrics.csv",
        {
            "train_round",
            "test_round",
            "coordinate_model",
            "samples",
            "mean_error_image_diagonal",
            "pck_at_005_image_diagonal",
        },
    )
    pose_summary = summary.loc[summary["metric_group"] == "pose"].set_index("model")
    if not set(MODELS).issubset(pose_summary.index):
        raise ValueError("Both training_mean and ridge summary rows are required.")
    activity_metrics = _weighted_activity_errors(per_session)

    cross_round = pd.concat([cross_l2_l3, cross_l3_l2], ignore_index=True)
    cross_round = cross_round.loc[cross_round["coordinate_model"].isin(MODELS)].copy()
    if len(cross_round) != 4:
        raise ValueError("Expected both pose models in both cross-round directions.")

    presence = summary.loc[summary["metric_group"] == "presence"]
    if len(presence) != 1:
        raise ValueError("Expected one pooled person-presence summary row.")
    presence = presence.iloc[0]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle(
        "CSI-to-pose evaluation — synchronized Lan2 + Lan3",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    overall_axis = axes[0, 0]
    x = np.arange(2)
    width = 0.34
    error_values = [
        float(pose_summary.loc[model, "mean_error_image_diagonal"])
        for model in MODELS
    ]
    pck_values = [
        float(pose_summary.loc[model, "pck_at_005_image_diagonal"])
        for model in MODELS
    ]
    bars = overall_axis.bar(
        x - width / 2,
        error_values,
        width,
        label="Mean joint error (lower is better)",
        color="#E45756",
    )
    overall_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    bars = overall_axis.bar(
        x + width / 2,
        pck_values,
        width,
        label="PCK@0.05 (higher is better)",
        color="#54A24B",
    )
    overall_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    overall_axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    overall_axis.set_ylim(0, 0.82)
    overall_axis.set_ylabel("Score / fraction of image diagonal")
    overall_axis.set_title("Leave-one-session-out across 12 verified sessions")
    overall_axis.grid(axis="y", alpha=0.25)
    overall_axis.set_axisbelow(True)
    overall_axis.legend(fontsize=8, loc="upper right")

    activity_axis = axes[0, 1]
    valid_activities = [
        activity
        for activity in ACTIVITIES
        if activity_metrics["activity"].eq(activity).any()
    ]
    positions = np.arange(len(valid_activities))
    bar_width = 0.36
    for offset, model in enumerate(MODELS):
        model_rows = activity_metrics.loc[
            activity_metrics["model"] == model
        ].set_index("activity")
        values = [
            float(model_rows.loc[activity, "mean_error_image_diagonal"])
            if activity in model_rows.index
            else np.nan
            for activity in valid_activities
        ]
        bars = activity_axis.bar(
            positions + (offset - 0.5) * bar_width,
            values,
            bar_width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        activity_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    activity_axis.set_xticks(positions, valid_activities, rotation=25, ha="right")
    activity_axis.set_ylabel("Mean joint error / image diagonal")
    activity_axis.set_title("Pose error by activity (valid labeled joints only)")
    activity_axis.grid(axis="y", alpha=0.25)
    activity_axis.set_axisbelow(True)
    activity_axis.legend(fontsize=8)

    cross_axis = axes[1, 0]
    round_order = [
        ("Lan2", "Lan3"),
        ("Lan3", "Lan2"),
    ]
    x_round = np.arange(len(round_order))
    for offset, model in enumerate(MODELS):
        values = []
        for train_round, test_round in round_order:
            row = cross_round.loc[
                (cross_round["train_round"] == train_round)
                & (cross_round["test_round"] == test_round)
                & (cross_round["coordinate_model"] == model)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Missing cross-round row for {train_round}->{test_round}, {model}."
                )
            values.append(float(row.iloc[0]["mean_error_image_diagonal"]))
        bars = cross_axis.bar(
            x_round + (offset - 0.5) * bar_width,
            values,
            bar_width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        cross_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    cross_axis.set_xticks(x_round, [f"{a} → {b}" for a, b in round_order])
    cross_axis.set_ylabel("Mean joint error / image diagonal")
    cross_axis.set_title("Train on one round, test on the other")
    cross_axis.grid(axis="y", alpha=0.25)
    cross_axis.set_axisbelow(True)
    cross_axis.legend(fontsize=8)

    presence_axis = axes[1, 1]
    presence_labels = ["Accuracy", "Balanced accuracy", "F1"]
    presence_values = [
        float(presence["presence_accuracy"]),
        float(presence["presence_balanced_accuracy"]),
        float(presence["presence_f1"]),
    ]
    bars = presence_axis.bar(
        presence_labels,
        presence_values,
        color=["#72B7B2", "#E45756", "#54A24B"],
    )
    presence_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    presence_axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    presence_axis.set_ylim(0, 1)
    presence_axis.set_ylabel("Score")
    presence_axis.set_title("Person-presence metrics (class imbalance matters)")
    presence_axis.grid(axis="y", alpha=0.25)
    presence_axis.set_axisbelow(True)

    num_sessions = int(pose_summary.loc["training_mean", "sessions"])
    num_samples = int(pose_summary.loc["training_mean", "samples"])
    figure.text(
        0.5,
        0.025,
        (
            f"{num_sessions} synchronized sessions, {num_samples} windows, one participant, "
            "two recording rounds. Pose targets are video-derived pseudo-labels. "
            "Training-mean is static and does not use CSI to reproduce motion; "
            "these plots diagnose the current experiment and do not improve the model."
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


def _read_pose(row: pd.Series, suffix: str) -> np.ndarray:
    coords = np.array(
        [
            [
                row[f"{landmark}_x_{suffix}"],
                row[f"{landmark}_y_{suffix}"],
            ]
            for landmark in LANDMARKS
        ],
        dtype=float,
    )
    return coords


def _read_validity(row: pd.Series) -> np.ndarray:
    return np.array(
        [bool(row[f"{landmark}_valid"]) for landmark in LANDMARKS],
        dtype=bool,
    )


def _pose_error_pixels(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> float:
    if not valid.any():
        return float("nan")
    delta = actual[valid] - predicted[valid]
    scaled = delta * np.array([IMAGE_WIDTH, IMAGE_HEIGHT])
    return float(np.linalg.norm(scaled, axis=1).mean())


def _draw_skeleton(
    axis: plt.Axes,
    points: np.ndarray,
    valid: np.ndarray,
    line_color: str,
    joint_color: str,
) -> None:
    for start, end in POSE_CONNECTIONS:
        start_index = LANDMARKS.index(start)
        end_index = LANDMARKS.index(end)
        if valid[start_index] and valid[end_index]:
            axis.plot(
                [points[start_index, 0], points[end_index, 0]],
                [points[start_index, 1], points[end_index, 1]],
                color=line_color,
                linewidth=2.2,
            )
    axis.scatter(
        points[valid, 0],
        points[valid, 1],
        s=18,
        color=joint_color,
        zorder=3,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(1, 0)
    axis.set_aspect(IMAGE_HEIGHT / IMAGE_WIDTH)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_walking_examples(results_dir: Path, output_path: Path) -> None:
    predictions_path = (
        results_dir / "cross-round-Lan2-to-Lan3" / "predictions.csv"
    )
    predictions = _read_required_csv(
        predictions_path,
        {
            "coordinate_model",
            "session_id",
            "activity",
            "timestamp_ms",
            "person_present_true",
            "person_present_predicted",
            *{
                f"{landmark}_{axis}_{suffix}"
                for landmark in LANDMARKS
                for axis in ("x", "y")
                for suffix in ("true", "predicted")
            },
            *{f"{landmark}_valid" for landmark in LANDMARKS},
        },
    )
    candidates = predictions.loc[
        predictions["coordinate_model"].isin(MODELS)
        & predictions["activity"].astype(str).str.lower().eq("walking")
        & predictions["person_present_true"].astype(bool)
    ].copy()
    if candidates.empty:
        raise ValueError("No held-out walking examples are available.")

    session_id = sorted(candidates["session_id"].unique())[0]
    session = candidates.loc[
        (candidates["session_id"] == session_id)
        & (candidates["coordinate_model"] == "training_mean")
    ].sort_values("timestamp_ms")
    valid_counts = session.apply(_read_validity, axis=1).map(np.count_nonzero)
    session = session.loc[valid_counts.to_numpy() >= 8]
    if len(session) < 4:
        raise ValueError("At least four valid held-out walking windows are required.")

    chosen_positions = np.linspace(0, len(session) - 1, num=4, dtype=int)
    selected = session.iloc[chosen_positions]
    ridge_session = candidates.loc[
        (candidates["session_id"] == session_id)
        & (candidates["coordinate_model"] == "ridge")
    ].set_index("window_index", verify_integrity=True)
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False)
    figure.suptitle(
        "Held-out walking session — video pseudo-pose vs CSI Ridge and mean-pose",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for column, (_, row) in enumerate(selected.iterrows()):
        actual = _read_pose(row, "true")
        valid = _read_validity(row)
        timestamp_s = (float(row["timestamp_ms"]) - float(session["timestamp_ms"].iloc[0])) / 1000.0
        _draw_skeleton(axes[0, column], actual, valid, "#2CA02C", "#65B84A")
        axes[0, column].set_title(
            f"Video pseudo-label\nwindow {int(row['window_index'])} | t={timestamp_s:.1f}s",
            fontsize=9,
            color="#287A28",
        )
        ridge_row = ridge_session.loc[int(row["window_index"])]
        for axis_row, prediction_row, model_name, line_color, joint_color in (
            (1, ridge_row, "CSI Ridge", "#F58518", "#17BECF"),
            (2, row, "Training-mean", "#4C78A8", "#72B7B2"),
        ):
            predicted = _read_pose(prediction_row, "predicted")
            error = _pose_error_pixels(actual, predicted, valid)
            _draw_skeleton(
                axes[axis_row, column],
                predicted,
                valid,
                line_color,
                joint_color,
            )
            axes[axis_row, column].set_title(
                f"{model_name} prediction\n{error:.0f}px error",
                fontsize=9,
                color=line_color,
            )

    figure.text(
        0.5,
        0.025,
        (
            f"One held-out Lan3 walking session ({session_id}); four matching windows span the session. "
            "Ground truth is MediaPipe-derived video pseudo-labeling. Ridge uses CSI; "
            "training-mean is static. A few illustrated frames are not a substitute for "
            "full-session metrics or independent-round evaluation."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
        wrap=True,
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.98, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_plots(results_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    metric_plot = output_dir / "verified_pose_metric_dashboard.png"
    pose_plot = output_dir / "heldout_walking_pose_examples.png"
    _plot_metric_dashboard(results_dir, metric_plot)
    _plot_walking_examples(results_dir, pose_plot)
    return metric_plot, pose_plot


def main() -> None:
    args = parse_args()
    for output in create_plots(args.results_dir, args.output_dir):
        print(f"Saved pose visualization: {output}")


if __name__ == "__main__":
    main()
