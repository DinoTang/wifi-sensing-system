
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CSI PREPROCESSING PIPELINE
# ============================================================
# Chức năng:
# 1. Đọc các file CSV CSI từ data/raw.
# 2. Gán nhãn kịch bản dựa trên tên file.
# 3. Kiểm tra chất lượng frame:
#       - Số sample thực tế.
#       - Sample index bị thiếu.
#       - Sample index bị trùng.
#       - Frame hoàn chỉnh hay không.
# 4. Kiểm tra amplitude và các sample index có amplitude = 0.
# 5. Xuất dataset tổng và dataset chỉ chứa frame hoàn chỉnh.
# 6. Tạo đặc trưng sơ bộ theo từng frame.
# 7. Xuất báo cáo CSV và biểu đồ.
#
# Lưu ý:
# - Frame không hoàn chỉnh vẫn được giữ trong dataset tổng.
# - Frame không hoàn chỉnh không được đưa vào dataset training.
# - Các sample index có amplitude = 0 vẫn được giữ lại.
# - Motion score chỉ là đặc trưng khám phá ban đầu,
#   chưa phải mô hình nhận diện chuyển động.
# ============================================================


REQUIRED_COLUMNS = [
    "frame_id",
    "timestamp_ms",
    "rssi",
    "len",
    "first_word",
    "sample_index",
    "i",
    "q",
    "amplitude",
]


SCENARIO_LABELS = {
    "phong_trong": "room_empty",
    "nguoi_ngoi_im": "person_static",
    "nguoi_di_vao": "person_entering",
    "nguoi_di_ra": "person_leaving",
    # Intel capture sessions (data/intel_raw): English scenario file stems.
    "room_empty": "room_empty",
    "person_static": "person_static",
    "person_moving": "person_moving",
    "multiple_people": "multiple_people",
}


# ============================================================
# 1. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CSI preprocessing and frame quality pipeline"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw CSI CSV files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Project root/output directory",
    )

    parser.add_argument(
        "--expected-samples",
        type=int,
        default=192,
        help=(
            "Expected number of unique samples in each frame "
            "(192 for the ESP32 dataset in data/raw, "
            "224 for the Intel VHT dataset in data/intel_raw)"
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help=(
            "Length in seconds of the trailing window used for the "
            "cross-frame (temporal) variability feature. The number of "
            "frames is derived from the measured sampling interval."
        ),
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "zscore", "minmax"],
        default="none",
        help=(
            "Optional scaling of the exported feature tables. "
            "'zscore' subtracts the mean and divides by the standard "
            "deviation, 'minmax' rescales to [0, 1]. Statistics are "
            "fitted ONLY on the sessions given to --train-sessions. "
            "Default 'none' keeps the raw feature values."
        ),
    )

    parser.add_argument(
        "--train-sessions",
        type=str,
        default="",
        help=(
            "Comma-separated session ids used to fit the normalization "
            "statistics, e.g. 'room_empty_01,person_static_01'. "
            "Must be set once the train/test split is agreed with the "
            "ML owner; when empty, statistics are fitted on all rows "
            "and the pipeline prints a leakage warning."
        ),
    )

    parser.add_argument(
        "--filter-frames",
        type=int,
        default=5,
        help=(
            "Width in frames of the centered moving-average filter used "
            "for the before/after noise-filtering experiment. "
            "Values below 2 disable filtering."
        ),
    )

    args = parser.parse_args()

    if args.expected_samples <= 0:
        raise ValueError("--expected-samples must be greater than 0.")

    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0.")

    args.train_sessions = [
        item.strip()
        for item in args.train_sessions.split(",")
        if item.strip()
    ]

    return args


# ============================================================
# 2. LABEL INFERENCE
# ============================================================

def infer_label(filename: str) -> str:
    """
    Infer scenario label from the input filename.

    Supports filenames such as:
        phong_trong.csv
        nguoi_ngoi_im.csv
        nguoi_di_vao.csv
        nguoi_di_ra.csv
    """

    stem = Path(filename).stem.lower().strip()

    # Normalize separators.
    normalized = stem.replace("-", "_").replace(" ", "_")

    # Prefer exact matches.
    if normalized in SCENARIO_LABELS:
        return SCENARIO_LABELS[normalized]

    # Also support names containing the scenario key.
    for key, label in SCENARIO_LABELS.items():
        if key in normalized:
            return label

    raise ValueError(
        f"Cannot infer scenario label from filename: {filename}. "
        f"Supported scenario names: {list(SCENARIO_LABELS.keys())}"
    )


# ============================================================
# 3. LOAD AND VALIDATE RAW DATA
# ============================================================

def load_csv_files(input_dir: str) -> list[tuple[str, str, pd.DataFrame]]:
    """
    Load all CSV files in input_dir.

    Returns:
        List of (session_id, scenario_label, dataframe).
    """

    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_path.resolve()}"
        )

    csv_files = sorted(
        path
        for path in input_path.glob("*.csv")
        # Files whose name starts with "_" are sidecar artefacts
        # (for example _sessions_metadata.csv), not capture sessions.
        if not path.name.startswith("_")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {input_path.resolve()}"
        )

    sessions = []

    for csv_path in csv_files:
        session_id = csv_path.stem
        scenario_label = infer_label(csv_path.name)

        df = pd.read_csv(csv_path)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{csv_path.name} is missing required columns: "
                f"{missing_columns}"
            )

        # Convert numerical columns.
        numeric_columns = [
            "frame_id",
            "timestamp_ms",
            "rssi",
            "len",
            "first_word",
            "sample_index",
            "i",
            "q",
            "amplitude",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # Rows without frame_id or sample_index cannot be assigned
        # reliably to a frame, so stop instead of silently dropping them.
        invalid_keys = df["frame_id"].isna() | df["sample_index"].isna()

        if invalid_keys.any():
            raise ValueError(
                f"{csv_path.name} contains {int(invalid_keys.sum())} "
                "rows with missing or invalid frame_id/sample_index."
            )

        # Frame IDs and sample indices are expected to be integers.
        non_integer_frame_ids = (
            df["frame_id"] % 1 != 0
        )

        non_integer_sample_indices = (
            df["sample_index"] % 1 != 0
        )

        if non_integer_frame_ids.any():
            raise ValueError(
                f"{csv_path.name} contains non-integer frame_id values."
            )

        if non_integer_sample_indices.any():
            raise ValueError(
                f"{csv_path.name} contains non-integer sample_index values."
            )

        df["frame_id"] = df["frame_id"].astype("int64")
        df["sample_index"] = df["sample_index"].astype("int64")

        df["session_id"] = session_id
        df["scenario_label"] = scenario_label

        sessions.append((session_id, scenario_label, df))

        print(
            f"[LOAD] {csv_path.name}: "
            f"{len(df):,} rows, label={scenario_label}"
        )

    return sessions


# ============================================================
# 4. FRAME QUALITY CHECK
# ============================================================

def inspect_session_frames(
    session_id: str,
    scenario_label: str,
    df: pd.DataFrame,
    expected_samples: int,
) -> pd.DataFrame:
    """
    Inspect each frame and return one quality-report row per frame.

    A frame is complete only if:
      - It has exactly expected_samples unique sample indices.
      - It has exactly expected_samples rows.
      - It has no duplicate (frame_id, sample_index) pairs.
      - It has no sample indices outside the expected range.
      - It has no missing sample indices in the expected range.
    """

    group = df.groupby("frame_id", sort=True)

    unique_counts = group["sample_index"].nunique()
    row_counts = group.size()
    timestamps = group["timestamp_ms"].first()

    # Detect duplicated frame/sample pairs.
    duplicate_pair_mask = df.duplicated(
        subset=["frame_id", "sample_index"],
        keep=False,
    )

    duplicate_frame_ids = set(
        df.loc[duplicate_pair_mask, "frame_id"].unique()
    )

    expected_indices = set(range(expected_samples))

    qc_rows = []

    for frame_id, unique_count in unique_counts.items():
        frame_rows = df.loc[
            df["frame_id"] == frame_id,
            "sample_index",
        ]

        present_indices = set(
            frame_rows.dropna().astype(int).tolist()
        )

        missing_indices = sorted(
            expected_indices - present_indices
        )

        unexpected_indices = sorted(
            present_indices - expected_indices
        )

        row_count = int(row_counts.loc[frame_id])
        has_duplicates = frame_id in duplicate_frame_ids

        is_complete = (
            int(unique_count) == expected_samples
            and row_count == expected_samples
            and not has_duplicates
            and not missing_indices
            and not unexpected_indices
        )

        timestamp = timestamps.loc[frame_id]

        qc_rows.append(
            {
                "session_id": session_id,
                "scenario_label": scenario_label,
                "frame_id": int(frame_id),
                "timestamp_ms": (
                    float(timestamp)
                    if pd.notna(timestamp)
                    else np.nan
                ),
                "sample_count": int(unique_count),
                "row_count": row_count,
                "expected_sample_count": expected_samples,
                "has_duplicate_sample_indices": bool(has_duplicates),
                "missing_sample_indices": ";".join(
                    map(str, missing_indices)
                ),
                "unexpected_sample_indices": ";".join(
                    map(str, unexpected_indices)
                ),
                "frame_complete": bool(is_complete),
            }
        )

    return pd.DataFrame(qc_rows)


def build_quality_report(
    sessions: list[tuple[str, str, pd.DataFrame]],
    expected_samples: int,
) -> pd.DataFrame:
    """
    Build the complete frame-quality report across all sessions.
    """

    reports = []

    for session_id, scenario_label, df in sessions:
        session_report = inspect_session_frames(
            session_id=session_id,
            scenario_label=scenario_label,
            df=df,
            expected_samples=expected_samples,
        )

        reports.append(session_report)

    if not reports:
        return pd.DataFrame()

    return pd.concat(reports, ignore_index=True)


# ============================================================
# 5. DATASET PREPARATION
# ============================================================

def attach_frame_quality(
    sessions: list[tuple[str, str, pd.DataFrame]],
    quality_report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach frame_complete to every raw sample row.

    The join uses both session_id and frame_id because frame IDs
    can overlap between different CSV files.
    """

    all_data = pd.concat(
        [df for _, _, df in sessions],
        ignore_index=True,
    )

    quality_keys = quality_report[
        [
            "session_id",
            "frame_id",
            "frame_complete",
        ]
    ].copy()

    if quality_keys.duplicated(["session_id", "frame_id"]).any():
        raise ValueError(
            "Quality report contains duplicated session_id/frame_id keys."
        )

    data = all_data.merge(
        quality_keys,
        on=["session_id", "frame_id"],
        how="left",
        validate="many_to_one",
    )

    if data["frame_complete"].isna().any():
        raise ValueError(
            "Some raw rows could not be matched to the frame-quality report."
        )

    data["frame_complete"] = data["frame_complete"].astype(bool)

    return data


# ============================================================
# 6. AMPLITUDE VALIDATION
# ============================================================

def validate_amplitude(data: pd.DataFrame) -> dict:
    """
    Compare provided amplitude with sqrt(i^2 + q^2).

    This is a consistency check, not a data-cleaning operation.
    """

    valid_mask = (
        data["i"].notna()
        & data["q"].notna()
        & data["amplitude"].notna()
    )

    if not valid_mask.any():
        return {
            "valid_rows": 0,
            "mean_absolute_error": np.nan,
            "max_absolute_error": np.nan,
        }

    expected_amplitude = np.sqrt(
        data.loc[valid_mask, "i"] ** 2
        + data.loc[valid_mask, "q"] ** 2
    )

    actual_amplitude = data.loc[valid_mask, "amplitude"]

    absolute_error = (
        actual_amplitude - expected_amplitude
    ).abs()

    return {
        "valid_rows": int(valid_mask.sum()),
        "mean_absolute_error": float(absolute_error.mean()),
        "max_absolute_error": float(absolute_error.max()),
    }


# ============================================================
# 7. ZERO-ONLY SAMPLE INDICES
# ============================================================

def find_zero_only_sample_indices(
    data: pd.DataFrame,
    expected_samples: int,
) -> list[int]:
    """
    Identify sample indices whose amplitude is zero or missing
    for all observed rows across the dataset.

    These indices are reported, not removed.
    """

    zero_only_indices = []

    for sample_index in range(expected_samples):
        sample_data = data.loc[
            data["sample_index"] == sample_index,
            "amplitude",
        ]

        if sample_data.empty:
            zero_only_indices.append(sample_index)
            continue

        values = sample_data.dropna()

        if values.empty or np.allclose(values.to_numpy(), 0):
            zero_only_indices.append(sample_index)

    return zero_only_indices


# ============================================================
# 8. PRELIMINARY FRAME FEATURES
# ============================================================

def calculate_frame_features(
    data: pd.DataFrame,
    expected_samples: int,
) -> pd.DataFrame:
    """
    Calculate preliminary features for each frame.

    Features:
      - mean_amplitude
      - std_amplitude
      - max_amplitude
      - mean_rssi
      - motion_score

    motion_score is computed only for complete frames. The score is
    the mean absolute amplitude difference from the previous complete
    frame in the same session. When an incomplete frame occurs, the
    comparison history is reset to avoid comparing across the gap.
    """

    feature_rows = []

    for session_id, session_df in data.groupby(
        "session_id",
        sort=False,
    ):
        session_df = session_df.sort_values(
            ["frame_id", "sample_index"]
        )

        frame_groups = session_df.groupby(
            "frame_id",
            sort=True,
        )

        previous_complete_amplitude = None

        for frame_id, frame_df in frame_groups:
            is_complete = bool(
                frame_df["frame_complete"].iloc[0]
            )

            amplitude = frame_df["amplitude"].dropna()
            rssi = frame_df["rssi"].dropna()

            mean_amplitude = (
                float(amplitude.mean())
                if not amplitude.empty
                else np.nan
            )

            std_amplitude = (
                float(amplitude.std(ddof=0))
                if not amplitude.empty
                else np.nan
            )

            max_amplitude = (
                float(amplitude.max())
                if not amplitude.empty
                else np.nan
            )

            mean_rssi = (
                float(rssi.mean())
                if not rssi.empty
                else np.nan
            )

            motion_score = np.nan

            if not is_complete:
                # Do not calculate motion features from incomplete frames.
                previous_complete_amplitude = None

            else:
                # Use a consistent sample-index order for each frame.
                ordered = frame_df.sort_values("sample_index")

                amplitude_series = (
                    ordered.set_index("sample_index")["amplitude"]
                    .reindex(range(expected_samples))
                )

                current_amplitude = amplitude_series.to_numpy(
                    dtype=float
                )

                # A frame is expected to have all samples. Still check
                # finite values before calculating the motion score.
                if np.isfinite(current_amplitude).all():
                    if previous_complete_amplitude is not None:
                        motion_score = float(
                            np.mean(
                                np.abs(
                                    current_amplitude
                                    - previous_complete_amplitude
                                )
                            )
                        )

                    previous_complete_amplitude = current_amplitude
                else:
                    previous_complete_amplitude = None

            timestamp_values = frame_df["timestamp_ms"].dropna()

            timestamp_ms = (
                float(timestamp_values.iloc[0])
                if not timestamp_values.empty
                else np.nan
            )

            scenario_label = frame_df["scenario_label"].iloc[0]

            feature_rows.append(
                {
                    "session_id": session_id,
                    "scenario_label": scenario_label,
                    "frame_id": int(frame_id),
                    "timestamp_ms": timestamp_ms,
                    "frame_complete": is_complete,
                    "mean_amplitude": mean_amplitude,
                    "std_amplitude": std_amplitude,
                    "max_amplitude": max_amplitude,
                    "mean_rssi": mean_rssi,
                    "motion_score": motion_score,
                }
            )

    return pd.DataFrame(feature_rows)


# ============================================================
# 9. SESSION QUALITY SUMMARY
# ============================================================

def build_session_summary(
    sessions: list[tuple[str, str, pd.DataFrame]],
    quality_report: pd.DataFrame,
    data: pd.DataFrame,
    zero_only_indices: list[int],
) -> pd.DataFrame:
    """
    Create one quality-summary row per input session.
    """

    summary_rows = []

    for session_id, scenario_label, raw_df in sessions:
        session_qc = quality_report.loc[
            quality_report["session_id"] == session_id
        ]

        session_data = data.loc[
            data["session_id"] == session_id
        ]

        complete_count = int(
            session_qc["frame_complete"].sum()
        )

        total_frames = int(len(session_qc))

        incomplete_count = total_frames - complete_count

        timestamp_values = (
            session_qc[["frame_id", "timestamp_ms"]]
            .dropna()
            .sort_values("frame_id")["timestamp_ms"]
        )

        timestamp_deltas = timestamp_values.diff()

        timestamp_zero_deltas = int(
            (timestamp_deltas == 0).sum()
        )

        timestamp_negative_deltas = int(
            (timestamp_deltas < 0).sum()
        )

        duplicate_pair_count = int(
            raw_df.duplicated(
                subset=["frame_id", "sample_index"],
                keep=False,
            ).sum()
        )

        summary_rows.append(
            {
                "session_id": session_id,
                "scenario_label": scenario_label,
                "total_rows": int(len(raw_df)),
                "unique_frames": total_frames,
                "complete_frames": complete_count,
                "incomplete_frames": incomplete_count,
                "incomplete_frame_rate": (
                    incomplete_count / total_frames
                    if total_frames > 0
                    else np.nan
                ),
                "duplicate_frame_sample_rows": duplicate_pair_count,
                "timestamp_zero_deltas": timestamp_zero_deltas,
                "timestamp_negative_deltas": timestamp_negative_deltas,
                "zero_only_sample_indices": ";".join(
                    map(str, zero_only_indices)
                ),
                "total_zero_only_sample_indices": len(
                    zero_only_indices
                ),
                "complete_rows": int(
                    session_data["frame_complete"].sum()
                ),
            }
        )

    return pd.DataFrame(summary_rows)


# ============================================================
# 9B. TEMPORAL (CROSS-FRAME) FEATURES
# ============================================================
#
# std_amplitude in section 8 varies BETWEEN samples inside one frame
# (it mostly reflects the frequency selectivity of the channel).
# To study human motion we also need the variability of the SAME
# CSI point across many frames inside a short time window.


def build_amplitude_matrix(
    complete_df: pd.DataFrame,
    expected_samples: int,
) -> pd.DataFrame:
    """
    Build a frames x sample_index amplitude matrix for one session.

    Only rows already flagged as frame_complete should be passed in.
    """

    pivot = complete_df.pivot_table(
        index="frame_id",
        columns="sample_index",
        values="amplitude",
        aggfunc="mean",
    )

    pivot = pivot.reindex(
        columns=range(expected_samples)
    )

    return pivot.sort_index()


def estimate_sampling_interval_ms(
    timestamps: pd.Series,
) -> tuple[float, float]:
    """
    Estimate the frame sampling interval from timestamps.

    Returns (median_delta_ms, fps). Non-positive deltas (clock glitches)
    are excluded before taking the median.
    """

    deltas = timestamps.diff().dropna()
    deltas = deltas.loc[deltas > 0]

    if deltas.empty:
        return np.nan, np.nan

    median_delta = float(deltas.median())

    fps = (
        1000.0 / median_delta
        if median_delta > 0
        else np.nan
    )

    return median_delta, fps


def rolling_temporal_std(
    matrix: np.ndarray,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame variability of each CSI point across a trailing window.

    For every frame position, the same sample_index is compared across
    all frames of the trailing window. Returns:
      - mean over sample_index of that cross-frame std
      - max  over sample_index of that cross-frame std
    """

    n_frames = matrix.shape[0]

    width = max(2, min(int(window_frames), n_frames))

    mean_std = np.full(n_frames, np.nan)
    max_std = np.full(n_frames, np.nan)

    for position in range(n_frames):
        start = max(0, position - width + 1)

        window = matrix[start : position + 1, :]

        with np.errstate(invalid="ignore"):
            per_point_std = np.nanstd(window, axis=0)

        per_point_std = per_point_std[
            np.isfinite(per_point_std)
        ]

        if per_point_std.size:
            mean_std[position] = float(per_point_std.mean())
            max_std[position] = float(per_point_std.max())

    return mean_std, max_std


def calculate_temporal_features(
    data: pd.DataFrame,
    expected_samples: int,
    window_seconds: float,
) -> pd.DataFrame:
    """
    Variability of the same CSI points across frames in a time window.

    Window length in frames is derived from the measured sampling
    interval (median timestamp delta of the session), so the requested
    `window_seconds` corresponds to real elapsed time.
    """

    rows = []

    for session_id, session_df in data.groupby(
        "session_id",
        sort=False,
    ):
        complete = session_df.loc[session_df["frame_complete"]]

        if complete.empty:
            continue

        matrix_df = build_amplitude_matrix(
            complete,
            expected_samples,
        )

        if len(matrix_df) < 2:
            continue

        timestamps = (
            complete.groupby("frame_id")["timestamp_ms"]
            .first()
            .reindex(matrix_df.index)
        )

        median_delta, fps = estimate_sampling_interval_ms(
            timestamps
        )

        if np.isfinite(fps) and fps > 0:
            window_frames = int(round(window_seconds * fps))
        else:
            window_frames = 2

        window_frames = max(
            2,
            min(window_frames, len(matrix_df)),
        )

        matrix = matrix_df.to_numpy(dtype=float)

        mean_std, max_std = rolling_temporal_std(
            matrix,
            window_frames,
        )

        scenario_label = session_df[
            "scenario_label"
        ].iloc[0]

        for position, frame_id in enumerate(matrix_df.index):
            rows.append(
                {
                    "session_id": session_id,
                    "scenario_label": scenario_label,
                    "frame_id": int(frame_id),
                    "timestamp_ms": float(
                        timestamps.iloc[position]
                    ),
                    "sampling_interval_ms": median_delta,
                    "sampling_fps": fps,
                    "window_frames": window_frames,
                    "window_seconds_requested": window_seconds,
                    "temporal_std_amplitude": mean_std[position],
                    "temporal_std_amplitude_max": (
                        max_std[position]
                    ),
                }
            )

    if not rows:
        print(
            "[TEMPORAL] No session had enough complete frames."
        )
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# 10b. OPTIONAL NORMALIZATION (FIT ON TRAIN SESSIONS ONLY)
# ============================================================
# Scaling is applied to the EXPORTED FEATURE TABLES only.
# The raw sample dataset (i, q, amplitude) is never rescaled, so the
# QC flags and the physical values stay interpretable.
#
# Statistics are always fitted on --train-sessions and then applied to
# every row. This keeps the scaling parameters free of test data and
# makes the same parameters reusable for live prediction later.

IDENTITY_COLUMNS = {
    "session_id",
    "scenario_label",
    "frame_id",
    "timestamp_ms",
    "frame_complete",
    "sampling_interval_ms",
    "sampling_fps",
    "window_frames",
    "window_seconds_requested",
}

NORMALIZATION_PARAM_COLUMNS = [
    "dataset",
    "column",
    "method",
    "center",
    "scale",
    "n_train_rows",
    "n_total_rows",
    "note",
]


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns that are measurements, not ids or labels."""

    return [
        column
        for column in frame.columns
        if column not in IDENTITY_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def normalize_features(
    features: pd.DataFrame,
    method: str,
    train_sessions: list[str],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Scale feature columns with statistics fitted on the train rows.

    Returns the normalized table, the fitted parameters and a list of
    human-readable notes (columns left unchanged, unknown sessions).
    Columns that are constant on the train rows keep their raw values,
    because their scale is undefined.
    """

    if features.empty:
        return (
            features.copy(),
            pd.DataFrame(columns=NORMALIZATION_PARAM_COLUMNS),
            [],
        )

    known_sessions = set(features["session_id"].unique())

    if train_sessions:
        unknown = sorted(set(train_sessions) - known_sessions)
        if unknown:
            raise ValueError(
                f"--train-sessions contains ids that are absent from "
                f"{dataset_name}: {unknown}"
            )
        train_mask = features["session_id"].isin(train_sessions)
        if not train_mask.any():
            raise ValueError(
                f"--train-sessions matched no row of {dataset_name}."
            )
        fitted_on = f"{int(train_mask.sum()):,}/{len(features):,} train rows"
    else:
        train_mask = pd.Series(True, index=features.index)
        fitted_on = f"ALL {len(features):,} rows (no split)"

    normalized = features.copy()
    param_rows: list[dict] = []
    notes: list[str] = []

    for column in select_feature_columns(features):
        values = pd.to_numeric(features[column], errors="coerce")

        train_values = (
            values.loc[train_mask]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if train_values.empty:
            notes.append(
                f"{dataset_name}.{column}: no finite train value, "
                "left unchanged."
            )
            continue

        if method == "zscore":
            center = float(train_values.mean())
            scale = float(train_values.std(ddof=0))
        else:  # minmax
            center = float(train_values.min())
            scale = float(train_values.max()) - center

        if not np.isfinite(scale) or scale == 0:
            notes.append(
                f"{dataset_name}.{column}: constant on the train rows, "
                "left unchanged."
            )
            param_rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "method": method,
                    "center": center,
                    "scale": np.nan,
                    "n_train_rows": int(train_values.size),
                    "n_total_rows": int(len(features)),
                    "note": "constant on train - not scaled",
                }
            )
            continue

        normalized[column] = (values - center) / scale

        param_rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "method": method,
                "center": center,
                "scale": scale,
                "n_train_rows": int(train_values.size),
                "n_total_rows": int(len(features)),
                "note": fitted_on,
            }
        )

    return normalized, pd.DataFrame(param_rows), notes


def moving_average_along_time(
    matrix: np.ndarray,
    filter_frames: int,
) -> np.ndarray:
    """
    Centered moving average along the frame axis.

    Offline smoothing only: the centered window uses future frames, so
    the filtered series must NOT be used for a real-time detector
    without changing it to a causal (trailing) window.
    """

    if filter_frames < 2:
        return matrix.copy()

    frame = pd.DataFrame(matrix)

    filtered = (
        frame.rolling(
            window=filter_frames,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    return filtered.to_numpy(dtype=float)


def evaluate_filter(
    data: pd.DataFrame,
    expected_samples: int,
    filter_frames: int,
    window_seconds: float,
) -> pd.DataFrame:
    """
    Before/after evaluation of the moving-average filter.

    Checks both sides of the trade-off:
      - noise reduction: high-frequency jitter, measured as the mean
        over sample points of std(diff(amplitude)) along time
      - motion retention: trend correlation between raw and filtered
        session-mean series, and the temporal variability that remains
    """

    rows = []

    for session_id, session_df in data.groupby(
        "session_id",
        sort=False,
    ):
        complete = session_df.loc[session_df["frame_complete"]]

        if complete.empty:
            continue

        matrix_df = build_amplitude_matrix(
            complete,
            expected_samples,
        )

        if len(matrix_df) < 3:
            continue

        timestamps = (
            complete.groupby("frame_id")["timestamp_ms"]
            .first()
            .reindex(matrix_df.index)
        )

        median_delta, fps = estimate_sampling_interval_ms(
            timestamps
        )

        if np.isfinite(fps) and fps > 0:
            window_frames = int(round(window_seconds * fps))
        else:
            window_frames = 2

        window_frames = max(
            2,
            min(window_frames, len(matrix_df)),
        )

        raw = matrix_df.to_numpy(dtype=float)

        filtered = moving_average_along_time(
            raw,
            filter_frames,
        )

        def high_frequency_jitter(matrix: np.ndarray) -> float:
            with np.errstate(invalid="ignore"):
                diffs = np.diff(matrix, axis=0)
                value = np.nanmean(np.nanstd(diffs, axis=0))
            return float(value)

        jitter_raw = high_frequency_jitter(raw)
        jitter_filtered = high_frequency_jitter(filtered)

        noise_reduction_pct = (
            100.0 * (1.0 - jitter_filtered / jitter_raw)
            if jitter_raw > 0
            else np.nan
        )

        raw_trend = np.nanmean(raw, axis=1)
        filtered_trend = np.nanmean(filtered, axis=1)

        if np.std(raw_trend) > 0 and np.std(filtered_trend) > 0:
            trend_correlation = float(
                np.corrcoef(raw_trend, filtered_trend)[0, 1]
            )
        else:
            trend_correlation = np.nan

        # Correlation of the SLOW content only (about 2 s of
        # smoothing applied to both series). This is the part a
        # motion detector relies on, so it must stay close to 1.
        slow_window = max(
            3,
            int(round(2.0 * fps)) if np.isfinite(fps) else 20,
        )
        slow_window = min(slow_window, len(raw_trend))

        raw_slow = (
            pd.Series(raw_trend)
            .rolling(
                slow_window,
                center=True,
                min_periods=1,
            )
            .mean()
        )
        filtered_slow = (
            pd.Series(filtered_trend)
            .rolling(
                slow_window,
                center=True,
                min_periods=1,
            )
            .mean()
        )

        if (
            float(np.std(raw_slow)) > 0
            and float(np.std(filtered_slow)) > 0
        ):
            low_frequency_correlation = float(
                np.corrcoef(raw_slow, filtered_slow)[0, 1]
            )
        else:
            low_frequency_correlation = np.nan

        retained_variance_pct = (
            100.0
            * float(np.var(filtered_trend))
            / float(np.var(raw_trend))
            if np.var(raw_trend) > 0
            else np.nan
        )

        temporal_raw, _ = rolling_temporal_std(
            raw,
            window_frames,
        )
        temporal_filtered, _ = rolling_temporal_std(
            filtered,
            window_frames,
        )

        temporal_raw_mean = float(
            np.nanmean(temporal_raw)
        )
        temporal_filtered_mean = float(
            np.nanmean(temporal_filtered)
        )

        temporal_retained_pct = (
            100.0 * temporal_filtered_mean / temporal_raw_mean
            if temporal_raw_mean > 0
            else np.nan
        )

        rows.append(
            {
                "session_id": session_id,
                "scenario_label": session_df[
                    "scenario_label"
                ].iloc[0],
                "frames": int(len(matrix_df)),
                "sampling_interval_ms": median_delta,
                "sampling_fps": fps,
                "filter_frames": filter_frames,
                "hf_jitter_raw": jitter_raw,
                "hf_jitter_filtered": jitter_filtered,
                "noise_reduction_pct": noise_reduction_pct,
                "trend_correlation": trend_correlation,
                "low_frequency_correlation": (
                    low_frequency_correlation
                ),
                "retained_variance_pct": retained_variance_pct,
                "temporal_std_raw": temporal_raw_mean,
                "temporal_std_filtered": temporal_filtered_mean,
                "temporal_std_retained_pct": (
                    temporal_retained_pct
                ),
            }
        )

    if not rows:
        print(
            "[FILTER] No session had enough complete frames."
        )
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# 10. PLOTS
# ============================================================

def save_session_heatmaps(
    sessions: list[tuple[str, str, pd.DataFrame]],
    results_dir: str,
    zero_only_indices: list[int],
):
    """
    Save one frame/sample amplitude heatmap per session.

    Zero-only sample indices are hidden from the visualization,
    but remain in the exported data.
    """

    os.makedirs(results_dir, exist_ok=True)

    for session_id, scenario_label, df in sessions:
        plot_df = df.copy()

        plot_df = plot_df.loc[
            ~plot_df["sample_index"].isin(zero_only_indices)
        ]

        if plot_df.empty:
            print(
                f"[PLOT] Skipping heatmap for {session_id}: no data."
            )
            continue

        pivot = plot_df.pivot_table(
            index="sample_index",
            columns="frame_id",
            values="amplitude",
            aggfunc="mean",
        )

        pivot = pivot.sort_index().sort_index(axis=1)

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))

        image = ax.imshow(
            pivot.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )

        ax.set_title(
            f"CSI Amplitude Heatmap - {session_id} ({scenario_label})"
        )
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Sample Index")

        # Use a limited number of readable ticks.
        column_count = len(pivot.columns)

        if column_count > 1:
            tick_positions = np.linspace(
                0,
                column_count - 1,
                min(10, column_count),
                dtype=int,
            )

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [str(pivot.columns[position]) for position in tick_positions],
                rotation=45,
                ha="right",
            )

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Amplitude")

        fig.tight_layout()

        output_path = os.path.join(
            results_dir,
            f"heatmap_{session_id}.png",
        )

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"[PLOT] Saved: {output_path}")


def save_mean_amplitude_plot(
    data: pd.DataFrame,
    results_dir: str,
):
    """
    Plot mean amplitude for each session.
    """

    session_means = (
        data.groupby(
            ["session_id", "scenario_label"],
            as_index=False,
        )["amplitude"]
        .mean()
        .sort_values("session_id")
    )

    if session_means.empty:
        return

    labels = session_means["session_id"].tolist()
    values = session_means["amplitude"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(labels, values)

    ax.set_title("Mean CSI Amplitude by Session")
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean Amplitude")

    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    output_path = os.path.join(
        results_dir,
        "mean_amplitude_by_session.png",
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT] Saved: {output_path}")


def save_motion_score_plot(
    features: pd.DataFrame,
    results_dir: str,
):
    """
    Plot preliminary motion score over frame IDs.
    """

    if features.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for session_id, session_features in features.groupby(
        "session_id",
        sort=False,
    ):
        session_features = session_features.sort_values("frame_id")

        ax.plot(
            session_features["frame_id"],
            session_features["motion_score"],
            label=session_id,
            linewidth=1,
        )

    ax.set_title("Preliminary CSI Motion Score by Frame")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Mean Absolute Amplitude Difference")

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = os.path.join(
        results_dir,
        "preliminary_motion_score.png",
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT] Saved: {output_path}")


def save_temporal_variability_plot(
    temporal_features: pd.DataFrame,
    results_dir: str,
):
    """
    Box plot of cross-frame temporal variability by scenario label.
    """

    if temporal_features.empty:
        return

    plot_df = temporal_features.dropna(
        subset=["temporal_std_amplitude"]
    )

    if plot_df.empty:
        return

    labels = list(
        dict.fromkeys(plot_df["scenario_label"])
    )

    series = [
        plot_df.loc[
            plot_df["scenario_label"] == label,
            "temporal_std_amplitude",
        ].to_numpy()
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(series, showfliers=False)

    ax.set_xticks(
        range(1, len(labels) + 1)
    )
    ax.set_xticklabels(labels)

    ax.set_title(
        "Temporal variability of CSI points across frames "
        "(trailing window)"
    )
    ax.set_xlabel("Scenario label")
    ax.set_ylabel("Mean cross-frame std of amplitude")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    output_path = os.path.join(
        results_dir,
        "temporal_variability_by_class.png",
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT] Saved: {output_path}")


def save_filter_comparison_plot(
    data: pd.DataFrame,
    expected_samples: int,
    filter_frames: int,
    results_dir: str,
):
    """
    Before/after panel: session-mean amplitude per scenario label.

    One representative session per class so the panels stay readable.
    """

    if filter_frames < 2:
        return

    selected = []

    for session_id, session_df in data.groupby(
        "session_id",
        sort=False,
    ):
        label = session_df["scenario_label"].iloc[0]

        if any(item[0] == label for item in selected):
            continue

        complete = session_df.loc[session_df["frame_complete"]]

        if len(complete) < 3:
            continue

        selected.append((label, session_id, complete))

    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(12, 3.2 * len(selected)),
        sharex=False,
    )

    if len(selected) == 1:
        axes = [axes]

    for ax, (label, session_id, complete) in zip(
        axes,
        selected,
    ):
        matrix_df = build_amplitude_matrix(
            complete,
            expected_samples,
        )

        raw = matrix_df.to_numpy(dtype=float)

        filtered = moving_average_along_time(
            raw,
            filter_frames,
        )

        timestamps = (
            complete.groupby("frame_id")["timestamp_ms"]
            .first()
            .reindex(matrix_df.index)
        )

        median_delta, _ = estimate_sampling_interval_ms(
            timestamps
        )

        if np.isfinite(median_delta) and median_delta > 0:
            elapsed_seconds = (
                np.arange(len(raw)) * median_delta / 1000.0
            )
        else:
            elapsed_seconds = np.arange(len(raw), dtype=float)

        raw_trend = np.nanmean(raw, axis=1)
        filtered_trend = np.nanmean(filtered, axis=1)

        with np.errstate(invalid="ignore"):
            jitter_raw = float(
                np.nanmean(np.nanstd(np.diff(raw, axis=0), axis=0))
            )
            jitter_filtered = float(
                np.nanmean(
                    np.nanstd(np.diff(filtered, axis=0), axis=0)
                )
            )

        reduction = (
            100.0 * (1.0 - jitter_filtered / jitter_raw)
            if jitter_raw > 0
            else np.nan
        )

        ax.plot(
            elapsed_seconds,
            raw_trend,
            linewidth=0.8,
            alpha=0.7,
            label="Raw",
        )
        ax.plot(
            elapsed_seconds,
            filtered_trend,
            linewidth=1.4,
            label=f"Moving average ({filter_frames} frames)",
        )

        ax.set_title(
            f"{session_id} ({label}) - "
            f"HF jitter reduced {reduction:.1f}%"
        )
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Mean amplitude")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = os.path.join(
        results_dir,
        "filter_before_after.png",
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT] Saved: {output_path}")


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():
    args = parse_args()

    if args.expected_samples <= 0:
        raise ValueError("--expected-samples must be greater than 0.")

    project_root = Path(args.output_dir)

    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CSI PREPROCESSING PIPELINE")
    print("=" * 65)

    print(f"Input directory : {Path(args.input_dir).resolve()}")
    print(f"Output directory: {project_root.resolve()}")
    print(f"Expected samples: {args.expected_samples}")
    print()

    # --------------------------------------------------------
    # Step 1: Load raw CSV files.
    # --------------------------------------------------------

    sessions = load_csv_files(args.input_dir)

    print()
    print(f"[INFO] Loaded {len(sessions)} sessions.")

    # --------------------------------------------------------
    # Step 2: Inspect frame quality.
    # --------------------------------------------------------

    quality_report = build_quality_report(
        sessions=sessions,
        expected_samples=args.expected_samples,
    )

    quality_report_path = (
        processed_dir / "frame_quality_report.csv"
    )

    quality_report.to_csv(
        quality_report_path,
        index=False,
    )

    print(f"[SAVE] {quality_report_path}")

    # --------------------------------------------------------
    # Step 3: Attach quality flags to every raw sample row.
    # --------------------------------------------------------

    data = attach_frame_quality(
        sessions=sessions,
        quality_report=quality_report,
    )

    # --------------------------------------------------------
    # Step 4: Validate amplitude consistency.
    # --------------------------------------------------------

    amplitude_validation = validate_amplitude(data)

    print()
    print("[AMPLITUDE VALIDATION]")
    print(
        f"Valid rows          : "
        f"{amplitude_validation['valid_rows']:,}"
    )
    print(
        f"Mean absolute error : "
        f"{amplitude_validation['mean_absolute_error']}"
    )
    print(
        f"Max absolute error  : "
        f"{amplitude_validation['max_absolute_error']}"
    )

    # --------------------------------------------------------
    # Step 5: Find sample indices that are zero-only.
    # --------------------------------------------------------

    zero_only_indices = find_zero_only_sample_indices(
        data=data,
        expected_samples=args.expected_samples,
    )

    print()
    print("[ZERO-ONLY SAMPLE INDICES]")
    print(zero_only_indices)
    print(
        "These indices are reported and hidden from heatmaps, "
        "but are not removed from the dataset."
    )

    # --------------------------------------------------------
    # Step 6: Export the complete dataset, retaining QC flags.
    # --------------------------------------------------------

    complete_dataset_path = (
        processed_dir / "csi_dataset_with_session_labels.csv"
    )

    data.to_csv(
        complete_dataset_path,
        index=False,
    )

    print(f"[SAVE] {complete_dataset_path}")

    # --------------------------------------------------------
    # Step 7: Export a separate dataset for training.
    # --------------------------------------------------------
    # Only rows belonging to complete frames are included.
    # A complete frame must pass all checks in the QC report.

    training_data = data.loc[
        data["frame_complete"]
    ].copy()

    training_dataset_path = (
        processed_dir / "csi_dataset_complete_frames.csv"
    )

    training_data.to_csv(
        training_dataset_path,
        index=False,
    )

    print(f"[SAVE] {training_dataset_path}")

    # --------------------------------------------------------
    # Step 8: Calculate preliminary frame-level features.
    # --------------------------------------------------------

    features = calculate_frame_features(
        data=data,
        expected_samples=args.expected_samples,
    )

    features_path = (
        processed_dir / "preliminary_frame_features.csv"
    )

    features.to_csv(
        features_path,
        index=False,
    )

    print(f"[SAVE] {features_path}")

    # --------------------------------------------------------
    # Step 9: Build and export session quality summary.
    # --------------------------------------------------------

    session_summary = build_session_summary(
        sessions=sessions,
        quality_report=quality_report,
        data=data,
        zero_only_indices=zero_only_indices,
    )

    summary_path = (
        processed_dir / "session_quality_summary.csv"
    )

    session_summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"[SAVE] {summary_path}")

    # --------------------------------------------------------
    # Step 10: Cross-frame (temporal) variability features.
    # --------------------------------------------------------
    # std_amplitude varies between samples inside one frame.
    # These features vary between frames for the SAME sample index,
    # which is the part that reacts to human motion.

    temporal_features = calculate_temporal_features(
        data=data,
        expected_samples=args.expected_samples,
        window_seconds=args.window_seconds,
    )

    temporal_path = (
        processed_dir / "temporal_features.csv"
    )

    temporal_features.to_csv(
        temporal_path,
        index=False,
    )

    print(f"[SAVE] {temporal_path}")

    if not temporal_features.empty:
        sampling = (
            temporal_features.groupby("session_id")
            [["sampling_interval_ms", "sampling_fps"]]
            .first()
        )

        print()
        print("[SAMPLING RATE CHECK]")
        print(sampling.round(2).to_string())

        window_values = temporal_features.groupby(
            "session_id"
        )["window_frames"].first()

        print(
            f"Window requested: {args.window_seconds} s -> "
            f"{sorted(set(window_values))} frames"
        )

        print()
        print("[TEMPORAL VARIABILITY BY CLASS]")
        print(
            temporal_features.groupby("scenario_label")[
                "temporal_std_amplitude"
            ]
            .describe()[
                ["count", "mean", "std", "min", "50%", "max"]
            ]
            .round(3)
            .to_string()
        )

    # --------------------------------------------------------
    # Step 10b: Optional normalization (statistics from train only).
    # --------------------------------------------------------
    # Off by default. Only the exported feature tables are scaled;
    # the raw sample rows (i, q, amplitude) are never rescaled.

    if args.normalization == "none":
        print()
        print("[NORMALIZATION] skipped (method=none).")
    else:
        print()
        print(
            f"[NORMALIZATION] method={args.normalization}, "
            f"train sessions={len(args.train_sessions) or 'ALL'}"
        )

        if not args.train_sessions:
            print(
                "[NORMALIZATION][WARNING] --train-sessions not set: "
                "statistics are fitted on every row, so any test split "
                "built from this output leaks. Set the split agreed "
                "with the ML owner before training."
            )

        param_tables = []

        for name, table in (
            ("preliminary_frame_features", features),
            ("temporal_features", temporal_features),
        ):
            if table.empty:
                print(f"[NORMALIZATION] {name}: empty, skipped.")
                continue

            scaled, params, notes = normalize_features(
                features=table,
                method=args.normalization,
                train_sessions=args.train_sessions,
                dataset_name=name,
            )

            scaled_path = processed_dir / f"{name}_normalized.csv"
            scaled.to_csv(scaled_path, index=False)
            print(f"[SAVE] {scaled_path}")

            if not params.empty:
                param_tables.append(params)

            for note in notes:
                print(f"[NORMALIZATION][NOTE] {note}")

        if param_tables:
            all_params = pd.concat(param_tables, ignore_index=True)
            params_path = processed_dir / "normalization_params.csv"
            all_params.to_csv(params_path, index=False)
            print(f"[SAVE] {params_path}")
            print()
            print("[NORMALIZATION PARAMETERS]")
            print(all_params.round(4).to_string(index=False))

    # --------------------------------------------------------
    # Step 11: Noise-filter experiment (before/after).
    # --------------------------------------------------------
    # The filtered series is written only to a separate evaluation
    # report. The exported dataset is never overwritten by it.

    filter_report = evaluate_filter(
        data=data,
        expected_samples=args.expected_samples,
        filter_frames=args.filter_frames,
        window_seconds=args.window_seconds,
    )

    filter_path = (
        processed_dir / "filter_evaluation.csv"
    )

    filter_report.to_csv(
        filter_path,
        index=False,
    )

    print(f"[SAVE] {filter_path}")

    if not filter_report.empty:
        print()
        print("[FILTER EVALUATION]")
        print(
            filter_report[
                [
                    "session_id",
                    "scenario_label",
                    "hf_jitter_raw",
                    "hf_jitter_filtered",
                    "noise_reduction_pct",
                    "low_frequency_correlation",
                    "retained_variance_pct",
                    "temporal_std_retained_pct",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

        print()
        print("[FILTER EVALUATION BY CLASS]")
        print(
            filter_report.groupby("scenario_label")
            [
                [
                    "noise_reduction_pct",
                    "low_frequency_correlation",
                    "retained_variance_pct",
                    "temporal_std_retained_pct",
                ]
            ]
            .mean()
            .round(3)
            .to_string()
        )

    # --------------------------------------------------------
    # Step 12: Generate plots.
    # --------------------------------------------------------

    save_session_heatmaps(
        sessions=sessions,
        results_dir=str(results_dir),
        zero_only_indices=zero_only_indices,
    )

    save_mean_amplitude_plot(
        data=data,
        results_dir=str(results_dir),
    )

    save_motion_score_plot(
        features=features,
        results_dir=str(results_dir),
    )

    save_temporal_variability_plot(
        temporal_features=temporal_features,
        results_dir=str(results_dir),
    )

    save_filter_comparison_plot(
        data=data,
        expected_samples=args.expected_samples,
        filter_frames=args.filter_frames,
        results_dir=str(results_dir),
    )

    # --------------------------------------------------------
    # Step 13: Print final summary.
    # --------------------------------------------------------

    total_frames = int(len(quality_report))
    total_complete = int(quality_report["frame_complete"].sum())
    total_incomplete = total_frames - total_complete

    total_rows = int(len(data))
    training_rows = int(len(training_data))

    print()
    print("=" * 65)
    print("PIPELINE SUMMARY")
    print("=" * 65)

    print(f"Sessions             : {len(sessions)}")
    print(f"Total raw rows       : {total_rows:,}")
    print(f"Total unique frames  : {total_frames:,}")
    print(f"Complete frames      : {total_complete:,}")
    print(f"Incomplete frames    : {total_incomplete:,}")

    incomplete_rate = (
        total_incomplete / total_frames * 100
        if total_frames > 0
        else 0.0
    )

    print(f"Incomplete rate      : {incomplete_rate:.2f}%")
    print(f"Training dataset rows: {training_rows:,}")
    print(
        f"Zero-only indices    : {len(zero_only_indices)}"
    )

    print()
    print("Frames by scenario:")
    print(
        quality_report.groupby("scenario_label")[
            "frame_complete"
        ].agg(["sum", "count"])
    )

    print()
    print("Session quality summary:")
    print(
        session_summary[
            [
                "session_id",
                "unique_frames",
                "complete_frames",
                "incomplete_frames",
            ]
        ].to_string(index=False)
    )

    print()
    print("[DONE] CSI preprocessing completed successfully.")


if __name__ == "__main__":
    main()
