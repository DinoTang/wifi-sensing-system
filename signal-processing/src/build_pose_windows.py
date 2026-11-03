"""Join CSI windows with the pose extracted from the video.

For every CSI window (default 2.0 s, step 0.2 s) the script attaches the pose
that belongs to the **last frame of the window**, because that is the pose a
real-time system would have when the window ends.

The time offset between CSI and video comes from ``sync_report.csv`` when its
verdict is ``usable``. In this capture round no session reached that verdict, so
the script falls back to ``--assume-offset 0`` and marks every sample
``sync_ok=0`` — the samples are still produced (so the pipeline is complete and
testable) but they must not be used for training until the offset is verified.

Output:
    data/pose/windows/<activity>_<repeat>.npz   csi, keypoints, visibility, mask, ...
    data/pose/windows_index.csv                 one row per window
    data/pose/windows_summary.csv               per-session counts

Run from anywhere::

    python signal-processing/src/build_pose_windows.py
    python signal-processing/src/build_pose_windows.py --window 2.0 --step 0.2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT / "data" / "pose" / "index.csv"
DEFAULT_SYNC = PROJECT / "data" / "pose" / "sync_report.csv"
DEFAULT_ROOT = PROJECT / "data" / "pose"

LANDMARK_COUNT = 33
KEY_JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
KEY_INDEX = [LANDMARKS.index(name) for name in KEY_JOINTS]

VISIBILITY_MIN = 0.3
MIN_KEY_JOINTS = 5
MAX_POSE_DT_S = 0.15
MAX_GAP_S = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ghép cửa sổ CSI với tư thế")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--sync-report", type=Path, default=DEFAULT_SYNC)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--window", type=float, default=2.0, help="Độ dài cửa sổ (giây)")
    parser.add_argument("--step", type=float, default=0.2, help="Bước trượt (giây)")
    parser.add_argument(
        "--assume-offset",
        type=float,
        default=0.0,
        help="Offset dùng khi không có phiên nào đủ tin cậy (giây)",
    )
    return parser.parse_args()


def read_sync(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            f"{row['activity']}_{row['repeat']}": row
            for row in csv.DictReader(handle)
        }


def load_csi(csv_path: Path):
    """Return (amplitudes [frames, tones], frame_ids, timestamps ms)."""
    frame = pd.read_csv(csv_path, usecols=["frame_id", "timestamp_ms", "sample_index", "amplitude"])
    pivot = frame.pivot_table(
        index="frame_id", columns="sample_index", values="amplitude", aggfunc="first"
    ).sort_index()
    meta = frame.groupby("frame_id", sort=True)["timestamp_ms"].first()
    return (
        pivot.to_numpy(dtype=np.float32),
        pivot.index.to_numpy(),
        meta.to_numpy(dtype=np.float64),
    )


def load_pose(path: Path):
    frame = pd.read_csv(path)
    times = frame["video_time_s"].to_numpy(dtype=np.float64)
    xs = np.column_stack([frame[f"{name}_x"].to_numpy() for name in LANDMARKS])
    ys = np.column_stack([frame[f"{name}_y"].to_numpy() for name in LANDMARKS])
    vis = np.column_stack([frame[f"{name}_vis"].to_numpy() for name in LANDMARKS])
    present = frame["person_present"].to_numpy(dtype=np.int8)
    return times, xs, ys, vis, present


def pose_at(times, xs, ys, vis, present, target_s):
    """Nearest pose sample; returns None if further than MAX_POSE_DT_S."""
    if len(times) == 0:
        return None
    index = int(np.argmin(np.abs(times - target_s)))
    delta = abs(times[index] - target_s)
    if delta > MAX_POSE_DT_S:
        return None
    return index, float(delta)


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Không thấy {index_path}. Chạy build_pose_index.py trước.")

    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    sync = read_sync(Path(args.sync_report))

    root = Path(args.root)
    windows_dir = root / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []

    print("=" * 84)
    print("GHÉP CỬA SỔ CSI + TƯ THẾ")
    print("=" * 84)

    for row in index_rows:
        key = f"{row['activity']}_{row['repeat']}"
        activity = row["activity"]
        csi_path = root / "raw_csv" / f"{key}.csv"
        pose_path = root / "landmarks" / f"{key}.csv"
        if not csi_path.exists() or not pose_path.exists():
            print(f"[SKIP] {key}: thiếu CSV CSI hoặc landmarks")
            continue

        sync_row = sync.get(key, {})
        if sync_row.get("verdict") == "usable":
            offset_s = float(sync_row["best_offset_s"])
            sync_ok = 1
            sync_source = "motion_correlation"
        else:
            offset_s = args.assume_offset
            sync_ok = 0
            sync_source = "assumed"

        amps, frame_ids, ts_ms = load_csi(csi_path)
        times, xs, ys, vis, present = load_pose(pose_path)

        dt = float(np.median(np.diff(ts_ms))) / 1000.0
        window = max(2, int(round(args.window / dt)))
        step = max(1, int(round(args.step / dt)))

        csi_windows = []
        keypoints = []
        visibility = []
        masks = []
        window_meta = []
        rejected = {"gap": 0, "no_pose": 0, "person_mismatch": 0, "few_joints": 0}

        for end in range(window - 1, len(frame_ids), step):
            start = end - window + 1
            chunk = amps[start : end + 1]
            if chunk.shape[0] != window or not np.isfinite(chunk).all():
                rejected["gap"] += 1
                continue
            gaps = np.diff(ts_ms[start : end + 1]) / 1000.0
            if gaps.max() > MAX_GAP_S:
                rejected["gap"] += 1
                continue

            # Pose time: CSI window end relative to CSI start, minus the offset.
            target = (ts_ms[end] - ts_ms[0]) / 1000.0 - offset_s
            found = pose_at(times, xs, ys, vis, present, target)
            if found is None:
                rejected["no_pose"] += 1
                continue
            index, delta = found

            detector_person = int(present[index])
            visible = (vis[index] >= VISIBILITY_MIN).astype(np.int8)
            joint_ok = int(visible[KEY_INDEX].sum())

            if activity == "empty":
                # The session itself says nobody is there, and in this room the
                # detector does produce false positives (89.7% of frames on
                # empty_01). The session label wins; the detector value is kept
                # next to it so the disagreement stays visible.
                person = 0
                visible = np.zeros(LANDMARK_COUNT, dtype=np.int8)
                joint_ok = 0
            else:
                if detector_person != 1:
                    rejected["person_mismatch"] += 1
                    continue
                if joint_ok < MIN_KEY_JOINTS:
                    rejected["few_joints"] += 1
                    continue
                person = 1

            kp = np.full((LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
            if person == 1:
                kp[:, 0] = xs[index]
                kp[:, 1] = ys[index]

            csi_windows.append(chunk.astype(np.float16))
            keypoints.append(kp)
            visibility.append(vis[index].astype(np.float32))
            masks.append(visible.astype(np.uint8))
            window_meta.append(
                {
                    "session": row["session_id"],
                    "activity": activity,
                    "repeat": row["repeat"],
                    "window_index": len(window_meta),
                    "end_frame_id": int(frame_ids[end]),
                    "timestamp_ms": float(ts_ms[end]),
                    "pose_time_s": round(float(target), 3),
                    "pose_dt_ms": round(delta * 1000, 1),
                    "person_present": person,
                    "detector_person_present": detector_person,
                    "key_joints_visible": joint_ok,
                    "sync_ok": sync_ok,
                    "sync_source": sync_source,
                    "offset_s": offset_s,
                }
            )

        if not csi_windows:
            print(f"[WARN] {key}: không có cửa sổ hợp lệ {rejected}")
            continue

        out_path = windows_dir / f"{key}.npz"
        np.savez_compressed(
            out_path,
            csi=np.stack(csi_windows),
            keypoints=np.stack(keypoints),
            visibility=np.stack(visibility),
            mask=np.stack(masks),
            activity=activity,
            session=row["session_id"],
            window_seconds=args.window,
            step_seconds=args.step,
            sampling_interval_ms=dt * 1000,
            offset_s=offset_s,
            sync_ok=sync_ok,
            landmarks=np.array(LANDMARKS),
        )
        all_rows.extend(window_meta)
        summaries.append(
            {
                "session": key,
                "activity": activity,
                "repeat": row["repeat"],
                "csi_frames": len(frame_ids),
                "windows": len(csi_windows),
                "rejected_gap": rejected["gap"],
                "rejected_no_pose": rejected["no_pose"],
                "rejected_person_mismatch": rejected["person_mismatch"],
                "rejected_few_joints": rejected["few_joints"],
                "sync_ok": sync_ok,
                "offset_s": offset_s,
                "file": out_path.name,
            }
        )
        print(
            f"[OK] {key:<16} windows {len(csi_windows):>4} "
            f"(bỏ: gap {rejected['gap']}, no_pose {rejected['no_pose']}, "
            f"person {rejected['person_mismatch']}, joints {rejected['few_joints']})"
        )

    if not all_rows:
        raise SystemExit("Không tạo được cửa sổ nào.")

    index_out = root / "windows_index.csv"
    with index_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_out = root / "windows_summary.csv"
    with summary_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print()
    print(f"windows: {len(all_rows):,}")
    print(f"index  : {index_out}")
    print(f"summary: {summary_out}")
    flagged = {row["sync_ok"] for row in all_rows}
    if 0 in flagged:
        print(
            "LƯU Ý: có mẫu sync_ok=0 — offset CHƯA được kiểm chứng. "
            "Không dùng để huấn luyện cho tới khi đồng bộ xong."
        )


if __name__ == "__main__":
    main()
