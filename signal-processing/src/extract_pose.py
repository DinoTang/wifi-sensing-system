"""Extract 2D pose landmarks from the session videos with MediaPipe.

Each session video is decoded at its native rate but only every ``--step``-th
frame is inferred (default 3 → 10 Hz, matching the ~9.9 Hz CSI frame rate).
Frames are downscaled before inference for speed; landmarks are stored in
**pixel coordinates of the original frame** plus the normalised value and the
visibility score, so nothing is lost.

Output:
    data/pose/landmarks/<activity>_<repeat>.csv   one row per inferred frame
    data/pose/landmarks_summary.csv               coverage per session

Landmarks follow the MediaPipe BlazePose order (33 points); the CSV column names
are the landmark names, so the mapping to COCO-17 can be done later without
re-reading the video.

Run from anywhere::

    python signal-processing/src/extract_pose.py
    python signal-processing/src/extract_pose.py --step 3 --width 640
    python signal-processing/src/extract_pose.py --sessions left_arm_01 --limit 90
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT / "data" / "pose" / "index.csv"
DEFAULT_MODEL = PROJECT / "models" / "pose_landmarker_lite.task"
DEFAULT_OUTPUT = PROJECT / "data" / "pose" / "landmarks"

# MediaPipe BlazePose landmark order.
LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# Joints that must be visible for the sample to be usable for an arm activity.
KEY_JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

VISIBILITY_MIN = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pose landmarks from videos")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step", type=int, default=3, help="Lấy mỗi N frame")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng khi suy luận")
    parser.add_argument(
        "--sessions",
        default="",
        help="Chỉ chạy các phiên này, ví dụ: left_arm_01,walking_01",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Giới hạn số frame suy luận (để thử)"
    )
    return parser.parse_args()


def build_landmarker(model_path: Path, running_mode):
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main() -> None:
    args = parse_args()
    if not Path(args.model).exists():
        raise SystemExit(
            f"Không thấy model {args.model}. Tải pose_landmarker_lite.task vào models/."
        )
    if not Path(args.index).exists():
        raise SystemExit(f"Không thấy {args.index}. Chạy build_pose_index.py trước.")

    import mediapipe as mp

    with Path(args.index).open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))

    wanted = {item.strip() for item in args.sessions.split(",") if item.strip()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header = ["frame_index", "video_time_s", "person_present"]
    for name in LANDMARKS:
        header += [f"{name}_x", f"{name}_y", f"{name}_vis"]

    summaries = []
    print("=" * 78)
    print("TRÍCH XUẤT TƯ THẾ (MediaPipe PoseLandmarker)")
    print("=" * 78)

    for row in index_rows:
        key = f"{row['activity']}_{row['repeat']}"
        if wanted and key not in wanted:
            continue
        video_path = PROJECT / row["video_file"]
        if not video_path.exists():
            print(f"[SKIP] {key}: không thấy video")
            continue

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # A fresh landmarker per session: VIDEO mode requires timestamps to
        # increase for the lifetime of the object, so reusing one across videos
        # that both start at t=0 raises "Input timestamp must be monotonically
        # increasing".
        landmarker = build_landmarker(
            Path(args.model), mp.tasks.vision.RunningMode.VIDEO
        )

        out_path = output_dir / f"{key}.csv"
        frame_index = 0
        rows_written = 0
        frames_with_person = 0
        frames_with_key_joints = 0
        started = time.time()

        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % args.step == 0:
                    scale = args.width / width
                    small = cv2.resize(
                        frame,
                        (args.width, int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int(round(frame_index / fps * 1000))
                    result = landmarker.detect_for_video(image, timestamp_ms)

                    values: list[float] = []
                    present = 0
                    visible_key_joints = 0
                    if result.pose_landmarks:
                        present = 1
                        marks = result.pose_landmarks[0]
                        for index, name in enumerate(LANDMARKS):
                            mark = marks[index]
                            values += [
                                round(mark.x * width, 2),
                                round(mark.y * height, 2),
                                round(mark.visibility, 3),
                            ]
                            if name in KEY_JOINTS and mark.visibility >= VISIBILITY_MIN:
                                visible_key_joints += 1
                    else:
                        values = [float("nan")] * (3 * len(LANDMARKS))

                    if present:
                        frames_with_person += 1
                        if visible_key_joints >= len(KEY_JOINTS) - 1:
                            frames_with_key_joints += 1

                    writer.writerow(
                        [
                            frame_index,
                            round(frame_index / fps, 3),
                            present,
                            *values,
                        ]
                    )
                    rows_written += 1
                    if args.limit and rows_written >= args.limit:
                        break
                frame_index += 1

        capture.release()
        landmarker.close()
        elapsed = time.time() - started
        coverage = (frames_with_person / rows_written * 100) if rows_written else 0
        summaries.append(
            {
                "session": key,
                "activity": row["activity"],
                "repeat": row["repeat"],
                "video_frames": total,
                "inferred_frames": rows_written,
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "frames_with_person": frames_with_person,
                "person_coverage_pct": round(coverage, 1),
                "frames_with_key_joints": frames_with_key_joints,
                "key_joint_coverage_pct": round(
                    frames_with_key_joints / rows_written * 100, 1
                )
                if rows_written
                else 0,
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"[OK] {key:<16} inferred {rows_written:>4}/{total:<5} "
            f"person {coverage:>5.1f}%  ({elapsed:.0f}s)"
        )

    if not summaries:
        raise SystemExit("Không trích xuất được phiên nào.")

    summary_path = output_dir.parent / "landmarks_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print()
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
