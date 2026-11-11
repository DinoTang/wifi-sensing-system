import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_pose_windows import LANDMARKS, load_pose, sync_alignment


def _landmark_frame(width=1920, height=1080):
    row = {
        "video_time_s": 0.0,
        "person_present": 1,
        "frame_width": width,
        "frame_height": height,
    }
    for landmark in LANDMARKS:
        row[f"{landmark}_x"] = 100.0
        row[f"{landmark}_y"] = 200.0
        row[f"{landmark}_vis"] = 0.9
    return pd.DataFrame([row])


def test_load_pose_returns_and_checks_video_dimensions(tmp_path):
    path = tmp_path / "pose.csv"
    _landmark_frame().to_csv(path, index=False)

    pose = load_pose(path)

    assert pose[0].tolist() == [0.0]
    assert pose[5:] == (1920, 1080)


def test_load_pose_rejects_missing_image_dimensions(tmp_path):
    path = tmp_path / "pose.csv"
    frame = _landmark_frame().drop(columns="frame_width")
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing frame_width"):
        load_pose(path)


def test_load_pose_rejects_inconsistent_image_dimensions(tmp_path):
    path = tmp_path / "pose.csv"
    frame = pd.concat([_landmark_frame(), _landmark_frame(width=1280)])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="inconsistent frame_width"):
        load_pose(path)


def test_sync_alignment_keeps_estimated_offset_unverified():
    assert sync_alignment(
        {
            "verdict": "estimated_unverified",
            "best_offset_s": "-35.2",
            "sync_source": "user_reported_video_start",
        },
        assumed_offset=0.0,
    ) == (-35.2, 0, "user_reported_video_start")


def test_sync_alignment_accepts_explicit_verified_offset():
    assert sync_alignment(
        {
            "verdict": "usable",
            "best_offset_s": "-44.5",
            "sync_source": "visual_clock_anchors",
        },
        assumed_offset=0.0,
    ) == (-44.5, 1, "visual_clock_anchors")


def test_sync_alignment_rejects_usable_row_without_offset():
    with pytest.raises(ValueError, match="must include best_offset_s"):
        sync_alignment({"verdict": "usable"}, assumed_offset=0.0)
