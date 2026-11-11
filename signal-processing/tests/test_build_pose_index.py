import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_pose_index import (
    CSI_FILES,
    capture_session_fields,
    capture_session_id,
)


def test_capture_session_uses_session_trial_and_keeps_capture_id():
    capture_id, trial, repeat = capture_session_fields(
        {"trial": "2026110511321713388"},
        {"trial": "03", "activity": "left_arm"},
    )

    assert capture_id == "2026110511321713388"
    assert trial == "03"
    assert repeat == "03"


def test_capture_session_falls_back_to_metadata_when_notes_are_missing():
    capture_id, trial, repeat = capture_session_fields(
        {"trial": "capture-123"},
        {},
    )

    assert (capture_id, trial, repeat) == (
        "capture-123",
        "capture-123",
        "capture-123",
    )


def test_capture_session_id_is_unique_per_activity_capture():
    notes = {"participant": "P01", "activity": "left_arm"}

    session_id = capture_session_id(
        notes,
        capture_id="2026110511321713388",
        trial="03",
        activity="left_arm",
    )

    assert session_id == "P01_trial03_left_arm_2026110511321713388"


def test_screen_clock_is_optional_for_pose_indexing():
    assert "screen-clock.csv" not in CSI_FILES
