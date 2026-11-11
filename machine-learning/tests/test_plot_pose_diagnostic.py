import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_pose_diagnostic import DEFAULT_RESULTS_DIR, _activity_label


def test_activity_label_supports_round_and_capture_session_names():
    assert _activity_label("Lan3_P01_trial03_left_arm") == "left_arm"
    assert _activity_label("P01_walking_01_20261103-025859_EiZQ") == "walking"


def test_default_results_use_combined_pose_sync_evaluation():
    assert DEFAULT_RESULTS_DIR.parts[-3:] == (
        "pose-sync-check",
        "combined-all-rounds",
        "training",
    )
