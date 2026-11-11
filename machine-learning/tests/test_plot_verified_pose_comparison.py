import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_verified_pose_comparison import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    create_plots,
)


def test_verified_pose_plots_use_saved_metrics_and_heldout_predictions(tmp_path):
    metric_plot, pose_plot = create_plots(DEFAULT_RESULTS_DIR, tmp_path)

    assert metric_plot.is_file()
    assert pose_plot.is_file()
    assert metric_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pose_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert DEFAULT_OUTPUT_DIR.name == "training"
