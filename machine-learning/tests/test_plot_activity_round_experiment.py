import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_activity_round_experiment import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    create_chart,
)


def test_default_round_experiment_results_create_a_png(tmp_path):
    output = tmp_path / "activity-round-results.png"

    create_chart(DEFAULT_RESULTS_DIR, output)

    assert output.is_file()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
