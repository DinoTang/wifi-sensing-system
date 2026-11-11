import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract_pose import frame_bounds


def test_frame_bounds_for_requested_video_segment():
    assert frame_bounds(430.0, 503.0, 30.0, 18000) == (12900, 15090)


@pytest.mark.parametrize(
    ("start", "end", "fps", "total"),
    [
        (-1.0, 5.0, 30.0, 300),
        (5.0, 5.0, 30.0, 300),
        (11.0, 0.0, 30.0, 300),
        (0.0, 0.0, 0.0, 300),
    ],
)
def test_frame_bounds_rejects_invalid_ranges(start, end, fps, total):
    with pytest.raises(ValueError):
        frame_bounds(start, end, fps, total)
