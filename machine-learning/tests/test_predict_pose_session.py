import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_pose_session import build_csi_windows


def test_build_csi_windows_uses_trailing_overlapping_windows():
    amplitudes = np.arange(24 * 224, dtype=np.float32).reshape(24, 224)
    timestamps_ms = np.arange(24, dtype=np.float64) * 100.0

    windows, end_timestamps, rejected_gaps = build_csi_windows(
        amplitudes, timestamps_ms, window_frames=20, step_frames=2
    )

    assert windows.shape == (3, 20, 224)
    assert end_timestamps.tolist() == [1900.0, 2100.0, 2300.0]
    assert rejected_gaps == 0
    np.testing.assert_array_equal(windows[0], amplitudes[:20])
    np.testing.assert_array_equal(windows[1], amplitudes[2:22])


def test_build_csi_windows_rejects_windows_crossing_large_gaps():
    amplitudes = np.ones((50, 224), dtype=np.float32)
    timestamps_ms = np.arange(50, dtype=np.float64) * 100.0
    timestamps_ms[25:] += 500.0

    windows, end_timestamps, rejected_gaps = build_csi_windows(
        amplitudes, timestamps_ms, window_frames=20, step_frames=5
    )

    assert end_timestamps.tolist() == [1900.0, 2400.0, 4900.0, 5400.0]
    assert len(windows) == 4
    assert rejected_gaps == 3


@pytest.mark.parametrize(
    ("amplitudes", "timestamps_ms", "message"),
    [
        (np.ones((4, 223)), np.arange(4), "shape"),
        (np.ones((4, 224)), np.arange(3), "one value per CSI frame"),
        (np.ones((4, 224)), np.asarray([0.0, 1.0, 1.0, 3.0]), "strictly increasing"),
    ],
)
def test_build_csi_windows_rejects_invalid_input(amplitudes, timestamps_ms, message):
    with pytest.raises(ValueError, match=message):
        build_csi_windows(amplitudes, timestamps_ms, window_frames=2, step_frames=1)
