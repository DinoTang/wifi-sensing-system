import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from estimate_sync import correlate, lag_bounds, lag_to_offset_s


def test_positive_lag_is_converted_to_window_offset():
    rng = np.random.default_rng(42)
    video = rng.normal(size=800)
    csi = video[120:720]
    lags = np.arange(-20.0, 40.1, 0.1)

    scores = correlate(csi, video, lags)
    best_lag = float(lags[int(np.nanargmax(scores))])

    assert np.isclose(best_lag, 12.0)
    assert np.isclose(lag_to_offset_s(best_lag), -12.0)


def test_lag_bounds_cover_long_video_capture_offsets():
    minimum, maximum = lag_bounds(csi_duration_s=60.0, video_duration_s=580.0)

    assert minimum == -40.0
    assert maximum == 560.0
