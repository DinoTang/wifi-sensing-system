"""A room reference recorded explicitly while the sensing area is empty."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

REFERENCE_VERSION = 1
CALIBRATION_SECONDS = 30
SETTLE_SECONDS = 5
MIN_WINDOWS = 30
MIN_SPAN_MS = 10_000
MAX_WINDOWS = 180
FEATURE_COUNT = 36


def room_features(window: np.ndarray) -> np.ndarray:
    # Per-link centering suppresses receiver gain changes. The median keeps a
    # single packet excursion out of the stored room spectrum.
    frames = np.log1p(np.asarray(window, dtype=np.float32)).reshape(20, 4, 56)
    frames -= frames.mean(axis=2, keepdims=True)
    spectrum = np.median(frames, axis=0).reshape(4, 8, 7).mean(axis=2).ravel()
    dynamics = np.median(np.abs(np.diff(frames, axis=0)), axis=(0, 2))
    return np.concatenate((spectrum, dynamics))


def write_json(path: Path, value: dict) -> None:
    """Replace the previous complete file only after the new JSON is closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class EmptyRoomReference:
    center: np.ndarray
    scale: np.ndarray
    threshold: float
    session_id: str
    created_ms: int
    windows: int
    configuration: dict

    @classmethod
    def fit(cls, features, timestamps, *, session_id, created_ms, configuration):
        rows = np.asarray(features, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        if (rows.ndim != 2 or rows.shape[1] != FEATURE_COUNT or not MIN_WINDOWS <= len(rows) <= MAX_WINDOWS
                or timestamps.shape != (len(rows),) or np.any(np.diff(timestamps) <= 0)
                or not np.isfinite(rows).all()
                or timestamps[-1] - timestamps[0] < MIN_SPAN_MS):
            raise ValueError("Not enough valid empty-room CSI for calibration")
        center = np.median(rows, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(rows - center), axis=0), .015)
        distances = np.sqrt(np.mean(((rows - center) / scale) ** 2, axis=1))
        threshold = max(2., float(np.quantile(distances, .99)) * 2.)
        return cls(center, scale, threshold, session_id, created_ms, len(rows), configuration)

    def compare(self, window: np.ndarray) -> dict:
        distance = float(np.sqrt(np.mean(((room_features(window) - self.center) / self.scale) ** 2)))
        return {"matched": distance <= self.threshold, "distance": round(distance, 4),
                "threshold": round(self.threshold, 4), "session_id": self.session_id}

    def summary(self) -> dict:
        return {"session_id": self.session_id, "created_ms": self.created_ms,
                "windows": self.windows, "configuration": self.configuration}

    def save(self, path: Path) -> None:
        write_json(path, {"reference_version": REFERENCE_VERSION, **self.summary(),
                          "center": self.center.tolist(), "scale": self.scale.tolist(),
                          "threshold": self.threshold})

    @classmethod
    def load(cls, path: Path, *, configuration: dict | None = None):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Invalid empty-room reference document")
        if value.get("reference_version") != REFERENCE_VERSION:
            raise ValueError("Unsupported empty-room reference version")
        stored_config = value.get("configuration")
        if not isinstance(stored_config, dict):
            raise ValueError("Invalid empty-room capture configuration")
        if configuration is not None and any(stored_config.get(k) != v for k, v in configuration.items()):
            raise ValueError("Empty-room reference uses another capture configuration")
        center, scale = (np.asarray(value[key], dtype=np.float32) for key in ("center", "scale"))
        threshold = float(value["threshold"])
        if (center.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,)
                or not np.isfinite(center).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0) or not np.isfinite(threshold) or threshold <= 0
                or not isinstance(value["session_id"], str) or not value["session_id"]
                or not isinstance(value["created_ms"], int) or value["created_ms"] < 0
                or not isinstance(value["windows"], int) or not MIN_WINDOWS <= value["windows"] <= MAX_WINDOWS):
            raise ValueError("Invalid empty-room reference")
        return cls(center, scale, threshold, value["session_id"], value["created_ms"],
                   value["windows"], value["configuration"])
