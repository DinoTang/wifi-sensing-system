"""Presence features and lightweight inference for Intel VHT20 windows."""

from dataclasses import dataclass

import numpy as np


def presence_features(windows: np.ndarray) -> np.ndarray:
    """Summarize pose-independent variations and the per-link spectrum."""
    windows = np.asarray(windows, dtype=np.float32)
    count = len(windows)
    links = windows.reshape(count, 20, 4, 56)
    rms = np.sqrt(np.mean(links ** 2, axis=-1, keepdims=True))
    shape = np.log1p(links / np.maximum(rms, 1))
    relative = windows / np.maximum(windows.mean(axis=1, keepdims=True), 12)
    log_amplitude = np.log1p(windows).mean(axis=-1)
    log_rms = np.log1p(rms[..., 0])
    return np.column_stack((
        windows.std(axis=1).mean(axis=1),
        np.abs(np.diff(windows, axis=1)).mean(axis=(1, 2)),
        relative.std(axis=1).mean(axis=1),
        np.abs(np.diff(relative, axis=1)).mean(axis=(1, 2)),
        log_amplitude.std(axis=1),
        np.diff(log_amplitude, axis=1).std(axis=1),
        shape.std(axis=1).mean(axis=-1),
        np.abs(np.diff(shape, axis=1)).mean(axis=(1, 3)),
        links.std(axis=1).mean(axis=-1) / np.maximum(rms.mean(axis=1)[..., 0], 1),
        shape.reshape(count, 20, 4, 8, 7).mean(axis=(1, 4)).reshape(count, 32),
        log_rms.mean(axis=1),
        log_rms.std(axis=1),
    ))


@dataclass(frozen=True)
class PresenceClassifier:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    @classmethod
    def from_config(cls, config: dict):
        center, scale, weights = (
            np.asarray(config[key], dtype=np.float64) for key in ("center", "scale", "weights")
        )
        intercept = float(config["intercept"])
        feature_count = presence_features(np.zeros((1, 20, 224), dtype=np.float32)).shape[1]
        if any(values.shape != (feature_count,) or not np.all(np.isfinite(values))
               for values in (center, scale, weights)) or np.any(scale <= 0) or not np.isfinite(intercept):
            raise ValueError("Invalid CSI presence classifier parameters")
        return cls(center, scale, weights, intercept)

    def probability(self, window: np.ndarray) -> float:
        features = presence_features(window[None, :, :])[0]
        logit = (features - self.center) @ (self.weights / self.scale) + self.intercept
        return float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))
