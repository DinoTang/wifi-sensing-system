"""Presence features and lightweight inference for Intel VHT20 windows."""

from dataclasses import dataclass

import numpy as np

SPIKE_STEP_RATIO = 4.0
SPIKE_STEP_FLOOR = .025
NULL_TONES = (7, 21, 34, 48, 63, 77, 90, 104, 119, 133, 146, 160, 175, 189, 202, 216)
SHAPE_FEATURE_COUNT = 76


def shape_presence_features(windows: np.ndarray) -> np.ndarray:
    """Summarize relative CSI shape after removing recorder fill and link gain."""
    values = np.asarray(windows, dtype=np.float32)
    active = np.ones(224, dtype=bool)
    active[list(NULL_TONES)] = False
    links = values[:, :, active].reshape(len(values), 20, 4, 52)
    rms = np.sqrt(np.mean(links ** 2, axis=3, keepdims=True))
    links = 100 * links / np.maximum(rms, 1)
    median = np.median(links, axis=1, keepdims=True)
    variation = (links - median) / np.maximum(median, 12)
    robust_scale = np.median(np.abs(variation), axis=1)
    step = np.median(np.abs(np.diff(variation, axis=1)), axis=1)
    sigma = np.std(variation, axis=1)
    log_shape = np.log1p(links)
    log_shape -= log_shape.mean(axis=3, keepdims=True)
    spectrum = np.median(log_shape, axis=1)
    distributions = np.concatenate([
        np.quantile(summary, (.1, .5, .9), axis=2).transpose(1, 0, 2).reshape(len(values), 12)
        for summary in (robust_scale, step, sigma)
    ], axis=1)
    dynamics = np.column_stack((
        np.median(np.abs(np.diff(variation, n=2, axis=1)), axis=(1, 3)),
        np.quantile(np.mean(np.abs(np.diff(variation, axis=1)), axis=3), .9, axis=1),
        np.quantile(log_shape.std(axis=1), .5, axis=2),
    ))
    frequency = np.column_stack((
        np.quantile(spectrum, (.1, .25, .5, .75, .9), axis=2).transpose(1, 0, 2).reshape(len(values), 20),
        spectrum.std(axis=2), np.abs(np.diff(spectrum, axis=2)).mean(axis=2),
    ))
    return np.column_stack((distributions, dynamics, frequency)).astype(np.float32)


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


def interpolate_isolated_spike(window: np.ndarray) -> np.ndarray | None:
    """Replace one large excursion whose two neighbours agree."""
    window = np.asarray(window, dtype=np.float32)
    log_amplitude = np.log1p(window)
    steps = np.abs(np.diff(log_amplitude, axis=0)).mean(axis=1)
    excursion = np.minimum(steps[:-1], steps[1:])
    neighbours = np.abs(log_amplitude[2:] - log_amplitude[:-2]).mean(axis=1)
    threshold = max(SPIKE_STEP_FLOOR, SPIKE_STEP_RATIO * float(np.median(steps)))
    isolated = (excursion > threshold) & (neighbours < .5 * excursion)
    if not isolated.any():
        return None
    index = int(np.argmax(np.where(isolated, excursion, 0))) + 1
    corrected = window.copy()
    corrected[index] = (window[index - 1] + window[index + 1]) / 2
    return corrected


@dataclass(frozen=True)
class PresenceClassifier:
    confidence_method = "csi_presence_logistic_probability"
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    @classmethod
    def from_config(cls, config: dict):
        if config.get("kind") == "gain_invariant_forest_v1":
            return PresenceForest.from_config(config)
        center, scale, weights = (
            np.asarray(config[key], dtype=np.float64) for key in ("center", "scale", "weights")
        )
        intercept = float(config["intercept"])
        feature_count = presence_features(np.zeros((1, 20, 224), dtype=np.float32)).shape[1]
        if any(values.shape != (feature_count,) or not np.all(np.isfinite(values))
               for values in (center, scale, weights)) or np.any(scale <= 0) or not np.isfinite(intercept):
            raise ValueError("Invalid CSI presence classifier parameters")
        return cls(center, scale, weights, intercept)

    def logit(self, window: np.ndarray) -> float:
        features = presence_features(window[None, :, :])[0]
        return float((features - self.center) @ (self.weights / self.scale) + self.intercept)

    def probability(self, window: np.ndarray) -> float:
        logit = self.logit(window)
        if logit >= 0:
            corrected = interpolate_isolated_spike(window)
            if corrected is not None:
                logit = min(logit, self.logit(corrected))
        return float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))


@dataclass(frozen=True)
class PresenceForest:
    """Portable Extra Trees inference using arrays stored with the checkpoint."""

    confidence_method = "csi_presence_forest_probability"
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    occupied: np.ndarray
    max_depth: int

    @classmethod
    def from_config(cls, config):
        feature, left, right = (np.asarray(config[k], dtype=np.int64) for k in ("feature", "left", "right"))
        threshold, occupied = (np.asarray(config[k], dtype=np.float64) for k in ("threshold", "occupied"))
        depth = int(config["max_depth"])
        if (feature.ndim != 2 or not 1 <= len(feature) <= 512 or feature.shape[1] < 1
                or any(a.shape != feature.shape for a in (left, right, threshold, occupied))
                or not 1 <= depth <= 32 or not np.isfinite(threshold).all()
                or not np.isfinite(occupied).all() or np.any((occupied < 0) | (occupied > 1))
                or np.any((feature < -2) | (feature >= SHAPE_FEATURE_COUNT))):
            raise ValueError("Invalid CSI presence forest")
        internal = feature >= 0
        if any(np.any((a[internal] < 0) | (a[internal] >= feature.shape[1])) for a in (left, right)):
            raise ValueError("Invalid CSI presence tree children")
        return cls(feature, threshold, left, right, occupied, depth)

    def probability(self, window: np.ndarray) -> float:
        values = shape_presence_features(window[None, :, :])[0]
        rows = np.arange(len(self.feature))
        nodes = np.zeros(len(rows), dtype=np.int64)
        for _ in range(self.max_depth + 1):
            features = self.feature[rows, nodes]
            active = features >= 0
            if not active.any():
                return float(self.occupied[rows, nodes].mean())
            selected, current = rows[active], nodes[active]
            goes_left = values[features[active]] <= self.threshold[selected, current]
            nodes[active] = np.where(goes_left, self.left[selected, current], self.right[selected, current])
        raise ValueError("CSI presence tree exceeds its recorded depth")

    def logit(self, window: np.ndarray) -> float:
        probability = np.clip(self.probability(window), 1e-9, 1 - 1e-9)
        return float(np.log(probability / (1 - probability)))
