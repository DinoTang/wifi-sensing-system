"""Presence features and lightweight inference for Intel VHT20 windows."""

from dataclasses import dataclass

import numpy as np

SPIKE_STEP_RATIO = 4.0
SPIKE_STEP_FLOOR = .025
NULL_TONES = (7, 21, 34, 48, 63, 77, 90, 104, 119, 133, 146, 160, 175, 189, 202, 216)
SHAPE_FEATURE_COUNT = 76


def active_link_amplitudes(windows: np.ndarray) -> np.ndarray:
    """Four antenna links and their 52 measured tones in the Intel VHT20 layout."""
    values = np.asarray(windows, dtype=np.float32).reshape(-1, 20, 4, 56)
    return np.delete(values, NULL_TONES[:4], axis=3)


def normalize_pose_links(windows: np.ndarray) -> np.ndarray:
    """Cancel per-packet receiver gain while preserving each antenna link's spectral shape."""
    values = np.asarray(windows, dtype=np.float32).reshape(-1, 20, 4, 56).copy()
    values[..., NULL_TONES[:4]] = 0
    rms = np.sqrt((values ** 2).sum(axis=-1, keepdims=True) / 52)
    return (100 * values / np.maximum(rms, 1.)).reshape(-1, 20, 224)


def coherence_features(windows: np.ndarray) -> np.ndarray:
    """Measure time variation shared by neighboring tones, relative to local noise."""
    links = active_link_amplitudes(windows)
    count = len(links)
    links /= np.maximum(np.sqrt(np.mean(links ** 2, axis=3, keepdims=True)), 1.)
    residual = links - np.median(links, axis=1, keepdims=True)
    # A transient bad packet should not dominate a complete two-second window.
    limit = 6 * np.median(np.abs(residual), axis=1, keepdims=True) + .008
    residual = np.clip(residual, -limit, limit)

    def quantiles(values):
        return np.quantile(values.reshape(count, -1), (.1, .5, .9), axis=1).T

    features = []
    for width in (4, 13):
        bands = residual.reshape(count, 20, 4, 52 // width, width)
        even = bands[..., ::2].mean(axis=-1)
        odd = bands[..., 1::2].mean(axis=-1)
        even -= even.mean(axis=1, keepdims=True)
        odd -= odd.mean(axis=1, keepdims=True)
        for lag in (0, 1, 3, 6):
            first, second = even[:, lag:], odd[:, :20 - lag]
            rms = np.sqrt((first ** 2).mean(axis=1) * (second ** 2).mean(axis=1))
            features.append(quantiles((first * second).mean(axis=1) / np.maximum(rms, 1e-7)))

        shared = (even + odd) / 2
        noise = np.median(np.abs(np.diff(even - odd, axis=1)), axis=1)
        for lag in (1, 4):
            change = np.median(np.abs(shared[:, lag:] - shared[:, :-lag]), axis=1)
            features.append(quantiles(change / np.maximum(noise, .0007)))

        # Fraction of the temporal variance carried by the strongest shared patterns.
        shared /= np.sqrt((shared ** 2).mean(axis=1, keepdims=True) + 1e-6)
        matrix = shared.reshape(count, 20, -1)
        covariance = (matrix @ matrix.transpose(0, 2, 1)) / matrix.shape[2]
        eigenvalues = np.linalg.eigvalsh(covariance)
        total = np.maximum(eigenvalues.sum(axis=1, keepdims=True), 1e-6)
        for components in (1, 3):
            features.append(eigenvalues[:, -components:].sum(axis=1, keepdims=True) / total)
    return np.log1p(np.maximum(np.column_stack(features), -.99)).astype(np.float32)


def motion_and_spectrum(windows):
    """Measure sustained change in both halves, and the local normalized spectrum."""
    links = active_link_amplitudes(windows)
    links /= np.maximum(np.sqrt(np.mean(links ** 2, axis=-1, keepdims=True)), 1.)
    bands = links.reshape(-1, 20, 4, 13, 4).mean(axis=-1)
    spectrum = np.median(bands, axis=1).reshape(len(links), 52)
    halves = bands.reshape(-1, 2, 10, 4, 13)
    changes = []
    for lag in (1, 3, 5):
        displacement = np.abs(halves[:, :, lag:] - halves[:, :, :-lag]).mean(axis=-1)
        quantiles = np.quantile(displacement, (.35, .65, .85), axis=(2, 3))
        changes.append(quantiles.min(axis=2).T)
    return np.log1p(32 * np.column_stack(changes)), spectrum


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
    features = staticmethod(presence_features)
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    @classmethod
    def from_config(cls, config: dict):
        kind = config.get("kind")
        if kind == "gain_invariant_forest_v1":
            return PresenceForest.from_config(config)
        if kind == "supported_motion_v1":
            return SupportedMotion.from_config(config)
        if kind == "temporal_coherence_v1":
            cls = CoherencePresenceClassifier
        elif kind is not None:
            raise ValueError(f"Unknown CSI presence classifier: {kind}")
        center, scale, weights = (
            np.asarray(config[key], dtype=np.float64) for key in ("center", "scale", "weights")
        )
        intercept = float(config["intercept"])
        feature_count = cls.features(np.zeros((1, 20, 224), dtype=np.float32)).shape[1]
        if any(values.shape != (feature_count,) or not np.all(np.isfinite(values))
               for values in (center, scale, weights)) or np.any(scale <= 0) or not np.isfinite(intercept):
            raise ValueError("Invalid CSI presence classifier parameters")
        return cls(center, scale, weights, intercept)

    def logit(self, window: np.ndarray) -> float:
        features = self.features(window[None, :, :])[0]
        return float((features - self.center) @ (self.weights / self.scale) + self.intercept)

    def probability(self, window: np.ndarray) -> float:
        logit = self.logit(window)
        if logit >= 0:
            corrected = interpolate_isolated_spike(window)
            if corrected is not None:
                logit = min(logit, self.logit(corrected))
        return float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))


class CoherencePresenceClassifier(PresenceClassifier):
    confidence_method = "csi_temporal_coherence_score"
    features = staticmethod(coherence_features)

    def probability(self, window: np.ndarray) -> float:
        return float(1 / (1 + np.exp(-np.clip(self.logit(window), -60, 60))))


@dataclass(frozen=True)
class SupportedMotion:
    """Use weak temporal evidence only near a learned, distinct still-person spectrum."""

    confidence_method = "csi_supported_motion_score"
    coherence: CoherencePresenceClassifier
    motion_weights: np.ndarray
    motion_bias: float
    occupied_spectra: np.ndarray
    empty_spectra: np.ndarray

    @classmethod
    def from_config(cls, config):
        coherence = PresenceClassifier.from_config(config["coherence"])
        weights = np.asarray(config["motion_weights"], dtype=np.float64)
        bias = float(config["motion_bias"])
        spectra = [np.asarray(config[k], dtype=np.float32) for k in ("occupied_spectra", "empty_spectra")]
        if (not isinstance(coherence, CoherencePresenceClassifier) or weights.shape != (9,)
                or not np.isfinite(weights).all() or not np.isfinite(bias)
                or any(s.ndim != 2 or s.shape[1] != 52 or not 1 <= len(s) <= 1024
                       or not np.isfinite(s).all() for s in spectra)):
            raise ValueError("Invalid motion or quiet-scene classifier")
        return cls(coherence, weights, bias, *spectra)

    def scores(self, temporal, changes, spectra):
        logits = changes @ self.motion_weights + self.motion_bias
        motion = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        logits = (temporal - self.coherence.center) @ (self.coherence.weights / self.coherence.scale)
        coherence = 1 / (1 + np.exp(-np.clip(logits + self.coherence.intercept, -60, 60)))

        def distance(templates):
            delta = spectra[:, None, :] - templates[None, :, :]
            return np.sqrt(np.min(np.mean(delta ** 2, axis=2), axis=1))

        occupied, empty = distance(self.occupied_spectra), distance(self.empty_spectra)
        # RMS distances use normalized link amplitudes. Outside the support of a
        # quiet occupied scene, or near an empty one, require sustained motion.
        support = np.clip((.12 - occupied) / .06, 0., 1.) * np.clip((empty - occupied) / .06, 0., 1.)
        return np.maximum(motion, np.minimum(coherence, support))

    def probability(self, window):
        windows = window[None, :, :]
        changes, spectra = motion_and_spectrum(windows)
        return float(self.scores(coherence_features(windows), changes, spectra)[0])


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
