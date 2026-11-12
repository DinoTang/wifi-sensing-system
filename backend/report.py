"""Bounded timing and lifecycle measurements for one CSI session."""

from collections import deque
from hashlib import sha256
from pathlib import Path
import time

import numpy as np

from .source import ROOT

TIMING_SAMPLE_LIMIT = 1024
TRANSITION_LIMIT = 64


def artifact_digests(checkpoint: Path, normalizer: Path) -> dict[str, str]:
    files = {
        "checkpoint": checkpoint,
        "normalizer": normalizer,
        "parser": ROOT / "signal-processing" / "src" / "parse_intel_csi.py",
    }
    hashes = {}
    for name, path in files.items():
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


def summarize(values: deque[float]) -> dict:
    if not values:
        return {"samples": 0, "p50": None, "p95": None, "max": None}
    p50, p95 = np.percentile(list(values), [50, 95])
    return {"samples": len(values), "p50": round(float(p50), 3),
            "p95": round(float(p95), 3), "max": round(max(values), 3)}


class SessionMetrics:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.started_ms = time.time_ns() // 1_000_000
        self.finished: float | None = None
        self.finished_ms: int | None = None
        self.first_prediction: float | None = None
        self.last_prediction: float | None = None
        self.first_timestamp_ms: int | None = None
        self.last_timestamp_ms: int | None = None
        self.max_gap_ms: int | None = None
        self.artifacts: dict[str, str] = {}
        self.transitions: deque[dict] = deque(maxlen=TRANSITION_LIMIT)
        self.transition_count = 0
        self.timings: dict[str, deque[float]] = {
            name: deque(maxlen=TIMING_SAMPLE_LIMIT)
            for name in ("preprocess", "queue", "inference", "backend", "ws_send")
        }
        self.websocket = dict(sent=0, discarded=0, send_timeouts=0)

    def transition(self, status: str, reason: str | None, timestamp_ms: int | None) -> None:
        self.transition_count += 1
        self.transitions.append({
            "status": status, "reason": reason, "timestamp_ms": timestamp_ms,
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3),
        })

    def finish(self) -> None:
        if self.finished is None:
            self.finished = time.perf_counter()
            self.finished_ms = time.time_ns() // 1_000_000

    def record_prediction(self, timing_ms: dict[str, float], completed: float) -> None:
        if self.first_prediction is None:
            self.first_prediction = completed
        self.last_prediction = completed
        for name, value in timing_ms.items():
            self.timings[name].append(value)

    def snapshot(self, predictions: int) -> dict:
        span = (
            self.last_prediction - self.first_prediction
            if self.first_prediction is not None and self.last_prediction is not None
            else 0
        )
        return {
            "started_ms": self.started_ms,
            "finished_ms": self.finished_ms,
            "elapsed_ms": round(((self.finished or time.perf_counter()) - self.started) * 1000, 3),
            "first_capture_timestamp_ms": self.first_timestamp_ms,
            "last_capture_timestamp_ms": self.last_timestamp_ms,
            "max_capture_gap_ms": self.max_gap_ms,
            "prediction_rate_hz": round((predictions - 1) / span, 3) if span > 0 else None,
            "timing_ms": {name: summarize(values) for name, values in self.timings.items()},
            "timing_sample_limit": TIMING_SAMPLE_LIMIT,
            "websocket": self.websocket.copy(),
            "transitions": list(self.transitions),
            "transition_count": self.transition_count,
        }
