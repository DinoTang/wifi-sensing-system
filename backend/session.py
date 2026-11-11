"""CSI session lifecycle, causal windows and bounded event delivery."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Literal
import uuid

import numpy as np

from .model import (
    DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER, MAX_GAP_MS, NOMINAL_INTERVAL_MS, STRIDE,
    WINDOW_SIZE, load_model,
)
from .source import CaptureError, CaptureReader, Record, VHT_RATE

LOGGER = logging.getLogger("uvicorn.error")
POLL_SECONDS = 0.025
READ_BATCH = 64


@dataclass(frozen=True)
class Settings:
    session_dir: Path
    source: Literal["replay", "live"]
    speed: float = 1.0
    timeout: float = 2.0
    checkpoint: Path = DEFAULT_CHECKPOINT
    normalizer: Path = DEFAULT_NORMALIZER
    autostart: bool = False


@dataclass(frozen=True)
class Window:
    generation: int
    timestamps: tuple[int, ...]
    amplitudes: np.ndarray
    chart_values: tuple[float, ...]
    received: float
    ready: float
    preprocess_ms: float


class Subscriber:
    """Keep only the newest message of each type for a slow client."""

    def __init__(self):
        self.pending: dict[str, dict] = {}
        self.changed = asyncio.Event()

    def put(self, message: dict) -> None:
        if message["type"] == "status" and message["status"] != "ready":
            self.pending.pop("chart", None)
            self.pending.pop("pose", None)
        self.pending[message["type"]] = message
        self.changed.set()

    async def get(self) -> dict:
        await self.changed.wait()
        message = min(self.pending.values(), key=lambda item: item["sequence"])
        del self.pending[message["type"]]
        if not self.pending:
            self.changed.clear()
        return message


class Session:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.subscribers: set[Subscriber] = set()
        self.sequence = 0
        self.reset()

    def reset(self) -> None:
        self.session_id: str | None = None
        self.state = "idle"
        self.reason: str | None = None
        self.model_version: str | None = None
        self.person_present: bool | None = None
        self.frames = deque(maxlen=WINDOW_SIZE)
        self.timestamps = deque(maxlen=WINDOW_SIZE)
        self.next_window = WINDOW_SIZE
        self.segment_frames = 0
        self.generation = 0
        self.target_ms: int | None = None
        self.last_received: float | None = None
        self.last_received_ms: int | None = None
        self.last_result_received: float | None = None
        self.last_gap_ms: int | None = None
        self.last_ws_send_ms: float | None = None
        self.capture_sequence: int | None = None
        self.counts = dict(records=0, accepted=0, filtered=0, backlog=0, gaps=0,
                           dropped_windows=0, predictions=0, model_failures=0)
        self.latest: dict[str, dict] = {}
        self.pending: asyncio.Queue[Window] = asyncio.Queue(maxsize=1)

    def envelope(self, kind: str) -> dict:
        return {
            "type": kind,
            "schema_version": 1,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "source": self.settings.source,
            "timestamp_ms": self.target_ms,
            "model_version": self.model_version,
        }

    def status(self) -> dict:
        return {
            **self.envelope("status"),
            "status": self.state,
            "reason": self.reason,
            "person_present": self.person_present,
            "keypoints": [],
            "capture": {"rate": VHT_RATE, "shape": [2, 2, 56], "features": 224,
                        "sequence": self.capture_sequence},
            "window": {"size": WINDOW_SIZE, "filled": len(self.frames), "stride": STRIDE},
            "counts": self.counts.copy(),
            "last_received_ms": self.last_received_ms,
            "last_gap_ms": self.last_gap_ms,
            "last_ws_send_ms": self.last_ws_send_ms,
        }

    def emit(self, kind: str, **payload) -> None:
        self.sequence += 1
        message = self.status() if kind == "status" else {**self.envelope(kind), **payload}
        self.latest[kind] = message
        for subscriber in self.subscribers:
            subscriber.put(message)

    def set_state(self, status: str, reason: str | None = None) -> None:
        if (status, reason) == (self.state, self.reason):
            return
        self.state, self.reason = status, reason
        if status != "ready":
            self.person_present = None
            self.latest.pop("pose", None)
            self.latest.pop("chart", None)
        self.emit("status")
        LOGGER.info("CSI session=%s source=%s status=%s reason=%s counts=%s",
                    self.session_id, self.settings.source, status, reason, self.counts)

    def invalidate_window(self, status: str, reason: str) -> None:
        self.generation += 1
        self.frames.clear()
        self.timestamps.clear()
        self.segment_frames = 0
        self.next_window = WINDOW_SIZE
        if not self.pending.empty():
            self.pending.get_nowait()
            self.counts["dropped_windows"] += 1
        self.set_state(status, reason)

    def subscribe(self) -> Subscriber:
        subscriber = Subscriber()
        subscriber.put(self.latest.get("status", self.status()))
        for kind in ("chart", "pose"):
            if kind in self.latest:
                subscriber.put(self.latest[kind])
        self.subscribers.add(subscriber)
        return subscriber

    async def start(self) -> bool:
        async with self.lock:
            if self.task is not None and not self.task.done():
                return False
            self.reset()
            self.session_id = uuid.uuid4().hex
            self.set_state("starting")
            self.task = asyncio.create_task(self.run())
            return True

    async def stop(self) -> None:
        async with self.lock:
            if self.task is not None and not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            if self.session_id is not None:
                self.invalidate_window("stopped", "REQUESTED_STOP")

    async def run(self) -> None:
        live = self.settings.source == "live"
        try:
            model = await asyncio.to_thread(load_model, self.settings.checkpoint, self.settings.normalizer)
            self.model_version = model.version
        except Exception:
            LOGGER.exception("Cannot load CSI model")
            self.set_state("model_error", "MODEL_LOAD_FAILED")
            return

        worker = asyncio.create_task(self.inference_worker(model)) if live else None
        try:
            self.set_state("warming_up", "WAITING_FOR_CSI")
            if live:
                await self.wait_for_capture()
            with CaptureReader(self.settings.session_dir, live=live) as reader:
                await self.consume(reader, model)
        except CaptureError:
            LOGGER.exception("CSI capture invalid")
            self.invalidate_window("source_error", "INVALID_CAPTURE")
        except OSError:
            LOGGER.exception("CSI capture unavailable")
            self.invalidate_window("source_error", "CAPTURE_UNAVAILABLE")
        finally:
            if worker is not None:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def wait_for_capture(self) -> None:
        deadline = time.perf_counter() + self.settings.timeout
        while not all((self.settings.session_dir / name).is_file()
                      for name in ("raw.csi", "recorder.log")):
            if time.perf_counter() >= deadline:
                self.set_state("disconnected", "WAITING_FOR_CAPTURE_FILES")
            await asyncio.sleep(POLL_SECONDS)

    async def consume(self, reader: CaptureReader, model) -> None:
        clock_start = time.perf_counter()
        first_capture_ns: int | None = None
        for_batch = 0
        while True:
            if for_batch % READ_BATCH == 0:
                reader.check_files()
                await asyncio.sleep(0)
            if reader.live and time.perf_counter() - (self.last_received or clock_start) >= self.settings.timeout:
                if self.state != "disconnected":
                    self.invalidate_window("disconnected", "CSI_TIMEOUT")
            if (reader.live and self.state == "ready"
                    and time.perf_counter() - self.last_result_received >= self.settings.timeout):
                self.set_state("insufficient_data", "RESULT_TIMEOUT")
            read_started = time.perf_counter()
            record = reader.read()
            if record is None:
                if not reader.live:
                    if self.state == "ready":
                        # Keep the last result available for one output interval.
                        await asyncio.sleep(STRIDE * NOMINAL_INTERVAL_MS / (1000 * self.settings.speed))
                    self.invalidate_window("ended", "REPLAY_COMPLETE")
                    return
                await asyncio.sleep(POLL_SECONDS)
                reader.check_files()
                continue
            for_batch += 1
            self.counts["records"] += 1
            if record.backlog:
                self.counts["backlog"] += 1
                continue
            parse_ms = (time.perf_counter() - read_started) * 1000
            if not reader.live:
                if first_capture_ns is None:
                    first_capture_ns, clock_start = record.timestamp_ns, time.perf_counter()
                due = clock_start + (record.timestamp_ns - first_capture_ns) / (1e9 * self.settings.speed)
                await asyncio.sleep(max(0, due - time.perf_counter()))
            window = self.ingest(record, parse_ms)
            if window is None:
                continue
            if not reader.live:
                await self.predict(model, window)
                continue
            if self.pending.full():
                self.pending.get_nowait()
                self.counts["dropped_windows"] += 1
            self.pending.put_nowait(window)

    def ingest(self, record: Record, parse_ms: float) -> Window | None:
        received = time.perf_counter()
        if not record.compatible:
            self.counts["filtered"] += 1
            if not record.rate.startswith("NOHT"):
                self.invalidate_window("incompatible", "CSI_CONFIGURATION")
            return None

        ts = record.timestamp_ns // 1_000_000
        previous = self.timestamps[-1] if self.timestamps else None
        self.target_ms = ts
        self.capture_sequence = record.sequence
        self.last_received = received
        self.last_received_ms = time.time_ns() // 1_000_000
        self.counts["accepted"] += 1
        if previous is not None and not 0 < ts - previous <= MAX_GAP_MS:
            self.last_gap_ms = ts - previous
            self.counts["gaps"] += 1
            reason = "CSI_GAP" if ts > previous else "CSI_TIMESTAMP_ORDER"
            self.invalidate_window("insufficient_data", reason)
        if self.state in {"disconnected", "incompatible"}:
            self.set_state("warming_up", "WAITING_FOR_WINDOW")

        self.timestamps.append(ts)
        self.frames.append(record.amplitude())
        self.segment_frames += 1
        if self.segment_frames < self.next_window:
            return None
        self.next_window += STRIDE
        timestamps = tuple(self.timestamps)
        # 20 frames at about 101 ms per interval. Reject sustained loss or a
        # different packet cadence even if every individual gap is <= 300 ms.
        if not (80 <= np.median(np.diff(timestamps)) <= 120 and 1500 <= ts - timestamps[0] <= 2300):
            self.set_state("insufficient_data", "CSI_CADENCE")
            self.generation += 1
            if not self.pending.empty():
                self.pending.get_nowait()
                self.counts["dropped_windows"] += 1
            return None
        # build_pose_windows stores float16. The model loads those values as float32.
        frames = np.stack(self.frames)
        amplitudes = frames.astype(np.float16).astype(np.float32)
        chart_values = tuple(round(float(value), 4) for value in frames.mean(axis=1))
        ready = time.perf_counter()
        return Window(self.generation, timestamps, amplitudes, chart_values, received, ready,
                      parse_ms + (ready - received) * 1000)

    async def inference_worker(self, model) -> None:
        while True:
            await self.predict(model, await self.pending.get())

    async def predict(self, model, window: Window) -> None:
        started = time.perf_counter()
        live = self.settings.source == "live"
        if live and started - window.received >= self.settings.timeout:
            self.counts["dropped_windows"] += 1
            return
        target = window.timestamps[-1]
        try:
            result = await asyncio.to_thread(model.predict_window, window.amplitudes, timestamp=target)
            if result["status"] != "success":
                raise ValueError(result.get("error_code", "Inference failed"))
            probability = float(result["presence_probability"])
            if not 0 <= probability <= 1:
                raise ValueError("Invalid model presence probability")
            present = bool(result["person_present"])
            coordinate_system = result["coordinate_system"]
            confidence_method = result["confidence_method"]
            # Upstream repeats presence probability on each joint. Send it
            # once at frame level so it cannot be read as joint confidence.
            keypoints = [
                {key: value for key, value in joint.items() if key != "confidence"}
                for joint in result["keypoints"]
            ] if present else []
            if present and (len(keypoints) != 33 or any(
                not (0 <= joint[axis] <= 1) for joint in keypoints for axis in ("x_norm", "y_norm")
            )):
                raise ValueError("Invalid model keypoints")
        except Exception:
            if window.generation == self.generation:
                self.counts["model_failures"] += 1
                if self.state != "model_error":
                    LOGGER.exception("CSI model inference failed")
                self.set_state("model_error", "MODEL_INFERENCE_FAILED")
            return

        completed = time.perf_counter()
        if window.generation != self.generation or (live and completed - window.received >= self.settings.timeout):
            self.counts["dropped_windows"] += 1
            return
        self.person_present = present
        self.last_result_received = window.received
        self.counts["predictions"] += 1
        self.set_state("ready")
        self.emit("chart", timestamp_ms=target, quantity="mean_raw_csi_amplitude",
                  points=[{"timestamp_ms": ts, "value": value}
                          for ts, value in zip(window.timestamps, window.chart_values)])
        self.emit(
            "pose", timestamp_ms=target, status="ready", person_present=self.person_present,
            presence_probability=probability, confidence_method=confidence_method,
            coordinate_system=coordinate_system, keypoints=keypoints,
            timing_ms={
                "preprocess": round(window.preprocess_ms, 3),
                "queue": round((started - window.ready) * 1000, 3),
                "inference": round((completed - started) * 1000, 3),
                "backend": round((completed - window.ready) * 1000 + window.preprocess_ms, 3),
            },
        )
