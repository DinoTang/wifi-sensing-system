"""CSI session lifecycle, causal windows and bounded event delivery."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
import logging
from pathlib import Path
import time
from typing import Literal
import uuid

import numpy as np

from .intel_capture import IntelCapture, IntelCaptureError, IntelCaptureSettings
from .model import (
    CADENCE_MEDIAN_MS, DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER, FEATURES, LANDMARK_COUNT,
    MAX_GAP_MS, NOMINAL_INTERVAL_MS, STRIDE, WINDOW_SIZE, WINDOW_SPAN_MS, load_model,
)
from .report import SessionMetrics, artifact_digests, save_session_report
from .source import CaptureError, CaptureReader, Record, VHT, VHT_RATE

LOGGER = logging.getLogger("uvicorn.error")
POLL_SECONDS = 0.025
READ_BATCH = 64
PENDING_WINDOWS = 1
PRESENCE_CONFIRM_WINDOWS = 3
PRESENCE_MAX_GAP_MS = STRIDE * MAX_GAP_MS
SIGNAL_STATES = {"warming_up", "ready", "insufficient_data", "model_error"}
AMPLITUDE_QUANTITY = "mean_raw_csi_amplitude"
# Exploratory threshold from signal-processing/docs/report_nguoi2.md.
MOTION_STD_THRESHOLD = 10.15


@dataclass(frozen=True)
class Settings:
    session_dir: Path
    source: Literal["replay", "live"]
    speed: float = 1.0
    timeout: float = 2.0
    checkpoint: Path = DEFAULT_CHECKPOINT
    normalizer: Path = DEFAULT_NORMALIZER
    autostart: bool = False
    intel_capture: IntelCaptureSettings | None = None


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

    def __init__(self, *, include_signal: bool = False):
        self.include_signal = include_signal
        self.pending: dict[str, dict] = {}
        self.changed = asyncio.Event()

    def put(self, message: dict) -> int:
        kind = message["type"]
        if kind == "signal" and not self.include_signal:
            return 0
        discarded = 0
        if kind == "status" and message["status"] != "ready":
            invalid = ("chart", "pose")
            if message["status"] not in SIGNAL_STATES and message["status"] not in ("ended", "stopped"):
                invalid += ("signal",)
            for old_kind in invalid:
                if self.pending.pop(old_kind, None) is not None:
                    discarded += 1
        discarded += kind in self.pending
        self.pending[kind] = message
        self.changed.set()
        return discarded

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
        self.capture_dir = self.settings.session_dir
        self.capture_settings = self.settings.intel_capture
        self.capture_started_ms: int | None = None
        self.state = "idle"
        self.reason: str | None = None
        self.model_version: str | None = None
        self.metrics: SessionMetrics | None = None
        self.person_present: bool | None = None
        self.presence_streak = 0
        self.presence_timestamp: int | None = None
        self.frames = deque(maxlen=WINDOW_SIZE)
        self.timestamps = deque(maxlen=WINDOW_SIZE)
        self.chart_values = deque(maxlen=WINDOW_SIZE)
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
                           timestamp_order_errors=0, incompatible_records=0, window_candidates=0,
                           rejected_cadence=0, dropped_windows=0, predictions=0,
                           present_predictions=0, empty_predictions=0, confirming_predictions=0, pose_predictions=0,
                           pose_suppressed=0, model_failures=0)
        self.latest: dict[str, dict] = {}
        self.pending: asyncio.Queue[Window] = asyncio.Queue(maxsize=PENDING_WINDOWS)

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
        defaults = self.settings.intel_capture
        duration_s = self.capture_settings.duration if self.capture_settings else None
        return {
            **self.envelope("status"),
            "status": self.state,
            "reason": self.reason,
            "person_present": self.person_present,
            "keypoints": [],
            "capture": {"rate": VHT_RATE, "shape": [VHT["nrx"], VHT["ntx"], VHT["ntone"]], "features": FEATURES,
                        "sequence": self.capture_sequence, "automatic": defaults is not None,
                        "duration_options_s": defaults.duration_options_s if defaults else [],
                        "default_duration_s": defaults.duration if defaults else None,
                        "duration_s": duration_s, "started_ms": self.capture_started_ms},
            "window": {"size": WINDOW_SIZE, "filled": len(self.frames), "stride": STRIDE},
            "counts": self.counts.copy(),
            "last_received_ms": self.last_received_ms,
            "last_gap_ms": self.last_gap_ms,
            "last_ws_send_ms": self.last_ws_send_ms,
        }

    def report(self) -> dict | None:
        if self.metrics is None:
            return None
        return {
            "report_version": 1, "session_id": self.session_id, "source": self.settings.source,
            "status": self.state, "reason": self.reason,
            "model": {"version": self.model_version, "artifact_sha256": self.metrics.artifacts.copy()},
            "configuration": {
                "rate": VHT_RATE, "payload_bytes": VHT["csi_len"],
                "window_shape": [WINDOW_SIZE, FEATURES], "stride": STRIDE,
                "maximum_gap_ms": MAX_GAP_MS, "cadence_median_ms": CADENCE_MEDIAN_MS,
                "window_span_ms": WINDOW_SPAN_MS, "landmarks": LANDMARK_COUNT,
                "presence_confirm_windows": PRESENCE_CONFIRM_WINDOWS,
                "presence_max_gap_ms": PRESENCE_MAX_GAP_MS,
                "pending_windows": PENDING_WINDOWS if self.settings.source == "live" else 0,
                "replay_speed": self.settings.speed if self.settings.source == "replay" else None,
                "live_timeout_s": self.settings.timeout if self.settings.source == "live" else None,
                "automatic_capture_s": self.capture_settings.duration if self.capture_settings else None,
                "recorder_started_ms": self.capture_started_ms,
            },
            "counts": self.counts.copy(), "last_gap_ms": self.last_gap_ms,
            **self.metrics.snapshot(self.counts["predictions"]),
        }

    def record_send(self, message: dict, duration_ms: float | None) -> None:
        if self.metrics is None or message["session_id"] != self.session_id:
            return
        if duration_ms is None:
            self.metrics.websocket["send_timeouts"] += 1
        else:
            self.last_ws_send_ms = round(duration_ms, 3)
            self.metrics.websocket["sent"] += 1
            self.metrics.timings["ws_send"].append(duration_ms)

    def emit(self, kind: str, **payload) -> None:
        self.sequence += 1
        message = self.status() if kind == "status" else {**self.envelope(kind), **payload}
        self.latest[kind] = message
        for subscriber in self.subscribers:
            discarded = subscriber.put(message)
            if self.metrics is not None:
                self.metrics.websocket["discarded"] += discarded

    def set_state(self, status: str, reason: str | None = None) -> None:
        if (status, reason) == (self.state, self.reason):
            return
        self.state, self.reason = status, reason
        if self.metrics is not None:
            self.metrics.transition(status, reason, self.target_ms)
        if status != "ready":
            self.person_present = None
            self.presence_streak = 0
            self.presence_timestamp = None
            self.latest.pop("pose", None)
            self.latest.pop("chart", None)
        self.emit("status")
        LOGGER.info("CSI session=%s source=%s status=%s reason=%s counts=%s",
                    self.session_id, self.settings.source, status, reason, self.counts)

    def drop_pending_window(self) -> None:
        if not self.pending.empty():
            self.pending.get_nowait()
            self.pending.task_done()
            self.counts["dropped_windows"] += 1

    def invalidate_window(self, status: str, reason: str) -> None:
        self.generation += 1
        self.frames.clear()
        self.timestamps.clear()
        self.chart_values.clear()
        self.latest.pop("signal", None)
        self.segment_frames = 0
        self.next_window = WINDOW_SIZE
        self.drop_pending_window()
        self.set_state(status, reason)

    def subscribe(self, *, include_signal: bool = False) -> Subscriber:
        subscriber = Subscriber(include_signal=include_signal)
        status = self.latest.get("status", self.status())
        subscriber.put(status)
        for kind in ("chart", "pose", "signal"):
            message = self.latest.get(kind)
            if message is not None and message["sequence"] > status["sequence"]:
                subscriber.put(message)
        self.subscribers.add(subscriber)
        return subscriber

    async def start(self, duration_s: int | None = None) -> bool:
        async with self.lock:
            if self.task is not None and not self.task.done():
                return False
            self.reset()
            self.session_id = uuid.uuid4().hex
            if self.capture_settings is not None:
                if duration_s is not None:
                    self.capture_settings = replace(self.capture_settings, duration=duration_s)
                self.capture_dir = self.settings.session_dir / self.session_id
            self.metrics = SessionMetrics()
            self.set_state("starting")
            self.task = asyncio.create_task(self.run())
            return True

    async def stop(self, session_id: str | None = None) -> bool:
        async with self.lock:
            if session_id is not None and session_id != self.session_id:
                return False
            if self.task is None or self.task.done():
                return True
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            if self.state != "stopped":
                self.invalidate_window("stopped", "REQUESTED_STOP")
                self.metrics.finish()
            return True

    async def run(self) -> None:
        try:
            await self.run_capture()
        except asyncio.CancelledError:
            self.invalidate_window("stopped", "REQUESTED_STOP")
            raise
        except Exception:
            LOGGER.exception("Unexpected error in CSI session")
            self.invalidate_window("source_error", "SESSION_FAILED")
        finally:
            self.metrics.finish()
            if self.capture_settings is not None and self.capture_dir.is_dir():
                try:
                    await asyncio.to_thread(save_session_report, self.capture_dir, self.report())
                except OSError:
                    LOGGER.exception("Cannot save CSI session report in %s", self.capture_dir)

    async def run_capture(self) -> None:
        live = self.settings.source == "live"
        try:
            self.metrics.artifacts = await asyncio.to_thread(
                artifact_digests, self.settings.checkpoint, self.settings.normalizer,
            )
            model = await asyncio.to_thread(load_model, self.settings.checkpoint, self.settings.normalizer)
            self.model_version = model.version
        except Exception:
            LOGGER.exception("Cannot load CSI model")
            self.set_state("model_error", "MODEL_LOAD_FAILED")
            return

        worker = None
        capture_completed = False
        try:
            async with AsyncExitStack() as stack:
                capture = None
                if self.capture_settings is not None:
                    self.set_state("starting", "PREPARING_CAPTURE")
                    capture = await stack.enter_async_context(IntelCapture(self.capture_settings, self.capture_dir))
                    self.capture_started_ms = capture.started_ms
                    LOGGER.info("Intel CSI files: %s", self.capture_dir)
                self.set_state("warming_up", "WAITING_FOR_CSI")
                if live and capture is None:
                    await self.wait_for_capture()
                if live:
                    worker = asyncio.create_task(self.inference_worker(model))
                with CaptureReader(self.capture_dir, live=live, skip_existing=capture is None) as reader:
                    await self.consume(reader, model, capture)
            if capture is not None:
                # The recorder can finish while its last complete window is
                # queued or running. Give that inference the usual live budget.
                try:
                    await asyncio.wait_for(self.pending.join(), timeout=self.settings.timeout)
                except asyncio.TimeoutError:
                    LOGGER.warning("Final CSI inference timed out for session %s", self.session_id)
                if self.counts["records"] == 0:
                    raise IntelCaptureError("CAPTURE_NO_DATA", "Recorder finished without any CSI frames")
                if self.counts["predictions"] == 0:
                    raise IntelCaptureError("CAPTURE_NO_RESULT", "Recorder finished without a valid prediction")
                capture_completed = True
        except IntelCaptureError as error:
            LOGGER.warning("Intel CSI session %s failed in %s: %s", self.session_id, self.capture_dir, error)
            self.invalidate_window("source_error", error.reason)
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
        if capture_completed:
            self.invalidate_window("ended", "CAPTURE_COMPLETE")

    async def wait_for_capture(self) -> None:
        deadline = time.perf_counter() + self.settings.timeout
        while not all((self.capture_dir / name).is_file()
                      for name in ("raw.csi", "recorder.log")):
            if time.perf_counter() >= deadline:
                self.set_state("disconnected", "WAITING_FOR_CAPTURE_FILES")
            await asyncio.sleep(POLL_SECONDS)

    async def consume(self, reader: CaptureReader, model, capture: IntelCapture | None = None) -> None:
        clock_start = time.perf_counter()
        first_capture_ns: int | None = None
        for_batch = 0
        while True:
            if for_batch % READ_BATCH == 0:
                reader.check_files()
                await asyncio.sleep(0)
            if reader.live and time.perf_counter() - (self.last_received or clock_start) >= self.settings.timeout:
                if self.state != "disconnected":
                    reason = "CAPTURE_NO_DATA" if capture is not None and self.counts["records"] == 0 else "CSI_TIMEOUT"
                    self.invalidate_window("disconnected", reason)
            if (reader.live and self.state == "ready"
                    and time.perf_counter() - self.last_result_received >= self.settings.timeout):
                self.set_state("insufficient_data", "RESULT_TIMEOUT")
            read_started = time.perf_counter()
            record = reader.read()
            if record is None:
                if capture is not None and capture.finished():
                    return
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
                await self.wait_for_replay_record(record, clock_start, first_capture_ns)
            window = self.ingest(record, parse_ms)
            if window is None:
                continue
            if not reader.live:
                await self.predict(model, window)
                continue
            if self.pending.full():
                self.drop_pending_window()
            self.pending.put_nowait(window)

    async def wait_for_replay_record(self, record: Record, clock_start: float, first_capture_ns: int) -> None:
        ns_per_second = 1e9 * self.settings.speed
        due = clock_start + (record.timestamp_ns - first_capture_ns) / ns_per_second
        if self.state == "ready" and self.timestamps:
            gap_ns = (self.timestamps[-1] + MAX_GAP_MS + 1) * 1_000_000
            if record.timestamp_ns >= gap_ns:
                gap_due = clock_start + (gap_ns - first_capture_ns) / ns_per_second
                await asyncio.sleep(max(0, gap_due - time.perf_counter()))
                # Clear cached output before waiting through a gap. Keep the
                # last sample timestamp so ingest can measure the actual gap.
                self.set_state("insufficient_data", "CSI_GAP")
        await asyncio.sleep(max(0, due - time.perf_counter()))

    def ingest(self, record: Record, parse_ms: float) -> Window | None:
        received = time.perf_counter()
        if not record.compatible:
            self.counts["filtered"] += 1
            if not record.rate.startswith("NOHT"):
                self.counts["incompatible_records"] += 1
                self.invalidate_window("incompatible", "CSI_CONFIGURATION")
            return None

        ts = record.timestamp_ns // 1_000_000
        previous = self.timestamps[-1] if self.timestamps else None
        self.target_ms = ts
        self.capture_sequence = record.sequence
        self.last_received = received
        self.last_received_ms = time.time_ns() // 1_000_000
        self.counts["accepted"] += 1
        if self.metrics is not None:
            if self.metrics.first_timestamp_ms is None:
                self.metrics.first_timestamp_ms = ts
            self.metrics.last_timestamp_ms = ts
        if previous is not None and not 0 < ts - previous <= MAX_GAP_MS:
            self.last_gap_ms = ts - previous
            self.counts["gaps"] += 1
            reason = "CSI_GAP" if ts > previous else "CSI_TIMESTAMP_ORDER"
            if reason == "CSI_TIMESTAMP_ORDER":
                self.counts["timestamp_order_errors"] += 1
            elif self.metrics is not None:
                self.metrics.max_gap_ms = max(self.metrics.max_gap_ms or 0, self.last_gap_ms)
            self.invalidate_window("insufficient_data", reason)
        if self.state in {"disconnected", "incompatible"}:
            self.set_state("warming_up", "WAITING_FOR_WINDOW")

        amplitudes = record.amplitude()
        self.timestamps.append(ts)
        self.frames.append(amplitudes)
        self.chart_values.append(round(float(amplitudes.mean()), 4))
        self.segment_frames += 1
        window = self.build_window(ts, received, parse_ms) if self.segment_frames >= self.next_window else None
        if self.settings.source == "live" and self.state in SIGNAL_STATES and (
            self.segment_frames == 1 or self.segment_frames % STRIDE == 0
        ):
            self.emit_trace("signal", self.timestamps, self.chart_values)
        return window

    def build_window(self, ts: int, received: float, parse_ms: float) -> Window | None:
        self.next_window += STRIDE
        self.counts["window_candidates"] += 1
        timestamps = tuple(self.timestamps)
        # 20 frames at about 101 ms per interval. Reject sustained loss or a
        # different packet cadence even if every individual gap is <= 300 ms.
        if not (
            CADENCE_MEDIAN_MS[0] <= np.median(np.diff(timestamps)) <= CADENCE_MEDIAN_MS[1]
            and WINDOW_SPAN_MS[0] <= ts - timestamps[0] <= WINDOW_SPAN_MS[1]
        ):
            self.counts["rejected_cadence"] += 1
            self.set_state("insufficient_data", "CSI_CADENCE")
            self.generation += 1
            self.drop_pending_window()
            return None
        # build_pose_windows stores float16. The model loads those values as float32.
        amplitudes = np.stack(self.frames).astype(np.float16).astype(np.float32)
        ready = time.perf_counter()
        return Window(self.generation, timestamps, amplitudes, tuple(self.chart_values), received, ready,
                      parse_ms + (ready - received) * 1000)

    def emit_trace(self, kind, timestamps, values) -> None:
        points = [{"timestamp_ms": ts, "value": value} for ts, value in zip(timestamps, values)]
        self.emit(kind, timestamp_ms=points[-1]["timestamp_ms"], quantity=AMPLITUDE_QUANTITY, points=points)

    async def inference_worker(self, model) -> None:
        while True:
            window = await self.pending.get()
            try:
                await self.predict(model, window)
            except asyncio.CancelledError:
                self.counts["dropped_windows"] += 1
                raise
            finally:
                self.pending.task_done()

    def confirm_presence(self, candidate: bool, target: int) -> bool:
        previous = self.presence_timestamp
        self.presence_timestamp = target
        if not candidate or previous is None or not 0 < target - previous <= PRESENCE_MAX_GAP_MS:
            self.presence_streak = 0
        if candidate:
            self.presence_streak = min(self.presence_streak + 1, PRESENCE_CONFIRM_WINDOWS)
        return self.presence_streak >= PRESENCE_CONFIRM_WINDOWS

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
            present = result["person_present"]
            if not isinstance(present, bool):
                raise ValueError("Invalid model presence result")
            coordinate_system = result["coordinate_system"]
            confidence_method = result["confidence_method"]
            pose_reason = result.get("pose_reason") if present else None
            pose_calibration = result.get("pose_calibration") if pose_reason else None
            # Upstream repeats presence probability on each joint. Send it
            # once at frame level so it cannot be read as joint confidence.
            keypoints = [
                {key: value for key, value in joint.items() if key != "confidence"}
                for joint in result["keypoints"]
            ] if present else []
            expected_joints = list(range(LANDMARK_COUNT)) if present and pose_reason is None else []
            if (pose_reason not in (None, "CSI_OUTSIDE_POSE_CALIBRATION")
                    or [joint["id"] for joint in keypoints] != expected_joints
                    or any(not 0 <= joint[axis] <= 1 for joint in keypoints for axis in ("x_norm", "y_norm"))):
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
        confirmed = self.confirm_presence(present, target)
        presence_pending = present and not confirmed
        self.person_present = None if presence_pending else confirmed
        if not confirmed:
            keypoints = []
            pose_reason = pose_calibration = None
        self.last_result_received = window.received
        self.counts["predictions"] += 1
        if confirmed:
            self.counts["present_predictions"] += 1
            self.counts["pose_predictions" if keypoints else "pose_suppressed"] += 1
        elif presence_pending:
            self.counts["confirming_predictions"] += 1
        else:
            self.counts["empty_predictions"] += 1
        timing_ms = {
            "preprocess": round(window.preprocess_ms, 3),
            "queue": round((started - window.ready) * 1000, 3),
            "inference": round((completed - started) * 1000, 3),
            "backend": round((completed - window.ready) * 1000 + window.preprocess_ms, 3),
        }
        if self.metrics is not None:
            self.metrics.record_prediction(timing_ms, completed)
        self.set_state("ready")
        self.emit_trace("chart", window.timestamps, window.chart_values)
        motion_score = float(window.amplitudes.std(axis=0).mean())
        self.emit(
            "pose", timestamp_ms=target, status="ready", person_present=confirmed, presence_pending=presence_pending,
            presence_probability=probability, confidence_method=confidence_method,
            coordinate_system=coordinate_system, keypoints=keypoints, pose_reason=pose_reason,
            pose_calibration=pose_calibration,
            motion={"score": round(motion_score, 3), "level": "high" if motion_score >= MOTION_STD_THRESHOLD else "low"},
            timing_ms=timing_ms,
        )
