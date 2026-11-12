"""Regression tests for the recorder boundary and frontend event contract."""

import asyncio
from dataclasses import replace
from hashlib import sha256
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app import create_app
from backend import intel_capture
from backend.intel_capture import IntelCaptureSettings
from backend.model import DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER, load_model
from backend.report import SessionMetrics, TIMING_SAMPLE_LIMIT
from backend.session import MOTION_STD_THRESHOLD, Session, Settings, Subscriber
from backend.source import CaptureError, CaptureReader, ROOT
from parse_intel_csi import convert_session

CAPTURES = ROOT / "hardware" / "csi_extracted" / "labeled"
EMPTY = CAPTURES / "empty_01_20261025-052422_4eki"
WALKING = CAPTURES / "walking_01_20261025-053626_PiI7"


def receive_until(socket, condition, limit=100):
    for _ in range(limit):
        message = socket.receive_json()
        if condition(message):
            return message
        if message.get("status") in {"ended", "stopped", "source_error", "model_error"}:
            pytest.fail(f"Session ended before the expected event: {message}")
    pytest.fail("Expected WebSocket event was not received")


def wait_for_status(client, state):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if status["status"] == state:
            return status
        time.sleep(0.02)
    pytest.fail(f"Did not reach {state}: {status}")


@pytest.fixture(scope="module")
def model():
    return load_model(DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER)


@pytest.mark.parametrize("capture", [EMPTY, WALKING], ids=["empty", "walking"])
def test_real_replay_matches_offline_parser_and_model(capture, model, monkeypatch):
    monkeypatch.setattr("backend.session.load_model", lambda *_: model)
    parsed, _ = convert_session(capture, "vht", include_test=True)
    indices = {frame["ts"] // 1_000_000: i for i, (_, frame, _) in enumerate(parsed["kept"])}

    with TestClient(create_app(Settings(capture, "replay", speed=8))) as client:
        assert "WebSocket" in client.get("/").text
        assert "Dashboard CSI" in client.get("/dashboard/").text
        assert "javascript" in client.get("/dashboard/dashboard.js").headers["content-type"]
        assert client.get("/api/session/report").status_code == 404
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["status"] == "idle"
            response = client.post("/api/session/start")
            assert response.status_code == 202
            assert client.post("/api/session/start").status_code == 409
            chart = None

            def matches_chart(item):
                nonlocal chart
                if item["type"] == "chart":
                    chart = item
                return item["type"] == "pose" and chart is not None and item["timestamp_ms"] == chart["timestamp_ms"]

            pose = receive_until(socket, matches_chart)
            assert len(chart["points"]) == 20
            assert chart["quantity"] == "mean_raw_csi_amplitude"
            assert pose["session_id"] == response.json()["session_id"]
            assert pose["source"] == "replay"
            assert pose["server_sent_ms"] > pose["timestamp_ms"]
            end = indices[pose["timestamp_ms"]] + 1
            window = np.asarray([
                [float(f"{(real * real + imag * imag) ** 0.5:.4f}") for real, imag in points]
                for _, _, points in parsed["kept"][end - 20:end]
            ], dtype=np.float16).astype(np.float32)
            offline = model.predict_window(window, timestamp=pose["timestamp_ms"])
            assert pose["person_present"] == offline["person_present"]
            assert pose["presence_probability"] == offline["presence_probability"]
            assert pose["keypoints"] == [
                {key: value for key, value in joint.items() if key != "confidence"}
                for joint in offline["keypoints"]
            ]
            motion = float(window.std(axis=0).mean())
            assert pose["motion"] == {
                "score": round(motion, 3), "level": "high" if motion >= MOTION_STD_THRESHOLD else "low",
            }
            with client.websocket_connect("/ws") as reconnected:
                receive_until(reconnected, lambda item: item["type"] == "status" and item["status"] == "ready")
                assert receive_until(reconnected, lambda item: item["type"] == "pose")["session_id"] == pose["session_id"]
            assert client.post("/api/session/stop", params={"session_id": "previous-session"}).status_code == 409
            client.post("/api/session/stop", params={"session_id": pose["session_id"]}).raise_for_status()
            stopped = receive_until(socket, lambda item: item.get("status") == "stopped")
            assert stopped["person_present"] is None and stopped["keypoints"] == []
            report = client.get("/api/session/report").json()
            assert report["session_id"] == pose["session_id"] and report["status"] == "stopped"
            assert report["finished_ms"] >= report["started_ms"]
            assert report["model"]["artifact_sha256"]["checkpoint"] == sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
            counts = report["counts"]
            assert counts["predictions"] == counts["present_predictions"] + counts["empty_predictions"]
            assert report["timing_ms"]["backend"]["samples"] == counts["predictions"]
            assert report["websocket"]["sent"] >= 3
            with client.websocket_connect("/ws") as reconnected:
                assert reconnected.receive_json()["status"] == "stopped"


def test_partial_binary_and_log_then_verify_pairing(tmp_path):
    original = (EMPTY / "raw.csi").read_bytes()
    log = (EMPTY / "recorder.log").read_bytes()
    first_end = 4 + int.from_bytes(original[:4], "big")
    log_end = log.index(b"\n", log.index(b"rate_bw(")) + 1
    raw_path, log_path = tmp_path / "raw.csi", tmp_path / "recorder.log"
    raw_path.touch()
    log_path.touch()
    with CaptureReader(tmp_path, live=True) as reader:
        with raw_path.open("ab") as raw, log_path.open("ab") as metadata:
            raw.write(original[:first_end - 1])
            raw.flush()
            metadata.write(log[:log_end - 1])
            metadata.flush()
            assert reader.read() is None
            raw.write(original[first_end - 1:first_end])
            raw.flush()
            assert reader.read() is None
            metadata.write(log[log_end - 1:log_end])
            metadata.flush()
            record = reader.read()
            assert record is not None and record.sequence == 130 and not record.backlog
            assert reader.read() is None
            raw.write(original[:first_end])
            raw.flush()
            second_end = log.index(b"\n", log.index(b"rate_bw(", log_end)) + 1
            metadata.write(log[log_end:second_end])
            metadata.flush()
            with pytest.raises(CaptureError, match="disagree"):
                reader.read()


def test_live_app_times_out_recovers_and_stops(tmp_path, monkeypatch, model):
    monkeypatch.setattr("backend.session.load_model", lambda *_: model)
    raw_path, log_path = tmp_path / "raw.csi", tmp_path / "recorder.log"
    raw_path.touch()
    log_path.touch()
    app = create_app(Settings(tmp_path, "live", timeout=0.8))
    with TestClient(app) as client:
        assert client.post("/api/session/start").status_code == 202
        wait_for_status(client, "disconnected")
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["status"] == "disconnected"
            with CaptureReader(WALKING, live=False) as reader:
                for _ in range(100):
                    reader.read()
                raw_end, log_end = reader.raw.tell(), reader.log.tell()
            with raw_path.open("ab") as raw, log_path.open("ab") as log:
                raw.write((WALKING / "raw.csi").read_bytes()[:raw_end])
                log.write((WALKING / "recorder.log").read_bytes()[:log_end])
            wait_for_status(client, "ready")
            pose = receive_until(socket, lambda item: item["type"] == "pose")
            assert pose["source"] == "live" and pose["timing_ms"]["backend"] >= 0
            disconnected = wait_for_status(client, "disconnected")
            assert disconnected["reason"] == "CSI_TIMEOUT"
            assert disconnected["person_present"] is None and disconnected["keypoints"] == []
            assert disconnected["counts"]["accepted"] > 20
            assert client.post("/api/session/stop").json()["status"] == "stopped"
            report = client.get("/api/session/report").json()
            assert report["source"] == "live" and report["status"] == "stopped"
            assert report["counts"]["dropped_windows"] > 0
            assert report["finished_ms"] is not None
            assert "CSI_TIMEOUT" in {transition["reason"] for transition in report["transitions"]}


@pytest.fixture
def automatic_capture(tmp_path, monkeypatch):
    iax = tmp_path / "iax"
    recorder = iax / "csi" / "iaxcsi" / "cpp" / "iaxcsi"
    activation = iax / "tools" / "iaxcsi-activate.sh"
    for path in (recorder, activation):
        path.parent.mkdir(parents=True)
        path.touch(mode=0o700)

    async def prepare(capture, log, script):
        assert script == activation

    monkeypatch.setattr(intel_capture, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(intel_capture, "shutil", SimpleNamespace(which=lambda name: name))
    monkeypatch.setattr(intel_capture.IntelCapture, "prepare", prepare)
    return Settings(tmp_path / "captures", "live", intel_capture=IntelCaptureSettings(iax))


def test_automatic_capture_start_stop_and_restart(automatic_capture, monkeypatch, model):
    monkeypatch.setattr("backend.session.load_model", lambda *_: model)
    spawn = intel_capture.spawn
    children = []
    # Write a real Intel fixture before the reader opens, just as a fast recorder can.
    recorder_script = """
from pathlib import Path
import sys
import time

raw, fixture = map(Path, sys.argv[1:])
raw.write_bytes((fixture / 'raw.csi').read_bytes())
sys.stdout.buffer.write((fixture / 'recorder.log').read_bytes())
sys.stdout.buffer.flush()
ready = raw.with_name('writer-ready')
ready.touch()
while not raw.with_name('writer-finished').exists():
    time.sleep(0.02)
sys.exit(124)
"""

    async def start_child(*args, stdout):
        if args[0] == "ping":
            assert args[2:] == ("-I", "wlp0s20f3", "-s", "56", "-i", "0.1", "-w", "66", "192.168.0.1")
            process = await spawn(sys.executable, "-c", "import time; time.sleep(60)", stdout=stdout)
        else:
            assert args[:13] == ("sudo", "-n", "--", "timeout", "-k", "5s", "-s", "INT",
                                 "60s", "stdbuf", "-oL", "-eL", automatic_capture.intel_capture.iax_dir /
                                 "csi" / "iaxcsi" / "cpp" / "iaxcsi")
            raw = args[-1]
            process = await spawn(sys.executable, "-c", recorder_script, raw, WALKING, stdout=stdout)
            deadline = time.monotonic() + 5
            while not raw.with_name("writer-ready").exists():
                if process.returncode is not None or time.monotonic() >= deadline:
                    await intel_capture.stop_process(process)
                    pytest.fail("Test recorder did not write its fixture")
                await asyncio.sleep(0.01)
        children.append(process)
        return process

    monkeypatch.setattr(intel_capture, "spawn", start_child)
    with TestClient(create_app(automatic_capture)) as client:
        assert not automatic_capture.session_dir.exists()
        first = client.post("/api/session/start").json()["session_id"]
        ready = wait_for_status(client, "ready")
        assert ready["capture"]["automatic"]
        assert ready["counts"]["backlog"] == 0
        assert ready["counts"]["predictions"] > 0
        assert client.post("/api/session/stop", params={"session_id": first}).json()["status"] == "stopped"
        assert all(process.returncode is not None for process in children)
        first_raw = automatic_capture.session_dir / first / "raw.csi"
        assert first_raw.stat().st_size > 0

        second = client.post("/api/session/start").json()["session_id"]
        assert second != first
        assert client.post("/api/session/stop", params={"session_id": first}).status_code == 409
        wait_for_status(client, "ready")
        (automatic_capture.session_dir / second / "writer-finished").touch()
        assert wait_for_status(client, "ended")["reason"] == "CAPTURE_COMPLETE"
        assert all(process.returncode is not None for process in children)
        assert first_raw.stat().st_size == (WALKING / "raw.csi").stat().st_size


def test_automatic_capture_reports_recorder_failure(automatic_capture, monkeypatch, model):
    monkeypatch.setattr("backend.session.load_model", lambda *_: model)
    spawn = intel_capture.spawn
    children = []

    async def start_child(*args, stdout):
        code = "import time; time.sleep(60)" if args[0] == "ping" else (
            "import sys; print('sudo: a password is required', file=sys.stderr); sys.exit(1)"
        )
        process = await spawn(sys.executable, "-c", code, stdout=stdout)
        children.append(process)
        return process

    monkeypatch.setattr(intel_capture, "spawn", start_child)
    with TestClient(create_app(automatic_capture)) as client:
        started = client.post("/api/session/start").json()["session_id"]
        assert wait_for_status(client, "source_error")["reason"] == "CAPTURE_SUDO"
        assert all(process.returncode is not None for process in children)
        assert (automatic_capture.session_dir / started / "raw.csi").stat().st_size == 0


def test_model_load_failure_is_distinct_from_empty(tmp_path):
    settings = Settings(EMPTY, "replay", checkpoint=tmp_path / "missing.pt")
    with TestClient(create_app(settings)) as client:
        with client.websocket_connect("/ws") as socket:
            client.post("/api/session/start").raise_for_status()
            message = receive_until(socket, lambda item: item.get("status") == "model_error")
            assert message["reason"] == "MODEL_LOAD_FAILED"
            assert message["person_present"] is None and message["keypoints"] == []


def test_gap_bad_configuration_and_inference_error(tmp_path, monkeypatch, model):
    with CaptureReader(EMPTY, live=False) as reader:
        records = [record for _ in range(100) if (record := reader.read()).compatible]
    session = Session(Settings(tmp_path, "replay"))
    for record in records[:8]:
        session.ingest(record, 0)
    gap = replace(records[8], timestamp_ns=records[7].timestamp_ns + 400_000_000)
    assert session.ingest(gap, 0) is None
    assert session.status()["reason"] == "CSI_GAP"
    assert session.status()["window"]["filled"] == 1
    assert session.ingest(replace(gap, rate="VHT,80,VHT80"), 0) is None
    assert session.status()["status"] == "incompatible"
    assert session.status()["window"]["filled"] == 0

    def fail(*args, **kwargs):
        raise RuntimeError("Injected model failure")

    monkeypatch.setattr(model, "predict_window", fail)
    with TestClient(create_app(Settings(EMPTY, "replay", speed=16))) as client:
        monkeypatch.setattr("backend.session.load_model", lambda *_: model)
        client.post("/api/session/start").raise_for_status()
        message = wait_for_status(client, "model_error")
        assert message["reason"] == "MODEL_INFERENCE_FAILED"
        assert message["person_present"] is None and message["keypoints"] == []


def test_slow_subscriber_keeps_latest_and_invalidates_pose():
    async def run():
        subscriber = Subscriber()
        for sequence in range(500):
            subscriber.put({"type": "pose", "sequence": sequence})
        assert (await subscriber.get())["sequence"] == 499
        subscriber.put({"type": "pose", "sequence": 500})
        subscriber.put({"type": "status", "sequence": 501, "status": "disconnected"})
        assert (await subscriber.get())["status"] == "disconnected"
        assert not subscriber.pending

    asyncio.run(run())


def test_empty_prediction_suppresses_keypoints():
    class EmptyPredictor:
        def predict_window(self, amplitudes, timestamp):
            return {
                "status": "success", "person_present": False, "presence_probability": 0.1,
                "confidence_method": "model_presence_sigmoid_probability",
                "coordinate_system": {}, "keypoints": [{"id": 0}],
            }

    async def run():
        session = Session(Settings(EMPTY, "replay"))
        with CaptureReader(EMPTY, live=False) as reader:
            window = None
            while (record := reader.read()) is not None:
                window = session.ingest(record, 0)
                if window is not None:
                    break
        assert window is not None
        await session.predict(EmptyPredictor(), window)
        assert session.latest["pose"]["person_present"] is False
        assert session.latest["pose"]["keypoints"] == []

    asyncio.run(run())


def test_replay_gap_clears_snapshot_before_next_record():
    async def run():
        session = Session(Settings(EMPTY, "replay"))
        with CaptureReader(EMPTY, live=False) as reader:
            record = reader.read()
        assert record is not None
        target = record.timestamp_ns // 1_000_000
        session.target_ms = target
        session.timestamps.append(target)
        session.set_state("ready")
        session.emit("chart")
        session.emit("pose")
        subscriber = session.subscribe()
        assert [(await subscriber.get())["type"] for _ in range(3)] == ["status", "chart", "pose"]

        late_record = replace(record, timestamp_ns=record.timestamp_ns + 2_000_000_000)
        task = asyncio.create_task(session.wait_for_replay_record(
            late_record, time.perf_counter() - .31, record.timestamp_ns,
        ))
        try:
            message = await asyncio.wait_for(subscriber.get(), timeout=.5)
            assert message["reason"] == "CSI_GAP" and not task.done()
            snapshot = session.subscribe()
            assert set(snapshot.pending) == {"status"}
            assert snapshot.pending["status"]["person_present"] is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_report_bounds_timing_samples():
    metrics = SessionMetrics()
    for value in range(TIMING_SAMPLE_LIMIT + 100):
        metrics.record_prediction({"backend": float(value)}, completed=float(value))
    snapshot = metrics.snapshot(TIMING_SAMPLE_LIMIT + 100)
    assert snapshot["timing_ms"]["backend"]["samples"] == TIMING_SAMPLE_LIMIT
    assert snapshot["timing_ms"]["backend"]["max"] == TIMING_SAMPLE_LIMIT + 99
    assert snapshot["prediction_rate_hz"] == 1


def test_stop_keeps_a_completed_replay_report(tmp_path, model, monkeypatch):
    monkeypatch.setattr("backend.session.load_model", lambda *_: model)
    (tmp_path / "raw.csi").touch()
    (tmp_path / "recorder.log").touch()
    with TestClient(create_app(Settings(tmp_path, "replay"))) as client:
        client.post("/api/session/start").raise_for_status()
        wait_for_status(client, "ended")
        before = client.get("/api/session/report").json()
        assert client.post("/api/session/stop").json()["status"] == "ended"
        assert before == client.get("/api/session/report").json()
