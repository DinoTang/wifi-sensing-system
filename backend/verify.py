"""Run one replay through HTTP and WebSocket and export its session report."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .model import LANDMARK_COUNT, WINDOW_SIZE


def http_json(base_url: str, route: str, method: str = "GET") -> dict:
    with urlopen(Request(base_url + route, method=method), timeout=5) as response:
        return json.load(response)


async def observe(websocket, session_id: str) -> dict:
    received = dict(status=0, chart=0, pose=0, matched_windows=0, present=0, empty=0)
    previous_sequence = -1
    chart_timestamp = None
    while True:
        message = json.loads(await websocket.recv())
        if message["session_id"] != session_id:
            continue
        sequence = message["sequence"]
        if message["source"] != "replay" or message["schema_version"] != 1 or sequence <= previous_sequence:
            raise ValueError("Unexpected WebSocket source, schema or sequence")
        previous_sequence = sequence

        kind = message["type"]
        if kind not in {"status", "chart", "pose"}:
            raise ValueError(f"Unexpected WebSocket message type: {kind}")
        received[kind] += 1
        if kind == "status":
            if message["status"] != "ready":
                chart_timestamp = None
            if message["status"] in {"model_error", "source_error", "stopped"}:
                raise ValueError(f"Replay did not finish: {message['status']} ({message['reason']})")
            if message["status"] == "ended":
                if not received["matched_windows"]:
                    raise ValueError("Replay finished without a matched chart and pose window")
                return received
        elif kind == "chart":
            times = [point["timestamp_ms"] for point in message["points"]]
            if (
                message["quantity"] != "mean_raw_csi_amplitude"
                or len(times) != WINDOW_SIZE
                or any(a >= b for a, b in zip(times, times[1:]))
                or times[-1] != message["timestamp_ms"]
            ):
                raise ValueError("Invalid CSI chart window")
            chart_timestamp = times[-1]
        elif kind == "pose":
            present = message["person_present"]
            if not isinstance(present, bool) or len(message["keypoints"]) != (LANDMARK_COUNT if present else 0):
                raise ValueError("Invalid presence and keypoint combination")
            received["present" if present else "empty"] += 1
            if message["timestamp_ms"] == chart_timestamp:
                received["matched_windows"] += 1
                chart_timestamp = None


async def verify(base_url: str, timeout: float) -> dict:
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.path not in {"", "/"} or url.query or url.fragment:
        raise ValueError("--base-url must be the origin of a CSI backend")
    base_url = base_url.rstrip("/")
    if (await asyncio.to_thread(http_json, base_url, "/api/status"))["source"] != "replay":
        raise ValueError("Start the backend with --source replay before running verification")

    ws_scheme = "wss" if url.scheme == "https" else "ws"
    session_id = None
    completed = False
    try:
        async with connect(f"{ws_scheme}://{url.netloc}/ws", open_timeout=5) as websocket:
            await asyncio.wait_for(websocket.recv(), timeout=5)
            accepted = await asyncio.to_thread(http_json, base_url, "/api/session/start", "POST")
            session_id = accepted["session_id"]
            observed = await asyncio.wait_for(observe(websocket, session_id), timeout=timeout)
            report = await asyncio.to_thread(http_json, base_url, "/api/session/report")
            if (
                report["session_id"] != session_id
                or report["status"] != "ended"
                or report["counts"]["predictions"] < observed["pose"]
            ):
                raise ValueError("Report does not match the replay observed by the client")
            completed = True
            return {"session": report, "websocket_client": observed}
    finally:
        if session_id is not None and not completed:
            try:
                query = urlencode({"session_id": session_id})
                await asyncio.to_thread(http_json, base_url, f"/api/session/stop?{query}", "POST")
            except (OSError, ValueError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120, help="Maximum replay duration in seconds")
    parser.add_argument("--output", type=Path, help="Write a new JSON report instead of printing it")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists")
    try:
        report = asyncio.run(verify(args.base_url, args.timeout))
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output is None:
            print(payload)
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            print(f"Verified replay report: {args.output}")
    except (OSError, ValueError, WebSocketException) as error:
        parser.exit(1, f"Verification failed: {error or type(error).__name__}\n")


if __name__ == "__main__":
    main()
