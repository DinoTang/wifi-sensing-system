"""Run the CSI service with a server-side capture directory."""

import argparse
from ipaddress import IPv4Address
import math
from pathlib import Path
import re
import sys

import uvicorn

from .app import create_app
from .intel_capture import IntelCaptureSettings
from .model import DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER
from .session import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["replay", "live"], required=True)
    parser.add_argument("--session", type=Path, required=True,
                        help="Capture directory, or parent directory for --auto-capture")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--timeout", type=float, default=2.0, help="Live CSI inactivity timeout in seconds")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--autostart", action="store_true")
    capture = parser.add_argument_group("Ubuntu Intel IAX capture")
    capture.add_argument("--auto-capture", action="store_true", help="Start and stop IAX with the live session")
    capture.add_argument("--iax-dir", type=Path, default=Path.home() / "iax")
    capture.add_argument("--interface", default="wlp0s20f3")
    capture.add_argument("--router", type=IPv4Address, default="192.168.0.1")
    capture.add_argument("--channel", type=int, default=36)
    capture.add_argument("--ssid", help="Require this Wi-Fi network before starting IAX")
    capture.add_argument("--duration", type=int, default=60, help="Capture length in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    intel_capture = None
    if args.auto_capture:
        if args.source != "live" or sys.platform != "linux":
            parser.error("--auto-capture requires --source live on Linux")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,14}", args.interface):
            parser.error("Invalid Wi-Fi interface name")
        if args.duration <= 0 or args.channel <= 0:
            parser.error("--duration and --channel must be positive")
        intel_capture = IntelCaptureSettings(
            args.iax_dir.expanduser().resolve(), interface=args.interface, router=str(args.router),
            channel=args.channel, ssid=args.ssid, duration=args.duration,
        )
    uvicorn.run(create_app(Settings(
        session_dir=args.session.resolve(), source=args.source,
        speed=args.speed, timeout=args.timeout, checkpoint=args.checkpoint,
        normalizer=args.normalizer, autostart=args.autostart, intel_capture=intel_capture,
    )), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
