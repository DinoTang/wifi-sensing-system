"""Run the CSI service with a server-side capture directory."""

import argparse
import math
from pathlib import Path

import uvicorn

from .app import create_app
from .model import DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER
from .session import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["replay", "live"], required=True)
    parser.add_argument("--session", type=Path, required=True, help="Directory containing raw.csi and recorder.log")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--timeout", type=float, default=2.0, help="Live CSI inactivity timeout in seconds")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--autostart", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    uvicorn.run(create_app(Settings(
        session_dir=args.session.resolve(), source=args.source,
        speed=args.speed, timeout=args.timeout, checkpoint=args.checkpoint,
        normalizer=args.normalizer, autostart=args.autostart,
    )), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
