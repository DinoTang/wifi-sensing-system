"""Run the Intel IAX recorder and its traffic generator for one live session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import sys

LOGGER = logging.getLogger("uvicorn.error")
SUDO = ("sudo", "-n", "--")
KILL_GRACE_SECONDS = 5


@dataclass(frozen=True)
class IntelCaptureSettings:
    iax_dir: Path
    interface: str = "wlp0s20f3"
    router: str = "192.168.0.1"
    channel: int = 36
    ssid: str | None = None
    duration: int = 60


class IntelCaptureError(RuntimeError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(detail)


async def stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=KILL_GRACE_SECONDS + 1)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


async def spawn(*args, stdout) -> asyncio.subprocess.Process:
    # Keep the controlling terminal so sudo can use its terminal-scoped ticket.
    # Wait for a child started during cancellation so we can stop it as well.
    task = asyncio.create_task(asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.DEVNULL, stdout=stdout,
        stderr=asyncio.subprocess.STDOUT, env={**os.environ, "LC_ALL": "C"},
    ))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            process = await task
        except OSError:
            pass
        else:
            await stop_process(process)
        raise


class IntelCapture:
    def __init__(self, settings: IntelCaptureSettings, directory: Path):
        self.settings = settings
        self.directory = directory
        self.recorder: asyncio.subprocess.Process | None = None
        self.ping: asyncio.subprocess.Process | None = None

    async def check(self, log, reason: str, *args, allowed=(0,)) -> tuple[int, str]:
        process = await spawn(*args, stdout=asyncio.subprocess.PIPE)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=12)
        except asyncio.TimeoutError as error:
            raise IntelCaptureError(reason, f"{args[0]} timed out") from error
        finally:
            await stop_process(process)
        message = output.decode(errors="replace").strip()
        log.write(f"{args[0]} exit={process.returncode}\n{message}\n")
        log.flush()
        if process.returncode not in allowed:
            if args[0] == "sudo" and "sudo:" in message:
                reason = "CAPTURE_SUDO"
            raise IntelCaptureError(reason, f"{args[0]} exit={process.returncode}; see setup.log")
        return process.returncode, message

    async def __aenter__(self) -> IntelCapture:
        config = self.settings
        if sys.platform != "linux":
            raise IntelCaptureError("CAPTURE_PLATFORM", "IAX capture requires Linux")
        recorder = config.iax_dir / "csi" / "iaxcsi" / "cpp" / "iaxcsi"
        activate = config.iax_dir / "tools" / "iaxcsi-activate.sh"
        missing = [name for name in ("iw", "pgrep", "sudo", "ping", "timeout", "stdbuf", "bash", "cat")
                   if shutil.which(name) is None]
        if missing:
            raise IntelCaptureError("CAPTURE_TOOLS", f"Missing commands: {', '.join(missing)}")
        if not recorder.is_file() or not os.access(recorder, os.X_OK) or not activate.is_file():
            raise IntelCaptureError("CAPTURE_TOOLS", f"Missing IAX executable or activation script in {config.iax_dir}")

        self.directory.mkdir(mode=0o700, parents=True)
        try:
            with (self.directory / "setup.log").open("x", encoding="utf-8") as log:
                await self.prepare(log, activate)
            # Preserve ownership when the elevated recorder opens this file.
            raw = self.directory / "raw.csi"
            raw.touch(mode=0o600, exist_ok=False)
            with (self.directory / "recorder.log").open("xb") as log:
                self.recorder = await spawn(
                    *SUDO, "timeout", "-k", f"{KILL_GRACE_SECONDS}s", "-s", "INT",
                    f"{config.duration}s", "stdbuf", "-oL", "-eL", recorder, config.interface, raw,
                    stdout=log,
                )
            with (self.directory / "ping.log").open("xb") as log:
                self.ping = await spawn(
                    "ping", "-n", "-I", config.interface, "-s", "56", "-i", "0.1",
                    "-w", str(config.duration + KILL_GRACE_SECONDS + 1), config.router, stdout=log,
                )
        except BaseException:
            await self.close()
            raise
        return self

    async def prepare(self, log, activate: Path) -> None:
        config = self.settings
        _, link = await self.check(log, "CAPTURE_WIFI", "iw", "dev", config.interface, "link")
        ssid = re.search(r"^\s*SSID: (.+)$", link, re.MULTILINE)
        if ssid is None or (config.ssid is not None and ssid[1] != config.ssid):
            raise IntelCaptureError("CAPTURE_WIFI", "Connect to the configured Wi-Fi network")
        _, info = await self.check(log, "CAPTURE_WIFI", "iw", "dev", config.interface, "info")
        channel = re.search(r"^\s*channel (\d+) .*\bwidth: (\d+) MHz\b", info, re.MULTILINE)
        if channel is None or (int(channel[1]), int(channel[2])) != (config.channel, 20):
            raise IntelCaptureError("CAPTURE_WIFI_CHANNEL", f"Expected channel {config.channel}, width 20 MHz")
        running, _ = await self.check(log, "CAPTURE_RECORDER_BUSY", "pgrep", "-x", "iaxcsi", allowed=(0, 1))
        if running == 0:
            raise IntelCaptureError("CAPTURE_RECORDER_BUSY", "Another iaxcsi recorder is already running")
        await self.check(log, "CAPTURE_ROUTER", "ping", "-n", "-I", config.interface, "-c", "2", "-W", "2", config.router)
        await self.check(log, "CAPTURE_ACTIVATION", *SUDO, "bash", activate)
        device = Path("/sys/class/net", config.interface, "device")
        if not device.is_symlink():
            raise IntelCaptureError("CAPTURE_ACTIVATION", f"No PCI device found for {config.interface}")
        enabled = Path("/sys/kernel/debug/iwlwifi", device.resolve().name, "iwlmvm", "csi_enabled")
        _, value = await self.check(log, "CAPTURE_ACTIVATION", *SUDO, "cat", enabled)
        if value != "1":
            raise IntelCaptureError("CAPTURE_ACTIVATION", "The Intel driver did not enable CSI")

    def finished(self) -> bool:
        code = self.recorder.returncode
        if code == 124:
            return True
        if code is not None:
            reason = "CAPTURE_RECORDER_EXITED"
            with (self.directory / "recorder.log").open("rb") as log:
                if b"sudo:" in log.read(4096):
                    reason = "CAPTURE_SUDO"
            raise IntelCaptureError(reason, f"iaxcsi exited with code {code}; see recorder.log")
        if self.ping.returncode is not None:
            raise IntelCaptureError("CAPTURE_ROUTER", f"ping exited with code {self.ping.returncode}; see ping.log")
        return False

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        processes = [process for process in (self.recorder, self.ping) if process is not None]
        if processes:
            await asyncio.gather(*(stop_process(process) for process in processes))
