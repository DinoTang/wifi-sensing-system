"""Join the Intel recorder's append-only binary and metadata streams."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "signal-processing" / "src"))
from parse_intel_csi import GROUPS, META_RE, decode_points  # noqa: E402

HEADER_SIZE = 276
TIMESTAMP_OFFSET = 212
DATA_PREFIX_SIZE = 4
VHT_RATE = "VHT,20,VHT20"
VHT = GROUPS["vht"]


class CaptureError(ValueError):
    """The two inputs cannot be safely paired."""


@dataclass(frozen=True)
class Record:
    timestamp_ns: int
    sequence: int
    rate: str
    shape: tuple[int, int, int]
    payload: bytes
    backlog: bool

    @property
    def compatible(self) -> bool:
        return (
            self.rate == VHT_RATE
            and self.shape == (VHT["nrx"], VHT["ntx"], VHT["ntone"])
            and len(self.payload) == VHT["csi_len"]
        )

    def amplitude(self) -> np.ndarray:
        iq = np.asarray(decode_points(self.payload), dtype=np.float64)
        # Match parse_intel_csi's four-decimal CSV before the training NPZ cast.
        return np.round(np.hypot(iq[:, 0], iq[:, 1]), 4).astype(np.float32)


def read_metadata(handle: BinaryIO) -> tuple[int, dict[str, str]] | None:
    while True:
        start = handle.tell()
        line = handle.readline(4097)
        if len(line) > 4096:
            raise CaptureError("recorder.log contains an oversized line")
        if not line.endswith(b"\n"):
            handle.seek(start)
            return None
        if not line.lstrip().startswith(b"csi_len("):
            continue
        lines = [line, *(handle.readline(4097) for _ in range(3))]
        if any(len(item) > 4096 for item in lines):
            raise CaptureError("recorder.log contains an oversized metadata line")
        if any(not item.endswith(b"\n") for item in lines):
            handle.seek(start)
            return None
        match = META_RE.match(b"".join(lines).decode("utf-8", errors="replace").lstrip())
        if match is None:
            raise CaptureError(f"Malformed metadata in recorder.log at byte {start}")
        return start, match.groupdict()


class CaptureReader:
    def __init__(self, folder: Path, *, live: bool, skip_existing: bool = True):
        self.raw_path = folder / "raw.csi"
        self.log_path = folder / "recorder.log"
        self.live = live
        self.skip_existing = skip_existing
        self.files = ExitStack()

    def __enter__(self) -> CaptureReader:
        try:
            self.raw = self.files.enter_context(self.raw_path.open("rb"))
            self.log = self.files.enter_context(self.log_path.open("rb"))
            stats = tuple(os.fstat(handle.fileno()) for handle in (self.raw, self.log))
            self.identity = tuple((stat.st_dev, stat.st_ino) for stat in stats)
            self.initial_sizes = tuple(stat.st_size if self.live and self.skip_existing else 0 for stat in stats)
            self.observed_sizes = tuple(stat.st_size for stat in stats)
        except BaseException:
            self.files.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.files.close()

    def check_files(self) -> None:
        stats = tuple(path.stat() for path in (self.raw_path, self.log_path))
        sizes = tuple(stat.st_size for stat in stats)
        if (
            tuple((stat.st_dev, stat.st_ino) for stat in stats) != self.identity
            or any(size < previous for size, previous in zip(sizes, self.observed_sizes))
        ):
            raise CaptureError("Recorder replaced or truncated the capture files")
        self.observed_sizes = sizes

    def read(self) -> Record | None:
        raw_start = self.raw.tell()
        prefix = self.raw.read(4)
        if len(prefix) < 4:
            self.raw.seek(raw_start)
            if not self.live and (prefix or read_metadata(self.log) is not None):
                raise CaptureError("Incomplete raw.csi record or extra recorder metadata")
            if not self.live and self.log.read(1):
                raise CaptureError("Incomplete line at the end of recorder.log")
            return None

        (length,) = struct.unpack(">I", prefix)
        if not HEADER_SIZE + DATA_PREFIX_SIZE < length <= 65536:
            raise CaptureError(f"Invalid raw.csi record length at byte {raw_start}")
        body = self.raw.read(length)
        if len(body) != length:
            self.raw.seek(raw_start)
            if not self.live:
                raise CaptureError(f"Incomplete raw.csi record at byte {raw_start}")
            return None
        metadata = read_metadata(self.log)
        if metadata is None:
            self.raw.seek(raw_start)
            if not self.live:
                raise CaptureError("raw.csi has more records than recorder.log")
            return None

        log_start, fields = metadata
        payload = body[HEADER_SIZE + DATA_PREFIX_SIZE:]
        (timestamp_ns,) = struct.unpack_from("<Q", body, TIMESTAMP_OFFSET)
        if len(payload) != int(fields["csi_len"]) or timestamp_ns != int(fields["ts"]):
            raise CaptureError(f"raw.csi and recorder.log disagree at byte {raw_start}")
        return Record(
            timestamp_ns=timestamp_ns,
            sequence=int(fields["seq"]),
            rate=fields["rate"].strip(),
            shape=tuple(int(fields[key]) for key in ("nrx", "ntx", "ntone")),
            payload=payload,
            backlog=raw_start < self.initial_sizes[0] or log_start < self.initial_sizes[1],
        )
