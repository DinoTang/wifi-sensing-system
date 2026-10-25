import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import serial


CSV_HEADER = ["frame_id", "timestamp_ms", "rssi", "len", "first_word", "data"]


def parse_csi_line(line):
    if not line.startswith("CSI_DATA,"):
        return None

    fields = line.rstrip("\r\n").split(",", 6)
    if len(fields) != 7:
        return None

    _, frame_id, timestamp_ms, rssi, length, first_word, data_field = fields
    data_hex = data_field.strip().strip('"')
    expected_length = int(length)
    if len(data_hex) != expected_length * 2:
        return None
    int(data_hex, 16)
    return [frame_id, timestamp_ms, rssi, expected_length, first_word, data_hex]


def capture(port, baud, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"csi_capture_{timestamp}.log"
    csv_path = output_dir / f"csi_capture_{timestamp}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = False
    completed = False

    with serial.Serial(port, baudrate=baud, timeout=1) as device, raw_path.open(
        "w", encoding="utf-8", newline=""
    ) as raw_file:
        device.reset_input_buffer()
        device.write(b"r\n")
        device.flush()
        print("Sent 'r'. Capturing for 3 minutes; do not open another monitor on this port.")

        while True:
            raw_line = device.readline()
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace")
            raw_file.write(line)
            raw_file.flush()
            print(line, end="")

            if line.startswith("CAPTURE,START"):
                started = True
            elif started and line.startswith("CAPTURE,complete"):
                completed = True
                break

            if started and line.startswith("CSI_DATA,"):
                try:
                    row = parse_csi_line(line)
                except (ValueError, IndexError):
                    row = None
                if row is not None:
                    rows.append(row)

    if not completed:
        raise RuntimeError("Capture ended before CAPTURE,complete was received")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"Saved raw log: {raw_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Valid CSI rows: {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 3-minute ESP32 CSI capture and save CSV")
    parser.add_argument("--port", default="COM5", help="ESP32 serial port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures"),
        help="Folder for .log and .csv files",
    )
    arguments = parser.parse_args()
    capture(arguments.port, arguments.baud, arguments.output_dir)
