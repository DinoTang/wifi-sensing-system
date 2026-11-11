"""Rebuild activity features and retrain from all valid CSI sessions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIGNAL_PROCESSING = ROOT / "signal-processing"
CSI_DATA = SIGNAL_PROCESSING / "csi-data"
POSE_DATA = SIGNAL_PROCESSING / "data" / "pose"
ML_OUTPUT = ROOT / "machine-learning" / "results" / "activity-baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run activity-only CSI indexing, parsing, preprocessing, training, "
            "and prediction as one fail-fast pipeline."
        )
    )
    parser.add_argument("--csi-data", type=Path, default=CSI_DATA)
    parser.add_argument("--pose-output", type=Path, default=POSE_DATA)
    parser.add_argument("--model-output", type=Path, default=ML_OUTPUT)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=224,
        help="Expected number of VHT samples per CSI frame.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str], log_handle) -> None:
    command_text = f"\n[{name}] {' '.join(command)}\n"
    sys.stdout.write(
        command_text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        .decode(sys.stdout.encoding or "utf-8")
    )
    sys.stdout.flush()
    log_handle.write(command_text)
    log_handle.flush()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(
        output.encode(sys.stdout.encoding or "utf-8", errors="replace")
        .decode(sys.stdout.encoding or "utf-8")
    )
    sys.stdout.flush()
    log_handle.write(output)
    log_handle.flush()
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )


def run_pipeline(
    csi_data: Path,
    pose_output: Path,
    model_output: Path,
    expected_samples: int,
) -> Path:
    if expected_samples <= 0:
        raise ValueError("--expected-samples must be positive.")
    if csi_data.resolve().parent != SIGNAL_PROCESSING.resolve():
        raise ValueError(
            "--csi-data must be a direct child of signal-processing so the "
            "generated index paths resolve correctly."
        )
    if not (csi_data / "labeled").is_dir():
        raise FileNotFoundError(f"CSI session directory not found: {csi_data / 'labeled'}")

    model_output.mkdir(parents=True, exist_ok=True)
    log_path = model_output / (
        f"pipeline_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    index_path = pose_output / "index.csv"
    raw_csv_dir = pose_output / "raw_csv"
    processed_dir = pose_output / "data" / "processed"

    steps = [
        (
            "Index CSI sessions (activity-only)",
            [
                sys.executable,
                str(SIGNAL_PROCESSING / "src" / "build_pose_index.py"),
                "--csi-data",
                str(csi_data),
                "--output",
                str(index_path),
                "--activity-only",
            ],
        ),
        (
            "Parse CSI sessions",
            [
                sys.executable,
                str(SIGNAL_PROCESSING / "src" / "parse_pose_sessions.py"),
                "--index",
                str(index_path),
                "--output-dir",
                str(pose_output),
            ],
        ),
        (
            "Preprocess and create features",
            [
                sys.executable,
                str(SIGNAL_PROCESSING / "src" / "csi_preprocessing_pipeline.py"),
                "--input-dir",
                str(raw_csv_dir),
                "--output-dir",
                str(pose_output),
                "--expected-samples",
                str(expected_samples),
            ],
        ),
        (
            "Train and evaluate activity model",
            [
                sys.executable,
                str(ROOT / "machine-learning" / "src" / "train_activity_baseline.py"),
                "--processed-dir",
                str(processed_dir),
                "--output-dir",
                str(model_output),
                "--evaluation-mode",
                "auto",
            ],
        ),
    ]

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                f"CSI activity pipeline started {datetime.now().isoformat()}\n"
            )
            for name, command in steps:
                run_step(name, command, log_handle)
            log_handle.write(
                f"\nPipeline completed {datetime.now().isoformat()}\n"
            )
    except subprocess.CalledProcessError as error:
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\nFAILED exit={error.returncode} command={error.cmd}\n"
            )
        print(
            f"\nPipeline stopped at the failed step (exit {error.returncode}). "
            f"Log: {log_path}",
            file=sys.stderr,
        )
        raise

    print(f"\nPipeline completed successfully. Log: {log_path}")
    return log_path


def main() -> None:
    args = parse_args()
    run_pipeline(
        csi_data=args.csi_data,
        pose_output=args.pose_output,
        model_output=args.model_output,
        expected_samples=args.expected_samples,
    )


if __name__ == "__main__":
    main()
