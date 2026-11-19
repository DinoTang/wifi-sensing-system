"""Fit presence on complete labeled Intel captures and check entire held-out days."""
import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.session import Session, Settings
from backend.source import CaptureReader
from presence import PresenceClassifier, coherence_features, motion_and_spectrum

VERSION = "smn-csiposenet-1.6"
THRESHOLD = .65


@dataclass
class Capture:
    day: str
    split: str
    activity: str
    name: str
    raw_hash: str
    log_hash: str
    trailing_raw_bytes: int
    timestamps: set
    targets_ms: np.ndarray
    excluded_initial_windows: int
    features: np.ndarray
    changes: np.ndarray
    spectra: np.ndarray


def read_capture(row, folder):
    session = Session(Settings(folder, "replay"))
    amplitudes, packets, targets = [], set(), []
    with CaptureReader(folder, live=True, skip_existing=False) as reader:
        while (record := reader.read()) is not None:
            window = session.ingest(record, 0)
            if window is not None:
                packets.update(window.timestamps)
                amplitudes.append(window.amplitudes)
                targets.append(window.timestamps[-1])
        remaining = reader.raw_path.stat().st_size - reader.raw.tell()
    if not amplitudes:
        raise ValueError(f"No valid windows in {folder}")
    # An audited clip may begin before the subject enters the camera's field of view.
    # Keep timestamps of *all* packets for the split-overlap check.
    trim_ms = float(row.get("skip_initial_s") or 0) * 1000
    if not np.isfinite(trim_ms) or trim_ms < 0:
        raise ValueError(f"Invalid initial exclusion for {folder}")
    excluded = next((i for i, t in enumerate(targets) if t - targets[0] >= trim_ms), len(targets))
    if excluded == len(targets):
        raise ValueError(f"No labeled windows after initial exclusion in {folder}")
    amplitudes = amplitudes[excluded:]
    temporal, changes, spectra = [], [], []
    for i in range(0, len(amplitudes), 128):
        batch = np.stack(amplitudes[i:i + 128])
        temporal.append(coherence_features(batch))
        motion, spectrum = motion_and_spectrum(batch)
        changes.append(motion)
        spectra.append(spectrum)
    hashes = [sha256((folder / name).read_bytes()).hexdigest() for name in ("raw.csi", "recorder.log")]
    return Capture(row["day"], row["split"], row["activity"], row["path"], *hashes,
                   remaining, packets, np.asarray(targets[excluded:]), excluded,
                   np.concatenate(temporal), np.concatenate(changes), np.concatenate(spectra))


def read_manifest(manifest, data_root):
    recordings, seen_files, seen_times = [], set(), set()
    roots = {"repo": ROOT, "data": data_root}
    day_splits = {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in ("fit", "reference") or not row["activity"] or not row["day"]:
                raise ValueError(f"Missing group/label or invalid split: {row}")
            if day_splits.setdefault(row["day"], row["split"]) != row["split"]:
                raise ValueError(f"Collection day occurs in both sets: {row['day']}")
            capture = read_capture(row, roots[row["root"]] / row["path"])
            if capture.raw_hash in seen_files or not seen_times.isdisjoint(capture.timestamps):
                raise ValueError(f"Repeated capture or packet timestamp across recordings: {capture.name}")
            seen_files.add(capture.raw_hash)
            seen_times.update(capture.timestamps)
            recordings.append(capture)
    return recordings


def fit_logistic(captures, features):
    days = np.concatenate([np.full(len(c.features), c.day) for c in captures])
    activity = np.concatenate([np.full(len(c.features), c.activity) for c in captures])
    occupied = activity != "empty"
    weights = np.zeros(len(activity))
    # Equal weight per collection day. Within a day, give empty twice the weight of
    # all occupied activities together, even when there are many more walking clips.
    for day in np.unique(days):
        for present in (False, True):
            selected = (days == day) & (occupied == present)
            labels = np.unique(activity[selected])
            if not len(labels):
                raise ValueError(f"{day} needs empty and occupied recordings")
            for label in labels:
                which = selected & (activity == label)
                weights[which] = (1 if present else 2) / (len(labels) * which.sum())
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(features, sample_weight=weights)
    regression = LogisticRegression(C=.1, random_state=42, max_iter=1000)
    regression.fit(scaler.transform(features), occupied, sample_weight=weights)
    return scaler, regression


def fit(captures):
    scaler, reg = fit_logistic(captures, np.concatenate([c.features for c in captures]))
    coherence = {"kind": "temporal_coherence_v1", "center": scaler.mean_.tolist(),
                 "scale": scaler.scale_.tolist(), "weights": reg.coef_[0].tolist(),
                 "intercept": float(reg.intercept_[0])}
    dynamic = [c for c in captures if c.activity not in ("standing", "seated")]
    scaler, reg = fit_logistic(dynamic, np.concatenate([c.changes for c in dynamic]))
    weights = reg.coef_[0] / scaler.scale_
    quiet = {True: [], False: []}
    for capture in captures:
        if capture.activity in ("standing", "seated", "empty"):
            centers = KMeans(n_clusters=min(6, len(capture.spectra)), n_init=3,
                             random_state=42).fit(capture.spectra).cluster_centers_
            quiet[capture.activity != "empty"].extend(centers.tolist())
    return {"kind": "supported_motion_v1", "coherence": coherence, "motion_weights": weights.tolist(),
            "motion_bias": float(reg.intercept_[0] - scaler.mean_ @ weights),
            "occupied_spectra": quiet[True], "empty_spectra": quiet[False]}


def evaluate(captures, config):
    model = PresenceClassifier.from_config(config)
    results = []
    for item in captures:
        scores = model.scores(item.features, item.changes, item.spectra)
        results.append({"capture": item.name, "day": item.day, "activity": item.activity,
                        "raw_sha256": item.raw_hash, "log_sha256": item.log_hash,
                        "trailing_unpaired_raw_bytes": item.trailing_raw_bytes,
                        "excluded_initial_windows": item.excluded_initial_windows,
                        "windows": len(scores), "positive": int(np.sum(scores >= THRESHOLD)),
                        "median_score": round(float(np.median(scores)), 4),
                        "max_score": round(float(scores.max()), 4)})
    return results


def evaluate_video_round(path, captures, fold_configs):
    """Score reviewed empty clips and visible participants against the whole-day held-out model."""
    with np.load(path, allow_pickle=False) as video:
        times = (video["timestamps"] * 1000).astype(np.int64)
        activities = video["activity"]
        visible = video["valid"].astype(bool)
        synchronized = np.abs(video["frame_errors_ms"]) <= 18
        if "sync_ok" in video:
            synchronized &= video["sync_ok"]
    by_time = {int(t): i for i, t in enumerate(times)}
    if len(by_time) != len(times):
        raise ValueError(f"Duplicate target timestamp in {path}")
    results, matched, days = {}, set(), set()
    excluded = 0
    for capture in captures:
        # The recorder stores nanoseconds and the video export uses float epoch seconds.
        indices = [next((by_time[t + offset] for offset in (0, -1, 1) if t + offset in by_time), -1)
                   for t in capture.targets_ms]
        if all(i == -1 for i in indices):
            continue
        if -1 in indices or len(set(indices)) != len(indices) or not matched.isdisjoint(indices):
            raise ValueError(f"Partial or duplicate raw/video timestamp match: {capture.name}")
        matched.update(indices)
        days.add(capture.day)
        excluded += capture.excluded_initial_windows
        if not np.all(activities[indices] == capture.activity):
            raise ValueError(f"Raw/video activity mismatch: {capture.name}")
        model = PresenceClassifier.from_config(fold_configs[capture.day])
        score = model.scores(capture.features, capture.changes, capture.spectra)
        # The caller visually reviews empty footage. A pose detector can hallucinate on furniture.
        eligible = synchronized[indices] & ((capture.activity == "empty") | visible[indices])
        row = results.setdefault(capture.activity, dict(scored=0, above_threshold=0, unscored_windows=0))
        row["scored"] += int(eligible.sum())
        row["above_threshold"] += int(((score >= THRESHOLD) & eligible).sum())
        row["unscored_windows"] += int((~eligible).sum()) + capture.excluded_initial_windows
    if len(days) != 1 or len(matched) + excluded != len(times):
        raise ValueError(f"Incomplete mapping of {path} to a held-out collection day")
    return {"npz": path.name, "sha256": sha256(path.read_bytes()).hexdigest(),
            "held_out_day": days.pop(), "activities": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--synced-round", type=Path, action="append", default=[],
                        help="NPZ from the same CSI/video recording with visually reviewed empty intervals")
    args = parser.parse_args()
    captures = read_manifest(args.manifest, args.data_root)
    train = [c for c in captures if c.split == "fit"]
    reference = [c for c in captures if c.split == "reference"]
    days = sorted({c.day for c in train})
    if len(days) < 2:
        parser.error("Provide at least two fit groups for leave-day-out comparison")
    folds, fold_configs = [], {}
    for day in days:
        other_days = [c for c in train if c.day != day]
        fold_configs[day] = fit(other_days)
        folds.extend(evaluate([c for c in train if c.day == day], fold_configs[day]))
    config = fit(train)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint["config"].update(version=VERSION, presence_classifier=config, presence_threshold=THRESHOLD)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {"version": VERSION, "threshold": THRESHOLD, "fit_days": days,
              "fit_windows": sum(len(c.features) for c in train),
              "reference_only_no_video": evaluate(reference, config), "leave_one_fit_day_out": folds,
              "matched_video_leave_day_out": [evaluate_video_round(path, train, fold_configs)
                                              for path in args.synced_round],
              "sha256": sha256(args.output.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["reference_only_no_video"], indent=2))


if __name__ == "__main__":
    main()
