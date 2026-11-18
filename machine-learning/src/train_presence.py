"""Train the automatic presence classifier while retaining the pose weights."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import torch

from presence import NULL_TONES, PresenceClassifier, shape_presence_features
from train_smn import load_windows

MODEL_VERSION = "smn-csiposenet-1.5"


def split_windows(group, train_fraction=.65, purge_windows=20):
    """Split chronological windows per activity, leaving a gap at the boundary."""
    labels = group["activity"]
    train, test = np.zeros(len(labels), bool), np.zeros(len(labels), bool)
    for activity in np.unique(labels):
        ids = np.flatnonzero(labels == activity)
        end = int(train_fraction * len(ids))
        train[ids[:end]] = True
        test[ids[end + purge_windows:]] = True
    return ({key: values[train] for key, values in group.items()},
            {key: values[test] for key, values in group.items()})


def fit_presence_forest(groups):
    features = np.concatenate([shape_presence_features(g["amplitudes"]) for g in groups])
    occupied = np.concatenate([g["activity"] != "empty" for g in groups])
    if not occupied.any() or occupied.all():
        raise ValueError("Presence training requires both empty and occupied captures")
    model = ExtraTreesClassifier(n_estimators=128, max_depth=7, min_samples_leaf=4,
                                 class_weight="balanced", random_state=42, n_jobs=2)
    model.fit(features, occupied)
    trees = [estimator.tree_ for estimator in model.estimators_]
    shape = len(trees), max(tree.node_count for tree in trees)
    feature = np.full(shape, -2, dtype=np.int64)
    left, right = np.full(shape, -1, np.int64), np.full(shape, -1, np.int64)
    threshold, probability = np.zeros(shape), np.zeros(shape)
    for i, tree in enumerate(trees):
        size = tree.node_count
        feature[i, :size] = tree.feature
        left[i, :size], right[i, :size] = tree.children_left, tree.children_right
        threshold[i, :size] = tree.threshold
        weights = tree.value[:, 0]
        probability[i, :size] = weights[:, 1] / weights.sum(axis=1)
    config = {"kind": "gain_invariant_forest_v1", "max_depth": max(tree.max_depth for tree in trees),
              "feature": feature.tolist(), "left": left.tolist(), "right": right.tolist(),
              "threshold": threshold.tolist(), "occupied": probability.tolist()}
    return config, model


def evaluate_group(group, classifier, reference):
    waves, labels = group["amplitudes"], group["activity"]
    if not len(labels):
        return {"windows": 0, "activities": {}}
    expected = reference.predict_proba(shape_presence_features(waves))[:, 1]
    scores = np.array([classifier.probability(wave) for wave in waves])
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
    rng = np.random.default_rng(42)
    altered = waves.reshape(-1, 20, 4, 56) * rng.uniform(.45, 2, (len(waves), 20, 4, 1))
    altered = altered.reshape(-1, 20, 224)
    altered[:, :, NULL_TONES] = 32700
    stressed = np.array([classifier.probability(wave) for wave in altered])
    np.testing.assert_array_equal(stressed >= .5, scores >= .5)
    activities = {}
    for activity in np.unique(labels):
        selected = labels == activity
        correct = (scores >= .5) == (activity != "empty")
        activities[str(activity)] = {"correct": int(correct[selected].sum()), "windows": int(selected.sum())}
    return {"windows": len(labels), "activities": activities,
            "max_gain_and_fill_score_change": float(np.abs(scores - stressed).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True,
                        help="Chronological NPZ with amplitudes and activity; repeated per capture group")
    parser.add_argument("--holdout", type=Path, action="append", default=[], help="Entirely held-out NPZ")
    parser.add_argument("--train-fraction", type=float, default=.65)
    parser.add_argument("--purge-windows", type=int, default=20,
                        help="Discard windows at the temporal train/validation boundary")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Existing SMN pose checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1 or args.purge_windows < 0:
        parser.error("train-fraction must be in (0, 1] and purge-windows must be nonnegative")
    seen, training, validation, sources = set(), [], [], []
    for path in args.train + args.holdout:
        digest = sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            parser.error(f"Duplicate data source: {path}")
        seen.add(digest)
        data = load_windows(path, with_pose=False)
        held_out = path in args.holdout
        train, test = (None, data) if held_out else split_windows(data, args.train_fraction, args.purge_windows)
        if train is not None and len(train["activity"]):
            training.append(train)
        validation.append(test)
        sources.append({"name": path.name, "sha256": digest, "entirely_held_out": held_out,
                        "training_windows": 0 if train is None else len(train["activity"]),
                        "validation_windows": len(test["activity"])})
    if not training:
        parser.error("No training windows")
    config, reference = fit_presence_forest(training)
    classifier = PresenceClassifier.from_config(config)
    metrics = [evaluate_group(group, classifier, reference) for group in validation]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint["config"]["presence_classifier"] = config
    checkpoint["config"]["version"] = MODEL_VERSION
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {"model_version": MODEL_VERSION, "checkpoint_sha256": sha256(args.output.read_bytes()).hexdigest(),
              "train_fraction": args.train_fraction, "purge_windows": args.purge_windows,
              "training_windows": sum(len(group["activity"]) for group in training),
              "sources": [dict(source, validation=metric) for source, metric in zip(sources, metrics)]}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
