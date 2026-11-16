"""Score the deployed SMN predictor against synchronized video joints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .predict import CSIPosePredictor
else:
    from predict import CSIPosePredictor

ARTIFACTS = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
ACTIVITIES = ("empty", "standing", "left_arm", "right_arm", "walking", "mixed")
SHOULDERS = (11, 12)
WRISTS = (15, 16)
HIPS = (23, 24)
RAISE_MARGIN = .015  # Fraction of the reference image height.


def wrist_lift(poses):
    """Positive y offsets mean the wrist is above the shoulder in the image."""
    return poses[:, SHOULDERS, 1] - poses[:, WRISTS, 1]


def rate(correct, selected):
    total = int(selected.sum())
    hits = int((correct & selected).sum())
    return {"correct": hits, "total": total, "fraction": hits / total if total else None}


def score_activity(activity, expected, visible, predicted, present, frame_size):
    # A suppressed or missing skeleton counts as a miss for every visible joint.
    drawn = np.isfinite(predicted).all(axis=(1, 2))
    result = {"windows": len(expected), "present": int(present.sum()), "drawn": int(drawn.sum())}
    if activity == "empty":
        result["false_positives"] = result["present"]
        return result

    result["false_negatives"] = result["windows"] - result["present"]
    distance = np.linalg.norm((predicted - expected) * frame_size, axis=-1)
    result["pck_05"] = rate(distance <= .05 * np.linalg.norm(frame_size), visible)
    for side, wrist in zip(("left", "right"), WRISTS):
        scored = visible[:, wrist] & drawn
        result[f"{side}_wrist_error_px"] = float(distance[scored, wrist].mean()) if scored.any() else None

    lift = wrist_lift(expected)
    predicted_raised = wrist_lift(predicted) > 0
    raised = lift > 0
    labeled = visible[:, SHOULDERS] & visible[:, WRISTS] & (np.abs(lift) >= RAISE_MARGIN)
    correct = (predicted_raised == raised) & drawn[:, None]
    for i, side in enumerate(("left", "right")):
        up = rate(correct[:, i], labeled[:, i] & raised[:, i])
        down = rate(correct[:, i], labeled[:, i] & ~raised[:, i])
        result[f"{side}_arm"] = {"raised": up, "lowered": down}
    result["arm_pair"] = rate(correct.all(axis=1), labeled.all(axis=1))
    result["both_raised"] = rate(correct.all(axis=1), labeled.all(axis=1) & raised.all(axis=1))

    hip_frames = visible[:, HIPS].all(axis=1) & drawn
    actual_x = expected[hip_frames][:, HIPS, 0].mean(axis=1) * frame_size[0]
    estimate_x = predicted[hip_frames][:, HIPS, 0].mean(axis=1) * frame_size[0]
    if len(estimate_x) >= 2:
        result["hip_x_span_px"] = float(np.diff(np.percentile(estimate_x, (5, 95)))[0])
        if np.std(actual_x) > 0 and np.std(estimate_x) > 0:
            result["hip_x_correlation"] = float(np.corrcoef(actual_x, estimate_x)[0, 1])
    return result


def evaluate_round(dataset, model):
    expected = dataset["poses"]
    valid = dataset["valid"].astype(bool)
    if "sync_ok" in dataset:
        valid &= np.broadcast_to(np.asarray(dataset["sync_ok"], dtype=bool), valid.shape)
    visible = dataset["mask"] & valid[:, None] & np.isfinite(expected).all(axis=-1)
    predicted = np.full_like(expected, np.nan, dtype=np.float32)
    present = np.zeros(len(expected), dtype=bool)
    for i, window in enumerate(dataset["amplitudes"]):
        prediction = model.predict_window(window)
        if prediction["status"] != "success":
            raise ValueError(f"Window {i}: {prediction['error_code']}")
        present[i] = prediction["person_present"]
        if prediction["keypoints"]:
            predicted[i] = [(joint["x_norm"], joint["y_norm"]) for joint in prediction["keypoints"]]

    activities = dataset["activity"]
    result = {}
    for activity in dict.fromkeys((*ACTIVITIES, *activities.tolist())):
        keep = activities == activity
        if keep.any():
            result[activity] = score_activity(activity, expected[keep], visible[keep], predicted[keep],
                                              present[keep], dataset["frame_size"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=Path, nargs="+", help="NPZ files with CSI and synchronized video joints")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "smn_pose.pt")
    parser.add_argument("--normalizer", type=Path, default=ARTIFACTS / "smn_normalization.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = CSIPosePredictor(args.checkpoint, args.normalizer)
    reports = []
    for path in args.round:
        with np.load(path, allow_pickle=False) as dataset:
            reports.append({"round": path.name, "activities": evaluate_round(dataset, model)})
    print(json.dumps({"model_version": model.version, "rounds": reports}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
