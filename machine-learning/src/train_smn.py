"""Fit the SMN pose checkpoint and the CSI presence classifier."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from dataset import CSINormalizer
from losses import MaskedPoseLoss
from models import CSIPoseNet, save_checkpoint
from presence import presence_features

MODEL_VERSION = "smn-csiposenet-1.3"
STD_FLOOR = 12
INPUT_CLIP = 6
EPOCHS = 30
MODEL_ARGS = {
    "input_features": 224, "conv_channels": [48, 64], "gru_hidden_dim": 64,
    "gru_num_layers": 1, "bidirectional": True, "dropout": .25, "num_keypoints": 33,
    "use_temporal_diff": True, "bound_pose": True, "pose_tail_steps": 4,
}
ARM_JOINTS = (13, 14, 15, 16)
ARM_SHOULDERS = (11, 12, 11, 12)


def load_windows(path: Path, *, with_pose: bool) -> dict:
    fields = ["amplitudes", "activity"]
    if with_pose:
        fields.extend(("poses", "mask", "valid", "frame_size"))
    with np.load(path, allow_pickle=False) as windows:
        data = {name: windows[name] for name in fields}
    frames, activities = data["amplitudes"], data["activity"]
    if (frames.ndim != 3 or frames.shape[1:] != (20, 224) or activities.shape != (len(frames),)
            or not np.isfinite(frames).all() or np.any(frames < 0)):
        raise ValueError(f"Expected 20 × 224 amplitude windows and activity labels in {path}")
    if with_pose:
        size = data["frame_size"]
        if size.shape != (2,) or np.any(size <= 0) or not np.isfinite(size).all():
            raise ValueError(f"Invalid reference image size in {path}")
        data["mask"] &= (data["valid"][:, None] & (activities != "empty")[:, None]
                         & np.isfinite(data["poses"]).all(axis=-1))
    return data


def fit_presence(groups: list[dict]) -> dict:
    features = np.concatenate([presence_features(group["amplitudes"]) for group in groups])
    occupied = np.concatenate([group["activity"] != "empty" for group in groups])
    weights = np.concatenate([np.full(len(group["activity"]), len(occupied) / (len(groups) * len(group["activity"])))
                              for group in groups])
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=.1, class_weight="balanced").fit(
        scaler.transform(features), occupied, sample_weight=weights,
    )
    return {"center": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}


def arm_pose_loss(predicted, expected, visible):
    """Supervise elbow and wrist offsets and the lift of each wrist independently."""
    visible_pairs = visible[:, ARM_JOINTS] & visible[:, ARM_SHOULDERS]
    actual_offsets = expected[:, ARM_JOINTS] - expected[:, ARM_SHOULDERS]
    predicted_offsets = predicted[:, ARM_JOINTS] - predicted[:, ARM_SHOULDERS]
    actual = actual_offsets[visible_pairs]
    relative_loss = F.smooth_l1_loss(predicted_offsets[visible_pairs], actual, beta=.05, reduction="sum")
    relative_loss = relative_loss / max(actual.numel(), 1)

    # Y increases downwards. Exclude wrists nearly level with their shoulders.
    actual_lift = -actual_offsets[:, 2:, 1]
    predicted_lift = -predicted_offsets[:, 2:, 1]
    labeled = visible_pairs[:, 2:] & (actual_lift.abs() >= .015)
    raised = actual_lift[labeled] > 0
    weights = torch.where(raised, 3., 1.)  # Raised wrists are rarer in these recordings.
    lift_loss = F.binary_cross_entropy_with_logits(24 * predicted_lift[labeled], raised.float(),
                                                   weight=weights, reduction="sum")
    return 2 * relative_loss + .35 * lift_loss / weights.sum().clamp(min=1)


def fit_pose(rounds: list[dict]):
    data = {key: np.concatenate([group[key] for group in rounds])
            for key in ("amplitudes", "poses", "mask", "activity")}
    normalizer = CSINormalizer().fit(data["amplitudes"])
    reference_tones = np.flatnonzero(normalizer.std < 2).tolist()
    normalizer.std = np.maximum(normalizer.std, STD_FLOOR)
    normalized = normalizer.transform(data["amplitudes"])
    max_distance = float(np.abs(normalized.mean(axis=1)).mean(axis=1).max()) * 1.25
    inputs = torch.from_numpy(normalized)
    joints = torch.from_numpy(data["poses"])
    joints_visible = torch.from_numpy(data["mask"])
    occupied = torch.from_numpy((data["activity"] != "empty").astype(np.float32))

    torch.manual_seed(42)
    net = CSIPoseNet(**MODEL_ARGS)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    loss_function = MaskedPoseLoss()
    subbands = torch.arange(224) // 14
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for batch in torch.randperm(len(inputs)).split(32):
            if len(batch) < 2:
                continue
            frames = inputs[batch]
            noise = .08 * torch.randn_like(frames)
            offsets = .55 * torch.randn(len(batch), 1, 16)[:, :, subbands]
            gain = (1 + .12 * torch.randn(len(batch), 1, 1)).clamp(.6, 1.4)
            prediction, coordinates = net((gain * (frames + noise) + offsets).clamp(-INPUT_CLIP, INPUT_CLIP))
            mask = joints_visible[batch]
            expected = joints[batch]
            loss, _, _ = loss_function(prediction, coordinates, occupied[batch], expected, mask, mask)
            loss = loss + arm_pose_loss(coordinates, expected, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Pose epoch {epoch}/{EPOCHS}", flush=True)
    return net, normalizer, reference_tones, max_distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-round", type=Path, action="append", required=True,
                        help="NPZ with synchronized video joints for an entire collection round")
    parser.add_argument("--presence-round", type=Path, action="append", nargs="+", default=[],
                        help="NPZ files from one additional round of labeled CSI captures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rounds = [load_windows(path, with_pose=True) for path in args.pose_round]
    width, height = (int(dimension) for dimension in rounds[0]["frame_size"])
    if any(not np.array_equal(group["frame_size"], (width, height)) for group in rounds):
        raise ValueError("Pose rounds must use the same reference image dimensions")
    extra_rounds = []
    for paths in args.presence_round:
        captures = [load_windows(path, with_pose=False) for path in paths]
        extra_rounds.append({key: np.concatenate([capture[key] for capture in captures])
                             for key in ("amplitudes", "activity")})
    classifier = fit_presence(rounds + extra_rounds)
    net, normalizer, reference_tones, max_distance = fit_pose(rounds)
    config = {
        "version": MODEL_VERSION,
        "model": {"name": "CSIPoseNet", **MODEL_ARGS},
        "data": {"window_size": 20, "num_subcarriers": 224, "num_keypoints": 33,
                 "frame_width": width, "frame_height": height, "normalized_input_clip": INPUT_CLIP,
                 "pose_reference_tones": reference_tones, "pose_max_distance": max_distance},
        "presence_classifier": classifier,
    }
    save_checkpoint(net, args.output / "smn_pose.pt", config=config, epoch=EPOCHS)
    normalizer.save(args.output / "smn_normalization.json")


if __name__ == "__main__":
    main()
