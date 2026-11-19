"""Train CSIPoseNet on synchronized 2D joints and their displacements between CSI windows."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset import CSINormalizer
from models import CSIPoseNet, save_checkpoint
from presence import NULL_TONES, normalize_pose_links
from train_smn import INPUT_CLIP, MODEL_ARGS, arm_pose_loss, load_windows

EPOCHS = 40
BODY = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def read_round(path):
    group = load_windows(path, with_pose=True)
    with np.load(path, allow_pickle=False) as data:
        times = data["timestamps"]
        errors = data["frame_errors_ms"]
        if times.shape != group["valid"].shape or errors.shape != times.shape or not np.isfinite(times).all():
            raise ValueError(f"Missing per-window synchronization in {path}")
        # Half of a 30 fps video frame plus rounding tolerance.
        synchronized = np.isfinite(errors) & (np.abs(errors) <= 18.)
        if "sync_ok" in data:
            synchronized &= data["sync_ok"]
        group["mask"] &= synchronized[:, None]
        group["timestamps"] = times
    if not group["mask"].any():
        raise ValueError(f"No synchronized visible joints in {path}")
    return group


def paired_indices(group):
    """Use an earlier labeled target in the same continuous activity, never across captures."""
    times, activity = group["timestamps"], group["activity"]
    previous = np.arange(len(times))
    seen = group["mask"].any(axis=1)
    for i in range(3, len(times)):
        prior = i - 3
        delta = np.diff(times[prior:i + 1])
        if (seen[i] and seen[prior] and np.all(activity[prior:i] == activity[i])
                and np.all((delta > 0) & (delta < .35)) and .35 < delta.sum() < .9):
            previous[i] = prior
    return previous


def train_pose(rounds):
    torch.manual_seed(42)
    frames = normalize_pose_links(np.concatenate([g["amplitudes"] for g in rounds]))
    normalizer = CSINormalizer().fit(frames)
    normalizer.std = np.maximum(normalizer.std, 5.)
    normalized = normalizer.transform(frames)
    max_distance = 1.25 * float(np.abs(normalized.mean(axis=1)).mean(axis=1).max())
    inputs = torch.from_numpy(normalized.clip(-INPUT_CLIP, INPUT_CLIP))
    expected = torch.from_numpy(np.nan_to_num(np.concatenate([g["poses"] for g in rounds])).astype(np.float32))
    visible = torch.from_numpy(np.concatenate([g["mask"] for g in rounds]))
    previous, start = [], 0
    for group in rounds:
        previous.append(paired_indices(group) + start)
        start += len(group["timestamps"])
    previous = torch.from_numpy(np.concatenate(previous))
    indices = torch.where(visible.any(dim=1))[0]
    # Both wrists above the shoulders are rare; revisit those camera-labeled examples in training.
    raised = expected[:, (11, 12), 1] - expected[:, (15, 16), 1] > .015
    both_raised = raised.all(dim=1) & visible[:, (11, 12, 15, 16)].all(dim=1)
    indices = torch.cat((indices, torch.where(both_raised)[0].repeat(2)))

    width, height = rounds[0]["frame_size"]
    axes = torch.tensor([1., height / width], dtype=torch.float32)
    importance = torch.ones(33)
    importance[list(BODY)] = 2.
    importance[[15, 16]] = 3.
    bands = torch.arange(224) // 14
    model = CSIPoseNet(**MODEL_ARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0009, weight_decay=.01)
    for epoch in range(EPOCHS):
        model.train()
        for current in indices[torch.randperm(len(indices))].split(32):
            prior = previous[current]
            pair = torch.cat((current, prior))
            offsets = .4 * torch.randn(len(current), 1, 16)[:, :, bands]
            augmented = inputs[pair] + offsets.repeat(2, 1, 1) + .07 * torch.randn_like(inputs[pair])
            augmented[:, :, NULL_TONES] = 0
            _, predicted = model(augmented.clamp(-INPUT_CLIP, INPUT_CLIP))
            now, before = predicted.chunk(2)
            weights = visible[current] * importance
            error = (now - expected[current]) * axes
            position = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.035, reduction="none")
            position = (position.mean(dim=-1) * weights).sum() / weights.sum()
            paired = (visible[current] & visible[prior])[:, BODY] & (current != prior)[:, None]
            change = (now - before) - (expected[current] - expected[prior])
            difference = change[:, BODY][paired] * axes
            velocity = F.smooth_l1_loss(difference, torch.zeros_like(difference), beta=.025, reduction="sum")
            velocity = velocity / (2 * paired.sum()).clamp_min(1)
            loss = 12 * (position + .65 * velocity) + arm_pose_loss(now, expected[current], visible[current])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        if epoch == 29:
            for group in optimizer.param_groups:
                group["lr"] *= .45
    return model, normalizer, max_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint with fitted live presence classifier")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rounds = [read_round(path) for path in args.round]
    frame_size = rounds[0]["frame_size"]
    if any(not np.array_equal(g["frame_size"], frame_size) for g in rounds):
        parser.error("Reference image dimensions must match across rounds")
    times = np.concatenate([g["timestamps"] for g in rounds])
    if len(np.unique(times)) != len(times):
        parser.error("Duplicate window timestamp across input recordings")
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]
    if config.get("presence_classifier", {}).get("kind") != "supported_motion_v1":
        parser.error("Fit the live presence classifier before packaging the pose model")
    net, normalizer, max_distance = train_pose(rounds)
    width, height = map(int, frame_size)
    config["model"] = {"name": "CSIPoseNet", **MODEL_ARGS}
    config["data"].update(pose_input="link_rms_v1", frame_width=width, frame_height=height,
                          pose_reference_tones=list(NULL_TONES), pose_max_distance=max_distance,
                          normalized_input_clip=INPUT_CLIP)
    save_checkpoint(net, args.output / "smn_pose.pt", config=config, epoch=EPOCHS)
    normalizer.save(args.output / "smn_normalization.json")
    print(f"Trained on {sum(len(g['timestamps']) for g in rounds)} windows, pose distance limit {max_distance:.4f}")


if __name__ == "__main__":
    main()
