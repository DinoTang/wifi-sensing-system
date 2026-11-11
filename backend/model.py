"""Reuse the checkpoint and preprocessing distributed on the Khanh branch."""

from pathlib import Path
import sys

from .source import ROOT

ML_ROOT = ROOT / "machine-learning"
DEFAULT_CHECKPOINT = ML_ROOT / "models" / "checkpoints" / "best_model.pt"
DEFAULT_NORMALIZER = DEFAULT_CHECKPOINT.with_name("normalization_params.json")
WINDOW_SIZE = 20
FEATURES = 224
STRIDE = 2
MAX_GAP_MS = 300
NOMINAL_INTERVAL_MS = 101


def load_model(checkpoint: Path, normalizer: Path):
    sys.path.insert(0, str(ML_ROOT))
    import torch
    from src.predict import CSIPosePredictor, LANDMARK_NAMES

    torch.set_num_threads(1)
    model = CSIPosePredictor(checkpoint_path=checkpoint, normalizer_path=normalizer)
    if (model.window_size, model.subcarriers) != (WINDOW_SIZE, FEATURES):
        raise ValueError("Checkpoint does not accept 20 × 224 CSI windows")
    if model.config.get("data", {}).get("num_keypoints") != len(LANDMARK_NAMES):
        raise ValueError("Checkpoint does not use the expected BlazePose landmark order")
    return model
