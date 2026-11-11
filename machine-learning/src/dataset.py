"""Dataset, DataLoader, and Normalization utilities for CSI Pose Estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CSINormalizer:
    """Z-score normalizer fit exclusively on training sessions to prevent data leakage."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, csi_data: np.ndarray) -> CSINormalizer:
        """Fit mean and std over training samples.
        
        Args:
            csi_data: Array of shape (N, window_size, subcarriers)
        """
        # Compute mean & std per subcarrier across all samples and time steps
        flat = csi_data.reshape(-1, csi_data.shape[-1])
        self.mean = np.mean(flat, axis=0, dtype=np.float32)
        self.std = np.std(flat, axis=0, dtype=np.float32)
        # Avoid division by zero on static/null subcarriers
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, csi_data: np.ndarray) -> np.ndarray:
        """Transform CSI data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted yet.")
        return ((csi_data - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> CSINormalizer:
        mean = np.array(data["mean"], dtype=np.float32) if data.get("mean") else None
        std = np.array(data["std"], dtype=np.float32) if data.get("std") else None
        return cls(mean=mean, std=std)

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> CSINormalizer:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class CSIPoseDataset(Dataset):
    """PyTorch Dataset for paired CSI windows and 2D Pose landmarks."""

    def __init__(
        self,
        csi_samples: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        mask: np.ndarray,
        presence: np.ndarray,
        activities: List[str],
        sessions: List[str],
        normalizer: Optional[CSINormalizer] = None,
        frame_width: float = 1920.0,
        frame_height: float = 1080.0,
        normalize_coordinates: bool = True,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.normalize_coordinates = normalize_coordinates

        if normalizer is not None:
            self.csi = normalizer.transform(csi_samples)
        else:
            self.csi = csi_samples.astype(np.float32)

        kpts = keypoints.astype(np.float32).copy()
        vis = visibility.astype(np.float32).copy()
        msk = mask.astype(np.float32).copy()
        pres = presence.astype(np.float32).copy()

        # Invalidate mask wherever coordinates/visibility are non-finite or room is empty
        finite_kpts = np.isfinite(kpts).all(axis=-1)
        finite_vis = np.isfinite(vis)
        person_present_mask = (pres > 0.5)[:, None]
        msk = np.where(finite_kpts & finite_vis & person_present_mask, msk, 0.0).astype(np.float32)

        # Sanitize NaNs/Infs so IEEE 754 NaN * 0.0 never poisons loss or gradients
        kpts = np.nan_to_num(kpts, nan=0.0, posinf=0.0, neginf=0.0)
        vis = np.nan_to_num(vis, nan=0.0, posinf=1.0, neginf=0.0)
        self.csi = np.nan_to_num(self.csi, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize_coordinates:
            kpts[:, :, 0] /= frame_width
            kpts[:, :, 1] /= frame_height

        self.keypoints = kpts
        self.visibility = vis
        self.mask = msk
        self.presence = pres
        self.activities = activities
        self.sessions = sessions

    def __len__(self) -> int:
        return len(self.csi)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "csi": torch.tensor(self.csi[idx], dtype=torch.float32),
            "keypoints": torch.tensor(self.keypoints[idx], dtype=torch.float32),
            "visibility": torch.tensor(self.visibility[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.float32),
            "presence": torch.tensor(self.presence[idx], dtype=torch.float32),
            "activity": self.activities[idx],
            "session": self.sessions[idx],
        }


def find_data_dir(candidate_dir: Union[str, Path] = "../signal-processing/data/pose/windows") -> Path:
    """Find the valid pose windows dataset directory across standard paths."""
    candidates = [
        Path(candidate_dir),
        Path("data/windows"),
        Path("../signal-processing/data/pose/windows"),
        Path("signal-processing/data/pose/windows"),
        Path(__file__).resolve().parents[2] / "signal-processing" / "data" / "pose" / "windows",
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.npz")):
            return p.resolve()

    raise FileNotFoundError(
        f"Could not find pose windows directory with .npz files. Checked: {[str(c) for c in candidates]}"
    )


def load_raw_session_data(
    data_dir: Union[str, Path]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all .npz session files into memory keyed by session name."""
    data_path = Path(data_dir)
    sessions_data = {}

    for npz_file in sorted(data_path.glob("*.npz")):
        session_name = npz_file.stem
        with np.load(npz_file, allow_pickle=True) as data:
            csi = data["csi"]
            kpts = data["keypoints"]
            vis = data["visibility"]
            mask = data["mask"]
            activity = str(data["activity"])
            session = str(data["session"]) if "session" in data else session_name

            # Binary presence label: 0 for empty, 1 for person present
            is_present = 0.0 if activity.lower() == "empty" else 1.0
            presence = np.full((len(csi),), is_present, dtype=np.float32)

            sessions_data[session_name] = {
                "csi": csi,
                "keypoints": kpts,
                "visibility": vis,
                "mask": mask,
                "presence": presence,
                "activity": [activity] * len(csi),
                "session": [session] * len(csi),
            }

    return sessions_data


def build_split_arrays(
    sessions_data: Dict[str, Dict[str, np.ndarray]], session_keys: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Concatenate samples for a list of session keys."""
    csi_list, kpt_list, vis_list, mask_list, pres_list = [], [], [], [], []
    act_list, sess_list = [], []

    for key in session_keys:
        # Match by prefix or exact stem (e.g. empty_01)
        matching_keys = [k for k in sessions_data if k == key or k.startswith(key)]
        for mk in matching_keys:
            d = sessions_data[mk]
            csi_list.append(d["csi"])
            kpt_list.append(d["keypoints"])
            vis_list.append(d["visibility"])
            mask_list.append(d["mask"])
            pres_list.append(d["presence"])
            act_list.extend(d["activity"])
            sess_list.extend(d["session"])

    if not csi_list:
        raise ValueError(f"No samples found for sessions: {session_keys}")

    return (
        np.concatenate(csi_list, axis=0),
        np.concatenate(kpt_list, axis=0),
        np.concatenate(vis_list, axis=0),
        np.concatenate(mask_list, axis=0),
        np.concatenate(pres_list, axis=0),
        act_list,
        sess_list,
    )


def create_dataloaders(
    data_dir: Union[str, Path],
    train_sessions: List[str],
    val_sessions: List[str],
    test_sessions: List[str],
    batch_size: int = 32,
    normalize_coordinates: bool = True,
    frame_width: float = 1920.0,
    frame_height: float = 1080.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, CSINormalizer]:
    """Create train, validation, and test DataLoaders with session-level split."""
    sessions_data = load_raw_session_data(data_dir)

    train_data = build_split_arrays(sessions_data, train_sessions)
    val_data = build_split_arrays(sessions_data, val_sessions)
    test_data = build_split_arrays(sessions_data, test_sessions)

    # Fit normalizer ONLY on train data
    normalizer = CSINormalizer()
    normalizer.fit(train_data[0])

    train_dataset = CSIPoseDataset(
        *train_data,
        normalizer=normalizer,
        frame_width=frame_width,
        frame_height=frame_height,
        normalize_coordinates=normalize_coordinates,
    )
    val_dataset = CSIPoseDataset(
        *val_data,
        normalizer=normalizer,
        frame_width=frame_width,
        frame_height=frame_height,
        normalize_coordinates=normalize_coordinates,
    )
    test_dataset = CSIPoseDataset(
        *test_data,
        normalizer=normalizer,
        frame_width=frame_width,
        frame_height=frame_height,
        normalize_coordinates=normalize_coordinates,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, normalizer
