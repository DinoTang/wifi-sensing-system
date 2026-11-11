"""Evaluation metrics for CSI Human Presence and 2D Pose Estimation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_mpjpe(
    pred_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    mask: np.ndarray,
    presence: np.ndarray,
    pixel_scale: Optional[Tuple[float, float]] = (1920.0, 1080.0),
) -> Tuple[float, np.ndarray, int]:
    """Compute Mean Per Joint Position Error (MPJPE).
    
    Args:
        pred_keypoints: (N, 33, 2) in normalized [0, 1] coords
        target_keypoints: (N, 33, 2) in normalized [0, 1] coords
        mask: (N, 33) binary joint validity mask
        presence: (N,) binary presence indicator
        pixel_scale: (width, height) to convert to pixel error
        
    Returns:
        (mean_error_pixels, per_joint_error_pixels, num_valid_joints)
    """
    person_idx = np.where(presence > 0.5)[0]
    if len(person_idx) == 0:
        return 0.0, np.zeros((pred_keypoints.shape[1],)), 0

    pred = pred_keypoints[person_idx].copy()
    target = target_keypoints[person_idx].copy()
    valid_mask = mask[person_idx].astype(bool)

    if pixel_scale is not None:
        scale = np.array(pixel_scale, dtype=np.float32)
        pred *= scale
        target *= scale

    # Euclidean distance per joint: (N_person, 33)
    dist = np.linalg.norm(pred - target, axis=-1)

    valid_dists = dist[valid_mask]
    if len(valid_dists) == 0:
        return 0.0, np.zeros((pred_keypoints.shape[1],)), 0

    overall_mpjpe = float(np.mean(valid_dists))

    # Per joint error
    per_joint = np.zeros((pred.shape[1],), dtype=np.float32)
    for j in range(pred.shape[1]):
        m = valid_mask[:, j]
        if np.any(m):
            per_joint[j] = float(np.mean(dist[m, j]))

    return overall_mpjpe, per_joint, len(valid_dists)


def compute_pck(
    pred_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    mask: np.ndarray,
    presence: np.ndarray,
    thresholds: List[float] = [0.05, 0.10, 0.20],
) -> Dict[str, float]:
    """Compute Percentage of Correct Keypoints (PCK) at normalized thresholds.
    
    Normalized by image diagonal (sqrt(1.0^2 + 1.0^2) ≈ 1.414).
    """
    person_idx = np.where(presence > 0.5)[0]
    if len(person_idx) == 0:
        return {f"pck@{int(t*100)}": 0.0 for t in thresholds}

    pred = pred_keypoints[person_idx]
    target = target_keypoints[person_idx]
    valid_mask = mask[person_idx].astype(bool)

    dist = np.linalg.norm(pred - target, axis=-1)  # normalized coords
    valid_dists = dist[valid_mask]

    if len(valid_dists) == 0:
        return {f"pck@{int(t*100)}": 0.0 for t in thresholds}

    results = {}
    for t in thresholds:
        correct = np.sum(valid_dists <= t)
        results[f"pck@{int(t*100)}"] = float(correct / len(valid_dists) * 100.0)

    return results


def compute_presence_metrics(
    pred_probs: np.ndarray, target_presence: np.ndarray, threshold: float = 0.5
) -> Dict[str, Union[float, list]]:
    """Compute binary classification metrics for human presence."""
    preds_binary = (pred_probs >= threshold).astype(int)
    targets_binary = (target_presence >= 0.5).astype(int)

    acc = float(accuracy_score(targets_binary, preds_binary))
    prec = float(precision_score(targets_binary, preds_binary, zero_division=0))
    rec = float(recall_score(targets_binary, preds_binary, zero_division=0))
    f1 = float(f1_score(targets_binary, preds_binary, zero_division=0))
    cm = confusion_matrix(targets_binary, preds_binary, labels=[0, 1]).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }
