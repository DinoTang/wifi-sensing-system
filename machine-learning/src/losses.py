"""Loss functions for CSI Multi-Task Pose and Presence Estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPoseLoss(nn.Module):
    """Multi-task loss: Binary presence cross-entropy + visibility-weighted pose regression.
    
    Guarantees:
      1. Pose loss is computed ONLY on frames with a person present.
      2. Keypoint errors are masked by joint visibility and valid detection flags.
      3. For empty room samples, keypoint loss is strictly 0.0.
    """

    def __init__(self, lambda_presence: float = 1.0, lambda_pose: float = 5.0, beta: float = 0.05):
        super().__init__()
        self.lambda_presence = lambda_presence
        self.lambda_pose = lambda_pose
        self.beta = beta  # SmoothL1 threshold
        self.presence_bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_presence_logits: torch.Tensor,
        pred_keypoints: torch.Tensor,
        target_presence: torch.Tensor,
        target_keypoints: torch.Tensor,
        target_visibility: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.
        
        Returns:
            (total_loss, presence_loss, pose_loss)
        """
        # 1. Presence Loss (support both (B,) and (B, 1) shapes)
        pred_pres_flat = pred_presence_logits.reshape(-1)
        target_pres_flat = target_presence.reshape(-1)
        loss_presence = self.presence_bce(pred_pres_flat, target_pres_flat)

        # 2. Masked Pose Regression Loss
        # Guard against NaN/Inf in ground truth coordinates (IEEE 754: NaN * 0.0 == NaN)
        finite_kpts = torch.isfinite(target_keypoints).all(dim=-1)  # (batch, 33)
        finite_vis = torch.isfinite(target_visibility)  # (batch, 33)
        safe_target_kpts = torch.where(
            torch.isfinite(target_keypoints), target_keypoints, pred_keypoints.detach()
        )
        safe_visibility = torch.nan_to_num(target_visibility, nan=0.0, posinf=1.0, neginf=0.0)

        # person_mask: (batch_size, 1, 1)
        person_mask = (target_pres_flat > 0.5).float().view(-1, 1, 1)
        # joint_mask: (batch_size, 33, 1)
        joint_mask = ((target_mask > 0.5) & finite_kpts & finite_vis).float().unsqueeze(-1)
        # visibility weights: (batch_size, 33, 1)
        vis_weights = safe_visibility.clamp(min=0.0, max=1.0).unsqueeze(-1)

        # Combined per-coordinate weight
        weights = person_mask * joint_mask * vis_weights  # (batch, 33, 1)

        # Smooth L1 loss per coordinate
        smooth_l1 = F.smooth_l1_loss(
            pred_keypoints, safe_target_kpts, beta=self.beta, reduction="none"
        )  # (batch, 33, 2)

        weighted_loss = smooth_l1 * weights
        total_weight = torch.sum(weights) * 2.0  # x and y coords

        if total_weight > 0:
            loss_pose = torch.sum(weighted_loss) / total_weight
        else:
            loss_pose = torch.tensor(0.0, device=pred_keypoints.device)

        total_loss = self.lambda_presence * loss_presence + self.lambda_pose * loss_pose
        return total_loss, loss_presence, loss_pose
