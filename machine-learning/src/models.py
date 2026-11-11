"""CSI Pose Estimation Models: Baselines and Deep Neural Network architectures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoseBaseline:
    """Naive baseline predicting the constant mean pose of training data."""

    def __init__(self, mean_pose: Optional[np.ndarray] = None, presence_prob: float = 0.5):
        self.mean_pose = mean_pose  # Shape: (33, 2)
        self.presence_prob = presence_prob

    def fit(self, keypoints: np.ndarray, presence: np.ndarray, mask: np.ndarray) -> MeanPoseBaseline:
        # Compute mean only over valid joints when person is present
        person_mask = (presence > 0.5)
        valid_kpts = keypoints[person_mask]
        valid_masks = mask[person_mask]

        if len(valid_kpts) > 0:
            # Weighted average per joint
            sum_kpts = np.sum(valid_kpts * valid_masks[:, :, None], axis=0)
            count = np.sum(valid_masks, axis=0)[:, None]
            count[count < 1] = 1.0
            self.mean_pose = sum_kpts / count
        else:
            self.mean_pose = np.zeros((33, 2), dtype=np.float32)

        self.presence_prob = float(np.mean(presence))
        return self

    def predict(self, csi_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (presence_probs, predicted_keypoints)."""
        N = len(csi_batch)
        presence = np.full((N,), self.presence_prob, dtype=np.float32)
        poses = np.tile(self.mean_pose[None, :, :], (N, 1, 1))
        return presence, poses


class LinearMLPBaseline(nn.Module):
    """Linear / MLP Baseline for CSI Presence & Pose Regression."""

    def __init__(
        self,
        window_size: int = 20,
        subcarriers: int = 224,
        num_keypoints: int = 33,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = window_size * subcarriers
        self.num_keypoints = num_keypoints

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        self.presence_head = nn.Linear(hidden_dim // 2, 1)
        self.pose_head = nn.Linear(hidden_dim // 2, num_keypoints * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, window_size, subcarriers)
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        feat = self.encoder(flat)

        presence_logits = self.presence_head(feat).squeeze(-1)  # (batch_size,)
        pose = self.pose_head(feat).view(batch_size, self.num_keypoints, 2)  # (batch_size, 33, 2)
        return presence_logits, pose


class TemporalAttention(nn.Module):
    """Attention pooling across time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, hidden_dim)
        weights = F.softmax(self.attn(x), dim=1)  # (batch, time, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch, hidden_dim)
        return pooled


class CSIPoseNet(nn.Module):
    """Deep Spatio-Temporal Network for Wi-Fi CSI 2D Pose Estimation.
    
    Architecture:
      1. 1D Temporal-Convolutional feature extractor for subcarrier spectrum dynamics.
      2. Bidirectional GRU to capture sequence velocity and acceleration.
      3. Attention-based temporal pooling.
      4. Dual-Head Decoders:
         - Binary presence classification head
         - 2D keypoint regression head with residual connections
    """

    def __init__(
        self,
        input_features: int = 224,
        conv_channels: Tuple[int, ...] = (64, 128),
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_keypoints: int = 33,
        use_temporal_diff: bool = True,
        bound_pose: bool = True,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.bidirectional = bidirectional
        self.use_temporal_diff = use_temporal_diff
        self.bound_pose = bound_pose

        conv_in_channels = input_features * 2 if use_temporal_diff else input_features

        # Conv1D across time steps: expects (batch, in_channels, seq_len)
        self.conv_net = nn.Sequential(
            nn.Conv1d(conv_in_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout / 2),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout / 2),
        )

        # Recurrent sequential modeling
        self.gru = nn.GRU(
            input_size=conv_channels[1],
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        rnn_out_dim = gru_hidden_dim * (2 if bidirectional else 1)
        self.temporal_attention = TemporalAttention(rnn_out_dim)

        # 4 gain-invariant temporal motion energy invariants appended for presence head:
        # [log1p(mean_sub(std_t(x))), log1p(std_sub(std_t(x))), log1p(rms(dx)), log1p(mean_sub(|dx|))]
        self.energy_dim = 4

        # Dual-stream presence classification: spatiotemporal branch + gain-invariant motion energy branch
        # Using heavy dropout on static feat stops it from silencing the generalizable energy branch.
        self.presence_head = nn.Sequential(
            nn.Dropout(min(0.85, dropout + 0.35)),
            nn.Linear(rnn_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.energy_bn = nn.BatchNorm1d(self.energy_dim)
        self.energy_presence_head = nn.Sequential(
            nn.Linear(self.energy_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # 2D Pose regression head
        self.pose_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_keypoints * 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, time_steps, subcarriers)
        dx = torch.zeros_like(x)
        dx[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        if self.use_temporal_diff:
            x_feat = torch.cat([x, dx], dim=-1)  # (batch_size, time_steps, subcarriers * 2)
        else:
            x_feat = x

        # Compute gain-invariant temporal motion energy descriptors (epsilon-guarded for smooth grads)
        t_std = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-6)  # (batch, subcarriers)
        dx_slice = dx[:, 1:, :] if x.size(1) > 1 else dx
        dx_rms = torch.sqrt(torch.mean(dx_slice ** 2, dim=1) + 1e-6)  # (batch, subcarriers)
        energy_feats = torch.stack(
            [
                torch.log1p(t_std.mean(dim=-1)),
                torch.log1p(torch.sqrt(torch.var(t_std, dim=-1, unbiased=False) + 1e-6)),
                torch.log1p(dx_rms.mean(dim=-1)),
                torch.log1p(torch.abs(dx_slice).mean(dim=(1, 2))),
            ],
            dim=-1,
        )  # (batch, 4)

        # Conv1D expects: (batch_size, channels, time_steps)
        x_trans = x_feat.transpose(1, 2)
        conv_out = self.conv_net(x_trans)  # (batch_size, conv_channels[-1], time_steps)
        gru_in = conv_out.transpose(1, 2)  # (batch_size, time_steps, conv_channels[-1])

        gru_out, _ = self.gru(gru_in)  # (batch_size, time_steps, rnn_out_dim)
        feat = self.temporal_attention(gru_out)  # (batch_size, rnn_out_dim)

        static_logit = 2.0 * torch.tanh(self.presence_head(feat) / 2.0)
        if self.training:
            branch_mask = (torch.rand_like(static_logit) > 0.5).float()
            static_logit = static_logit * branch_mask
        ef_norm = self.energy_bn(energy_feats) if (not self.training or x.size(0) > 1) else energy_feats
        energy_logit = self.energy_presence_head(ef_norm)

        presence_logits = (static_logit + energy_logit).squeeze(-1)  # (batch_size,)
        pose_raw = self.pose_head(feat).view(-1, self.num_keypoints, 2)  # (batch_size, 33, 2)
        pose = torch.sigmoid(pose_raw) if self.bound_pose else pose_raw

        return presence_logits, pose


def save_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    config: Optional[Dict] = None,
    epoch: int = 0,
    val_loss: float = 0.0,
) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(payload, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})

    model_name = model_cfg.get("name", "CSIPoseNet")
    if model_name == "LinearMLPBaseline":
        model = LinearMLPBaseline(
            window_size=config.get("data", {}).get("window_size", 20),
            subcarriers=model_cfg.get("input_features", 224),
            num_keypoints=model_cfg.get("num_keypoints", 33),
        )
    else:
        model = CSIPoseNet(
            input_features=model_cfg.get("input_features", 224),
            conv_channels=tuple(model_cfg.get("conv_channels", [64, 128])),
            gru_hidden_dim=model_cfg.get("gru_hidden_dim", 128),
            gru_num_layers=model_cfg.get("gru_num_layers", 2),
            bidirectional=model_cfg.get("bidirectional", True),
            dropout=model_cfg.get("dropout", 0.2),
            num_keypoints=model_cfg.get("num_keypoints", 33),
            use_temporal_diff=model_cfg.get("use_temporal_diff", True),
            bound_pose=model_cfg.get("bound_pose", True),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config
