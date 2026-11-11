"""Training pipeline for CSI Human Presence and Pose Estimation models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

try:
    from .dataset import create_dataloaders, find_data_dir
    from .losses import MaskedPoseLoss
    from .metrics import compute_mpjpe, compute_pck, compute_presence_metrics
    from .models import CSIPoseNet, LinearMLPBaseline, MeanPoseBaseline, save_checkpoint
except ImportError:
    from dataset import create_dataloaders, find_data_dir
    from losses import MaskedPoseLoss
    from metrics import compute_mpjpe, compute_pck, compute_presence_metrics
    from models import CSIPoseNet, LinearMLPBaseline, MeanPoseBaseline, save_checkpoint


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MaskedPoseLoss,
    device: str,
) -> Tuple[float, float, float, Dict]:
    model.eval()
    total_loss, total_pres_loss, total_pose_loss = 0.0, 0.0, 0.0
    all_preds_presence, all_target_presence = [], []
    all_preds_pose, all_target_pose = [], []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            csi = batch["csi"].to(device)
            kpts = batch["keypoints"].to(device)
            vis = batch["visibility"].to(device)
            mask = batch["mask"].to(device)
            presence = batch["presence"].to(device)

            pres_logits, pred_kpts = model(csi)
            loss, p_loss, k_loss = criterion(pres_logits, pred_kpts, presence, kpts, vis, mask)

            total_loss += loss.item() * len(csi)
            total_pres_loss += p_loss.item() * len(csi)
            total_pose_loss += k_loss.item() * len(csi)

            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            all_preds_presence.append(pres_probs)
            all_target_presence.append(presence.cpu().numpy())

            all_preds_pose.append(pred_kpts.cpu().numpy())
            all_target_pose.append(kpts.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_pres_loss = total_pres_loss / num_samples
    avg_pose_loss = total_pose_loss / num_samples

    all_preds_presence = np.concatenate(all_preds_presence, axis=0)
    all_target_presence = np.concatenate(all_target_presence, axis=0)
    all_preds_pose = np.concatenate(all_preds_pose, axis=0)
    all_target_pose = np.concatenate(all_target_pose, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    pres_metrics = compute_presence_metrics(all_preds_presence, all_target_presence)
    mpjpe_px, _, _ = compute_mpjpe(all_preds_pose, all_target_pose, all_masks, all_target_presence)
    pck_dict = compute_pck(all_preds_pose, all_target_pose, all_masks, all_target_presence)

    metrics = {
        "loss": avg_loss,
        "presence_loss": avg_pres_loss,
        "pose_loss": avg_pose_loss,
        "mpjpe_px": mpjpe_px,
        **pres_metrics,
        **pck_dict,
    }
    return avg_loss, mpjpe_px, pres_metrics["accuracy"], metrics


def train_neural_model(
    model: torch.nn.Module,
    model_label: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: MaskedPoseLoss,
    checkpoint_path: Path,
    config_to_save: Dict,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
    gain_shift_aug_std: float = 0.0,
    subcarrier_shift_aug_std: float = 0.0,
    noise_aug_std: float = 0.0,
) -> Tuple[Dict, int, float, float]:
    """Train a PyTorch CSI presence + pose model with early stopping and checkpointing."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_presence_loss": [],
        "train_pose_loss": [],
        "val_loss": [],
        "val_presence_loss": [],
        "val_pose_loss": [],
        "val_mpjpe_px": [],
        "val_presence_acc": [],
        "lr": [],
    }

    print(f"\n--- Training {model_label} ---")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, train_pres_loss, train_pose_loss = 0.0, 0.0, 0.0

        for batch in train_loader:
            csi = batch["csi"].to(device)
            kpts = batch["keypoints"].to(device)
            vis = batch["visibility"].to(device)
            mask = batch["mask"].to(device)
            presence = batch["presence"].to(device)

            if gain_shift_aug_std > 0:
                csi = csi + torch.randn((csi.size(0), 1, 1), device=device) * gain_shift_aug_std
            if subcarrier_shift_aug_std > 0:
                csi = csi + torch.randn((csi.size(0), 1, csi.size(2)), device=device) * subcarrier_shift_aug_std
            if noise_aug_std > 0:
                csi = csi + torch.randn_like(csi) * noise_aug_std

            optimizer.zero_grad()
            pres_logits, pred_kpts = model(csi)
            loss, p_loss, k_loss = criterion(pres_logits, pred_kpts, presence, kpts, vis, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * len(csi)
            train_pres_loss += p_loss.item() * len(csi)
            train_pose_loss += k_loss.item() * len(csi)

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_pres_loss /= n_train
        train_pose_loss /= n_train

        val_loss, val_mpjpe, val_pres_acc, val_metrics = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        curr_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_presence_loss"].append(train_pres_loss)
        history["train_pose_loss"].append(train_pose_loss)
        history["val_loss"].append(val_loss)
        history["val_presence_loss"].append(val_metrics["presence_loss"])
        history["val_pose_loss"].append(val_metrics["pose_loss"])
        history["val_mpjpe_px"].append(val_mpjpe)
        history["val_presence_acc"].append(val_pres_acc)
        history["lr"].append(curr_lr)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MPJPE: {val_mpjpe:.1f} px | "
                f"Val Pres Acc: {val_pres_acc*100:.1f}% | "
                f"LR: {curr_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model=model,
                filepath=checkpoint_path,
                config=config_to_save,
                epoch=epoch,
                val_loss=val_loss,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    train_duration = time.time() - start_time
    print(f"[{model_label}] Completed in {train_duration:.1f}s. Best Epoch: {best_epoch} | Val Loss: {best_val_loss:.4f}")
    return history, best_epoch, best_val_loss, train_duration


def train_model(config_path: Union[str, Path]) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f"CSI POSE ESTIMATION - TRAINING PIPELINE (PyTorch {torch.__version__} on {device})")
    print("=" * 78)

    # 1. Resolve and verify data directory
    data_cfg = config["data"]
    data_dir = find_data_dir(data_cfg.get("data_dir", "../signal-processing/data/pose/windows"))
    print(f"Data directory: {data_dir}")
    print(f"Train sessions: {data_cfg['train_sessions']}")
    print(f"Val sessions  : {data_cfg['val_sessions']}")
    print(f"Test sessions : {data_cfg['test_sessions']}")

    # 2. Build DataLoaders & Normalizer
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        data_dir=data_dir,
        train_sessions=data_cfg["train_sessions"],
        val_sessions=data_cfg["val_sessions"],
        test_sessions=data_cfg["test_sessions"],
        batch_size=config["training"]["batch_size"],
        normalize_coordinates=data_cfg.get("normalize_coordinates", True),
        frame_width=data_cfg.get("frame_width", 1920.0),
        frame_height=data_cfg.get("frame_height", 1080.0),
    )

    print(f"Loaded samples -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Save Normalizer
    checkpoints_dir = Path("models/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    norm_path = checkpoints_dir / "normalization_params.json"
    normalizer.save(norm_path)
    print(f"Saved normalizer to {norm_path}")

    # 3. Naive Baseline (MeanPoseBaseline - Step 4)
    print("\n--- Baseline 1: Naive MeanPoseBaseline ---")
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    mean_baseline = MeanPoseBaseline()
    mean_baseline.fit(train_dataset.keypoints, train_dataset.presence, train_dataset.mask)

    val_pres_pred, val_pose_pred = mean_baseline.predict(val_dataset.csi)
    baseline_val_mpjpe, _, _ = compute_mpjpe(
        val_pose_pred, val_dataset.keypoints, val_dataset.mask, val_dataset.presence
    )
    baseline_val_pck = compute_pck(
        val_pose_pred, val_dataset.keypoints, val_dataset.mask, val_dataset.presence
    )
    baseline_val_pres = compute_presence_metrics(val_pres_pred, val_dataset.presence)

    test_pres_pred, test_pose_pred = mean_baseline.predict(test_dataset.csi)
    baseline_test_mpjpe, _, _ = compute_mpjpe(
        test_pose_pred, test_dataset.keypoints, test_dataset.mask, test_dataset.presence
    )
    baseline_test_pck = compute_pck(
        test_pose_pred, test_dataset.keypoints, test_dataset.mask, test_dataset.presence
    )
    baseline_test_pres = compute_presence_metrics(test_pres_pred, test_dataset.presence)
    print(f"[MeanPoseBaseline] Val MPJPE: {baseline_val_mpjpe:.2f} px | Test MPJPE: {baseline_test_mpjpe:.2f} px")

    t_cfg = config["training"]
    criterion = MaskedPoseLoss(
        lambda_presence=t_cfg["loss_weights"]["presence"],
        lambda_pose=t_cfg["loss_weights"]["pose"],
    )

    # 4. Simple Neural Baseline (LinearMLPBaseline - Step 4)
    m_cfg = config["model"]
    mlp_model = LinearMLPBaseline(
        window_size=data_cfg.get("window_size", 20),
        subcarriers=m_cfg.get("input_features", 224),
        num_keypoints=m_cfg.get("num_keypoints", 33),
    ).to(device)
    mlp_config = {**config, "model": {**m_cfg, "name": "LinearMLPBaseline"}}
    mlp_ckpt = checkpoints_dir / "mlp_baseline.pt"
    _, mlp_best_ep, mlp_best_val, mlp_dur = train_neural_model(
        model=mlp_model,
        model_label="Baseline 2: LinearMLPBaseline",
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        checkpoint_path=mlp_ckpt,
        config_to_save=mlp_config,
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        max_epochs=min(30, t_cfg["max_epochs"]),
        patience=10,
        device=device,
    )
    mlp_best_loaded = torch.load(mlp_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    mlp_model.load_state_dict(mlp_best_loaded)
    _, mlp_val_mpjpe, _, mlp_val_metrics = evaluate_epoch(mlp_model, val_loader, criterion, device)
    _, mlp_test_mpjpe, _, mlp_test_metrics = evaluate_epoch(mlp_model, test_loader, criterion, device)

    # 5. Main Model (CSIPoseNet - Step 5)
    set_seed(seed)
    model = CSIPoseNet(
        input_features=m_cfg.get("input_features", 224),
        conv_channels=tuple(m_cfg.get("conv_channels", [64, 128])),
        gru_hidden_dim=m_cfg.get("gru_hidden_dim", 96),
        gru_num_layers=m_cfg.get("gru_num_layers", 2),
        bidirectional=m_cfg.get("bidirectional", True),
        dropout=m_cfg.get("dropout", 0.35),
        num_keypoints=m_cfg.get("num_keypoints", 33),
        use_temporal_diff=m_cfg.get("use_temporal_diff", True),
        bound_pose=m_cfg.get("bound_pose", True),
    ).to(device)
    best_model_path = checkpoints_dir / "best_model.pt"
    history, best_epoch, best_val_loss, train_duration = train_neural_model(
        model=model,
        model_label="Main Model: CSIPoseNet (Conv1D + BiGRU + Attention + TemporalDiff)",
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        checkpoint_path=best_model_path,
        config_to_save=config,
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        max_epochs=t_cfg["max_epochs"],
        patience=t_cfg["early_stopping_patience"],
        device=device,
        gain_shift_aug_std=t_cfg.get("gain_shift_aug_std", 0.5),
        subcarrier_shift_aug_std=t_cfg.get("subcarrier_shift_aug_std", 0.5),
        noise_aug_std=t_cfg.get("noise_aug_std", 0.08),
    )

    best_loaded = torch.load(best_model_path, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(best_loaded)
    _, csi_train_mpjpe, csi_train_acc, csi_train_metrics = evaluate_epoch(model, train_loader, criterion, device)
    _, csi_val_mpjpe, _, csi_val_metrics = evaluate_epoch(model, val_loader, criterion, device)
    _, csi_test_mpjpe, _, csi_test_metrics = evaluate_epoch(model, test_loader, criterion, device)

    print("\n--- Model Comparison Summary (Val: mixed_01 | Independent Test: walking_01) ---")
    print(f"MeanPoseBaseline  | Val MPJPE: {baseline_val_mpjpe:6.2f} px | Test MPJPE: {baseline_test_mpjpe:6.2f} px")
    print(f"LinearMLPBaseline | Val MPJPE: {mlp_val_mpjpe:6.2f} px | Test MPJPE: {mlp_test_mpjpe:6.2f} px")
    print(f"CSIPoseNet        | Val MPJPE: {csi_val_mpjpe:6.2f} px | Test MPJPE: {csi_test_mpjpe:6.2f} px")

    # Save model config and training history
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoints_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(results_dir / "experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_duration_sec": train_duration,
                "history": history,
                "model_comparison": {
                    "MeanPoseBaseline": {
                        "val": {"mpjpe_px": baseline_val_mpjpe, **baseline_val_pres, **baseline_val_pck},
                        "test": {"mpjpe_px": baseline_test_mpjpe, **baseline_test_pres, **baseline_test_pck},
                    },
                    "LinearMLPBaseline": {
                        "best_epoch": mlp_best_ep,
                        "train_duration_sec": mlp_dur,
                        "val": mlp_val_metrics,
                        "test": mlp_test_metrics,
                    },
                    "CSIPoseNet": {
                        "best_epoch": best_epoch,
                        "train_duration_sec": train_duration,
                        "train_diagnostic": csi_train_metrics,
                        "val": csi_val_metrics,
                        "test": csi_test_metrics,
                    },
                },
            },
            f,
            indent=2,
        )

    # Plot Loss Curve
    plot_loss_curves(history, results_dir / "loss_curve.png")
    return history


def plot_loss_curves(history: Dict, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = history["epoch"]

    # 1. Total Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Total Loss", color="#00bcd4", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Total Loss", color="#ff5722", lw=2)
    axes[0].set_title("Total Loss (Presence + Pose)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Pose MPJPE (Pixels)
    axes[1].plot(epochs, history["val_mpjpe_px"], label="Val MPJPE (px)", color="#4caf50", lw=2)
    axes[1].set_title("Validation MPJPE (Pixel Error)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (px)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # 3. Presence Accuracy
    axes[2].plot(epochs, [x * 100 for x in history["val_presence_acc"]], label="Val Presence Acc (%)", color="#9c27b0", lw=2)
    axes[2].set_title("Validation Human Presence Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(50, 102)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSI Pose Estimation Model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.json"),
        help="Path to configuration JSON",
    )
    args = parser.parse_args()
    train_model(args.config)
