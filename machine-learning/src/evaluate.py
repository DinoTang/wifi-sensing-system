"""Comprehensive Evaluation on Independent Test Sessions (Person 3)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .dataset import CSINormalizer, CSIPoseDataset, find_data_dir, load_raw_session_data
    from .metrics import compute_mpjpe, compute_pck, compute_presence_metrics
    from .models import load_checkpoint
    from .predict import CSIPosePredictor
except ImportError:
    from dataset import CSINormalizer, CSIPoseDataset, find_data_dir, load_raw_session_data
    from metrics import compute_mpjpe, compute_pck, compute_presence_metrics
    from models import load_checkpoint
    from predict import CSIPosePredictor

# 33 BlazePose connections for skeleton drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def measure_latency(model: torch.nn.Module, sample_shape: tuple, device: str, n_warmup: int = 10, n_iter: int = 100) -> Dict[str, float]:
    """Benchmark inference latency per CSI window."""
    dummy = torch.randn(*sample_shape, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

        timings = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return {
        "mean_latency_ms": float(np.mean(timings)),
        "std_latency_ms": float(np.std(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "throughput_fps": float(1000.0 / np.mean(timings)),
    }


def plot_confusion_matrix(cm: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)

    classes = ["No Human (Empty)", "Human Present"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Human Presence Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_mpjpe_by_activity(activity_errors: Dict[str, float], save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    acts = list(activity_errors.keys())
    errs = [activity_errors[a] for a in acts]

    bars = ax.bar(acts, errs, color="#2196f3", edgecolor="#1565c0", width=0.5)
    ax.set_ylabel("MPJPE (Pixels, 1920x1080 scale)")
    ax.set_title("Mean Per Joint Position Error by Activity")
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} px",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _draw_skeleton_matplotlib(ax, coords_norm: np.ndarray, mask_row: Optional[np.ndarray], limb_color: str, joint_color: str) -> None:
    for p1, p2 in POSE_CONNECTIONS:
        if mask_row is not None and (mask_row[p1] < 0.5 or mask_row[p2] < 0.5):
            continue
        ax.plot([coords_norm[p1, 0], coords_norm[p2, 0]], [coords_norm[p1, 1], coords_norm[p2, 1]], color=limb_color, lw=2.5)
    if mask_row is not None:
        valid = mask_row > 0.5
        ax.scatter(coords_norm[valid, 0], coords_norm[valid, 1], color=joint_color, s=25, zorder=3)
    else:
        ax.scatter(coords_norm[:, 0], coords_norm[:, 1], color=joint_color, s=25, zorder=3)


def plot_skeleton_comparison(
    pred_kpts_norm: np.ndarray,
    target_kpts_norm: np.ndarray,
    masks: np.ndarray,
    presence: np.ndarray,
    activities: List[str],
    save_path: Path,
    n_samples: int = 4,
    indices: Optional[List[int]] = None,
    title: str = "Ground Truth (Green) vs CSIPoseNet Prediction (Cyan/Orange)",
    sample_errors_px: Optional[np.ndarray] = None,
):
    """Draw side-by-side skeleton comparisons for Ground Truth vs Prediction."""
    if indices is None:
        person_idx = [i for i, p in enumerate(presence) if p > 0.5]
        if len(person_idx) == 0:
            return
        selected = np.linspace(0, len(person_idx) - 1, min(n_samples, len(person_idx)), dtype=int)
        chosen = [person_idx[s] for s in selected]
    else:
        chosen = list(indices)
        if not chosen:
            return

    fig, axes = plt.subplots(2, len(chosen), figsize=(5 * len(chosen), 9), facecolor="#1a1a24", squeeze=False)
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold")

    for col, idx in enumerate(chosen):
        act = activities[idx]
        gt = target_kpts_norm[idx]
        pred = pred_kpts_norm[idx]
        msk = masks[idx]

        # Row 0: Ground Truth
        ax_gt = axes[0, col]
        ax_gt.set_facecolor("#111116")
        ax_gt.set_title(f"GT: {act} (Win #{idx})", color="#4caf50", fontsize=11, fontweight="bold")
        ax_gt.set_xlim(0, 1)
        ax_gt.set_ylim(1, 0)
        ax_gt.axis("off")
        _draw_skeleton_matplotlib(ax_gt, gt, msk, "#4caf50", "#8bc34a")

        # Row 1: Prediction
        ax_pred = axes[1, col]
        ax_pred.set_facecolor("#111116")
        err_str = f" | Err: {sample_errors_px[idx]:.1f}px" if sample_errors_px is not None else ""
        ax_pred.set_title(f"Pred: {act}{err_str}", color="#00e5ff", fontsize=11, fontweight="bold")
        ax_pred.set_xlim(0, 1)
        ax_pred.set_ylim(1, 0)
        ax_pred.axis("off")
        _draw_skeleton_matplotlib(ax_pred, pred, None, "#ff9100", "#00e5ff")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_comparison_video(
    session_name: str,
    activity_name: str,
    pred_kpts_norm: np.ndarray,
    target_kpts_norm: np.ndarray,
    masks: np.ndarray,
    presence_target: np.ndarray,
    presence_probs: np.ndarray,
    save_path: Path,
    fps: float = 5.0,
    pixel_scale: tuple = (1920.0, 1080.0),
) -> str:
    """Export side-by-side Ground Truth vs Prediction comparison video (MP4/AVI)."""
    panel_w, panel_h = 540, 480
    canvas_w, canvas_h = panel_w * 2, panel_h + 70

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = save_path.with_suffix(".mp4")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_path = save_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))

    scale_arr = np.array(pixel_scale, dtype=np.float32)

    for i in range(len(pred_kpts_norm)):
        frame = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
        # Divider line
        cv2.line(frame, (panel_w, 70), (panel_w, canvas_h), (70, 70, 90), 2)
        cv2.line(frame, (0, 70), (canvas_w, 70), (70, 70, 90), 2)

        gt = target_kpts_norm[i]
        pred = pred_kpts_norm[i]
        msk = masks[i]
        is_gt_present = presence_target[i] > 0.5
        is_pred_present = presence_probs[i] >= 0.5

        if is_gt_present and np.any(msk > 0.5):
            dists = np.linalg.norm((pred - gt) * scale_arr, axis=-1)
            mpjpe_i = float(np.mean(dists[msk > 0.5]))
            err_text = f"MPJPE: {mpjpe_i:.1f} px"
        else:
            err_text = "MPJPE: N/A (empty)"

        # Header HUD
        cv2.putText(
            frame,
            f"Session: {session_name} ({activity_name}) | Win {i+1}/{len(pred_kpts_norm)} | {err_text}",
            (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 245), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"GT Presence: {int(is_gt_present)} (Green)",
            (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 230, 120), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Pred Presence: {int(is_pred_present)} (prob={presence_probs[i]:.2f}) (Orange/Cyan)",
            (panel_w + 16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 60), 1, cv2.LINE_AA
        )

        # Draw GT skeleton (left panel)
        if is_gt_present:
            for p1, p2 in POSE_CONNECTIONS:
                if msk[p1] > 0.5 and msk[p2] > 0.5:
                    pt1 = (int(np.clip(gt[p1, 0], 0, 1) * panel_w), 70 + int(np.clip(gt[p1, 1], 0, 1) * panel_h))
                    pt2 = (int(np.clip(gt[p2, 0], 0, 1) * panel_w), 70 + int(np.clip(gt[p2, 1], 0, 1) * panel_h))
                    cv2.line(frame, pt1, pt2, (80, 200, 90), 2, cv2.LINE_AA)
            for j in range(len(gt)):
                if msk[j] > 0.5:
                    pt = (int(np.clip(gt[j, 0], 0, 1) * panel_w), 70 + int(np.clip(gt[j, 1], 0, 1) * panel_h))
                    cv2.circle(frame, pt, 4, (130, 255, 140), -1, cv2.LINE_AA)

        # Draw Pred skeleton (right panel)
        if is_pred_present:
            for p1, p2 in POSE_CONNECTIONS:
                pt1 = (panel_w + int(np.clip(pred[p1, 0], 0, 1) * panel_w), 70 + int(np.clip(pred[p1, 1], 0, 1) * panel_h))
                pt2 = (panel_w + int(np.clip(pred[p2, 0], 0, 1) * panel_w), 70 + int(np.clip(pred[p2, 1], 0, 1) * panel_h))
                cv2.line(frame, pt1, pt2, (0, 150, 255), 2, cv2.LINE_AA)
            for j in range(len(pred)):
                pt = (panel_w + int(np.clip(pred[j, 0], 0, 1) * panel_w), 70 + int(np.clip(pred[j, 1], 0, 1) * panel_h))
                cv2.circle(frame, pt, 4, (255, 230, 0), -1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    return str(out_path)


def _compute_split_aggregates(
    preds_pose: list, targets_pose: list, masks: list, pres_preds: list, pres_targets: list
) -> Dict:
    if not preds_pose:
        return {}
    p_pose = np.concatenate(preds_pose, axis=0)
    t_pose = np.concatenate(targets_pose, axis=0)
    m = np.concatenate(masks, axis=0)
    pr_p = np.concatenate(pres_preds, axis=0)
    pr_t = np.concatenate(pres_targets, axis=0)

    pres_m = compute_presence_metrics(pr_p, pr_t)
    mpjpe_val, _, n_eval = compute_mpjpe(p_pose, t_pose, m, pr_t)
    pck_m = compute_pck(p_pose, t_pose, m, pr_t)
    return {
        "total_samples": int(len(pr_t)),
        "mpjpe_pixels": mpjpe_val,
        "evaluated_joints": int(n_eval),
        **pres_m,
        **pck_m,
    }


def _match_split(sess_name: str, split_keys: List[str]) -> bool:
    return any(sess_name == k or sess_name.startswith(k) for k in split_keys)


def run_evaluation(
    checkpoint_path: Union[str, Path] = "models/checkpoints/best_model.pt",
    norm_path: Union[str, Path] = "models/checkpoints/normalization_params.json",
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "results",
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("EVALUATING MODEL ON INDEPENDENT SESSIONS (Person 3)")
    print("=" * 78)

    model, config = load_checkpoint(checkpoint_path, device=device)
    normalizer = CSINormalizer.load(norm_path)
    predictor = CSIPosePredictor(checkpoint_path=checkpoint_path, normalizer_path=norm_path, device=device)

    data_cfg = config.get("data", {})
    resolved_data_dir = find_data_dir(data_dir or data_cfg.get("data_dir", "../signal-processing/data/pose/windows"))
    sessions_data = load_raw_session_data(resolved_data_dir)

    train_sessions = data_cfg.get("train_sessions", [])
    val_sessions = data_cfg.get("val_sessions", ["mixed_01"])
    test_sessions = data_cfg.get("test_sessions", ["walking_01"])
    frame_w = float(data_cfg.get("frame_width", 1920.0))
    frame_h = float(data_cfg.get("frame_height", 1080.0))

    print(f"Train Sessions (in-sample diagnostic): {train_sessions}")
    print(f"Val Sessions                         : {val_sessions}")
    print(f"Independent Test Sessions            : {test_sessions}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_metrics = {}
    activity_mpjpe = {}

    split_buckets = {
        "train": ([], [], [], [], [], []),
        "val": ([], [], [], [], [], []),
        "test": ([], [], [], [], [], []),
    }
    diagnostic_all = ([], [], [], [], [], [])
    exported_videos = []

    model.eval()
    with torch.no_grad():
        for sess_name, sdata in sorted(sessions_data.items()):
            split_tag = "other"
            if _match_split(sess_name, test_sessions):
                split_tag = "test"
            elif _match_split(sess_name, val_sessions):
                split_tag = "val"
            elif _match_split(sess_name, train_sessions):
                split_tag = "train"

            csi_norm = normalizer.transform(sdata["csi"])
            csi_t = torch.tensor(csi_norm, dtype=torch.float32, device=device)

            pres_logits, pred_kpts = model(csi_t)
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            pred_kpts_np = pred_kpts.cpu().numpy()

            act_name = sdata["activity"][0]
            kpts_target = np.nan_to_num(sdata["keypoints"].copy(), nan=0.0)
            kpts_target[:, :, 0] /= frame_w
            kpts_target[:, :, 1] /= frame_h
            mask_clean = np.where(
                np.isfinite(sdata["keypoints"]).all(axis=-1) & (sdata["presence"] > 0.5)[:, None],
                sdata["mask"],
                0.0,
            )

            pres_m = compute_presence_metrics(pres_probs, sdata["presence"])
            mpjpe_val, _, valid_joints = compute_mpjpe(
                pred_kpts_np, kpts_target, mask_clean, sdata["presence"]
            )
            pck_m = compute_pck(pred_kpts_np, kpts_target, mask_clean, sdata["presence"])

            session_metrics[sess_name] = {
                "split": split_tag,
                "activity": act_name,
                "samples": len(csi_norm),
                "mpjpe_px": mpjpe_val,
                "valid_joint_evaluations": valid_joints,
                "presence_accuracy": pres_m["accuracy"],
                "presence_confusion_matrix": pres_m["confusion_matrix"],
                **pck_m,
            }

            if act_name.lower() != "empty":
                activity_mpjpe[f"{act_name} ({split_tag})"] = mpjpe_val

            for bucket in ([split_buckets[split_tag]] if split_tag in split_buckets else []) + [diagnostic_all]:
                bucket[0].append(pred_kpts_np)
                bucket[1].append(kpts_target)
                bucket[2].append(mask_clean)
                bucket[3].append(pres_probs)
                bucket[4].append(sdata["presence"])
                bucket[5].extend(sdata["activity"])

            # Export side-by-side comparison videos for independent Val and Test sessions
            if split_tag in ("val", "test"):
                vid_path = export_comparison_video(
                    session_name=sess_name,
                    activity_name=act_name,
                    pred_kpts_norm=pred_kpts_np,
                    target_kpts_norm=kpts_target,
                    masks=mask_clean,
                    presence_target=sdata["presence"],
                    presence_probs=pres_probs,
                    save_path=out_dir / f"pose_comparison_{sess_name}.mp4",
                    fps=5.0,
                    pixel_scale=(frame_w, frame_h),
                )
                exported_videos.append(vid_path)

    test_summary = _compute_split_aggregates(*split_buckets["test"][:5])
    val_summary = _compute_split_aggregates(*split_buckets["val"][:5])
    all_diag_summary = _compute_split_aggregates(*diagnostic_all[:5])

    # Benchmark both raw forward and end-to-end CSIPosePredictor latency
    model_latency = measure_latency(model, (1, 20, 224), device=device)
    dummy_raw = np.zeros((20, 224), dtype=np.float32)
    e2e_timings = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = predictor.predict_window(dummy_raw, timestamp=time.time())
        e2e_timings.append((time.perf_counter() - t0) * 1000.0)
    e2e_latency = {
        "mean_e2e_latency_ms": float(np.mean(e2e_timings)),
        "p95_e2e_latency_ms": float(np.percentile(e2e_timings, 95)),
        "e2e_throughput_fps": float(1000.0 / np.mean(e2e_timings)),
    }

    # Plots: representative test/val predictions + top failure cases on unseen sessions
    unseen_preds = np.concatenate(split_buckets["val"][0] + split_buckets["test"][0], axis=0)
    unseen_targs = np.concatenate(split_buckets["val"][1] + split_buckets["test"][1], axis=0)
    unseen_masks = np.concatenate(split_buckets["val"][2] + split_buckets["test"][2], axis=0)
    unseen_pres = np.concatenate(split_buckets["val"][4] + split_buckets["test"][4], axis=0)
    unseen_acts = split_buckets["val"][5] + split_buckets["test"][5]

    scale_vec = np.array([frame_w, frame_h], dtype=np.float32)
    per_sample_dists = np.linalg.norm((unseen_preds - unseen_targs) * scale_vec, axis=-1)
    per_sample_err = np.array([
        float(np.mean(per_sample_dists[i, unseen_masks[i] > 0.5])) if np.any(unseen_masks[i] > 0.5) else 0.0
        for i in range(len(unseen_preds))
    ])

    plot_confusion_matrix(all_diag_summary["confusion_matrix"], out_dir / "presence_confusion_matrix.png")
    plot_mpjpe_by_activity(activity_mpjpe, out_dir / "mpjpe_by_activity.png")
    plot_skeleton_comparison(
        unseen_preds, unseen_targs, unseen_masks, unseen_pres, unseen_acts,
        out_dir / "pose_prediction_comparison.png",
        title="Unseen Sessions (Val/Test): Ground Truth vs CSIPoseNet Prediction",
        sample_errors_px=per_sample_err,
    )

    worst_indices = np.argsort(per_sample_err)[-4:][::-1].tolist()
    plot_skeleton_comparison(
        unseen_preds, unseen_targs, unseen_masks, unseen_pres, unseen_acts,
        out_dir / "failure_cases.png",
        indices=worst_indices,
        title="Top Failure Cases on Unseen Sessions (Highest MPJPE)",
        sample_errors_px=per_sample_err,
    )

    evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "independent_test": test_summary,
        "validation": val_summary,
        "all_sessions_diagnostic_only": all_diag_summary,
        "presence_evaluation_note": (
            "empty_01 is the only empty session and is assigned to train; therefore independent "
            "validation (mixed_01) and test (walking_01) contain only human-present windows. "
            "Independent empty false-positive testing requires recording an unseen empty_02 session."
        ),
        "latency": {**model_latency, **e2e_latency},
        "per_session": session_metrics,
        "per_activity_mpjpe": activity_mpjpe,
        "exported_videos": exported_videos,
    }

    with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print("\n--- Independent Test Summary (walking_01) ---")
    print(f"Test MPJPE: {test_summary.get('mpjpe_pixels', 0):.2f} pixels (1920x1080 scale)")
    print(f"Test PCK@5%: {test_summary.get('pck@5', 0):.1f}% | PCK@10%: {test_summary.get('pck@10', 0):.1f}% | PCK@20%: {test_summary.get('pck@20', 0):.1f}%")
    print(f"Test Presence Accuracy: {test_summary.get('accuracy', 0)*100:.1f}%")
    print(f"Validation MPJPE (mixed_01): {val_summary.get('mpjpe_pixels', 0):.2f} pixels")
    print(f"End-to-End Window Latency: {e2e_latency['mean_e2e_latency_ms']:.2f} ms ({e2e_latency['e2e_throughput_fps']:.1f} FPS)")
    print(f"Exported Videos: {exported_videos}")
    print(f"Saved artifacts to {out_dir}/")

    return evaluation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CSI Pose Estimation Model")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--normalizer", default="models/checkpoints/normalization_params.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.normalizer, output_dir=args.output_dir)
