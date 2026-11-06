"""Visualize synchronized video with skeleton overlay and CSI amplitude.

Fulfills Step 7 / "Việc nên làm trước tiên" of Người 2:
- Processes a complete left_arm session.
- Draws MediaPipe 33-joint skeleton on video frames.
- Plots CSI amplitude across subcarriers side-by-side with a time cursor.
- Outputs a side-by-side video demo (MP4) and a composite PNG figure.

Usage:
    python signal-processing/src/visualize_pose_csi_sample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = (
    PROJECT
    / "csi-data"
    / "video-sessions"
    / "P01_left_arm_01_20261103-025544_ttVs"
    / "copy_75958A5C-7861-4AFE-AAF1-F76CF2805543.mov"
)
DEFAULT_LANDMARKS = PROJECT / "data" / "pose" / "landmarks" / "left_arm_01.csv"
DEFAULT_CSI = PROJECT / "data" / "pose" / "raw_csv" / "left_arm_01.csv"
DEFAULT_OUTPUT_MP4 = PROJECT / "data" / "pose" / "results" / "left_arm_synced_demo.mp4"
DEFAULT_OUTPUT_PNG = PROJECT / "data" / "pose" / "results" / "left_arm_synced_sample.png"

# BlazePose skeleton connections
POSE_CONNECTIONS = [
    # Shoulders and torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


def load_csi_matrix(csv_path: Path):
    df = pd.read_csv(csv_path, usecols=["frame_id", "timestamp_ms", "sample_index", "amplitude"])
    pivot = df.pivot_table(index="frame_id", columns="sample_index", values="amplitude", aggfunc="first")
    pivot = pivot.sort_index()
    ts = df.groupby("frame_id", sort=True)["timestamp_ms"].first().values
    ts_sec = (ts - ts[0]) / 1000.0
    return pivot.values, ts_sec, pivot.index.values


def draw_skeleton(frame, landmarks_row, orig_w=1920, orig_h=1080):
    h, w = frame.shape[:2]
    scale_x = w / orig_w
    scale_y = h / orig_h

    # Extract coordinates
    pts = {}
    for i in range(33):
        col_x = f"landmark_{i}_x" if f"landmark_{i}_x" in landmarks_row else None
        # In extract_pose.py, landmark names are used
        pass

    # Find columns by landmark names
    from extract_pose import LANDMARKS
    coords = []
    visibilities = []
    for name in LANDMARKS:
        x_val = landmarks_row.get(f"{name}_x", 0.0)
        y_val = landmarks_row.get(f"{name}_y", 0.0)
        vis_val = landmarks_row.get(f"{name}_vis", 0.0)
        coords.append((int(x_val * scale_x), int(y_val * scale_y)))
        visibilities.append(vis_val)

    # Draw lines
    for p1_idx, p2_idx in POSE_CONNECTIONS:
        if visibilities[p1_idx] > 0.3 and visibilities[p2_idx] > 0.3:
            # Highlight left arm in green/cyan, others in yellow
            if p1_idx in (11, 13, 15) and p2_idx in (11, 13, 15):
                color = (0, 255, 255)  # Cyan for left arm
                thickness = 3
            else:
                color = (0, 200, 200)
                thickness = 2
            cv2.line(frame, coords[p1_idx], coords[p2_idx], color, thickness)

    # Draw points
    for idx, (pt, vis) in enumerate(coords):
        if vis > 0.3:
            if idx in (11, 13, 15):  # left shoulder, elbow, wrist
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                cv2.circle(frame, pt, 7, (0, 255, 255), 1)
            else:
                cv2.circle(frame, pt, 3, (0, 140, 255), -1)

    return frame


def create_demo_video(
    video_path: Path,
    landmarks_csv: Path,
    csi_csv: Path,
    out_mp4: Path,
    start_sec: float = 15.0,
    duration_sec: float = 15.0,
):
    print(f"Loading data for {video_path.name}...")
    csi_matrix, csi_ts, csi_frame_ids = load_csi_matrix(csi_csv)
    landmarks_df = pd.read_csv(landmarks_csv)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    target_fps = 10.0
    frame_step = int(fps / target_fps)
    start_frame = int(start_sec * fps)
    total_steps = int(duration_sec * target_fps)

    # Target frame dimensions
    vid_w, vid_h = 540, 540
    plot_w, plot_h = 660, 540
    out_w = vid_w + plot_w
    out_h = vid_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), fourcc, target_fps, (out_w, out_h))

    # Pre-render CSI background heatmap
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=100)
    plt.tight_layout()

    print(f"Rendering demo video ({total_steps} frames, {duration_sec} s)...")
    for step_i in range(total_steps):
        curr_frame_idx = start_frame + step_i * frame_step
        curr_time_s = curr_frame_idx / fps

        # Find closest landmark row
        closest_lm_idx = (landmarks_df["video_time_s"] - curr_time_s).abs().idxmin()
        lm_row = landmarks_df.iloc[closest_lm_idx]

        # Read video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # Crop center square and resize
        h, w = frame.shape[:2]
        crop_size = min(h, w)
        cy, cx = h // 2, w // 2
        y1, y2 = cy - crop_size // 2, cy + crop_size // 2
        x1, x2 = cx - crop_size // 2, cx + crop_size // 2
        cropped = frame[y1:y2, x1:x2].copy()

        # Adjust landmarks to cropped coordinate space
        lm_adjusted = lm_row.copy()
        for i in range(33):
            col_x, col_y = f"{landmarks_df.columns[3+i*3]}", f"{landmarks_df.columns[4+i*3]}"
            if col_x in lm_adjusted:
                lm_adjusted[col_x] = (lm_adjusted[col_x] - x1) * (vid_w / crop_size)
                lm_adjusted[col_y] = (lm_adjusted[col_y] - y1) * (vid_h / crop_size)

        resized_vid = cv2.resize(cropped, (vid_w, vid_h))

        # Draw skeleton on cropped & resized video
        # We can draw directly on resized_vid using scaled coordinates
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            col_x1, col_y1, col_v1 = f"{landmarks_df.columns[3+p1_idx*3]}", f"{landmarks_df.columns[4+p1_idx*3]}", f"{landmarks_df.columns[5+p1_idx*3]}"
            col_x2, col_y2, col_v2 = f"{landmarks_df.columns[3+p2_idx*3]}", f"{landmarks_df.columns[4+p2_idx*3]}", f"{landmarks_df.columns[5+p2_idx*3]}"
            v1 = lm_row[col_v1]
            v2 = lm_row[col_v2]
            if v1 > 0.3 and v2 > 0.3:
                pt1 = (int((lm_row[col_x1] - x1) * vid_w / crop_size), int((lm_row[col_y1] - y1) * vid_h / crop_size))
                pt2 = (int((lm_row[col_x2] - x1) * vid_w / crop_size), int((lm_row[col_y2] - y1) * vid_h / crop_size))
                if 0 <= pt1[0] < vid_w and 0 <= pt1[1] < vid_h and 0 <= pt2[0] < vid_w and 0 <= pt2[1] < vid_h:
                    color = (0, 255, 255) if (p1_idx in (11, 13, 15) and p2_idx in (11, 13, 15)) else (0, 200, 200)
                    cv2.line(resized_vid, pt1, pt2, color, 2)

        for p_idx in range(33):
            col_x, col_y, col_v = f"{landmarks_df.columns[3+p_idx*3]}", f"{landmarks_df.columns[4+p_idx*3]}", f"{landmarks_df.columns[5+p_idx*3]}"
            if lm_row[col_v] > 0.3:
                pt = (int((lm_row[col_x] - x1) * vid_w / crop_size), int((lm_row[col_y] - y1) * vid_h / crop_size))
                if 0 <= pt[0] < vid_w and 0 <= pt[1] < vid_h:
                    color = (0, 255, 0) if p_idx in (11, 13, 15) else (0, 140, 255)
                    rad = 5 if p_idx in (11, 13, 15) else 3
                    cv2.circle(resized_vid, pt, rad, color, -1)

        # Draw overlays on video
        cv2.putText(resized_vid, "Video + MediaPipe Pose (33 joints)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(resized_vid, f"Time: {curr_time_s:5.2f}s | Frame: {curr_frame_idx}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(resized_vid, "Activity: left_arm_01", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Plot CSI amplitude and time cursor
        ax.clear()
        # Find closest CSI frame
        csi_idx = (np.abs(csi_ts - curr_time_s)).argmin()
        window_start_t = max(0.0, curr_time_s - 5.0)
        window_end_t = min(csi_ts[-1], curr_time_s + 5.0)
        mask_t = (csi_ts >= window_start_t) & (csi_ts <= window_end_t)

        # Mean amplitude curve
        mean_amp = np.mean(csi_matrix[mask_t, :], axis=1)
        ax.plot(csi_ts[mask_t], mean_amp, color="#00ffff", lw=2, label="Mean CSI Amplitude")
        ax.axvline(curr_time_s, color="#ff3366", lw=2.5, linestyle="--", label=f"Current Cursor ({curr_time_s:.2f}s)")

        # Plot subcarrier bands (sample tones)
        for tone_i, col in [(20, "#ffaa00"), (50, "#00ff88"), (100, "#ff88ff")]:
            ax.plot(csi_ts[mask_t], csi_matrix[mask_t, tone_i], color=col, alpha=0.4, lw=1, label=f"Tone #{tone_i}")

        ax.set_facecolor("#1a1a24")
        fig.patch.set_facecolor("#121218")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(f"Wi-Fi CSI Signal (VHT 224 tones) | Frame ID: {csi_frame_ids[csi_idx]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_xlim(window_start_t, window_end_t)
        ax.grid(True, color="#333344", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", facecolor="#222230", edgecolor="#444455", labelcolor="white", fontsize=8)

        # Convert matplotlib canvas to OpenCV image
        fig.canvas.draw()
        plot_rgba = np.asarray(fig.canvas.buffer_rgba())
        plot_rgb = plot_rgba[:, :, :3]
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)
        plot_resized = cv2.resize(plot_bgr, (plot_w, plot_h))

        # Combine side-by-side
        combined = np.hstack([resized_vid, plot_resized])
        writer.write(combined)

    writer.release()
    cap.release()
    plt.close(fig)
    print(f"[OK] Saved synchronized demo video: {out_mp4}")


def create_snapshot_figure(
    video_path: Path,
    landmarks_csv: Path,
    csi_csv: Path,
    out_png: Path,
):
    print("Generating composite snapshot figure for report...")
    csi_matrix, csi_ts, csi_frame_ids = load_csi_matrix(csi_csv)
    landmarks_df = pd.read_csv(landmarks_csv)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Pick 4 representative timestamps during the arm raising cycle
    sample_times = [16.0, 18.0, 20.0, 22.0]
    titles = [
        "(A) Arm at Rest (t = 16.0s)",
        "(B) Arm Raising (t = 18.0s)",
        "(C) Peak Arm Height (t = 20.0s)",
        "(D) Arm Lowering (t = 22.0s)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="#0e1117")
    fig.suptitle(
        "Xác Thực Đồng Bộ Tư Thế Video ↔ Tín Hiệu CSI (Phiên left_arm_01)",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    for col_i, (t_sec, title) in enumerate(zip(sample_times, titles)):
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        crop_size = min(h, w)
        cy, cx = h // 2, w // 2
        y1, y2 = cy - crop_size // 2, cy + crop_size // 2
        x1, x2 = cx - crop_size // 2, cx + crop_size // 2
        cropped = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

        closest_lm_idx = (landmarks_df["video_time_s"] - t_sec).abs().idxmin()
        lm_row = landmarks_df.iloc[closest_lm_idx]

        # Draw skeleton on cropped RGB
        vid_h, vid_w = cropped.shape[:2]
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            col_x1, col_y1, col_v1 = f"{landmarks_df.columns[3+p1_idx*3]}", f"{landmarks_df.columns[4+p1_idx*3]}", f"{landmarks_df.columns[5+p1_idx*3]}"
            col_x2, col_y2, col_v2 = f"{landmarks_df.columns[3+p2_idx*3]}", f"{landmarks_df.columns[4+p2_idx*3]}", f"{landmarks_df.columns[5+p2_idx*3]}"
            if lm_row[col_v1] > 0.3 and lm_row[col_v2] > 0.3:
                pt1 = (int(lm_row[col_x1] - x1), int(lm_row[col_y1] - y1))
                pt2 = (int(lm_row[col_x2] - x1), int(lm_row[col_y2] - y1))
                color = (0, 255, 255) if (p1_idx in (11, 13, 15) and p2_idx in (11, 13, 15)) else (255, 200, 0)
                cv2.line(cropped, pt1, pt2, color, 3)

        for p_idx in range(33):
            col_x, col_y, col_v = f"{landmarks_df.columns[3+p_idx*3]}", f"{landmarks_df.columns[4+p_idx*3]}", f"{landmarks_df.columns[5+p_idx*3]}"
            if lm_row[col_v] > 0.3:
                pt = (int(lm_row[col_x] - x1), int(lm_row[col_y] - y1))
                color = (0, 255, 0) if p_idx in (11, 13, 15) else (255, 100, 0)
                rad = 6 if p_idx in (11, 13, 15) else 4
                cv2.circle(cropped, pt, rad, color, -1)

        ax_top = axes[0, col_i]
        ax_top.imshow(cropped)
        ax_top.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax_top.axis("off")

        # Bottom plot: CSI around that timestamp
        ax_bot = axes[1, col_i]
        t_span = (csi_ts >= t_sec - 2.5) & (csi_ts <= t_sec + 2.5)
        mean_amp = np.mean(csi_matrix[t_span, :], axis=1)
        ax_bot.set_facecolor("#181b24")
        ax_bot.plot(csi_ts[t_span], mean_amp, color="#00ffcc", lw=2, label="Mean CSI")
        ax_bot.plot(csi_ts[t_span], csi_matrix[t_span, 30], color="#ff9900", alpha=0.5, lw=1, label="Tone 30")
        ax_bot.plot(csi_ts[t_span], csi_matrix[t_span, 100], color="#ff33aa", alpha=0.5, lw=1, label="Tone 100")
        ax_bot.axvline(t_sec, color="#ff3366", lw=2, linestyle="--", label="Cursor")
        ax_bot.tick_params(colors="white")
        ax_bot.xaxis.label.set_color("white")
        ax_bot.yaxis.label.set_color("white")
        ax_bot.set_xlabel("Time (s)", color="white", fontsize=9)
        if col_i == 0:
            ax_bot.set_ylabel("CSI Amplitude", color="white", fontsize=9)
            ax_bot.legend(loc="upper right", facecolor="#222230", edgecolor="#444455", labelcolor="white", fontsize=7)
        ax_bot.grid(True, color="#333344", linestyle=":", alpha=0.5)

    cap.release()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved snapshot figure: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Render synchronized Pose + CSI video and snapshot")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--csi", type=Path, default=DEFAULT_CSI)
    parser.add_argument("--out-mp4", type=Path, default=DEFAULT_OUTPUT_MP4)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    args = parser.parse_args()

    create_snapshot_figure(args.video, args.landmarks, args.csi, args.out_png)
    create_demo_video(args.video, args.landmarks, args.csi, args.out_mp4, start_sec=15.0, duration_sec=10.0)


if __name__ == "__main__":
    main()
