# Task 3 handoff — Final_CSI rebuild

## Result in one sentence

The Task 3 data pipeline is rebuilt from `Final_CSI`, and a CSI/pseudo-pose baseline has been trained and evaluated. The current recordings do **not** establish reliable CSI-driven moving-pose prediction: the training-mean pose baseline beats Ridge on both rounds and in both held-out-round tests. Do not present the selected coordinate output as a successful CSI pose estimate.

## Rebuilt data and synchronization

- Source: `D:\CSI values\Final_CSI`; 18 CSI captures, six activities per round.
- Parsed CSI: 18 per-session CSVs under `Lan1/raw_csv`, `Lan2/raw_csv`, and `Lan3/raw_csv`.
- Video pose pseudo-labels: 18 segment-specific landmark CSVs; 10,932 inferred frames total.
- Windowing: 2-second windows, 0.2-second step, 20 CSI frames × 224 features per window.
- Windows created: Lan1 1,043; Lan2 991; Lan3 983.
- **Lan1:** all six offsets remain estimates from the reported video start; `sync_ok=0`. Its training run is diagnostic only and excluded from verified training.
- **Lan2:** six sessions use three visible laptop-clock anchors (video t=0, 5, and 614.8 s) and the session host/display clock log. A fit to the displayed timestamps has slope 1.000001614, maximum anchor residual 1.009 ms, and endpoint drift about 0.992 ms. The entire video was decoded (18,475/18,475 frames) and 616 one-fps samples were reviewed; activity content agrees broadly with the mapped labels.
- **Lan3:** six sessions use five visible phone-clock anchors. The fitted slope is 1.000019842; maximum anchor residual is 6.774 ms, drift across the anchor span is 10.891 ms (about 11.5 ms extrapolated over the full clip). Its session host/display clock log has slope 0.999999709 and maximum absolute difference 0.999 ms. The entire video was decoded (17,397/17,397 frames) and 580 one-fps samples were reviewed; activity content agrees broadly with the mapped labels.
- Lan2 and Lan3 are marked `sync_ok=1`; combined verified training contains 12 sessions and 1,974 windows. Lan1's 1,043 windows are kept separately as diagnostic data.

The published clock evidence is in [clock_quality.json](./clock_quality.json), [video_clock_anchors.csv](./video_clock_anchors.csv), and the privacy-minimized [sync_summary_public.csv](./sync_summary_public.csv). The original per-session sync report, which contains local file paths, is not distributed. The Lan2/Lan3 visual reviews use one sample per second after a complete decode; they do not represent manual review of every frame or pose. See the [Lan2 audit](../../activity-round-experiment/Lan2-video-audit/LAN2_VIDEO_AUDIT.md) and [Lan3 audit](../../activity-round-experiment/Lan3-video-audit/LAN3_VIDEO_AUDIT.md).

## Existing Người 2 `pose_windows_dataset`

The original workspace contained `signal-processing/data/pose/data/pose_windows_dataset`. I inspected and used those files without modifying them; their filesystem modification times are November 6, 2026, before this Final_CSI rebuild. The input folder contained six existing NPZ sessions, the session index, sync report, windows index, and summary. These source data files are not included in the repository. I cannot prove byte-for-byte identity with the original ZIP because that archive/hash is not present here.

Important caveat: this dataset is **not sync-verified**. Its `sync_report.csv` marks offsets weak or unreliable, every row of `windows_summary.csv` has `sync_ok=0` and `offset_s=0.0`, and each NPZ has `sync_ok=0`. I therefore used the dataset as-is for a diagnostic run only, not as verified training data.

- Diagnostic run: `nguoi2-pose-windows-dataset/training-diagnostic/` (6 sessions, 1,054 windows).
- Mean-pose baseline: normalized image-diagonal error 0.03745; PCK@0.05 0.69882.
- Ridge: error 0.04858; PCK@0.05 0.69875.
- Person presence: accuracy 0.69829; balanced accuracy 0.42593; F1 0.82235.
- Inference smoke test: 201 predictions at `nguoi2-pose-windows-dataset/training-diagnostic/inference_left_arm_01.csv`.

All of these metrics are diagnostic only because synchronization is unverified. The source dataset remains untouched; training outputs are stored separately.

## Evaluation

Coordinate error is normalized by the image diagonal; PCK uses a 0.05 image-diagonal threshold. Leave-one-session-out results:

| Dataset | Sessions / windows | Mean-pose error / PCK | Ridge error / PCK | Person-presence accuracy / balanced accuracy / F1 |
|---|---:|---:|---:|---:|
| Lan1 (diagnostic; unverified sync) | 6 / 1,043 | 0.0359 / 0.7060 | 0.0574 / 0.6974 | 0.6750 / 0.4131 / 0.8060 |
| Lan2 (verified) | 6 / 991 | 0.0409 / 0.7039 | 0.0557 / 0.7148 | 0.7548 / 0.4675 / 0.8603 |
| Lan3 (verified) | 6 / 983 | 0.0457 / 0.7053 | 0.0879 / 0.4333 | 0.7202 / 0.4442 / 0.8374 |
| Lan2 + Lan3 (verified) | 12 / 1,974 | 0.0423 / 0.7143 | 0.0579 / 0.5519 | 0.7791 / 0.5028 / 0.8744 |

Two additional independent-round tests were run. Training on Lan2 and testing on Lan3 gave mean-pose error 0.0462 / PCK 0.7198, versus Ridge 0.1143 / 0.2083. Training on Lan3 and testing on Lan2 gave mean-pose error 0.0389 / PCK 0.7128, versus Ridge 0.1271 / 0.1663. The coordinate model selected from training-only validation was `training_mean` in both directions. The mean-pose baseline does not use CSI to move the predicted joints.

Presence F1/accuracy should not be read alone: balanced accuracy is much lower and varies substantially across rounds, indicating class imbalance and weak round-to-round generalization.

## Inference and visual check

- Final verified model: [pose_baseline.joblib](./combined-verified/training/pose_baseline.joblib) with privacy-minimized [model metadata](./combined-verified/training/model_metadata_public.json) (model version `20261110-020608-922645`).
- Smoke inference wrote 184 windows to a local CSV output (not distributed with the repository).
- CPU model-only latency over 100 windows: median 0.493 ms/window; p95 1.025 ms/window. This excludes CSI parsing/window creation, model loading, and CSV I/O.
- The local `left_arm_03_overlay.mp4` shows held-out Lan3 video pseudo-labels against the selected Lan2-trained mean-pose baseline. The video is not distributed; the published [skeleton comparison figure](./combined-verified/training/heldout_walking_pose_examples.png) is a compact illustration of the same limitation, not evidence that CSI predicts the moving pose.

## Interpretation and limits

1. The input labels are MediaPipe pose estimates from video, not motion-capture ground truth.
2. Recordings cover one participant, one room/camera setup, and only two verified rounds. Leave-one-session-out results are not a broad external validation.
3. The present feature set and data do not show that CSI predicts changing joint coordinates better than the mean pose. A pose-inference claim is therefore not supported yet.
4. No ground-truth gender or physical height labels are present. These outputs do not classify gender or estimate a person's real-world height.

## Published artifacts and local-only data

The repository includes this report, the clock-quality summary, visible clock
anchors, a privacy-minimized synchronization table, the verified model,
compact metric tables, and figures. The source `Final_CSI` recordings,
per-round parsed CSVs/landmarks/windows, full out-of-fold prediction tables,
and video overlays remain local and are not included. The cross-round folders
retain their compact `metrics.csv` tables; the larger prediction tables are
not distributed.

## Re-run training

Run these commands from the project root after preparing the corresponding
verified windows from CSI/video source data:
(`D:\CSI values\Latest_edition\latest_edition` was the original workstation
path; use the root of your own clone instead).

```powershell
python machine-learning/src/train_pose_baseline.py --window-dir machine-learning/results/pose-sync-check/final-csi-rebuild/Lan2/windows --output-dir machine-learning/results/pose-sync-check/final-csi-rebuild/Lan2/training --evaluation-mode session
python machine-learning/src/train_pose_baseline.py --window-dir machine-learning/results/pose-sync-check/final-csi-rebuild/Lan3/windows --output-dir machine-learning/results/pose-sync-check/final-csi-rebuild/Lan3/training --evaluation-mode session
python machine-learning/src/train_pose_baseline.py --window-dir machine-learning/results/pose-sync-check/final-csi-rebuild/combined-verified/windows --output-dir machine-learning/results/pose-sync-check/final-csi-rebuild/combined-verified/training --evaluation-mode session
```

Lan1 is intentionally excluded from verified runs. To reproduce its diagnostic run only, explicitly add `--allow-unverified-sync` and retain the diagnostic-only warning.
