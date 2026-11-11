# Lan1 video/CSI audit — sequential pass

Date: 2026-11-10

## Purpose and safety scope

This pass checked whether Lan1's existing per-activity videos can provide a
better CSI/video alignment than the original whole-video estimate. It read
the existing parsed CSI and decoded the six short activity clips one at a
time. No video batch parallelism, model training, or source-data edits were
performed. Sparse frames and derived audit outputs are stored in this folder.

## What the project documents establish

- `signal-processing/docs/capture_checklist.md` requires a clock synchronized
  to the CSI host to be visible in the video, plus a shared gesture/clap where
  possible. A clean `screen-clock.csv` only verifies the host/display clock;
  it does not, by itself, reveal when the camera recording began.
- `signal-processing/cong_viec_nguoi_2.md` specifies checking video/CSI anchors,
  filtering while preserving motion, extracting timestamped pose labels, and
  rejecting poorly synchronized windows. It explicitly distinguishes
  session-level activity labels from frame-level pose labels.
- `signal-processing/docs/video_pose_spec.md` and
  `signal-processing/docs/report_nguoi2.md` require whole-session splits before
  making overlapping windows, and warn that the initial one-repeat-per-class
  collection is for validating the pipeline, not proving generalization.
- The Lan1 `sessions.xlsx` only maps six activities to their session folders;
  it has no video start time or action transition timestamps. Each
  `session-notes.txt` says `synchronization_status=not_yet_verified` and does
  not record event-level movement times.
- A separate filled-in form, `D:\CSI values\Phieu_bo_sung_thong_tin_CSI_video.docx`,
  adds a **reported** common start: the recorder Enter time and the phone
  record button were both noted as 02:52:28 on the laptop clock. It maps the
  continuous video's t=0 to that time and lists CSI starts at video t=00:35,
  02:13, 03:50, 05:28, 07:05, and 08:43 for the six activities. On
  2026-11-11, the operator clarified that they intentionally pressed Enter
  and the phone's record button at essentially the same time to start both
  procedures. This is a useful first-hand synchronization anchor and no
  known delay is being reported. However, the press times were not measured
  against each other, and no shared event or readable synchronized clock in
  the video independently records the first captured frame.

## Facts verified from the Lan1 files

- All six activity folders have a matching approximately 60-second video,
  `screen-clock.csv`, session notes, and parsed CSI capture.
- The six videos decoded successfully, sequentially: five are 59.033 s and
  `mixed` is 60.000 s. CSI spans are 59.353–59.861 s, at approximately
  9.9–10 Hz.
- The screen-clock logs contain 967–969 rows and span approximately
  97.28–97.41 s. CSI begins 34.38–34.96 s after each clock log starts, and
  ends about 3.02–3.10 s before the log ends. This proves the CSI/host-clock
  timeline is present, but not the video start position within that timeline.
- A seven-frame sparse check of the Lan1 `left_arm` clip shows a person and
  an analog wall clock in the background. The clock face is obscured and its
  time cannot be read or tied to the CSI host clock; it is not a verified
  synchronization anchor.
- All 18 CSI session summaries from the existing parsed tables were already
  checked separately; the six Lan1 activity captures contain 569–601 VHT
  frames with no incomplete CSI frames.

## Motion-correlation diagnostic

For each activity, the per-tone CSI amplitude change was compared with
low-resolution video frame differences, both over the full frame and over a
person-centered region. The diagnostic decoded one video at a time, smoothed
the series by 0.5 s, and searched candidate mappings of video time at CSI
time zero over ±15 s. The table reports the best ROI correlation and the
best candidate more than one second away. Candidate lags are not accepted as
offsets.

| Activity | CSI frames | Video duration (s) | Best candidate lag (s) | Best r | Runner-up lag (s) | Runner-up r |
|---|---:|---:|---:|---:|---:|---:|
| empty | 569 | 59.033 | -14.6 | 0.106 | -12.6 | 0.102 |
| left_arm | 596 | 59.033 | +3.9 | 0.279 | -1.4 | 0.259 |
| mixed | 570 | 60.000 | +0.2 | 0.317 | -6.2 | 0.281 |
| right_arm | 572 | 59.033 | +0.8 | 0.343 | -1.6 | 0.308 |
| standing | 581 | 59.033 | +12.6 | 0.074 | +11.5 | 0.064 |
| walking | 601 | 59.033 | -3.8 | 0.616 | +1.1 | 0.570 |

`walking` has a moderate correlation, but its runner-up is close (peak ratio
about 1.08). The remaining sessions have weak correlations or similarly
competitive peaks. Best candidate lags vary across activities; the
correlation method therefore cannot establish a common camera/CSI offset or
the required <=100 ms alignment. The result is consistent with the earlier
warning that activity motion alone is an ambiguous synchronization anchor.

## Conclusion and next step

Lan1 is **not missing usable CSI/video data**. Its `sync_ok=0` status means only
that frame-level CSI-to-video pose alignment has not been independently
verified; the per-activity labels and CSI remain usable for activity-level
experiments. There is a useful, documented **approximate alignment** in the
filled-in collection form, strengthened by the operator's first-hand
clarification that the two recording procedures were started together. This
is stronger evidence than an offset inferred from activity correlation and
supports using the reported mapping for review and exploratory processing.
It still does not independently measure the relation between the two button
presses and the phone's first captured frame, so it does not demonstrate a
<=100 ms offset or justify changing `sync_ok` by itself. This does not mean a
delay is known to have occurred; the residual uncertainty simply was not
measured.

To promote Lan1 pose windows to verified training data, measure or otherwise
independently verify the remaining start-time uncertainty (for example, with
a visible shared clock/event or a known phone-to-laptop clock comparison),
then check the alignment near both the start and end of the recordings.

Detailed numeric output: `lan1_all_sessions_sync_diagnostic.csv` and
`left_arm_sync_motion_correlation.csv`. Sparse visual samples from the one
manually inspected `left_arm` clip are in this folder.

## Sequential visual review — 2026-11-10

After the initial correlation pass, all six per-activity clips and the
separate round master video were decoded sequentially. The clips were decoded
to end-of-file and sampled into contact sheets at one frame per second. Those
contact sheets were then visually reviewed in order. This is a complete
decode/integrity pass plus a 1-fps visual review, **not** a manual inspection
of every decoded frame.

| Video | Decoded frames / duration | Visual sample review |
|---|---:|---|
| `empty` | 1,771 / 59.033 s | 60 samples; room appears empty |
| `left_arm` | 1,771 / 59.033 s | 60 samples; participant repeatedly raises an arm |
| `mixed` | 1,800 / 60.000 s | 60 samples; varied reaching, turning, and arm movement |
| `right_arm` | 1,771 / 59.033 s | 60 samples; repeated one-arm raises |
| `standing` | 1,771 / 59.033 s | 60 samples; participant mostly stands in place |
| `walking` | 1,771 / 59.033 s | 60 samples; participant changes position and walks in the room |
| `RNZD8449.MOV` (round master) | 17,811 / 593.700 s | 594 samples; continuous round timeline |

The master-video samples show an empty-room opening, a person entering at
roughly 110 seconds, and later standing, arm-raise, and movement periods.
Because these are one-second samples, they cannot establish exact action
boundaries or rule out brief events between samples. No readable, independent
camera/CSI clock anchor or timed shared event was established by this pass.
The master video therefore adds useful context for reviewing the round but
does not, by itself, resolve the Lan1 CSI/video offset or justify changing
`sync_ok=0`.

Contact sheets are stored under `master_review/` for the master video, and
beside this report for each per-activity clip (`lan1_*_fullpass_1fps.jpg`).
