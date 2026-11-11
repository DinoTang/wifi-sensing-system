# Lan2 video/CSI audit — sequential full decode and 1-fps visual review

Date: 2026-11-10

## Initial sparse pass

The first pass sought to three visible-clock anchors and the start, midpoint,
and end of each of the six CSI captures (21 sparse frames total). It did not
decode the video from beginning to end. The follow-up below completes that
decode and documents a sequential visual review of 1-fps samples. Only Lan2
was processed; Lan3 was not touched, and no model training was run.

## Visible clock fit

The laptop clock is readable in the video at:

| Video time | Displayed clock |
|---:|---|
| 0 s | 2026-11-05 11:07:48.138 +07:00 |
| 5 s | 2026-11-05 11:07:53.136 +07:00 |
| 614.8 s | 2026-11-05 11:18:02.938 +07:00 |

A least-squares fit to the three displayed timestamps gives a slope of
1.000001614, maximum absolute residual 1.009 ms, and endpoint drift about
0.992 ms over 614.8 seconds. The 12,815-row `session-clock.csv` covers
2026-11-05 11:07:13.011 through 11:18:02.122 +07:00 and bridges the receiver
host clock to the displayed clock. All six CSI captures fall inside its
recorded interval.

## CSI intervals mapped onto the round video

| Activity | CSI frames | Video time for CSI capture (s) | Approx. capture duration (s) |
|---|---:|---:|---:|
| empty | 573 | 66.516–126.363 | 59.847 |
| standing | 574 | 159.200–218.851 | 59.651 |
| left_arm | 564 | 252.078–311.316 | 59.239 |
| right_arm | 557 | 343.988–403.439 | 59.451 |
| walking | 557 | 436.451–496.215 | 59.764 |
| mixed | 561 | 528.928–588.796 | 59.868 |

Each mapped interval is inside the 615.833-second video. At the mapped
capture start/mid/end, the sparse frames visually agree with the session
labels: empty room; standing; arm-raise activities; walking; and mixed
movement. The contact sheet also includes the three clock-anchor frames.

## Interpretation

This is strong independent evidence that the Lan2 receiver timestamps and
round-video timeline are aligned. It is substantially stronger than using
session filenames, modification times, or an activity-motion correlation.
It supports retaining `sync_ok=1` for the six Lan2 sessions and using these
CSI/video mappings to build pose windows.

The sparse visual check does not inspect every video frame or every pose
track sample. The clock fit is well within the project's 100 ms threshold,
but the visual anchors are at 0, 5, and 614.8 seconds; only the endpoints are
widely separated. Keep that sampling limitation explicit rather than claiming
the whole video was manually reviewed frame by frame.

## Artifacts

- `lan2_sparse_clock_and_csi_samples.jpg`: three clock anchors and 18
  sparse frames at the starts, midpoints, and ends of the six CSI intervals.
- `lan2_csi_video_time_mapping.csv`: absolute CSI time and mapped video
  intervals for each activity.

## Sequential full-decode and visual review — 2026-11-10

The single round video `CVAL0593.MOV` was decoded sequentially from start to
end in one low-priority process. OpenCV/FFmpeg and OpenCV image processing
were restricted to one thread. The process used a duty-cycle throttle
targeting 28% of one logical CPU; CPU and working-set measurements were
recorded every 30 seconds. Decoded frames were discarded immediately after
processing; only resized 320-pixel-wide samples were buffered, in batches of
60, to avoid retaining the video in memory.

| Measure | Result |
|---|---:|
| Video metadata | 30 fps, 1920 × 1080, 18,475 frames, 615.833 s |
| Decoded | 18,475 / 18,475 frames |
| Visual review samples | 616 frames, one per second from t=0 through t=615 |
| Contact sheets | 11, split into sequential one-minute ranges |
| Elapsed time | 3,163 s (52.7 min) |
| Decoder process CPU | 28.0% average of one logical CPU (26.6–30.9%) |
| Whole-system CPU sampled | 16.2% average (12.7–26.4%) |
| Decoder peak working set | 232.2 MB |

These are distinct measures: the 28% figure is the decoder process as a share
of one logical CPU, while whole-system CPU is the sampled machine-wide
percentage. They are measured averages/ranges, not a hard system-wide cap.
The resource log is `fullpass_1fps/lan2_fullpass_resource_log.csv`; contact
sheets are `fullpass_1fps/lan2_timeline_1fps_01.jpg` through
`lan2_timeline_1fps_11.jpg`.

### Activity-by-activity observations

| Activity interval in video | Visual review |
|---|---|
| `empty`, 66.516–126.363 s | Room remains empty across the 1-fps samples. |
| `standing`, 159.200–218.851 s | Participant stands in place throughout; there are brief arm/upper-body movements early in the interval, followed by mostly still standing. |
| `left_arm`, 252.078–311.316 s | Repeated single-arm raises are visible; the sequence is consistent with the activity label. |
| `right_arm`, 343.988–403.439 s | Repeated single-arm raises are visible; the sequence is consistent with the activity label. |
| `walking`, 436.451–496.215 s | Participant repeatedly changes position and walks around the room. |
| `mixed`, 528.928–588.796 s | Varied reaching/arm motions, turns, and position changes are visible, consistent with mixed activity. |

The unsupervised gaps between CSI captures contain setup, pauses, and
additional movements; they should not be treated as labeled CSI activity
windows. At 1-fps sampling, exact sub-second action boundaries cannot be
determined, and individual poses in every decoded frame were not inspected.
The full decode establishes that the video is readable through its reported
end; the visual review is a regular sample, not a frame-by-frame human review.

The activity intervals remain inside the clock-fitted timeline and their
sampled contents match the labels. This independently reinforces the existing
CSI-to-video clock mapping. It does not alter the approximately 1 ms clock-fit
error, and it does not establish exact pose labels for every frame.
