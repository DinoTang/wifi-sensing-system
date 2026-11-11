# Lan3 video/CSI audit — sequential full decode and 1-fps visual review

Date: 2026-11-10

## Clock evidence and mapped activity intervals

Lan3 video `UCZS0429.MOV` is 579.900 seconds long at 30 fps. The existing
clock audit read five visible phone-clock anchors at video times 30, 120, 240,
480, and 578.9 seconds. The fitted slope is 1.000019842; maximum absolute
anchor residual is 6.774 ms, and drift across the 548.9-second anchor span is
10.891 ms (about 11.5 ms extrapolated over the full clip). The host/display
clock log independently differs by at most 0.999 ms.

The six CSI sessions map to the following approximate video start times under
the existing fitted clock:

| Activity | Approx. video start (s) | Visual samples |
|---|---:|---|
| `empty` | 44.453 | 60 |
| `standing` | 136.995 | 60 |
| `left_arm` | 229.607 | 60 |
| `right_arm` | 322.029 | 60 |
| `walking` | 414.532 | 60 |
| `mixed` | 507.056 | 60 |

These are session mapping times from the clock fit, not frame-accurate
movement onset annotations. The activities last approximately one minute.

## Sequential full decode and visual review

The 1920 × 1080 source was decoded sequentially from start to end with one
OpenCV/FFmpeg thread. The process ran at below-normal priority and used a
duty-cycle throttle targeting 28% of one logical CPU. Each decoded frame was
discarded after use; only resized 320-pixel-wide samples were buffered in
groups of 60.

| Measure | Result |
|---|---:|
| Video metadata | 30 fps, 17,397 frames, 579.900 s |
| Decoded | 17,397 / 17,397 frames |
| Visual review samples | 580 frames, one per second from t=0 through t=579 |
| Contact sheets | 10, split into sequential one-minute ranges |
| Elapsed time | 3,328 s (55.5 min) |
| Decoder process CPU | 28.0% average of one logical CPU (27.6–28.4%) |
| Whole-system CPU sampled | 16.3% average (14.2–20.7%) |
| Decoder peak working set | 232.6 MB |

These measurements describe this decoder process and sampled machine load;
the 28% figure is not a hard cap on total system CPU. The resource log is
`fullpass_1fps/lan3_fullpass_resource_log.csv`; the contact sheets are
`fullpass_1fps/lan3_timeline_1fps_01.jpg` through
`fullpass_1fps/lan3_timeline_1fps_10.jpg`.

### Activity-by-activity observations

| Activity | 1-fps visual review |
|---|---|
| `empty` | The mapped interval shows the room without a visible participant, consistent with the activity label. |
| `standing` | The participant enters/settles near the start of the mapped interval, then is mostly stationary. Sparse samples do not establish exact entry or action-transition frames. |
| `left_arm` | Repeated arm-raise movements are visible during the mapped interval, broadly consistent with the label. |
| `right_arm` | Repeated arm-raise movements are visible during the mapped interval, broadly consistent with the label. |
| `walking` | The participant changes position and walks during the mapped interval. |
| `mixed` | The participant shows varied arm and body movements during the mapped interval. |

The intervals between CSI sessions include pauses and other visible activity.
Those portions should not be treated as labeled CSI windows. A one-second
sample cadence can miss brief events and cannot supply frame-accurate pose
labels. Although every encoded frame decoded successfully, only the 580
regularly spaced samples were manually reviewed; this is not human review of
every frame or pose.

The sampled content is broadly consistent with all six session labels. This
supports the existing clock-based video mapping but does not replace the five
clock anchors, change the fitted offset, or prove that every CSI/pose window
is perfectly aligned.
