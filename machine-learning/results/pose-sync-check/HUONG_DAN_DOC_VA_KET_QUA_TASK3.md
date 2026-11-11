# Hướng dẫn đọc tài liệu và kết quả nghiên cứu Task 3

Tài liệu này là bản đồ để thành viên mới có thể hiểu nghiên cứu CSI-to-pose
mà không phải mở lần lượt mọi CSV, model và báo cáo. Tài liệu trả lời bốn câu
hỏi:

1. Nên đọc những tài liệu nào, theo thứ tự nào?
2. Các thư mục và file kết quả quan trọng dùng để làm gì?
3. Nhóm đã chạy được những gì và kết quả hiện nói lên điều gì?
4. Làm sao xem hình, chạy lại một bước hoặc thử inference?

> **Kết luận quan trọng:** pipeline offline từ capture CSI đã lưu đến CSV dự
> đoán đã chạy được. Tuy nhiên, kết quả hiện tại **chưa chứng minh CSI dự đoán
> được chuyển động pose đáng tin cậy**. Đây là nghiên cứu/baseline phục vụ
> prototype, không phải một bộ đếm người, nhận diện giới tính/chiều cao hay
> hệ thống CSI-to-pose đã được nghiệm thu.

## 1. Đọc nhanh trong 5 phút

Nếu chỉ có ít thời gian, hãy đọc theo thứ tự:

1. [Báo cáo bàn giao chi tiết](./final-csi-rebuild/BAO_CAO_BAN_GIAO_TASK_NGUOI_3.md)
   — số liệu, đồng bộ, thí nghiệm, giới hạn và checklist công việc.
2. [Bản handoff kỹ thuật](./final-csi-rebuild/TASK3_HANDOFF.md) — cấu trúc
   dữ liệu đã dựng lại, cách đánh giá và lệnh chạy lại.
3. [Kết quả phân loại hoạt động](../activity-round-experiment/round-heldout-comparison/ACTIVITY_CLASSIFICATION_RESULTS.md)
   — thí nghiệm khác với CSI-to-pose, dùng để xem CSI có dấu hiệu phân loại
   nhãn hoạt động hay không.

Các bài audit video theo lượt nằm trong
[Lan1](../activity-round-experiment/Lan1-video-audit/LAN1_VIDEO_AUDIT.md),
[Lan2](../activity-round-experiment/Lan2-video-audit/LAN2_VIDEO_AUDIT.md) và
[Lan3](../activity-round-experiment/Lan3-video-audit/LAN3_VIDEO_AUDIT.md).
Chúng giải thích cơ sở và giới hạn của việc ghép thời gian video với CSI.

## 2. Bức tranh tổng thể: dữ liệu đi qua những bước nào?

```text
Capture CSI + video
        │
        ├── Parse CSI ──> bảng CSI theo frame
        ├── Kiểm tra clock/video ──> trạng thái đồng bộ theo phiên
        └── Trích pose từ video ──> landmarks/pseudo-labels
                                  │
                    Ghép theo thời gian và chia cửa sổ
                                  │
                    Huấn luyện/evaluate baseline
                                  │
         CSV metric, model, hình so sánh và inference offline
```

Pose trong video được ước lượng tự động (MediaPipe), nên là **pseudo-label**,
không phải phép đo khớp chuẩn từ motion-capture. Đồng bộ tốt giúp so sánh đúng
thời điểm CSI và video; nó không tự làm nhãn pose thành ground truth và cũng
không đảm bảo mô hình học được quan hệ CSI-pose.

Hai hướng thí nghiệm cần phân biệt:

| Hướng | Câu hỏi | Đầu ra chính |
|---|---|---|
| CSI-to-pose | CSI có giúp dự đoán trạng thái có người và 17 điểm khớp 2D tốt hơn baseline không? | Metric pose/presence, model, dự đoán theo cửa sổ |
| Phân loại hoạt động | CSI có phân biệt được các nhãn phiên `empty`, `standing`, `left_arm`, `right_arm`, `walking`, `mixed` không? | Metric phân loại hoạt động và confusion matrix |

Phân loại hoạt động **không đồng nghĩa** với xác định chính xác thời điểm một
hành động bắt đầu/kết thúc, dự đoán tọa độ pose hoặc nhận diện người.

## 3. Tài liệu nào dùng để làm gì?

| Tài liệu | Dùng khi muốn biết | Cách đọc |
|---|---|---|
| [README của machine-learning](../../README.md) | Cách chạy các pipeline ML khác nhau, lệnh train/inference và test | Tài liệu hướng dẫn kỹ thuật; trong đó có cả thí nghiệm Intel/hoạt động/pose, không phải tất cả đều cùng một dataset |
| [BAO_CAO_BAN_GIAO_TASK_NGUOI_3.md](./final-csi-rebuild/BAO_CAO_BAN_GIAO_TASK_NGUOI_3.md) | Việc đã làm, bằng chứng đồng bộ, metric, audit, inference và checklist | Báo cáo đầy đủ nhất; dùng làm căn cứ khi trình bày trạng thái Task 3 |
| [TASK3_HANDOFF.md](./final-csi-rebuild/TASK3_HANDOFF.md) | Chi tiết cấu trúc dữ liệu rebuild, bảng metric và cách chạy train lại | Handoff kỹ thuật; dùng khi muốn tái hiện thí nghiệm |
| [ACTIVITY_CLASSIFICATION_RESULTS.md](../activity-round-experiment/round-heldout-comparison/ACTIVITY_CLASSIFICATION_RESULTS.md) | Kết quả phân loại sáu nhãn hoạt động khi giữ nguyên cả lượt làm test | Đọc riêng với báo cáo pose; đây là bài toán khác |
| [LAN1_VIDEO_AUDIT.md](../activity-round-experiment/Lan1-video-audit/LAN1_VIDEO_AUDIT.md), [LAN2_VIDEO_AUDIT.md](../activity-round-experiment/Lan2-video-audit/LAN2_VIDEO_AUDIT.md), [LAN3_VIDEO_AUDIT.md](../activity-round-experiment/Lan3-video-audit/LAN3_VIDEO_AUDIT.md) | Video/clock được kiểm tra ra sao và kiểm tra đó có giới hạn gì | Xem audit của đúng lượt cần giải thích |
| `signal-processing/docs/` | Quy trình thu, checklist và quy ước đồng bộ/pose | Xem [capture checklist](../../../signal-processing/docs/capture_checklist.md), [video-pose spec](../../../signal-processing/docs/video_pose_spec.md) và [báo cáo Người 2](../../../signal-processing/docs/report_nguoi2.md) |

Không nên gộp các số trong những báo cáo/snapshot cũ thành một bảng duy nhất
nếu chưa kiểm tra dataset, trạng thái đồng bộ và phương pháp chia train/test.
Khi có khác biệt, ưu tiên kết quả của lần rebuild mới nhất trong thư mục
`final-csi-rebuild` và ghi rõ tên thí nghiệm.

## 4. Cấu trúc kết quả Task 3

Thư mục chính:

```text
machine-learning/results/pose-sync-check/
├── HUONG_DAN_DOC_VA_KET_QUA_TASK3.md
└── final-csi-rebuild/
    ├── BAO_CAO_BAN_GIAO_TASK_NGUOI_3.md
    ├── TASK3_HANDOFF.md
    ├── sync_summary_public.csv
    ├── video_clock_anchors.csv
    ├── clock_quality.json
    ├── BIEU_DO_TOM_TAT_TASK_NGUOI_3.png
    ├── combined-verified/training/
    │   ├── pose_baseline.joblib
    │   ├── model_metadata_public.json
    │   ├── summary_metrics.csv
    │   ├── verified_pose_metric_dashboard.png
    │   └── heldout_walking_pose_examples.png
    └── cross-round-Lan2-to-Lan3/ and cross-round-Lan3-to-Lan2/
        └── metrics.csv
```

The original workspace also has per-round directories, raw parsed tables,
landmarks, windows, detailed predictions, and run snapshots. Those generated
data-heavy files are local-only and are intentionally not part of the public
repository; the checked-in tree contains the compact artifacts shown above.

### Các file ở thư mục gốc `final-csi-rebuild`

| File | Ý nghĩa |
|---|---|
| `sync_summary_public.csv` | Trạng thái đồng bộ theo lượt/hoạt động, đã lược bỏ đường dẫn máy cá nhân |
| `video_clock_anchors.csv` | Các mốc giờ đọc từ video dùng để đối chiếu đồng hồ và ước lượng ánh xạ thời gian |
| `clock_quality.json` | Tóm tắt chất lượng clock/fit, residual và các số đo liên quan |
| `pose_extraction_summary.csv` | Tóm tắt kết quả trích pose theo phiên |
| `BIEU_DO_TOM_TAT_TASK_NGUOI_3.png` | Hình tổng quan được lưu cùng hồ sơ bàn giao |

### Thư mục từng lượt: `Lan1`, `Lan2`, `Lan3`

Mỗi lượt lưu các bước trung gian và kết quả theo sáu hoạt động:

| Thành phần | Nội dung |
|---|---|
| `index.csv`, `_pose_metadata.csv`, `raw_csv/`, `landmarks/` | Trong workspace gốc: chỉ mục, CSI đã parse và keypoints/pseudo-pose trích từ video; không được phân phối cùng repo |
| `windows/`, `windows_index.csv`, `windows_summary.csv` | Trong workspace gốc: các cửa sổ CSI-pose và trạng thái đồng bộ; cần dữ liệu nguồn để dựng lại |
| `training/` hoặc `training-diagnostic/` | Trong workspace gốc: kết quả từng lượt; Lan1 chỉ dùng chẩn đoán |

Các tệp `summary_metrics.csv`, `per_session_metrics.csv`,
`out_of_fold_predictions.csv` và `session_splits.csv` lần lượt chứa metric
tổng hợp, metric theo phiên, dự đoán out-of-fold và cách chia fold. `model_metadata.json`
mô tả model/run; `pose_baseline.joblib` là model đã lưu. Các thư mục `runs/<run_id>/`
là snapshot của từng lần chạy, giúp phân biệt lần train thay vì chỉ dựa vào
các file tiện dụng ở thư mục cấp cao hơn.

### Kết quả gộp và các phép thử riêng

| Thư mục | Dùng để làm gì | Lưu ý |
|---|---|---|
| `combined-verified/` | Tập gộp 12 phiên Lan2+Lan3 có `sync_ok=1`; repo giữ model, metadata đã lược định danh, metric tổng hợp và hình | Tập chính để xem kết quả pose đã đồng bộ |
| `retrained-verified-lan2-lan3-20261110/` | Snapshot train lại trên 12 phiên/1.974 windows | Trong workspace gốc; snapshot và dữ liệu chi tiết không được phân phối |
| `cross-round-Lan2-to-Lan3/`, `cross-round-Lan3-to-Lan2/` | Train trên một lượt, test trên lượt còn lại | Repo giữ bảng metric gọn; dự đoán chi tiết/video không được phân phối |
| `inference-smoke-lan3-left-arm/` | Smoke-test inference từ capture Lan3 đã lưu | Kết quả và capture ở workspace gốc; không phải test độc lập |
| `nguoi2-pose-windows-dataset/` | Chẩn đoán riêng trên dataset windows Người 2 | Dữ liệu không được phân phối; kết quả có `sync_ok=0` |

Trong mỗi thư mục kết quả, `.npz` thường lưu cửa sổ số liệu, `.csv` lưu bảng
metric/index/prediction có thể mở bằng Excel hoặc pandas, `.json` lưu cấu hình
và metadata, `.joblib` là model đã serialize, còn `.png`/`.mp4` là hình/video
minh họa. Không nên sửa tay file metric hoặc model đã sinh; nếu cần kết quả
mới, chạy pipeline vào một thư mục output mới.

### Hình để xem trước

- [Dashboard metric pose](./final-csi-rebuild/combined-verified/training/verified_pose_metric_dashboard.png)
  so sánh pose trung bình và Ridge theo tập/hoạt động và phép thử giữa hai lượt.
- [Ví dụ skeleton walking được giữ lại](./final-csi-rebuild/combined-verified/training/heldout_walking_pose_examples.png)
  đặt pseudo-label video, Ridge và pose trung bình cạnh nhau ở cùng cửa sổ.
- [Dashboard phân loại hoạt động theo lượt](../activity-round-experiment/round-heldout-comparison/activity_round_results.png)
  hiển thị metric theo lượt, recall và nhầm lẫn giữa các nhãn.

Hình skeleton giúp nhìn lỗi và giải thích baseline; nó không phải chứng minh
định lượng độc lập, không tạo thêm dữ liệu và không tự cải thiện mô hình.

## 5. Dữ liệu nào được dùng và vì sao đồng bộ quan trọng?

Các capture CSI/video gốc không được đóng gói trong repo vì kích thước và
quyền chia sẻ dữ liệu. Báo cáo, metric tóm tắt và hình nghiên cứu có thể xem
trực tiếp sau khi clone; để tái chạy train hoặc inference, người dùng cần
được cung cấp dữ liệu tương thích riêng. Vì vậy, chỉ clone repo sẽ không tự
tái tạo được các lần train trong báo cáo.

| Dữ liệu | Quy mô | Cách dùng |
|---|---:|---|
| `Final_CSI` Lan1 | 6 hoạt động, 1.043 cửa sổ pose | Giữ riêng để chẩn đoán; `sync_ok=0`, không xem là pose ground truth đã ghép chắc chắn |
| `Final_CSI` Lan2 | 6 hoạt động, 991 cửa sổ pose | Đã kiểm tra clock/video; dùng trong tập verified |
| `Final_CSI` Lan3 | 6 hoạt động, 983 cửa sổ pose | Đã kiểm tra clock/video; dùng trong tập verified |
| Lan2 + Lan3 | 12 phiên, 1.974 cửa sổ | Tập verified chính cho phép thử pose |
| Dataset windows Người 2 có trong workspace | 6 phiên, 1.054 cửa sổ | Chẩn đoán riêng; `sync_ok=0`, không tính là verified pose |

Toàn bộ `Final_CSI` có 18 capture (sáu hoạt động ở mỗi lượt), 3.017 cửa sổ
pose được tạo ra và 10.932 frame pose được trích. Một cửa sổ CSI dùng 2 giây
dữ liệu, bước trượt 0,2 giây và có dạng `[20, 224]` (20 frame CSI, 224 đặc
trưng mỗi frame). Trong số các cửa sổ pose, chỉ 1.974 cửa sổ Lan2/Lan3 được
dùng làm tập verified chính.

Một `window` là đoạn dữ liệu ngắn dùng làm một mẫu mô hình, không phải một
người hoặc một lần quay độc lập. Các cửa sổ được lấy chồng lấn, nên 1.974
windows không tương đương 1.974 thí nghiệm độc lập.

`sync_ok=1` chỉ cho biết phiên đủ điều kiện đồng bộ theo tiêu chí/bằng chứng
được ghi trong audit; không đồng nghĩa mọi frame hoặc mọi keypoint đã được
người kiểm tra thủ công. Lan2/Lan3 đã giải mã đầy đủ video và kiểm tra mẫu
hình thưa 1 Hz; Lan1 có lời xác nhận bắt đầu hai thiết bị gần nhau nhưng thiếu
mốc chung đo độc lập đủ mạnh, nên vẫn giữ `sync_ok=0` cho bài toán pose.

## 6. Kết quả hiện tại và cách diễn giải đúng

### CSI-to-pose: baseline CSI chưa thắng baseline pose đơn giản

Trong đánh giá trên 12 phiên Lan2/Lan3:

| Mô hình tọa độ | Sai số chuẩn hóa (thấp hơn tốt hơn) | PCK@0.05 (cao hơn tốt hơn) |
|---|---:|---:|
| Pose trung bình (`training_mean`) | **0,0423** | **0,714** |
| Ridge dùng đặc trưng CSI | 0,0579 | 0,552 |

PCK là tỷ lệ keypoint nằm trong ngưỡng sai số đã định; ở đây ngưỡng là 0,05
đường chéo ảnh. Trong hai phép giữ nguyên cả một lượt để test, kết quả cũng
nghiêng về baseline pose trung bình:

| Train → test | Pose trung bình: lỗi / PCK | Ridge: lỗi / PCK |
|---|---:|---:|
| Lan2 → Lan3 | 0,0462 / 0,720 | 0,1143 / 0,208 |
| Lan3 → Lan2 | 0,0389 / 0,713 | 0,1271 / 0,166 |

**Ý nghĩa:** model được chọn để xuất tọa độ là `training_mean`, tức pose trung
bình tĩnh, không tạo chuyển động khớp dựa trên CSI. Ridge hiện không cho thấy
lợi ích đủ tốt so với baseline đơn giản. Vì vậy, không được diễn giải các
tọa độ đầu ra như bằng chứng CSI đã nhận biết tư thế động.

Person-presence có balanced accuracy khoảng **0,503** trong đánh giá gộp. F1
khoảng **0,874** trông cao hơn, nhưng dữ liệu mất cân bằng; cần xem balanced
accuracy, recall từng lớp và confusion matrix, không dùng riêng F1/accuracy
để khẳng định phát hiện người đáng tin cậy.

### Phân loại hoạt động: có tín hiệu ban đầu nhưng còn yếu

Trong thí nghiệm giữ nguyên cả lượt làm test trên 18 phiên/10.237 frame,
Random Forest với frame + temporal features đạt balanced accuracy **0,518**
và macro-F1 **0,512**; baseline đoán theo tần suất đạt balanced accuracy
**0,164**. Điểm theo lượt test dao động (0,584; 0,527; 0,444).

Đây là bằng chứng thăm dò rằng đặc trưng CSI có thể chứa tín hiệu liên quan
đến nhãn hoạt động, nhưng chưa đủ để gọi là hệ thống nhận diện đáng tin cậy.
Nhãn hoạt động được gán theo phiên, không phải nhãn chính xác tại từng thời
điểm của video; kết quả không chứng minh mô hình nhận ra thời điểm người vừa
bước vào phòng.

### Các giới hạn không được bỏ qua

- Chỉ có một người tham gia, một bối cảnh/phòng và hai lượt pose đã xác minh.
- Pose là ước lượng từ video, không phải ground truth motion-capture.
- Chưa có nhãn thật về giới tính, chiều cao hoặc tình huống nhiều người.
- Smoke-test inference trên capture đã dùng trong nghiên cứu chỉ kiểm tra
  pipeline có chạy; không đo khả năng dự đoán trên người/phiên mới.
- Chưa tích hợp/đánh giá end-to-end với receiver CSI live, backend và frontend.

Do đó, phát biểu phù hợp là: **pipeline nghiên cứu offline và prototype chạy
được; bằng chứng hiện tại chưa hỗ trợ tuyên bố sản phẩm đã đếm người, nhận
diện giới tính/chiều cao hoặc dự đoán pose chuyển động đáng tin cậy.**

## 7. Xem kết quả hoặc chạy lại

### Chỉ xem kết quả đã có

1. Đọc phần kết luận ở đầu hướng dẫn này.
2. Mở `final-csi-rebuild/BAO_CAO_BAN_GIAO_TASK_NGUOI_3.md` để xem bằng chứng
   và trạng thái công việc.
3. Mở `combined-verified/training/summary_metrics.csv` để xem kết quả gộp.
   Tệp `model_metadata_public.json` mô tả cấu hình model mà không chứa định
   danh các phiên quay.
4. Mở `cross-round-Lan2-to-Lan3/metrics.csv` và
   `cross-round-Lan3-to-Lan2/metrics.csv` để xem test giữa hai lượt.
5. Mở các CSV metric cùng `ACTIVITY_CLASSIFICATION_RESULTS.md` ở thư mục
   `activity-round-experiment/round-heldout-comparison/` để đọc kết quả phân
   loại hoạt động. Xem ba hình được liên kết ở mục 4 để nắm kết quả trực quan.

### Dựng lại dashboard pose

Từ thư mục gốc repo, với môi trường Python đã cài dependencies của
`machine-learning`:

```powershell
python machine-learning/src/plot_verified_pose_comparison.py
```

Script đọc kết quả đã lưu và tạo lại hai hình dashboard/skeleton; nó **không
train lại model**. Xem
[README machine-learning](../../README.md) để cài môi trường, train, chạy
inference trên capture CSI đã lưu và chạy test.

### Train lại có kiểm soát

Các lệnh và tùy chọn đầy đủ nằm trong
[TASK3_HANDOFF.md](./final-csi-rebuild/TASK3_HANDOFF.md). Train lại có thể
ghi ra thư mục kết quả; hãy chọn output mới để không ghi đè snapshot. Chỉ
dùng windows Lan2/Lan3 đã xác minh cho kết quả pose chính, không thêm cờ
`--allow-unverified-sync` trừ khi chủ ý chạy chẩn đoán và ghi rõ điều đó.

## 8. Trước khi đưa tài liệu/kết quả lên main

- Giữ hướng dẫn này cùng các báo cáo, mã nguồn, test và hình được dẫn chiếu
  trong repo để link tương đối không bị hỏng.
- Kiểm tra `git status` và dung lượng thay đổi trước khi push. Các capture
  CSI/video gốc và môi trường ảo thường rất lớn hoặc chứa dữ liệu nghiên cứu;
  không đưa chúng lên Git nếu chưa được nhóm đồng ý và chưa kiểm tra chính
  sách chia sẻ.
- Nếu thêm một lần train mới, ghi rõ dataset, trạng thái đồng bộ, cách chia
  train/test, ID run và metric; không thay số cũ mà không lưu snapshot.
- Khi báo cáo kết quả, trình bày cả baseline, kết quả theo từng lượt và giới
  hạn dữ liệu; không chỉ chọn metric đẹp nhất hoặc một hình minh họa.
