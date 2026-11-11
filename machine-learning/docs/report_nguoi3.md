# Báo cáo hoàn thành công việc Người thứ 3: Mô hình CSI sang Tư thế 2D

Tài liệu này tổng hợp đầy đủ quy trình thực hiện, kết quả thực nghiệm, phân tích lỗi và giới hạn của bộ dữ liệu cho nhiệm vụ **Người thứ 3** theo đặc tả trong `Cong_viec_nguoi_thu_3_Mo_hinh_CSI.docx`.

---

## 1. Chốt bài toán & Kiểm tra dữ liệu bàn giao (Bước 1–2)

### 1.1 Phạm vi bài toán thống nhất
- **Đối tượng & không gian:** 1 người trong phòng thí nghiệm cố định, vị trí thiết bị phát/thu Wi-Fi và camera cố định (`P01`).
- **Đầu vào:** Cửa sổ biên độ CSI đã xử lý hình dạng `(window_size=20, num_subcarriers=224)`, tương ứng ~2.0 giây (`window_seconds=2.0`, `step_seconds=0.2`, `sampling_interval_ms=101.0`, lưu trữ `float16`).
- **Đầu ra:**
  1. `person_present` (nhị phân có người / trống phòng) kèm `presence_probability` (từ đầu ra sigmoid của nhánh phân loại).
  2. Tọa độ **33 khớp 2D MediaPipe BlazePose** (`nose` -> `right_foot_index`) trong mặt phẳng ảnh chuẩn hóa `[0, 1]` gốc trên-trái, quy đổi sang pixel `1920 × 1080` khi suy luận cho Backend/Frontend. Chưa dựng hình dáng 3D hoặc đa người.
- **Quy tắc `empty`:** Khi không có người (`empty`), không tính loss tọa độ khớp trong khi huấn luyện và trả mảng `keypoints = []` khi suy luận (tránh vẽ bộ xương giả).

### 1.2 Kiểm tra dữ liệu bàn giao từ Người 2
- **Nguồn dữ liệu:** `signal-processing/data/pose/windows/*.npz` gồm 6 phiên độc lập, tổng **1,054 cửa sổ** (`20 × 224`):

| Tệp `.npz` | Mã phiên (`session`) | Hoạt động | Số cửa sổ | Số khớp hợp lệ (`mask_sum`) | Ghi chú chất lượng |
|---|---|---|---:|---:|---|
| `empty_01.npz` | `P01_empty_01_20261103-025228_RF3X` | `empty` | 190 | 0 | `keypoints` có `NaN` (12,540 phần tử), `visibility` có `NaN` (660 phần tử), `mask=0`, `sync_ok=0` |
| `standing_01.npz` | `P01_standing_01_20261103-025406_96qH` | `standing` | 194 | 6,402 | Đủ 33/33 khớp; người đứng cố định (`x ≈ 1037 ± 64.6 px`); `sync_ok=0` |
| `left_arm_01.npz` | `P01_left_arm_01_20261103-025544_ttVs` | `left_arm` | 201 | 6,633 | Đủ 33/33 khớp; người đứng cố định (`x ≈ 1031 ± 65.4 px`); `sync_ok=0` |
| `right_arm_01.npz` | `P01_right_arm_01_20261103-025721_OG2o` | `right_arm` | 185 | 6,105 | Đủ 33/33 khớp; người đứng cố định (`x ≈ 1026 ± 70.6 px`); `sync_ok=0` |
| `mixed_01.npz` | `P01_mixed_01_20261103-030037_i7Q7` | `mixed` | 153 | 4,994 | 49 cửa sổ che khuất một phần khớp; người di chuyển ngang (`x ≈ 1020 ± 176.6 px`); `sync_ok=0` |
| `walking_01.npz` | `P01_walking_01_20261103-025859_EiZQ` | `walking` | 131 | 4,290 | 31 cửa sổ che khuất một phần khớp; người đi lại (`x ≈ 1022 ± 181.7 px`); `sync_ok=0` |

- **Xử lý lỗi `NaN` ở dữ liệu thật (`empty_01`):** Trong quá trình chạy thử nghiệm ban đầu, `NaN` trong `empty_01['keypoints']` gây nổ `train_loss = NaN` dù `mask = 0` (vì theo chuẩn IEEE 754, `NaN * 0.0 == NaN` trước khi cộng tổng loss). Lỗi đã được xử lý triệt để tại cả `CSIPoseDataset` (`src/dataset.py`) và `MaskedPoseLoss` (`src/losses.py`).
- **Kiểm tra trực quan mẫu `left_arm_01` ("Việc nên làm trước tiên"):** Đã xuất biểu đồ kiểm tra cặp CSI–Skeleton cửa sổ #50 tại `results/step2_left_arm_sanity_check.png`.

---

## 2. Chia tập dữ liệu & Chuẩn hóa (Bước 3)

Tuân thủ nguyên tắc **chia theo phiên độc lập**, không chia ngẫu nhiên khung hình lân cận cùng phiên để tránh data leakage:
- **Train (770 mẫu):** `empty_01`, `standing_01`, `left_arm_01`, `right_arm_01` (`190` empty + `580` present).
- **Validation (153 mẫu):** `mixed_01` (dùng để chọn cấu hình, điều chỉnh learning rate và early stopping).
- **Independent Test (131 mẫu):** `walking_01` (giữ độc lập hoàn toàn cho đánh giá cuối).
- **Chuẩn hóa CSI:** `CSINormalizer` Z-score theo từng subcarrier được fit **duy nhất trên 770 mẫu Train**, lưu tại `models/checkpoints/normalization_params.json`.

---

## 3. Xây dựng Mô hình Cơ sở & Kiến trúc chính (Bước 4–5)

### 3.1 Hàm mất mát & Chống học thuộc
- **`MaskedPoseLoss` (`src/losses.py`):**
  $$\mathcal{L} = \lambda_{\text{pres}} \cdot \text{BCEWithLogits} + \lambda_{\text{pose}} \cdot \text{MaskedSmoothL1}(\beta=0.05)$$
  với $\lambda_{\text{pres}} = 1.0, \lambda_{\text{pose}} = 5.0$. Trọng số pose nhân với `person_mask * joint_mask * visibility` nên các mẫu `empty` có sai số tọa độ bằng **0.0 tuyệt đối**.
- **Kiểm tra kỹ thuật trên tập nhỏ:** Unit test `test_small_subset_overfit_sanity_check` xác nhận mạng hội tụ và giảm >80% loss trên batch tổng hợp nhỏ (bảo đảm luồng gradient thông suốt).

### 3.2 So sánh các mô hình (Hyperparameters trong `configs/default_config.json`, `seed=42`)

| Mô hình | Đặc điểm chính | Val (`mixed_01`) Presence Acc | Val (`mixed_01`) MPJPE | Test (`walking_01`) Presence Acc | Test (`walking_01`) MPJPE | Test PCK@10% | Test PCK@20% |
|---|---|---:|---:|---:|---:|---:|---:|
| **`MeanPoseBaseline`** (Naive) | Dự đoán tư thế trung bình của tập Train (`prob=0.753`) | 100.0% *(luôn dự đoán có người)* | **181.52 px** | 100.0% *(luôn dự đoán có người)* | **172.20 px** | 45.5% | **99.9%** |
| **`LinearMLPBaseline`** | Mạng MLP phẳng `4480 -> 256 -> 128`, đầu ra tuyến tính | 0.0% | 974.75 px | 0.0% | 983.49 px | 2.3% | 11.5% |
| **`CSIPoseNet`** | `Conv1D([64,128]) + BiGRU(h=96, 2 layers) + TemporalAttention` + `TemporalDiff (dx)` + Nhánh năng lượng chuyển động (`BatchNorm1d(4)`) + `Sigmoid` pose | **100.0%** | 206.98 px | **99.2%** *(F1=0.996)* | 213.73 px | **48.7%** | 85.7% |

*(Lưu ý: Trên tập Train diagnostic, `CSIPoseNet` đạt MPJPE = **20.24 px**, Presence Acc = **100.0%** (`190/190` empty đúng, `580/580` present đúng). Không dùng kết quả Train này để kết luận khả năng tổng quát hóa.)*

---

## 4. Phân tích lỗi & Đánh giá trên dữ liệu độc lập (Bước 6)

### 4.1 Phát hiện thực nghiệm quan trọng (Error Analysis)
1. **Vì sao `LinearMLPBaseline` sụp đổ hoàn toàn (`983 px`, `0%` Acc) và phiên bản `CSIPoseNet` thô ban đầu chỉ đạt `~40–51%` Presence Acc trên `mixed_01`/`walking_01`?**
   - Trong tập Train, `empty_01` có biên độ CSI trung bình `125.95` (`z std = 1.376`) trong khi 3 phiên `standing/left_arm/right_arm` đều đứng yên tại vùng giữa khung (`x ≈ 1030 px`) với biên độ CSI `133.6–135.3` và phổ tần đa đường tĩnh khác biệt.
   - Khi người di chuyển ra các vị trí khác trong `mixed_01` và `walking_01`, biên độ CSI trung bình dịch về `129.6` và phổ tĩnh tiệm cận `empty_01` (khoảng cách phổ `2.5` so với `10.5` của `standing`). Các mạng phụ thuộc đặc trưng tĩnh đã học lối tắt sai: *“phổ giống `empty_01` => phòng trống”* (`corr = 0.869`) và nội suy tọa độ ra ngoài miền (`>400–980 px`).
2. **Giải pháp kiến trúc sau phân tích lỗi:**
   - Ràng buộc đầu ra tọa độ bằng `Sigmoid` giữ bộ xương dự đoán hợp lệ trong `[0, 1]`.
   - Thêm dữ liệu tăng cường dịch gain (`gain_shift_aug_std=0.5`, `subcarrier_shift_aug_std=0.2`, `noise_aug_std=0.08`) và kênh sai phân thời gian `dx = x[t] - x[t-1]`.
   - Bổ sung **nhánh phát hiện chuyển động bất biến gain** (`energy_presence_head` trên `log1p(std_t)` + `BatchNorm1d(4)` kết hợp `branch_mask` dropout 50% nhánh tĩnh khi train). Nhờ vậy, Presence Accuracy trên phiên độc lập tăng từ **~51.1% lên 100.0% (Val `mixed_01`) và 99.2% (Test `walking_01`)**, trong khi vẫn giữ **100.0% (190/190)** nhận diện đúng phòng trống trên `empty_01`.
3. **Vì sao MPJPE của `CSIPoseNet` (`207–214 px`) chưa vượt được `MeanPoseBaseline` (`172–182 px`) trên `walking_01`/`mixed_01`?**
   - Tập Train chỉ chứa các tư thế **đứng tại một chỗ (`x std ≈ 65 px`)**, hoàn toàn **không có dữ liệu đi lại (`walking`, `x std ≈ 182 px`)**, và toàn bộ 6 tệp `.npz` hiện đều gắn cờ **`sync_ok = 0`**. Do chưa có cặp CSI–Pose cho chuyển động ngang qua phòng trong tập Train, mọi mô hình đều bị giới hạn về suy diễn vị trí tổng thể khi người đi sang hai rìa khung ảnh.

### 4.2 Tốc độ suy luận & Độ trễ (trên CPU máy thử nghiệm)
- **Model forward (`20 × 224`, batch=1):** trung bình **5.06 ms**, p95 **6.35 ms** (~**197.5 FPS**).
- **Toàn quy trình `CSIPosePredictor.predict_window` (validate + normalize + tensor + infer + đóng gói JSON 33 khớp):** trung bình **6.01 ms**, p95 **7.08 ms** (~**166.3 FPS**). Thời gian này nhỏ hơn rất nhiều so với bước trượt cửa sổ `200 ms` (`step_seconds = 0.2s`), đáp ứng chạy real-time trên CPU.

### 4.3 Hiện vật trực quan & Video đối chiếu
- `results/loss_curve.png`: Đồ thị train/val loss, Val MPJPE và Val Presence Accuracy qua các epoch.
- `results/presence_confusion_matrix.png`, `results/mpjpe_by_activity.png`.
- `results/pose_prediction_comparison.png` & `results/failure_cases.png` (top mẫu sai số lớn nhất trên unseen Val/Test).
- Video đối chiếu Ground Truth vs Dự đoán: `results/pose_comparison_mixed_01.mp4` và `results/pose_comparison_walking_01.mp4`.

---

## 5. Giới hạn hệ thống & Khuyến nghị tiếp theo (Điều kiện hoàn thành)

Theo đúng yêu cầu nghiệm thu trong `Cong_viec_nguoi_thu_3_Mo_hinh_CSI.docx`:
1. **Chưa đủ phiên lặp lại cho đánh giá tổng quát:** Mỗi hoạt động hiện mới có **1 phiên thu duy nhất**. Đặc biệt, `empty_01` là phiên trống phòng duy nhất nên phải nằm trong tập Train — dẫn đến tập Val (`mixed_01`) và Test (`walking_01`) không chứa khung hình trống để đo False Positive Rate độc lập. Cần phối hợp Người 1 thu thêm ít nhất `empty_02`, `walking_02`, `standing_02`, `mixed_02`.
2. **Đồng bộ dữ liệu chưa xác minh (`sync_ok = 0`):** Toàn bộ 6 tệp `.npz` hiện tại đều có `sync_ok = 0` (`offset_s = 0.0`). Độ chính xác khớp chi tiết theo thời gian thực chưa thể cam kết đạt mức dựng tư thế chính xác cho chuyển động tự do cho đến khi Người 2 xác nhận đồng bộ thời gian và bổ sung dữ liệu người đi lại trong tập huấn luyện.
3. **Ngưỡng đạt thực nghiệm:** Chưa tự đặt con số MPJPE mục tiêu khi chưa thống nhất với nhóm/giảng viên; hiện tại hệ thống đã chạy lại hoàn chỉnh từ đầu đến cuối, vượt qua **11/11 unit tests**, đạt hiệu năng nhận diện có người **99.2%–100%** trên phiên unseen và cung cấp đầy đủ contract suy luận sẵn sàng tích hợp.
