# Wi-Fi CSI Human Presence & 2D Pose Estimation (`machine-learning`)

Module của **Thành viên 3 (Người thứ 3)** trong hệ thống Wi-Fi Sensing: tiếp nhận tập cửa sổ CSI–Pose đồng bộ từ Người 2, huấn luyện mô hình dự đoán trạng thái có người (`person_present`) và tọa độ 33 khớp 2D (`keypoints`), đánh giá trên phiên độc lập và cung cấp engine suy luận chuẩn hóa cho Backend/Frontend (Người 4).

## Checkpoint đang dùng trên backend Jayson

Backend nạp `models/checkpoints/smn_pose.pt` và `smn_normalization.json`, phiên bản `smn-csiposenet-1.6`. Ngưỡng hiện diện là 0,65. Hai nửa của mỗi cửa sổ CSI cung cấp mức biến đổi theo thời gian. Với tín hiệu ít biến động, mô hình còn kiểm tra mức tương đồng với phổ các ca yên tĩnh đã học. Mạng dự đoán 33 khớp từ CSI chuẩn hóa theo liên kết anten, học cả vị trí và độ dịch chuyển giữa hai khung video có nhãn.

Để huấn luyện lại, đặt các bản thu từ Drive theo cột `path` của [danh mục phiên](configs/live-captures.csv) dưới `$CAPTURES`. Hai tệp `Lan2_windows.npz` và `Lan3_windows.npz` chứa nhãn khớp từ video đồng bộ của chính đợt đó. Chạy từ gốc repo với các thư viện trong `machine-learning/requirements.txt`.

```bash
.venv/bin/python machine-learning/src/train_live_presence.py \
  --manifest machine-learning/configs/live-captures.csv --data-root "$CAPTURES" \
  --checkpoint machine-learning/models/checkpoints/smn_pose.pt \
  --synced-round "$CAPTURES/Lan2_windows.npz" --synced-round "$CAPTURES/Lan3_windows.npz" \
  --output "$CAPTURES/presence.pt" --report "$CAPTURES/presence.json"
.venv/bin/python machine-learning/src/train_live_pose.py \
  --round "$CAPTURES/Lan2_windows.npz" --round "$CAPTURES/Lan3_windows.npz" \
  --checkpoint "$CAPTURES/presence.pt" --output "$CAPTURES/model"
```

Các phiên cùng ngày được giữ chung khi đánh giá hiện diện. Lần 1 chỉ đóng góp nhãn hoạt động. Hai phiên Ubuntu ngày 17/11 lúc 11:47 và 11:49 có vai trò `reference` trong danh mục. Chúng chỉ dùng để xem lại hành vi phần mềm. Lệnh trên ghép timestamp CSI với khung hình đồng bộ của Lần 2 và 3 để chấm hiện diện. Bản live chỉ cần gói CSI mới của phiên. [Kết quả trên bản ghi và giới hạn](../backend/NGHIEM-THU.md)

## Lịch sử bản 1.5

Backend mặc định nạp `models/checkpoints/smn_pose.pt` và `smn_normalization.json`, phiên bản `smn-csiposenet-1.5`. Hiện diện dùng 76 đặc trưng hình dạng/biến thiên tương đối CSI, bỏ 16 vị trí điền cố định và chuẩn hóa RMS từng gói, từng liên kết anten. Extra Trees 128 cây được lưu vào checkpoint dưới dạng các mảng; `src/presence.py` suy luận bằng NumPy. Ngưỡng điểm là 0,5. Backend cần ba kết quả dương liên tiếp trước khi báo có người, và xóa khớp ngay khi có dự đoán phòng trống.

Để học lại phần hiện diện và giữ trọng số tư thế, tạo các NPZ theo nhóm phiên với `amplitudes` có shape `(N, 20, 224)` và `activity` chứa `empty` hoặc tên hoạt động có người. Giữ thứ tự thời gian trong từng hoạt động. Ví dụ sau khi chuẩn bị các tệp cửa sổ đã gắn nhãn:

```bash
.venv/bin/python machine-learning/src/train_presence.py \
  --train "$HOME/smn-windows/Lan2_windows.npz" \
  --train "$HOME/smn-windows/Lan3_windows.npz" \
  --train "$HOME/smn-windows/Oct1_windows.npz" \
  --train "$HOME/smn-windows/Oct2_windows.npz" \
  --train "$HOME/smn-windows/Lan1_windows.npz" \
  --train "$HOME/smn-windows/Ubuntu16_windows.npz" \
  --train "$HOME/smn-windows/Ubuntu17_windows.npz" \
  --holdout "$HOME/smn-windows/Oct3_windows.npz" \
  --train-fraction 0.65 --purge-windows 20 \
  --checkpoint machine-learning/models/checkpoints/smn_pose.pt \
  --output "$HOME/smn-windows/smn_pose_candidate.pt" \
  --report "$HOME/smn-windows/presence-validation.json"
```

Công cụ dùng 65% đầu mỗi hoạt động để học và bỏ 20 cửa sổ ở ranh giới. `--holdout` giữ cả tệp ngoài tập học. Báo cáo ghi SHA-256 tệp, số mẫu từng nhóm/hoạt động, đối chiếu bộ suy luận NumPy với scikit-learn và thử biến đổi mức khuếch đại/giá trị điền. Chạy công cụ cần các thư viện huấn luyện trong `requirements.txt`; chạy backend live không cần cài thêm scikit-learn. Bản phân chia thực tế và số liệu checkpoint 1.5 nằm trong [báo cáo JSON](results/presence-1.5-validation.json) và [nghiệm thu](../backend/NGHIEM-THU.md#nghiệm-thu-smn-15-với-luồng-một-nút).

`Ubuntu17_windows.npz` ghép hai phiên `1c7cb0d9ecee49c290d0d787255b5779` (phòng trống 11:47) và `7005d27a0bd84b67923d1cf9df0d7f0c` (đi, giơ tay 11:49) theo nhãn người thu xác nhận. `Ubuntu16_windows.npz` ghép hai bản ghi ZIP phòng trống/đi lại trước đó. `Oct1–3` ghép các phiên `empty`, `seated`, `walking` cùng số thứ tự trong `hardware/csi_extracted/labeled`; `Lan1–3` là ba đợt thu của nhóm. Các NPZ chỉ phục vụ học và kiểm tra offline; backend live đọc gói mới của phiên đang thu.

Điểm hiện diện chưa được hiệu chỉnh thành xác suất đúng ở môi trường mới. Kết quả chia đoạn cùng phiên không thay thế kiểm tra bằng một đợt thu độc lập trên máy Ubuntu. Trọng số tọa độ vẫn có giới hạn khi giơ tay tự do ở phiên mới; xem số liệu giữ riêng theo đợt trong nghiệm thu.

---

## 1. Phạm vi bài toán & Quy ước dữ liệu (Bước 1–2)

- **Phạm vi:** 1 người trong phòng cố định, thiết bị Wi-Fi & camera cố định; dự đoán trạng thái có người (`empty` vs `present`) và **bộ xương 2D trong mặt phẳng ảnh** (chưa hỗ trợ nhiều người hoặc 3D).
- **Đầu vào (`CSI window`):** Mảng thực `(window_size=20, num_subcarriers=224)` (~2.0 giây, bước trượt `step_seconds=0.2s`, `sampling_interval_ms=101.0ms`).
- **Chuẩn hóa CSI:** Z-score theo từng subcarrier (`mean`, `std` lưu trong `models/checkpoints/normalization_params.json`), **chỉ fit trên tập Train** (kẹp `std < 1e-6 -> 1.0`). Tương thích phiên bản tiền xử lý hiện tại của `signal-processing` (`224` subcarriers, cửa sổ 20 bước).
- **Số khớp & thứ tự:** 33 khớp chuẩn MediaPipe BlazePose (`0: nose` đến `32: right_foot_index` — danh sách đầy đủ trong `src/predict.py::LANDMARK_NAMES`).
- **Hệ tọa độ:**
  - Mô hình hồi quy ở không gian chuẩn hóa `[0, 1]` gốc trên-trái (`origin: top_left`).
  - Khi xuất payload cho Backend/Frontend, tự động quy đổi sang pixel gốc `1920 × 1080` (`x_pixel`, `y_pixel`) kèm bản chuẩn hóa (`x_norm`, `y_norm`).
- **Quy tắc `empty` & dữ liệu thiếu/lỗi:**
  - Khi `empty` (`person_present = False`): Loss tọa độ khớp bằng `0.0` (không backprop sai số bộ xương giả); khi suy luận trả về `keypoints = []`.
  - Dữ liệu thiếu, NaN/Inf, sai shape hoặc timestamp quá cũ (`max_age_sec`) được chặn ở `CSIPoseDataset`, `MaskedPoseLoss` và `CSIPosePredictor.predict_window` với mã lỗi rõ ràng (`MISSING_DATA`, `INVALID_SHAPE`, `NUMERICAL_INSTABILITY`, `STALE_INPUT`) để giao diện không giữ bộ xương cũ.

---

## 2. Cấu trúc thư mục

```text
machine-learning/
├── configs/
│   └── default_config.json         # Cấu hình phiên chia tập, kiến trúc mạng & hyperparameters
├── docs/
│   └── report_nguoi3.md            # Báo cáo chi tiết quá trình xây dựng & đánh giá
├── models/
│   └── checkpoints/
│       ├── best_model.pt           # Trọng số CSIPoseNet tốt nhất theo Val Loss
│       ├── mlp_baseline.pt         # Trọng số LinearMLPBaseline để đối chiếu
│       ├── model_config.json       # Cấu hình mô hình tương ứng checkpoint
│       └── normalization_params.json
├── results/
│   ├── experiment_log.json         # Nhật ký train/val/test & so sánh các mô hình
│   ├── evaluation_metrics.json     # Báo cáo chỉ số chi tiết theo từng phiên/split & độ trễ
│   ├── loss_curve.png              # Biểu đồ Loss, Val MPJPE, Presence Accuracy
│   ├── mpjpe_by_activity.png       # MPJPE theo hoạt động/phiên
│   ├── presence_confusion_matrix.png
│   ├── pose_prediction_comparison.png
│   ├── failure_cases.png           # Các trường hợp sai số lớn nhất trên tập unseen
│   └── pose_comparison_*.mp4       # Video đối chiếu Ground Truth vs Dự đoán
├── src/
│   ├── dataset.py                  # Đọc .npz, làm sạch NaN, CSINormalizer, chia theo phiên
│   ├── models.py                   # MeanPoseBaseline, LinearMLPBaseline, CSIPoseNet
│   ├── losses.py                   # MaskedPoseLoss (BCE + Masked Smooth L1)
│   ├── metrics.py                  # MPJPE, PCK@5/10/20%, Presence Accuracy/Precision/Recall/F1
│   ├── train.py                    # Huấn luyện baselines & CSIPoseNet, early stopping
│   ├── evaluate.py                 # Đánh giá phiên độc lập, benchmark độ trễ, xuất biểu đồ & video
│   └── predict.py                  # CSIPosePredictor & CSIStreamBuffer cho real-time/offline
├── tests/
│   └── test_pipeline.py            # Kiểm thử đơn vị pytest
└── requirements.txt
```

---

## 3. Cài đặt & Lệnh chạy

### Cài đặt thư viện
```bash
cd machine-learning
pip install -r requirements.txt
```

### 1) Huấn luyện (Baselines + CSIPoseNet)
```bash
python src/train.py --config configs/default_config.json
```

### 2) Đánh giá trên phiên độc lập & xuất video/biểu đồ
```bash
python src/evaluate.py --checkpoint models/checkpoints/best_model.pt --normalizer models/checkpoints/normalization_params.json --output-dir results
```

### 3) Suy luận mẫu (Đọc 1 cửa sổ `.npz`)
```bash
python src/predict.py --sample-npz ../signal-processing/data/pose/windows/left_arm_01.npz
```

### 4) Chạy bộ kiểm thử đơn vị
```bash
python -m pytest tests/ -v
```

---

## 4. Hợp đồng dữ liệu suy luận (Contract cho Người 4)

```python
from src.predict import CSIPosePredictor, CSIStreamBuffer

predictor = CSIPosePredictor(
    checkpoint_path="models/checkpoints/best_model.pt",
    normalizer_path="models/checkpoints/normalization_params.json",
)
result = predictor.predict_window(csi_window, timestamp=1700000000.0, max_age_sec=5.0)
```

Ví dụ kết quả thành công (`status == "success"`):
```json
{
  "status": "success",
  "timestamp": 1700000000.0,
  "person_present": true,
  "presence_probability": 0.9981,
  "confidence_method": "model_presence_sigmoid_probability",
  "keypoints": [
    {"id": 0, "name": "nose", "x_pixel": 945, "y_pixel": 280, "x_norm": 0.4922, "y_norm": 0.2593, "confidence": 0.9981}
  ],
  "num_keypoints": 33,
  "latency_ms": 4.2,
  "coordinate_system": {
    "type": "2D_pixel",
    "reference_width": 1920.0,
    "reference_height": 1080.0,
    "origin": "top_left"
  },
  "model_version": "1.0.0"
}
```
