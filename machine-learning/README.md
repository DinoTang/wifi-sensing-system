# Wi-Fi CSI Human Presence & 2D Pose Estimation (`machine-learning`)

Module của **Thành viên 3 (Người thứ 3)** trong hệ thống Wi-Fi Sensing: tiếp nhận tập cửa sổ CSI–Pose đồng bộ từ Người 2, huấn luyện mô hình dự đoán trạng thái có người (`person_present`) và tọa độ 33 khớp 2D (`keypoints`), đánh giá trên phiên độc lập và cung cấp engine suy luận chuẩn hóa cho Backend/Frontend (Người 4).

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
