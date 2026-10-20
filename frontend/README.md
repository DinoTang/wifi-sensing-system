# Wi-Fi Sensing System - Frontend Dashboard

Giao diện Web Dashboard thời gian thực phục vụ giám sát sóng **Wi-Fi CSI (Channel State Information)** và **Phát hiện có người trong phòng (Human Presence Detection)**.

Được xây dựng 100% bằng **Python** sử dụng framework **NiceGUI** (nền tảng FastAPI, WebSockets và Apache ECharts chạy trên trình duyệt).

---

## 🌟 Tính năng nổi bật

### 1. Màn hình 1: Giám sát tín hiệu CSI (CSI Signal Monitor)
- **Time-Series Waveforms:** Vẽ dạng sóng biên độ thời gian thực của 5 subcarriers tiêu biểu (-20, -10, 0, 10, 20) và đường biên độ trung bình.
- **Spectrum across 64 Subcarriers:** Biểu đồ phổ toàn bộ 64 sóng mang con (chuẩn OFDM 20MHz).
- **Thước đo truyền thông:** Tốc độ gói tin (Sampling Rate Hz), Cường độ tín hiệu RSSI (dBm), Chỉ số biến thiên CSI (Variance).

### 2. Màn hình 2: Phát hiện có người (Human Presence Detection)
- **Thẻ trạng thái trung tâm:** Đèn báo và biểu tượng động:
  - 🟢/⚪ **Phòng trống (Empty Room)**
  - 🟠 **Có người ngồi yên / Thở (Stationary / Breathing)**
  - 🔴 **Có người đang di chuyển (Moving / Walking)**
- **Độ tin cậy mô hình AI (Confidence %):** Thanh hiển thị xác suất dự đoán.
- **Thanh trượt độ nhạy (Sensitivity Threshold):** Tinh chỉnh ngưỡng phát hiện ngay trên giao diện.
- **Hiệu chuẩn nền (Calibrate Baseline):** Nút bấm lấy mẫu phòng trống khi khởi động.
- **Nhật ký sự kiện (Activity Log):** Bảng ghi nhận mốc thời gian người ra/vào phòng.

### 3. Thanh điều khiển Demo nhanh (Bottom Toolbar)
- Cho phép bấm chuyển đổi nhanh giữa 3 trạng thái (*Phòng trống*, *Ngồi thở*, *Di chuyển*) để demo trực quan mà không cần đợi phần cứng.

---

## 🚀 Hướng dẫn cài đặt và Chạy

### 1. Cài đặt thư viện cần thiết

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Khởi chạy ứng dụng

```bash
python main.py
```

Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở trình duyệt tại địa chỉ:
👉 **`http://localhost:8080`**

---

## 📂 Cấu trúc thư mục

```text
frontend/
├── main.py              # File chạy chính của Dashboard (NiceGUI + ECharts)
├── csi_source.py        # Module quản lý dữ liệu CSI (Hỗ trợ Mock và WebSocket)
├── requirements.txt     # Danh sách thư viện phụ thuộc
└── README.md            # Tài liệu hướng dẫn
```
