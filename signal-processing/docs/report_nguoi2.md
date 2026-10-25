# Báo cáo Người 2 — Thu thập và xử lý dữ liệu CSI

## 1. Mục tiêu

Chuẩn bị dữ liệu CSI cho các bước phân tích và xây dựng mô hình của nhóm: kiểm tra schema, đánh giá chất lượng, tạo đặc trưng cơ bản và trực quan hóa amplitude theo frame/thời gian.

## 2. Dữ liệu hiện có

Bộ dữ liệu gồm bốn phiên: phòng trống, người ngồi im, người đi vào và người đi ra. Các nhãn này được gán ở cấp phiên dựa trên tên file. Chưa có mốc thời gian chính xác cho sự kiện đi vào/đi ra nên không thể dùng chúng làm nhãn sự kiện cho từng frame.

## 3. Pipeline

1. Đọc CSV và kiểm tra các cột bắt buộc.
2. Bổ sung `session_id`, `scenario_label`, `source_file` và thông tin phạm vi nhãn.
3. Đánh giá frame đủ/thiếu mẫu, cặp frame-sample trùng, timestamp trùng và sai lệch amplitude so với I/Q.
4. Đánh dấu các sample index có amplitude bằng 0 trong toàn bộ phiên dữ liệu; không tự xóa.
5. Xuất dataset đã bổ sung metadata, báo cáo QC, đặc trưng theo frame và biểu đồ.

## 4. Đầu ra

- `data/processed/csi_dataset_with_session_labels.csv`
- `data/processed/frame_quality_report.csv`
- `data/processed/session_quality_summary.csv`
- `data/processed/preliminary_frame_features.csv`
- `results/*.png`

## 5. Giới hạn và rủi ro

- Số mẫu kỳ vọng 192/frame cần được xác nhận với người phụ trách thu CSI.
- Ý nghĩa của các `sample_index` luôn bằng 0 chưa được xác minh; không được tự động loại bỏ.
- Motion score là chỉ số khám phá, không phải mô hình phát hiện người đã đánh giá độ chính xác.
- Cần bổ sung metadata thiết bị, vị trí, khoảng cách, điều kiện phòng và thời điểm sự kiện.
- Nên chia tập huấn luyện/kiểm tra theo phiên thu để giảm rò rỉ dữ liệu.

## 6. Việc cần xác nhận với Người 1

1. Thiết bị, firmware và phương thức giải mã CSI.
2. Số mẫu CSI kỳ vọng mỗi frame và ý nghĩa của `sample_index`.
3. Nguyên nhân frame không đủ mẫu, timestamp trùng và các sample index bằng 0.
4. Timestamp thực tế của các sự kiện và metadata điều kiện thu.

## 7. Bàn giao cho Người 3

Bàn giao các đặc trưng sơ bộ và báo cáo QC để nghiên cứu mô hình. Người 3 cần tự đánh giá khả năng phân biệt các phiên, chia train/test theo `session_id` và không coi nhãn phiên là nhãn đúng cho từng thời điểm.
