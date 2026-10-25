# Người 2 — CSI Data Processing

Module thu thập/kiểm tra chất lượng, tiền xử lý cơ bản và trực quan hóa dữ liệu Wi-Fi CSI phục vụ đồ án phát hiện sự hiện diện/chuyển động của con người.

## Phạm vi hiện tại

- Đọc CSV CSI từ `data/raw/`.
- Gắn nhãn **cấp phiên thu** dựa trên tên file.
- Kiểm tra số mẫu theo frame, cặp `(frame_id, sample_index)` trùng, timestamp trùng và độ lệch giữa `amplitude` với `sqrt(i² + q²)`.
- Gắn cờ các `sample_index` có amplitude bằng 0 trong toàn bộ dữ liệu; không xóa chúng khỏi dữ liệu xuất.
- Xuất dữ liệu có nhãn phiên, báo cáo chất lượng và đặc trưng thống kê sơ bộ theo frame.
- Sinh heatmap amplitude, so sánh amplitude và biểu đồ motion score khám phá.

**Giới hạn:** nhãn hiện tại là nhãn phiên, không phải nhãn sự kiện theo từng thời điểm. Motion score chỉ là đặc trưng khám phá, chưa phải detector đã được xác thực. `expected-samples=192` là giả định cấu hình ban đầu và cần được xác nhận với Người 1/phụ trách phần cứng. Không tự kết luận mọi sample index là subcarrier hợp lệ.

## Cấu trúc

```text
Nguoi2_CSI_DataProcessing/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # CSV gốc, giữ nguyên
│   └── processed/    # CSV đầu ra của pipeline
├── src/
│   └── csi_preprocessing_pipeline.py
├── tests/
│   └── test_pipeline_smoke.py
├── results/          # PNG biểu đồ
└── docs/
    ├── report_nguoi2.md
    └── metadata_template.csv
```

## Cài đặt

Khuyến nghị Python 3.10 hoặc mới hơn.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell không cho phép activate môi trường, có thể dùng `\.venv\Scripts\python.exe -m pip install -r requirements.txt` và gọi Python trong môi trường đó trực tiếp.

## Chạy pipeline

Từ thư mục gốc `Nguoi2_CSI_DataProcessing`:

```bash
python src/csi_preprocessing_pipeline.py --input-dir data/raw --output-dir . --expected-samples 192
```

Kết quả được lưu vào:

- `data/processed/csi_dataset_with_session_labels.csv`
- `data/processed/frame_quality_report.csv`
- `data/processed/session_quality_summary.csv`
- `data/processed/preliminary_frame_features.csv`
- `results/*.png`

Tham số `--expected-samples` phải được điều chỉnh nếu Người 1 xác nhận số mẫu kỳ vọng mỗi frame khác 192.

## Các cột dữ liệu chính

- `frame_id`: định danh frame trong dữ liệu nguồn.
- `timestamp_ms`: timestamp theo millisecond được thiết bị ghi lại.
- `sample_index`: chỉ số mẫu trong frame; cần xác nhận ý nghĩa vật lý với người thu dữ liệu.
- `i`, `q`: thành phần I/Q được giải mã.
- `amplitude`: biên độ nguồn; thường kỳ vọng gần `sqrt(i² + q²)`, có thể khác nhỏ do làm tròn.
- `scenario_label`: nhãn cấp phiên suy ra từ tên file.
- `frame_complete`: cờ QC dựa trên số lượng mẫu kỳ vọng.
- `motion_score_raw`: trung bình độ thay đổi tuyệt đối amplitude giữa hai frame hoàn chỉnh liên tiếp; chỉ dùng khám phá.

## Chạy kiểm thử

```bash
python -m pytest -q
```

## Cần xác nhận trước khi kết luận

1. Thiết bị/firmware và phương thức giải mã CSI.
2. Vì sao có một số `sample_index` luôn bằng 0 và liệu chúng có phải mẫu padding/không hợp lệ.
3. Số mẫu kỳ vọng thật sự của mỗi frame.
4. Ý nghĩa và đơn vị của `timestamp_ms`, cách xử lý timestamp trùng.
5. Timestamp thực tế khi người đi vào/đi ra và điều kiện môi trường của từng phiên.
6. Cách chia train/test theo phiên để tránh rò rỉ giữa các frame cùng một phiên.

## Bàn giao cho Người 3

Dùng `preliminary_frame_features.csv` để khảo sát đặc trưng, nhưng hãy chia dữ liệu train/test theo `session_id`, không chia ngẫu nhiên từng frame của cùng phiên. Không coi `scenario_label` là nhãn thời gian chính xác cho từng frame khi sự kiện chỉ diễn ra trong một phần phiên.

## Chi tiết
Mảng Người 2 đã hoàn thành phần xử lý và kiểm tra chất lượng dữ liệu CSI Intel rồi nhé mọi người.

Hiện tại dataset có 5.053 frame, gồm 3 lớp `room_empty`, `person_static`, `person_moving`. Dữ liệu đã được kiểm tra, không có frame thiếu mẫu hay lỗi chất lượng theo các tiêu chí đã kiểm tra.

Người 3 có thể lấy dataset chính tại:

`signal-processing/data/intel/data/processed/csi_dataset_complete_frames.csv`

Ngoài ra, có các file hỗ trợ:
- `preliminary_frame_features.csv`: các đặc trưng đã tính theo frame.
- `frame_quality_report.csv`: báo cáo chất lượng frame.
- `session_quality_summary.csv`: tổng hợp chất lượng các phiên thu.

Metadata các phiên nằm tại `signal-processing/data/intel_raw/_sessions_metadata.csv`.

**Lưu ý:** Dataset hiện có 3 lớp, chưa có `multiple_people`; dữ liệu chưa được lọc nhiễu hoặc chuẩn hóa. Người 3 có thể bắt đầu kiểm tra dữ liệu và xây dựng mô hình baseline trên dataset hiện tại trước nhé. Nếu cần bổ sung hoặc thay đổi cách tiền xử lý thì báo lại nhóm để thống nhất.