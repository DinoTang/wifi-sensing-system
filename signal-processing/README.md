# Người 2 — CSI Data Processing

Module thu thập/kiểm tra chất lượng, tiền xử lý cơ bản và trực quan hóa dữ liệu Wi-Fi CSI phục vụ đồ án phát hiện sự hiện diện/chuyển động của con người.

## Phạm vi hiện tại

Repo đang xử lý **hai bộ dữ liệu riêng biệt**, dùng chung một pipeline nhưng với cấu hình khác nhau:

| | Bộ ESP32 (cũ) | Bộ Intel (mới, đang dùng) |
|---|---|---|
| Vị trí | `data/raw/` | `data/intel_raw/` (sinh từ `hardware/csi_extracted/labeled/`) |
| Nguồn | Serial 921600 baud (`hardware/capture_csi_serial.py`) | `raw.csi` nhị phân + `recorder.log` |
| Số mẫu/frame | **192** | **224** (nhóm VHT: 2 RX × 2 TX × 56 tone) |
| Nhãn | `phong_trong`, `nguoi_ngoi_im`, `nguoi_di_vao`, `nguoi_di_ra` | `room_empty`, `person_static`, `person_moving` |
| Đầu ra | `data/processed/`, `results/` | `data/intel/data/processed/`, `data/intel/results/` |

Các bước pipeline:

- Đọc CSV CSI, gán nhãn **cấp phiên thu** theo tên file.
- Kiểm tra số mẫu theo frame, cặp `(frame_id, sample_index)` trùng, timestamp trùng và độ lệch giữa `amplitude` với `sqrt(i² + q²)`.
- Gắn cờ các `sample_index` có amplitude bằng 0 trong toàn bộ dữ liệu; **không xóa** chúng khỏi dữ liệu xuất.
- Xuất dữ liệu có nhãn phiên, báo cáo chất lượng và đặc trưng thống kê sơ bộ theo frame.
- Tính **đặc trưng biến thiên theo thời gian** (cùng một điểm CSI so qua nhiều frame trong cửa sổ 1–2 giây).
- Chạy **thử nghiệm lọc nhiễu** before/after và xuất số liệu đánh giá.
- (**Tùy chọn**) **chuẩn hóa** bảng đặc trưng theo `--normalization zscore|minmax`, thống kê chỉ fit trên `--train-sessions` — mặc định tắt.
- Sinh heatmap amplitude, so sánh amplitude, motion score, boxplot biến thiên thời gian và biểu đồ trước/sau lọc.

Hai script kiểm chứng độc lập (chạy riêng, không thuộc pipeline):

- `analyze_ping_artifact.py` — kiểm tra xem các cú nhọn amplitude ở phòng trống có chu kỳ 100 ms (bằng lưu lượng ping 10 Hz) hay không (ACF + periodogram + khoảng cách giữa các cú nhọn).
- `analyze_session_drift.py` — kiểm tra xem 9 phiên có bị “trôi nền” theo thứ tự thu hay không (RSSI, biên độ nền, hồ sơ amplitude theo tone).

**Giới hạn:** nhãn là nhãn **phiên**, không phải nhãn sự kiện theo từng thời điểm. Motion score và temporal std là đặc trưng khám phá, chưa phải detector đã được xác thực. Pipeline **không chuẩn hóa theo mặc định** (xem mục "Chuẩn hóa").

## Nguồn gốc của 5.053 frame (bộ Intel)

Chuyển từ **9 phiên thu** (bộ `labeled/`, thu ngày 2026-10-25, mỗi phiên 60 giây). Phiên `test_01` là lần chạy thử nên **bị loại** (chỉ thêm `--include-test` mới lấy). Parser mặc định giữ **nhóm VHT** (2×2×56, 896 B, 224 mẫu); nhóm NOHT (1×1×52, 208 B, 52 mẫu) bị loại khỏi CSV — số này vẫn thấy trong cột `frames_filtered_out` của metadata.

| Lớp (nhãn) | Phiên thu | Frame/phiên | Tổng frame |
|---|---|---|---:|
| `room_empty` | `room_empty_01`, `_02`, `_03` | 572 · 568 · 558 | **1698** |
| `person_static` | `person_static_01`, `_02`, `_03` | 572 · 555 · 557 | **1684** |
| `person_moving` | `person_moving_01`, `_02`, `_03` | 558 · 561 · 552 | **1671** |
| **Tổng** | **9 phiên** | ~561/phiên | **5053** |

Tổng số dòng: 1.131.872 (= 5.053 × 224). Frame thiếu mẫu: **0**. Tỷ lệ lớp xấp xỉ cân bằng (33,6% / 33,3% / 33,1%).

Mỗi phiên ghi được ~1100 frame cả hai nhóm (18 frame/giây), trong đó ~570 frame VHT được giữ lại → **nhịp lấy mẫu thực tế 101 ms ≈ 9,9 frame/giây** (đã kiểm tra lại trong pipeline, xem mục biến thiên thời gian).

## Cấu trúc

```text
signal-processing/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # CSV gốc của bộ ESP32 — giữ nguyên
│   ├── processed/            # đầu ra pipeline bộ ESP32
│   ├── intel_raw/            # CSV sinh từ raw.csi (bộ Intel)
│   │   └── _sessions_metadata.csv   # metadata từng phiên (bắt đầu bằng "_" = không phải phiên thu)
│   └── intel/
│       ├── data/processed/   # đầu ra pipeline bộ Intel
│       └── results/          # PNG biểu đồ bộ Intel
├── src/
│   ├── parse_intel_csi.py            # raw.csi + recorder.log -> CSV
│   └── csi_preprocessing_pipeline.py # CSV -> dataset + QC + đặc trưng + biểu đồ
├── analyze_ping_artifact.py          # kiểm chứng nghi vấn cú nhọn do ping
├── analyze_session_drift.py          # kiểm chứng nghi vấn trôi nền theo thứ tự thu
├── tests/
│   └── test_pipeline_smoke.py
├── results/                  # PNG biểu đồ bộ ESP32
└── docs/
    └── report_nguoi2.md
```

## Cài đặt

Khuyến nghị Python 3.10 hoặc mới hơn.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell không cho phép activate môi trường, có thể dùng `.venv\Scripts\python.exe -m pip install -r requirements.txt` và gọi Python trong môi trường đó trực tiếp.

## Chạy lại từ đầu (thứ tự bắt buộc)

Từ thư mục `signal-processing/`:

**Bước 1 — chuyển `raw.csi` thành CSV** (chỉ cần làm lại khi có phiên thu mới):

```bash
python src/parse_intel_csi.py                  # mặc định: nhóm VHT -> data/intel_raw/
python src/parse_intel_csi.py --group noht     # nếu muốn nhóm NOHT (52 mẫu)
python src/parse_intel_csi.py --include-test   # thêm phiên test_01
```

> Chạy lại parser **không xoá** 4 cột metadata điền tay (`distance_m`, `position`, `environment`, `notes`): parser đọc file `_sessions_metadata.csv` cũ và giữ lại các giá trị khác `chưa ghi nhận`, chỉ sinh `chưa ghi nhận` cho phiên hoàn toàn mới.

**Bước 2 — chạy pipeline tiền xử lý:**

```bash
# Bộ Intel (dang dùng):
python src/csi_preprocessing_pipeline.py \
    --input-dir data/intel_raw \
    --output-dir data/intel \
    --expected-samples 224

# Bộ ESP32 cũ:
python src/csi_preprocessing_pipeline.py \
    --input-dir data/raw \
    --output-dir . \
    --expected-samples 192

# Thêm chuẩn hóa (CHỈ khi đã chốt cách chia train/test với Người 3):
python src/csi_preprocessing_pipeline.py \
    --input-dir data/intel_raw \
    --output-dir data/intel \
    --expected-samples 224 \
    --normalization zscore \
    --train-sessions room_empty_01,room_empty_02,room_empty_03,person_static_01
```

**Bước 3 — chạy kiểm thử:**

```bash
python -m pytest -q
```

> `--expected-samples` phải khớp **số mẫu thực tế của bộ dữ liệu đang xử lý**: 192 với CSV ESP32, **224** với CSV Intel nhóm VHT. Sai giá trị này thì mọi frame bị báo "thiếu mẫu".

## Đầu ra

Bộ Intel (`--output-dir data/intel`):

- `data/intel/data/processed/csi_dataset_with_session_labels.csv` — toàn bộ 1.131.872 dòng, có cờ QC.
- `data/intel/data/processed/csi_dataset_complete_frames.csv` — **file chính bàn giao cho Người 3** (chỉ frame đủ mẫu).
- `data/intel/data/processed/frame_quality_report.csv` — QC theo frame.
- `data/intel/data/processed/session_quality_summary.csv` — QC theo phiên.
- `data/intel/data/processed/preliminary_frame_features.csv` — đặc trưng theo frame (mean/std/max amplitude, rssi, motion_score).
- `data/intel/data/processed/temporal_features.csv` — **biên động theo thời gian**: `temporal_std_amplitude`, `temporal_std_amplitude_max`, cùng `sampling_interval_ms`, `sampling_fps`, `window_frames`.
- `data/intel/data/processed/filter_evaluation.csv` — **đánh giá lọc nhiễu** theo phiên.
- `data/intel/data/processed/*_normalized.csv` + `normalization_params.csv` — chỉ có khi chạy với `--normalization zscore|minmax`.
- `data/intel/results/*.png` — heatmap từng phiên + `mean_amplitude_by_session.png` + `preliminary_motion_score.png` + `temporal_variability_by_class.png` + `filter_before_after.png`.
- `data/intel/results/ping_periodicity_check.{csv,png}` và `session_drift_check.{csv,png}` — đầu ra của 2 script kiểm chứng (chạy riêng).

Bộ ESP32: cùng danh sách file nhưng ở `data/processed/` và `results/`.

## Biến thiên theo thời gian và thử lọc nhiễu

Hai tham số (đặt trong lúc gọi pipeline):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--window-seconds` | `2.0` | Độ dài cửa sổ theo **giây**; pipeline tự đổi sang frame bằng nhịp lấy mẫu đo được (2,0 s × 9,9 fps → 20 frame). |
| `--filter-frames` | `5` | Độ rộng bộ lọc trung bình động (≈0,5 s tại 9,9 fps). Đặt `< 2` để tắt lọc. |

- `temporal_std_amplitude`: độ lệch chuẩn của **cùng một điểm CSI** qua các frame trong cửa sổ trailing — đây là phần phản ánh chuyển động, khác với `std_amplitude` (độ lệch giữa các điểm **trong** một frame, phản ánh chọn lọc tần số của kênh).
- `filter_evaluation.csv`: so sánh trước/sau lọc với 3 nhóm chỉ số — giảm nhiễu tần số cao (`noise_reduction_pct`), giữ tín hiệu chậm (`low_frequency_correlation`, tương quan sau khi làm mượt 2 s), giữ biến động liên quan chuyển động (`temporal_std_retained_pct`).
- **Bộ lọc là centered moving average (dùng cả quá khứ lẫn tương lai)** → chỉ dùng cho phân tích ngoại tuyến; nếu đưa vào hệ thống thời gian thực phải đổi sang cửa sổ trailing.
- **Dữ liệu xuất ra KHÔNG bị thay đổi bởi bộ lọc**: series đã lọc chỉ nằm trong biểu đồ `filter_before_after.png` và báo cáo `filter_evaluation.csv`. Dataset gốc giữ nguyên để Người 3 tự quyết định tiền xử lý.

### Chuẩn hóa (thống nhất với Người 3)

**Mặc định: tắt.** Pipeline không chuẩn hóa gì cả — `csi_dataset_complete_frames.csv` vẫn là amplitude thô.

Khi nào cần thì thêm hai tham số:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--normalization` | `none` (mặc định) / `zscore` / `minmax` | Cách scale **bảng đặc trưng** (`preliminary_frame_features`, `temporal_features`). Dữ liệu thô `i/q/amplitude` **không bao giờ** bị scale. |
| `--train-sessions` | danh sách `session_id` cách nhau bằng dấu `,` | Các phiên dùng để **fit** thống kê (mean/std hoặc min/max). Thống kê áp dụng cho **tất cả** các dòng, kể cả dòng test. |

Kết quả:

- `preliminary_frame_features_normalized.csv`, `temporal_features_normalized.csv` — bảng đã scale (chỉ tạo khi bật).
- `normalization_params.csv` — từng cột: `center`, `scale`, số dòng fit, phiên dùng để fit — để Người 3 tái sử dụng đúng scaler khi dự đoán dữ liệu mới.

Ba quy tắc:

1. **Không fit trên toàn bộ dữ liệu.** Nếu thiếu `--train-sessions` thì pipeline in cảnh báo rò rỉ và vẫn scale theo toàn bộ — đừng dùng kết quả đó cho báo cáo.
2. `--train-sessions` chứa `session_id` không tồn tại sẽ **báo lỗi ngay** (không âm thầm dùng tất cả).
3. Cột nào không đổi trên tập train (vd `mean_rssi` = −52 everywhere) bị **bỏ qua**, giữ nguyên giá trị thô, ghi chú trong `normalization_params.csv`.

> Chưa chốt cách chia train/test với Người 3 → **chưa chạy thật** với hai tham số này; code và test đã có sẵn, họp xong chỉ cần truyền tham số vào.

## Các cột dữ liệu chính

- `frame_id`: định danh frame trong phiên thu.
- `timestamp_ms`: timestamp theo millisecond do thiết bị ghi lại.
- `sample_index`: chỉ số mẫu trong frame (bộ Intel: 0–223 ứng 4 khối × 56 tone).
- `i`, `q`: thành phần I/Q giải mã — **lưu ý:** ESP32 là int8 (±127), Intel là int16 (±300).
- `amplitude`: biên độ nguồn; thường gần `sqrt(i² + q²)`, có thể khác nhỏ do làm tròn.
- `scenario_label`: nhãn cấp phiên suy ra từ tên file.
- `frame_complete`: cờ QC dựa trên số mẫu kỳ vọng.
- `motion_score_raw`: trung bình độ thay đổi tuyệt đối amplitude giữa hai frame hoàn chỉnh liên tiếp; chỉ dùng khám phá.
- `temporal_std_amplitude` (file `temporal_features.csv`): độ biến động cùng điểm CSI qua nhiều frame trong cửa sổ.

## Chạy kiểm thử

```bash
python -m pytest -q
```

## Cần xác nhận trước khi kết luận

1. **Metadata phiên**: 4 cột `distance_m`, `position`, `environment`, `notes` trong `_sessions_metadata.csv` **đã bổ sung hồi tố ngày 2026-10-26** (khoảng cách AP–laptop ≈ 2 m — ước lượng; người ngồi im nằm đúng trên đường thẳng AP–laptop; đi lại liên tục suốt phiên; chỉ 1 người trong phòng; không đổi thiết bị/Wi-Fi giữa các lần thu). **Vẫn còn thiếu**: kích thước/vật liệu phòng và một phép đo khoảng cách thật sự. Các lần thu sau điền trực tiếp vào phiếu [docs/capture_checklist.md](docs/capture_checklist.md); khi chạy lại `parse_intel_csi.py`, các cột đã điền tay **được giữ nguyên** (parser tự đọc lại file metadata cũ).
2. Thứ tự thu hiện là dồn từng lớp (`room_empty` 05:24–05:27 → `person_static` 05:31–05:34 → `person_moving` 05:36–05:39). Đã chạy `analyze_session_drift.py`: **không thấy trôi nền theo thời gian toàn cục** (RSSI −52 → −53 dBm; biên độ nền của `room_empty` và `person_moving` ổn định tới 0,2–0,24 đơn vị; tương quan hồ sơ tone không giảm theo độ lệch thời gian). **Nhưng** bên trong khối `person_static` có dịch chuyển thật (biên độ trung bình 135,0 → 134,3 → 132,3), nên vẫn chưa loại trừ được hoàn toàn “dịch chuyển môi trường trùng với nhãn”. Lần thu sau nên trộn thứ tự lớp.
3. Vì sao một số `sample_index` luôn bằng 0 (16 chỉ số = 16 tone null của OFDM) và xác nhận chúng với Người 1.
4. Ý nghĩa và đơn vị của `timestamp_ms`, cách xử lý timestamp trùng.
5. Cách chia train/test theo phiên để tránh rò rỉ giữa các frame cùng một phiên.

## Bàn giao cho Người 3

- Dataset chính: `data/intel/data/processed/csi_dataset_complete_frames.csv`.
- Đặc trưng: `preliminary_frame_features.csv` (trong frame) và `temporal_features.csv` (qua frame, cửa sổ 2 s).
- Báo cáo QC: `frame_quality_report.csv`, `session_quality_summary.csv`.
- Đánh giá lọc nhiễu: `filter_evaluation.csv`.

Yêu cầu khi dùng:

1. Chia train/test theo `session_id` (không chia ngẫu nhiên từng frame của cùng phiên).
2. Không coi `scenario_label` là nhãn đúng cho từng thời điểm.
3. Nếu chuẩn hóa: dùng `--normalization zscore|minmax --train-sessions <id>` của pipeline (hoặc fit scaler riêng của Người 3) — **chỉ fit trên tập train**; tham số đã fit nằm ở `normalization_params.csv` để tái dùng cho dữ liệu mới.
4. Chưa dùng series đã lọc như dữ liệu đầu vào mặc định — xem số liệu `filter_evaluation.csv` trước.

## Chi tiết

Kết quả phân tích chi tiết (3 lớp khác nhau ở đâu, có lặp lại giữa các phiên không, motion score tách tới đâu) nằm trong [docs/report_nguoi2.md](docs/report_nguoi2.md).
