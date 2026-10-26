# Báo cáo Người 2 — Thu thập và xử lý dữ liệu CSI

## 1. Mục tiêu

Chuẩn bị dữ liệu CSI cho các bước phân tích và mô hình của nhóm: kiểm tra schema, đánh giá chất lượng, tạo đặc trưng theo frame và theo thời gian, thử lọc nhiễu, trực quan hóa và rút ra nhận xét dựa trên số liệu thật.

## 2. Dữ liệu hiện có

### 2.1 Bộ ESP32 (cũ) — `data/raw/`

Bốn phiên: phòng trống, người ngồi im, người đi vào, người đi ra. Nhãn gán ở **cấp phiên** dựa trên tên file; không có mốc thời gian cho sự kiện nên không dùng làm nhãn sự kiện theo từng frame. Bộ này đã được điều tra kỹ (45 frame chỉ có 64 mẫu) và kết luận là **frame hợp lệ của nhóm gói NOHT**, không phải lỗi mất dữ liệu; chi tiết nằm ở các script `analyze_*.py` và `verify_log_to_csv.py` trong thư mục repo.

### 2.2 Bộ Intel (mới, đang dùng) — `data/intel_raw/`

Sinh từ `hardware/csi_extracted/labeled/` bằng `src/parse_intel_csi.py`:

| Hạng mục | Giá trị |
|---|---|
| Số phiên dùng chính | 9 (loại `test_01` — lần chạy thử) |
| Lớp | `room_empty` (3 phiên), `person_static` (3), `person_moving` (3) |
| Frame VHT giữ lại | **5053** (1698 / 1684 / 1671) |
| Mẫu mỗi frame | **224** (2 RX × 2 TX × 56 tone) |
| Tổng dòng | 1.131.872 |
| Frame thiếu mẫu | **0** |
| Nhịp lấy mẫu đo được | 101 ms ≈ **9,9 frame/giây** |
| Nhóm bị loại | NOHT (1×1×52, 208 B) — thấy trong cột `frames_filtered_out` (~520/phiên) |

## 3. Pipeline

1. Đọc CSV, kiểm tra các cột bắt buộc, gán `session_id` và `scenario_label` theo tên file.
2. Đánh giá frame đủ/thiếu mẫu, cặp `(frame_id, sample_index)` trùng, timestamp trùng, sai lệch `amplitude` so với `sqrt(i² + q²)`.
3. Gắn cờ các `sample_index` luôn bằng 0 (16 chỉ số = 16 tone null OFDM); **không tự xóa**.
4. Xuất dataset có cờ QC, báo cáo QC theo frame và theo phiên.
5. Đặc trưng theo frame: `mean/std/max_amplitude`, `mean_rssi`, `motion_score`.
6. **Đặc trưng theo thời gian**: `temporal_std_amplitude` — độ lệch chuẩn của **cùng một điểm CSI** qua nhiều frame trong cửa sổ trailing 2 giây (20 frame). Khác với `std_amplitude` (dao động **giữa các điểm trong một frame**, phản ánh chọn lọc tần số).
7. **Thử lọc nhiễu**: trung bình động có tâm 5 frame (≈0,5 s), đánh giá trước/sau.
8. Sinh biểu đồ: heatmap từng phiên, so sánh mean amplitude, motion score, boxplot biến thiên thời gian, before/after lọc.
9. **(Tùy chọn, mặc định tắt)** Chuẩn hóa bảng đặc trưng bằng `--normalization zscore|minmax`, thống kê **chỉ fit trên `--train-sessions`**; xuất thêm `*_normalized.csv` và `normalization_params.csv`. Dữ liệu thô `i/q/amplitude` không bao giờ bị scale.

Hai script kiểm chứng chạy riêng (không thuộc pipeline): `analyze_ping_artifact.py` (nghi vấn cú nhọn do ping) và `analyze_session_drift.py` (nghi vấn trôi nền theo thứ tự thu).

## 4. Đầu ra

Bộ Intel:

- `data/intel/data/processed/csi_dataset_complete_frames.csv` — file chính bàn giao cho Người 3.
- `.../csi_dataset_with_session_labels.csv`, `frame_quality_report.csv`, `session_quality_summary.csv`, `preliminary_frame_features.csv`.
- `.../temporal_features.csv` — biến thiên theo thời gian (kèm `sampling_interval_ms`, `window_frames`).
- `.../filter_evaluation.csv` — đánh giá lọc nhiễu theo phiên.
- `.../*_normalized.csv` + `normalization_params.csv` — chỉ có khi chạy với `--normalization` (mặc định tắt).
- `data/intel/results/*.png`.
- `data/intel/results/ping_periodicity_check.{csv,png}` và `session_drift_check.{csv,png}` — đầu ra của 2 script kiểm chứng (`analyze_ping_artifact.py`, `analyze_session_drift.py`).

Cách chạy lại đúng thứ tự nằm trong `README.md`.

## 5. Nhận xét từ kết quả thực tế

Toàn bộ số liệu dưới đây lấy từ `preliminary_frame_features.csv`, `temporal_features.csv` và `filter_evaluation.csv` sau khi chạy pipeline với `--expected-samples 224 --window-seconds 2 --filter-frames 5`. Khi phân tích phân bố, **bỏ 19 frame đầu của mỗi phiên** (chưa đủ cửa sổ 20 frame).

### 5.1 Ba lớp khác nhau ở đâu trên biểu đồ

**Heatmap** (`heatmap_room_empty_01.png`, `heatmap_person_static_01.png`, `heatmap_person_moving_01.png`):

- `room_empty`: các dải ngang (chọn lọc tần số của kênh) gần như **không đổi theo thời gian**; chỉ có vân dọc mảnh. Nhịp điệu rất tĩnh.
- `person_static`: vẫn thấy dải ngang nhưng có **vết đứt dọc** và gợn dịch chuyển chậm của dải (vd phiên `person_static_01` có đứt nét quanh frame ~200).
- `person_moving`: vân dọc **dày và đổi liên tục từng frame** — nhìn bằng mắt là phân biệt được ngay.

> Ghi chú kỹ thuật: trục hoạ heatmap là `frame_id` gốc (lên tới ~1100) vì `frame_id` do firmware ghi và xen kẽ cả gói NOHT; số frame VHT thực dùng chỉ ~570/phiên. Đây không phải lỗi trục.

**Biểu đồ motion score** (`preliminary_motion_score.png`): ba phiên `person_moving` (đường cao, đỉnh 60–80) tách hẳn lên trên; sáu phiên còn lại (static + empty) chụm dưới 10.

**Boxplot biến thiên thời gian** (`temporal_variability_by_class.png`):

| Lớp | Trung vị | Hộp IQR | Râu |
|---|---:|---|---|
| `person_moving` | 17,70 | 14,63 – 26,13 | 0,0 – 36,2 |
| `person_static` | 4,74 | 3,90 – 6,46 | 2,06 – 10,46 |
| `room_empty` | 3,41 | 2,62 – 3,97 | 2,05 – 5,56 |

Điểm thú vị: `std_amplitude` (trong frame) **không** tách được `person_static` khỏi `room_empty` — mà còn **đảo chiều**: phòng trống có std trong frame **cao hơn** ngồi im (AUC 0,151, tức phân bố trái ngược). `mean_amplitude` gần như vô dụng (AUC 0,31 – 0,68). Nghĩa là thông tin chuyển động nằm ở **phần biến thiên theo thời gian**, không ở biên độ trung bình.

### 5.2 Khác biệt có lặp lại giữa các phiên không?

Trung bình theo từng phiên (một dòng = một lần thu):

| Lớp | Phiên | motion_score | temporal_std |
|---|---|---:|---:|
| `room_empty` | 01 / 02 / 03 | 3,33 / 3,24 / 3,06 | 3,42 / 3,38 / 3,18 |
| `person_static` | 01 / 02 / 03 | 4,67 / 4,69 / 4,64 | 5,19 / 5,50 / 6,47 |
| `person_moving` | 01 / 02 / 03 | 20,85 / 20,40 / 20,11 | 20,60 / 19,51 / 19,85 |

**Có — khác biệt lặp lại khá ổn định.** Ba phiên `person_moving` chỉ chênh nhau ~4%; `room_empty` ~8%; xếp thứ tự `empty < static < moving` đúng ở cả 9 phiên, trên cả hai đặc trưng.

Hai điểm cần nói thật:

1. `person_static` có xu hướng **tăng dần** theo phiên (temporal_std 5,19 → 5,50 → 6,47). Có thể do người ngồi lâu thêm cựa quày/dễ thở mạnh hơn; **đây là suy đoán**, chưa có bằng chứng — cần hỏi lại người thu hoặc thu lại có giám sát.
2. **Thứ tự thu dồn theo lớp** — đã kiểm tra bằng `analyze_session_drift.py` (bảng theo đúng thứ tự thời gian, `session_drift_check.csv`):
   - **Không thấy trôi nền theo thời gian toàn cục**: RSSI trung vị đi từ −52 → −53 dBm trong 14 phút (1 dB, rất nhỏ); biên độ nền của `room_empty` chỉ đổi 0,24 đơn vị (131,48 → 131,56 → 131,72, tức 0,18%) và của `person_moving` chỉ 0,21 đơn vị; tương quan hồ sơ amplitude theo tone giữa `room_empty_01` và `person_moving_03` (cách nhau 14 phút) vẫn là 0,985 — **không giảm theo độ lệch thời gian**.
   - **Nhưng có một dịch chuyển thật bên trong khối `person_static`**: biên độ trung bình 135,03 → 134,25 → 132,27 (biên độ 2,76, gấp ~11 lần biên độ trong khối `room_empty`), và `person_static_01` có hồ sơ tone lệch nhất (tương quan 0,81–0,87 với các phiên khác).
   - **Kết luận trung thực**: bằng chứng hiện có **mâu thuẫn với giả thuyết trôi nền theo thời gian**, nhưng vì lớp và thời gian trùng nhau hoàn toàn (thu dồn theo block) nên bài kiểm tra này **không thể chứng minh được** là không có ảnh hưởng thời gian. Việc thu lại theo **thứ tự trộn** vẫn là cách duy nhất để đóng lại nghi vấn này.

### 5.3 Motion score và temporal_std tách được tới đâu?

Tính AUC cho từng đặc trưng (mỗi dòng so một cặp lớp, trên frame-level):

| Đặc trưng | moving vs empty | moving vs static | static vs empty |
|---|---:|---:|---:|
| `temporal_std_amplitude` | **1,000** | 0,973 | 0,760 |
| `motion_score` | 0,989 | 0,973 | 0,621 |
| `std_amplitude` | 0,638 | 0,824 | 0,151 (nghịch chiều) |
| `mean_amplitude` | 0,456 | 0,313 | 0,678 |

Dùng ngưỡng đơn trên `temporal_std_amplitude`:

- `≥ 10,15` để tách **đi lại vs hai lớp còn lại**: chính xác **96,0%** (sai 4% còn lại chủ yếu là frame đi lại chậm).
- Ngưỡng tốt nhất cho **ngồi im vs phòng trống**: **74,4%** — mức này **chưa đủ dùng** cho một ngưỡng đơn.

Kết luận trung thực:

- **Đi lại ↔ còn lại**: tách tốt, cả bằng ngưỡng đơn lẫn bằng AUC.
- **Ngồi im ↔ phòng trống**: **vẫn chồng lấn** ở frame-level (AUC 0,76 với temporal_std, 0,62 với motion_score). Tách được ở mức trung bình phiên, nhưng từng frame thì không chắc. Người 3 cần đặc trưng mạnh hơn (miền tần số, đa khung cửa sổ) hoặc mô hình học máy thay vì một ngưỡng.

### 5.4 Kết quả thử lọc nhiễu

Bộ lọc đã thử: **trung bình động có tâm, 5 frame (0,5 s)** trên từng điểm CSI dọc trục thời gian.

Theo phiên (`filter_evaluation.csv`): giảm nhiễu tần số cao **70,5 – 80,1%**; tương quan tín hiệu chậm (sau khi làm mượt 2 s) **0,977 – 0,998**.

Theo lớp:

| Lớp | Giảm nhiễu TB | Jitter sau lọc | Biến thiên thời gian còn lại |
|---|---:|---:|---:|
| `person_moving` | 77,5% | 6,31 | 47,6% |
| `person_static` | 73,1% | 2,17 | 58,8% |
| `room_empty` | 80,0% | 1,00 | 40,1% |

**Bộ lọc có giữ được biến động liên quan chuyển động không? Có** — kiểm chứng bằng hai cách:

1. Tương quan tín hiệu chậm trước/sau lọc ≥ 0,977 ở cả 9 phiên (nội dung chậm — phần phản ánh trạng thái — không bị phá).
2. Tương phản giữa các lớp **không giảm mà còn tăng nhẹ**: jitter `moving/empty` đi từ 5,61 lên 6,31; biến thiên thời gian `moving/empty` đi từ 5,94 lên 7,04. Nghĩa là bộ lọc bỏ nhiễu đều ở cả ba lớp chứ không "ăn" mất tín hiệu của người đi lại.

Kiểm chứng thêm bằng cách đổi độ rộng bộ lọc (chạy lại với `--filter-frames`):

| `filter-frames` | Giảm nhiễu (moving / empty) | Tương quan tín hiệu chậm (thấp nhất) | Tương phản jitter moving/empty |
|---:|---|---:|---:|
| 3 (~0,3 s) | 63,3% / 66,2% | 0,986 | 6,10 |
| **5 (~0,5 s)** | **77,5% / 80,0%** | **0,977** | **6,31** |
| 7 (~0,7 s) | 83,7% / 85,6% | 0,969 | 6,38 |
| 9 (~0,9 s) | 87,3% / 88,7% | 0,960 | 6,28 |
| 13 (~1,3 s) | 91,2% / 92,3% | 0,940 | 6,36 |

Hai điều đáng chú ý:

1. Mức giảm nhiễu sát với **lý thuyết nhiễu trắng** (giảm ≈ `1 − 1/w`: 66,7% / 80,0% / 88,9% cho w = 3/5/9). Nghĩa là jitter giữa các frame **gần như trắng** (không có cấu trúc thời gian đáng kể ở thang ~0,1–1 s) — bộ lọc bỏ đúng phần đó.
2. Vì jitter chủ yếu là nhiễu trắng, **`retained_variance_pct` thấp (~20–42%) là bình thường và đúng**: nó chỉ ra phần bị bỏ đi chính là nhiễu trắng, không phải tín hiệu chậm. Đã chọn `filter-frames = 5` vì giảm nhiễu ~80% mà vẫn giữ tương quan tín hiệu chậm ≥ 0,977; rộng hơn nữa thì bắt đầu ăn vào băng tần đi bộ (chu kỳ bước ~1–2 Hz).

**Lưu ý quan trọng cho Người 3:** dữ liệu xuất (`csi_dataset_complete_frames.csv`) **không bị lọc**. Nếu lọc để tính đặc trưng jitter thì nên tính jitter trên **dữ liệu thô** (vì lọc bỏ mất ~50–60% biến thiên thời gian), còn lọc chỉ dùng để trích tín hiệu chậm/đường nền.

### 5.5 Biểu đồ CHƯA chứng minh được điều gì

Nói thẳng để khỏi bị hỏi ngược:

1. **Chưa chứng minh lọc nhiễu làm mô hình tốt hơn.** Mới chỉ chứng minh lọc giảm nhiễu và **giữ được** nội dung chậm/tương phản lớp. Việc lọc có giúp accuracy tăng hay không là việc của Người 3 phải thử.
2. **Nhãn là nhãn phiên, không phải nhãn thời điểm.** Trong 60 s của `person_moving`, người có thể đứng nghỉ; ta không biết frame nào đang thực sự di chuyển nên không thể khẳng định "frame này bị gán sai nhãn".
3. **Chưa tổng quát hóa được.** Một phòng, một người, một AP, một ngày, một vị trí. Chưa nói được gì về người khác, phòng khác, ngày khác.
4. **Thứ tự thu dồn theo lớp** (xem 5.2): bài kiểm tra `analyze_session_drift.py` không tìm thấy trôi nền theo thời gian toàn cục, nhưng vì lớp và thời gian trùng nhau nên vẫn **chưa loại trừ được** phỏng đoán "dịch chuyển môi trường trùng với nhãn" — chỉ thu lại theo thứ tự trộn mới đóng được.
5. **Các cú nhọn amplitude của `room_empty`** — đã kiểm chứng bằng `analyze_ping_artifact.py` (`ping_periodicity_check.csv`); **giả thuyết beat giữa ping 10 Hz với nhịp mẫu 9,9 Hz KHÔNG được dữ liệu hỗ trợ**:

   | Kiểm tra | Kỳ vọng nếu do ping 10 Hz | Thực tế (3 phiên `room_empty`)
   |---|---|---|
   | Khoảng cách giữa các cú nhọn | ~100 ms | **906 / 1210 / 1310 ms** (IQR 428–2317 ms) |
   | ACF tại đúng chu kỳ alias (10,1 s ≈ 100 frame) | dương, vượt ngưỡng 0,082 | **−0,015 / +0,018 / −0,031** — bằng 0 |
   | Công suất phổ tại 0,099 Hz so với lân cận | ≥ 2 và lặp lại ở cả 3 phiên | **0,91 / 4,76 / 1,30** — không ổn định (đối chứng `person_moving` cũng cho 2,62 / 2,06) |
   | Số đỉnh ACF vượt ngưỡng trong 0,3–12 s | tập trung ở một độ trễ cố định | 1 / 1 / 5 đỉnh, độ trễ **2121 / 8080 / 4545 ms** — không trùng nhau |

   **Vẫn còn thật**: khoảng 36–39 cú nhọn mỗi phút ở cả 3 phiên phòng trống (z ≈ 2,4–2,8 trên nền 131,5), trung vị khoảng 1 s một cú nhưng phân bố rất rộng (IQR 0,43–2,3 s). Tức là **có hiện tượng nhọn lặp lại xấp xỉ mỗi 1 s, không phải mỗi 100 ms** — và nguyên nhân thì **chưa rõ** (không phải ping). Cách xác nhận cuối cùng vẫn là thu 30 s với ping tắt hoặc đổi tần số ping 5/20 Hz; nếu chu kỳ các cú nhọn không đổi thì loại hẳn nguyên nhân ping.
6. **`person_static` có std trong frame thấp hơn `room_empty`** (AUC 0,151, nghịch chiều) — mới là quan sát, chưa có cơ chế giải thích. Có thể thử nghiệm: cho người ra vào phòng đúng vị trí đó nhiều lần xem hiệu ứng có lặp lại không.

## 6. Metadata các phiên thu — đã bổ sung hồi tố

`_sessions_metadata.csv` điền sẵn mọi thứ **đọc được từ file** (thời gian bắt đầu/kết thúc, AP, tần số, kênh, độ rộng, số ăng-ten, RSSI, số frame, số dòng, tone null…). Bốn cột còn lại do người thu bổ sung **ngày 2026-10-26 (hồi tố, không ghi tại chỗ)**:

| Cột | Giá trị đã điền (cả 9 phiên) |
|---|---|
| `distance_m` | **2.0 m** (AP ↔ laptop) — **ước lượng, không đo bằng thước** |
| `position` | `room_empty`: không có người giữa AP và laptop, đường thẳng thông thoáng · `person_static`: người ngồi im tại một vị trí cố định **trên đường thẳng AP–laptop** · `person_moving`: người đi lại liên tục khắp phòng |
| `environment` | Chỉ có duy nhất **1 người** trong phòng (người được đo); không có người khác; không di chuyển thiết bị, không đổi cấu hình Wi-Fi giữa các lần thu; kích thước và vật liệu phòng **vẫn chưa ghi nhận** |
| `notes` | Hoạt động diễn ra **suốt phiên** (phòng trống / ngồi im không rời chỗ / đi lại liên tục không ngắt quãng); ghi bổ sung ngày 2026-10-26 |

**Ba điều những thông tin này giải thích được cho số liệu:**

1. Người ngồi im **đúng trên đường thẳng AP–laptop** (2 m, LOS bị chặn bởi thân người) → giải thích vì sao `person_static` làm biên độ nền nhảy lên 135,0 so với 131,5 của `room_empty`. Nếu sau này muốn kiểm chứng, chỉ cần cho người ngồi lệch khỏi đường thẳng đó.
2. `person_moving` **di chuyển liên tục, không ngắt quãng** → nhãn phiên hợp lệ cho cả 60 s; ta không phải lo "trong phiên đi lại có khoảng đứng yên".
3. **Không có gì thay đổi giữa các lần thu** → củng cố kết quả của `analyze_session_drift.py` (không trôi nền theo thời gian).

**Vẫn còn thiếu:** kích thước và vật liệu phòng (cho `environment`); khoảng cách chỉ là ước lượng (nếu cần dùng làm biến số phải đo lại). Và vì khoảng cách luôn là 2 m trong cả 9 phiên, **bộ dữ liệu này không nói được gì về ảnh hưởng của khoảng cách** — đó là giới hạn, không phải lỗi.

## 7. Giới hạn và rủi ro

- Nhãn cấp phiên, không có nhãn theo thời điểm.
- Số mẫu kỳ vọng 192 (ESP32) và 224 (Intel VHT) đã được **xác nhận từ dữ liệu thật**; mọi code dùng `--expected-samples` phải khớp bộ dữ liệu đang xử lý.
- 16 `sample_index` luôn bằng 0 tương ứng 16 tone null OFDM; vẫn giữ trong dữ liệu, chỉ ẩn khỏi heatmap.
- Motion score và temporal std là đặc trưng khám phá, chưa phải detector đã đánh giá độ chính xác trên dữ liệu ngoài.
- Chưa chuẩn hóa dữ liệu ở dạng xuất mặc định; pipeline đã có `--normalization zscore|minmax` + `--train-sessions` (thống kê **chỉ fit trên các phiên train**), mặc định tắt — chờ chốt cách chia train/test với Người 3.

## 8. Việc cần xác nhận với Người 1

1. ~~Vị trí và khoảng cách thiết bị khi thu~~ — **đã bổ sung hồi tố ngày 2026-10-26** (xem mục 6); còn thiếu kích thước/vật liệu phòng và một phép đo khoảng cách thật sự.
2. Có thể thu thêm 2 phiên `entering`/`leaving` trên chính bộ Intel để đủ 4 lớp như bộ ESP32 không.
3. Khi thu lần sau: **trộn thứ tự lớp**, ghi lại thời điểm bắt đầu/kết thúc hoạt động của người. Riêng nghi vấn ping: số liệu hiện tại **không ủng hộ** giả thuyết này (xem 5.5) nên việc tắt/đổi tần số ping chỉ còn cần nếu muốn loại trừ chắc chắn 100%.
4. Xác nhận lại 16 tone null và ý nghĩa `sample_index` trong khung 2×2×56.

## 9. Bàn giao cho Người 3

- Dataset chính: `data/intel/data/processed/csi_dataset_complete_frames.csv` (5053 frame × 224 mẫu, nhãn 3 lớp).
- Đặc trưng: `preliminary_frame_features.csv` (trong frame) + `temporal_features.csv` (cửa sổ 2 s).
- QC: `frame_quality_report.csv`, `session_quality_summary.csv`. Đánh giá lọc: `filter_evaluation.csv`.

Yêu cầu sử dụng:

1. Chia train/test theo `session_id`, không chia ngẫu nhiên từng frame.
2. Không coi `scenario_label` là nhãn đúng cho từng thời điểm.
3. Nếu chuẩn hóa, dùng `--normalization zscore|minmax --train-sessions <id>` của pipeline (hoặc scaler riêng của Người 3) — **fit chỉ trên tập train**; tham số đã fit nằm ở `normalization_params.csv` để tái dùng cho dữ liệu mới.
4. `temporal_std_amplitude` hiện là đặc trưng mạnh nhất cho lớp `person_moving`; `person_static` vs `room_empty` cần đặc trưng khác hoặc mô hình học máy.
5. Muốn thêm lớp `multiple_people` thì phải thu — hiện chưa có dữ liệu.
