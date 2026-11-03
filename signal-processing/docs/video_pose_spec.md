# Bước 3–5–6: chốt thông số trước khi xử lý hàng loạt

Tài liệu này ghi **những gì đã có**, **những gì còn thiếu**, và **các thông số phải thống nhất với Người 3** trước khi ghép CSI với tư thế lấy từ video. Mục đích: khi video được bàn giao, việc xử lý chỉ còn là chạy lệnh, không phải quyết định lại.

Trạng thái tính đến 2026-11-03. Số liệu CSI lấy từ `check_dataset_readiness.py` trên 10 phiên hiện có.

---

## 1. Đầu vào: có gì, thiếu gì

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| `raw.csi`, `recorder.log` | **có** (10 phiên) | 1.100 bản ghi/phiên, khớp số bản ghi ở cả 10 phiên |
| `metadata.txt`, `start-time.txt`, `end-time.txt` | **có** | nhãn, trial, thời lượng khai báo |
| `wifi.txt`, `ping.log` | **có** | SSID, kênh, −48 dBm, ping 10 Hz |
| CSI đã đọc + QC | **có** | `parse_intel_csi.py` + `csi_preprocessing_pipeline.py` |
| `screen-clock.csv` | **thiếu ở cả 10 phiên** | không có file nào trong repo |
| Video gốc | **thiếu ở cả 10 phiên** | không có `.mp4/.mov/.mkv/.avi` nào |
| Ghi chú phiên quay | **thiếu** | chỉ có 4 cột metadata điền hồi tố |
| Phiếu thông tin buổi đo | **có mẫu** | [capture_checklist.md](capture_checklist.md) |
| Hoạt động `standing`, `left_arm`, `right_arm`, `mixed` | **thiếu dữ liệu** | mới có `empty` (3 phiên) và `walking` (3 phiên); `seated` **không** thay được `standing` |
| Phiên mẫu `P01_left_arm_01` | **có video + clock + notes, thiếu CSI** | `csi-folder.txt` trỏ tới `/home/toan/csi-data/...` — chưa copy về; chưa đồng bộ được (mục 2.1) |

Hệ quả: bước 3, 5, 6 **chưa chạy được trên bất kỳ phiên nào**. Không viết code đoán trước định dạng video khi chưa có một mẫu thật.

---

## 2. Bước 3 — đồng bộ thời gian CSI ↔ video

Thời gian CSI **đã xác minh trên dữ liệu thật**: `recorder.log` ghi `ts` là **Unix epoch tính bằng nanosecond** (đối chiếu `ts` của bản ghi đầu với `start-time.txt` khớp đến từng giây), nhịp VHT trung vị **100,7 ms ≈ 9,93 frame/s**, không có timestamp lùi ở cả 10 phiên. Nhưng mỗi phiên còn **8–11 khoảng trống dài 700–810 ms** (≈ 7 frame VHT liên tiếp) — khi ghép cửa sổ phải xử lý các khoảng này, không được coi chuỗi là liên tục tuyệt đối.

Vì `ts` là giờ thật (epoch), về lý thuyết có thể ghép bằng giờ tuyệt đối **nếu** biết chính xác thời điểm bắt đầu của video. Thực tế thì không: xem mục 2.1.

Quy tắc đề xuất (cần Người 3 đồng ý):

1. `screen-clock.csv` ghi **ít nhất 5 mốc** trải đều video: `video_time_s,clock_iso`. Đọc giờ trên đồng hồ laptop **nhìn thấy trong hình**, không lấy mtime file, không lấy thời điểm nhấn Enter.
2. Fit tuyến tính `clock = a · video_time + b` trên các mốc. **Kiểm tra riêng đầu và cuối video**, không chỉ fit một lần rồi tin toàn bộ.
3. Báo cáo `residual_max_ms` và `drift_ms` (chênh lệch dự đoán – thực tế ở mốc cuối). **Ngưỡng chấp nhận: 100 ms (1 frame CSI)**. Vượt ngưỡng → phiên đó đánh dấu `sync_ok = no`, không đưa vào tập huấn luyện.
4. Mọi mốc, hệ số `a`, `b`, residual và các đoạn không đồng bộ được ghi vào `sync_report.csv` — đây là đầu ra bắt buộc của bước 3.
5. Đồng hồ máy quay và đồng hồ laptop có thể lệch pha; nếu `a ≠ 1` quá 0,1 % thì phải hiệu chỉnh drift theo thời gian, không dùng offset cố định.

### 2.1 Kết quả kiểm tra phiên mẫu `P01_left_arm_01_20261103-025544_ttVs`

| Kiểm tra | Kết quả |
|---|---|
| Video giải mã được | **có** — 1920×1080, HEVC, 30 fps, 1.771 frame, **59,03 s** |
| `screen-clock.csv` | **sạch**: 968 mốc, cách nhau 100,53 ms, **0 khoảng trống**, không lùi, `displayed − host` trong khoảng −1…0 ms |
| Đồng hồ trong khung hình | **KHÔNG CÓ** → không thể áp dụng cách đồng bộ ở mục 2 |
| Thời gian tạo file video | 12:42:52 UTC, trong khi phiên thu lúc 07:55 UTC → **lệch 4 giờ 47 phút**, không dùng được |
| `screen-clock` so với video | log dài **97,28 s**, video chỉ **59,07 s** → ~38 s log nằm ngoài video |

Kết luận: `synchronization_status=not_yet_verified` trong `session-notes.txt` là **đúng**. Muốn đồng bộ thì lần quay sau **phải để đồng hồ trong khung hình** (10 s đầu và 10 s cuối là đủ), kèm 1 lần vẫy tay/clap làm mốc. Xem [capture_checklist.md](capture_checklist.md) mục E.

Phương án dự phòng (chưa dùng, chỉ khi không thể quay lại): tương quan giữa năng lượng chuyển động trong video (chênh lệch frame) và `motion_score` của CSI để tìm độ lệch. Nhược điểm: động tác giơ tay lặp lại nhiều lần trong phiên nên có **nhiều đỉnh tương quan cách nhau 15–20 s**, dễ chọn sai. Chỉ dùng khi có mốc clap để loại nhập nhằng.

---

## 3. Bước 5 — tư thế từ video

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Bộ khớp | **17 điểm COCO**, thứ tự cố định: mũi, mắt L/R, tai L/R, vai L/R, khuỷu L/R, cổ tay L/R, hông L/R, gối L/R, cổ chân L/R | chuẩn phổ biến, mọi thư viện pose đều xuất được; đủ vai–khuỷu–cổ tay cho `left_arm`/`right_arm` |
| Hệ toạ độ | **pixel** `(x, y)` + lưu `frame_width`, `frame_height` để chuẩn hoá sau | không mất thông tin, đổi độ phân giải vẫn tái tính được |
| Khớp không thấy | `NaN` + `mask = 0`, **không nội suy** | nội suy tạo bộ xương không có thật |
| Ngưỡng tin cậy | `confidence < 0.3` → coi như không thấy | cần Người 3 xác nhận |
| `empty` | **không tạo bộ xương**, ghi `no_person = 1` | tránh model học "xương giả" |
| Kiểm tra chất lượng | đếm % frame có ≥ 10/17 khớp hợp lệ; đếm đoạn công cụ nhận sai (khớp nhảy > 0,5 chiều cao người trong 1 frame) | phát hiện tracking mất dấu |
| Trái/phải | dùng nhãn L/R của công cụ, **không tự suy từ vị trí ảnh** (người quay ngược chiều là sai ngay) | |

Nhãn `left_arm`/`right_arm` chỉ nói **đoạn quay đó** giơ tay nào; mô hình vẫn cần toạ độ khớp tại từng thời điểm — không dùng nhãn đoạn thay cho xương.

---

## 4. Bước 6 — ghép cửa sổ CSI với tư thế

| Tham số | Đề xuất | Ghi chú |
|---|---|---|
| Độ dài cửa sổ | **2,0 s = 20 frame** VHT (9,93 fps) | khớp với `--window-seconds 2.0` đang dùng ở pipeline |
| Bước trượt | **0,2 s = 2 frame** | 90 % chồng lấn; nếu quá nhiều mẫu thì tăng lên 0,5 s |
| Thời điểm nhãn | **tư thế tại frame cuối cửa sổ** | giống hệ thống chạy trực tiếp; nếu dùng tư thế giữa cửa sổ thì lúc chạy thật phải chờ thêm nửa cửa sổ |
| Cửa sổ không hợp lệ | loại nếu **> 20 % frame thiếu** hoặc cửa sổ cắt qua khoảng trống > 300 ms | mỗi phiên có 8–11 khoảng như vậy |
| Định dạng mẫu | `.npz`: `csi [20,224] float32`, `keypoints [17,2] float32`, `mask [17] uint8`, `confidence [17] float32`, `timestamp_ms`, `session_id`, `activity`, `sync_ok` | kèm `README` mô tả từng trường + shape |
| Chia tập | **theo phiên quay**, trước khi tạo cửa sổ chồng lấn | tránh rò rỉ: hai cửa sổ liền nhau chia khác tập là gần trùng dữ liệu |
| Chuẩn hoá | fit **chỉ trên tập train**, lưu `center/scale` như `normalization_params.csv` | đã có sẵn cơ chế ở pipeline |

Mỗi mẫu phải truy ngược được: `session_id` → thư mục phiên → `raw.csi` + video gốc.

---

## 5. Danh sách phiên tối thiểu để huấn luyện

| Hoạt động | Phiên tối thiểu | Hiện có |
|---|---:|---:|
| `empty` | 3 | **3** |
| `standing` | 3 | 0 |
| `left_arm` | 3 | 0 |
| `right_arm` | 3 | 0 |
| `walking` | 3 | **3** |
| `mixed` | 3 | 0 |
| **Tổng** | **18 phiên** (6 × 3) | 6 |

Mỗi phiên: một video + `raw.csi` cùng lúc, cùng bố trí thiết bị, mỗi hoạt động **lặp 3 lần** và **trộn thứ tự** giữa các hoạt động (để tránh trùng lặp vấn đề "thứ tự thu dồn theo lớp" đã gặp ở bộ 3 lớp).

---

## 6. Cần Người 3 trả lời trước khi chạy

1. Bộ khớp 17 COCO + thứ tự trên — dùng được chưa?
2. Toạ độ **pixel** hay đã chuẩn hoá `[0,1]`?
3. Ngưỡng `confidence` và quy tắc khớp thiếu: bao nhiêu khớp hợp lệ thì giữ mẫu?
4. Cửa sổ **2,0 s / bước 0,2 s / nhãn ở cuối cửa sổ** — có đúng kiểu đầu vào mô hình sẽ dùng?
5. `empty`: bỏ hẳn khỏi bài toán tư thế, hay giữ thành lớp "không có người" với vector xương toàn mask 0?
6. Định dạng `.npz` như mục 4 hay cần `h5`/`parquet`?

Trả lời xong 6 câu này thì bước 3/5/6 chỉ còn là chạy script, không phải sửa thiết kế.

---

## 7. Điều kiện hoàn thành và giới hạn

- Hoàn thành khi: chạy lại được toàn bộ từ `raw.csi` + video → bộ mẫu, sai số đồng bộ và mẫu lỗi ghi rõ, Người 3 đọc được đúng shape.
- 6 hoạt động hiện tại **chỉ để kiểm chứng quy trình**, không đủ kết luận mô hình dựng đúng tư thế người thật; cần thêm phiên độc lập (người khác, bố trí khác) để đánh giá.
- Chất lượng kết quả phụ thuộc đồng bộ thời gian, chất lượng nhãn, chất lượng CSI — cả ba đều phải báo cáo riêng, không gộp thành một con số accuracy.

### 2.2 Kết quả sau khi nhận đủ dữ liệu (2026-11-03)

Đã nhận đủ 6 thư mục CSI (`csi-data/labeled/`) và 6 video (`csi-data/video-sessions/`). Đã chạy `estimate_sync.py` trên cả 6 phiên: **không phiên nào đạt mức `usable`**.

- r tốt nhất 0,11–0,28; đỉnh phụ gần bằng đỉnh chính (tỉ lệ 1,09–1,33).
- Offset giữa các phiên mâu thuẫn: −38,1 s … +1,5 s, dù quay cùng buổi, cùng quy trình.
- Thử làm mượt 0,3 / 0,5 / 1,0 / 2,0 s: kết quả vẫn không ổn định.

Vì vậy `build_pose_windows.py` dùng **giả định offset = 0** và ghi `sync_ok = 0` cho mọi mẫu. Quy tắc ở mục 2 (đọc đồng hồ trong hình) **vẫn là bắt buộc** cho mọi lần thu sau; không có nó thì không có cách nào kiểm chứng.

Cách kiểm chứng nhanh khi đã có đồng hồ: chênh lệch `residual_max_ms` phải ≤ 100 ms **và** CSI phải biến động mạnh hơn khi tay đang giơ (hiện tại với giả định 0 thì kết quả ngược lại: 7,99 khi giơ tay so với 8,60 khi không).
