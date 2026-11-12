# Backend CSI cho người thứ 4

Backend chạy mô hình CSIPoseNet đã bàn giao trên nhánh `Khanh`. Một phiên đọc CSI từ Intel recorder, dựng cửa sổ, suy luận trên CPU và gửi trạng thái, biên độ và tọa độ khớp qua WebSocket. Trang kiểm thử tích hợp hiển thị cả ba loại thông điệp.

## Cài đặt và phát lại

Chạy từ thư mục gốc repo với Python 3.10–3.12. Nếu dùng Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r backend/requirements.txt
.\.venv\Scripts\python -m backend --source replay --session hardware/csi_extracted/labeled/walking_01_20261025-053626_PiI7
```

Mở <http://127.0.0.1:8000/dashboard/> rồi bấm **Bắt đầu phiên**. Dùng bản ghi phòng trống ngày 03/11 tại `signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE` để kiểm tra hiển thị 0 khớp. Đây là phiên đã dùng khi huấn luyện. Phiên phòng trống độc lập ngày 25/10 dùng để kiểm tra khả năng tổng quát hóa. Mặc định tốc độ phát bằng tốc độ thu. `--speed 8` tăng tốc phát lại. `--autostart` tự bắt đầu khi server khởi động.

Trên Linux dùng `.venv/bin/python` thay cho `.\.venv\Scripts\python`. Backend mặc định chỉ lắng nghe loopback. Để thử từ máy frontend trên LAN, đặt `--host` bằng địa chỉ mạng phù hợp trong môi trường nhóm tin cậy. API điều khiển phiên chưa có xác thực. Trình duyệt có thể dùng trang cùng origin hoặc reverse proxy cùng origin để truy cập API và WebSocket. Backend frontend (ví dụ NiceGUI) có thể kết nối WebSocket từ phía máy chủ.

## Thu trực tiếp từ Intel recorder

1. Tạo một thư mục mới cho phiên trên máy thu. Trỏ backend vào thư mục đó bằng lệnh sau.

   ```sh
   .venv/bin/python -m backend --source live --session /path/to/new-capture --autostart
   ```

2. Chạy lệnh Intel recorder mà nhóm đang dùng trong tiến trình riêng. Cấu hình recorder ghi `raw.csi` và chuyển nhật ký của **cùng tiến trình** vào `recorder.log` trong thư mục trên. Kiểm tra cả hai tệp tăng kích thước khi đang thu. Nếu recorder dùng stdout dạng stdio có bộ đệm, có thể chạy dưới `stdbuf -oL -eL` trên Linux để log được ghi theo dòng. Nếu recorder chỉ flush tệp nhị phân khi thoát, cần bật flush ở chương trình thu để có luồng trực tiếp.
3. Kiểm tra `GET /api/status` hoặc mở trang kiểm thử. `--timeout 2` là thời gian chờ mẫu hợp lệ mặc định. Dừng phiên từ UI/API chỉ dừng backend đọc CSI. Dừng tiến trình recorder riêng theo quy trình thu của nhóm.

`backend/source.py` là bộ chuyển tiếp từ hai tệp đang được ghi vào bộ đệm backend. Adapter đợi đủ byte và đủ các dòng metadata, ghép theo thứ tự, rồi kiểm tra kích thước và timestamp nhúng trong header. Nếu một tệp bị thay thế, thu nhỏ hoặc lệch cặp, bắt đầu lại bằng thư mục phiên mới. Khi mở tệp ở chế độ live, các bản ghi có từ trước lúc mở được đọc để căn vị trí và đếm vào `backlog`. Chỉ bản ghi mới sau mốc đó đi vào mô hình. Dùng thư mục mới khi chuyển phiên để nhận ngay từ đầu.

Mã thu ESP32 serial hiện tại xuất định dạng khác. Checkpoint trong gói bàn giao dùng CSI Intel VHT20. Phiên thu ESP32 cần parser và mô hình tương thích riêng.

## Hợp đồng xử lý và mô hình

- Nền tích hợp là `Khanh@5769e40`. Dùng layout và `decode_points` của `signal-processing/src/parse_intel_csi.py`. Bản ghi hợp lệ có `VHT,20,VHT20`, `2×2×56`, payload 896 byte. Bản ghi NOHT xen kẽ được bỏ qua.
- Cửa sổ dùng 20 mẫu × 224 biên độ I/Q theo thứ tự gốc của parser, bước 2 mẫu, nhịp danh định khoảng 101 ms. Biên độ làm tròn 4 chữ số như CSV và qua float16 → float32 như NPZ huấn luyện. Giữ các vị trí zero và không chạy pipeline lọc/đặc trưng dành cho bài toán phân loại hoạt động.
- Dùng `machine-learning/src/predict.py`, checkpoint `best_model.pt`, Z-score từ `normalization_params.json` và threshold presence 0,5 của mô hình 1.0.0. Tọa độ 2D theo 33 điểm BlazePose, gốc trên trái, có cả giá trị chuẩn hóa và pixel theo ảnh tham chiếu 1920 × 1080. `presence_probability` là sigmoid của đầu ra có người, không phải độ tin cậy từng khớp.
- Đầu vào là CSI. Timestamp mục tiêu là timestamp của mẫu CSI cuối cửa sổ, giữ nguyên cơ sở thời gian của recorder và đổi từ ns sang ms bằng phép chia nguyên. Mẫu lùi/trùng timestamp, gap trên 300 ms hoặc cấu hình sai làm mất hiệu lực tư thế cũ. Cửa sổ còn phải có trung vị khoảng mẫu 80–120 ms và thời gian từ mẫu đầu đến mẫu cuối 1.500–2.300 ms để chặn nhịp sai hoặc thiếu mẫu kéo dài. Khi gặp gap, hệ thống tích lũy lại 20 mẫu nên vị trí stride sau gap có thể khác batch offline.
- Replay đọc theo thứ tự và lập lịch theo timestamp của bản ghi đầu. Khi lịch phát gặp khoảng trống quá 300 ms theo thời gian bản ghi, backend xóa kết quả đã lưu trước khi chờ đến gói tiếp theo. Live dùng một cửa sổ đang suy luận và tối đa một cửa sổ chờ. Cửa sổ mới nhất thay thế cửa sổ chờ khi quá tải. Mẫu quá cũ hoặc cửa sổ đã bị vô hiệu hóa sẽ bị loại. Mỗi client chỉ giữ tối đa một thông điệp chờ cho mỗi loại.
- `main@e94e396` bàn giao một mô hình nghiên cứu riêng với 17 điểm và tọa độ `training_mean` cố định. Phiên bản này dùng checkpoint 33 điểm đi kèm nhánh được yêu cầu. Tùy chọn `--checkpoint` và `--normalizer` chỉ dành cho cặp tệp tương thích, đến từ nguồn được nhóm tin cậy.

## HTTP và WebSocket

- `GET /api/status` trả trạng thái, lý do, số bản ghi đọc/nhận/lọc/bỏ, số gap, số cửa sổ suy luận, cấu hình và timestamp nhận gần nhất.
- `GET /api/session/report` trả báo cáo JSON của phiên đang chạy hoặc phiên gần nhất. Báo cáo có cấu hình đầu vào, SHA-256 của checkpoint, bộ chuẩn hóa và parser, thống kê cửa sổ, chuyển trạng thái, nhịp suy luận và độ trễ từng giai đoạn. API trả 404 trước phiên đầu tiên. Xuất JSON trước khi bắt đầu phiên mới.
- `POST /api/session/start` trả HTTP 202 và ID phiên. Khi đã có phiên đang chạy, trả 409. Server chọn thư mục và nguồn từ tham số CLI.
- `POST /api/session/stop` dừng phiên và trả trạng thái hiện tại. Gọi lại sau khi phiên đã kết thúc sẽ trả trạng thái kết thúc đã ghi nhận. Có thể gửi `?session_id=<id>` để chỉ dừng đúng phiên đó. API trả 409 nếu phiên đã đổi.
- `WS /ws` gửi snapshot hiện tại lúc kết nối, rồi cập nhật theo thời gian thực. `sequence` tăng trong vòng đời của tiến trình server. Client có thể thấy khoảng nhảy khi bỏ thông điệp chờ. Khi đổi `session_id`, nhận trạng thái khác `ready` hoặc mất WebSocket, xóa ngay tư thế cũ.
- Tài liệu HTTP tự sinh tại <http://127.0.0.1:8000/docs>.

Mỗi thông điệp có `type`, `schema_version`, `session_id`, `sequence`, `source` (`replay` hoặc `live`), `timestamp_ms`, `server_sent_ms` và `model_version`. Timestamp của `pose` và `chart` cùng cửa sổ giống nhau. `server_sent_ms` là đồng hồ hệ điều hành trên máy backend lúc gửi. Khi nhận `status`, `person_present: null` nghĩa là chưa có kết quả hợp lệ. Trạng thái `ready` trên `pose` với `person_present: false` là dự đoán phòng trống và có `keypoints: []`.

Ví dụ một thông điệp `pose` cho phòng trống:

```json
{
  "type": "pose", "schema_version": 1, "session_id": "demo", "sequence": 20,
  "source": "replay", "timestamp_ms": 1792920264420, "server_sent_ms": 1794463200123,
  "model_version": "1.0.0", "status": "ready", "person_present": false,
  "presence_probability": 0.02, "confidence_method": "model_presence_sigmoid_probability",
  "coordinate_system": {"type": "2D_pixel", "reference_width": 1920, "reference_height": 1080, "origin": "top_left"},
  "keypoints": [], "timing_ms": {"preprocess": 0.4, "queue": 0.1, "inference": 4.5, "backend": 5.0}
}
```

`chart` có `quantity: "mean_raw_csi_amplitude"` và tối đa 20 `points` dạng `{"timestamp_ms": 1792920264420, "value": 125.4}`. Đây là biên độ trung bình của 224 vị trí ở mỗi mẫu, trước Z-score. Khi có người, mỗi khớp có `id`, `name`, `x_pixel`, `y_pixel`, `x_norm` và `y_norm`. [Dashboard hai màn hình](../frontend/README-dashboard.md) đọc hợp đồng này tại `/dashboard/`.

### Trạng thái và lỗi

- `idle`, `starting`, `warming_up`: chưa chạy, đang tải mô hình, đang đợi CSI hoặc cửa sổ.
- `ready`: đang xuất kết quả hợp lệ, gồm cả phòng trống và có người.
- `insufficient_data`: `CSI_GAP`, `CSI_TIMESTAMP_ORDER`, `CSI_CADENCE` hoặc `RESULT_TIMEOUT`.
- `incompatible`: `CSI_CONFIGURATION`, cấu hình không khớp đầu vào checkpoint.
- `disconnected`: `WAITING_FOR_CAPTURE_FILES` hoặc `CSI_TIMEOUT`. Phiên live có thể tiếp tục khi nguồn hoạt động lại.
- `source_error`: `INVALID_CAPTURE` (lệch cặp, tệp bị thay thế hoặc dữ liệu hỏng), `CAPTURE_UNAVAILABLE` (tệp không đọc được) hoặc `SESSION_FAILED` (phiên gặp lỗi ngoài dự kiến). Dùng log backend để tìm nguyên nhân rồi chạy lại.
- `model_error`: `MODEL_LOAD_FAILED` hoặc `MODEL_INFERENCE_FAILED`.
- `ended` / `stopped`: `REPLAY_COMPLETE` / `REQUESTED_STOP`.

## Kiểm thử

```sh
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/test_backend.py
```

`timing_ms` đo trên backend bằng đồng hồ đơn điệu. `preprocess` gồm ghép bản ghi mục tiêu và dựng cửa sổ, `queue` là thời gian chờ suy luận, `inference` gồm chuẩn hóa và chạy mô hình, và `backend` cộng ba giai đoạn. `last_ws_send_ms` trong API trạng thái đo thời gian lời gọi gửi ASGI gần nhất. Để đo độ trễ từ thu đến hiển thị trên hai máy, phải xác minh đồng hồ recorder và máy nhận trước khi lấy hiệu timestamp.

Xem [quy trình nghiệm thu và xuất bằng chứng JSON](NGHIEM-THU.md) để chạy các tình huống phòng trống, có người và mất nguồn.

### Kết quả trên máy thử

Windows, Intel Core i5-11300H, Python 3.12.14 và PyTorch 2.6.0 CPU, một luồng PyTorch. Hai phiên Intel bên dưới được ghi ngày 25/10 và nằm ngoài sáu phiên dùng trong báo cáo huấn luyện của nhánh `Khanh`.

- **Đi bộ, replay 1×.** 1.080 bản ghi, 558 VHT phù hợp, 10 gap. Client nhận 187 kết quả qua WebSocket, trung bình 3,21 Hz tính từ kết quả đầu đến kết quả cuối, gồm các khoảng phải tích lũy lại sau gap. Backend p50/p95 là 5,11/8,00 ms, suy luận p50/p95 là 4,56/7,40 ms. Từ thời điểm gửi đến nhận trên loopback cùng máy, p95 là 1 ms theo timestamp ms. Giao diện đã vẽ biểu đồ và 33 khớp. Lệnh dừng xóa cả hai khỏi màn hình.
- **Phòng trống, replay 8×.** 1.100 bản ghi, 572 VHT phù hợp, 10 gap, 181 kết quả. Backend p50/p95 là 6,94/9,22 ms ở tốc độ thử tải này. **Checkpoint dự đoán có người ở 181/181 cửa sổ**, dù phiên được gắn nhãn phòng trống. So sánh cửa sổ phát lại với parser và suy luận offline xác nhận backend chuyển đúng đầu vào. Trường hợp engine trả `person_present: false` được kiểm thử riêng và không phát khớp. Backend giữ nguyên đầu ra của checkpoint thay vì dùng nhãn phiên làm đầu vào.
- **Khôi phục và lỗi.** Kiểm thử tự động gồm bản ghi thiếu byte, log thiếu dòng, lệch cặp, sai cấu hình, gap, timeout khi dừng ghi, ghi trở lại sau timeout, lỗi tải/chạy mô hình, kết nối lại và dừng. Test live ghi thêm các byte từ bản ghi Intel thật vào hai tệp trong khi backend đang đọc.
- **Còn cần xác nhận tại hiện trường.** Chạy đúng tiến trình Intel recorder trên máy thu để xác minh flush, tốc độ và độ trễ từ lúc thu tới lúc frontend hiển thị. Kiểm tra lại mô hình trên nhiều phiên phòng trống độc lập. Tập huấn luyện của checkpoint `Khanh` có `sync_ok=0` theo báo cáo bàn giao, nên phần kiểm thử này xác minh vận chuyển dữ liệu và suy luận. Đánh giá độ chính xác tư thế cần dữ liệu đối chiếu đã đồng bộ.

Trích log thử nghiệm với phiên đi bộ:

```text
source=replay status=warming_up reason=WAITING_FOR_CSI
source=replay status=ready reason=None
source=replay status=insufficient_data reason=CSI_GAP
source=replay status=stopped reason=REQUESTED_STOP
```
