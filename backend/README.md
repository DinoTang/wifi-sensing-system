# Backend CSI cho người thứ 4

Backend dùng mô hình `smn-csiposenet-1.5` có sẵn trong nhánh `Jayson`. Một phiên đọc CSI từ Intel recorder, dựng cửa sổ, suy luận trên CPU và gửi trạng thái, biên độ và tọa độ khớp qua WebSocket.

## Cài đặt và phát lại

Chạy từ thư mục gốc repo với Python 3.10–3.12. Nếu dùng Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r backend/requirements.txt
$round = 'signal-processing/SMN-20261105T143709Z-1-001/SMN/csi-data-lan3'
.\.venv\Scripts\python -m backend --source replay --session "$round/video-sessions/P01_trial03_20261105-112653_fKOx/walking/csi"
```

Mở <http://127.0.0.1:8000/dashboard/> rồi bấm **Bắt đầu phiên**. Bản ghi đi bộ Lần 3 có đủ hai tệp và dùng cùng bố trí máy thu với mô hình. Để phát bản ghi đứng yên Lần 3, dùng `$round/labeled/test_2026110511304425340_20261105-113047_skHz`. Để thử phòng trống, dùng `signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE`. Đây là các ca phát lại kiểm tra hệ thống. Mặc định tốc độ phát bằng tốc độ thu. `--speed 8` tăng tốc phát lại. `--autostart` tự bắt đầu khi server khởi động.

Trên Linux dùng `.venv/bin/python` thay cho `.\.venv\Scripts\python`. Backend mặc định chỉ lắng nghe loopback. Để thử từ máy frontend trên LAN, đặt `--host` bằng địa chỉ mạng phù hợp trong môi trường nhóm tin cậy. API điều khiển phiên chưa có xác thực. Trình duyệt có thể dùng trang cùng origin hoặc reverse proxy cùng origin để truy cập API và WebSocket. Backend frontend (ví dụ NiceGUI) có thể kết nối WebSocket từ phía máy chủ.

## Thu trực tiếp bằng nút Bắt đầu phiên

Trên Ubuntu, kết nối card Intel đã cài IAX với Wi-Fi 5 GHz ở độ rộng 20 MHz. Lệnh dưới đây dùng cấu hình của script thu gốc: card `wlp0s20f3`, kênh 36, router `192.168.0.1` và bộ thu ở `~/iax/csi/iaxcsi/cpp/iaxcsi`.

```bash
iw dev wlp0s20f3 info | grep -F 'width:'
sudo -v && .venv/bin/python -m backend \
  --source live --auto-capture --session "$HOME/csi-data/live"
```

Giữ terminal này mở. Vào <http://127.0.0.1:8000/dashboard/>, chọn **30 giây**, **1 phút** hoặc **2 phút**, rồi bấm **Bắt đầu phiên**. Backend chuẩn bị Wi-Fi và bật IAX, sau đó bắt đầu tính thời lượng khi khởi chạy bộ thu. Backend ping router mỗi 100 ms với tải 56 byte trong lúc thu. **Dừng phiên** kết thúc sớm. `--duration` đặt thời lượng mặc định (60 giây nếu bỏ qua). Dùng `--iax-dir`, `--interface`, `--router`, `--channel` và `--ssid` khi máy hoặc mạng dùng cấu hình khác.

Một lần bấm **Bắt đầu phiên** tự thu CSI mới và chạy bộ phân loại hiện diện. Không có bước hiệu chuẩn phòng trống bắt buộc. Backend tích lũy khoảng 2 giây CSI hợp lệ trước kết quả đầu tiên; trạng thái đợi dữ liệu được hiển thị trong thời gian đó. Khi mô hình nhận là phòng trống, dashboard báo **Phòng trống** và xóa bộ xương.

Mỗi lần bấm bắt đầu tạo thư mục `~/csi-data/live/<session_id>`. Lấy ID đầy đủ ở mục Phiên trên dashboard hoặc qua `GET /api/status`. Dùng `setup.log` để kiểm tra Wi-Fi và kích hoạt, `recorder.log` để kiểm tra bộ thu, và `ping.log` để xem lưu lượng tới router. `raw.csi` tăng kích thước trong lúc thu nếu driver gửi khung CSI. Khi phiên dừng hoặc hết giờ, backend tự lưu `session-report.json` cạnh hai tệp thu; tệp có mã phiên, phiên bản mô hình, SHA-256 và thống kê dự đoán. Khi báo `CAPTURE_SUDO`, dừng backend bằng Ctrl+C rồi chạy lại lệnh trên trong terminal Ubuntu để gia hạn quyền sudo.

### Đọc từ bộ thu do nhóm chạy riêng

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
- Mặc định `machine-learning/src/predict.py` nạp cặp `smn_pose.pt` và `smn_normalization.json`. Tọa độ 2D theo 33 điểm BlazePose, gốc trên trái, có cả giá trị chuẩn hóa và pixel theo ảnh tham chiếu 1920 × 1080. Bản 1.5 dùng bộ phân loại hiện diện Extra Trees 128 cây, độ sâu tối đa 7, ngưỡng điểm 0,5. Đặc trưng loại 16 vị trí điền cố định của recorder, chuẩn hóa RMS riêng từng gói và từng liên kết anten, rồi mô tả biến thiên tương đối và hình dạng phổ. Backend chạy phần suy luận này bằng NumPy từ tham số trong checkpoint; không cần cài scikit-learn để chạy live.
- Mô hình tư thế ưu tiên bốn gói mới nhất trong cửa sổ. Lúc chuẩn hóa đầu vào, các sóng mang cố định trong dữ liệu huấn luyện được đưa về giá trị chuẩn hóa 0. Bộ kiểm tra độ lệch tín hiệu dựa vào các sóng mang còn lại, do giá trị điền của recorder ở vị trí cố định có thể khác lúc huấn luyện.
- Đầu vào là CSI. Timestamp mục tiêu là timestamp của mẫu CSI cuối cửa sổ, giữ nguyên cơ sở thời gian của recorder và đổi từ ns sang ms bằng phép chia nguyên. Mẫu lùi/trùng timestamp, gap trên 300 ms hoặc cấu hình sai làm mất hiệu lực tư thế cũ. Cửa sổ còn phải có trung vị khoảng mẫu 80–120 ms và thời gian từ mẫu đầu đến mẫu cuối 1.500–2.300 ms để chặn nhịp sai hoặc thiếu mẫu kéo dài. Khi gặp gap, hệ thống tích lũy lại 20 mẫu nên vị trí stride sau gap có thể khác batch offline.
- Replay đọc theo thứ tự và lập lịch theo timestamp của bản ghi đầu. Khi lịch phát gặp khoảng trống quá 300 ms theo thời gian bản ghi, backend xóa kết quả đã lưu trước khi chờ đến gói tiếp theo. Live dùng một cửa sổ đang suy luận và tối đa một cửa sổ chờ. Cửa sổ mới nhất thay thế cửa sổ chờ khi quá tải. Mẫu quá cũ hoặc cửa sổ đã bị vô hiệu hóa sẽ bị loại. Mỗi client chỉ giữ tối đa một thông điệp chờ cho mỗi loại.
- Trọng số tư thế học từ 2.237 cửa sổ của Lần 2 và Lần 3, gồm phòng trống, đứng yên, giơ tay trái, giơ tay phải, đi lại và phối hợp động tác. Đánh giá bằng video đồng bộ của cả hai đợt và cách chạy lại nằm trong [hướng dẫn nghiệm thu](NGHIEM-THU.md#kiểm-tra-giơ-tay-và-chuyển-động-hỗn-hợp).
- Khi CSI cách xa dải dùng để học tư thế, backend gửi kết quả hiện diện, mảng khớp rỗng và lý do `CSI_OUTSIDE_POSE_CALIBRATION`. Trường `pose_calibration` báo độ lệch tính trên CSI đã chuẩn hóa (`distance`) và giới hạn của checkpoint (`limit`). Khi thử phiên thu mới, giữ vị trí router và máy thu gần với bố trí đã dùng để huấn luyện và kiểm tra riêng trường hợp phòng trống, đứng yên và đi lại.
- `motion` trong thông điệp `pose` dùng trung bình độ lệch chuẩn theo thời gian của từng vị trí CSI trong cửa sổ 20 mẫu, trước Z-score. `level` là `high` từ ngưỡng khảo sát 10,15 trở lên và `low` ở dưới ngưỡng. Chỉ số mô tả biến thiên của kênh vô tuyến. Nhiễu, thay đổi môi trường và chuyển động đều có thể ảnh hưởng đến chỉ số. Xem [báo cáo tín hiệu](../signal-processing/docs/report_nguoi2.md).

## HTTP và WebSocket

- `GET /api/status` trả trạng thái, lý do, số bản ghi đọc/nhận/lọc/bỏ, số gap, số cửa sổ suy luận, cấu hình và timestamp nhận gần nhất.
- `GET /api/session/report` trả báo cáo JSON của phiên đang chạy hoặc phiên gần nhất. Báo cáo có cấu hình đầu vào, SHA-256 của checkpoint, bộ chuẩn hóa và parser, thống kê cửa sổ, chuyển trạng thái, nhịp suy luận và độ trễ từng giai đoạn. API trả 404 trước phiên đầu tiên. Nếu tải qua API, tải trước khi bắt đầu phiên mới. Thu tự động còn tự lưu báo cáo vào thư mục của đúng phiên khi kết thúc.
- `POST /api/session/start` trả HTTP 202 và ID phiên. Thu tự động nhận `?duration_s=30`, `60` hoặc `120`. Giá trị ngoài các lựa chọn hoặc truyền thời lượng cho phiên phát lại bị từ chối với 422. Khi đã có phiên đang chạy, API trả 409. `capture.started_ms` là mốc bắt đầu của bộ thu để dashboard tính thời gian còn lại.
- `POST /api/session/stop` dừng phiên và trả trạng thái hiện tại. Gọi lại sau khi phiên đã kết thúc sẽ trả trạng thái kết thúc đã ghi nhận. Có thể gửi `?session_id=<id>` để chỉ dừng đúng phiên đó. API trả 409 nếu phiên đã đổi.
- `WS /ws` gửi snapshot hiện tại lúc kết nối, rồi cập nhật theo thời gian thực. `sequence` tăng trong vòng đời của tiến trình server. Client có thể thấy khoảng nhảy khi bỏ thông điệp chờ. Khi đổi `session_id`, nhận trạng thái khác `ready` hoặc mất WebSocket, xóa ngay tư thế cũ.
- Dashboard mở `/ws?include_signal=true` để nhận thêm `signal` trong phiên live. Thông điệp có cùng `quantity` và `points` như `chart`, phát từ các gói VHT20 hợp lệ kể cả lúc tích lũy mẫu hoặc suy luận gặp lỗi. `chart` vẫn đi cùng `pose` theo đúng timestamp của cửa sổ suy luận. Khi hết thời lượng thu tự động, backend dừng bộ thu rồi chờ tối đa theo `--timeout` để xử lý cửa sổ cuối còn chờ.
- Tài liệu HTTP tự sinh tại <http://127.0.0.1:8000/docs>.

Mỗi thông điệp có `type`, `schema_version`, `session_id`, `sequence`, `source` (`replay` hoặc `live`), `timestamp_ms`, `server_sent_ms` và `model_version`. Timestamp của `pose` và `chart` cùng cửa sổ giống nhau. `server_sent_ms` là đồng hồ hệ điều hành trên máy backend lúc gửi. Backend đợi ba cửa sổ liên tiếp có tín hiệu hiện diện rồi mới báo có người. Trong lúc xác nhận, `status` có `person_present: null`, còn `pose` có `presence_pending: true`, `person_present: false` và `keypoints: []`. Một dự đoán âm tính trở về phòng trống và xóa bộ xương ngay.

Ví dụ một thông điệp `pose` cho phòng trống:

```json
{
  "type": "pose", "schema_version": 1, "session_id": "demo", "sequence": 20,
  "source": "replay", "timestamp_ms": 1792920264420, "server_sent_ms": 1794463200123,
  "model_version": "smn-csiposenet-1.5", "status": "ready", "person_present": false,
  "presence_probability": 0.02, "confidence_method": "csi_presence_forest_probability",
  "coordinate_system": {"type": "2D_pixel", "reference_width": 1920, "reference_height": 1080, "origin": "top_left"},
  "keypoints": [], "motion": {"score": 3.46, "level": "low"},
  "timing_ms": {"preprocess": 0.4, "queue": 0.1, "inference": 4.5, "backend": 5.0}
}
```

`chart` có `quantity: "mean_raw_csi_amplitude"` và tối đa 20 `points` dạng `{"timestamp_ms": 1792920264420, "value": 125.4}`. Đây là biên độ trung bình của 224 vị trí ở mỗi mẫu, trước chuẩn hóa. Khi mô hình có kết quả tư thế, mỗi khớp có `id`, `name`, `x_pixel`, `y_pixel`, `x_norm` và `y_norm`. [Dashboard hai màn hình](../frontend/README-dashboard.md) đọc hợp đồng này tại `/dashboard/`.

`presence_probability` là điểm do bộ phân loại xuất ra. Phần trăm trên dashboard mang nhãn **Điểm hiện diện của mô hình**; điểm cao chưa chứng minh mô hình đúng trên một phòng hoặc phiên thu chưa kiểm tra. Xem cách chia tập và giới hạn trong [nghiệm thu SMN 1.5](NGHIEM-THU.md#nghiệm-thu-smn-15-với-luồng-một-nút).

### Trạng thái và lỗi

- `idle`, `starting`, `warming_up`: chưa chạy, đang tải mô hình, đang đợi CSI hoặc cửa sổ.
- `ready`: đang xuất kết quả hợp lệ, gồm cả phòng trống và có người.
- `insufficient_data`: `CSI_GAP`, `CSI_TIMESTAMP_ORDER`, `CSI_CADENCE` hoặc `RESULT_TIMEOUT`.
- `incompatible`: `CSI_CONFIGURATION`, cấu hình không khớp đầu vào checkpoint.
- `disconnected`: `WAITING_FOR_CAPTURE_FILES` hoặc `CSI_TIMEOUT`. Phiên live có thể tiếp tục khi nguồn hoạt động lại.
- `source_error`: `INVALID_CAPTURE` (lệch cặp, tệp bị thay thế hoặc dữ liệu hỏng), `CAPTURE_UNAVAILABLE` (tệp không đọc được) hoặc `SESSION_FAILED` (phiên gặp lỗi ngoài dự kiến). Dùng log backend để tìm nguyên nhân rồi chạy lại.
- Thu tự động báo lỗi riêng khi Wi-Fi sai kênh hoặc độ rộng, router mất kết nối, driver chưa bật CSI, thiếu quyền sudo, recorder thoát sớm hoặc không ghi được CSI. Dashboard hiển thị hướng kiểm tra tương ứng.
- `model_error`: `MODEL_LOAD_FAILED` hoặc `MODEL_INFERENCE_FAILED`.
- `ended`: `REPLAY_COMPLETE` hoặc `CAPTURE_COMPLETE` khi đã hết thời lượng thu tự động.
- `stopped`: `REQUESTED_STOP`.

## Kiểm thử

```sh
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/test_backend.py
```

`timing_ms` đo trên backend bằng đồng hồ đơn điệu. `preprocess` gồm ghép bản ghi mục tiêu và dựng cửa sổ, `queue` là thời gian chờ suy luận, `inference` gồm chuẩn hóa và chạy mô hình, và `backend` cộng ba giai đoạn. `last_ws_send_ms` trong API trạng thái đo thời gian lời gọi gửi ASGI gần nhất. Để đo độ trễ từ thu đến hiển thị trên hai máy, phải xác minh đồng hồ recorder và máy nhận trước khi lấy hiệu timestamp.

Xem [quy trình nghiệm thu và xuất bằng chứng JSON](NGHIEM-THU.md) để chạy các tình huống phòng trống, có người và mất nguồn.

### Kết quả kiểm thử với checkpoint Khanh 1.0.0

Windows, Intel Core i5-11300H, Python 3.12.14 và PyTorch 2.6.0 CPU, một luồng PyTorch. Hai phiên Intel bên dưới được ghi ngày 25/10 và nằm ngoài sáu phiên dùng trong báo cáo huấn luyện của nhánh `Khanh`.

- **Đi bộ, replay 1×.** 1.080 bản ghi, 558 VHT phù hợp, 10 gap. Client nhận 187 kết quả qua WebSocket, trung bình 3,21 Hz tính từ kết quả đầu đến kết quả cuối, gồm các khoảng phải tích lũy lại sau gap. Backend p50/p95 là 5,11/8,00 ms, suy luận p50/p95 là 4,56/7,40 ms. Từ thời điểm gửi đến nhận trên loopback cùng máy, p95 là 1 ms theo timestamp ms. Giao diện đã vẽ biểu đồ và 33 khớp. Lệnh dừng xóa cả hai khỏi màn hình.
- **Phòng trống, replay 8×.** 1.100 bản ghi, 572 VHT phù hợp, 10 gap, 181 kết quả. Backend p50/p95 là 6,94/9,22 ms ở tốc độ thử tải này. **Checkpoint dự đoán có người ở 181/181 cửa sổ**, dù phiên được gắn nhãn phòng trống. So sánh cửa sổ phát lại với parser và suy luận offline xác nhận backend chuyển đúng đầu vào. Trường hợp engine trả `person_present: false` được kiểm thử riêng và không phát khớp. Backend giữ nguyên đầu ra của checkpoint thay vì dùng nhãn phiên làm đầu vào.
- **Khôi phục và lỗi.** Kiểm thử tự động gồm bản ghi thiếu byte, log thiếu dòng, lệch cặp, sai cấu hình, gap, timeout khi dừng ghi, ghi trở lại sau timeout, lỗi tải/chạy mô hình, kết nối lại và dừng. Test live ghi thêm các byte từ bản ghi Intel thật vào hai tệp trong khi backend đang đọc.
- **Thu tự động.** Kiểm thử với tiến trình thu thay thế đã đọc được dữ liệu từ lúc bắt đầu, dừng cả bộ thu lẫn ping, bắt đầu lại trong thư mục mới, kết thúc theo thời lượng và phân biệt lỗi sudo với mất nguồn CSI.
- **Biến thiên CSI.** Ở các phiên ngày 25/10 và 03/11, mức cao xuất hiện trong 708/728 cửa sổ đi lại. Mức thấp xuất hiện trong 506/563 cửa sổ ngồi yên và 714/714 cửa sổ phòng trống. Các số đếm này dựa trên nhãn của toàn phiên và dùng để kiểm tra cách hiển thị chỉ số.
- **Còn cần xác nhận tại hiện trường.** Chạy đúng tiến trình Intel recorder trên máy thu để xác minh flush, tốc độ và độ trễ từ lúc thu tới lúc frontend hiển thị. Kiểm tra lại mô hình trên nhiều phiên phòng trống độc lập. Tập huấn luyện của checkpoint `Khanh` có `sync_ok=0` theo báo cáo bàn giao, nên phần kiểm thử này xác minh vận chuyển dữ liệu và suy luận. Đánh giá độ chính xác tư thế cần dữ liệu đối chiếu đã đồng bộ.

Trích log thử nghiệm với phiên đi bộ:

```text
source=replay status=warming_up reason=WAITING_FOR_CSI
source=replay status=ready reason=None
source=replay status=insufficient_data reason=CSI_GAP
source=replay status=stopped reason=REQUESTED_STOP
```
