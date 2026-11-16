# Dashboard tín hiệu CSI và bộ xương 2D

Dashboard web mới nằm trong `frontend/dashboard/` và chạy tại `/dashboard/` trên backend FastAPI. Mã giao diện NiceGUI trước đó nằm ở nhánh `feature/frontend`.

## Cài đặt và chạy

Chạy từ thư mục gốc của repo. Cài Python và thư viện theo [hướng dẫn backend](../backend/README.md). Dashboard dùng HTML, CSS và JavaScript, nên chỉ cần Node.js khi chạy kiểm thử frontend.

```bash
smn="signal-processing/SMN-20261105T143709Z-1-001/SMN/csi-data-lan3"
round="$smn/video-sessions/P01_trial03_20261105-112653_fKOx"
.venv/bin/python -m backend --source replay --session "$round/walking/csi"
```

Mở <http://127.0.0.1:8000/dashboard/> và bấm **Bắt đầu phiên**. Bản ghi đi bộ Lần 3 chạy với mô hình có sẵn trên `Jayson`. Để thử phòng trống, dừng backend rồi mở lại với bản ghi ngày 03/11.

```bash
empty="signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE"
.venv/bin/python -m backend --source replay --session "$empty"
```

Trên Ubuntu có Intel IAX, bật chế độ thu tự động. Mở dashboard, chọn thời lượng **30 giây**, **1 phút** hoặc **2 phút**, rồi bấm **Bắt đầu phiên**. Bấm **Dừng phiên** để kết thúc sớm.

```bash
sudo -v && .venv/bin/python -m backend \
  --source live --auto-capture --session "$HOME/csi-data/live"
```

Xem [cấu hình và nhật ký thu trên Ubuntu](../backend/README.md#thu-trực-tiếp-bằng-nút-bắt-đầu-phiên).

Địa chỉ backend mặc định là `127.0.0.1:8000`. Khi đổi `--host` hoặc `--port`, mở `/dashboard/` tại địa chỉ tương ứng. Dashboard kết nối `/ws` và các API điều khiển trên cùng địa chỉ đang mở. Nếu dùng reverse proxy, chuyển tiếp cả HTTP và WebSocket trên cùng origin.

Để thử giao diện độc lập, chạy `python -m http.server 8080 --directory frontend/dashboard` rồi mở <http://127.0.0.1:8080/?source=demo>. Có thể chọn có người, phòng trống, chờ dữ liệu hoặc lỗi mô hình. Dải **Giả lập** đánh dấu tín hiệu và tư thế mẫu.

## Hai màn hình

- **Tín hiệu CSI** hiển thị biên độ I/Q thô trung bình của 224 giá trị mà backend cung cấp. Trục ngang tính giây tương đối so với mẫu mới nhất. Chọn cửa sổ hiển thị từ 5 giây đến 2 phút. Khi bắt đầu thu tự động, cửa sổ biểu đồ lấy theo thời lượng đã chọn. Rê chuột qua đường để xem biên độ và timestamp.
- **Hiện diện** hiển thị dự đoán **Có người** hoặc **Phòng trống**, mức biến thiên CSI và bộ xương 2D thử nghiệm nếu tọa độ hợp lệ. Mức biến thiên giúp quan sát sự khác biệt giữa đi lại và ngồi yên trong dữ liệu đã thu. Độ chính xác của từng khớp khi đổi tư thế còn hạn chế. Màu xanh biểu thị phía trái của cơ thể và màu vàng biểu thị phía phải. Hai màn hình dùng chung phiên và nguồn dữ liệu.

Nhãn **Trực tiếp**, **Phát lại** và **Giả lập** mô tả nguồn. Nút bắt đầu và dừng gọi API phiên của backend và chỉ có trong chế độ kết nối backend. Khi lỗi nguồn, lỗi mô hình hoặc mất WebSocket, dashboard báo **Chưa có kết quả** và xóa khớp. Mất WebSocket kích hoạt kết nối lại. Khi backend báo thiếu mẫu, biểu đồ tạm chờ. Khi có mẫu mới, đường biểu đồ tách đoạn ở khoảng thiếu. Khung trạng thái ở cuối màn hình giải thích các tình huống đang kết nối, đang tích lũy, đang nhận, dữ liệu cũ, lỗi và dừng.

## Đồng bộ và ánh xạ

- WebSocket `/ws` dùng `schema_version: 1` với ba loại thông điệp `status`, `chart` và `pose`. Dashboard hiển thị một dự đoán khi có `chart` cùng `session_id` và `timestamp_ms`. Timestamp là mẫu CSI cuối cửa sổ, tính bằng mili giây trên đồng hồ bộ thu. Luồng bỏ sequence trùng hoặc thấp hơn và dữ liệu của phiên khác.
- Biểu đồ giữ tối đa 120 giây và 2.000 điểm của phiên đang xem, kể cả sau khi kết thúc. Khoảng cách trên 300 ms hoặc đoạn thiếu do backend báo được vẽ thành khoảng trống. Bộ đệm chờ ghép giữ tối đa tám kết quả tư thế. Backend báo `CSI_OUTSIDE_POSE_CALIBRATION` khi tín hiệu nằm ngoài dải đã học cho tọa độ khớp.
- Các kết quả hết hạn sau 2,5 giây tính từ lúc trình duyệt nhận được, phù hợp nhịp 0,2 giây/cửa sổ và timeout backend mặc định 2 giây. Khi phát lại rất chậm, dashboard có thể báo dữ liệu cũ trong thời gian chờ cửa sổ kế tiếp.
- Bộ xương dùng thứ tự ID BlazePose `0–32`. Ví dụ vai–khuỷu–cổ tay trái là `11–13–15` và phải là `12–14–16`. Tọa độ `x_norm` và `y_norm` có gốc trên trái, với x tăng sang phải và y tăng xuống dưới. Tọa độ pixel được chia theo kích thước tham chiếu khi thiếu tọa độ chuẩn hóa. Canvas giữ tỉ lệ khung tham chiếu và chỉ nối xương khi có cả hai khớp.
- Đồng hồ trình duyệt đo thời gian từ nhận thông điệp đến gọi lệnh vẽ. Nhịp cập nhật ở chân trang lấy từ cửa sổ đến gần đây. Cả hai chỉ số đo ở frontend, còn thời gian xử lý backend hiển thị riêng.

Chế độ đối chiếu video cần video gốc và ánh xạ thời gian đã được xác minh cho chính phiên CSI. Bản dashboard này sử dụng tín hiệu CSI và đầu ra mô hình.

## Kiểm tra

```powershell
node --test frontend/tests/dashboard-stream.test.mjs
```

Kiểm thử mặc định phát lại các bản ghi đi bộ và đứng yên Lần 3 với mô hình SMN. Các ca giơ tay trái, tay phải, cả hai tay và phối hợp đi lại dùng thêm hai đợt CSI của nhóm. Cách chạy và kết quả đối chiếu với video nằm trong [hướng dẫn nghiệm thu](../backend/NGHIEM-THU.md#kiểm-tra-giơ-tay-và-chuyển-động-hỗn-hợp).

Trên trình duyệt, đã kiểm tra hai màn hình ở 1280 × 820 và 390 × 844, bộ xương 33 khớp khi có người, 0 khớp khi phòng trống, xóa khớp khi dừng hoặc mất WebSocket, và kết nối lại với nguồn mới. Kiểm thử giả lập xác nhận trạng thái chờ và lỗi mô hình. Trong một đoạn phát lại đi bộ 1×, dashboard quan sát khoảng 5,2 cập nhật/giây và 1,1 ms từ lúc nhận đến lúc vẽ tại một cửa sổ. Kết quả này đo trên máy kiểm thử. Việc kiểm tra trọn đường truyền từ bộ thu Intel đang ghi trên phần cứng thật vẫn cần thực hiện tại hiện trường.
