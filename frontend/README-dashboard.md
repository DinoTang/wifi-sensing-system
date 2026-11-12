# Dashboard tín hiệu CSI và bộ xương 2D

Dashboard web mới nằm trong `frontend/dashboard/` và chạy tại `/dashboard/` trên backend FastAPI. Mã giao diện NiceGUI trước đó nằm ở nhánh `feature/frontend`.

## Cài đặt và chạy

Chạy từ thư mục gốc của repo. Cài Python và thư viện theo [hướng dẫn backend](../backend/README.md). Dashboard dùng HTML, CSS và JavaScript, nên chỉ cần Node.js khi chạy kiểm thử frontend.

```powershell
python -m backend --source replay --session hardware/csi_extracted/labeled/walking_01_20261025-053626_PiI7
```

Mở <http://127.0.0.1:8000/dashboard/> và bấm **Bắt đầu phiên**. Bản ghi đi bộ cho phép thử biểu đồ và bộ xương. Để thử phòng trống, dùng bản ghi ngày 03/11.

```powershell
python -m backend --source replay --session signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE
```

Để chạy với máy thu, chọn thư mục nơi Intel recorder đang ghi `raw.csi` và `recorder.log`.

```powershell
python -m backend --source live --session "D:/duong-dan-den-phien-csi"
```

Địa chỉ backend mặc định là `127.0.0.1:8000`. Khi đổi `--host` hoặc `--port`, mở `/dashboard/` tại địa chỉ tương ứng. Dashboard kết nối `/ws` và các API điều khiển trên cùng địa chỉ đang mở. Nếu dùng reverse proxy, chuyển tiếp cả HTTP và WebSocket trên cùng origin.

Để thử giao diện độc lập, chạy `python -m http.server 8080 --directory frontend/dashboard` rồi mở <http://127.0.0.1:8080/?source=demo>. Có thể chọn có người, phòng trống, chờ dữ liệu hoặc lỗi mô hình. Dải **Giả lập** đánh dấu tín hiệu và tư thế mẫu.

## Hai màn hình

- **Tín hiệu CSI** hiển thị biên độ I/Q thô trung bình của 224 giá trị mà backend cung cấp. Trục ngang tính giây tương đối so với mẫu mới nhất. Chọn cửa sổ 5, 10 hoặc 20 giây. Rê chuột qua đường để xem biên độ và timestamp. Tần số lấy mẫu tính từ trung vị khoảng cách giữa các mẫu hợp lệ trong cửa sổ.
- **Hiện diện** hiển thị dự đoán **Có người** hoặc **Phòng trống**, xác suất có người nếu backend cung cấp, và bộ xương 2D nếu tọa độ hợp lệ. Màu xanh biểu thị phía trái của cơ thể và màu vàng biểu thị phía phải. Hai màn hình dùng chung phiên và nguồn dữ liệu.

Nhãn **Trực tiếp**, **Phát lại** và **Giả lập** mô tả nguồn. Nút bắt đầu và dừng gọi API phiên của backend và chỉ có trong chế độ kết nối backend. Khi lỗi nguồn, lỗi mô hình hoặc mất WebSocket, dashboard báo **Chưa có kết quả** và xóa khớp. Mất WebSocket kích hoạt kết nối lại. Khi backend báo thiếu mẫu, biểu đồ tạm chờ. Khi có mẫu mới, đường biểu đồ tách đoạn ở khoảng thiếu. Khung trạng thái ở cuối màn hình giải thích các tình huống đang kết nối, đang tích lũy, đang nhận, dữ liệu cũ, lỗi và dừng.

## Đồng bộ và ánh xạ

- WebSocket `/ws` dùng `schema_version: 1` với ba loại thông điệp `status`, `chart` và `pose`. Dashboard hiển thị một dự đoán khi có `chart` cùng `session_id` và `timestamp_ms`. Timestamp là mẫu CSI cuối cửa sổ, tính bằng mili giây trên đồng hồ bộ thu. Luồng bỏ sequence trùng hoặc thấp hơn và dữ liệu của phiên khác.
- Biểu đồ ghép các cửa sổ chồng lên nhau theo timestamp, giữ tối đa 20 giây và 300 điểm. Khoảng cách trên 300 ms hoặc đoạn thiếu do backend báo được vẽ thành khoảng trống. Bộ đệm chờ ghép giữ tối đa tám kết quả tư thế.
- Các kết quả hết hạn sau 2,5 giây tính từ lúc trình duyệt nhận được, phù hợp nhịp 0,2 giây/cửa sổ và timeout backend mặc định 2 giây. Khi phát lại rất chậm, dashboard có thể báo dữ liệu cũ trong thời gian chờ cửa sổ kế tiếp.
- Bộ xương dùng thứ tự ID BlazePose `0–32`. Ví dụ vai–khuỷu–cổ tay trái là `11–13–15` và phải là `12–14–16`. Tọa độ `x_norm` và `y_norm` có gốc trên trái, với x tăng sang phải và y tăng xuống dưới. Tọa độ pixel được chia theo kích thước tham chiếu khi thiếu tọa độ chuẩn hóa. Canvas giữ tỉ lệ khung tham chiếu và chỉ nối xương khi có cả hai khớp.
- Đồng hồ trình duyệt đo thời gian từ nhận thông điệp đến gọi lệnh vẽ. Nhịp cập nhật ở chân trang lấy từ cửa sổ đến gần đây. Cả hai chỉ số đo ở frontend, còn thời gian xử lý backend hiển thị riêng.

Chế độ đối chiếu video cần video gốc và ánh xạ thời gian đã được xác minh cho chính phiên CSI. Bản dashboard này sử dụng tín hiệu CSI và đầu ra mô hình.

## Kiểm tra

```powershell
node --test frontend/tests/dashboard-stream.test.mjs
```

Kiểm thử WebSocket đã ghép được **1.333 cửa sổ** từ bảy phiên phát lại, gồm phòng trống, đứng, tay trái, tay phải, đi bộ và mixed. Phiên phòng trống ngày 03/11 có 187 cửa sổ đều hiện phòng trống. Phiên phòng trống độc lập ngày 25/10 có 181 cửa sổ bị mô hình báo có người. Dashboard hiển thị đầu ra của checkpoint cho cả hai phiên.

Trên trình duyệt, đã kiểm tra hai màn hình ở 1280 × 820 và 390 × 844, bộ xương 33 khớp khi có người, 0 khớp khi phòng trống, xóa khớp khi dừng hoặc mất WebSocket, và kết nối lại với nguồn mới. Kiểm thử giả lập xác nhận trạng thái chờ và lỗi mô hình. Trong một đoạn phát lại đi bộ 1×, dashboard quan sát khoảng 5,2 cập nhật/giây và 1,1 ms từ lúc nhận đến lúc vẽ tại một cửa sổ. Kết quả này đo trên máy kiểm thử. Việc kiểm tra trọn đường truyền từ bộ thu Intel đang ghi trên phần cứng thật vẫn cần thực hiện tại hiện trường.
