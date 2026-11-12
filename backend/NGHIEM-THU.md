# Nghiệm thu backend CSI

Backend nhận `raw.csi` và `recorder.log` từ cùng một tiến trình Intel recorder, ghép đủ một bản ghi, giữ các mẫu VHT20 phù hợp, dựng cửa sổ 20 × 224 và đưa vào CSIPoseNet. Dashboard nhận `status`, `chart` và `pose` qua WebSocket. Đối chiếu `session_id` và `timestamp_ms` của biểu đồ và tư thế để xác định chúng thuộc cùng cửa sổ.

## Phát lại và xuất bằng chứng

Chạy từ gốc repo sau khi cài môi trường Python theo [README backend](README.md).

1. Mở terminal thứ nhất và chạy backend với bản ghi đi bộ. Đây là phiên ngày 25/10, ngoài sáu phiên được liệt kê trong báo cáo mô hình.

   ```powershell
   python -m backend --source replay --session hardware/csi_extracted/labeled/walking_01_20261025-053626_PiI7
   ```

2. Mở [dashboard](http://127.0.0.1:8000/dashboard/), giữ phiên ở trạng thái chưa bắt đầu. Trong terminal thứ hai chạy lệnh dưới đây. Lệnh kết nối WebSocket trước, bắt đầu phiên qua HTTP, kiểm tra trình tự thông điệp, ghép biểu đồ và tư thế theo timestamp, rồi tải báo cáo khi backend báo `REPLAY_COMPLETE`.

   ```powershell
   python -m backend.verify --output replay-walking.json
   ```

3. Quan sát hai màn hình trong lúc lệnh chạy. Khi kết thúc, kiểm tra `websocket_client.matched_windows` lớn hơn 0 và `session.status` bằng `ended`. Đọc báo cáo ngay trên server bằng `GET /api/session/report` hoặc mở tệp JSON đã lưu.
4. Dừng backend và chạy lại với `signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE` để trình diễn phòng trống. Dùng một tên JSON mới. Phiên 03/11 này đã xuất hiện trong tập huấn luyện. Lặp lại với `hardware/csi_extracted/labeled/empty_01_20261025-052422_4eki` để kiểm tra riêng khả năng tổng quát hóa.

`backend.verify` yêu cầu backend chạy ở chế độ replay và chưa có phiên hoạt động. Nếu đổi cổng hoặc chạy chậm, dùng `--base-url http://127.0.0.1:8015` hoặc `--timeout 180`. Tốc độ mặc định là 1×. Ghi rõ `--speed` khi đưa số liệu vào báo cáo.

## Đọc báo cáo

- `model.artifact_sha256` định danh tệp checkpoint, bộ chuẩn hóa và parser. Có thể đối chiếu từng giá trị trên Windows bằng `Get-FileHash -Algorithm SHA256 <tệp>`.
- `counts.records`, `accepted`, `filtered` và `backlog` mô tả đầu vào. `window_candidates`, `rejected_cadence`, `dropped_windows`, `model_failures` và `predictions` mô tả quá trình dựng cửa sổ và suy luận. `present_predictions` và `empty_predictions` là số đầu ra mô hình theo hai trạng thái. `gaps` gồm khoảng cách vượt ngưỡng hoặc timestamp không tăng. `timestamp_order_errors` đếm riêng lỗi thứ tự.
- `prediction_rate_hz` lấy số khoảng giữa dự đoán chia thời gian đơn điệu từ dự đoán đầu đến cuối. Thời gian này bao gồm giai đoạn chờ dữ liệu giữa các dự đoán. `transitions` giữ tối đa 64 lần đổi trạng thái gần nhất và `transition_count` ghi tổng số lần đổi.
- `timing_ms` có số mẫu, p50, p95 và tối đa trên 1.024 mẫu gần nhất cho mỗi giai đoạn. `backend` tính ghép bản ghi mục tiêu, dựng cửa sổ, chờ và suy luận. `ws_send` đo lời gọi gửi ASGI theo từng client. Số đo này chưa bao gồm truyền mạng tới trình duyệt, vẽ giao diện, hoặc thời gian Intel recorder đợi ghi đủ cửa sổ.
- `websocket.discarded` đếm thông điệp bị thay thế trong hộp thư client hoặc bị loại khi trạng thái mất hiệu lực. `counts.dropped_windows` là cửa sổ bị bỏ ở hàng đợi suy luận. Hai loại số đếm có phạm vi khác nhau.

## Kiểm thử lỗi và thu trực tiếp

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/test_backend.py
```

Các ca tích hợp kiểm tra bản ghi nhị phân ghi dở, log chưa đủ dòng, lệch timestamp giữa hai tệp, mất mẫu, ghi trở lại sau timeout, lỗi mô hình, kết nối lại và dừng. Ca live trong bộ kiểm thử dùng byte Intel thật được ghi thêm vào thư mục tạm để thử bộ đọc tệp. Khi nguồn mất, client kết nối lại chỉ nhận trạng thái hiện tại. Cửa sổ hợp lệ mới được dựng sau khi nguồn phục hồi.

Để nghiệm thu với thiết bị, chạy Intel recorder và backend ở chế độ live theo [hướng dẫn thu trực tiếp](README.md#thu-trực-tiếp-từ-intel-recorder). Dùng thư mục phiên mới, xác nhận hai tệp tăng kích thước trong lúc thu, rồi quan sát `warming_up` → `ready`. Tạm ngừng nguồn đủ lâu để thấy `CSI_TIMEOUT` và kiểm tra dashboard xóa tư thế. Nối lại, chờ đủ một cửa sổ mới và xuất JSON trước khi bắt đầu phiên khác.

Ghi model máy thu, cấu hình Wi-Fi, hệ điều hành, phiên bản PyTorch và tốc độ thu vào biên bản. Nếu đo độ trễ từ thời điểm thu tới hiển thị, ghi cách xác minh đồng hồ và sai số đồng bộ. Dùng số đo từng giai đoạn riêng khi chưa xác minh đồng hồ.

Một phiên phát được bộ xương chứng minh dữ liệu đi qua parser, cửa sổ, mô hình và giao diện. Báo cáo độ chính xác hiện diện cần nhãn thật của các phiên độc lập. Báo cáo sai số khớp cần nhãn tư thế đã đồng bộ. Giữ riêng số liệu kiểm thử kỹ thuật, ví dụ trong tập huấn luyện và kết quả kiểm tra tổng quát hóa.

## Kết quả đo trên máy kiểm thử

Các số dưới đây đo trên Windows 11, Intel Core i5-11300H, Python 3.12.14 và PyTorch 2.6.0 CPU. Client HTTP và WebSocket chạy qua loopback tới một tiến trình backend riêng.

- **Đi bộ độc lập ngày 25/10, replay 1×.** Đọc 1.080 bản ghi, giữ 558 mẫu VHT20 và ghi nhận 10 khoảng mất mẫu. Client ghép đủ 187/187 cặp biểu đồ–tư thế. Nhịp trung bình là 3,215 dự đoán/giây, gồm các khoảng phải tích lũy lại. Xử lý backend p50/p95 là 5,991/8,615 ms, suy luận p95 7,860 ms và gửi ASGI p95 0,340 ms. Mức tối đa của thời gian xử lý backend là 88,181 ms.
- **Phòng trống độc lập ngày 25/10, replay 8×.** Client ghép 181/181 cửa sổ. Checkpoint báo có người ở cả 181 cửa sổ. Backend loại 11 cửa sổ do cadence không phù hợp. Kết quả này cho thấy checkpoint cần được đánh giá thêm trên các phiên phòng trống độc lập.
- **Phòng trống trong dữ liệu huấn luyện ngày 03/11, replay 8×.** Client ghép 187/187 cửa sổ. Cả 187 dự đoán là phòng trống và có danh sách khớp rỗng. Đây là ca kiểm tra giao diện và đường truyền.
- **Mô phỏng bộ thu ghi nối tệp.** Ghi nối 200 bản ghi từ phiên đi bộ, tách phần đầu của một bản ghi và dòng log để thử đọc dở. Backend phát 13 dự đoán trước khi ngừng ghi và 28 dự đoán tổng cộng sau khi ghi tiếp. Client mới trong lúc mất dữ liệu nhận `disconnected / CSI_TIMEOUT` và không nhận tư thế cũ. Chuỗi trạng thái cuối là `ready → disconnected → warming_up → insufficient_data → ready → stopped`. Ca này chạy từ byte CSI đã ghi. Nghiệm thu thu CSI trực tiếp và độ trễ từ thiết bị tới màn hình vẫn cần bộ thu Intel.
