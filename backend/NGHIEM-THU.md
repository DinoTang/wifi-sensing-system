# Nghiệm thu backend CSI

Backend nhận `raw.csi` và `recorder.log` từ cùng một tiến trình Intel recorder, ghép đủ một bản ghi, giữ các mẫu VHT20 phù hợp, dựng cửa sổ 20 × 224 và đưa vào CSIPoseNet. Dashboard nhận `status`, `chart` và `pose` qua WebSocket. Đối chiếu `session_id` và `timestamp_ms` của biểu đồ và tư thế để xác định chúng thuộc cùng cửa sổ.

## Phát lại và xuất bằng chứng

Chạy từ gốc repo sau khi cài môi trường Python theo [README backend](README.md).

1. Mở terminal thứ nhất và chạy backend với bản ghi đi bộ Lần 3. Bản ghi này nằm trong dữ liệu huấn luyện tư thế SMN và dùng để kiểm tra đường truyền đến dashboard.

   ```powershell
   $round = 'signal-processing/SMN-20261105T143709Z-1-001/SMN/csi-data-lan3'
   python -m backend --source replay --session "$round/video-sessions/P01_trial03_20261105-112653_fKOx/walking/csi"
   ```

2. Mở [dashboard](http://127.0.0.1:8000/dashboard/), giữ phiên ở trạng thái chưa bắt đầu. Trong terminal thứ hai chạy lệnh dưới đây. Lệnh kết nối WebSocket trước, bắt đầu phiên qua HTTP, kiểm tra trình tự thông điệp, ghép biểu đồ và tư thế theo timestamp, rồi tải báo cáo khi backend báo `REPLAY_COMPLETE`.

   ```powershell
   python -m backend.verify --output replay-walking.json
   ```

3. Quan sát hai màn hình trong lúc lệnh chạy. Khi kết thúc, kiểm tra `websocket_client.matched_windows` lớn hơn 0 và `session.status` bằng `ended`. Đọc báo cáo ngay trên server bằng `GET /api/session/report` hoặc mở tệp JSON đã lưu.
4. Dừng backend và chạy lại với `signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE` để trình diễn phòng trống. Dùng một tên JSON mới. Phiên 03/11 này đã xuất hiện trong tập huấn luyện. Lặp lại với `hardware/csi_extracted/labeled/empty_01_20261025-052422_4eki` để kiểm tra riêng khả năng tổng quát hóa.

`backend.verify` yêu cầu backend chạy ở chế độ replay và chưa có phiên hoạt động. Nếu đổi cổng hoặc chạy chậm, dùng `--base-url http://127.0.0.1:8015` hoặc `--timeout 180`. Tốc độ mặc định là 1×. Ghi rõ `--speed` khi đưa số liệu vào báo cáo.

## Kiểm tra giơ tay và chuyển động hỗn hợp

Trong phiên thu trực tiếp có người, để người thử tự chọn động tác, thứ tự và thời điểm đổi tư thế trong vùng thu. Trên màn hình **Hiện diện**, đối chiếu cổ tay so với vai cùng bên và vị trí hông với chuyển động thực tế. Trái và phải tính theo bên của người thực hiện. Chạy riêng một phiên phòng trống để kiểm tra báo nhầm.

Để phát lại, dùng `python -m backend --source replay --session <thu-muc-phien>` với cặp `raw.csi` và `recorder.log` từ cùng phiên trong bộ SMN. Chẳng hạn, ba thư mục trong `csi-data-lan3/labeled` dành cho giơ tay trái, tay phải và phối hợp động tác lần lượt là `test_2026110511321713388_20261105-113219_unjU`, `test_2026110511334930064_20261105-113352_WWrN` và `test_202611051136547003_20261105-113657_oRku`.

- Phát lại qua WebSocket đủ 12 bản ghi từ hai đợt đã dùng để học. Phòng trống có 0/381 cửa sổ vẽ khớp, đứng yên vẽ khớp ở 364/369 và hông thay đổi khoảng 14–17 px từ phân vị 5 đến 95. Khi đi bộ, tương quan vị trí hông theo chiều ngang với video đạt 0,98 ở cả hai lượt.
- Với phiên giơ từng tay, cổ tay trái đúng lúc giơ ở 184/184 và lúc hạ ở 170/171. Cổ tay phải đạt 195/195 và 155/168. Trong phiên hỗn hợp, 252/262 cửa sổ khớp cả hai trạng thái cổ tay và 85/85 cửa sổ giơ đồng thời được nhận đúng. Chỉ tính cửa sổ thấy rõ vai, cổ tay và cổ tay cách độ cao vai ít nhất 1,5% chiều cao ảnh.
- Khi huấn luyện trên một đợt và kiểm tra trên đợt kia, kết quả phối hợp hai tay là 86/137 và 83/125. Riêng ca giơ cùng lúc đạt 40/50 và 1/35. Hai đợt có cùng người và bố trí thiết bị, nên cần kiểm thử tiếp trên phiên thu Intel mới. Bộ video đồng bộ hiện thiếu phiên ngồi có nhãn.
- Để chấm sai số trước buổi thi, dùng một phiên cử động tự do độc lập ở bố trí sẽ thi. Video đồng bộ cung cấp tọa độ tham chiếu khi đánh giá. Backend thu trực tiếp suy luận từ các cửa sổ CSI. Lần 1 có video và mốc bắt đầu do người thu ghi nhận, với sai số đồng bộ từng khung hình chưa được xác minh.

Để tính lại điểm với NPZ đã đồng bộ, chạy từ gốc repo. Mỗi tệp chứa `amplitudes`, `activity`, `poses`, `mask`, `valid` và `frame_size`. Công cụ chấm các nhãn có trong tệp, gồm cả ngồi yên và chuyển động tự do nếu phiên có nhãn tương ứng. Gán `empty` cho cửa sổ tham chiếu phòng trống. Khi tệp có cờ `sync_ok=0`, cửa sổ tương ứng chỉ được đếm theo nhãn hoạt động, không dùng để chấm sai số khớp. Cửa sổ đồng bộ có người nhưng khớp bị chặn vẫn nằm trong mẫu số của PCK và các chỉ số giơ tay khi video thấy khớp.

```bash
.venv/bin/python machine-learning/src/evaluate_smn.py \
  "$HOME/smn-windows/Lan2_windows.npz" "$HOME/smn-windows/Lan3_windows.npz"
```

## Đối chiếu với nhánh Khanh trên ba lần thu

Đã phát lại 18 ca từ ba lần thu trong nhóm, gồm phòng trống, đứng yên, giơ từng tay, đi lại và phối hợp. Cả 3.389 cửa sổ suy luận ghép được `chart` và `pose` theo timestamp và mã phiên. Từng gói VHT20 hợp lệ khớp timestamp và 224 biên độ với kết quả parser ngoại tuyến của `Khanh@5769e40`.

- Checkpoint `1.0.0` của Khanh báo có người trong cả 381 cửa sổ phòng trống của Lần 2 và Lần 3. Checkpoint SMN 1.3 của Jayson báo phòng trống ở cả 568 cửa sổ gắn nhãn trống của ba lần. Trong các ca có người, SMN 1.3 xuất 2.813/2.821 bộ khớp. Tám lượt bỏ sót gồm hai cửa sổ đi lại Lần 1, một cửa sổ hỗn hợp Lần 2 và năm cửa sổ đứng yên Lần 3.
- Ở hai video đồng bộ Lần 2 và Lần 3, SMN 1.3 đạt PCK@0,05 đường chéo ảnh là 10.734/11.136 khớp đi lại. Checkpoint Khanh đạt 3.267/11.136 khớp trên cùng bộ nhãn. Hai video này đã dùng để huấn luyện tư thế SMN 1.3. Kết quả kiểm tra giữ riêng theo đợt được ghi ở mục giơ tay và chuyển động hỗn hợp.
- Sáu clip Lần 1 dùng để kiểm tra hoạt động. Mốc ghép từng khung hình còn chưa xác minh. Độ trải hông theo chiều ngang từ phân vị 5 đến 95 khi đứng yên là khoảng 4 px trên video và 93 px trên CSI 1.3. Khi đi lại, các giá trị tương ứng là khoảng 497 px và 91 px. Đây là thống kê theo ca. Cần phiên thu mới với video đồng bộ để chấm sai lệch từng khớp trong bố trí thi.
- Gói nghiên cứu `main@e94e396` chọn `training_mean` cho tọa độ 17 khớp. Chạy hàm tọa độ trên 185 cửa sổ đi lại của Lần 2 cho cùng một bộ tọa độ ở cả 185 cửa sổ. Bản chạy trực tiếp trên Jayson dùng CSIPoseNet với 33 khớp dự đoán từ CSI.

## Đối chiếu hai phiên thu Ubuntu

Hai phiên thu 60 giây trên kênh 36, độ rộng 20 MHz cho thấy SMN 1.2 báo có người ở 192/192 dự đoán phòng trống và chặn tọa độ ở cả 201 dự đoán đi lại. Giá trị điền ở các sóng mang cố định của recorder khác giá trị trong tập huấn luyện. Bản 1.3 chuẩn hóa các vị trí này theo trung bình huấn luyện và học lại bộ phân loại hiện diện từ dữ liệu thu mới.

- Phát lại qua WebSocket với SMN 1.3 cho kết quả 193/193 cửa sổ phòng trống và 201/201 cửa sổ đi lại có đủ 33 khớp. Vị trí hông theo chiều ngang thay đổi 122 px từ phân vị 5 đến 95 trong bản ghi đi lại.
- Kiểm tra hiện diện trên hai đoạn thời gian chừa riêng, cách dữ liệu huấn luyện ít nhất 3 giây, cho kết quả 142/142 cửa sổ phòng trống và 144/144 cửa sổ đi lại. Các đoạn này thuộc cùng hai phiên thu. Đánh giá độ chính xác tọa độ trên máy đó cần video tham chiếu đồng bộ.

## Đọc báo cáo

- `model.artifact_sha256` định danh tệp checkpoint, bộ chuẩn hóa và parser. Có thể đối chiếu từng giá trị trên Windows bằng `Get-FileHash -Algorithm SHA256 <tệp>`.
- `counts.records`, `accepted`, `filtered` và `backlog` mô tả đầu vào. `window_candidates`, `rejected_cadence`, `dropped_windows`, `model_failures` và `predictions` mô tả quá trình dựng cửa sổ và suy luận. `present_predictions` và `empty_predictions` đếm kết quả hiện diện. Trong số `present_predictions`, `pose_predictions` đếm các kết quả có đủ 33 khớp và `pose_suppressed` đếm các cửa sổ bị chặn tọa độ do tín hiệu nằm ngoài dải kiểm tra. `gaps` gồm khoảng cách vượt ngưỡng hoặc timestamp không tăng. `timestamp_order_errors` đếm riêng lỗi thứ tự.
- `prediction_rate_hz` lấy số khoảng giữa dự đoán chia thời gian đơn điệu từ dự đoán đầu đến cuối. Thời gian này bao gồm giai đoạn chờ dữ liệu giữa các dự đoán. `transitions` giữ tối đa 64 lần đổi trạng thái gần nhất và `transition_count` ghi tổng số lần đổi.
- `timing_ms` có số mẫu, p50, p95 và tối đa trên 1.024 mẫu gần nhất cho mỗi giai đoạn. `backend` tính ghép bản ghi mục tiêu, dựng cửa sổ, chờ và suy luận. `ws_send` đo lời gọi gửi ASGI theo từng client. Số đo này chưa bao gồm truyền mạng tới trình duyệt, vẽ giao diện, hoặc thời gian Intel recorder đợi ghi đủ cửa sổ.
- `websocket.discarded` đếm thông điệp bị thay thế trong hộp thư client hoặc bị loại khi trạng thái mất hiệu lực. `counts.dropped_windows` là cửa sổ bị bỏ ở hàng đợi suy luận. Hai loại số đếm có phạm vi khác nhau.

## Kiểm thử lỗi và thu trực tiếp

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/test_backend.py
```

Các ca tích hợp kiểm tra bản ghi nhị phân ghi dở, log chưa đủ dòng, lệch timestamp giữa hai tệp, mất mẫu, ghi trở lại sau timeout, lỗi mô hình, kết nối lại và dừng. Ca live trong bộ kiểm thử dùng byte Intel thật được ghi thêm vào thư mục tạm để thử bộ đọc tệp. Khi nguồn mất, client kết nối lại chỉ nhận trạng thái hiện tại. Cửa sổ hợp lệ mới được dựng sau khi nguồn phục hồi.

Để nghiệm thu với thiết bị, chạy backend với `--source live --auto-capture` theo [hướng dẫn thu trực tiếp](README.md#thu-trực-tiếp-bằng-nút-bắt-đầu-phiên). Lần lượt chọn 30 giây, 1 phút và 2 phút, rồi kiểm tra đồng hồ đếm ngược. Xác nhận hai tệp trong thư mục phiên mới tăng kích thước, biểu đồ hiện mẫu đầu khi backend còn tích lũy và trạng thái chuyển từ `warming_up` sang `ready` khi có dự đoán. Kiểm tra biểu đồ sau khi phiên kết thúc và xuất JSON trước khi bắt đầu phiên khác.

Ghi model máy thu, cấu hình Wi-Fi, hệ điều hành, phiên bản PyTorch và tốc độ thu vào biên bản. Nếu đo độ trễ từ thời điểm thu tới hiển thị, ghi cách xác minh đồng hồ và sai số đồng bộ. Dùng số đo từng giai đoạn riêng khi chưa xác minh đồng hồ.

Một phiên phát được bộ xương chứng minh dữ liệu đi qua parser, cửa sổ, mô hình và giao diện. Báo cáo độ chính xác hiện diện cần nhãn thật của các phiên độc lập. Báo cáo sai số khớp cần nhãn tư thế đã đồng bộ. Giữ riêng số liệu kiểm thử kỹ thuật, ví dụ trong tập huấn luyện và kết quả kiểm tra tổng quát hóa.

## Kết quả đo với checkpoint Khanh 1.0.0

Các số dưới đây đo trên Windows 11, Intel Core i5-11300H, Python 3.12.14 và PyTorch 2.6.0 CPU. Client HTTP và WebSocket chạy qua loopback tới một tiến trình backend riêng.

- **Đi bộ độc lập ngày 25/10, replay 1×.** Đọc 1.080 bản ghi, giữ 558 mẫu VHT20 và ghi nhận 10 khoảng mất mẫu. Client ghép đủ 187/187 cặp biểu đồ–tư thế. Nhịp trung bình là 3,215 dự đoán/giây, gồm các khoảng phải tích lũy lại. Xử lý backend p50/p95 là 5,991/8,615 ms, suy luận p95 7,860 ms và gửi ASGI p95 0,340 ms. Mức tối đa của thời gian xử lý backend là 88,181 ms.
- **Phòng trống độc lập ngày 25/10, replay 8×.** Client ghép 181/181 cửa sổ. Checkpoint báo có người ở cả 181 cửa sổ. Backend loại 11 cửa sổ do cadence không phù hợp. Kết quả này cho thấy checkpoint cần được đánh giá thêm trên các phiên phòng trống độc lập.
- **Phòng trống trong dữ liệu huấn luyện ngày 03/11, replay 8×.** Client ghép 187/187 cửa sổ. Cả 187 dự đoán là phòng trống và có danh sách khớp rỗng. Đây là ca kiểm tra giao diện và đường truyền.
- **Mô phỏng bộ thu ghi nối tệp.** Ghi nối 200 bản ghi từ phiên đi bộ, tách phần đầu của một bản ghi và dòng log để thử đọc dở. Backend phát 13 dự đoán trước khi ngừng ghi và 28 dự đoán tổng cộng sau khi ghi tiếp. Client mới trong lúc mất dữ liệu nhận `disconnected / CSI_TIMEOUT` và không nhận tư thế cũ. Chuỗi trạng thái cuối là `ready → disconnected → warming_up → insufficient_data → ready → stopped`. Ca này chạy từ byte CSI đã ghi. Nghiệm thu thu CSI trực tiếp và độ trễ từ thiết bị tới màn hình vẫn cần bộ thu Intel.
