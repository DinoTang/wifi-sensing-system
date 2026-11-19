# Nghiệm thu backend CSI

Backend nhận `raw.csi` và `recorder.log` từ cùng một tiến trình Intel recorder, ghép đủ một bản ghi, giữ các mẫu VHT20 phù hợp, dựng cửa sổ 20 × 224 và đưa vào CSIPoseNet. Dashboard nhận `status`, `chart` và `pose` qua WebSocket. Đối chiếu `session_id` và `timestamp_ms` của biểu đồ và tư thế để xác định chúng thuộc cùng cửa sổ.

## Bản 1.6 trên nhánh Jayson

Bản này dùng parser Intel của Khanh và mô hình CSIPoseNet. Bộ hiện diện học từ 5.410 cửa sổ của 29 bản ghi. Mạng tư thế được huấn luyện lại bằng 1.842 cửa sổ có nhãn khớp nhìn thấy trong hai video, trên tổng số 2.237 cửa sổ của Lần 2 và 3. Lần 1 có nhãn hoạt động và được dùng để học hiện diện. Khi bấm **Bắt đầu phiên**, đầu vào là 20 gói CSI mới hợp lệ. Bộ thu và mô hình chạy trên Ubuntu. Hệ thống không yêu cầu thao tác hiệu chuẩn phòng trước. [Danh mục bản ghi](../machine-learning/configs/live-captures.csv)

- **Đạt trên video đối chiếu cho ca trống và đi bộ.** Khi bỏ cả ngày quay Lần 2 và 3 khỏi tập học hiện diện, có 0/381 cửa sổ báo có người trong hai đoạn video trống. Có 372/372 cửa sổ đi bộ xác nhận được người trong ảnh đạt ngưỡng hiện diện. Đây là số cửa sổ trước bước xác nhận ba lần của dashboard. Các cửa sổ trong cùng bản ghi chồng lấn.
- **Chưa đạt khi đứng yên.** Cùng phép thử trên nhận 188/188 cửa sổ đứng yên ở Lần 2 và 100/173 ở Lần 3. Đầu clip đứng yên Lần 3 có cảnh người thử đi vào. Bảy cửa sổ đầu được bỏ khỏi dữ liệu học hiện diện. Ca ngồi yên còn thiếu video đồng bộ để đánh giá khớp.
- **Chưa đạt cho mọi động tác tự do.** Học tư thế bằng Lần 2 rồi chấm Lần 3 cho 3.640/5.705 khớp đi bộ nằm trong bán kính 110 pixel so với khớp video. Đổi chiều cho 4.505/5.431 khớp. Trong hai bộ kiểm tra tương ứng, mô hình nhận đúng tay trái giơ ở 0/88 và 9/96 trường hợp có nhãn rõ, tay phải ở 93/95 và 97/100. Khi vừa di chuyển vừa giơ cả hai tay, kết quả là 27/46 và 3/23. Số khớp được vẽ là phép kiểm tra đầu ra. Các tọa độ cần được chấm theo video.
- **Đạt kiểm thử tích hợp.** Đã bấm bắt đầu và dừng trong Chrome ở hai kích thước màn hình, với một bộ thu thay thế ghi nối tiếp byte Intel thật vào tệp của phiên mới. Đã kiểm tra các lựa chọn 30, 60 và 120 giây, hiện biểu đồ trong ca trống, nối biểu đồ với tư thế theo mã phiên và timestamp, và xóa khớp khi mất tín hiệu hoặc khi chuyển sang cửa sổ trống. Báo cáo kiểm thử gồm 41 test Python và 8 test dashboard. **Chưa kiểm chứng trên thiết bị** thời lượng thực của IAX và độ chính xác của bản 1.6 khi thu một phiên Ubuntu mới. [Hướng dẫn chạy trên Ubuntu](README.md#thu-trực-tiếp-bằng-nút-bắt-đầu-phiên)

**Hai phiên 17/11 chỉ để tham khảo.** Bạn xác nhận hai phiên 11:47 và 11:49 không có video. Chúng đã giúp khảo sát hành vi báo nhầm của bản cũ. Bản 1.6 loại cả hai khỏi tập học và mọi chỉ số độ chính xác theo video. Khi replay phiên 11:47, giao diện báo trống ở 189/191 cửa sổ. Hai cửa sổ còn lại đang xác nhận, màn hình không hiện người hoặc bộ xương lần nào. Ở phiên 11:49, 165 cửa sổ có 33 khớp và 22 cửa sổ đang xác nhận. Không có căn cứ video để chấm các cổ tay của phiên này. Quy ước chỉ tham khảo cũng áp dụng cho mọi số liệu của hai phiên này ở các mục lịch sử bên dưới.

Video đối chiếu là `Lan2.MOV` và `Lan3.MOV`. Mốc đồng hồ xuất hiện trong từng video cho sai số khớp đồng hồ lớn nhất lần lượt 1,01 và 6,78 ms. Khung hình gần nhất cách thời điểm CSI tối đa 16,67 ms. Đã xem các khung hình tương ứng của hai đoạn trống. Riêng Lần 3, công cụ gắn nhãn ảnh vẫn xuất ít nhất sáu khớp thân người ở 115/190 khung trống. Mã học tư thế loại mọi nhãn khớp trong những đoạn trống này. Hai lần quay có cùng người và bố trí phòng. [Số liệu, SHA-256 và cách chia tập](../machine-learning/results/live-1.6-validation.json)

### Đối chiếu với yêu cầu của nhóm

Các phép thử giữ riêng theo đợt quay đã được dùng để so các phương án trong quá trình phát triển. Độ chính xác khi thi còn cần một phiên thu mới ở bố trí thực tế, có video cùng phiên và thứ tự động tác do người thử tự chọn.

Ở hai phép thử đi bộ, mô hình xuất bộ xương ở 192/192 và 180/180 cửa sổ có người nhìn thấy trên video. Sai số trung bình trên các khớp nhìn thấy là 109 và 72 pixel trên ảnh 1920 × 1080. Khi thử tăng trọng số cho từng bên tay, số lần nhận đúng tay trái tăng, nhưng số lần xác định đúng trạng thái hai cổ tay trong cảnh đứng của Lần 3 giảm từ 96/172 xuống 28/172. Checkpoint bàn giao giữ phương án có kết quả tốt hơn trong cảnh đứng và khi vừa đi vừa giơ cả hai tay.

- **“Chọn 30 giây, 1 phút hoặc 2 phút rồi bấm Bắt đầu phiên”.** Dashboard gửi thời lượng đến API, mỗi phiên có ID và thư mục riêng. IAX nhận thời lượng đã chọn. Đồng hồ bắt đầu khi recorder được khởi chạy. [Bộ chọn](../frontend/dashboard/dashboard.js#L78) · [API bắt đầu và dừng](app.py#L58) · [Mốc khởi chạy bộ thu](intel_capture.py#L124)
- **“Hiển thị CSI đang thu ngay cả khi phòng trống”.** Bộ đọc ghép gói raw và metadata đầy đủ. Backend xuất biên độ và tiến hành suy luận từ cùng các gói VHT20 hợp lệ. [Đọc cặp tệp Intel](source.py#L113) · [Tiếp nhận và dựng cửa sổ](session.py#L432)
- **“Phòng trống thì ẩn bộ xương. Có người đi thì hiện khớp người đó”.** Bộ xác nhận yêu cầu ba lần dương liên tiếp. Dashboard ghép kết quả hiện tại theo phiên và timestamp, vẽ các khớp của mô hình. Ca rời vùng đã kiểm thử bằng cách nối các đoạn CSI trống–đi bộ–trống. Độ trễ khi người thật rời phòng còn cần video cùng phiên. [Xác nhận và xuất tư thế](session.py#L517) · [Ghép tín hiệu](../frontend/dashboard/stream.js#L96) · [Hiển thị](../frontend/dashboard/dashboard.js#L339)
- **“Lấy đúng động tác, không lấy lại video lần khác”.** Mô hình nhận CSI, học quan hệ giữa các khớp video và sự thay đổi theo thời gian. Tài liệu Word còn đề cập trình xem video đối chiếu chung con trỏ thời gian. Bản này chấm video ngoại tuyến. Trình xem so sánh trực tiếp trên dashboard chưa được triển khai. [Đường suy luận](../machine-learning/src/predict.py#L73) · [Chấm trên tập CSI–video ghép thời gian](../machine-learning/src/train_live_presence.py#L154)

## Nghiệm thu SMN 1.5 với luồng một nút

Yêu cầu người dùng xác nhận: chọn thời lượng rồi bấm **Bắt đầu phiên** phải tự thu CSI trực tiếp và phân tích, bao gồm nhận ra phòng trống. Bản 1.5 bỏ bước hiệu chuẩn bắt buộc đã thêm trước đó. Hệ thống tích lũy 20 mẫu hợp lệ rồi suy luận bằng bộ phân loại trong checkpoint; không dùng tên phiên, nhãn kiểm thử hoặc video làm đầu vào live.

Bộ phân loại mới loại các vị trí điền cố định của recorder và chuẩn hóa RMS theo từng gói, từng liên kết anten trước khi tính 76 đặc trưng. Cách này giảm phụ thuộc vào mức khuếch đại biên độ. Trọng số dự đoán tọa độ của bản trước được giữ nguyên; kết quả hiện diện mới không xác nhận độ chính xác của các khớp.

- **Kiểm tra giữ riêng theo thời gian:** dùng 65% đầu mỗi hoạt động để học, bỏ tiếp 20 cửa sổ ở ranh giới, dùng phần sau để kiểm tra. Dữ liệu học có 3.410 cửa sổ. Riêng cả nhóm Oct3 gồm ba phiên phòng trống, ngồi và đi lại ngày 25/10 được giữ ngoài tập học. Các nhóm còn lại có đoạn học và kiểm tra thuộc cùng phiên. Khoảng bỏ 20 cửa sổ dài hơn độ chồng lấn 20 mẫu với bước trượt 2 mẫu.
- **Kết quả:** nhận đúng 486/486 cửa sổ phòng trống và 1.339/1.339 cửa sổ có người trong phần kiểm tra trên. Đây là các cửa sổ có chồng lấn với nhau trong cùng tập kiểm tra, không phải 1.825 lần thu độc lập. Số liệu, số mẫu từng hoạt động, SHA-256 dữ liệu NPZ và checkpoint nằm trong [JSON nghiệm thu](../machine-learning/results/presence-1.5-validation.json).
- **Kiểm tra biến đổi biên độ:** nhân riêng từng gói/liên kết với hệ số ngẫu nhiên 0,45–2 và đặt 16 vị trí điền thành 32.700. Cả 1.825 quyết định giữ nguyên. Đầu ra bộ suy luận NumPy khớp `predict_proba` của mô hình huấn luyện với sai số tuyệt đối tối đa được cho phép là 1e-12.
- **Kiểm tra luồng ứng dụng:** 39 test Python và 8 test frontend đã chạy qua. Hai ca mở Chrome ở độ rộng 390 và 1.280 px bấm bắt đầu trực tiếp, đưa byte CSI của phiên `1c7cb0d9...` và `7005d27a...` vào tiến trình thu thay thế; dashboard lần lượt hiển thị **Phòng trống**, 0/33 khớp và **Có người**, 33/33 khớp. Biểu đồ cập nhật và nút dừng kết thúc phiên. Đây là kiểm tra trên các bản ghi thực qua bộ thu thay thế, chưa phải thử lại Intel IAX trên Ubuntu sau bản sửa.
- **Phát lại đủ hai phiên Ubuntu:** phiên phòng trống `1c7cb0d9...` có 191/191 dự đoán phòng trống, 0 bộ xương; phiên `7005d27a...` có 165 dự đoán hiện diện kèm đủ khớp và 22 cửa sổ đang xác nhận sau khi tích lũy lại, không có dự đoán phòng trống. Xem [thống kê và SHA-256 tệp nguồn](../machine-learning/results/presence-1.5-replay.json). Hai phiên này có đoạn đã dùng học, nên replay là kiểm tra hồi quy và đường truyền.
- **Chưa đạt tổng quát hóa theo đợt thu:** khi học lại cùng cấu trúc nhưng bỏ hẳn Ubuntu16, cả 193/193 cửa sổ trống của nhóm đó bị báo có người. Bỏ hẳn Ubuntu17 cho kết quả tương tự ở 191/191 cửa sổ trống. Đây là hai mô hình kiểm tra riêng, không phải điểm của checkpoint đã học đoạn đầu hai nhóm. Xem [kết quả giữ riêng cả đợt](../machine-learning/results/presence-1.5-group-holdout.json). Dữ liệu hiện có chưa đủ chứng minh nhận diện phòng trống tin cậy khi điều kiện thu thay đổi.
- **Giới hạn tại hiện trường:** chưa có phiên thu mới trên Ubuntu sau bản 1.5 để xác nhận tại hiện trường. Mức điểm phần trăm cao là đầu ra mô hình và chưa được hiệu chỉnh để phản ánh độ chính xác ngoài tập dữ liệu. Giơ tay tự do vẫn cần video đồng bộ của đúng phiên để chấm.

Các mục dưới đây ghi cả kết quả lịch sử, với phiên bản checkpoint được nêu ở từng mục.

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
4. Dừng backend và chạy lại với `signal-processing/csi-data/labeled/test_2026110302522810528_20261103-025233_JoZE` để trình diễn phòng trống. Dùng một tên JSON mới. Phiên 03/11 này đã xuất hiện trong tập huấn luyện. Lặp lại với `hardware/csi_extracted/labeled/empty_03_20261025-052627_Yatl` để kiểm tra một phiên giữ ngoài tập học hiện diện 1.5.

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

## Nghiệm thu bản SMN 1.4 với hai phiên thu mới

Hai phiên VHT20 trên máy Ubuntu thu quanh 11:47 và 11:49 dùng kênh 36, độ rộng 20 MHz. SMN 1.3 báo có người ở 191/191 cửa sổ của phiên phòng trống 11:47. SMN 1.4 hiệu chỉnh bộ phân loại hiện diện bằng phiên phòng trống này cùng các phiên có nhãn trước đó. Phiên vừa đi vừa giơ tay 11:49 được giữ riêng để kiểm tra.

- Khi phát lại phiên phòng trống, SMN 1.4 có 190 cửa sổ báo phòng trống và một cửa sổ đang xác nhận. Số cửa sổ hiển thị có người là 0/191. Hai lần thử chừa đoạn thời gian của phiên này để kiểm tra có 0/92 và 0/67 cửa sổ phòng trống bị báo nhầm ở đầu ra bộ phân loại. Đây là các đoạn của cùng một phiên.
- Phiên đi lại giữ riêng có 187/187 cửa sổ được bộ phân loại nhận là có người. Backend hiển thị có người ở 165 cửa sổ và chờ xác nhận ở 22 cửa sổ, gồm lúc mới bắt đầu và sau khoảng mất gói. Các khoảng cách giữa gói VHT hợp lệ vượt 300 ms ở phiên này kéo dài khoảng 399–807 ms.
- Bộ phân loại kiểm tra lại cửa sổ nếu một gói nhảy biên độ đột ngột rồi tín hiệu trở về gần mức trước đó. Biểu đồ dùng biên độ đo được. Backend chờ ba kết quả hiện diện liên tiếp để loại tín hiệu dương tính thoáng qua.
- Trong 187 cửa sổ của phiên 11:49, tọa độ suy luận có 0 dự đoán đặt cả hai cổ tay trên vai. Video đồng bộ của Lần 2 và Lần 3 cho phép chấm riêng những cửa sổ người đổi vị trí theo chiều ngang và giơ cả hai tay. Khi học trên một đợt rồi chấm đợt kia, hai chiều đạt 36/46 và 0/23 cửa sổ. Phiên 11:49 còn thiếu video cùng mốc thu để đo sai số cổ tay theo thời gian.

## Đọc báo cáo

- `model.artifact_sha256` định danh tệp checkpoint, bộ chuẩn hóa và parser. Có thể đối chiếu từng giá trị trên Windows bằng `Get-FileHash -Algorithm SHA256 <tệp>`.
- `counts.records`, `accepted`, `filtered` và `backlog` mô tả đầu vào. `window_candidates`, `rejected_cadence`, `dropped_windows`, `model_failures` và `predictions` mô tả quá trình dựng cửa sổ và suy luận. Kết quả hiện diện gồm `present_predictions`, `empty_predictions` và `confirming_predictions`. Trong số `present_predictions`, `pose_predictions` đếm các kết quả có đủ 33 khớp và `pose_suppressed` đếm các cửa sổ bị chặn tọa độ do tín hiệu nằm ngoài dải kiểm tra. `gaps` gồm khoảng cách vượt ngưỡng hoặc timestamp không tăng. `timestamp_order_errors` đếm riêng lỗi thứ tự.
- `prediction_rate_hz` lấy số khoảng giữa dự đoán chia thời gian đơn điệu từ dự đoán đầu đến cuối. Thời gian này bao gồm giai đoạn chờ dữ liệu giữa các dự đoán. `transitions` giữ tối đa 64 lần đổi trạng thái gần nhất và `transition_count` ghi tổng số lần đổi.
- `timing_ms` có số mẫu, p50, p95 và tối đa trên 1.024 mẫu gần nhất cho mỗi giai đoạn. `backend` tính ghép bản ghi mục tiêu, dựng cửa sổ, chờ và suy luận. `ws_send` đo lời gọi gửi ASGI theo từng client. Số đo này chưa bao gồm truyền mạng tới trình duyệt, vẽ giao diện, hoặc thời gian Intel recorder đợi ghi đủ cửa sổ.
- `websocket.discarded` đếm thông điệp bị thay thế trong hộp thư client hoặc bị loại khi trạng thái mất hiệu lực. `counts.dropped_windows` là cửa sổ bị bỏ ở hàng đợi suy luận. Hai loại số đếm có phạm vi khác nhau.

## Kiểm thử lỗi và thu trực tiếp

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest -q backend/test_backend.py
```

Các ca tích hợp kiểm tra bản ghi nhị phân ghi dở, log chưa đủ dòng, lệch timestamp giữa hai tệp, mất mẫu, ghi trở lại sau timeout, lỗi mô hình, kết nối lại và dừng. Ca live trong bộ kiểm thử dùng byte Intel thật được ghi thêm vào thư mục tạm để thử bộ đọc tệp. Khi nguồn mất, client kết nối lại chỉ nhận trạng thái hiện tại. Cửa sổ hợp lệ mới được dựng sau khi nguồn phục hồi.

Để nghiệm thu với thiết bị, chạy backend với `--source live --auto-capture` theo [hướng dẫn thu trực tiếp](README.md#thu-trực-tiếp-bằng-nút-bắt-đầu-phiên). Lần lượt chọn 30 giây, 1 phút và 2 phút, rồi kiểm tra đồng hồ đếm ngược. Xác nhận hai tệp trong thư mục phiên mới tăng kích thước, biểu đồ hiện mẫu đầu khi backend còn tích lũy và trạng thái chuyển từ `warming_up` sang `ready` khi có dự đoán. Kiểm tra biểu đồ sau khi phiên kết thúc và tệp `session-report.json` tự lưu trong thư mục phiên.

Trên Ubuntu, dừng backend cũ bằng Ctrl+C. Mở terminal ở thư mục repo có `.git`, `backend/` và `.venv/`, rồi chạy:

```bash
git switch Jayson
git pull --ff-only origin Jayson
iw dev wlp0s20f3 info | grep -F 'width:'
sudo -v && .venv/bin/python -m backend \
  --source live --auto-capture --session "$HOME/csi-data/live"
```

Mở `/dashboard/`, tải lại bằng Ctrl+Shift+R và kiểm tra mô hình `smn-csiposenet-1.6`. Lần lượt chọn 30 giây, 1 phút và 2 phút. Thử ca trống, ca một người đi rồi tự chọn thời điểm giơ và hạ từng tay, và ca người rời khung hình. Để chấm, quay trọn vùng kiểm tra và ghi lại mốc đồng hồ Ubuntu trong video của từng phiên. Kiểm tra đồng hồ kết thúc, mã phiên, biểu đồ CSI, số khớp và thời điểm xóa khớp khi người rời đi. Mỗi thư mục phiên tự lưu báo cáo có `presence_trace` để so theo thời điểm.

Nếu muốn tải báo cáo qua API sau khi phiên kết thúc, trước khi bấm phiên tiếp theo, chạy:

```bash
curl -fsS http://127.0.0.1:8000/api/session/report -o session-report.json
```

Tệp tải theo cách này nằm trong thư mục terminal hiện tại. Bản tự lưu nằm ở `~/csi-data/live/<session_id>/session-report.json`, cùng `raw.csi` và `recorder.log`.

Ghi model máy thu, cấu hình Wi-Fi, hệ điều hành, phiên bản PyTorch và tốc độ thu vào biên bản. Nếu đo độ trễ từ thời điểm thu tới hiển thị, ghi cách xác minh đồng hồ và sai số đồng bộ. Dùng số đo từng giai đoạn riêng khi chưa xác minh đồng hồ.

Một phiên phát được bộ xương chứng minh dữ liệu đi qua parser, cửa sổ, mô hình và giao diện. Báo cáo độ chính xác hiện diện cần nhãn thật của các phiên độc lập. Báo cáo sai số khớp cần nhãn tư thế đã đồng bộ. Giữ riêng số liệu kiểm thử kỹ thuật, ví dụ trong tập huấn luyện và kết quả kiểm tra tổng quát hóa.

## Kết quả đo với checkpoint Khanh 1.0.0

Các số dưới đây đo trên Windows 11, Intel Core i5-11300H, Python 3.12.14 và PyTorch 2.6.0 CPU. Client HTTP và WebSocket chạy qua loopback tới một tiến trình backend riêng.

- **Đi bộ độc lập ngày 25/10, replay 1×.** Đọc 1.080 bản ghi, giữ 558 mẫu VHT20 và ghi nhận 10 khoảng mất mẫu. Client ghép đủ 187/187 cặp biểu đồ–tư thế. Nhịp trung bình là 3,215 dự đoán/giây, gồm các khoảng phải tích lũy lại. Xử lý backend p50/p95 là 5,991/8,615 ms, suy luận p95 7,860 ms và gửi ASGI p95 0,340 ms. Mức tối đa của thời gian xử lý backend là 88,181 ms.
- **Phòng trống độc lập ngày 25/10, replay 8×.** Client ghép 181/181 cửa sổ. Checkpoint báo có người ở cả 181 cửa sổ. Backend loại 11 cửa sổ do cadence không phù hợp. Kết quả này cho thấy checkpoint cần được đánh giá thêm trên các phiên phòng trống độc lập.
- **Phòng trống trong dữ liệu huấn luyện ngày 03/11, replay 8×.** Client ghép 187/187 cửa sổ. Cả 187 dự đoán là phòng trống và có danh sách khớp rỗng. Đây là ca kiểm tra giao diện và đường truyền.
- **Mô phỏng bộ thu ghi nối tệp.** Ghi nối 200 bản ghi từ phiên đi bộ, tách phần đầu của một bản ghi và dòng log để thử đọc dở. Backend phát 13 dự đoán trước khi ngừng ghi và 28 dự đoán tổng cộng sau khi ghi tiếp. Client mới trong lúc mất dữ liệu nhận `disconnected / CSI_TIMEOUT` và không nhận tư thế cũ. Chuỗi trạng thái cuối là `ready → disconnected → warming_up → insufficient_data → ready → stopped`. Ca này chạy từ byte CSI đã ghi. Nghiệm thu thu CSI trực tiếp và độ trễ từ thiết bị tới màn hình vẫn cần bộ thu Intel.
