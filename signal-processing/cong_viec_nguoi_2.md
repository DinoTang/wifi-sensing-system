Công việc người thứ 2
Xử lý CSI và chuẩn bị dữ liệu tư thế từ video
Sau khi người thứ 1 bàn giao CSI, video và thông tin buổi đo, người thứ 2 chuyển dữ liệu thô thành bộ dữ liệu sạch, ghép đúng thời gian để người thứ 3 huấn luyện mô hình dự đoán tư thế. Tài liệu này mô tả các bước thực hiện và kết quả cần bàn giao.
1 Nhận dữ liệu và kiểm tra CSI
Bước 1 Nhận và sắp xếp dữ liệu
Nhận raw.csi, wifi.txt, metadata.txt, recorder.log, ping.log, start-time.txt, end-time.txt, screen-clock.csv, ghi chú phiên quay, video gốc và phiếu thông tin buổi đo. Giữ nguyên dữ liệu gốc; tạo thư mục riêng cho kết quả xử lý.
Lập danh sách sáu hoạt động: empty, standing, left_arm, right_arm, walking và mixed. Mỗi mục cần có mã phiên, người thực hiện, lần lặp, đường dẫn CSI và đường dẫn video tương ứng. Nếu metadata ghi test, đối chiếu ghi chú phiên để xác định hoạt động thực tế.
Kết quả: danh sách phiên đo có thể truy ngược từ dữ liệu xử lý về đúng bản ghi gốc.
Bước 2 Đọc và kiểm tra CSI
Dùng parser chuyển raw.csi thành dữ liệu số. Kiểm tra số bản ghi hợp lệ, đơn vị và thứ tự timestamp, số anten, số subcarrier, tốc độ nhận thực tế và các khoảng thiếu dữ liệu. Không dùng số gói ping thay cho số bản ghi CSI hợp lệ.
Tách các cấu hình CSI khác nhau trước khi xử lý, chẳng hạn loại khung, số anten hoặc số subcarrier khác nhau. Không ghép trực tiếp các bản ghi khác kích thước thành một chuỗi đầu vào.
Kết quả: CSI đọc được, thống kê chất lượng từng phiên và danh sách bản ghi bị loại kèm lý do.
Lưu ý: một file raw.csi có dữ liệu chưa đủ chứng minh dữ liệu phù hợp để học tư thế. Cần kiểm tra cấu trúc, thời gian và mức độ liên tục trước.

2 Đồng bộ và tạo nhãn tư thế
Bước 3 Đồng bộ CSI với video
Đối chiếu đồng hồ laptop nhìn thấy trong video với screen-clock.csv để xác định thời điểm trên video tương ứng với thời gian máy tính. Xác minh timestamp CSI có cùng cơ sở thời gian hoặc xây dựng ánh xạ cần thiết trước khi ghép.
Kiểm tra mốc ở cả đầu và cuối video để phát hiện độ lệch thay đổi theo thời gian. Ghi lại mốc tham chiếu, cách tính độ lệch và sai số ước tính. Không coi thời điểm nhấn Enter hoặc thời điểm tạo thư mục là thời điểm bắt đầu video chính xác.
Kết quả: bảng ánh xạ thời gian CSI và video, các mốc đã kiểm tra và những đoạn chưa thể đồng bộ tin cậy.
Bước 4 Xử lý tín hiệu
Tính biên độ từ CSI phức; xử lý bản ghi lỗi và giá trị bất thường, sau đó lọc nhiễu phù hợp. So sánh tín hiệu trước và sau xử lý để kiểm tra vẫn giữ được biến đổi do chuyển động. Nếu dùng pha, cần có quy trình hiệu chỉnh và kiểm chứng riêng.
Lưu cấu hình lọc, cách xử lý khoảng trống và mọi phép đổi tần số lấy mẫu. Với chuẩn hóa có học tham số từ dữ liệu, chỉ ước lượng tham số trên tập huấn luyện rồi áp dụng cho tập kiểm định và kiểm tra.
Kết quả: CSI đã xử lý, biểu đồ đối chiếu và cấu hình có thể chạy lại. Thống nhất cách xử lý dùng khi chạy trực tiếp để tránh khác biệt với lúc huấn luyện.
Bước 5 Trích xuất tư thế từ video
Dùng công cụ nhận diện tư thế để lấy tọa độ các khớp theo từng khung hình. Lưu thời gian khung hình, tọa độ, độ tin cậy và trạng thái hợp lệ. Kiểm tra tay trái và phải, cơ thể bị che, khớp ngoài khung hình và các đoạn công cụ nhận sai.
Thống nhất với người thứ 3 danh sách khớp, thứ tự khớp và hệ tọa độ. Với mục tiêu ban đầu là bộ xương 2D, cần ghi rõ tọa độ dùng pixel hay đã chuẩn hóa và quy tắc xử lý khớp không nhìn thấy.
Kết quả: file nhãn tư thế có timestamp và các khớp hợp lệ. Đây là nhãn do công cụ ước lượng từ video, cần kiểm tra chất lượng trước khi dùng.
Nhãn left_arm chỉ cho biết hoạt động của đoạn quay. Để học tư thế, mô hình cần tọa độ vai, khuỷu tay, cổ tay và các khớp khác tại từng thời điểm. Với empty, ghi rõ không có người; không tự tạo bộ xương.

3 Ghép mẫu và bàn giao
Bước 6 Ghép CSI với tư thế
Ghép một cửa sổ CSI với tư thế tương ứng theo quy tắc thống nhất với người thứ 3, chẳng hạn tư thế tại cuối cửa sổ nếu mô hình sẽ chạy trực tiếp. Thống nhất độ dài cửa sổ, bước trượt, thời điểm mục tiêu và cách biểu diễn dữ liệu trước khi xử lý toàn bộ.
Loại hoặc đánh dấu các mẫu đồng bộ kém, thiếu CSI hoặc mất khớp. Chia dữ liệu theo phiên quay độc lập trước khi tạo các cửa sổ chồng lấn để tránh rò rỉ giữa tập huấn luyện và tập kiểm tra.
Mỗi mẫu cần có: CSI đầu vào; tọa độ khớp mục tiêu; độ tin cậy và mặt nạ khớp hợp lệ; mã phiên, người thực hiện, hoạt động và timestamp. Có thể lưu dạng NPZ hoặc định dạng khác đã thống nhất, kèm mô tả từng trường và kích thước.
Bước 7 Kiểm tra và bàn giao cho người thứ 3
Cung cấp lệnh chạy toàn bộ quy trình, thông số xử lý, ví dụ đọc một mẫu và báo cáo chất lượng. Người thứ 3 phải đọc được dữ liệu và xác nhận kích thước đầu vào, đầu ra trước khi huấn luyện.
    • Danh sách phiên và liên kết tới CSI, video gốc.
    • Báo cáo kiểm tra CSI và bảng đồng bộ thời gian.
    • CSI đã xử lý và nhãn khớp có độ tin cậy.
    • Bộ mẫu đã ghép cùng danh sách train, validation và test theo phiên.
    • Mã xử lý, cấu hình, hướng dẫn chạy và ví dụ trực quan đối chiếu.
Việc nên làm trước tiên
Xử lý trọn vẹn một đoạn left_arm: đọc CSI → kiểm tra chất lượng → đồng bộ video → lấy khớp → ghép mẫu. Hiển thị video có bộ xương cạnh tín hiệu CSI và con trỏ thời gian chung để kiểm tra chuyển động tương ứng. Khi đoạn mẫu đúng, mới áp dụng cho cả sáu hoạt động.
Điều kiện hoàn thành và giới hạn dữ liệu
Người thứ 2 hoàn thành khi quy trình chạy lại được, các sai số và mẫu lỗi được ghi rõ, và người thứ 3 đọc được bộ mẫu đúng định dạng. Lọc nhiễu xong mới hoàn thành một phần của công việc chuẩn bị dữ liệu.
Sáu đoạn hiện tại phục vụ kiểm chứng quy trình. Nhóm cần thêm các phiên thu độc lập để huấn luyện và đánh giá; bộ dữ liệu này chưa đủ để kết luận mô hình sẽ dựng đúng tư thế người thật. Kết quả còn phụ thuộc chất lượng đồng bộ, nhãn, CSI và đánh giá của người thứ 3.