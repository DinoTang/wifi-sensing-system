# PHIẾU GHI ĐIỀU KIỆN THU — CSI SENSING

> **Trạng thái (2026-10-26):** 9 phiên ngày 25/10 **đã được bổ sung hồi tố** vào `_sessions_metadata.csv` (khoảng cách ≈ 2 m — ước lượng; người ngồi im nằm trên đường thẳng AP–laptop; đi lại liên tục suốt phiên; 1 người trong phòng; không đổi thiết bị/Wi-Fi). **Vẫn thiếu** kích thước và vật liệu phòng, và khoảng cách chưa được đo thật. Phiếu này dành cho **các lần thu sau**.

Điền file này **ngay trong lúc thu**, mỗi phiên một dòng. Sau đó copy 4 cột `distance_m`, `position`, `environment`, `notes` vào `_sessions_metadata.csv`.

Sai còn tốt hơn thiếu: nếu không chắc một thông tin, ghi **`không chắc`**, đừng đoán số.

---

## A. Ghi một lần cho cả buổi thu (nếu không đổi giữa các phiên)

| Hạng mục | Cách đo / ghi | Ví dụ |
|---|---|---|
| Phòng | Tên phòng + kích thước ước lượng (m) | `phòng lab 4×5 m` |
| Vật liệu | Tường, sàn, có kính/gỗ/kim loại không | `tường sơn, sàn gạch, 1 cửa kính` |
| Vị trí AP | Đặt trên đâu, cao bao nhiêu | `router để trên kệ cao ~1,2 m` |
| Vị trí laptop thu CSI | Đặt trên đâu, hướng ăng-ten | `laptop mở nắp 90°, bàn bên kia phòng` |
| Khoảng cách AP ↔ laptop | **Thước dây hoặc đo trên bản đồ phòng** — ghi mét, làm tròn 0,5 m | `3,0 m` |
| Người trong phòng lúc thu | Ngoài người được đo còn ai không | `không có ai khác` |
| Lưu lượng nền | Ping đang bật hay tắt, tần số | `ping 10 Hz liên tục` |
| Thời điểm | Buổi sáng/chiều/tối | `chiều` |

**Bắt buộc chụp 2–3 tấm ảnh mỗi buổi thu** (toàn phòng, vị trí AP, vị trí laptop + người). Ảnh là metadata tốt nhất, không tốn thời gian ghi chữ.

---

## B. Ghi cho từng phiên (60 s)

| Cột trong CSV | Ghi gì | Đơn vị |
|---|---|---|
| `scenario` | `room_empty` / `person_static` / `person_moving` / `multiple_people` / … | — |
| `trial` | Số thứ tự trong lớp | `01`, `02`, `03` |
| `distance_m` | Khoảng cách người ↔ laptop (hoặc ↔ đường thẳng AP–laptop) | mét |
| `position` | Người ở đâu, theo 1 trong 3 mẫu sau: `gần AP` / `giữa đường thẳng` / `gần laptop` | — |
| `environment` | Có đổi gì so với mục A không (mở/đóng cửa, có người đi qua ngoài, quạt/tivi bật…) | — |
| `notes` | **Hoạt động có suốt 60 s không**: `liên tục` / `nghỉ ~X giây ở khoảng giây Y` / `chỉ làm X giây`. Với `multiple_people`: `đều ngồi im` hay `có người đi lại` | — |

### Thứ tự thu

- **Trộn/ngẫu nhiên thứ tự lớp** — KHÔNG thu dồn từng lớp (vd: `empty, moving, static, moving, empty, static…`).
- Nếu có thể, **trộn luôn giữa các buổi thu** (một lớp ở đầu buổi, một lớp ở cuối).

### Nếu thu `entering` / `leaving`

- Bật bấm giây.
- Ghi vào `notes`: **`người bắt đầu đi vào lúc giây X, ra khỏi vùng đo lúc giây Y`**.
- Không ghi nhãn "đi vào" cho cả phiên nếu người chỉ đi vào trong 10 s giữa phiên.

### Kiểm tra nhanh trước khi rời phòng

Chạy trong thư mục `signal-processing/`:

```bash
python src/parse_intel_csi.py
```

Nếu dòng in ra có `frames_kept` thấp bất thường, `ts_backwards > 0`, hoặc số frame/classes không như dự kiến → **thu lại ngay lúc đó**, đừng đợi về nhà mới phát hiện.

---

## C. Phiên bản điền nhanh (copy-paste)

```text
Ngày / giờ bắt đầu: __________
Người thu: __________

Phiên 1: scenario=________ trial=__ distance_m=__ position=________ environment=________ notes=________
Phiên 2: scenario=________ trial=__ distance_m=__ position=________ environment=________ notes=________
Phiên 3: scenario=________ trial=__ distance_m=__ position=________ environment=________ notes=________
...
Ảnh đã chụp:  [ ] toàn phòng  [ ] vị trí AP  [ ] vị trí laptop & người
Thứ tự thu có bị dồn từng lớp không?  [ ] không (đúng)  [ ] có (ghi rõ)
```

---

## D. Với 9 phiên đã thu ngày 2026-10-25 (không có phiếu)

Nếu **thiết bị vẫn còn nguyên vị trí như ngày đó**: có thể **đo bù** bây giờ, nhưng phải ghi rõ là đo bù:

- `distance_m`: đi thước đo lại khoảng cách AP ↔ laptop.
- `position`: mô tả lại chỗ ngồi/đi lại.
- `environment`: ghi kích thước phòng + vật liệu.
- `notes`: thêm hậu tố `(đo bù ngày __/__, chưa xác nhận có giữ nguyên vị trí từ 25/10 hay không)`.

Nếu **đã đổi chỗ hoặc không nhớ**: giữ nguyên `chưa ghi nhận` trong CSV — báo cáo đã ghi rõ đây là giới hạn của bộ dữ liệu, và cách sửa duy nhất là **thu lại có phiếu ghi**. Không được điền phỏng đoán: số sai trong metadata sẽ làm hỏng cả phần kết luận về ảnh hưởng của khoảng cách/vị trí.
