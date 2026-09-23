# ASL Studio · Dual-Hand HCI

**[Mở web app ASL Studio](https://asl-studio-hci.onrender.com/)** · [Kiểm tra trạng thái mô hình](https://asl-studio-hci.onrender.com/api/health)

Web app nhận diện bảng chữ cái ASL bằng camera, dựa trên mô hình SVM phân cấp của đồ án HCI. **Tay trái** tạo ký hiệu; **tay phải** mở rồi nắm để chốt chữ. Ứng dụng còn có bảng tham khảo A–Z, đánh vần từ bằng ảnh, sao chép và phát âm văn bản.

> Đây là bản thử nghiệm học tập, nhận diện từng khung hình và chủ yếu phù hợp ký hiệu tĩnh. J và Z có chuyển động nên có thể nhận diện chưa chính xác. Công cụ đánh vần từng chữ cái, không dịch ngữ pháp ASL hay thay thế phiên dịch ngôn ngữ ký hiệu.

## Chạy trên máy

Yêu cầu Python 3.11 và trình duyệt có camera. Từ **thư mục gốc của repo**:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r Phan_tich_va_nhan_dang_mau/Phase2_OnlineWebService/backend/requirements.txt
uvicorn main:app --app-dir Phan_tich_va_nhan_dang_mau/Phase2_OnlineWebService/backend --reload
```

Mở **http://127.0.0.1:8000** rồi bấm **Bật camera** và cấp quyền. Không cần Live Server hay PyCharm; giao diện và API chạy cùng một địa chỉ. `/api/health` trả trạng thái mô hình. Cần kết nối mạng để tải MediaPipe từ jsDelivr và phông chữ Google ở lần mở trang.

Video được MediaPipe xử lý trên trình duyệt; chỉ tọa độ 21 điểm của tay trái được gửi đến API cùng máy chủ, không gửi khung hình camera. Khi triển khai công khai, người dùng cần mở trang bằng HTTPS để trình duyệt cấp quyền camera.

## Triển khai web app

Repo có [`render.yaml`](render.yaml) để triển khai FastAPI và giao diện trên **một Render Web Service**:

1. Đăng nhập [Render](https://dashboard.render.com/) và chọn **New → Blueprint**.
2. Kết nối tài khoản GitHub, chọn repo `bachkhi29/HCI_ASL_Recognition`, nhánh `master` và áp dụng blueprint.
3. Chờ build và kiểm tra `https://<tên-dịch-vụ>.onrender.com/api/health` trả `ready: true`; sau đó mở trang gốc và cấp quyền camera.
4. URL đang chạy: **https://asl-studio-hci.onrender.com/**. Ghim URL này vào **GitHub repo → About → Edit → Website**.

Gói miễn phí Render có thể ngủ khi không có lượt truy cập; lần mở đầu sau một thời gian không dùng sẽ chậm. Mô hình `joblib` chỉ nên tải từ nguồn tin cậy; các file `.pkl` hiện có được lưu trong repo. Nếu `/api/health` trả 503, xem log khởi động để kiểm tra tương thích phiên bản scikit-learn của mô hình.

## Cấu trúc

- `Phan_tich_va_nhan_dang_mau/Phase2_OnlineWebService/frontend/`: giao diện web, MediaPipe trong trình duyệt, ảnh ký hiệu.
- `Phan_tich_va_nhan_dang_mau/Phase2_OnlineWebService/backend/`: FastAPI, chuẩn hóa điểm mốc, mô hình SVM.
- `Phan_tich_va_nhan_dang_mau/Phase1_OfflineTraining/`: dữ liệu và mã huấn luyện nguyên bản.

**Tác giả đồ án:** Lý Đình Bách và Thái Hoàng Ân, Đại học Sài Gòn.
