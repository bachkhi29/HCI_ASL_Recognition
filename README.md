# 🤟 Hệ Thống Nhận Diện Ký Hiệu ASL (Dual-Hand HCI)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-f7931e.svg)](https://scikit-learn.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://developers.google.com/mediapipe)

Dự án Nhận diện Ngôn ngữ Ký hiệu Mỹ (ASL) theo thời gian thực (Real-time). Hệ thống là sự kết hợp giữa các thuật toán Học máy kinh điển và thiết kế Tương tác Người - Máy (HCI) độc đáo, nhằm giải quyết bài toán "nhận diện rác" của các ứng dụng theo dõi cử chỉ truyền thống.

## 👨‍💻 Đội ngũ phát triển
* **Thành viên:** Lý Đình Bách & Thái Hoàng Ân
* **Chuyên ngành:** Trí tuệ Nhân tạo (Artificial Intelligence)
* **Đơn vị:** Đại học Sài Gòn (SGU)



## ✨ Tính năng cốt lõi (HCI - Tương tác Kép)
Hệ thống loại bỏ hoàn toàn sự phụ thuộc vào chuột và bàn phím vật lý thông qua cơ chế phân chia nhiệm vụ cho hai tay:
* 🤚 **Tay trái (Sign Input):** Đảm nhiệm việc ra dấu các ký tự ASL (A-Z) và các lệnh tĩnh (Space, Delete). Hệ thống liên tục quét tọa độ 3D và đưa vào mô hình phân loại.
* ✊ **Tay phải (Action Trigger):** Đóng vai trò như phím "Enter". Người dùng xòe tay để ở trạng thái *Sẵn sàng* và **nắm tay lại** để chốt ký tự hiện tại từ tay trái xuất ra màn hình.
* 🛠 **Trải nghiệm Người dùng (UX/UI):** 
  * Giao diện Modern Minimalist tinh tế.
  * Tích hợp Bảng từ điển mã ASL (Chế độ Lightbox phóng to cấu trúc xương).
  * Thanh công cụ tiện ích: Phát âm thanh (Text-to-Speech), Sao chép (Copy), Xóa (Del) và Làm mới (Reset).

---

## 🧠 Dữ liệu & Huấn luyện Mô hình học máy

Dự án không sử dụng dữ liệu ảnh thô để nhận diện, mà áp dụng pipeline trích xuất đặc trưng hình học để tối ưu hóa tốc độ và độ chính xác:

1. **Trích xuất Đặc trưng (Feature Extraction):** Sử dụng MediaPipe Hands để lấy tọa độ không gian (X, Y, Z) của 21 điểm mốc khớp xương.
2. **Tiền xử lý (Preprocessing):** Tọa độ thô được dời tâm về mốc cổ tay (Landmark 0) và chuẩn hóa tỷ lệ (Scaling) để mô hình không bị ảnh hưởng bởi kích thước bàn tay hay khoảng cách đến Camera.
3. **Mô hình Phân loại (Classification):** Quá trình huấn luyện (Offline Training) được tiến hành và đối chiếu hiệu năng trên 3 thuật toán kinh điển:
   * **Random Forest (RF)**
   * **Support Vector Machine (SVM)**
   * **K-Nearest Neighbors (KNN)**
```
---
## 🚀 Hướng dẫn Cài đặt & Vận hành (Localhost)

### 1. Khởi chạy Backend (Máy chủ AI)
Cài đặt các thư viện cần thiết và khởi động máy chủ API tại thư mục `Phase2_OnlineWebService/backend/`:
```bash
pip install fastapi uvicorn mediapipe scikit-learn joblib
uvicorn main:app --reload
```
*API Server sẽ hoạt động ngầm tại cổng `http://127.0.0.1:8000`.*

### 2. Khởi chạy Frontend (Giao diện)
* Khởi động tính năng **Live Server** trên IDE (VS Code/PyCharm) cho file `Phase2_OnlineWebService/frontend/index.html`.
* Cấp quyền truy cập Camera cho trình duyệt và bắt đầu trải nghiệm tương tác.