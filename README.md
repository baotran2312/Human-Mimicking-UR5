# Human-Mimicking-UR5: Delay-Robust 25-DoF Dexterous Teleoperation System

Dự án này hiện thực hóa thuật toán **Closed-Loop Inverse Kinematics (CLIK) kháng trễ** cho hệ thống điều khiển từ xa tích hợp cánh tay robot UR5 (6-DoF) và bàn tay khéo léo (19-DoF) thời gian thực từ Webcam Laptop qua mạng Wi-Fi nội bộ.

---

## 🛠️ Hướng dẫn cài đặt môi trường (Setup)

### 1. Trên máy Laptop (Windows - Máy gửi tọa độ Webcam)
Yêu cầu cài đặt Python 3.10 hoặc 3.11 và cài đặt các thư viện cần thiết:
```bash
pip install opencv-python mediapipe numpy
```

### 2. Trên máy PC Ubuntu (Ubuntu - Máy nhận và mô phỏng Isaac Sim)
Sử dụng môi trường ảo Isaac Sim (Conda/Anaconda):
```bash
conda activate env_isaacsim
# Hoặc chạy trực tiếp thông qua python.sh của Isaac Sim
```

---

## 🚀 Hướng dẫn các bước chạy thực nghiệm HIL (Hardware-in-the-Loop)

Dự án hỗ trợ 2 kịch bản chạy chính dưới đây:

### Kịch bản A: Truyền nhận 25-DoF mượt mà qua Wi-Fi (Tần số 30 FPS)
*Mô tả: Giải IK trực tiếp trên Laptop và gửi phẳng mảng nhị phân 100 bytes (25 góc khớp dạng float) sang PC mô phỏng.*

1.  **Bước 1: Khởi chạy Receiver trên máy PC (Ubuntu)**
    ```bash
    python src/hil/isaac_client_wifi.py
    ```
2.  **Bước 2: Khởi chạy Sender trên máy Laptop (Windows)**
    Nhập địa chỉ IP của máy PC Ubuntu (ví dụ: `192.168.2.47`):
    ```bash
    python src/hil/camera_stream_wifi.py --ip 192.168.2.47
    ```
    *Kết quả: Camera laptop sẽ hiển thị, góc khớp tay của bạn được trích xuất và gửi sang PC để điều khiển robot ảo.*

---

### Kịch bản B: Mô phỏng Kịch bản 1 (Đo lường sai số MAE & Rung giật JVCI thời gian thực)
*Mô tả: Laptop gửi tọa độ tay gốc (Raw Pos), PC Ubuntu tự động thêm trễ nhân tạo (150ms + Jitter 50ms), chạy thuật toán đối chứng AnyTeleop vs Proposed CLIK để xuất báo cáo.*

1.  **Bước 1: Khởi chạy Isaac Client trên máy PC (Ubuntu)**
    ```bash
    python src/hil/isaac_client_scenario1.py
    ```
    *(PC sẽ mở cổng `5005` chờ nhận dữ liệu từ Laptop và chuẩn bị mô phỏng).*
2.  **Bước 2: Khởi chạy Raw Sender trên máy Laptop (Windows)**
    Bật camera quét tay và vẽ khung xương tay trực quan:
    ```bash
    python src/hil/camera_stream_raw.py --ip 192.168.2.47
    ```
3.  **Bước 3: Nhận kết quả so sánh**
    Di chuyển tay của bạn trước camera Laptop. Trên màn hình terminal của máy **PC** sẽ liên tục in ra bảng đối chứng và tự động ghi đè dữ liệu xuất ra CSV sau mỗi 2 giây:
    *   **AnyTeleop Standard:** Gặp sai số bám trễ lớn (~30mm) và chattering khớp cao.
    *   **Proposed CLIK (Đề xuất):** Nhờ bộ lọc LPF nội suy và vận tốc đón đầu *Feedforward*, sai số bám MAE được triệt tiêu về mức cực nhỏ (< 2mm) và chuyển động mượt mà.

*Dữ liệu chi tiết của lượt chạy sẽ được lưu tự động tại: `data/hil_scenario1_results_latest.csv` và `data/hil_scenario1_summary_latest.csv`.*

---

## 📈 Hướng dẫn chạy mô phỏng số học (Numerical Simulation)

Nếu không có webcam hoặc muốn kiểm thử nhanh các phương trình toán học của bài báo trên môi trường máy tính nội bộ:
```bash
python src/hil/simulate_scenario1.py
```
*Kết quả: Chương trình sẽ tự động tính toán vi phân quỹ đạo vòng tròn, áp đặt trễ biến thiên + Jitter và in ra bảng đánh giá hiệu suất so sánh AnyTeleop vs Proposed CLIK.*

---

## 📂 Cấu trúc thư mục dự án liên quan HIL

*   `src/hil/camera_stream_wifi.py`: Sender 25 góc khớp đã giải IK.
*   `src/hil/isaac_client_wifi.py`: Receiver góc khớp hiển thị phẳng.
*   `src/hil/camera_stream_raw.py`: Sender tọa độ bàn tay người thô (đã vẽ xương tay MediaPipe).
*   `src/hil/isaac_client_scenario1.py`: Receiver tích hợp LPF nội suy, bù trễ Feedforward và xuất CSV.
*   `src/hil/simulate_scenario1.py`: Script chạy mô phỏng số học độc lập.
*   `data/`: Thư mục chứa các kết quả đo lường `.csv` từ các lượt thực nghiệm.
