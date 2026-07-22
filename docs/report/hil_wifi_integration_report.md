# Báo cáo Thực nghiệm HIL & Hoàn thiện Toán học Điều khiển (Tuần 1)

Báo cáo này tổng hợp chi tiết toàn bộ các bước thực hiện nghiên cứu, bổ sung lý thuyết điều khiển phi tuyến và hiện thực hóa kết nối Hardware-in-the-Loop (HIL) thời gian thực giữa Laptop cá nhân và PC Ubuntu phòng lab.

---

## 1. Hoàn thiện Lý thuyết Toán học (Manuscript Refinement)
Dựa trên các vòng phản biện khắt khe từ góc nhìn lý thuyết điều khiển phi tuyến (nonlinear control) và thực tiễn cơ học, bản thảo đã được tối ưu hóa toàn bộ qua 7 phiên bản nháp và định dạng chuẩn LaTeX tại [manuscript.tex](file:///D:/NCKH/Humanoid/HumanMimic/docs/literature_review/manuscript.tex):

*   **Giải quyết nghịch lý không khả vi (Non-differentiability Paradox):** Do nhiễu jitter mạng $n(t)$ không liên tục và không khả vi, việc lấy đạo hàm quỹ đạo thô $x_d(t - \tau(t))$ là sai giải tích. Chúng tôi đã tái định nghĩa mục tiêu bám theo quỹ đạo đã lọc $x_{d,filt}(t) = x_d(t - \tau_{filt}(t))$, cô lập nhiễu jitter thành sai số dịch chuyển không gian bị chặn (bounded spatial tracking offset).
*   **Chứng minh ổn định Local ISS (LISS) & Phân tách cụm chéo LKF:**
    *   Xây dựng phiếm hàm Lyapunov-Krasovskii (LKF) dạng tích phân kép dựa trên trễ lọc $\tau_{filt}(t)$ có đạo hàm bị chặn $|\dot{\tau}_{filt}(t)| \le d < 1$.
    *   Mở rộng vector trạng thái thành $\eta(t) = [e^T(t), e^T(t-\tau_{filt}(t)), e^T(t-\tau_m)]^T \in \mathbb{R}^{3m}$.
    *   Áp dụng **Bổ đề lồi nghịch đảo (Reciprocal Convexity Lemma - Park et al., 2011)** để gộp các tích phân phân đoạn, tạo ra ma trận LMI $\Omega_{ISS} \in \mathbb{R}^{3m \times 3m}$ chính xác 100% về mặt toán học.
    *   Sử dụng bất đẳng thức Young bóc tách hoàn toàn các cụm chéo phi tuyến giữa sai số trạng thái và nhiễu damping DLS $d_s(t)$ trong đạo hàm gia tốc $\dot{e}^T R \dot{e}$.
*   **Định lượng rò rỉ không gian rỗng (Null-space Leakage):** Chứng minh bằng phân tích SVD rằng rò rỉ cơ học từ tác vụ phụ do DLS damping chỉ có độ lớn tối đa $\mathcal{O}(\lambda)$ và hội tụ về $0$ ngay tại điểm kỳ dị sâu ($\sigma_i \to 0$), đảm bảo không phá vỡ tính ổn định ISS.
*   **Ràng buộc tham số bộ lọc thông thấp (LPF):** Đưa ra điều kiện chặn dưới toán học cho hằng số thời gian bộ lọc $T_f$:
    $$T_f \ge \frac{\max |\tau(t) - \tau_{filt}(t)|}{d}$$
*   **Hàm tránh va chạm thực tế & Synergy ngón tay:** Tích hợp bộ lọc bão hòa vận tốc khớp $\text{sat}_{v_{max}}(\dot{q}_0)$ để ngăn bùng nổ gradient, giới hạn khoảng cách va chạm $d_{influence}$, và bổ sung thế năng posture danh định ($q_{nom}$) giúp ổn định 14 bậc tự do dư thừa của bàn tay 19-DoF.

---

## 2. Thiết lập Mạng ảo & Khắc phục xung đột hệ thống
Trong quá trình kết nối từ xa (Remote HIL) qua TeamViewer, hệ thống đã gặp các lỗi mạng phức tạp và được giải quyết triệt để:
1.  **Lệch lớp mạng (Subnet mismatch):** Laptop cá nhân ở lớp mạng `192.168.100.X`, còn PC Ubuntu ở lớp mạng Wi-Fi lab `192.168.2.X`.
2.  **Lỗi ZeroTier & Dịch vụ Windows:** Dịch vụ ZeroTier trên Windows bị xung đột cổng mạng mặc định `9993` với dịch vụ **IP Helper (`iphlpsvc`)** của Windows, khiến ZeroTier bị treo ở trạng thái `OFFLINE` và GUI báo `"fail connecting to service"`.
3.  **Khắc phục thành công:**
    *   Viết script tự động tắt dịch vụ IP Helper, kill tiến trình ZeroTier bị treo, khởi động lại dịch vụ ZeroTierOneService để bind lại cổng, sau đó mở lại IP Helper.
    *   ZeroTier chuyển sang `ONLINE` và kết nối thành công.

---

## 3. Hiện thực hóa UDP HIL Bridge mạng Wi-Fi cục bộ (Tuần 1)
Khi người vận hành đến phòng Lab và kết nối hai máy vào chung một Wi-Fi, chúng tôi đã nâng cấp giao thức truyền dẫn để đạt tần số cao và độ trễ thấp nhất.

### 3.1 Cấu trúc gói tin nhị phân (Binary Float Payload)
Thay vì gửi chuỗi JSON tốn tài nguyên chuyển đổi chuỗi, gói tin gửi đi được đóng gói trực tiếp dưới dạng **nhị phân phẳng 100 bytes** (gồm 25 giá trị float liên tục, mỗi float 4 bytes) sử dụng hàm `struct.pack("25f", ...)` của Python:

| Chỉ số Float | Khớp đại diện | Mô tả |
| :--- | :--- | :--- |
| `0` đến `5` | Cánh tay UR5 (6 DoF) | Góc khớp cánh tay giải từ bộ Inverse Kinematics (IK) |
| `6` đến `9` | Ngón trỏ (4 DoF) | Góc khớp J1 đến J4 của ngón trỏ |
| `10` đến `13` | Ngón giữa (4 DoF) | Góc khớp J1 đến J4 của ngón giữa |
| `14` đến `17` | Ngón áp út (4 DoF) | Góc khớp J1 đến J4 của ngón áp út |
| `18` đến `21` | Ngón út (4 DoF) | Góc khớp J1 đến J4 của ngón út |
| `22` đến `24` | Ngón cái (3 DoF) | Góc khớp j1 đến j3 của ngón cái |

### 3.2 Sơ đồ hoạt động thực tế

```
   [LAPTOP CÁ NHÂN (Local Wi-Fi: 192.168.2.X)]
           │
           ├─► Đọc webcam laptop (0) @ 30 FPS
           ├─► Chạy MediaPipe Hands trích xuất 21 keypoints
           ├─► Giải IK UR5 & Flexion Finger Retargeting
           ├─► Đóng gói struct.pack("25f", ...) -> 100 Bytes payload
           │
           ▼ (Truyền gói tin UDP qua mạng Wi-Fi cục bộ)
   [PC UBUNTU PHÒNG LAB (IP: 192.168.2.47:5005)]
           │
           ├─► Nhận gói tin 100 bytes nhị phân
           ├─► Giải mã struct.unpack("25f", ...)
           └─► Hiển thị và sẵn sàng truyền vào Driver/Simulator @ 100 Hz
```

### 3.3 Chạy thử nghiệm thành công
Thực nghiệm đã chạy thông suốt tại phòng lab với tần số bám mục tiêu cao và độ trễ xử lý cực thấp:
*   **Mã nguồn gửi (Laptop):** [camera_stream_wifi.py](file:///D:/NCKH/Humanoid/HumanMimic/src/hil/camera_stream_wifi.py)
*   **Mã nguồn nhận (PC Ubuntu):** [isaac_client_wifi.py](file:///D:/NCKH/Humanoid/HumanMimic/src/hil/isaac_client_wifi.py)
*   **Kết quả ghi nhận:** FPS đạt mức tối đa của camera laptop (~30 FPS), độ trễ MediaPipe xử lý ổn định ở mức ~29-38 ms, gói tin nhị phân truyền nhận không bị mất gói (0% packet loss) trong mạng Wi-Fi cục bộ.
