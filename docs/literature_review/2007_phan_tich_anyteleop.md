# Phân tích AnyTeleop & Kế hoạch áp dụng cho UR5 + Tay 19-DOF (HIL Simulation)

> Nguồn: `docs/literature_review/AnyTeleop A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System/AnyTeleop A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System.md`
> (Qin et al., UCSD & NVIDIA). Project page: <https://yzqin.github.io/anyteleop/>
>
> **Tài nguyên sẵn có (kiểm tra 2026-07-20):**
> * Mã nguồn chính thức của AnyTeleop: <https://github.com/yzqin/anyteleop>
> * Các thư viện bổ trợ: **cuRobo** (tránh va chạm và sinh quỹ đạo thời gian thực trên GPU), **Meshcat** (Web visualizer).
>
> → Ta sẽ **ứng dụng kiến trúc phân tầng của AnyTeleop** để xây dựng hệ thống mô phỏng Hardware-in-the-Loop (HIL) cho UR5 + bàn tay 19-DoF.

---

## 1. AnyTeleop làm gì — bức tranh tổng thể

AnyTeleop là một hệ thống điều khiển từ xa (Teleoperation) bằng thị giác máy tính, được thiết kế để điều khiển **bất kỳ cấu hình cánh tay + bàn tay khéo léo nào** trong cả mô phỏng (Sim) lẫn thực tế (Real) mà không cần thiết bị đeo (gloves/mocap).

```
  [Camera RGB-D / RGB]
           │
           ▼
  [Hand Pose Detection] ────► 21 điểm khớp tay người (local) + Lòng bàn tay (global)
           │
           ▼ (Nếu dùng nhiều camera)
  [Detection Fusion] ───────► Tự động hiệu chuẩn và trộn dữ liệu (SMPL-X shape consistency)
           │
           ▼
  [Hand Pose Retargeting] ──► Tối ưu hóa khoảng cách đầu ngón (Eq. 1) -> 19 góc khớp tay robot
           │
           ▼
  [Motion Generation] ──────► Cảm nhận wrist pose -> cuRobo/IK -> 6 góc khớp UR5 (Tránh va chạm)
           │
           ▼
  [Robot Client] ───────────► Gửi lệnh điều khiển (Isaac Sim / Robot thật UR5)
```

### Kết quả chính:
* **Hỗ trợ đa cấu hình**: Chạy được trên nhiều hệ thống arm-hand (Franka+Allegro, Franka+DClaw, v.v.).
* **Độ trễ thấp**: Chu kỳ tính toán hand pose và retargeting đạt ~25-35 ms trên GPU RTX 3090 / Laptop RTX 2070.
* **Độ chính xác cao**: Trong thực tế, AnyTeleop đạt tỷ lệ thành công cao hơn hệ thống Telekinesis (Sivakumar et al.) trên 8/10 tác vụ thao tác khéo léo.

---

## 2. Từng thành phần kỹ thuật (phân tích sâu)

### 2.1 Hand Pose Detection (Nhận diện tay người)
* **Finger Keypoints**: Sử dụng **MediaPipe Hands** để trích xuất tọa độ 3D của 21 khớp xương tay trong hệ tọa độ cổ tay (wrist frame).
* **Global Wrist Pose**:
  * **Với camera RGB-D (RealSense)**: Đọc giá trị depth tại các pixel chứa keypoint 2D, giải bài toán **PnP (Perspective-n-Point)** để tìm vị trí và hướng cổ tay trong hệ tọa độ camera.
  * **Với camera RGB thường**: Sử dụng bộ dự đoán weak perspective scale (tương tự FrankMocap) để ước lượng khoảng cách từ tay đến camera qua kích thước bàn tay.

### 2.2 Detection Fusion (Trộn dữ liệu nhiều camera - Nếu áp dụng)
* **Auto-calibration**: Sử dụng chuyển động tương đối của bàn tay người trong $N$ khung hình đầu tiên để tự động tính toán góc xoay giữa các camera trong không gian $SO(3)$ mà không cần bảng căn chỉnh checkerboard.
* **Confidence Scoring**: So sánh sai số của tham số hình dáng bàn tay (SMPL-X hand shape parameters) ước lượng được với hình dáng chuẩn của người vận hành. Camera nào bị che khuất nhiều sẽ có sai số lớn và bị giảm trọng số.

### 2.3 Hand Pose Retargeting (Ánh xạ động học bàn tay)
AnyTeleop không ánh xạ góc khớp 1-1. Thay vào đó, thuật toán giải bài toán tối ưu hóa bám điểm đầu ngón (keypoint-based optimization):

$$\min_{q_t} \sum_{i=0}^{N} ||\alpha v_t^i - f_i(q_t)||^2 + \beta ||q_t - q_{t-1}||^2$$

$$\text{s.t.} \quad q_l \le q_t \le q_u$$

* $v_t^i$: Vector hướng từ cổ tay đến đầu ngón tay người $i$.
* $f_i(q_t)$: Hàm động học thuận (FK) tính vị trí đầu ngón tay robot tương ứng.
* $\alpha$: Hệ số co giãn tỉ lệ kích thước (scaling factor) giữa tay người và tay robot.
* $\beta ||q_t - q_{t-1}||^2$: Thành phần phạt gia tốc (smoothing term) giúp ngón tay robot không bị giật.

### 2.4 Motion Generation (Điều khiển cánh tay UR5)
* Nhận tọa độ Cartesian của cổ tay người ở tần số 25Hz.
* Sử dụng thư viện **cuRobo** (chạy song song trên GPU) để tìm quỹ đạo tránh va chạm tự động (collision-free trajectory) trong không gian khớp của cánh tay robot, nâng tần số điều khiển lên 120Hz trước khi gửi xuống robot.

---

## 3. Ánh xạ sang hệ của mình (UR5 + Tay 19-DOF + Isaac Sim)

Ta sẽ đơn giản hóa kiến trúc AnyTeleop để phù hợp với phần cứng Tuần 1:

| Thành phần | AnyTeleop (Paper) | Hệ của mình (HIL Prototype) |
|---|---|---|
| **Camera** | Nhiều camera RGB-D | **1 Camera RealSense D435/D455** đặt ở vị trí $(0, 1.7, 1.8)$ |
| **Nhận diện ngón** | MediaPipe | **MediaPipe Hands** (cài đặt trực tiếp vào `env_isaacsim`) |
| **Nhận diện cổ tay** | PnP với Depth | **PnP solver (OpenCV `cv2.solvePnP`)** kết hợp camera intrinsics |
| **Retargeting ngón** | Tối ưu hóa khoảng cách | **Scipy Optimization (SLSQP)** giải phương trình tối ưu hóa dựa trên FK của file `ur5dex.urdf` |
| **Điều khiển UR5** | cuRobo | **Pinocchio IK / DifferentialIKController** có sẵn trong Isaac Lab |
| **Simulator** | Isaac Gym / SAPIEN | **Isaac Sim 5.1** chạy file cảnh `ur5withscene.usd` |
| **Giao tiếp** | Custom Client/Server | **UDP Socket** (gửi mảng 25 giá trị float: 6 arm joints + 19 hand joints) |

---

## 4. Cấu trúc Code đề xuất cho HIL System

Ta sẽ tạo các file code mới trực tiếp trong thư mục `src/` của workspace:

```text
HumanMimic/
└── src/
    └── hil/
        ├── camera_stream.py      # Đọc RealSense / WebCam, chạy MediaPipe trích xuất 21 keypoints
        ├── retargeting_solver.py # Đọc file ur5dex.urdf, giải IK cho UR5 & giải tối ưu hóa khớp ngón tay
        ├── udp_bridge.py         # Đóng gói vector góc khớp và truyền qua mạng UDP
        └── isaac_client.py       # Script chạy trong Isaac Sim để nhận góc khớp UDP và áp vào robot ảo
```

---

## 5. Lộ trình thực hiện thực nghiệm (Roadmap)

1. **Bước 1: Trích xuất & Hiệu chuẩn (Vision Validation)**:
   * Viết script `camera_stream.py` kiểm tra độ trễ trích xuất MediaPipe. 
   * Đo đạc hệ số tỉ lệ $\alpha$ giữa bàn tay người điều khiển và bàn tay robot 19-DoF.
2. **Bước 2: Viết Solver Retargeting (Retargeting Development)**:
   * Sử dụng **Pinocchio** để load file `ur5dex.urdf`.
   * Viết thuật toán giải tối ưu hóa góc khớp tay dựa trên khoảng cách Euclid từ lòng bàn tay đến 5 đầu ngón tay.
3. **Bước 3: Kết nối Truyền thông (Bridge Integration)**:
   * Xây dựng luồng gửi-nhận nhị phân phẳng (flat binary 100 bytes / 25 floats) để giảm thiểu độ trễ mạng Wi-Fi cục bộ giữa Laptop và PC phòng lab.
4. **Bước 4: Vận hành thử nghiệm (Teleoperation Demo - WiFi HIL)**:
   * Khởi chạy mô phỏng Isaac Sim trên PC phòng Lab (`isaac_client_wifi.py`) đóng vai trò là bên nhận (Receiver).
   * Khởi chạy bộ đọc camera và xử lý thị giác (MediaPipe + IK) trên Laptop cá nhân (`camera_stream_wifi.py` hoặc `realsense_stream_wifi.py`) đóng vai trò là bên gửi (Sender), truyền tham số góc khớp nhị phân phẳng qua UDP WiFi nội bộ tới IP của PC để điều khiển robot ảo bám theo thời gian thực.
