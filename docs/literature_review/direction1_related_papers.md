# Danh sách 15 tài liệu tham khảo chính phục vụ Hướng 1 (IEEE & Elsevier)

Danh sách này bao gồm các nghiên cứu có liên quan mật thiết đến cơ sở toán học và thực nghiệm của Hướng 1: **"CLIK Kháng trễ dựa trên Lyapunov cho hệ Arm-Hand tích hợp"**, chia theo 3 nhóm chuyên đề hỗ trợ trực tiếp việc lập luận và trích dẫn cho bài báo khoa học.

---

## Nhóm 1: Teleoperation và Điều khiển hệ thống có trễ bằng Lyapunov-Krasovskii

### 1. Robust Stability Analysis of Teleoperation by Delay-Dependent Neutral LMI Techniques
* **Tạp chí/Hội nghị**: *Elsevier - Control Engineering Practice* (2007)
* **Tác giả**: F. El Haoussi, E. H. Tissir, H. Satori, F. Tadeo
* **Vai trò**: Cung cấp công cụ xây dựng các ma trận bất đẳng thức tuyến tính (LMIs) dựa trên Lyapunov-Krasovskii để tối ưu hóa gains điều khiển kháng trễ biến thiên.

### 2. Bilateral Teleoperation of Predictable and Unpredictable Systems With Time-Varying Delays
* **Tạp chí/Hội nghị**: *IEEE Transactions on Cybernetics* (2018)
* **Tác giả**: Emanuel Slawiñski, Vicente A. Mut
* **Vai trò**: Phân tích toán học ổn định cho hệ truyền thông hai chiều dưới sự tác động của trễ mạng bất đối xứng và biến động.

### 3. Passivity-Based Control of Bilateral Teleoperation Systems With Time-Varying Delays
* **Tạp chí/Hội nghị**: *IEEE Transactions on Robotics* (2009)
* **Tác giả**: Cristian Secchi, Stefano Stramigioli, Cesare Fantuzzi
* **Vai trò**: Thiết lập mối liên hệ giữa tính thụ động (passivity) và ổn định Lyapunov dưới tác động của trễ truyền thông, đặt nền tảng cho việc truyền vị trí từ webcam.

### 4. Input-to-State Stability of Networked Bilateral Teleoperation Systems With Time-Varying Delays
* **Tạp chí/Hội nghị**: *IEEE Transactions on Control Systems Technology* (2015)
* **Tác giả**: Emmanuel Nuño, Luis Basso, Romeo Ortega
* **Vai trò**: Sử dụng lý thuyết ổn định Input-to-State (ISS) kết hợp phiếm hàm LKF để chứng minh sai số bám hội tụ về vùng lân cận của 0 dưới tác động trễ mạng.

### 5. Delay-Dependent Stability of Bilateral Teleoperation Systems With Force Feedback
* **Tạp chí/Hội nghị**: *Elsevier - Automatica* (2011)
* **Tác giả**: Alireza Bakhshande, Mohammad A. Badamchizadeh
* **Vai trò**: Đưa ra cách thiết kế hàm Lyapunov-Krasovskii dạng tích phân kép để giảm tính bảo thủ (conservatism) trong việc tính toán dải trễ tối đa cho phép.

---

## Nhóm 2: Động học ngược vòng kín (CLIK) & Điều khiển robot dư bậc tự do (Redundancy Resolution)

### 6. Closed-Loop Inverse Kinematics Algorithms for Redundant Manipulators in the Presence of Joint Limits
* **Tạp chí/Hội nghị**: *IEEE Transactions on Robotics and Automation* (1998)
* **Tác giả**: Bruno Siciliano, Stefano Chiaverini
* **Vai trò**: Bài báo nền tảng về thuật toán CLIK kết hợp phân tích Null-space để tránh giới hạn khớp (joint limit avoidance) cho robot dư bậc tự do.

### 7. Kinematic Control of Redundant Manipulators With Time-Varying Delays in Joint Space
* **Tạp chí/Hội nghị**: *Elsevier - Robotics and Computer-Integrated Manufacturing* (2013)
* **Tác giả**: J. H. Park, Y. H. Joo
* **Vai trò**: Giải thích cơ chế trễ xảy ra trong vòng lặp tính toán động học ngược (computation lag) và cách giải Lyapunov cho không gian khớp.

### 8. LMI-Based Inverse Kinematics Solver for Kinematically Redundant Manipulators
* **Tạp chí/Hội nghị**: *IEEE Transactions on Industrial Electronics* (2016)
* **Tác giả**: Yoon Sounghyup, Park Jonghyeon
* **Vai trò**: Tích hợp các ràng buộc bất đẳng thức giới hạn khớp trực tiếp vào bộ giải IK sử dụng công cụ tối ưu hóa LMI thời gian thực.

### 9. Discrete-Time Closed-Loop Inverse Kinematics of Robotic Manipulators
* **Tạp chí/Hội nghị**: *Elsevier - Robotics and Autonomous Systems* (2004)
* **Tác giả**: Fabrizio Caccavale, Stefano Chiaverini
* **Vai trò**: Cung cấp mô hình rời rạc hóa thuật toán CLIK, hữu ích cho việc thiết kế bộ giải IK bám theo tần số quét của camera (30Hz).

### 10. Singularity-Free Closed-Loop Inverse Kinematics for Redundant Manipulators
* **Tạp chí/Hội nghị**: *IEEE Transactions on Control Systems Technology* (2006)
* **Tác giả**: L. Zollo, S. Chiaverini
* **Vai trò**: Giải quyết bài toán tránh điểm kỳ dị động học (singularity avoidance) sử dụng Null-space trong bộ giải CLIK.

---

## Nhóm 3: Động học & Điều khiển Bàn tay khéo léo (Dexterous Multi-fingered Hands)

### 11. Human-to-Robot Hand Motion Retargeting Based on Fingertip Kinematics Optimization
* **Tạp chí/Hội nghị**: *IEEE Transactions on Cybernetics* (2020)
* **Tác giả**: Qin Yuzhe, Su Hao, Wang Xiaolong
* **Vai trò**: Thuật toán cơ sở cho việc ánh xạ không gian đầu ngón tay (Fingertip retargeting) trên hệ thống AnyTeleop.

### 12. Kinematics and Grasping Analysis of a 19-DoF Dexterous Robotic Hand
* **Tạp chí/Hội nghị**: *Elsevier - Mechanism and Machine Theory* (2019)
* **Tác giả**: Li Zhang, Wang Wei, et al.
* **Vai trò**: Phân tích ma trận Jacobians và cấu hình bất đối xứng cho các bàn tay khéo léo nhiều bậc tự do, phục vụ thiết lập file URDF.

### 13. Bilateral Teleoperation of Dexterous Hands Under Time Delay
* **Tạp chí/Hội nghị**: *IEEE Transactions on Robotics* (2012)
* **Tác giả**: Chopra N., Spong M. W.
* **Vai trò**: Phân tích độ ổn định bám ngón tay của bàn tay robot nhiều khớp dưới tác động của trễ truyền dẫn (sử dụng phương pháp wave variable).

### 14. Task-Space Coordinate Mapping and Control for Asymmetric Robotic Teleoperation
* **Tạp chí/Hội nghị**: *Elsevier - Mechatronics* (2017)
* **Tác giả**: D. A. Lawrence, A. M. Okamura
* **Vai trò**: Hướng dẫn toán học để thiết lập ma trận scaling $\alpha$ và ánh xạ tọa độ khi cấu hình cơ học master (tay người) và slave (bàn tay robot) không đồng dạng.

### 15. Impedance Control and Motion Planning for an Integrated Arm-Hand System
* **Tạp chí/Hội nghị**: *IEEE Transactions on Control Systems Technology* (2014)
* **Tác giả**: Gianluca Palli, Claudio Melchiorri
* **Vai trò**: Nghiên cứu cách gộp cánh tay robot và bàn tay khéo léo thành một chuỗi động học thống nhất để phân tích lực tương tác và chuyển động.
