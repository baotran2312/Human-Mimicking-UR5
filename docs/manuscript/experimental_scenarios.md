# Đề xuất Kịch bản Mô phỏng và Thực nghiệm HIL cho Hệ UR5 + Tay 19-DoF

Dưới đây là 3 kịch bản mô phỏng (trong Isaac Sim 5.1) và thực nghiệm (trên hệ robot thật UR5 + tay khéo léo) được thiết kế đối xứng nhằm làm nổi bật ưu thế vượt trội của thuật toán đề xuất so với các phương pháp SOTA khác.

---

## Kịch bản 1: Điều khiển bám quỹ đạo dưới tác động của trễ lớn và Jitter mạng
*Mục tiêu: Chứng minh tính ổn định kháng trễ (Delay-Robustness) và khả năng khử nhiễu rung giật (Jitter Attenuation).*

### 1. Nhiệm vụ (Task)
*   Người vận hành giơ tay trước camera để điều khiển đầu wrist của robot vẽ một quỹ đạo hình tròn (đường kính 20 cm) trong không gian tác vụ và thực hiện thao tác nhấp nhả các ngón tay bám theo chu kỳ.

### 2. Thiết lập môi trường truyền thông (Network Configuration)
*   **Trễ mô phỏng:** Thiết lập độ trễ nhân tạo trung bình $\tau_{true} = 150 \text{ ms}$, kết hợp thêm nhiễu jitter ngẫu nhiên tần số cao $n(t)$ có biên độ $\pm 50 \text{ ms}$, và giả lập tỷ lệ mất gói tin (packet loss) 10%.
*   **Cấu hình phần cứng:** Mạng Wi-Fi nội bộ phòng Lab được cấu hình định tuyến qua một node trung gian để tạo trễ vật lý tương tự.

### 3. Phương pháp đối chứng (Comparison Methods)
1.  **Luật CLIK tiêu chuẩn (như AnyTeleop [8]):** Bộ điều khiển không có bù trễ vi phân và không có bộ lọc LPF làm mịn trễ.
2.  **Proposed CLIK (Thuật toán đề xuất):** Sử dụng bộ lọc LPF thông số $T_f$ thỏa mãn điều kiện ổn định và bù trễ vi phân vòng kín dựa trên trễ lọc $\tau_{filt}(t)$.

### 4. Chỉ số đo lường (Metrics)
*   **Mean Absolute Error (MAE):** Sai số bám vị trí trung bình trong không gian tác vụ ($e_p(t)$).
*   **Joint Velocity Chattering Index (JVCI):** Chỉ số đo độ rung giật và chattering của khớp:
    $$\text{JVCI} = \int_{0}^{T} \|\ddot{q}(t)\|^2 dt$$
    *(Giá trị JVCI càng nhỏ chứng tỏ chuyển động của khớp càng mượt mà).*
*   **Trajectory Smoothness (Vận tốc tác vụ):** Sự trơn tru của quỹ đạo đầu ngón tay robot.

---

## Kịch bản 2: Vượt điểm kỳ dị động học tại biên vùng làm việc (Singularity Traversal)
*Mục tiêu: Chứng minh tính kháng kỳ dị vững (Singularity-Robustness) và giới hạn rò rỉ Null-space ở mức $\mathcal{O}(\lambda)$.*

### 1. Nhiệm vụ (Task)
*   Người vận hành điều khiển cánh tay robot duỗi thẳng tối đa ra biên vùng làm việc (workspace boundary) và dịch chuyển cổ tay đi qua các điểm kỳ dị sâu của UR5 (nơi trị riêng bé nhất của Jacobian $\sigma_{min}(J) \to 0$). Trong lúc này, các ngón tay robot vẫn phải thực hiện cử động bóp-nhả liên tục.

### 2. Phương pháp đối chứng (Comparison Methods)
1.  **CLIK dùng nghịch đảo Moore-Penrose (như Li et al. [4]):** Sử dụng nghịch đảo giả thông thường $J^{\dagger} = J^T(J J^T)^{-1}$.
2.  **Proposed CLIK (Thuật toán đề xuất):** Sử dụng nghịch đảo Damped Least-Squares $J^{\dagger}_{DLS}$ kết hợp chứng minh rò rỉ Null-space bị chặn bởi $\lambda/2$.

### 3. Chỉ số đo lường (Metrics)
*   **Maximum Joint Velocity ($\max \|\dot{q}\|$):** Theo dõi xem vận tốc khớp có bị bùng nổ (vượt quá giới hạn vật lý của động cơ UR5 $\pm 3.14 \text{ rad/s}$) khi đi qua điểm kỳ dị hay không.
*   **Task-Space Tracking Deviation:** Sai số lệch hướng của cổ tay khi robot đi qua điểm kỳ dị.
*   **Null-Space Task Tracking Error:** Sai số bám góc khớp ngón tay phụ ($q_h$) để kiểm tra xem rò rỉ cơ học $\mathcal{O}(\lambda)$ từ tác vụ bám chính có làm hỏng chuyển động của các ngón tay hay không.

---

## Kịch bản 3: Cầm nắm khéo léo trong không gian hẹp có vật cản (Multi-Objective Teleoperation)
*Mục tiêu: Chứng minh khả năng thỏa mãn đồng thời nhiều ràng buộc phi tuyến trong không gian Null-space dư thừa (Joint Limits, Self-Collision, Posture Synergy).*

### 1. Nhiệm vụ (Task)
*   Người vận hành điều khiển hệ robot arm-hand luồn lách qua các chướng ngại vật (ví dụ: các thanh chắn hoặc hộp đặt gần robot) để tiếp cận và nhấc một chiếc cốc lên. Nhiệm vụ này bắt buộc cánh tay phải gập sát người (dễ tự va chạm giữa các link) và các ngón tay phải khép chặt để giữ cốc (dễ vượt quá giới hạn khớp ngón).

### 2. Phương pháp đối chứng (Comparison Methods)
1.  **CLIK tối ưu cục bộ không có thế năng Synergy (như AnyTeleop sửa đổi):** Chỉ có tránh giới hạn khớp và va chạm, không có thế năng định hướng tư thế tự nhiên $H_{post}$.
2.  **Proposed CLIK (Thuật toán đề xuất):** Tích hợp đầy đủ $H(q)$ gồm tránh va chạm Capsule (BVH), giới hạn khớp bão hòa sat, và thế năng synergy $H_{post}$ hướng các khớp ngón tay về tư thế tự nhiên của bàn tay người.

### 3. Chỉ số đo lường (Metrics)
*   **Minimum Self-Collision Distance ($d_{min}$):** Khoảng cách nhỏ nhất ghi nhận được giữa các link robot trong suốt quá trình thao tác.
*   **Joint Limit Proximity Index (JLPI):** Chỉ số đánh giá độ an toàn của khớp đối với biên vật lý (JLPI $\to 0$ là tối ưu).
*   **Anthropomorphic Posture Similarity (APS):** Chỉ số đo độ tương đồng hình học giữa bàn tay robot và tay người vận hành (Cosine similarity của các vector hướng ngón tay).
