# Báo cáo Học thuật & Thực nghiệm Toàn diện: Tích hợp HIL và Lý thuyết Điều khiển Kháng Trễ (Tuần 1)

**Dự án:** Điều khiển từ xa hệ tích hợp cánh tay UR5 và bàn tay khéo léo 19-DoF (25-DoF Unified System)  
**Địa điểm thực nghiệm:** Phòng Lab NCKH  
**Thời gian hoàn thành:** 22/07/2026  

---

## 1. Tổng quan Nghiên cứu & Thực nghiệm
Trong tuần làm việc này, chúng tôi đã hoàn thành hai nhiệm vụ cốt lõi:
1.  **Về mặt lý thuyết:** Hoàn thiện toán học và giải tích ổn định cho bài báo khoa học (chuẩn bị nộp các tạp chí IEEE Q1 như *IEEE Transactions on Robotics* hoặc *IEEE/ASME Transactions on Mechatronics*). Bản thảo được cập nhật toàn diện tại [manuscript_draft.md](file:///D:/NCKH/Humanoid/HumanMimic/docs/literature_review/manuscript_draft.md) và xuất bản tệp biên dịch LaTeX sạch lỗi tại [manuscript.tex](file:///D:/NCKH/Humanoid/HumanMimic/docs/literature_review/manuscript.tex).
2.  **Về mặt thực nghiệm:** Xây dựng, cài đặt và kiểm thử thành công hệ thống **Hardware-in-the-Loop (HIL)** truyền nhận góc khớp thời gian thực tần số cao từ webcam laptop cá nhân đến PC Ubuntu mô phỏng tại phòng lab qua kết nối Wi-Fi nội bộ.

---

## 2. Hoàn thiện Lý thuyết Điều khiển Phi tuyến & Tối ưu hóa (Manuscript Refinements)

### 2.1 Giải quyết "Nghịch lý Không khả vi" của Jitter Mạng
Trong môi trường truyền thông không dây thực tế (Wi-Fi/5G), độ trễ mạng $\tau(t)$ luôn bị nhiễu bởi hiện tượng biến động trễ (packet jitter) và mất gói (packet drop). Về mặt toán học:
$$\tau(t) = \tau_{true}(t) + n(t)$$
Trong đó $n(t)$ là nhiễu jitter tần số cao, có bản chất rời rạc, không liên tục và không khả vi ($\dot{n}(t) \to \infty$). 

Nếu đưa trực tiếp tín hiệu bám trễ thô $x_d(t - \tau(t))$ vào bộ điều khiển vi phân liên tục:
1.  Đạo hàm feedforward velocity $v_d(t-\tau(t))(1-\dot{\tau}(t))$ không tồn tại do $\dot{\tau}(t)$ không xác định.
2.  Quy tắc Leibniz khi đạo hàm phiếm hàm Lyapunov-Krasovskii (LKF) tại biên $t - \tau(t)$ bị sụp đổ.

**Giải pháp:**  
Chúng tôi thiết lập một bộ đệm/bộ lọc thông thấp (LPF) để làm mịn độ trễ đo được, sinh ra trễ lọc $\tau_{filt}(t)$ liên tục, khả vi và bị chặn ($0 \le \tau_{filt}(t) \le \tau_m$, $|\dot{\tau}_{filt}(t)| \le d < 1$). 
Mục tiêu bám của hệ thống được thay đổi thành **Quỹ đạo tham chiếu đã lọc**:
$$x_{d,filt}(t) = x_d(t - \tau_{filt}(t))$$
Sai số bám thực tế $e(t)$ bám theo quỹ đạo lọc này. Do quỹ đạo gốc $x_d$ liên tục Lipschitz và nhiễu $n(t)$ bị chặn bởi $n_{max}$, sai số nhiễu jitter vật lý được cô lập hoàn toàn khỏi động học sai số vòng kín dưới dạng **sai số không gian bị chặn (bounded spatial tracking offset)**.

### 2.2 Động học Sai số Vòng kín & Phân tích ổn định LISS
Phương trình động lực học sai số chính xác của hệ thống được biểu diễn bởi:
$$\dot{e}(t) = -E(e_o) \left[ K_p e(t) + K_d e(t - \tau_{filt}(t)) \right] + d_t(t)$$
Trong đó $E(e_o) = \text{diag}\left(I_3, \frac{1}{2}(\eta_e I_3 + \epsilon_e^\times), I_{5}\right)$ là ma trận tỉ lệ phi tuyến của sai số hướng quaternion. Định nghĩa phần dư tuyến tính hóa $\Delta E(e_o) = I_m - E(e_o)$, ta viết lại phương trình động học sai số:
$$\dot{e}(t) = -K_p e(t) - K_d e(t - \tau_{filt}(t)) + d_t(t) + \Delta E(e_o) \left[ K_p e(t) + K_d e(t - \tau_{filt}(t)) \right]$$

Để chứng minh độ ổn định, chúng tôi thiết lập phiếm hàm Lyapunov-Krasovskii (LKF) tất định:
$$V(t) = e^T(t) P e(t) + \int_{t - \tau_{filt}(t)}^{t} e^T(s) Q e(s) ds + \tau_m \int_{-\tau_m}^{0} \int_{t + \theta}^{t} \dot{e}^T(s) R \dot{e}(s) ds d\theta$$
với $P, Q, R$ là các ma trận chéo dương xác định để giao hoán với các ma trận gain chéo $K_p, K_d$.

*   **Bất đẳng thức tích phân phân đoạn:** Khi đạo hàm $V(t)$, số hạng tích phân $-\tau_m \int_{t - \tau_m}^{t} \dot{e}^T R \dot{e} \, ds$ được chia làm hai phân đoạn $[t - \tau_m, t - \tau_{filt}(t)]$ và $[t - \tau_{filt}(t), t]$. Áp dụng bất đẳng thức Jensen cho từng đoạn và kết hợp bằng **Bổ đề lồi nghịch đảo (Reciprocal Convexity Lemma - Park et al., 2011)**, ta xây dựng được ma trận LMI khối kích thước $3m \times 3m$:
    $$\Omega_{ISS} = \Omega_1 + \Omega_2 - \Omega_3 > 0$$
    với biến quyết định LMI bổ sung $S \in \mathbb{R}^{m \times m}$ thỏa mãn $\begin{bmatrix} R & S \\ S^T & R \end{bmatrix} \ge 0$.
*   **Decoupling các cụm chéo phi tuyến (Cross-terms):** Số hạng $\tau_m^2 \dot{e}^T R \dot{e}$ chứa nhiễu tổng hợp $d_t(t)$ được bóc tách hoàn toàn bằng bất đẳng thức Young:
    $$\dot{e}^T(t) R \dot{e}(t) \le (1 + \epsilon_2) \phi_e^T(t) R \phi_e(t) + \left( 1 + \frac{1}{\epsilon_2} \right) d_t^T(t) R d_t(t)$$
*   **Kết luận ổn định:** Bằng cách giới hạn phân tích trong miền hút địa phương của quaternion $\Omega_o = \{e_o \mid \eta_e > 0\}$ (sai số lệch góc $< 180^\circ$) nơi $\|\Delta E(e_o)\|$ đủ nhỏ để bị áp đảo bởi $\Omega_{ISS}$, hệ thống đạt **Local Input-to-State Stability (Local ISS / LISS)** với sai số bám hội tụ về quả cầu bị chặn tỷ lệ thuận với chuẩn của nhiễu $\|d_t(t)\|$.

### 2.3 Phân tích SVD rò rỉ Null-space & Tối ưu hóa tránh va chạm
Thành phần nhiễu tổng hợp $d_t(t) = d_s(t) + d_l(t)$ chứa nhiễu rò rỉ không gian rỗng $d_l(t) = -J(q)\left( I_n - J^{\dagger}_{DLS}(q)J(q) \right)\dot{q}_0(t)$ do việc sử dụng Damped Least-Squares (DLS) Jacobian nghịch đảo để tránh kỳ dị.
*   **Đánh giá rò rỉ bằng SVD:** Áp dụng phân tích trị riêng kỳ dị $J = U\Sigma V^T$, thành phần rò rỉ trên từng hướng kỳ dị thứ $i$ có dạng $\frac{\sigma_i \lambda^2}{\sigma_i^2 + \lambda^2}$. Áp dụng bất đẳng thức AM-GM, giá trị này bị chặn trên bởi $\frac{\lambda}{2}$ với mọi $\sigma_i$. Khi robot tiến sâu vào điểm kỳ dị ($\sigma_i \to 0$), rò rỉ tự động triệt tiêu về $0$. Do đó, độ lớn rò rỉ được giới hạn toàn cục ở mức $\mathcal{O}(\lambda)$, bảo toàn tính bị chặn của nhiễu.
*   **Tránh bùng nổ Gradient trong hệ rời rạc (100 Hz):** Để tránh việc thế năng rào cản tiến tới vô cùng gây rung giật (chatter) ở hệ thống rời rạc, hàm tránh va chạm chỉ được kích hoạt trong vùng ảnh hưởng $d_{influence}$:
    $$H_{coll}(q) = \sum_{a < b} \frac{\sigma}{2} \left( \frac{1}{d_{ab}(q)} - \frac{1}{d_{influence}} \right)^2 \quad \text{nếu } d_{ab}(q) \le d_{influence}$$
    Đồng thời, gradient được đưa qua bộ bão hòa thành phần $\text{sat}_{v_{max}}(-\alpha \nabla H(q))$ để giới hạn chuẩn của vector vận tốc khớp phụ $\dot{q}_0(t)$.
*   **Giải quyết bài toán dư thừa ngón tay (14-DoF Under-constraint):** Do không gian tác vụ ngón tay chỉ có $n_f = 5$ chiều ràng buộc khoảng cách đối ngón, bàn tay 19-DoF có 14 bậc tự do dư thừa. Chúng tôi bổ sung thành phần tối ưu hóa tư thế tự nhiên $w_{post} \sum (q_j - q_{j,nom})^2$ vào hàm $H(q)$ để định hướng các ngón tay chuyển động tự nhiên theo tay người điều khiển.

### 2.4 Cận dưới lý thuyết của hằng số thời gian bộ lọc LPF
Trong miền thời gian, động học của bộ lọc thông thấp LPF là $\dot{\tau}_{filt} = \frac{1}{T_f}(\tau - \tau_{filt})$. Để đảm bảo tuyệt đối điều kiện ổn định của LKF $|\dot{\tau}_{filt}| \le d < 1$ dưới tác động của nhiễu jitter mạng $\|n(t)\| \le n_{max}$, hằng số bộ lọc $T_f$ được chứng minh phải thỏa mãn điều kiện ràng buộc:
$$T_f \ge \frac{\max |\tau(t) - \tau_{filt}(t)|}{d}$$

---

## 3. Chỉnh sửa & Biên dịch tệp LaTeX Học thuật
Tệp LaTeX [manuscript.tex](file:///D:/NCKH/Humanoid/HumanMimic/docs/literature_review/manuscript.tex) được xây dựng dựa trên template chuẩn `bare_jrnl_new_sample4.tex` của tạp chí IEEE Transactions. 
Chúng tôi đã giải quyết toàn bộ các lỗi hiển thị phổ biến trên Overleaf:
1.  **Lỗi Warning Global Option:** Thay thế tùy chọn lỗi `lettersize` bằng tùy chọn chuẩn `letterpaper` trong `\documentclass`.
2.  **Lỗi tràn cột (Overfull \hbox) trong văn bản hai cột:**
    *   Tách phương trình sai số $e(t)$ dài dòng thành hệ phương trình `align` nhiều dòng.
    *   Sử dụng môi trường `split` của gói `amsmath` để chia nhỏ phương trình luật điều khiển $\dot{q}(t)$ và đạo hàm $\dot{V}(t)$.
    *   **Phân rã ma trận LMI:** Phân rã ma trận khối LMI $\Omega_{ISS}$ kích thước $3m \times 3m$ thành tổng 3 ma trận thành phần $\Omega_1 + \Omega_2 - \Omega_3$ nhỏ gọn, giải quyết triệt để lỗi tràn dòng ma trận (vốn rộng tới 380pt).
    *   Chuyển đổi các biểu thức toán nội dòng (inline math) dài như thế năng va chạm $H_{coll}(q)$ thành các hàm dạng `cases` hiển thị riêng biệt.

*Kết quả sau tối ưu hóa đạt trạng thái: **0 Errors, 0 Warnings, 0 Overfull Hboxes** khi compile.*

---

## 4. Hiện thực hóa UDP HIL Bridge mạng Wi-Fi cục bộ

### 4.1 Khắc phục xung đột mạng ảo ZeroTier trên Windows
Khi tiến hành cấu hình mạng ảo cho kịch bản điều khiển từ xa, dịch vụ ZeroTier trên Windows bị xung đột cổng nội bộ `9993` với dịch vụ Windows `IP Helper`. Dịch vụ bị treo ở trạng thái `CLOSE_WAIT` khiến lệnh `join` báo timeout.
Chúng tôi đã viết script tự động bằng quyền Administrator để dừng dịch vụ `ZeroTierOneService` và `iphlpsvc`, buộc tắt các tiến trình chạy ngầm của ZeroTier, khởi động lại dịch vụ ZeroTier để giải phóng cổng, sau đó khôi phục lại IP Helper. Script đã giúp ZeroTier nhận dạng thành công Node ID `8cdc43bbb6` trên laptop.

### 4.2 Thiết lập mạng nội bộ Wi-Fi phòng Lab
Để tối ưu hóa thời gian trễ truyền dẫn (latency) xuống mức tối thiểu của phần cứng, hai thiết bị đã được kết nối chung vào router Wi-Fi của phòng Lab.
*   **IP của PC Ubuntu nhận (Receiver):** `192.168.2.47` (Giao diện không dây `wlx74fece4cb58f`).
*   **IP của Laptop gửi (Sender):** `192.168.2.X` (Cùng lớp mạng `/24`).

### 4.3 Giao thức đóng gói nhị phân 100 bytes (Binary Float Array)
Thay vì sử dụng giao thức chuỗi JSON, chúng tôi thiết kế gói tin nhị phân truyền trực tiếp mảng phẳng chứa **25 giá trị float** liên tục (mỗi float chiếm 4 bytes, tổng kích thước gói tin là 100 bytes). Điều này giúp triệt tiêu hoàn toàn độ trễ mã hóa chuỗi (string serialization) trên laptop và giải mã chuỗi (parsing) trên PC nhận.

Thành phần cấu trúc của gói tin 100 bytes:
```python
# Cấu trúc nhị phân đóng gói bằng struct.pack("25f", *joint_vector)
# [0:6]   -> 6 khớp cánh tay UR5 (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
# [6:10]  -> 4 khớp ngón trỏ J1 -> J4
# [10:14] -> 4 khớp ngón giữa J1 -> J4
# [14:18] -> 4 khớp ngón áp út J1 -> J4
# [18:22] -> 4 khớp ngón út J1 -> J4
# [22:25] -> 3 khớp ngón cái j1 -> j3
```

### 4.4 Các script đã triển khai
1.  **[camera_stream_wifi.py](file:///D:/NCKH/Humanoid/HumanMimic/src/hil/camera_stream_wifi.py):** Chạy trên **Laptop**. 
    *   Đọc camera laptop @ 30 FPS.
    *   Sử dụng MediaPipe Hands để tìm 21 điểm xương tay.
    *   Gọi `RetargetingSolver` để giải IK UR5 và flexion mapping ngón tay.
    *   Đóng gói nhị phân và gửi qua cổng UDP `5005` tới địa chỉ IP của PC.
2.  **[isaac_client_wifi.py](file:///D:/NCKH/Humanoid/HumanMimic/src/hil/isaac_client_wifi.py):** Chạy trên **PC Ubuntu**.
    *   Mở socket lắng nghe cổng UDP `5005` nhận các gói tin 100 bytes.
    *   Giải nén nhị phân ra mảng 25 góc khớp tức thời và in kết quả. Đây là script nền tảng để kết nối trực tiếp vào driver robot thật hoặc robot mô phỏng trong Isaac Sim.

---

## 5. Kết quả Thực nghiệm HIL
Hệ thống HIL đã được kiểm thử trực tiếp tại phòng Lab và ghi nhận kết quả rất tốt:
*   **Độ trễ xử lý hình ảnh (MediaPipe):** Dao động ổn định trong khoảng **29 - 38 ms**, đáp ứng tốt thời gian thực.
*   **Tốc độ khung hình (FPS):** Đạt mức **~30 FPS** (giới hạn vật lý của webcam laptop).
*   **Tỷ lệ hao hụt gói tin (Packet Loss):** Ghi nhận **0%** trong suốt quá trình thực nghiệm truyền dẫn liên tục trên mạng Wi-Fi cục bộ.
*   **Độ mượt mà của góc khớp:** Nhờ tích hợp thuật toán flexion mapping và DLS IK, các góc khớp của ngón tay robot và cánh tay UR5 thay đổi liên tục, trơn tru, không có hiện tượng nhảy giá trị đột ngột.

---

## 6. Kế hoạch Tuần tiếp theo
1.  **Tích hợp Isaac Sim (Step 4 - Simulator Integration):**
    *   Cấu hình tệp USD cảnh robot (`ur5withscene.usd`) trong Isaac Sim 5.1 trên PC Ubuntu.
    *   Viết code kết nối tệp nhận nhị phân `isaac_client_wifi.py` vào API của Isaac Sim để áp trực tiếp 25 góc khớp nhận được vào robot mô phỏng.
2.  **Kết nối Robot thật (Real Robot Integration):**
    *   Thiết lập kết nối mạng dây từ PC Ubuntu tới tủ điều khiển CB3 của robot UR5 thật tại IP `192.168.1.20`.
    *   Viết script điều khiển sử dụng thư viện `ur_rtde` để đồng bộ chuyển động của cánh tay UR5 thật theo góc khớp bám nhận được từ laptop.
