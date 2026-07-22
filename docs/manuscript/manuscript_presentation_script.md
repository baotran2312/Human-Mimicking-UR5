# Kịch bản Thuyết trình: Điều khiển CLIK Kháng Trễ cho Hệ Arm-Hand 25-DoF

*   **Thời lượng tối đa:** 10 phút (Khoảng 1200 - 1400 từ nói, tốc độ vừa phải, mạch lạc).
*   **Ngôn ngữ thuyết trình:** Tiếng Việt (Văn phong học thuật, trang trọng).
*   **Tài liệu trình chiếu đi kèm:** [manuscript_slides.md](file:///D:/NCKH/Humanoid/HumanMimic/docs/manuscript/manuscript_slides.md)

---

## slide 1: Tiêu đề & Giới thiệu chung (Thời gian: 0:00 - 0:45)
**Lời thoại nói:**
> Kính thưa Thầy và các bạn, em tên là Trần Thiên Bảo. Hôm nay, thay mặt nhóm nghiên cứu, em xin phép trình bày đề tài: *"Delay-Robust Closed-Loop Inverse Kinematics Control for Unified 25-DoF Arm-Hand Dexterous Teleoperation Systems"* - Hệ thống điều khiển động học ngược vòng kín kháng trễ cho hệ tích hợp cánh tay robot UR5 và bàn tay khéo léo 19 bậc tự do. 
> 
> Đề tài này được thực hiện dưới sự hướng dẫn của TS. Ngô Hà Quang Thịnh. Nghiên cứu tập trung giải quyết bài toán bám quỹ đạo thời gian thực dưới tác động của trễ truyền thông biến thiên và singularity (kỳ dị động học) cho hệ robot arm-hand có số bậc tự do cao.

**Gợi ý trình diễn:**
*   Nói rõ ràng, tự tin, hướng mắt về phía Hội đồng. Nhấn mạnh vào cụm từ "unified 25-DoF" và "delay-robust".

---

## slide 2: Đặt vấn đề & Thách thức (Thời gian: 0:45 - 1:45)
**Lời thoại nói:**
> Hiện nay, việc thu thập dữ liệu thao tác khéo léo (dexterous manipulation) từ con người để huấn luyện các chính sách học máy như 3D Diffusion Policy hay 3D Flow Diffusion Policy đang là một xu hướng lớn. Để thu thập dữ liệu này, chúng ta cần các hệ thống teleoperation (điều khiển từ xa) bằng thị giác máy tính như AnyTeleop hay DexSim2Real.
> 
> Tuy nhiên, khi truyền dữ liệu vận tốc và góc khớp qua các đường truyền không dây như Wi-Fi hay 4G/5G, hệ thống luôn phải đối mặt với hai thách thức lớn:
> 1. Trễ biến thiên theo thời gian (time-varying latency) và nhiễu biến động trễ (packet jitter). Jitter làm cho đồ thị trễ không liên tục, không khả vi, phá hỏng các thuật toán giải tích điều khiển thông thường.
> 2. Hệ arm-hand 25 bậc tự do là một chuỗi động học cực kỳ dư thừa, rất dễ rơi vào vùng kỳ dị, tự va chạm (self-collisions) hoặc chuyển động không tự nhiên.

**Gợi ý trình diễn:**
*   Chỉ vào slide để nhấn mạnh nghịch lý không khả vi của nhiễu Jitter và độ phức tạp của 25 bậc tự do.

---

## slide 3: Thống kê SOTA & Đóng góp (Thời gian: 1:45 - 2:45)
**Lời thoại nói:**
> Để làm nổi bật nghiên cứu, nhóm đã xây dựng bảng so sánh State-of-the-Art (SOTA) với 11 nghiên cứu tiêu biểu gần đây trên IEEE Transactions. Các hệ thống hiện tại, hoặc là chỉ giải quyết hệ arm-hand không có trễ như AnyTeleop, hoặc nếu có kháng trễ thì chỉ áp dụng cho manipulator bậc tự do thấp (2 đến 7 DoF) và thiếu các ràng buộc tránh va chạm phức tạp.
>
> Đóng góp chính của nhóm bao gồm:
> * Thứ nhất, đề xuất mô hình phân tách trễ "True-and-Noise" để cô lập hoàn toàn nhiễu jitter không khả vi thông qua bộ lọc thông thấp LPF.
> * Thứ hai, chứng minh toán học chặt chẽ độ ổn định Local ISS (Input-to-State Stability) bằng phiếm hàm Lyapunov-Krasovskii kết hợp bất đẳng thức lồi nghịch đảo LMI.
> * Thứ ba, thiết lập bộ điều khiển CLIK tích hợp tránh kỳ dị DLS và tối ưu hóa tránh va chạm vật lý thời gian thực.

---

## slide 4: Mô hình động học hệ 25-DoF (Thời gian: 2:45 - 3:45)
**Lời thoại nói:**
> Tiếp theo, em xin trình bày mô hình toán học của hệ thống. Cấu hình khớp $q(t)$ của robot được tích hợp phẳng gồm 6 khớp cánh tay UR5 và 19 khớp bàn tay khéo léo. 
> 
> Để tránh việc hệ thống bị ràng buộc quá mức (kinematic over-constraint) khi giải động học thuận trong không gian tác vụ, nhóm thiết kế không gian tác vụ $x(t)$ có số chiều $m = 11$, bao gồm 3 chiều vị trí cổ tay, 3 chiều hướng quay wrist dưới dạng đơn vị Quaternion, và chỉ sử dụng 5 chiều khoảng cách tương đối giữa các đầu ngón tay (finger opposition distances). Phương trình vi phân động học thuận có dạng $v(t) = J(q) \dot{q}$, trong đó $J$ là ma trận Jacobian hệ thống kích thước 11 x 25.

---

## slide 5: Mô hình phân tách trễ mạng (Thời gian: 3:45 - 4:45)
**Lời thoại nói:**
> Dưới tác động của trễ mạng $\tau(t)$, quỹ đạo người điều khiển nhận được bị trễ. Chúng em mô hình hóa trễ đo được gồm trễ thực $\tau_{true}(t)$ khả vi và nhiễu jitter $n(t)$ bị chặn bởi $n_{max}$.
>
> Để xử lý nhiễu jitter, chúng em đưa $\tau(t)$ qua một bộ lọc thông thấp LPF để trích xuất tín hiệu trễ lọc $\tau_{filt}(t)$ có đạo hàm bị chặn dưới 1. Quỹ đạo tham chiếu mới được định nghĩa là $x_{d,filt}(t) = x_d(t - \tau_{filt}(t))$. Bằng cách này, sai số bám $e(t)$ được tính toán dựa trên quỹ đạo lọc, giúp triệt tiêu hoàn toàn nhiễu jitter khỏi vòng điều khiển vi phân. Jitter lúc này chỉ đóng vai trò là một sai số không gian bị chặn.

---

## slide 6: Bộ điều khiển CLIK kháng kỳ dị (Thời gian: 4:45 - 5:45)
**Lời thoại nói:**
> Luật điều khiển động học ngược vòng kín (CLIK) đề xuất có dạng như trên slide (Slide 6). Để đối phó với hiện tượng nổ vận tốc khớp tại điểm kỳ dị, nhóm sử dụng ma trận nghịch đảo Damped Least-Squares (DLS) với hệ số cản lambda dương. 
>
> Luật điều khiển gồm 2 phần: Phần không gian tác vụ chính bám theo quỹ đạo lọc có bù trễ vi phân $v_{d,filt}$, và phần không gian con (Null-space projector) dùng để thực hiện các nhiệm vụ phụ $\dot{q}_0(t)$ mà không làm ảnh hưởng đến độ chính xác bám của wrist và các đầu ngón tay.

---

## slide 7: Chứng minh ổn định LKF phi tuyến (Thời gian: 5:45 - 6:45)
**Lời thoại nói:**
> Để chứng minh hệ thống ổn định dưới tác động của trễ lọc $\tau_{filt}(t)$ và các thành phần nhiễu damping DLS, nhóm xây dựng phiếm hàm năng lượng Lyapunov-Krasovskii Functional (LKF) gồm 3 thành phần: Năng lượng sai số hiện tại, năng lượng tích lũy trễ của sai số, và năng lượng tích phân kép của gia tốc sai số. 
> 
> Đạo hàm phiếm hàm này theo quỹ đạo sai số phi tuyến chứa các thành phần chéo phức tạp liên quan đến nhiễu damping $d_t(t)$ và gia tốc.

---

## slide 8: Báo cáo ổn định LISS qua hệ LMI (Thời gian: 6:45 - 7:45)
**Lời thoại nói:**
> Để đánh giá đạo hàm phiếm hàm, chúng em thực hiện các bước biến đổi toán học sau:
> * Áp dụng **Bổ đề Young (Lemma 2)** để phân tách các số hạng chéo phi tuyến của nhiễu.
> * Sử dụng **Bất đẳng thức Jensen (Lemma 3)** để chặn dưới phần năng lượng gia tốc tích phân kép.
> * Cuối cùng, áp dụng **Bổ đề lồi nghịch đảo (Lemma 1 - Park et al., 2011)** để gộp các tích phân phân đoạn trễ mà không làm tăng độ bảo thủ của hệ thống.
> 
> Kết quả, chúng em thu được hệ điều kiện ma trận LMI $\Omega_{ISS} > 0$ kích thước $3m \times 3m$. Nếu hệ LMI này có nghiệm, đạo hàm phiếm hàm $\dot{V}(t)$ sẽ âm xác định ngoài quả cầu nhiễu, chứng minh hệ thống đạt ổn định **Local Input-to-State Stability (LISS)**.

---

## slide 9: Tối ưu đa mục tiêu trong Null-space (Thời gian: 7:45 - 8:30)
**Lời thoại nói:**
> Tận dụng không gian con Null-space dư thừa (14 bậc tự do), nhóm thiết kế thế năng tối ưu đa mục tiêu $H(q)$ gồm 3 thành phần:
> 1. Tránh giới hạn khớp vật lý $H_{lim}(q)$.
> 2. Dẫn hướng ngón tay robot về tư thế tự nhiên của người thông qua posture synergy $H_{post}(q)$.
> 3. Tránh tự va chạm thời gian thực $H_{coll}(q)$ sử dụng hình học bao Capsule. Thế năng va chạm này chỉ kích hoạt trong vùng ảnh hưởng $d_{inf}$ để tránh rung giật tín hiệu.

---

## slide 10: Giới hạn rò rỉ Null-space (Thời gian: 8:30 - 9:15)
**Lời thoại nói:**
> Vận tốc khớp phụ $\dot{q}_0(t)$ được đưa qua bộ bão hòa sat để giới hạn dòng lệnh, đảm bảo nhiễu Null-space luôn bị chặn. 
> 
> Đặc biệt, chúng em đưa ra chứng minh toán học bằng phân tích phân rã trị riêng kỳ dị SVD (Remark 2), chỉ ra rằng việc sử dụng nghịch đảo DLS cản sẽ tạo ra một lượng rò rỉ không gian rỗng (Null-space leakage) có độ lớn tối đa là $\mathcal{O}(\lambda)$ ngay tại điểm gần kỳ dị, và tự động triệt tiêu về $0$ tại điểm kỳ dị sâu. Điều này chứng minh tác vụ phụ Null-space không làm mất ổn định hệ thống bám chính.

---

## slide 11 & 12: Hiện thực hóa HIL & Kết luận (Thời gian: 9:15 - 10:00)
**Lời thoại nói:**
> Để chạy thuật toán thời gian thực ở tần số 100 Hz, nhóm sử dụng cấu trúc phân cấp BVH (Bounding Volume Hierarchy) của Isaac Sim để loại bỏ nhanh các cặp liên kết không có nguy cơ va chạm ($d_{ab} > d_{inf}$). 
> 
> Về thực nghiệm Hardware-in-the-Loop (HIL) tại phòng lab, nhóm đã thiết lập thành công đường truyền nhị phân UDP 100 bytes (chứa 25 giá trị float) trực tiếp qua Wi-Fi nội bộ từ Laptop (chạy MediaPipe giải góc khớp) đến PC Ubuntu (mô phỏng nhận góc khớp). Hệ thống đạt tần số bám ~30 FPS, độ trễ xử lý ảnh chỉ từ 29 - 38 ms và tỷ lệ mất gói tin đạt 0%.
> 
> Kết quả này chứng minh thuật toán đề xuất hoàn toàn khả thi cho các ứng dụng teleoperation thực tế có độ trễ. Em xin chân thành cảm ơn Hội đồng đã lắng nghe và em xin nhận các câu hỏi đóng góp ý kiến từ Thầy Cô.

---

## Các câu hỏi Hội đồng thường gặp & Cách trả lời nhanh:

1.  **Tại sao ma trận LMI $\Omega_{ISS}$ lại có kích thước $3m \times 3m$?**
    *   *Trả lời:* Do vector trạng thái mở rộng của LKF tích phân kép được định nghĩa gồm 3 thành phần: sai số hiện tại $e(t)$, sai số tại thời điểm trễ lọc $e(t - \tau_{filt}(t))$ và sai số tại giới hạn trễ tối đa $e(t - \tau_m)$. Kích thước $3m \times 3m$ là kết quả trực tiếp của việc áp dụng Bổ đề lồi nghịch đảo Park để gộp các đoạn trễ này.
2.  **Tại sao lại chọn ma trận trọng số $P, Q, R$ trong LKF là ma trận chéo?**
    *   *Trả lời:* Trong hệ thống CLIK, ma trận Jacobian và ma trận gain điều khiển $K_p, K_d$ là các ma trận chéo. Để các ma trận trọng số trong đạo hàm $\dot{V}(t)$ có tính chất giao hoán chéo (ví dụ: $P K_p = K_p P$) giúp rút gọn các biểu thức quadratic về dạng LMI thuận tiện, việc chọn $P, Q, R$ chéo là bắt buộc về mặt toán học.
3.  **Làm sao để biết robot UR5 thật có bám kịp tần số gửi 30Hz từ Laptop qua mạng không?**
    *   *Trả lời:* Cánh tay robot UR5 thật nhận lệnh điều khiển thông qua driver `ur_rtde` chạy ở tần số lên đến 125Hz. Tín hiệu gửi từ Laptop ở tần số 30Hz sẽ được nội suy mượt mà trong mô phỏng/driver bằng bộ lọc hoặc hàm nội suy spline trước khi áp trực tiếp vào động cơ khớp, do đó robot hoàn toàn bám kịp mà không bị giật.
