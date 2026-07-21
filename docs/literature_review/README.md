# Tổng hợp Nghiên cứu Khoa học theo cấu trúc IMRAD (UR5 & Dexterous Hand)

Tài liệu này tổng hợp chi tiết **14 bài báo nghiên cứu khoa học** cốt lõi phục vụ dự án dưới cấu trúc **IMRAD** (Introduction, Methods, Results, Discussion) cho từng bài báo riêng biệt.

---

## 1. 3D Diffusion Policy (DP3): Generalizable Visuomotor Policy Learning via Simple 3D Representations

* **I (Introduction)**: Việc học bắt chước bằng thị giác 2D đòi hỏi hàng trăm demo cho mỗi tác vụ và kém linh hoạt khi thay đổi không gian hoặc camera. Bài báo đề xuất DP3 nhằm tối ưu hóa số lượng demo cần thiết bằng cách kết hợp biểu diễn không gian 3D với mô hình sinh hành động khuếch tán.
* **M (Methods)**: Trích xuất Point Cloud thô từ camera độ sâu, đưa qua Point Encoder (PointNet đơn giản) để nén thành vector đặc trưng 3D đại diện cho không gian. Vector này kết hợp với trạng thái khớp của robot để làm điều kiện (Condition) cho mô hình khuếch tán (Diffusion Policy) sinh ra chuỗi hành động robot.
* **R (Results)**: Trên 72 tác vụ mô phỏng, DP3 chỉ cần **10 demo** để học thành công, cải thiện 24.2% so với baseline. Trên robot thật (bao gồm bàn tay khéo léo cầm vật mềm), DP3 đạt tỉ lệ thành công **85% với chỉ 40 demo**, vượt trội về khả năng tổng quát hóa vật thể và góc camera.
* **D (Discussion)**: DP3 là thuật toán cốt lõi chúng ta nên dùng để huấn luyện UR5 + Hand. Nó chứng minh Point Cloud từ RealSense hiệu quả hơn nhiều so với ảnh màu 2D khi robot cần thao tác cầm nắm 3D.

---

## 2. 3D Flow Diffusion Policy: Visuomotor Policy Learning via Generating Flow in 3D Space

* **I (Introduction)**: Mặc dù DP3 hiệu quả, việc dự đoán trực tiếp tọa độ hành động tuyệt đối vẫn gặp khó khăn trong các tác vụ động hoặc khi vật thể di chuyển liên tục. Bài báo đề xuất dự đoán trường dòng chảy 3D (3D Flow) để robot bám đuổi vật thể mượt mà hơn.
* **M (Methods)**: Thay vì dự đoán vị trí khớp tuyệt đối, mạng neural học cách dự đoán vector dịch chuyển động học (3D Flow field) của các điểm trên robot hướng tới các điểm trên vật thể mục tiêu thông qua cơ chế Diffusion.
* **R (Results)**: Đạt độ chính xác bám đuổi cao hơn trong các tác vụ tương tác động (như gạt vật thể đang lăn, bắt vật thể). Giảm giật khớp đáng kể so với việc dự đoán vị trí tuyệt đối.
* **D (Discussion)**: Có thể áp dụng khi chúng ta nâng cấp UR5 sang các tác vụ động (thao tác trực tiếp với vật thể chuyển động trên băng chuyền hoặc khi người đưa vật cho robot).

---

## 3. 3D Hand Pose Estimation in Everyday Egocentric Images

* **I (Introduction)**: Ước lượng tư thế tay 3D từ góc nhìn thứ nhất (Egocentric - góc nhìn người đeo kính/camera gắn ngực) cực kỳ khó do tay người thường xuyên tương tác gần, bị vật thể che khuất và có độ biến dạng lớn trong sinh hoạt hàng ngày.
* **M (Methods)**: Thu thập và dán nhãn tập dữ liệu egocentric lớn trong đời thực. Sử dụng mạng CNN kết hợp thông tin phân đoạn vật thể (segmentation) và tay để dự đoán 21 tọa độ khớp tay 3D dạng heatmap.
* **R (Results)**: Thuật toán cải thiện sai số vị trí khớp trung bình từ 3.2cm xuống còn 1.8cm trên các tập dữ liệu egocentric thực tế phức tạp có tương tác với vật thể.
* **D (Discussion)**: Giúp chúng ta hiểu cách thiết lập camera RealSense góc nhìn chéo/egocentric sao cho thuật toán nhận diện khớp ngón tay ít bị lỗi nhất khi người làm demo.

---

## 4. A Dexterous Hand-Arm Teleoperation System Based on Hand Pose Estimation and Active Vision

* **I (Introduction)**: Hệ thống điều khiển từ xa (Teleoperation) cho cả cánh tay và bàn tay khéo léo thường bị giới hạn góc nhìn camera cố định. Bài báo đề xuất tích hợp thêm cơ cấu "Thị giác chủ động" (Active Vision) để camera tự động điều chỉnh theo tay người.
* **M (Methods)**: 
  * Dùng camera RGB-D trích xuất 21 keypoints tay người.
  * Giải IK cho cánh tay UR5 và ánh xạ khớp cho bàn tay khéo léo.
  * Tích hợp camera lên một cánh tay robot phụ (hoặc gắn trên cổ tay) tự động di chuyển để giữ bàn tay người luôn ở trung tâm góc nhìn camera.
* **R (Results)**: Giảm tỷ lệ mất dấu tay (tracking loss) từ 45% xuống dưới 5% trong suốt quá trình vận hành phức tạp, tăng tốc độ hoàn thành tác vụ thêm 30%.
* **D (Discussion)**: Rất thực tế cho hệ thống HIL của chúng ta. Nếu camera RealSense cố định bị che bởi UR5, ta có thể viết thuật toán điều khiển camera ảo di chuyển bám theo đầu bàn tay.

---

## 5. A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation

* **I (Introduction)**: Nhận diện khớp tay khi tay đang tương tác/cầm nắm vật thể thường bị sai lệch lớn do vật che mất ngón tay. Bài báo cải tiến thuật toán A2J (Anchor-to-Joint) bằng cơ chế Transformer để tăng độ chính xác 3D.
* **M (Methods)**: Thiết lập các điểm neo (anchors) dày đặc trên ảnh. Dùng Transformer Encoder để tính toán mối quan hệ ngữ cảnh giữa các điểm neo này, từ đó dự đoán vị trí khớp ngón tay bị che khuất dựa vào thông tin của các phần ngón tay còn nhìn thấy.
* **R (Results)**: Đạt vị trí Top 1 trên các bảng xếp hạng ước lượng tư thế tay tương tác (như HO3D và HANDS datasets) với sai số trung bình chỉ khoảng 1.1 - 1.3 cm.
* **D (Discussion)**: Đây là thuật toán thị giác rất mạnh để trích xuất đặc trưng ngón tay từ video người thật khi họ đang thực hiện thao tác cầm nắm vật thể trước camera.

---

## 6. AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

* **I (Introduction)**: Thiết kế hàm thưởng (reward function) thủ công để robot/nhân vật chuyển động tự nhiên như người là cực kỳ khó. AMP đề xuất học dáng chuyển động từ dữ liệu thô (Motion Capture) mà không cần hàm thưởng mô phỏng chi tiết.
* **M (Methods)**: Sử dụng Học tăng cường đối nghịch (Adversarial RL). Một mạng Discriminator liên tục so sánh chuyển động của robot ảo với tập dữ liệu cử chỉ người thật, sinh ra phần thưởng đối nghịch (adversarial reward) ép robot phải di chuyển giống người.
* **R (Results)**: Robot tự động học được dáng đi, chạy, nhảy và các cử chỉ tay vô cùng mượt mà, tự nhiên và bám sát vật lý mà không bị hiện tượng giật cục hay chuyển động kỳ dị.
* **D (Discussion)**: Đây là thuật toán cốt lõi cho giai đoạn sau của dự án khi chúng ta có **Unitree G1**. Chúng ta sẽ dùng AMP để huấn luyện G1 đi đứng bằng cách bắt chước video dáng đi của người.

---

## 7. Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation

* **I (Introduction)**: Nghiên cứu các hàm mục tiêu (objectives) khác nhau khi chuyển đổi tư thế tay người sang tay robot khéo léo để tìm ra phương pháp giúp robot cầm nắm vật tốt nhất.
* **M (Methods)**: So sánh 3 phương pháp retargeting: (1) Khớp-sang-Khớp (Joint-space), (2) Vị trí đầu ngón tay tuyệt đối (Fingertip-space), (3) Khoảng cách tương đối giữa các ngón (Opposition-space). Thử nghiệm trên mô phỏng tương tác vật lý.
* **R (Results)**: Phương pháp tối ưu hóa khoảng cách tương đối (Opposition-space / Vector Alignment) đạt tỷ lệ cầm nắm thành công cao nhất (vượt trội hơn hẳn so với joint-space) vì nó giữ được lực kẹp đối kháng giữa các ngón tay.
* **D (Discussion)**: Xác nhận cơ sở toán học cho thuật toán Retargeting của chúng ta: Chúng ta phải code ánh xạ dựa trên khoảng cách giữa các đầu ngón tay (Fingertip distance) thay vì ép các khớp ngón tay robot xoay theo đúng góc khớp người.

---

## 8. AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System

* **I (Introduction)**: Các hệ thống điều khiển robot bằng tay người trước đây thường được code cứng cho một loại robot cụ thể. AnyTeleop đề xuất một hệ thống tổng quát, cắm-và-chạy (plug-and-play) cho mọi cấu hình cánh tay + bàn tay khéo léo.
* **M (Methods)**:
  * Trích xuất keypoints 3D từ camera RGB thường.
  * Sử dụng một bộ giải tối ưu động học ngược (Unified IK Solver) dựa trên khoảng cách tương đối để tự động ánh xạ sang cấu hình robot đích bất kỳ (đọc từ file URDF của robot đó).
* **R (Results)**: Chạy thử nghiệm thành công trên nhiều cấu hình thực tế như cánh tay Franka/UR5 + bàn tay Allegro/Robotiq chỉ bằng 1 camera đơn, đạt hiệu năng điều khiển thời gian thực mượt mà.
* **D (Discussion)**: Đây là bài báo sát nhất với mục tiêu HIL của chúng ta. Chúng ta sẽ xây dựng hệ thống bám sát theo kiến trúc module của AnyTeleop (đọc file `ur5dex.urdf` để tự động cấu hình bộ giải IK).

---

## 9. Benchmarks and Challenges in Pose Estimation for Egocentric Hand Interactions with Objects

* **I (Introduction)**: Đánh giá hiệu năng và thiết lập benchmark chuẩn hóa cho các thuật toán nhận diện tư thế tay khi có tương tác trực tiếp với vật thể (Hand-Object Interaction - HOI).
* **M (Methods)**: So sánh các backbone thị giác (ResNet, ViT, PointNet) trên các tập dữ liệu lớn như Ego4D, Assembly101. Đánh giá sai số dưới các điều kiện ánh sáng, góc camera và kích thước vật thể khác nhau.
* **R (Results)**: Chỉ ra rằng các thuật toán thị giác 2D bị giảm 40% độ chính xác khi tay cầm vật thể nhỏ hoặc khi có va chạm mạnh. Việc bổ sung thông tin hình học 3D (Depth/Point Cloud) giúp ổn định kết quả.
* **D (Discussion)**: Củng cố lý do tại sao chúng ta nên sử dụng camera depth RealSense thay vì webcam thường cho hệ thống HIL để tránh mất dấu ngón tay khi tay robot chạm vào vật thể.

---

## 10. Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping

* **I (Introduction)**: Khi chuyển từ mô phỏng sang thực tế (Sim-to-Real), robot thường làm rơi vật do thiếu phản hồi lực chính xác. Bài báo nghiên cứu cách deploy không cần train lại (Zero-Shot) bằng cách tích hợp mô phỏng lực kẹp.
* **M (Methods)**:
  * Huấn luyện chính sách điều khiển dựa trên cảm biến lực ảo trong simulator.
  * Áp dụng Domain Randomization cực hạn lên các thuộc tính vật lý (ma sát, khối lượng vật, độ cứng).
  * Sử dụng thuật toán điều khiển vị trí/lực lai (Hybrid Position/Force Control) khi deploy thật.
* **R (Results)**: Robot bàn tay khéo léo thật thực hiện thành công các tác vụ cầm nắm vật thể mềm, dễ vỡ và trơn trượt ngay lần thử đầu tiên với tỷ lệ thành công >80%.
* **D (Discussion)**: Giúp định hình cách thiết lập vật lý trong Isaac Sim của chúng ta (Tuần 3): Chúng ta phải cài đặt độ ma sát ngẫu nhiên và khối lượng vật ngẫu nhiên để sau này nạp code xuống UR5 thật không bị trượt tay.

---

## 11. DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation

* **I (Introduction)**: Việc huấn luyện cầm nắm nhiều vật thể khác nhau trong mô phỏng mất rất nhiều thời gian. DexSim2Real đề xuất sử dụng Foundation Model (mô hình nền tảng như GPT-4V/VILA) để định dẫn hướng và tạo kỹ năng cầm nắm tổng quát.
* **M (Methods)**: Sử dụng Foundation Model để phân tích hình ảnh vật thể, tự động xác định các điểm đặt ngón tay (grasping points) hợp lý, sau đó sinh code/quỹ đạo mô phỏng tự động để robot thực hiện trong Isaac Sim, giảm thiểu thời gian thử sai.
* **R (Results)**: Rút ngắn thời gian thiết lập môi trường và huấn luyện đi 10 lần. Robot có khả năng cầm nắm các vật thể lạ hoàn toàn chưa từng thấy trong tập train với tỷ lệ thành công cao.
* **D (Discussion)**: Gợi ý cho chúng ta hướng phát triển dài hạn: Khi hệ thống HIL đã chạy ổn định, chúng ta có thể dùng AI để tự động sinh ra các demo cầm nắm thay vì phải tự tay làm demo trước camera liên tục.

---

## 12. Deformer: Dynamic Fusion Transformer for Robust Hand Pose Estimation

* **I (Introduction)**: Ước lượng tư thế tay 3D từ video gặp thách thức lớn khi bị che khuất hoặc mờ chuyển động (motion blur). Các phương pháp temporal self-attention thông thường thất bại khi đặc trưng từng khung hình bị biến dạng nặng. Bài báo đề xuất mô hình Deformer để giải quyết vấn đề này.
* **M (Methods)**: Tích hợp một Module Dynamic Fusion dự đoán sự biến dạng (deformation) của bàn tay và dịch chuyển (warp) các lưới dự đoán bàn tay từ các khung hình lân cận rõ ràng sang hỗ trợ khung hình hiện tại. Đồng thời giới thiệu hàm loss **maxMSE** để tự động tăng trọng số tập trung vào các điểm khớp khó nhận diện (như đầu ngón tay).
* **R (Results)**: Đạt độ chính xác vượt trội hơn 10% so với SOTA trên tập DexYCB và HO3D, đồng thời tăng 14% khả năng chống chịu các tình huống che khuất.
* **D (Discussion)**: Thuật toán này rất hữu ích cho hệ thống HIL của chúng ta. Khi người vận hành tương tác với vật thể làm che khuất ngón tay, Deformer giúp tái tạo tư thế tay robot mượt mà bằng cách bù đắp thông tin từ các frame trước/sau.

---

## 13. DeltaDorsal: Enhancing Hand Pose Estimation with Dorsal Features in Egocentric Views

* **I (Introduction)**: Góc nhìn thứ nhất (Egocentric) trên các thiết bị kính XR thường xuyên làm ngón tay bị che khuất bởi chính bàn tay hoặc vật thể cầm nắm. Bài báo đề xuất ước lượng tư thế tay chỉ dựa trên biến dạng da mu bàn tay (dorsum) mà không cần nhìn trực tiếp ngón tay.
* **M (Methods)**: Thiết lập bộ mã hóa delta hai luồng (dual-stream delta encoder). Mạng so sánh các đặc trưng thị giác mu bàn tay hiện tại với tư thế baseline thả lỏng của người dùng để phân tách tín hiệu biến dạng da, từ đó suy luận ra góc khớp ngón tay 3D bên dưới.
* **R (Results)**: Giảm 18% sai số góc khớp (MPJAE) trong các tình huống ngón tay bị che khuất nặng (>= 50%) so với mô hình toàn bàn tay SOTA như HaMeR.
* **D (Discussion)**: Cực kỳ hữu dụng cho thiết lập camera đơn của chúng ta. Khi camera RealSense ở vị trí chéo cao nhìn xuống lòng bàn tay (làm khuất ngón tay bên dưới), ta có thể áp dụng ý tưởng của DeltaDorsal để suy luận góc ngón tay từ chuyển động mu bàn tay.

---

## 14. DexHiL: A Human-in-the-Loop Framework for Vision-Language-Action Model Post-Training in Dexterous Manipulation

* **I (Introduction)**: Triển khai các mô hình VLA (Vision-Language-Action) trên bàn tay khéo léo gặp lỗi dịch chuyển phân phối (covariate shift) và tích lũy sai số. Học bắt chước offline đơn thuần (SFT) dễ bị bão hòa hiệu năng và sập khi gặp trạng thái lạ (OOD).
* **M (Methods)**: Đề xuất bộ khung học trực tuyến có con người can thiệp (Human-in-the-Loop - HiL) thông qua vòng lặp DAgger. Thiết lập cơ chế can thiệp thời gian thực (nhấp chuột/phím để lấy quyền kiểm soát robot) và thuật toán lấy mẫu trọng số ưu tiên cho các phân đoạn dữ liệu sửa sai (intervention-aware reweighting) trong quá trình train.
* **R (Results)**: Đạt hiệu năng vượt trội hơn 25% tỷ lệ thành công trung bình so với các mô hình chỉ tinh chỉnh offline trên robot thật.
* **D (Discussion)**: Trực tiếp chứng minh tính thực tiễn của quy trình HIL chúng ta đang xây dựng. Chúng ta sẽ dùng chính hệ thống webcam teleop thời gian thực để người dùng có thể can thiệp sửa lỗi cho policy trong Isaac Sim, thu thập dữ liệu hiệu quả để tối ưu mô hình.
