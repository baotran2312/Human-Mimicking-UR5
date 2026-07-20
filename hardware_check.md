Chào bạn, tôi đã quét toàn bộ hệ thống máy tính này và phát hiện hai dự án chính liên quan đến UR5, bàn tay khéo léo
  (Allegro Hand 16/19-DOF) và các mô hình thị giác máy tính/ước lượng pose (YOLO/RealSense).

  Dưới đây là báo cáo chi tiết về những tài nguyên đang có sẵn trên máy để phục vụ cho nhiệm vụ Tuần 1: tìm hiểu thuật toán
  training, trích xuất đặc trưng từ video để chuẩn bị cho humanoid và test trên robot UR5.
  ──────
  ### 1. Tài nguyên Mô phỏng & Điều khiển UR5 + Bàn tay ( Seqhandisaac )

  Dự án chính nằm tại thư mục Seqhandisaac. Đây là nơi chứa mô phỏng và code điều khiển UR5 cùng bàn tay khéo léo tự modify
  (cấu hình Allegro-like hand):
  • Mô hình USD mô phỏng:
      • ur5withscene.usd: File mô phỏng lắp ghép UR5 + bàn tay 19 khớp trong không gian làm việc thực tế (bàn, robot pedestal,
      v.v.).
      • ur5dex.usd: File mô phỏng robot gốc.
  • Định nghĩa Khớp & Robot ( seq3 ):
      • assets.py: Định nghĩa tên khớp ( ARM_JOINTS  và  FINGER_JOINTS  gồm 6 khớp UR5 và 19 khớp bàn tay chia theo 5 ngón: 
      thumb ,  index ,  middle ,  ring ,  pinky ).
      • expert.py: Kịch bản điều khiển (scripted expert) 3 giai đoạn cầm nắm: nhón bằng ngón cái+trỏ ( pinch_ti ), kẹp kéo
      bằng trỏ+giữa ( scissor_im ), và ôm bằng áp út+út+lòng bàn tay ( power_rp ). Hiện tại Stage 1 & 2 đã PASS hoàn toàn,
      Stage 3 đang viết dở (bị va chạm giữa các vật).
  • ROS 2 & MoveIt Workspace:
      • ros2_ur5dex: Không gian làm việc ROS 2 (Jazzy) để điều khiển MoveIt kết hợp với cầu nối Isaac Sim hoặc robot thật.
  • Kết nối Robot thật (Sim2Real):
      • sim2real_ur5.py: Kết nối robot thật thông qua thư viện  ur_rtde  và GUI hiệu chuẩn.

  ──────
  ### 2. Thuật toán Training (Học máy / Chính sách điều khiển)

  Để thực hiện huấn luyện cử chỉ như con người, máy tính đã có sẵn các cấu trúc phục vụ các thuật toán:

  • Mô hình học sâu khuếch tán (Diffusion Policy + DPPO):
      • Thư mục  seq_three_stages/third_party/dppo  chứa mã nguồn của DPPO (Diffusion Policy Policy Optimization) và tham
      khảo DP3 (3D Diffusion Policy).
      • Chi tiết kế hoạch chuyển đổi sang Diffusion Policy theo phương pháp MoDex (Lu et al. 2026) được lưu tại
      01_phan_tich_modex.md và kế hoạch thực thi tại 02_ke_hoach_thuc_thi.md.
      • Phương pháp MoDex: Sử dụng PointNet để trích xuất đặc trưng point cloud của môi trường kết hợp với MLP mã hóa trạng
      thái khớp và lịch sử cầm nắm, đầu ra là hành động khuếch tán (diffusion) điều khiển vị trí bàn tay.
  • Học tăng cường truyền thống (PPO / RSL-RL):
      • Mã nguồn cũ (mang tính tham khảo lịch sử) nằm ở  train_pinch.py  sử dụng PPO của  rsl_rl  tích hợp với Isaac Lab.

  ──────
  ### 3. Trích xuất đặc trưng tư thế (Pose Extraction) từ Video/Camera

  Mặc dù không có phần cứng Motion Capture (Mocap), trên máy tính này đang có sẵn các công cụ thị giác máy tính rất mạnh mẽ
  để phục vụ ý tưởng trích xuất từ video của bạn:

  • Mô hình YOLOv11 Pose:
      • File trọng số yolo11n-pose.pt đã được tải xuống và cấu hình thành công trong thư mục dự án  digitaltwin_warehouse .
  • Code trích xuất khung xương 3D thời gian thực:
      • realsense_person_detect.py: Đọc hình ảnh màu & depth từ camera Intel RealSense (D435), chạy model YOLOv11-pose để
      trích xuất 17 điểm khung xương người (COCO skeleton) và tính toán tọa độ 3D (X, Y, Z tính bằng mét) trong không gian
      thực.
      • isaac_people_twin.py: Sử dụng dữ liệu trích xuất từ YOLO để tạo ra một "digital twin" của con người di chuyển thời
      gian thực trong mô phỏng Isaac Sim.
  • Video Test mẫu:
      • Hai video thực tế đã quay sẵn ở thư mục gốc: 2026-07-11 15-27-06.mp4 (51 giây) và 2026-07-11 17-08-32.mp4 (67 giây)
      dùng để test thuật toán nhận diện người và robot bằng YOLOv11-pose.

  ──────
  ### 4. Môi trường Python (Conda Environments)

  Có 3 môi trường conda chính, trong đó  env_isaacsim  là môi trường hoạt động tốt nhất trên phần cứng mới (GPU Blackwell/RTX
  5060 Ti):

  1.  env_isaacsim  (Isaac Sim 5.1): Môi trường khuyên dùng từ nay. Đã được cài đặt đầy đủ các thư viện quan trọng:
      •  torch  (2.7.0+cu128) &  torchvision  (0.22.0) cho học máy.
      •  ultralytics  (8.4.75) cho YOLO Pose.
      •  opencv-python  (4.9.0) cho xử lý ảnh/video.
      •  pyrealsense2  (2.58.2) để đọc camera RealSense vật lý.
      •  ur_rtde  (1.6.3) để gửi lệnh trực tiếp xuống robot thật.
  2.  env_isaaclab  (Isaac Sim 4.5): Chỉ nên dùng cho chế độ headless (không bật GUI) do bị lỗi viewport đứng yên (bug-088)
  trên GPU mới.
  ──────
  ### Hướng tiếp cận gợi ý cho tuần này (Tuần 1):

  Với mục tiêu "tìm hiểu thuật toán training, trích xuất đặc trưng", bạn có thể tận dụng các tài nguyên trên như sau:

  1. Về trích xuất đặc trưng từ video (Feature Extraction):
      • Bạn đã có YOLOv11 Pose cài sẵn. Tuy nhiên, YOLO chỉ trích xuất các điểm trên cơ thể (khủy tay, cổ tay, vai, v.v.). Để
      bàn tay 5 ngón thực hiện cử chỉ giống người, bạn sẽ cần trích xuất khớp ngón tay chi tiết (21 điểm khớp tay).
      • Gợi ý: Cài đặt thêm thư viện  mediapipe  vào  env_isaacsim  ( pip install mediapipe ). MediaPipe Hand Tracking rất
      mạnh trong việc lấy tọa độ 3D của 21 khớp ngón tay từ camera thường / video. Sau đó, viết một script ánh xạ (mapping)
      khoảng cách/góc giữa các khớp tay người sang góc khớp tương ứng của Allegro Hand (19 khớp đã được định nghĩa trong
      assets.py).
  2. Về thuật toán training:
      • Đọc và phân tích kỹ tài liệu 01_phan_tich_modex.md. Đây chính là mô hình thuật toán mới nhất (2026) được thiết kế
      riêng cho việc huấn luyện cầm nắm tuần tự bằng bàn tay khéo léo thông qua Diffusion Policy (DP3 + DPPO).
      • Mô hình này giải thích cách encode Point Cloud (đặc trưng 3D của vật thể) và trạng thái robot để sinh action chuỗi
      thời gian (Action Chunking).
