import cv2
import time
import math
import numpy as np
import os
import sys
import argparse
import socket
import struct

# Thêm thư mục hiện tại vào path để import các module local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] Thư viện 'pyrealsense2' chưa được cài đặt.")
    exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] Thư viện 'mediapipe' chưa được cài đặt.")
    exit(1)

try:
    from retargeting_solver import RetargetingSolver
except ImportError as e:
    print(f"[ERROR] Không thể import RetargetingSolver: {e}")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="AnyTeleop HIL Local WiFi Sender with RealSense (Binary 25-DoF)")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the remote PC (Ubuntu)")
    parser.add_argument("--port", type=int, default=5005, help="UDP target port")
    parser.add_argument("--width", type=int, default=640, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=480, help="Camera resolution height")
    parser.add_argument("--fps", type=int, default=30, help="Camera frames per second")
    args = parser.parse_args()

    # Khởi tạo Socket UDP ở chế độ gửi binary
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Khởi tạo bộ giải động học ngược (IK)
    urdf_path = "config/ur5dex.urdf"
    try:
        solver = RetargetingSolver(urdf_path)
    except Exception as e:
        print(f"[ERROR] Lỗi khởi tạo RetargetingSolver: {e}")
        return

    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Khởi tạo Intel RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Không thể mở camera Intel RealSense: {e}")
        return

    # Lấy depth scale từ cảm biến
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Khởi tạo bộ căn chỉnh depth sang color
    align = rs.align(rs.stream.color)

    print("\n==================================================================")
    print(" SENDER REALSENSE (WIFI LOCAL): TRUYỀN DỮ LIỆU BINARY 25-DOF QUA MẠNG")
    print(f" Đang truyền gói tin binary (100 bytes) đến -> {args.ip}:{args.port}")
    print(" Cử động tay trước camera RealSense để thấy góc khớp tính toán liên tục.")
    print(" Nhấn phím 'q' để THOÁT.")
    print("==================================================================\n")

    prev_time = time.time()
    
    try:
        while True:
            # Chờ lấy frames từ RealSense
            frames = pipeline.wait_for_frames()
            # Căn chỉnh frames
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Chuyển đổi sang numpy array
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Xử lý ảnh bằng MediaPipe
            start_process = time.time()
            results = hands.process(img_rgb)
            latency = (time.time() - start_process) * 1000

            # Tính toán FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Nếu phát hiện tay người
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 1. Giải góc khớp ngón tay (19 DoF)
                    finger_angles = solver.map_finger_joints(hand_landmarks.landmark)
                    
                    # 2. Giải IK cho cánh tay UR5 (6 DoF) với mapping trục chuẩn xác sử dụng depth thật
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    
                    # Lấy pixel tọa độ của wrist trong ảnh depth (đã aligned)
                    # Do frame bị flip nên tọa độ x phải lấy (1.0 - wrist.x) để ánh xạ vào ảnh gốc chưa flip
                    # (Nhưng do depth_image cũng được lấy trực tiếp từ camera, nếu ta flip color frame thì ta
                    # cũng nên flip depth frame hoặc điều chỉnh tọa độ pixel đọc depth).
                    # Để đơn giản và chính xác, ta flip cả depth_image tương tự color frame:
                    # depth_image = cv2.flip(depth_image, 1)
                    depth_image_flipped = cv2.flip(depth_image, 1)
                    
                    px = int(wrist.x * w)
                    py = int(wrist.y * h)
                    px = max(0, min(w - 1, px))
                    py = max(0, min(h - 1, py))
                    
                    # Lấy giá trị depth trung vị trong vùng 5x5 xung quanh wrist
                    depth_patch = depth_image_flipped[max(0, py-2):min(h, py+3), max(0, px-2):min(w, px+3)].astype(np.float32) * depth_scale
                    valid_depths = depth_patch[depth_patch > 0.1]
                    
                    if len(valid_depths) > 0:
                        real_depth = np.median(valid_depths)
                    else:
                        # Fallback nếu mất depth (sử dụng ước lượng 2D từ size bàn tay)
                        middle_mcp = hand_landmarks.landmark[9]
                        hand_size_2d = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
                        hand_size_2d = max(0.05, min(0.3, hand_size_2d))
                        real_depth = 0.025 / hand_size_2d  # Hệ số tỉ lệ thực nghiệm
                        
                    # Giới hạn khoảng cách depth hợp lệ [0.4, 1.0] m
                    real_depth = max(0.4, min(1.0, real_depth))
                    
                    # Ánh xạ độ sâu thực tế sang robot Y (tiến-lùi)
                    # Càng gần camera (depth nhỏ) -> robot đi về phía trước (Y tăng)
                    d_min, d_max = 0.4, 1.0
                    y_min, y_max = 0.3, 0.6
                    depth_norm = (real_depth - d_min) / (d_max - d_min)
                    target_y = y_max - depth_norm * (y_max - y_min)
                    
                    # Ánh xạ tọa độ ngang (camera X) sang robot X (trái-phải)
                    target_x = (wrist.x - 0.5) * 0.8
                    
                    # Ánh xạ tọa độ dọc (camera Y) sang robot Z (lên-xuống)
                    z_min, z_max = 0.8, 1.1
                    target_z = z_min + (1.0 - wrist.y) * (z_max - z_min)
                    
                    # Giải hướng bàn tay người làm target_quat cho robot
                    target_quat = solver.solve_hand_orientation(hand_landmarks.landmark)
                    
                    arm_angles = solver.solve_arm_ik([target_x, target_y, target_z], target_quat)
                    
                    # 3. Gom 25 góc khớp thành mảng phẳng theo thứ tự cố định
                    joint_vector = []
                    
                    # UR5 (6 DoF)
                    joint_vector.extend(arm_angles)
                    
                    # Index finger (4 DoF)
                    joint_vector.extend([
                        finger_angles.get("index_J1", 0.0),
                        finger_angles.get("index_J2", 0.0),
                        finger_angles.get("index_J3", 0.0),
                        finger_angles.get("index_J4", 0.0)
                    ])
                    # Middle finger (4 DoF)
                    joint_vector.extend([
                        finger_angles.get("middle_J1", 0.0),
                        finger_angles.get("middle_J2", 0.0),
                        finger_angles.get("middle_J3", 0.0),
                        finger_angles.get("middle_J4", 0.0)
                    ])
                    # Ring finger (4 DoF)
                    joint_vector.extend([
                        finger_angles.get("ring_J1", 0.0),
                        finger_angles.get("ring_J2", 0.0),
                        finger_angles.get("ring_J3", 0.0),
                        finger_angles.get("ring_J4", 0.0)
                    ])
                    # Pinky finger (4 DoF)
                    joint_vector.extend([
                        finger_angles.get("pinky_J1", 0.0),
                        finger_angles.get("pinky_J2", 0.0),
                        finger_angles.get("pinky_J3", 0.0),
                        finger_angles.get("pinky_J4", 0.0)
                    ])
                    # Thumb (3 DoF)
                    joint_vector.extend([
                        finger_angles.get("thumb_j1", 0.0),
                        finger_angles.get("thumb_j2", 0.0),
                        finger_angles.get("thumb_j3", 0.0)
                    ])
                    
                    # 4. Đóng gói thành nhị phân (25 số float = 100 bytes)
                    packet = struct.pack("25f", *joint_vector)
                    
                    # Gửi qua UDP
                    try:
                        sock.sendto(packet, (args.ip, args.port))
                    except Exception as e:
                        print(f"[ERROR] Không thể gửi gói tin: {e}")

                    # In kiểm tra nhanh trên console
                    print(f"FPS: {fps:.1f} | Latency: {latency:.1f}ms")
                    print(f"  UR5 Joint 1-3: {np.round(arm_angles[:3], 3)}")
                    print(f"  Finger J2 Check: Index={joint_vector[7]:.3f} | Middle={joint_vector[11]:.3f}")
                    print("-" * 60)

                    # Vẽ thông tin lên camera frame
                    cv2.putText(frame, f"TCP: [{target_x:.2f}, {target_y:.2f}, {target_z:.2f}]", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, "Sending 25-DoF Binary Packet...", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("AnyTeleop - Local WiFi Sender (RealSense Binary)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        solver.disconnect()
        sock.close()
        print("[INFO] Đã đóng cổng gửi.")

if __name__ == "__main__":
    main()
