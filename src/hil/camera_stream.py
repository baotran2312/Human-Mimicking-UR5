import cv2
import time
import math
import numpy as np
import os
import sys
import argparse

# Thêm thư mục hiện tại vào path để import các module local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

try:
    from udp_bridge import UDPBridge
except ImportError as e:
    print(f"[ERROR] Không thể import UDPBridge: {e}")
    exit(1)

def calculate_distance(p1, p2):
    """Tính khoảng cách Euclid giữa 2 điểm 3D (x, y, z)"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    # Sử dụng argparse để tùy chỉnh cấu hình mạng qua terminal
    parser = argparse.ArgumentParser(description="AnyTeleop HIL Camera Stream and UDP Bridge")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target IP address for UDP receiver (PC)")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP port")
    args = parser.parse_args()

    # Khởi tạo bộ giải động học ngược (IK)
    urdf_path = "config/ur5dex.urdf"
    try:
        solver = RetargetingSolver(urdf_path)
    except Exception as e:
        print(f"[ERROR] Lỗi khởi tạo RetargetingSolver: {e}")
        return

    # Khởi tạo UDP Bridge
    udp = UDPBridge(ip=args.ip, port=args.port)

    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Mở Webcam mặc định
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam laptop.")
        return

    print("\n==================================================================")
    print(" BẮT ĐẦU ĐIỀU KHIỂN & GIẢI ĐỘNG HỌC NGƯỢC THỜI GIAN THỰC (HIL)")
    print(f" Đang truyền gói tin UDP đến -> {args.ip}:{args.port}")
    print(" Cử động tay trước camera để thấy góc khớp tính toán liên tục.")
    print(" Nhấn phím 'q' trong cửa sổ hiển thị để THOÁT.")
    print("==================================================================\n")

    prev_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

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
                # Vẽ khung xương tay
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- 1. GIẢI ÁNH XẠ KHỚP NGÓN TAY (FINGER RETARGETING) ---
                finger_angles = solver.map_finger_joints(hand_landmarks.landmark)
                
                # --- 2. GIẢI IK CHO CÁNH TAY UR5 (ARM IK) ---
                # Lấy vị trí cổ tay (WRIST) trong ảnh làm đích điều khiển (Target TCP)
                # Chuẩn hóa tọa độ x, y về dải làm việc của UR5 trong sim (mét)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
                # Ánh xạ tọa độ ảnh [0, 1] sang dải làm việc thực tế của UR5:
                # X: [-0.4, 0.4] m | Y: [0.3, 0.7] m | Z: [0.8, 1.2] m
                target_x = (wrist.x - 0.5) * 0.8
                target_y = 0.5 + (1.0 - wrist.y - 0.5) * 0.4
                target_z = 0.9 - (wrist.z * 0.5)  # wrist.z là độ sâu tương đối
                
                # Hướng cổ tay mặc định (hướng thẳng xuống mặt bàn)
                target_quat = [0, 0, 0, 1] 
                
                # Giải động học ngược cánh tay UR5
                arm_angles = solver.solve_arm_ik([target_x, target_y, target_z], target_quat)
                
                # --- 3. TRUYỀN GÓC KHỚP QUA UDP BRIDGE ---
                udp.send_joint_angles(arm_angles, finger_angles)

                # --- 4. IN KẾT QUẢ LIÊN TỤC RA TERMINAL ---
                print(f"FPS: {fps:.1f} | Latency: {latency:.1f}ms")
                print(f"  Target TCP: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}")
                print(f"  UR5 Joints: {np.round(arm_angles, 3)}")
                print(f"  Index J2-J4: {finger_angles['index_J2']:.3f} rad | Thumb j1: {finger_angles['thumb_j1']:.3f} rad")
                print("-" * 60)

                # Vẽ thông tin lên màn hình OpenCV
                cv2.putText(frame, f"TCP Target: [{target_x:.2f}, {target_y:.2f}, {target_z:.2f}]", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"UR5 Joints: {np.round(arm_angles, 2)}", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Hiển thị FPS và Latency
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Latency: {latency:.1f} ms", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("AnyTeleop - HIL Step 2: Real-time Retargeting & IK", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    solver.disconnect()
    udp.close()
    print("[INFO] Đã ngắt kết nối và đóng cổng UDP.")

if __name__ == "__main__":
    main()
