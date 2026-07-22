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
    parser = argparse.ArgumentParser(description="AnyTeleop HIL Local WiFi Sender (Binary 25-DoF)")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the remote PC (Ubuntu)")
    parser.add_argument("--port", type=int, default=5005, help="UDP target port")
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

    # Mở Webcam trên laptop
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam laptop.")
        return

    print("\n==================================================================")
    print(" SENDER (WIFI LOCAL): TRUYỀN DỮ LIỆU BINARY 25-DOF QUA MẠNG")
    print(f" Đang truyền gói tin binary (100 bytes) đến -> {args.ip}:{args.port}")
    print(" Cử động tay trước camera để thấy góc khớp tính toán liên tục.")
    print(" Nhấn phím 'q' để THOÁT.")
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
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 1. Giải góc khớp ngón tay (19 DoF)
                finger_angles = solver.map_finger_joints(hand_landmarks.landmark)
                
                # 2. Giải IK cho cánh tay UR5 (6 DoF)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                target_x = (wrist.x - 0.5) * 0.8
                target_y = 0.5 + (1.0 - wrist.y - 0.5) * 0.4
                target_z = 0.9 - (wrist.z * 0.5)
                target_quat = [0, 0, 0, 1] 
                
                arm_angles = solver.solve_arm_ik([target_x, target_y, target_z], target_quat)
                
                # 3. Gom 25 góc khớp thành mảng phẳng theo thứ tự cố định
                # 6 Khớp Arm + 16 Khớp Ngón (4 ngón x J1-J4) + 3 Khớp Ngón Cái (j1-j3)
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
        cv2.imshow("AnyTeleop - Local WiFi Sender (Binary)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    solver.disconnect()
    sock.close()
    print("[INFO] Đã đóng cổng gửi.")

if __name__ == "__main__":
    main()
