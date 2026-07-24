import cv2
import time
import socket
import struct
import sys
import os
import argparse
import math
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] Thư viện 'mediapipe' chưa được cài đặt.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Laptop Webcam Raw Target Sender for HIL")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the remote PC (Ubuntu)")
    parser.add_argument("--port", type=int, default=5005, help="UDP target port")
    args = parser.parse_args()

    # Khởi tạo Socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Khởi tạo MediaPipe Hands và Drawing Utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam laptop.")
        return

    print("\n==================================================================")
    print(" SENDER (RAW POSE): PHÁT TỌA ĐỘ TAY GỐC TỪ WEBCAM LAPTOP")
    print(f" Đang truyền tọa độ thô đến PC -> {args.ip}:{args.port}")
    print(" Nhấn phím 'q' để THOÁT.")
    print("==================================================================\n")

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ các điểm khớp tay lên màn hình
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Lấy tọa độ cổ tay (Wrist) đại diện cho x_d
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
                # Ánh xạ tọa độ không gian làm việc (Workspace mapping)
                target_x = (wrist.x - 0.5) * 0.8
                target_y = 0.5 + (1.0 - wrist.y - 0.5) * 0.4
                target_z = 0.9 - (wrist.z * 0.5)
                
                # Tính toán hướng 3D thực tế của bàn tay từ MediaPipe landmarks
                p0 = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
                p5 = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
                p9 = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
                p17 = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
                
                v_palm = p9 - p0
                v_len = np.linalg.norm(v_palm)
                y_axis = v_palm / v_len if v_len > 1e-6 else np.array([0.0, 1.0, 0.0])
                
                v_lat = p17 - p5
                z_axis = np.cross(y_axis, v_lat)
                z_len = np.linalg.norm(z_axis)
                z_axis = z_axis / z_len if z_len > 1e-6 else np.array([0.0, 0.0, 1.0])
                x_axis = np.cross(y_axis, z_axis)
                
                R = np.column_stack((x_axis, -y_axis, -z_axis))
                tr = np.trace(R)
                if tr > 0:
                    S = math.sqrt(tr + 1.0) * 2
                    qw = 0.25 * S
                    qx = (R[2, 1] - R[1, 2]) / S
                    qy = (R[0, 2] - R[2, 0]) / S
                    qz = (R[1, 0] - R[0, 1]) / S
                else:
                    qx, qy, qz, qw = 1.0, 0.0, 0.0, 0.0
                
                # Thu thập độ gập (flexion) của 5 ngón tay (mỗi ngón 1 float từ 0.0 đến 1.0)
                # Dùng làm tọa độ khớp ngón tay thô
                fingers_flex = []
                fingers_mcp_tip = [(5, 8), (9, 12), (13, 16), (17, 20)]
                for mcp, tip in fingers_mcp_tip:
                    mcp_pt = hand_landmarks.landmark[mcp]
                    tip_pt = hand_landmarks.landmark[tip]
                    dist = math.sqrt((tip_pt.x - mcp_pt.x)**2 + (tip_pt.y - mcp_pt.y)**2)
                    flex = clip(1.0 - (dist / 0.15), 0.0, 1.0)
                    fingers_flex.append(flex)
                
                # Thêm ngón cái (Pinch distance)
                thumb_tip = hand_landmarks.landmark[4]
                index_mcp = hand_landmarks.landmark[5]
                t_dist = math.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
                thumb_flex = clip(1.0 - (t_dist / 0.15), 0.0, 1.0)
                fingers_flex.append(thumb_flex)

                # Gói tin gồm: timestamp (1f) + position (3f) + orientation (4f) + finger flex (5f)
                # Tổng cộng: 1 + 3 + 4 + 5 = 13 floats (52 bytes)
                timestamp = time.time() - start_time
                payload = [timestamp, target_x, target_y, target_z, qx, qy, qz, qw] + fingers_flex
                
                packet = struct.pack("13f", *payload)
                sock.sendto(packet, (args.ip, args.port))

                print(f"[SEND] Pos: [{target_x:.2f}, {target_y:.2f}, {target_z:.2f}] | Flex: {[round(f, 2) for f in fingers_flex]}")

        cv2.imshow("Laptop Raw Pose Sender", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

def clip(val, vmin, vmax):
    return max(vmin, min(val, vmax))

if __name__ == "__main__":
    main()
