import socket
import struct
import numpy as np

def main():
    # Cấu hình Socket nhận dữ liệu UDP
    UDP_IP = "0.0.0.0"  # Lắng nghe tất cả các card mạng của PC Ubuntu
    UDP_PORT = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print("\n==================================================================")
    print(" ISAAC CLIENT (WIFI RECEIVER): LẮNG NGHE GÓI TIN NHỊ PHÂN 25-DOF")
    print(f" Đang chờ nhận mảng phẳng 25 float (100 bytes) trên cổng: {UDP_PORT}")
    print(" Nhấn Ctrl+C để THOÁT.")
    print("==================================================================\n")
    
    # Định nghĩa nhãn các khớp để hiển thị rõ ràng
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", # UR5
        "idx_J1", "idx_J2", "idx_J3", "idx_J4",                                    # Index
        "mid_J1", "mid_J2", "mid_J3", "mid_J4",                                    # Middle
        "ring_J1", "ring_J2", "ring_J3", "ring_J4",                                # Ring
        "pnk_J1", "pnk_J2", "pnk_J3", "pnk_J4",                                    # Pinky
        "thb_j1", "thb_j2", "thb_j3"                                               # Thumb
    ]
    
    try:
        while True:
            # Nhận gói tin 100 bytes (25 floats * 4 bytes)
            data, addr = sock.recvfrom(1024)
            
            if len(data) == 100:
                # Giải mã gói tin nhị phân
                joint_values = struct.unpack("25f", data)
                
                # In thông tin nhận được
                print(f"[RECEIVED] Nhận gói tin từ {addr[0]}:{addr[1]}")
                print(f"  UR5 Arm Joints : {np.round(joint_values[:6], 3)}")
                print(f"  Index Finger   : {np.round(joint_values[6:10], 3)}")
                print(f"  Middle Finger  : {np.round(joint_values[10:14], 3)}")
                print(f"  Ring Finger    : {np.round(joint_values[14:18], 3)}")
                print(f"  Pinky Finger   : {np.round(joint_values[18:22], 3)}")
                print(f"  Thumb joints   : {np.round(joint_values[22:25], 3)}")
                print("-" * 60)
                
                # CHÚ THÍCH: Trong Isaac Sim, bạn sẽ tích hợp phần code sau để điều khiển robot thật/mô phỏng:
                # for i, val in enumerate(joint_values):
                #     my_robot.set_joint_position(joint_names[i], val)
            else:
                print(f"[WARN] Nhận gói tin sai kích thước ({len(data)} bytes) từ {addr}")
                
    except KeyboardInterrupt:
        print("\n[INFO] Đang đóng cổng nhận và thoát.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
