import argparse
from isaaclab.app import AppLauncher

# Cấu hình Argument Parser cho Isaac Sim
parser = argparse.ArgumentParser(description="Isaac Sim UDP Client for Teleoperation")
parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP bind IP address")
parser.add_argument("--port", type=int, default=5005, help="UDP bind port")
parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed factor")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Khởi chạy simulation app trước khi import các thư viện Isaac khác
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
import torch
import time
import socket
import json
import numpy as np

# Thêm thư mục mã nguồn seq3 từ Seqhandisaac vào sys.path
seq3_path = "/home/nhglab/Tri/Seqhandisaac/seq_three_stages/source/seq3"
if seq3_path not in sys.path:
    sys.path.insert(0, seq3_path)

try:
    from seq3.assets import ARM_JOINTS, FINGER_JOINTS, HAND_JOINTS, CAMERA_POS, camera_target
    from seq3.scene import build_scene
except ImportError as e:
    print(f"[ERROR] Không thể import các module seq3 từ {seq3_path}: {e}")
    simulation_app.close()
    sys.exit(1)

def main():
    device = args_cli.device
    
    # Khởi dựng môi trường giả lập (Sim, Robot Articulation, Objects)
    print("[INFO] Đang khởi dựng scene và robot trong Isaac Sim...")
    sim, robot, objects = build_scene(device=device, with_objects=True)
    
    if not args_cli.headless:
        # Cấu hình góc nhìn camera trùng góc nhìn camera RealSense
        sim.set_camera_view(eye=CAMERA_POS, target=camera_target(1.2))

    physics_dt = sim.get_physics_dt()
    realtime = not args_cli.headless

    # Tìm chỉ số joints trên mô hình robot
    arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
    hand_ids = {}
    for finger_name, joint_names in FINGER_JOINTS.items():
        ids, _ = robot.find_joints(joint_names, preserve_order=True)
        hand_ids[finger_name] = ids

    # Đưa robot về vị trí HOME ban đầu và đợi ổn định
    print("[INFO] Đang đưa robot về tư thế HOME ban đầu...")
    q = robot.data.default_joint_pos.clone()
    HOME_ARM = [1.5708, -1.5708, 1.5708, 0.0, 1.5708, 0.0]
    q[:, arm_ids] = torch.tensor(HOME_ARM, device=device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    
    for _ in range(120):
        sim.step()
        robot.update(physics_dt)
        for o in objects.values():
            o.update(physics_dt)

    # Khởi tạo UDP Socket ở chế độ non-blocking để đọc gói tin mới nhất mà không gây trễ sim
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args_cli.ip, args_cli.port))
        sock.setblocking(False)
        print("\n==================================================================")
        print(f" UDP CLIENT ĐANG LẮNG NGHE TRÊN CỔNG: {args_cli.port}")
        print(" Hãy chạy realsense_stream.py hoặc camera_stream.py để truyền dữ liệu.")
        print(" Nhấn Ctrl+C hoặc đóng cửa sổ mô phỏng để THOÁT.")
        print("==================================================================\n")
    except Exception as e:
        print(f"[ERROR] Không thể khởi tạo UDP socket: {e}")
        simulation_app.close()
        sys.exit(1)

    try:
        while simulation_app.is_running():
            t0 = time.time()

            # Đọc toàn bộ gói tin UDP hiện có trong hàng đợi để lấy gói mới nhất
            latest_payload = None
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    latest_payload = json.loads(data.decode("utf-8"))
                except BlockingIOError:
                    break
                except Exception as e:
                    print(f"[WARN] Lỗi khi nhận gói tin UDP: {e}")
                    break

            # Cập nhật góc khớp mục tiêu nếu nhận được gói tin mới
            if latest_payload is not None:
                # Cập nhật cánh tay UR5
                arm_angles = latest_payload.get("arm", [])
                if len(arm_angles) == 6:
                    q[:, arm_ids] = torch.tensor(arm_angles, device=device)

                # Cập nhật bàn tay 19-DoF
                hand_data = latest_payload.get("hand", {})
                for finger_name, ids in hand_ids.items():
                    angles = hand_data.get(finger_name, [])
                    if len(angles) == len(ids):
                       q[:, ids] = torch.tensor(angles, device=device)
                       
                # Hiển thị log kiểm tra định kỳ (ở tốc độ hiển thị vừa phải)
                if int(time.time() * 2) % 4 == 0:
                    print(f"[RECV] TCP/IK updated. Arm angles: {np.round(arm_angles, 2)}")

            # Gửi target pos mới xuống simulator và thực hiện bước sim tiếp theo
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            sim.step()
            
            # Cập nhật dữ liệu trạng thái robot và vật thể
            robot.update(physics_dt)
            for o in objects.values():
                o.update(physics_dt)

            # Đồng bộ hóa real-time với GUI
            if realtime:
                remaining = physics_dt / args_cli.speed - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[INFO] Đang dừng Isaac Sim UDP Client.")
    finally:
        sock.close()
        simulation_app.close()
        print("[INFO] Đã đóng cổng UDP và tắt mô phỏng.")

if __name__ == "__main__":
    main()
