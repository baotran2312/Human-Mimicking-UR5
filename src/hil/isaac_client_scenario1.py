import argparse
import os
import sys

# Thêm đường dẫn tìm kiếm package pybullet phòng trường hợp môi trường chưa load
for site_pkg in [
    "/home/nhglab/anaconda3/envs/env_isaacsim/lib/python3.11/site-packages",
    "/home/nhglab/anaconda3/envs/hil_env/lib/python3.10/site-packages",
    "/home/nhglab/anaconda3/lib/python3.13/site-packages",
]:
    if os.path.exists(site_pkg) and site_pkg not in sys.path:
        sys.path.append(site_pkg)

# Khởi chạy AppLauncher của Isaac Sim trước khi import PyTorch & Isaac Sim Modules
parser = argparse.ArgumentParser(description="Isaac Sim HIL Scenario 1 Client")
parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP bind IP address")
parser.add_argument("--port", type=int, default=5005, help="UDP bind port")
parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed factor")
try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    HAS_ISAAC_SIM = True
except (ImportError, Exception):
    args_cli = parser.parse_args()
    simulation_app = None
    HAS_ISAAC_SIM = False

import socket
import struct
import time
import numpy as np
from collections import deque

# Import modules từ hệ thống HIL & Seqhandisaac
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
seq3_path = "/home/nhglab/Tri/Seqhandisaac/seq_three_stages/source/seq3"
if os.path.exists(seq3_path) and seq3_path not in sys.path:
    sys.path.insert(0, seq3_path)

try:
    import torch
    from seq3.assets import ARM_JOINTS, FINGER_JOINTS, CAMERA_POS, camera_target
    from seq3.scene import build_scene
    from retargeting_solver import RetargetingSolver
    HAS_ROBOT_SIM = HAS_ISAAC_SIM
except Exception as e:
    HAS_ROBOT_SIM = False
    print(f"[WARN] Đang chạy chế độ đo đạc số liệu thuần (không có giao diện Isaac Sim 3D): {e}")

class HILSimulationScenario1:
    def __init__(self):
        # Mạng nhận UDP Non-blocking
        self.UDP_IP = args_cli.ip if args_cli else "0.0.0.0"
        self.UDP_PORT = args_cli.port if args_cli else 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((self.UDP_IP, self.UDP_PORT))
            self.sock.setblocking(False)
        except Exception as e:
            print(f"[ERROR] Lỗi bind UDP port: {e}")
        
        # Hộp lưu trữ gói tin thô nhận được từ Laptop
        self.packet_queue = deque()
        self.packet_history = deque(maxlen=2000)
        self.running = True
        
        # Các thông số trễ mạng mô phỏng (Kịch bản 1)
        self.tau_mean = 0.150  # 150ms trễ
        self.jitter_amp = 0.050  # 50ms jitter
        
        # Trạng thái của Robot ảo trong mô phỏng (Arm TCP position)
        self.x_curr_std = np.array([0.18, 0.40, 0.90])
        self.x_curr_prop = np.array([0.18, 0.40, 0.90])
        self.prev_pos_filt = None
        
        # Gains của bộ điều khiển
        self.K_p = 5.0 * np.eye(3)
        self.K_d = 2.0 * np.eye(3)
        self.T_f = 0.25  # Hằng số lọc LPF
        
        # Trạng thái trễ lọc
        self.tau_filt = self.tau_mean
        self.start_time = None
        
        # Các chỉ số tích lũy đo lường
        self.err_std_history = []
        self.err_prop_history = []
        self.q_dot_std_history = []
        self.q_dot_prop_history = []
        
        # Danh sách lưu trữ dữ liệu để xuất ra CSV
        self.csv_records = []
        from datetime import datetime
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Khởi tạo mô phỏng Isaac Sim 3D nếu có môi trường
        self.has_sim = HAS_ROBOT_SIM
        if self.has_sim:
            print("[INFO] Đang dựng Robot UR5 + Bàn tay 19-DoF trong Isaac Sim...")
            self.device = args_cli.device
            self.sim, self.robot, self.objects = build_scene(device=self.device, with_objects=True)
            
            if not args_cli.headless:
                self.sim.set_camera_view(eye=CAMERA_POS, target=camera_target(1.2))

            self.physics_dt = self.sim.get_physics_dt()
            self.arm_ids, _ = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
            self.hand_ids = {}
            for finger_name, joint_names in FINGER_JOINTS.items():
                ids, _ = self.robot.find_joints(joint_names, preserve_order=True)
                self.hand_ids[finger_name] = ids

            urdf_path = "config/ur5dex.urdf"
            if not os.path.exists(urdf_path):
                urdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "ur5dex.urdf")
            self.solver = RetargetingSolver(urdf_path)

            self.q = self.robot.data.default_joint_pos.clone()
            HOME_ARM = [1.5708, -1.5708, 1.5708, 0.0, 1.5708, 0.0]
            self.q[:, self.arm_ids] = torch.tensor(HOME_ARM, device=self.device)
            self.robot.write_joint_state_to_sim(self.q, torch.zeros_like(self.q))
            self.robot.set_joint_position_target(self.q)
            self.robot.write_data_to_sim()
            
            for _ in range(60):
                self.sim.step()
                self.robot.update(self.physics_dt)
                for o in self.objects.values():
                    o.update(self.physics_dt)

    def save_csv(self):
        import csv
        os.makedirs("data", exist_ok=True)
        ts = self.run_timestamp
        
        csv_file = os.path.join("data", f"hil_scenario1_results_{ts}.csv")
        csv_file_latest = os.path.join("data", "hil_scenario1_results_latest.csv")
        
        summary_file = os.path.join("data", f"hil_scenario1_summary_{ts}.csv")
        summary_file_latest = os.path.join("data", "hil_scenario1_summary_latest.csv")

        if self.csv_records:
            fieldnames = list(self.csv_records[0].keys())
            for path in [csv_file, csv_file_latest]:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.csv_records)
            print(f"\n[INFO] Đã xuất thành công {len(self.csv_records)} dòng kết quả ra file: {csv_file}")
        else:
            fieldnames = ["timestamp", "target_x", "target_y", "target_z", "std_x", "std_y", "std_z", "prop_x", "prop_y", "prop_z", "err_std_mm", "err_prop_mm", "delay_ms", "tau_filt_ms"]
            for path in [csv_file, csv_file_latest]:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            print(f"\n[INFO] Đã tạo file CSV kết quả (đang chờ gói tin): {csv_file}")

        # Đồng thời tạo file tổng hợp summary.csv
        if self.err_std_history and self.err_prop_history:
            dt = 0.01
            mae_std = np.mean(np.linalg.norm(self.err_std_history, axis=1)) * 1000
            mae_prop = np.mean(np.linalg.norm(self.err_prop_history, axis=1)) * 1000
            q_ddot_std = np.diff(self.q_dot_std_history, axis=0) / dt if len(self.q_dot_std_history) > 1 else np.zeros((1, 6))
            q_ddot_prop = np.diff(self.q_dot_prop_history, axis=0) / dt if len(self.q_dot_prop_history) > 1 else np.zeros((1, 6))
            jvci_std = np.sum(np.square(np.linalg.norm(q_ddot_std, axis=1))) * dt
            jvci_prop = np.sum(np.square(np.linalg.norm(q_ddot_prop, axis=1))) * dt

            for path in [summary_file, summary_file_latest]:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "AnyTeleop_Standard", "Proposed_CLIK", "Improvement_Percent"])
                    writer.writerow(["MAE_mm", round(mae_std, 2), round(mae_prop, 2), round(((mae_std - mae_prop)/mae_std)*100, 1)])
                    writer.writerow(["JVCI", round(jvci_std, 2), round(jvci_prop, 2), round(((jvci_std - jvci_prop)/jvci_std)*100, 1)])
            print(f"[INFO] Đã xuất file tổng hợp kết quả: {summary_file}")

    def interpolate_target_position(self, target_time):
        """Tìm vị trí mục tiêu bằng cách nội suy tuyến tính trong lịch sử thời gian phát từ Laptop"""
        if not self.packet_history:
            return None, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(5)
        
        history = list(self.packet_history)
        
        # Nếu thời gian tìm kiếm cũ hơn gói tin cũ nhất
        if target_time <= history[0]["pkt_time"]:
            return history[0]["pos"], history[0]["quat"], history[0]["flex"]
        
        # Nếu mới hơn gói tin mới nhất
        if target_time >= history[-1]["pkt_time"]:
            return history[-1]["pos"], history[-1]["quat"], history[-1]["flex"]
        
        # Tìm khoảng lân cận để nội suy
        for i in range(len(history) - 1):
            p1 = history[i]
            p2 = history[i+1]
            if p1["pkt_time"] <= target_time <= p2["pkt_time"]:
                gap = p2["pkt_time"] - p1["pkt_time"]
                if gap < 1e-6:
                    return p1["pos"], p1["quat"], p1["flex"]
                alpha = (target_time - p1["pkt_time"]) / gap
                pos_interp = (1.0 - alpha) * p1["pos"] + alpha * p2["pos"]
                quat_interp = p1["quat"]
                flex_interp = (1.0 - alpha) * p1["flex"] + alpha * p2["flex"]
                return pos_interp, quat_interp, flex_interp
                
        return history[-1]["pos"], history[-1]["quat"], history[-1]["flex"]

    def run(self):
        """Vòng lặp chính đơn luồng (Single-thread loop) tương thích 100% với Omniverse GUI"""
        dt = 0.01
        print("\n==================================================================")
        print(" PC RECEIVER (HIL SCENARIO 1): ĐANG CHỜ NHẬN DỮ LIỆU TỪ WEBCAM LAPTOP...")
        print(" Chạy camera_stream_raw.py trên Laptop để bắt đầu thực nghiệm.")
        print("==================================================================\n")
        
        try:
            while self.running and (simulation_app is None or simulation_app.is_running()):
                loop_start = time.time()
                curr_real_time = time.time()
                
                # 1. Đọc tất cả gói tin UDP thô khả dụng (non-blocking)
                while True:
                    try:
                        data, addr = self.sock.recvfrom(1024)
                        if len(data) == 52:
                            payload = struct.unpack("13f", data)
                            pkt_time = payload[0]
                            target_pos = np.array(payload[1:4])
                            target_quat = np.array(payload[4:8])
                            finger_flex = np.array(payload[8:13])
                            
                            delay = self.tau_mean + np.random.uniform(-self.jitter_amp, self.jitter_amp)
                            scheduled_time = curr_real_time + delay
                            
                            self.packet_queue.append({
                                "scheduled_time": scheduled_time,
                                "pos": target_pos,
                                "quat": target_quat,
                                "flex": finger_flex,
                                "delay": delay,
                                "pkt_time": pkt_time
                            })
                            self.packet_history.append({
                                "pkt_time": pkt_time,
                                "pos": target_pos,
                                "quat": target_quat,
                                "flex": finger_flex
                            })
                    except BlockingIOError:
                        break
                    except Exception as e:
                        break
                
                # 2. Lấy gói tin đến hạn nhận từ hàng đợi trễ
                target_pos = None
                target_quat = np.array([0.0, 0.0, 0.0, 1.0])
                finger_flex = np.zeros(5)
                target_delay = 0.0
                pkt = None
                
                while self.packet_queue and self.packet_queue[0]["scheduled_time"] <= curr_real_time:
                    pkt = self.packet_queue.popleft()
                    target_pos = pkt["pos"]
                    target_quat = pkt["quat"]
                    finger_flex = pkt["flex"]
                    target_delay = pkt["delay"]
                
                # 3. Tính toán điều khiển bám vị trí nếu có gói tin đến hạn
                if target_pos is not None and pkt is not None:
                    if self.start_time is None:
                        self.start_time = curr_real_time
                    
                    # --- A. ĐIỀU KHIỂN TIÊU CHUẨN (AnyTeleop - Không bù trễ) ---
                    e_std = target_pos - self.x_curr_std
                    self.err_std_history.append(e_std)
                    
                    q_dot_std_step = 6.0 * e_std
                    self.x_curr_std += q_dot_std_step * dt
                    self.q_dot_std_history.append(np.tile(q_dot_std_step[:1], 6))
                    
                    # --- B. ĐIỀU KHIỂN ĐỀ XUẤT (Proposed CLIK - LPF + Bù trễ) ---
                    tau_filt_dot = (target_delay - self.tau_filt) / self.T_f
                    self.tau_filt += tau_filt_dot * dt
                    
                    # Nội suy mục tiêu tại thời điểm trễ đã lọc LPF để khử jitter
                    t_target_filt = pkt["pkt_time"] + (target_delay - self.tau_filt)
                    pos_filt, _, _ = self.interpolate_target_position(t_target_filt)
                    if pos_filt is None:
                        pos_filt = target_pos
                    
                    # Tính vận tốc feedforward (vd_filt) bằng đạo hàm số của vị trí lọc
                    if self.prev_pos_filt is not None:
                        vd_filt = (pos_filt - self.prev_pos_filt) / dt
                        # Giới hạn vận tốc tránh nổ gai đột ngột do nhiễu MediaPipe
                        vd_filt = np.clip(vd_filt, -0.5, 0.5)
                    else:
                        vd_filt = np.zeros(3)
                    self.prev_pos_filt = pos_filt.copy()
                    
                    e_prop = pos_filt - self.x_curr_prop
                    self.err_prop_history.append(e_prop)
                    
                    hist_steps = int(self.tau_filt / dt)
                    if len(self.err_prop_history) > hist_steps:
                        e_prop_delayed = self.err_prop_history[-hist_steps]
                    else:
                        e_prop_delayed = e_prop
                    
                    # Luật Proposed CLIK đầy đủ: feedforward + proportional + delay-feedback
                    q_dot_prop_step = vd_filt + self.K_p.dot(e_prop) + self.K_d.dot(e_prop_delayed)
                    self.x_curr_prop += q_dot_prop_step * dt
                    self.q_dot_prop_history.append(np.tile(q_dot_prop_step[:1], 6))

                    # Ghi lại dòng dữ liệu CSV
                    elapsed = curr_real_time - self.start_time
                    self.csv_records.append({
                        "timestamp": round(elapsed, 4),
                        "target_x": round(float(target_pos[0]), 5),
                        "target_y": round(float(target_pos[1]), 5),
                        "target_z": round(float(target_pos[2]), 5),
                        "std_x": round(float(self.x_curr_std[0]), 5),
                        "std_y": round(float(self.x_curr_std[1]), 5),
                        "std_z": round(float(self.x_curr_std[2]), 5),
                        "prop_x": round(float(self.x_curr_prop[0]), 5),
                        "prop_y": round(float(self.x_curr_prop[1]), 5),
                        "prop_z": round(float(self.x_curr_prop[2]), 5),
                        "err_std_mm": round(float(np.linalg.norm(e_std) * 1000), 2),
                        "err_prop_mm": round(float(np.linalg.norm(e_prop) * 1000), 2),
                        "delay_ms": round(float(target_delay * 1000), 1),
                        "tau_filt_ms": round(float(self.tau_filt * 1000), 1)
                    })

                    # Cập nhật khớp robot theo kết quả IK mới
                    if self.has_sim:
                        try:
                            arm_angles = self.solver.solve_arm_ik(self.x_curr_prop, target_quat)
                            self.q[:, self.arm_ids] = torch.tensor(arm_angles, device=self.device)
                            
                            # Ánh xạ ngón tay thô
                            for i, f_name in enumerate(["index", "middle", "ring", "pinky"]):
                                flex_val = finger_flex[i] * 1.309
                                if f_name in self.hand_ids:
                                    for hid in self.hand_ids[f_name][1:]:
                                        self.q[0, hid] = flex_val
                            if "thumb" in self.hand_ids:
                                self.q[0, self.hand_ids["thumb"][0]] = -finger_flex[4] * 1.309
                                for hid in self.hand_ids["thumb"][1:]:
                                    self.q[0, hid] = finger_flex[4] * 1.571
                        except Exception:
                            pass
                    
                    # Tính toán và xuất kết quả đối chứng tức thời mỗi 2 giây
                    if len(self.err_std_history) % 200 == 0:
                        mae_std = np.mean(np.linalg.norm(self.err_std_history, axis=1)) * 1000
                        mae_prop = np.mean(np.linalg.norm(self.err_prop_history, axis=1)) * 1000
                        
                        q_ddot_std = np.diff(self.q_dot_std_history, axis=0) / dt
                        q_ddot_prop = np.diff(self.q_dot_prop_history, axis=0) / dt
                        jvci_std = np.sum(np.square(np.linalg.norm(q_ddot_std, axis=1))) * dt
                        jvci_prop = np.sum(np.square(np.linalg.norm(q_ddot_prop, axis=1))) * dt
                        
                        print(f"Time: {elapsed:.1f}s | AnyTeleop MAE: {mae_std:.2f}mm | Proposed MAE: {mae_prop:.2f}mm (Imp: {((mae_std-mae_prop)/mae_std)*100:.1f}%)")
                        print(f"            | AnyTeleop JVCI: {jvci_std:.1f}  | Proposed JVCI: {jvci_prop:.1f} (Imp: {((jvci_std-jvci_prop)/jvci_std)*100:.1f}%)")
                        print("-" * 75)
                        self.save_csv()
                
                # 4. LUÔN LUÔN CẬP NHẬT ISAAC SIM VÀ PHẬT THỂ MỖI BƯỚC THỜI GIAN (Tránh lỗi đóng ứng dụng)
                if self.has_sim:
                    self.robot.set_joint_position_target(self.q)
                    self.robot.write_data_to_sim()
                    self.sim.step()
                    self.robot.update(self.physics_dt)
                    for o in self.objects.values():
                        o.update(self.physics_dt)

                # Đảm bảo tần số chạy ổn định ở 100 Hz
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, dt - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Đang dừng hệ thống HIL Scenario 1...")
        finally:
            self.running = False
            self.sock.close()
            self.save_csv()
            if simulation_app is not None:
                simulation_app.close()
            print("[INFO] Đã đóng tài nguyên và tắt mô phỏng hoàn tất.")

if __name__ == "__main__":
    sim = HILSimulationScenario1()
    sim.run()
