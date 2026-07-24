import argparse
import os
import sys
from datetime import datetime

# Add site-packages search paths for pybullet
for site_pkg in [
    "/home/nhglab/anaconda3/envs/env_isaacsim/lib/python3.11/site-packages",
    "/home/nhglab/anaconda3/envs/hil_env/lib/python3.10/site-packages",
    "/home/nhglab/anaconda3/lib/python3.13/site-packages",
]:
    if os.path.exists(site_pkg) and site_pkg not in sys.path:
        sys.path.append(site_pkg)

# Launch Isaac Sim AppLauncher
parser = argparse.ArgumentParser(description="Isaac Sim HIL Scenario 2 Dynamic Grasping Client")
parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP bind IP address")
parser.add_argument("--port", type=int, default=5005, help="UDP bind port")
parser.add_argument("--usd", type=str, default="config/grasp_scene.usd", help="USD grasp scene path")
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
    print(f"[WARN] Running in numerical HIL mode (No Isaac Sim 3D UI): {e}")

class HILSimulationScenario2:
    def __init__(self):
        # UDP Setup
        self.UDP_IP = args_cli.ip if args_cli else "0.0.0.0"
        self.UDP_PORT = args_cli.port if args_cli else 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((self.UDP_IP, self.UDP_PORT))
            self.sock.setblocking(False)
        except Exception as e:
            print(f"[ERROR] UDP bind error: {e}")

        # Latency buffers
        self.packet_queue = deque()
        self.packet_history = deque(maxlen=2000)
        self.running = True

        # Master clock sync
        self.latest_pkt_time = None
        self.latest_pkt_arrival_time = None
        self.latest_delay = 0.150
        self.tau_mean = 0.150
        self.jitter_amp = 0.050

        # Robot states
        self.x_curr_std = np.array([0.18, 0.40, 0.90])
        self.x_curr_prop = np.array([0.18, 0.40, 0.90])
        self.prev_pos_filt = None
        self.vd_filt_smooth = np.zeros(3)
        self.T_vd = 0.10

        # Controller parameters
        self.K_p = 5.0 * np.eye(3)
        self.K_d = 2.0 * np.eye(3)
        self.T_f = 0.25
        self.tau_filt = self.tau_mean
        self.start_time = None

        # Scenario 2 - Dynamic Ball definition (Oscillating suspended target)
        self.ball_pos = np.array([0.18, 0.45, 0.85])
        self.ball_freq = 0.15  # 0.15 Hz sinusoidal oscillation

        # Grasping state machine
        self.grasp_triggered_std = False
        self.grasp_triggered_prop = False
        self.grasp_success_std = False
        self.grasp_success_prop = False
        
        self.ttg_std = 0.0
        self.ttg_prop = 0.0
        
        self.err_std_history = []
        self.err_prop_history = []
        self.q_dot_std_history = []
        self.q_dot_prop_history = []

        # CSV Logging Settings
        self.csv_records = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # HIL Simulation scene
        self.has_sim = HAS_ROBOT_SIM
        if self.has_sim:
            usd_path = args_cli.usd
            if not os.path.exists(usd_path):
                usd_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "grasp_scene.usd")
            
            print(f"[INFO] Initializing Isaac Sim Scene from: {usd_path}")
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

    def update_ball(self, elapsed_time):
        """Simulates the dynamic suspended ball oscillation in 3D space"""
        r = 0.08  # 8cm oscillation radius
        self.ball_pos[0] = 0.18 + r * np.cos(2 * np.pi * self.ball_freq * elapsed_time)
        self.ball_pos[1] = 0.45 + r * np.sin(2 * np.pi * self.ball_freq * elapsed_time)
        self.ball_pos[2] = 0.85 + 0.03 * np.cos(2 * np.pi * 0.5 * elapsed_time)

    def interpolate_target_position(self, target_time):
        if not self.packet_history:
            return None, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(5)
        history = list(self.packet_history)
        if target_time <= history[0]["pkt_time"]:
            return history[0]["pos"], history[0]["quat"], history[0]["flex"]
        if target_time >= history[-1]["pkt_time"]:
            return history[-1]["pos"], history[-1]["quat"], history[-1]["flex"]
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

    def save_csv(self):
        import csv
        os.makedirs("data", exist_ok=True)
        ts = self.run_timestamp
        
        csv_file = os.path.join("data", f"hil_scenario2_results_{ts}.csv")
        csv_file_latest = os.path.join("data", "hil_scenario2_results_latest.csv")
        
        summary_file = os.path.join("data", f"hil_scenario2_summary_{ts}.csv")
        summary_file_latest = os.path.join("data", "hil_scenario2_summary_latest.csv")

        # 1. Export detailed frame-by-frame data
        if self.csv_records:
            fieldnames = list(self.csv_records[0].keys())
            for path in [csv_file, csv_file_latest]:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.csv_records)
            print(f"\n[INFO] Successfully exported {len(self.csv_records)} rows to: {csv_file}")
        else:
            fieldnames = ["timestamp", "ball_x", "ball_y", "ball_z", "std_x", "std_y", "std_z", "prop_x", "prop_y", "prop_z", "dist_std_mm", "dist_prop_mm", "delay_ms"]
            for path in [csv_file, csv_file_latest]:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

        # 2. Export summary KPIs
        if self.err_std_history and self.err_prop_history:
            mae_std = np.mean(np.linalg.norm(self.err_std_history, axis=1)) * 1000
            mae_prop = np.mean(np.linalg.norm(self.err_prop_history, axis=1)) * 1000
            impr_mae = round(((mae_std - mae_prop) / mae_std) * 100, 1) if mae_std > 0 else 0.0
            
            for path in [summary_file, summary_file_latest]:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "AnyTeleop_Standard", "Proposed_CLIK", "Improvement"])
                    writer.writerow(["Grasp_Success", "Yes" if self.grasp_success_std else "No", "Yes" if self.grasp_success_prop else "No", "N/A"])
                    writer.writerow(["Time_to_Grasp_s", round(self.ttg_std, 2), round(self.ttg_prop, 2), round(self.ttg_std - self.ttg_prop, 2)])
                    writer.writerow(["Dynamic_MAE_mm", round(mae_std, 2), round(mae_prop, 2), f"{impr_mae}%"])
            print(f"[INFO] Saved Scenario 2 summary to: {summary_file}")

    def run(self):
        dt = 0.01
        print("\n==================================================================")
        print(" PC RECEIVER (HIL SCENARIO 2): DYNAMIC GRASPING SIMULATOR...")
        print(" Using scene: config/grasp_scene.usd")
        print(" Guide your hand on Laptop camera to grasp the oscillating ball!")
        print("==================================================================\n")

        try:
            step_count = 0
            while self.running and (simulation_app is None or simulation_app.is_running()):
                loop_start = time.time()
                curr_real_time = time.time()
                step_count += 1

                # 1. Read non-blocking UDP packets
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
                            
                            self.latest_pkt_time = pkt_time
                            self.latest_pkt_arrival_time = curr_real_time
                            self.latest_delay = delay
                            
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
                    except Exception:
                        break

                # 2. Control execution & physics updates
                if self.latest_pkt_time is not None:
                    if self.start_time is None:
                        self.start_time = curr_real_time
                    
                    elapsed = curr_real_time - self.start_time
                    self.update_ball(elapsed)

                    # Estimate laptop time
                    t_laptop = self.latest_pkt_time + (curr_real_time - self.latest_pkt_arrival_time)

                    # --- A. STANDARD CONTROL ---
                    t_target_raw = t_laptop - self.latest_delay
                    pos_raw, quat_raw, flex_raw = self.interpolate_target_position(t_target_raw)
                    if pos_raw is None:
                        pos_raw = self.x_curr_std

                    dist_to_ball_std = np.linalg.norm(self.x_curr_std - self.ball_pos)
                    self.err_std_history.append(self.x_curr_std - self.ball_pos)
                    
                    if not self.grasp_triggered_std:
                        q_dot_std_step = 6.0 * (pos_raw - self.x_curr_std)
                        if dist_to_ball_std <= 0.045:
                            self.grasp_triggered_std = True
                            self.ttg_std = elapsed
                            if np.linalg.norm(q_dot_std_step) < 0.25:
                                self.grasp_success_std = True
                            print(f"[EVENT] Standard Controller reached ball at {self.ttg_std:.2f}s! Success: {self.grasp_success_std}")
                    else:
                        q_dot_std_step = np.zeros(3)

                    self.x_curr_std += q_dot_std_step * dt
                    self.q_dot_std_history.append(np.tile(q_dot_std_step[:1], 6))

                    # --- B. PROPOSED CONTROL ---
                    tau_filt_dot = (self.latest_delay - self.tau_filt) / self.T_f
                    self.tau_filt += tau_filt_dot * dt

                    t_target_filt = t_laptop - self.tau_filt
                    pos_filt, _, flex_filt = self.interpolate_target_position(t_target_filt)
                    if pos_filt is None:
                        pos_filt = self.x_curr_prop

                    # Smooth feedforward
                    if self.prev_pos_filt is not None:
                        vd_filt_raw = (pos_filt - self.prev_pos_filt) / dt
                        vd_filt_raw = np.clip(vd_filt_raw, -0.5, 0.5)
                        alpha_vd = dt / (self.T_vd + dt)
                        self.vd_filt_smooth = (1.0 - alpha_vd) * self.vd_filt_smooth + alpha_vd * vd_filt_raw
                    else:
                        self.vd_filt_smooth = np.zeros(3)
                    self.prev_pos_filt = pos_filt.copy()

                    dist_to_ball_prop = np.linalg.norm(self.x_curr_prop - self.ball_pos)
                    self.err_prop_history.append(self.x_curr_prop - self.ball_pos)

                    if not self.grasp_triggered_prop:
                        e_prop = pos_filt - self.x_curr_prop
                        hist_steps = int(self.tau_filt / dt)
                        e_prop_delayed = self.err_prop_history[-hist_steps] if len(self.err_prop_history) > hist_steps else e_prop
                        
                        q_dot_prop_step = self.vd_filt_smooth + self.K_p.dot(e_prop) + self.K_d.dot(e_prop_delayed)
                        
                        if dist_to_ball_prop <= 0.045:
                            self.grasp_triggered_prop = True
                            self.ttg_prop = elapsed
                            if np.linalg.norm(q_dot_prop_step) < 0.20:
                                self.grasp_success_prop = True
                            print(f"[EVENT] Proposed Controller reached ball at {self.ttg_prop:.2f}s! Success: {self.grasp_success_prop}")
                    else:
                        q_dot_prop_step = np.zeros(3)

                    self.x_curr_prop += q_dot_prop_step * dt
                    self.q_dot_prop_history.append(np.tile(q_dot_prop_step[:1], 6))

                    # Append metrics to CSV records
                    self.csv_records.append({
                        "timestamp": f"{elapsed:.3f}",
                        "ball_x": round(self.ball_pos[0], 4),
                        "ball_y": round(self.ball_pos[1], 4),
                        "ball_z": round(self.ball_pos[2], 4),
                        "std_x": round(self.x_curr_std[0], 4),
                        "std_y": round(self.x_curr_std[1], 4),
                        "std_z": round(self.x_curr_std[2], 4),
                        "prop_x": round(self.x_curr_prop[0], 4),
                        "prop_y": round(self.x_curr_prop[1], 4),
                        "prop_z": round(self.x_curr_prop[2], 4),
                        "dist_std_mm": round(dist_to_ball_std * 1000, 2),
                        "dist_prop_mm": round(dist_to_ball_prop * 1000, 2),
                        "delay_ms": round(self.latest_delay * 1000, 1)
                    })

                    # Command Simulator
                    if self.has_sim:
                        try:
                            # 1. Update ball position in Isaac Sim physics
                            if "ball" in self.objects:
                                self.objects["ball"].set_position(self.ball_pos)
                            
                            # 2. Retarget arm angles & wrist orientation
                            q_arm_prop = self.solver.solve_arm_ik(self.x_curr_prop, quat_raw)
                            self.q[:, self.arm_ids] = torch.tensor(q_arm_prop, device=self.device)
                            
                            # 3. Retarget hand fingers + abduction
                            hand_angles = self.solver.solve_hand_orientation(flex_filt)
                            for fname, ids in self.hand_ids.items():
                                if fname in hand_angles:
                                    vals = hand_angles[fname]
                                    for idx, hid in enumerate(ids):
                                        if idx < len(vals):
                                            self.q[0, hid] = float(vals[idx])
                                            
                            if self.grasp_triggered_prop:
                                # Close fingers when grasp triggered
                                for fname, ids in self.hand_ids.items():
                                    for hid in ids[1:]:
                                        self.q[0, hid] = 1.309
                        except Exception:
                            pass

                    # Periodic reporting & CSV saving
                    if step_count % 200 == 0:
                        print(f"Time: {elapsed:.1f}s | Ball: [{self.ball_pos[0]:.2f}, {self.ball_pos[1]:.2f}, {self.ball_pos[2]:.2f}]")
                        print(f"            | Standard Dist: {dist_to_ball_std*1000:.1f}mm | Proposed Dist: {dist_to_ball_prop*1000:.1f}mm")
                        print("-" * 75)
                        self.save_csv()

                # Step simulation continuously (prevents Isaac Sim 5.1 timeout)
                if self.has_sim:
                    self.robot.set_joint_position_target(self.q)
                    self.robot.write_data_to_sim()
                    self.sim.step()
                    self.robot.update(self.physics_dt)
                    for o in self.objects.values():
                        o.update(self.physics_dt)

                elapsed_loop = time.time() - loop_start
                sleep_time = max(0.001, dt - elapsed_loop)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            self.running = False
            self.sock.close()
            self.save_csv()
            if simulation_app is not None:
                simulation_app.close()
            print("[INFO] Exit completed.")

if __name__ == "__main__":
    sim = HILSimulationScenario2()
    sim.run()
