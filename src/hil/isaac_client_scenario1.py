import socket
import struct
import time
import threading
import numpy as np
from collections import deque

class HILSimulationScenario1:
    def __init__(self):
        # Mạng nhận
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        
        # Hộp lưu trữ gói tin thô nhận được từ Laptop
        self.packet_queue = deque()
        self.running = True
        
        # Các thông số trễ mạng mô phỏng (Kịch bản 1)
        self.tau_mean = 0.150  # 150ms trễ
        self.jitter_amp = 0.050  # 50ms jitter
        
        # Trạng thái của Robot ảo trong mô phỏng (Arm TCP position)
        self.x_curr_std = np.array([0.18, 0.40, 0.90])
        self.x_curr_prop = np.array([0.18, 0.40, 0.90])
        
        # Gains của bộ điều khiển
        self.K_p = 5.0 * np.eye(3)
        self.K_d = 2.0 * np.eye(3)
        self.T_f = 0.25  # Hằng số lọc LPF
        
        # Trạng thái trễ lọc
        self.tau_filt = self.tau_mean
        self.prev_time = time.time()
        self.start_time = None
        
        # Các chỉ số tích lũy đo lường
        self.err_std_history = []
        self.err_prop_history = []
        self.q_dot_std_history = []
        self.q_dot_prop_history = []
        
        # Khóa đồng bộ
        self.lock = threading.Lock()

    def receive_packets(self):
        """Luồng nhận gói tin thô liên tục từ Laptop qua mạng"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if len(data) == 52:  # 13 floats * 4 bytes
                    payload = struct.unpack("13f", data)
                    
                    # Đọc dữ liệu từ gói tin
                    pkt_time = payload[0]
                    target_pos = np.array(payload[1:4])
                    target_quat = np.array(payload[4:8])
                    finger_flex = np.array(payload[8:13])
                    
                    # Mô phỏng độ trễ mạng thực tế (True + Jitter) cho gói tin này
                    # Trễ = trễ trung bình + Jitter ngẫu nhiên
                    delay = self.tau_mean + np.random.uniform(-self.jitter_amp, self.jitter_amp)
                    scheduled_time = time.time() + delay
                    
                    with self.lock:
                        self.packet_queue.append({
                            "scheduled_time": scheduled_time,
                            "pos": target_pos,
                            "quat": target_quat,
                            "flex": finger_flex,
                            "delay": delay
                        })
            except Exception as e:
                if self.running:
                    print(f"[ERROR] Lỗi nhận gói tin: {e}")

    def control_loop(self):
        """Luồng điều khiển mô phỏng HIL chạy ở tần số 100 Hz"""
        dt = 0.01
        print("\n==================================================================")
        print(" PC RECEIVER (HIL SCENARIO 1): ĐANG CHỜ NHẬN DỮ LIỆU TỪ WEBCAM LAPTOP...")
        print(" Chạy camera_stream_raw.py trên Laptop để bắt đầu thực nghiệm.")
        print("==================================================================\n")
        
        while self.running:
            loop_start = time.time()
            curr_real_time = time.time()
            
            # Lấy gói tin đã đến hạn nhận (sau khi đi qua độ trễ mạng mô phỏng)
            target_pos = None
            target_delay = 0.0
            
            with self.lock:
                # Tìm gói tin đã đạt đến scheduled_time
                while self.packet_queue and self.packet_queue[0]["scheduled_time"] <= curr_real_time:
                    pkt = self.packet_queue.popleft()
                    target_pos = pkt["pos"]
                    target_delay = pkt["delay"]
            
            if target_pos is not None:
                if self.start_time is None:
                    self.start_time = curr_real_time
                
                # --- A. ĐIỀU KHIỂN TIÊU CHUẨN (AnyTeleop - Không bù trễ) ---
                e_std = target_pos - self.x_curr_std
                self.err_std_history.append(e_std)
                
                q_dot_std_step = 6.0 * e_std
                self.x_curr_std += q_dot_std_step * dt
                self.q_dot_std_history.append(np.tile(q_dot_std_step[:1], 6))
                
                # --- B. ĐIỀU KHIỂN ĐỀ XUẤT (Proposed CLIK - LPF + Bù trễ) ---
                # Động học lọc thông thấp LPF
                tau_filt_dot = (target_delay - self.tau_filt) / self.T_f
                self.tau_filt += tau_filt_dot * dt
                
                e_prop = target_pos - self.x_curr_prop
                self.err_prop_history.append(e_prop)
                
                # Lấy sai số trễ trong quá khứ từ lịch sử (Delay feedback term)
                hist_steps = int(self.tau_filt / dt)
                if len(self.err_prop_history) > hist_steps:
                    e_prop_delayed = self.err_prop_history[-hist_steps]
                else:
                    e_prop_delayed = e_prop
                
                # Vận tốc điều khiển bám
                q_dot_prop_step = self.K_p.dot(e_prop) + self.K_d.dot(e_prop_delayed)
                self.x_curr_prop += q_dot_prop_step * dt
                self.q_dot_prop_history.append(np.tile(q_dot_prop_step[:1], 6))
                
                # Tính toán và xuất kết quả đối chứng tức thời mỗi 2 giây
                elapsed = curr_real_time - self.start_time
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
            
            # Đảm bảo tần số chạy ổn định ở 100 Hz
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)

    def start(self):
        self.t_recv = threading.Thread(target=self.receive_packets)
        self.t_ctrl = threading.Thread(target=self.control_loop)
        
        self.t_recv.start()
        self.t_ctrl.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Đang dừng hệ thống HIL...")
            self.running = False
            self.sock.close()
            self.t_recv.join()
            self.t_ctrl.join()
            print("[INFO] Đã dừng hoàn toàn.")

if __name__ == "__main__":
    sim = HILSimulationScenario1()
    sim.start()
