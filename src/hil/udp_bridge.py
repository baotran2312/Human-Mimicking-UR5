import socket
import json
import time

class UDPBridge:
    def __init__(self, ip="127.0.0.1", port=5005):
        """Khởi tạo UDP Bridge để truyền góc khớp đến simulator/robot"""
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[INFO] UDP Bridge initialized. Target: {self.ip}:{self.port}")

    def send_joint_angles(self, arm_angles, finger_angles):
        """Đóng gói và truyền góc khớp 25 DoF qua UDP"""
        # Định dạng gói dữ liệu JSON
        payload = {
            "timestamp": time.time(),
            "arm": list(arm_angles),
            "hand": {
                "index": [
                    finger_angles.get("index_J1", 0.0),
                    finger_angles.get("index_J2", 0.0),
                    finger_angles.get("index_J3", 0.0),
                    finger_angles.get("index_J4", 0.0)
                ],
                "middle": [
                    finger_angles.get("middle_J1", 0.0),
                    finger_angles.get("middle_J2", 0.0),
                    finger_angles.get("middle_J3", 0.0),
                    finger_angles.get("middle_J4", 0.0)
                ],
                "ring": [
                    finger_angles.get("ring_J1", 0.0),
                    finger_angles.get("ring_J2", 0.0),
                    finger_angles.get("ring_J3", 0.0),
                    finger_angles.get("ring_J4", 0.0)
                ],
                "pinky": [
                    finger_angles.get("pinky_J1", 0.0),
                    finger_angles.get("pinky_J2", 0.0),
                    finger_angles.get("pinky_J3", 0.0),
                    finger_angles.get("pinky_J4", 0.0)
                ],
                "thumb": [
                    finger_angles.get("thumb_j1", 0.0),
                    finger_angles.get("thumb_j2", 0.0),
                    finger_angles.get("thumb_j3", 0.0)
                ]
            }
        }
        
        # Chuyển sang chuỗi JSON và gửi đi
        try:
            data = json.dumps(payload).encode("utf-8")
            self.sock.sendto(data, (self.ip, self.port))
        except Exception as e:
            print(f"[ERROR] Failed to send UDP packet: {e}")

    def close(self):
        self.sock.close()
        print("[INFO] UDP socket closed.")
