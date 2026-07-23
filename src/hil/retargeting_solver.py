import pybullet as p
import pybullet_data
import numpy as np
import os

class RetargetingSolver:
    def __init__(self, urdf_path):
        """Khởi tạo bộ giải động học ngược (IK) và ánh xạ khớp sử dụng PyBullet"""
        # Khởi tạo PyBullet ở chế độ DIRECT (không giao diện để chạy nhanh/headless)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load mô hình robot UR5 + Hand
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"Không tìm thấy file URDF tại: {urdf_path}")
            
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        
        # Định danh các khớp và link
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = {}
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        
        # UR5 Arm Joint Names
        arm_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Quét toàn bộ khớp từ URDF
        for idx in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, idx)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            
            # Lưu cấu hình giới hạn khớp (joint limits)
            self.joint_info[joint_name] = {
                "index": idx,
                "type": joint_type,
                "lower_limit": info[8],
                "upper_limit": info[9],
                "max_force": info[10],
                "max_velocity": info[11]
            }
            
            # Phân loại khớp Arm và khớp Finger
            if joint_name in arm_names:
                self.arm_joint_indices.append(idx)
            elif "index" in joint_name or "middle" in joint_name or "ring" in joint_name or "pinky" in joint_name or "thumb" in joint_name:
                if joint_type == p.JOINT_REVOLUTE:
                    self.finger_joint_indices.append(idx)

        # Lấy chỉ số link Tool Center Point (TCP) - DH_base_link nối với wrist_3
        self.tcp_link_index = -1
        for idx in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, idx)
            link_name = info[12].decode("utf-8")
            if link_name == "DH_base_link":
                self.tcp_link_index = idx
                break
        
        print(f"[INFO] URDF loaded successfully. Found {len(self.arm_joint_indices)} UR5 joints and {len(self.finger_joint_indices)} active finger joints.")

    def solve_arm_ik(self, target_pos, target_quat):
        """Giải động học ngược (IK) cho 6 khớp cánh tay UR5"""
        # Sử dụng giải thuật IK của PyBullet
        ik_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.tcp_link_index,
            target_pos,
            target_quat,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        # Chỉ lấy 6 góc khớp của UR5
        return [ik_angles[i] for i in range(6)]

    def map_finger_joints(self, landmarks):
        """Ánh xạ cử chỉ tay người (21 keypoints từ MediaPipe) sang 19 khớp ngón tay robot"""
        # landmarks là danh sách 21 điểm (x, y, z) từ MediaPipe
        
        joint_targets = {}
        
        # --- 1. TÍNH GÓC DẠNG/KHÉP (ABDUCTION/ADDUCTION - J1 JOINTS) ---
        # Trích xuất tọa độ 3D của các điểm mốc chính trên bàn tay
        p0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])  # Cổ tay (Wrist)
        p5 = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])  # Gốc ngón trỏ (Index MCP)
        p9 = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])  # Gốc ngón giữa (Middle MCP)
        p17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])  # Gốc ngón út (Pinky MCP)
        
        # Định nghĩa hệ trục tọa độ của lòng bàn tay (Palm Local Frame)
        v_palm = p9 - p0
        v_palm_len = np.linalg.norm(v_palm)
        y_palm = v_palm / v_palm_len if v_palm_len > 1e-6 else np.array([0.0, 1.0, 0.0])
        
        v_lat = p17 - p5
        z_palm = np.cross(y_palm, v_lat)
        z_palm_len = np.linalg.norm(z_palm)
        z_palm = z_palm / z_palm_len if z_palm_len > 1e-6 else np.array([0.0, 0.0, 1.0])
        
        x_palm = np.cross(y_palm, z_palm)
        
        # Cặp chỉ số MCP -> PIP của các ngón để tính hướng đốt ngón thứ nhất
        finger_mcp_pip = {
            "index": (5, 6),
            "middle": (9, 10),
            "ring": (13, 14),
            "pinky": (17, 18)
        }
        
        finger_angles_raw = {}
        for f_name, (mcp_idx, pip_idx) in finger_mcp_pip.items():
            pmcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])
            ppip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y, landmarks[pip_idx].z])
            v_seg = ppip - pmcp
            
            # Chiếu đốt ngón lên hệ trục lòng bàn tay
            x_seg = np.dot(v_seg, x_palm)
            y_seg = np.dot(v_seg, y_palm)
            finger_angles_raw[f_name] = np.arctan2(x_seg, y_seg)

        # Mốc trung tính (neutral spread angle) khi tay xòe tự nhiên (đối chiếu trực tiếp với y_palm)
        neutral_bias = {
            "index": -0.15,
            "middle": 0.0,
            "ring": 0.15,
            "pinky": 0.30
        }
        
        spread_gain = 1.5
        j1_limit = 0.1396
        
        # --- 2. TÍNH GÓC GẬP NGÓN (FLEXION - J2->J4 JOINTS) ---
        def get_finger_flexion(mcp_idx, tip_idx, wrist_idx=0):
            wrist_pos = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y, landmarks[wrist_idx].z])
            mcp_pos = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])
            tip_pos = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
            
            max_len = np.linalg.norm(mcp_pos - wrist_pos) + np.linalg.norm(tip_pos - mcp_pos)
            curr_len = np.linalg.norm(tip_pos - wrist_pos)
            
            flexion = np.clip((max_len - curr_len) / (max_len * 0.4), 0.0, 1.0)
            return flexion

        # Tính độ gập để làm mượt / dập nhiễu dạng khép (fade) khi ngón tay co lại
        index_flex = get_finger_flexion(5, 8)
        middle_flex = get_finger_flexion(9, 12)
        ring_flex = get_finger_flexion(13, 16)
        pinky_flex = get_finger_flexion(17, 20)
        
        # Hệ số fade: 1.0 khi duỗi thẳng (dạng khép tự do), về 0.0 khi co lại (khóa về trung tính)
        index_fade = 1.0 - np.clip(index_flex, 0.0, 1.0)
        middle_fade = 1.0 - np.clip(middle_flex, 0.0, 1.0)
        ring_fade = 1.0 - np.clip(ring_flex, 0.0, 1.0)
        pinky_fade = 1.0 - np.clip(pinky_flex, 0.0, 1.0)

        # Gán góc J1 độc lập kết hợp hệ số fade
        joint_targets["index_J1"] = np.clip((finger_angles_raw["index"] - neutral_bias["index"]) * spread_gain * index_fade, -j1_limit, j1_limit)
        joint_targets["middle_J1"] = np.clip((finger_angles_raw["middle"] - neutral_bias["middle"]) * spread_gain * middle_fade, -j1_limit, j1_limit)
        joint_targets["ring_J1"] = np.clip((finger_angles_raw["ring"] - neutral_bias["ring"]) * spread_gain * ring_fade, -j1_limit, j1_limit)
        joint_targets["pinky_J1"] = np.clip((finger_angles_raw["pinky"] - neutral_bias["pinky"]) * spread_gain * pinky_fade, -j1_limit, j1_limit)

        fingers = {
            "index": (5, 8),
            "middle": (9, 12),
            "ring": (13, 16),
            "pinky": (17, 20)
        }
        
        for f_name, (mcp, tip) in fingers.items():
            flex = get_finger_flexion(mcp, tip)
            max_limit = self.joint_info[f"{f_name}_J2"]["upper_limit"]
            angle = flex * max_limit
            joint_targets[f"{f_name}_J2"] = angle
            joint_targets[f"{f_name}_J3"] = angle
            joint_targets[f"{f_name}_J4"] = angle

        # 3. Ánh xạ ngón cái (Thumb - có 3 khớp j1->j3)
        # Tính độ gập ngón cái dựa vào khoảng cách ngón cái đến ngón trỏ (cho cử chỉ Pinch)
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        thumb_flex = np.clip(1.0 - (np.linalg.norm(thumb_tip - index_mcp) / 0.15), 0.0, 1.0)
        
        # thumb_j1 giới hạn [-1.309, 0]
        joint_targets["thumb_j1"] = -thumb_flex * 1.309
        # thumb_j2, j3 giới hạn [0, 1.571]
        joint_targets["thumb_j2"] = thumb_flex * 1.571
        joint_targets["thumb_j3"] = thumb_flex * 1.571

        return joint_targets

    def solve_hand_orientation(self, landmarks):
        """Tính toán hướng quaternion của bàn tay người từ MediaPipe landmarks"""
        p0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])  # Wrist
        p5 = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])  # Index MCP
        p9 = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])  # Middle MCP
        p17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])  # Pinky MCP
        
        v_palm = p9 - p0
        v_palm_len = np.linalg.norm(v_palm)
        y_palm = v_palm / v_palm_len if v_palm_len > 1e-6 else np.array([0.0, 1.0, 0.0])
        
        v_lat = p17 - p5
        z_palm = np.cross(y_palm, v_lat)
        z_palm_len = np.linalg.norm(z_palm)
        z_palm = z_palm / z_palm_len if z_palm_len > 1e-6 else np.array([0.0, 0.0, 1.0])
        
        x_palm = np.cross(y_palm, z_palm)
        
        # Tạo ma trận xoay tương đối R_rel = [X_palm, -Y_palm, -Z_palm]
        # Hướng mặc định (identity) của robot là hướng thẳng xuống bàn
        R = np.zeros((3, 3))
        R[:, 0] = x_palm
        R[:, 1] = -y_palm
        R[:, 2] = -z_palm
        
        # Trực giao hóa ma trận xoay bằng SVD để tránh sai số cơ học
        try:
            U, _, Vt = np.linalg.svd(R)
            R = np.dot(U, Vt)
        except Exception:
            pass
            
        # Chuyển đổi R sang quaternion [qx, qy, qz, qw] theo chuẩn PyBullet
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        return [qx, qy, qz, qw]

    def disconnect(self):
        p.disconnect(self.physics_client)

if __name__ == "__main__":
    # Test độc lập xem solver có load được URDF không
    urdf = "config/ur5dex.urdf"
    if not os.path.exists(urdf):
        # Nếu chạy từ thư mục root thì tạo đường dẫn tương đối
        urdf = "../../config/ur5dex.urdf"
        
    try:
        solver = RetargetingSolver(urdf)
        # Test IK ngẫu nhiên
        test_pos = [0.1, 0.5, 0.9]
        test_quat = [0, 0, 0, 1]
        arm_angles = solver.solve_arm_ik(test_pos, test_quat)
        print(f"[TEST SUCCESS] Solved UR5 joint angles: {np.round(arm_angles, 4)}")
        solver.disconnect()
    except Exception as e:
        print(f"[TEST FAILED] Error occurred: {e}")
