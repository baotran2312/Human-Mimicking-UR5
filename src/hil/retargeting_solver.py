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
        # Ta áp dụng phương pháp ánh xạ góc co duỗi tương đối (Heuristic Flexion Mapping)
        
        joint_targets = {}
        
        # Định nghĩa các ngón tay và các khớp tương ứng từ MediaPipe
        # Ngón cái: 1, 2, 3, 4 | Ngón trỏ: 5, 6, 7, 8 | Ngón giữa: 9, 10, 11, 12
        # Ngón nhẫn: 13, 14, 15, 16 | Ngón út: 17, 18, 19, 20
        
        # Hàm tính góc gập (flexion) của một ngón dựa vào khoảng cách từ gốc đến đầu ngón
        def get_finger_flexion(mcp_idx, tip_idx, wrist_idx=0):
            # Tính khoảng cách từ cổ tay đến đầu ngón và gốc ngón
            wrist_pos = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y, landmarks[wrist_idx].z])
            mcp_pos = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])
            tip_pos = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
            
            # Khoảng cách tối đa (khi ngón duỗi thẳng)
            max_len = np.linalg.norm(mcp_pos - wrist_pos) + np.linalg.norm(tip_pos - mcp_pos)
            # Khoảng cách hiện tại từ cổ tay đến đầu ngón
            curr_len = np.linalg.norm(tip_pos - wrist_pos)
            
            # Chuẩn hóa giá trị flexion từ 0.0 (duỗi thẳng) đến 1.0 (co hoàn toàn)
            flexion = np.clip((max_len - curr_len) / (max_len * 0.4), 0.0, 1.0)
            return flexion

        # 1. Ánh xạ các ngón trỏ, giữa, áp út, út (Mỗi ngón có 4 khớp J1->J4)
        fingers = {
            "index": (5, 8),
            "middle": (9, 12),
            "ring": (13, 16),
            "pinky": (17, 20)
        }
        
        for f_name, (mcp, tip) in fingers.items():
            flex = get_finger_flexion(mcp, tip)
            
            # J1 (Adduction/Abduction - khép/xòe ngón): Mặc định giữ thẳng (0.0)
            joint_targets[f"{f_name}_J1"] = 0.0
            
            # J2, J3, J4 (Flexion - Gập ngón): Ánh xạ tuyến tính vào dải giới hạn [0.0, 1.309] radian
            max_limit = self.joint_info[f"{f_name}_J2"]["upper_limit"]
            angle = flex * max_limit
            joint_targets[f"{f_name}_J2"] = angle
            joint_targets[f"{f_name}_J3"] = angle
            joint_targets[f"{f_name}_J4"] = angle

        # 2. Ánh xạ ngón cái (Thumb - có 3 khớp j1->j3)
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
