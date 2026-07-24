import argparse
import os
import sys

# Khởi chạy AppLauncher / SimulationApp của Isaac Sim trước khi import thư viện USD
try:
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Isaac Sim Assembly Scene Builder")
    parser.add_argument("--headless", action="store_true", help="Run simulation headless")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except (ImportError, Exception):
    try:
        from omni.isaac.kit import SimulationApp
        parser = argparse.ArgumentParser(description="Isaac Sim Assembly Scene Builder")
        parser.add_argument("--headless", action="store_true", help="Run simulation headless")
        args_cli, unknown = parser.parse_known_args()
        simulation_app = SimulationApp({"headless": args_cli.headless})
    except ImportError:
        print("\n[ERROR] Thư viện Isaac Sim (SimulationApp) không khả dụng khi chạy lệnh 'python' trực tiếp.")
        print(" Vui lòng sử dụng script wrapper 'isaaclab.sh' của Isaac Sim trên PC Ubuntu bằng lệnh sau:\n")
        print("   /home/nhglab/IsaacLab/isaaclab.sh -p src/hil/build_grasp_scene.py\n")
        sys.exit(1)

import omni
import numpy as np
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import create_new_stage, save_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim, define_prim
from pxr import UsdGeom, Gf, UsdPhysics

def main():
    # 1. Tạo Stage mới cho Scene
    create_new_stage()
    stage = omni.usd.get_context().get_stage()
    
    # Thiết lập hướng trục Z là trục hướng lên (Up-Axis)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    
    # 2. Định nghĩa các đường dẫn tệp USD của Arm và Hand
    ws_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ws_assets_dir = os.path.join(ws_dir, "data", "assets")
    
    ur5_usd_path = os.path.join(ws_assets_dir, "ur5only.usd")
    hand_usd_path = os.path.join(ws_assets_dir, "DexterousHandBase.usd")
    
    # Fallback nếu không tìm thấy ở workspace thì dùng đường dẫn cũ trên PC
    if not os.path.exists(ur5_usd_path):
        pc_ref_dir = "/home/nhglab/Tri/Seqhandisaac"
        ur5_usd_path = os.path.join(pc_ref_dir, "ur5only.usd")
        hand_usd_path = os.path.join(pc_ref_dir, "DexterousHandBase.usd")

    print(f"[INFO] Đang tải UR5 Arm từ: {ur5_usd_path}")
    print(f"[INFO] Đang tải Dexterous Hand từ: {hand_usd_path}")

    # 3. Add Reference UR5 vào Prim `/World/UR5`
    add_reference_to_stage(usd_path=ur5_usd_path, prim_path="/World/UR5")
    ur5_prim = stage.GetPrimAtPath("/World/UR5")
    if not ur5_prim.IsValid():
        print("[ERROR] Không thể load UR5 USD reference.")
        simulation_app.close()
        return

    # Gán tọa độ gốc cho UR5
    xform_ur5 = UsdGeom.Xformable(ur5_prim)
    xform_ur5.ClearXformOpOrder()
    xform_ur5.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform_ur5.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # 4. Tìm Prim khớp Wrist 3 của UR5 để làm điểm gắn Bàn tay
    # Thông thường đường dẫn prim cổ tay có dạng: /World/UR5/wrist_3_link hoặc tương tự
    wrist_prim_path = "/World/UR5/wrist_3_link"
    wrist_prim = stage.GetPrimAtPath(wrist_prim_path)
    
    if not wrist_prim.IsValid():
        # Fallback dò tìm wrist link
        for prim in stage.Traverse():
            if "wrist_3" in prim.GetName() or "wrist_3_link" in prim.GetName():
                wrist_prim_path = prim.GetPath().pathString
                wrist_prim = prim
                print(f"[INFO] Đã tìm thấy Wrist Prim tại: {wrist_prim_path}")
                break
                
    if not wrist_prim.IsValid():
        # Nếu vẫn không tìm thấy, gắn Hand trực tiếp vào gốc World và dịch chuyển tương đối
        wrist_prim_path = "/World"
        print("[WARN] Không tìm thấy wrist_3_link của UR5. Gắn Bàn tay trực tiếp vào /World.")

    # 5. Gắn Reference Bàn tay khéo léo (Dexterous Hand) vào dưới wrist link
    hand_prim_path = f"{wrist_prim_path}/DexterousHand"
    add_reference_to_stage(usd_path=hand_usd_path, prim_path=hand_prim_path)
    hand_prim = stage.GetPrimAtPath(hand_prim_path)
    
    if hand_prim.IsValid():
        # Căn chỉnh tư thế gá lắp của bàn tay (Translation & Rotation relative to wrist link)
        xform_hand = UsdGeom.Xformable(hand_prim)
        xform_hand.ClearXformOpOrder()
        # Dịch chuyển nhô ra theo trục nối của cổ tay (thường là Z hoặc Y)
        xform_hand.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.12)) # Dịch lên 12 cm
        xform_hand.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0)) # Xoay 90 độ để hướng lòng tay ra ngoài
        print("[SUCCESS] Đã lắp gá ráp bàn tay khéo léo thành công vào UR5.")
    else:
        print("[ERROR] Không thể load Dexterous Hand USD reference.")

    # 6. Tạo quả bóng (Sphere) phục vụ kịch bản 2 chụp bóng
    ball_prim_path = "/World/ball"
    create_prim(
        prim_path=ball_prim_path,
        prim_type="Sphere",
        position=np.array([0.18, 0.45, 0.85]),
        attributes={"radius": 0.035} # Bán kính bóng 3.5 cm (Đường kính 7cm giống bóng tennis)
    )
    ball_prim = stage.GetPrimAtPath(ball_prim_path)
    
    # Tô màu quả bóng màu đỏ cam nổi bật (Visual material)
    color_op = UsdGeom.Gprim(ball_prim).GetDisplayColorAttr()
    color_op.Set([Gf.Vec3f(1.0, 0.35, 0.0)]) # Orange-Red

    # Thêm đặc tính Vật lý Rigid Body (Kinematic) cho quả bóng để điều khiển vị trí
    physics_api = UsdPhysics.RigidBodyAPI.Apply(ball_prim)
    physics_api.CreateKinematicEnabledAttr().Set(True)

    # 7. Lưu lại file Stage hoàn chỉnh vào workspace
    save_path = os.path.join(ws_assets_dir, "grasp_scene.usd")
    save_stage(save_path)
    
    config_save_path = os.path.join(ws_dir, "config", "grasp_scene.usd")
    save_stage(config_save_path)
    
    print(f"\n[SUCCESS] Đã tạo và lưu thành công Scene lắp ráp mới tại: {save_path}")
    print(f"[SUCCESS] Đã sao chép Scene lắp ráp vào config: {config_save_path}")
    print(" Bạn có thể mở tệp USD này trực tiếp bằng Isaac Sim để kiểm tra visual.")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
