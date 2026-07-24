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
    ur5dex_usd_path = os.path.join(ws_assets_dir, "ur5dex.usd")
    
    if os.path.exists(ur5dex_usd_path):
        print(f"[INFO] Tải Robot UR5 + Bàn tay khéo léo hoàn chỉnh từ: {ur5dex_usd_path}")
        add_reference_to_stage(usd_path=ur5dex_usd_path, prim_path="/World/UR5")
    else:
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

        # 4. Tìm chính xác Prim Link cổ tay cuối cùng (wrist_3_link - LOẠI TRỪ JOINT)
        wrist_link_prim = None
        wrist_link_path = None
        
        # Danh sách ưu tiên các đường dẫn chuẩn tới link cuối của UR5
        for candidate in ["/World/UR5/ur5/wrist_3_link", "/World/UR5/wrist_3_link"]:
            p = stage.GetPrimAtPath(candidate)
            if p.IsValid():
                wrist_link_prim = p
                wrist_link_path = candidate
                break

        if not wrist_link_prim:
            # Fallback dò tìm prim là Link chứa 'wrist_3_link' hoặc 'flange' (LOẠI TRỪ PRIM JOINT)
            for prim in stage.Traverse():
                name = prim.GetName().lower()
                if ("wrist_3_link" in name or "flange" in name) and "joint" not in name:
                    wrist_link_prim = prim
                    wrist_link_path = prim.GetPath().pathString
                    break

        if wrist_link_prim:
            print(f"[SUCCESS] Đã tìm thấy Wrist 3 Link Prim tại: {wrist_link_path}")
            hand_prim_path = f"{wrist_link_path}/DexterousHand"
            add_reference_to_stage(usd_path=hand_usd_path, prim_path=hand_prim_path)
            hand_prim = stage.GetPrimAtPath(hand_prim_path)
            if hand_prim.IsValid():
                xform_hand = UsdGeom.Xformable(hand_prim)
                xform_hand.ClearXformOpOrder()
                xform_hand.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                xform_hand.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                print(f"[SUCCESS] Đã gắn Bàn tay khéo léo vào đúng link cuối: {hand_prim_path}")
        else:
            print("[WARN] Không tìm thấy wrist_3_link. Gắn Bàn tay trực tiếp vào /World.")
            add_reference_to_stage(usd_path=hand_usd_path, prim_path="/World/DexterousHand")

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
