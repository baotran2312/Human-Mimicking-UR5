import argparse
import os
import sys

# Khởi chạy AppLauncher của Isaac Sim trước khi import PyTorch & Isaac Sim Modules
parser = argparse.ArgumentParser(description="Isaac Sim Assembly Scene Viewer")
parser.add_argument("--usd", type=str, default="config/grasp_scene.usd", help="USD scene file path")
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

if not HAS_ISAAC_SIM or simulation_app is None:
    print("\n[ERROR] Thư viện Isaac Sim (AppLauncher) không khả dụng khi chạy lệnh 'python' trực tiếp.")
    print(" Vui lòng sử dụng script wrapper 'isaaclab.sh' của Isaac Sim trên PC Ubuntu bằng lệnh sau:\n")
    print("   /home/nhglab/IsaacLab/isaaclab.sh -p src/hil/run_grasp_scene.py\n")
    sys.exit(1)

import omni
import omni.usd

def main():
    usd_path = args_cli.usd
    if not os.path.exists(usd_path):
        usd_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "assets", "grasp_scene.usd")
    usd_path = os.path.abspath(usd_path)

    print(f"\n==================================================================")
    print(f" ISAAC SIM SCENE VIEWER: ĐANG MỞ TỆP USD DỰ ÁN")
    print(f" Đường dẫn file: {usd_path}")
    print("==================================================================\n")

    # Mở stage USD bằng API chuẩn của Omniverse Kit USD Context
    omni.usd.get_context().open_stage(usd_path)

    print("[INFO] Scene 3D đã được tải thành công. Đang giữ cửa sổ Isaac Sim mở...")
    print(" Nhấn Ctrl+C trong Terminal hoặc ĐÓNG CỬA SỔ Isaac Sim để thoát.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[INFO] Đang đóng cửa sổ xem Scene...")
    finally:
        simulation_app.close()
        print("[INFO] Đã đóng tài nguyên hoàn tất.")

if __name__ == "__main__":
    main()
