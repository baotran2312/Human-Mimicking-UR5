import socket
import json

def main():
    # Khởi tạo Socket UDP Receiver
    UDP_IP = "0.0.0.0"  # Lắng nghe trên tất cả card mạng
    UDP_PORT = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print("\n==================================================================")
    print(f" UDP RECEIVER ĐANG LẮNG NGHE TRÊN CỔNG: {UDP_PORT}")
    print(" Chạy camera_stream.py từ laptop và trỏ IP về máy này để nhận dữ liệu.")
    print(" Nhấn Ctrl+C để THOÁT.")
    print("==================================================================\n")
    
    try:
        while True:
            data, addr = sock.recvfrom(4096)  # Nhận tối đa 4096 bytes
            try:
                payload = json.loads(data.decode("utf-8"))
                # In thông tin nhận được để kiểm tra
                timestamp = payload.get("timestamp", 0.0)
                arm_joints = payload.get("arm", [])
                hand_joints = payload.get("hand", {})
                
                print(f"[RECEIVED] From {addr[0]}:{addr[1]} | Time: {timestamp:.2f}")
                print(f"  UR5 Arm: {['{:.3f}'.format(x) for x in arm_joints]}")
                print(f"  Hand - Index: {['{:.3f}'.format(x) for x in hand_joints.get('index', [])]}")
                print(f"  Hand - Thumb: {['{:.3f}'.format(x) for x in hand_joints.get('thumb', [])]}")
                print("-" * 50)
            except json.JSONDecodeError:
                print(f"[WARN] Nhận dữ liệu không hợp lệ từ {addr}: {data}")
    except KeyboardInterrupt:
        print("\n[INFO] Đang đóng cổng nhận UDP.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
