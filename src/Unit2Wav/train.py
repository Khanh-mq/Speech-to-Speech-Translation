import os
import subprocess
import argparse
import sys

BASE_DIR = '/mnt/e/AI/khanh'

def train_vocoder(mode):
    """ Huấn luyện Vocoder từ Unit IDs 
    thưc hiện gắn thêm cờ dur-prediction để mô hình học cả duration """
    print(f"\n>>> Huấn luyện Vocoder từ Unit IDs cho: {mode.upper()}...")
    
    VOCODER_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", f"vocoder_{mode}_kmean500")
    
    # Tạo thư mục checkpoint trước nếu chưa có, để tránh lỗi khi tạo file log
    os.makedirs(VOCODER_CKPT_DIR, exist_ok=True)
    
    # Định nghĩa file log
    log_file_path = os.path.join(VOCODER_CKPT_DIR, f"train_vocoder_{mode}.log")
    print(f">>> Toàn bộ log sẽ được lưu tại: {log_file_path}")

    cmd = [
            "python", "speech-resynthesis/train.py",
            "--config", os.path.join(BASE_DIR, "src/Unit2Wav", "config.json"),
            "--checkpoint_path", VOCODER_CKPT_DIR,
            "--training_epochs", "300",
            "--stdout_interval", "50",
            "--checkpoint_interval", "10000",
            "--validation_interval", "10000",
            # "--dur_prediction", 
    ]
    
    # Mở file log ở chế độ 'append' hoặc 'write'
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        # Dùng Popen để bắt output real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, # Gộp chung Error vào Output
            text=True, 
            bufsize=1
        )
        
        # Đọc từng dòng log được sinh ra
        for line in process.stdout:
            # 1. In ra màn hình terminal để theo dõi
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # 2. Ghi vào file log
            log_file.write(line)
            log_file.flush()
            
        # Đợi process chạy xong và lấy mã lỗi (nếu có)
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[LỖI] Tiến trình train bị dừng đột ngột với mã lỗi: {process.returncode}")
        else:
            print(f"\n--- Done Vocoder Training: Checkpoints saved at {VOCODER_CKPT_DIR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Vocoder từ Unit IDs")
    parser.add_argument("--lang", choices=["source", "target"], help="Chọn nguồn (en) hoặc đích (vn)")

    args = parser.parse_args()
    if args.lang:
        train_vocoder(args.lang)
    else:
        print("Vui lòng truyền tham số --lang")