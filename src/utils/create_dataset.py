import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Đường dẫn thư mục chứa 74k file âm thanh gốc
INPUT_DIR = "/mnt/e/AI/khanh/audio_data/train"
OUTPUT_DIR = "/mnt/e/AI/khanh/audio_data/train_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def change_speed(input_path, output_path, speed_factor):
    # atempo dùng để đổi tốc độ phát âm (0.9 hoặc 1.1)
    cmd = [
        "ffmpeg", "-i", input_path, 
        "-filter:a", f"atempo={speed_factor}", 
        "-vn", output_path,
        "-loglevel", "error" # Ẩn bớt log cho đỡ rối màn hình
    ]
    subprocess.run(cmd)

def process_file(filename):
    if filename.endswith(".wav"):
        input_path = os.path.join(INPUT_DIR, filename)
        name, ext = os.path.splitext(filename)
        
        # Tạo file chậm 0.9x
        out_09 = os.path.join(OUTPUT_DIR, f"{name}_0.9{ext}")
        if not os.path.exists(out_09):
            change_speed(input_path, out_09, 0.9)
            
        # Tạo file nhanh 1.1x
        out_11 = os.path.join(OUTPUT_DIR, f"{name}_1.1{ext}")
        if not os.path.exists(out_11):
            change_speed(input_path, out_11, 1.1)

# Chạy đa luồng (Multi-threading) để xử lý nhanh 74k files
wav_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
print(f"Bắt đầu xử lý {len(wav_files)} files...")

with ThreadPoolExecutor(max_workers=20) as executor: # Chỉnh max_workers tùy số nhân CPU của bạn
    executor.map(process_file, wav_files)

print("Đã hoàn thành nhân bản dữ liệu!")