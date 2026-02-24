import os
import subprocess
import argparse

BASE_DIR = "/mnt/e/AI/khanh"
FAIRSEQ_DIR = os.path.join(BASE_DIR, "fairseq")
 
# Đường dẫn thư mục checkpoint
BASE_SOURCE_HIFI_GAN_TARGET = os.path.join(BASE_DIR, 'checkpoints/vocoder_target') # Model tiếng Việt (Custom)
BASE_TARGET_HIFI_GAN_SOURCE = os.path.join(BASE_DIR, 'checkpoints/vocoder_source') # Model tiếng Anh (Fairseq)

def infer(mode):
    print(f"\n>>> Đang khởi chạy suy luận chế độ: {mode.upper()}...")

    # 1. CẤU HÌNH CHO TARGET (TIẾNG VIỆT) - Dùng Code Custom
    if mode == "target":
        # Input/Output
        input_file = os.path.join(BASE_DIR, 'final/unit2wav/target/predicted_unit.txt')
        output_file = os.path.join(BASE_DIR, 'final/unit2wav/target/predicted_wav/result_vn.wav') # Thêm tên file .wav
        
        # Checkpoint (Custom)
        hifi_gan_ckpt = os.path.join(BASE_SOURCE_HIFI_GAN_TARGET, "g_00027000")
        config_file = os.path.join(BASE_SOURCE_HIFI_GAN_TARGET, "config.json")

        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Lệnh gọi (Sửa tên tham số cho khớp với infer.py custom)
        cmd = [
            "python", os.path.join(BASE_DIR, "speech-resynthesis/infer.py"),
            "--input_file", input_file,       
            "--output_file", output_file,     
            "--checkpoint_file", hifi_gan_ckpt, 
            "--config", config_file           
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # 2. CẤU HÌNH CHO SOURCE (TIẾNG ANH) - Dùng Code Fairseq 
    elif mode == "source":
        # Input/Output
        input_file = os.path.join(BASE_DIR, 'final/unit2wav/source/predicted_unit.txt') # Sửa đường dẫn cho đúng logic source
        output_file = os.path.join(BASE_DIR, 'final/unit2wav/source/predicted_wav/result_en.wav')

        # Checkpoint (Fairseq gốc)
        hifi_gan_ckpt = os.path.join(BASE_TARGET_HIFI_GAN_SOURCE, "g_00500000") 
        config_file = os.path.join(BASE_TARGET_HIFI_GAN_SOURCE, "config.json")
        
        # Đảm bảo thư mục output tồn tại
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Lệnh gọi 
        cmd = [
            "python", os.path.join(FAIRSEQ_DIR, "examples/speech_to_speech/generate_waveform_from_code.py"),
            "--in-code-file", input_file,
            "--vocoder", hifi_gan_ckpt,
            "--vocoder-cfg", config_file,
            "--results-path", output_file,
            "--dur-prediction" 
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print(f"\n--- XONG! File âm thanh đã lưu tại: {output_file} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer Unit to Wav Wrapper")
    parser.add_argument("--lang", choices=["source", "target"], help="Chọn 'source' (Tiếng Anh - Fairseq) hoặc 'target' (Tiếng Việt - Custom)")
    args = parser.parse_args()
    
    infer(args.lang)