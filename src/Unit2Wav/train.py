import os
import subprocess
import argparse
BASE_DIR = '/mnt/e/AI/khanh'

VOCODER_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "vocoder_target")
def train_vocoder(mode):
    """ Huấn luyện Vocoder từ Unit IDs 
    thưc hiện gắn thêm cờ dur-prediction để mô hình học cả duration và """
    print(f"\n>>> Huấn luyện Vocoder từ Unit IDs...")
    
    
    VOCODER_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", f"vocoder_{mode}")
    cmd = [
            "python", "speech-resynthesis/train.py",
            "--config", os.path.join(BASE_DIR, "src/Unit2Wav", "config.json"),
            "--checkpoint_path", VOCODER_CKPT_DIR,
            "--training_epochs", "300",
            "--stdout_interval", "50",
            "--checkpoint_interval", "2000",
            "--validation_interval", "5000"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"--- Done Vocoder Training: Checkpoints saved at {VOCODER_CKPT_DIR} ---")
if __name__ == "__main__":
    #  thuộc tính mode để xác định nguồn hoặc đích
    parser = argparse.ArgumentParser(description="Huấn luyện Vocoder từ Unit IDs")
    parser.add_argument("--lang", choices=["source", "target"], help="Chọn nguồn (en) hoặc đích (vn)")

    args = parser.parse_args()
    if args.lang == "source":
        train_vocoder("source")
    elif args.lang == "target":
        train_vocoder("target")