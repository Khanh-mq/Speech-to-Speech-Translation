import os
import subprocess
import torch
import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_TARGET = '/mnt/e/AI/khanh/checkpoints'
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit")
CHECKPOINT_DIR = os.path.join(BASE_TARGET, "unit2unit") 


def resume_training_optimized():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_last.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Lỗi: Không tìm thấy file {checkpoint_path}!")
        return

    print(f"Đã tìm thấy checkpoint. Chuẩn bị Fine-tuning từ: {checkpoint_path}")

    cmd = [
        "fairseq-train", DATA_BIN,
        "--task", "translation",
        "--arch", "transformer",
        "--share-decoder-input-output-embed",
        
        # Dùng finetune-from-model thay vì restore-file
        "--finetune-from-model", checkpoint_path,
        
        # --- CẤU HÌNH INPUT ---
        "--max-source-positions", "1024",
        "--max-target-positions", "1024",
        "--skip-invalid-size-inputs-valid-test",
        
        # --- TỐI ƯU HÓA ---
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--adam-eps", "1e-8",
        "--clip-norm", "0.0",
        
        # --- FINE-TUNING CONFIG ---
        "--lr", "1e-4",
        # "--min-lr", "1e-6",
        "--warmup-updates", "1000",
        "--lr-scheduler", "inverse_sqrt",
        
        "--dropout", "0.1",
        "--weight-decay", "0.0001",
        
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1",
        "--max-tokens", "4096",
        "--update-freq", "4",
        "--fp16",
        
        "--save-dir", CHECKPOINT_DIR,
        "--max-epoch", "200",
        "--patience", "50",
        "--keep-last-epochs", "10",
        "--keep-best-checkpoints", "5",
        "--log-interval", "50",
        "--save-interval", "5",
        "--validate-interval", "1",
        
        # --- METRIC ---
        "--eval-bleu",
        "--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 200}',
        "--best-checkpoint-metric", "loss",
        
        # KHÔNG cần reset vì finetune-from-model không load optimizer state
    ]

    print("Đang chạy lệnh train fine-tune...")
    
    log_path = os.path.join(CHECKPOINT_DIR, "train_finetune.log")
    
    try:
        with open(log_path, "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"FINE-TUNING WITH --finetune-from-model: {timestamp}\n")
            log_file.write(f"Base Checkpoint: {checkpoint_path}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"{'='*80}\n\n")
            log_file.flush()
            
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  LỆNH TRAIN BỊ LỖI (Exit Code: {e.returncode})")
        
        # Đọc lỗi từ log
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                print("\n📝 50 dòng cuối của log:")
                print("".join(lines[-50:]))
        except:
            pass


if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    resume_training_optimized()