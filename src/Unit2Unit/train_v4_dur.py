import os
import subprocess
import torch
import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_TARGET = '/mnt/e/AI/khanh/checkpoints'
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit_Dedup_Dur") 
CHECKPOINT_DIR = os.path.join(BASE_TARGET, "unit2unit_Dedup_Dur_v1") 

# Trỏ đến thư mục chứa các file custom của Fairseq
USER_DIR = "./custom_u2u" 

def start_training():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Bắt đầu Training Unit2Unit - Mô hình ĐỐI XỨNG (Dedup) với Duration Predictor (Base Arch).")

    cmd = [
        "fairseq-train", 
        "--user-dir", USER_DIR, 
        DATA_BIN,
        "--task", "translation_with_duration",  
        "--arch", "transformer_base_with_duration",  # Đã hạ xuống Base chống Overfitting
        "--share-decoder-input-output-embed",
        
        # --- BỘ NHỚ CHIỀU DÀI ---
        "--max-source-positions", "1024",
        "--max-target-positions", "1024",
        "--skip-invalid-size-inputs-valid-test",
        
        # --- BATCHING & PHẦN CỨNG ---
        "--max-tokens", "4096",    
        "--update-freq", "8",      
        "--fp16",                  # Sử dụng Bfloat16 siêu xịn của RTX 3090 chống tràn số
        
        # --- OPTIMIZATION (Đã tinh chỉnh chống sốc Gradient) ---
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--adam-eps", "1e-8",
        "--clip-norm", "0.5",      # Giảm từ 1.0 xuống 0.5
        "--lr", "3e-4",            # Giảm từ 5e-4 xuống 3e-4
        "--warmup-updates", "4000",
        "--lr-scheduler", "inverse_sqrt",
        
        # --- REGULARIZATION (Đã tinh chỉnh chống Overfitting) ---
        "--dropout", "0.2",                  # Xóa chữ 'n' thừa, tăng từ 0.1 lên 0.2
        "--attention-dropout", "0.1",        # Thêm mới
        "--weight-decay", "0.001",           # Tăng từ 0.0001 lên 0.001
        "--label-smoothing", "0.1",       
        "--criterion", "unit_and_duration_loss",  
        
        # --- ĐIỀU KHIỂN & LƯU TRỮ ---
        "--save-dir", CHECKPOINT_DIR,
        "--max-epoch", "200",
        "--patience", "30",
        "--keep-last-epochs", "5",
        "--keep-best-checkpoints", "3",
        "--log-interval", "50",
        "--save-interval", "10",              # Đổi thành 1 để lưu sát sao epoch tốt nhất
        "--validate-interval", "1",
        "--best-checkpoint-metric", "loss", 
        "--tensorboard-logdir", os.path.join(CHECKPOINT_DIR, "tb_logs"),
    ]

    log_path = os.path.join(CHECKPOINT_DIR, "train_dedup_dur.log")
    
    try:
        with open(log_path, "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"TRAIN SYMMETRIC DEDUP WITH DURATION: {timestamp}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"{'='*80}\n\n")
            log_file.flush()
            
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nLỆNH TRAIN BỊ LỖI (Exit Code: {e.returncode})")

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    start_training()