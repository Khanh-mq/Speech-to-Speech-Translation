import os
import subprocess
import torch
import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN MỚI ---
BASE_TARGET = '/mnt/e/AI/khanh/checkpoints'
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit_Asym") # Trỏ vào bộ data vừa tạo
CHECKPOINT_DIR = os.path.join(BASE_TARGET, "unit2unit_BIG_Asym") # Thư mục lưu model mới

def start_training():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Bắt đầu Training Unit2Unit - Mô hình BẤT ĐỐI XỨNG.")

    cmd = [
        "fairseq-train", DATA_BIN,
        "--task", "translation",
        "--arch", "transformer_wmt_en_de_big", 
        "--share-decoder-input-output-embed",
        
        # --- BỘ NHỚ CHIỀU DÀI ---
        # Vì Tiếng Việt giữ nguyên nên chuỗi sẽ dài, cần tăng max-positions
        "--max-source-positions", "2048",
        "--max-target-positions", "2048",
        "--skip-invalid-size-inputs-valid-test",
        
        "--max-tokens", "2500",    
        "--update-freq", "16",     
        "--fp16", 
        
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--adam-eps", "1e-8",
        "--clip-norm", "1.0", # Cắt Gradient lớn hơn 1.0 để chống nổ
        
        "--lr", "5e-4",
        "--warmup-updates", "4000",
        "--lr-scheduler", "inverse_sqrt",
        
        "--dropout", "0.1",               
        "--weight-decay", "0.0001",        
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1",       
        
        # --- ĐIỀU KHIỂN & LƯU TRỮ ---
        "--save-dir", CHECKPOINT_DIR,
        "--max-epoch", "200",
        "--patience", "30",
        "--keep-last-epochs", "5",
        "--keep-best-checkpoints", "3",
        "--log-interval", "50",
        "--save-interval", "30",
        "--validate-interval", "1",
        "--best-checkpoint-metric", "loss",
        "--tensorboard-logdir", os.path.join(CHECKPOINT_DIR, "tb_logs"),
    ]

    log_path = os.path.join(CHECKPOINT_DIR, "train_asym.log")
    
    try:
        with open(log_path, "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"TRAIN ASYMMETRIC (DEDUP SRC -> RAW TGT): {timestamp}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"{'='*80}\n\n")
            log_file.flush()
            
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nLỆNH TRAIN BỊ LỖI (Exit Code: {e.returncode})")

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    start_training()