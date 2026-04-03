import os
import subprocess
import torch
import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_TARGET = '/mnt/e/AI/khanh/checkpoints'
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit_dedup")
CHECKPOINT_DIR = os.path.join(BASE_TARGET, "unit2unit_dedup_v2")

def start_training_from_scratch_optimized():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Chuẩn bị Training Unit2Unit từ đầu (Nâng cấp).")
    print(f"Checkpoints lưu tại: {CHECKPOINT_DIR}")

    cmd = [
        "fairseq-train", DATA_BIN,
        "--task", "translation",
        "--arch", "transformer", 
        "--share-decoder-input-output-embed",
        
        # 1. FIX LỖI TRÀN RAM HỆ THỐNG (BẮT BUỘC)
        "--num-workers", "0",
        
        # --- CẤU HÌNH INPUT ---
        "--max-source-positions", "1536",
        "--max-target-positions", "1536",
        "--skip-invalid-size-inputs-valid-test",
        
        # 2. TỐI ƯU CHO RTX 3090 (TÙY CHỌN NHƯNG KHUYÊN DÙNG)
        # Tăng max-tokens để tận dụng 24GB VRAM, giảm update-freq để giữ nguyên Effective Batch Size
        "--max-tokens", "4096",    # Cũ: 1536
        "--update-freq", "6",      # Cũ: 16 (4096 * 6 ~ 24,576 tokens/update)
        
        "--fp16", 
        
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--adam-eps", "1e-8",
        "--clip-norm", "0.1",
        
        "--lr", "5e-4",
        
        # 3. GIẢM WARMUP ĐỂ MODEL HỘI TỤ NHANH HƠN (TÙY CHỌN)
        # 4000 updates ~ 4 epochs đầu tiên thay vì 8.5 epochs như cũ
        "--warmup-updates", "4000", # Cũ: 8000
        
        "--lr-scheduler", "inverse_sqrt",
        
        # NỚI LỎNG CÁC RÀNG BUỘC ĐỂ MÔ HÌNH HỌC VÀO
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
        
        # --- METRIC ---
        "--best-checkpoint-metric", "loss",

        "--tensorboard-logdir", os.path.join(CHECKPOINT_DIR, "tb_logs"),
    ]

    print("Đang chạy lệnh train from scratch...")
    
    log_path = os.path.join(CHECKPOINT_DIR, "train_scratch.log")
    
    try:
        with open(log_path, "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"TRAINING UNIT2UNIT BIG FROM SCRATCH: {timestamp}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"{'='*80}\n\n")
            log_file.flush()
            
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n LỆNH TRAIN BỊ LỖI (Exit Code: {e.returncode})")
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                print("\n 50 dòng cuối của log để debug:")
                print("".join(lines[-50:]))
        except:
            pass

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    start_training_from_scratch_optimized()