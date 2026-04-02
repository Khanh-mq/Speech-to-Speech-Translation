import os
import random
import argparse

# --- CẤU HÌNH ---
BASE_DIR = "/mnt/e/AI/khanh"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "src/Unit2Wav/processed_data") 

# Cấu hình dữ liệu đầu vào (Source & Target)
DATASETS = {
    "target": { # Tiếng Việt
        "tsv": '/mnt/g/khanh/manifest_temp/target/train.tsv',
        "km": '/mnt/g/khanh/kmean500/train_0_1.km',
        "wav_root": "/mnt/g/data_final/data/target/train", 
    },
    "source": { # Tiếng Anh 
        "tsv": '/mnt/g/khanh/manifest_temp/source/train.tsv',
        "km": '/mnt/g/khanh/hubert_feats/en/train_0_1.km',
        "wav_root": "/mnt/g/data_final/data/source/train", 
    }
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_dataset(lang):
    print(f"\n================ XỬ LÝ DỮ LIỆU: {lang.upper()} (FAIRSEQ FORMAT) ================")
    
    cfg = DATASETS[lang]
    output_dir = os.path.join(OUTPUT_BASE_DIR, lang)
    ensure_dir(output_dir)

    # 1. Đọc và Ghép nối
    print(f"--> Đang đọc file TSV và KM...")
    merged_data = []
    
    try:
        with open(cfg['tsv'], 'r', encoding='utf-8') as f_tsv, \
             open(cfg['km'], 'r', encoding='utf-8') as f_km:
            
            root_in_tsv = f_tsv.readline().strip() # Bỏ qua header
            
            for line_tsv, line_km in zip(f_tsv, f_km):
                # Lấy tên file
                filename = line_tsv.split('\t')[0]
                if filename.endswith('.wav'):
                    filename = filename[:-4] # Bỏ đuôi wav tạm thời
                
                units = line_km.strip()
                merged_data.append((filename, units))
                
    except FileNotFoundError as e:
        print(f"[LỖI] Không tìm thấy file: {e}")
        return

    print(f"--> Tìm thấy tổng: {len(merged_data)} cặp.")

    # 2. Xáo trộn và Chia tập (30k / 1k)
    print(f"--> Đang xáo trộn và chia tập...")
    random.seed(42) 
    random.shuffle(merged_data)

    train_set = merged_data[:50000]
    val_set = merged_data[50000:52000]

    # 3. Hàm ghi file chuẩn Dictionary
    def write_fairseq_manifest(data_list, output_filename):
        out_path = os.path.join(output_dir, output_filename)
        
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for filename, units in data_list:
                # 1. Tạo đường dẫn tuyệt đối
                full_path = os.path.join(cfg['wav_root'], filename + ".wav")
                
                # 2. Tạo Dictionary (Format bạn yêu cầu)
                # Lưu ý: Code gốc của Fairseq thường dùng ast.literal_eval để đọc dòng này
                data_dict = {
                    "audio": full_path,
                    "hubert": units, # Hoặc key là 'code' tùy dataset của bạn
                    "duration": len(units.split()) * 0.02 # Ước lượng 20ms/frame
                }
                
                # 3. Ghi ra file (Convert Dict -> String)
                f_out.write(str(data_dict) + "\n")
                
        print(f"--> Đã lưu: {out_path} ({len(data_list)} dòng)")
        return out_path

    # Thực hiện ghi
    path_train = write_fairseq_manifest(train_set, "train.manifest")
    path_val = write_fairseq_manifest(val_set, "valid.manifest")

    print(f"\n--> Cấu hình mẫu cho config.json:")
    print(f'    "training_files": "{path_train}",')
    print(f'    "validation_files": "{path_val}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xử lý dữ liệu Unit2Wav")
    parser.add_argument("--lang", choices=["source", "target", "both"], default="both",
                        help="Chọn ngôn ngữ để xử lý: 'source' (Anh), 'target' (Việt), hoặc 'both' (Cả hai)")
    args = parser.parse_args()
    
    
    if args.lang == "both":
        process_dataset("target") # Xử lý tiếng Việt
        process_dataset("source") # Xử lý tiếng Anh
    else:
        process_dataset(args.lang)