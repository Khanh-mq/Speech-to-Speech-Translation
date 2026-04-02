# import os
# import shutil
# import subprocess

# # --- CẤU HÌNH ĐƯỜNG DẪN ---
# BASE_DATA = '/mnt/g/khanh'
# BASE_TARGET = '/mnt/e/AI/khanh/checkpoints'
# FEAT_DIR = os.path.join(BASE_DATA, "hubert_feats")

# # Thư mục mới chứa data chuẩn (Asym = Asymmetric)
# DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit_Asym") 

# def deduplicate_units(line):
#     """Hàm nén: '15 15 15 20' -> '15 20'"""
#     units = line.strip().split()
#     if not units: return ""
#     dedup_units = [units[0]]
#     for u in units[1:]:
#         if u != dedup_units[-1]:
#             dedup_units.append(u)
#     return " ".join(dedup_units)

# def process_source_file(input_path, output_path):
#     """TIẾNG ANH (SOURCE): Bắt buộc nén để Transformer hiểu nghĩa"""
#     print(f"  -> RÚT GỌN (Deduplicating) Tiếng Anh: {os.path.basename(input_path)}")
#     with open(input_path, 'r', encoding='utf-8') as fin, \
#          open(output_path, 'w', encoding='utf-8') as fout:
#         for line in fin:
#             fout.write(deduplicate_units(line) + "\n")

# def process_target_file(input_path, output_path):
#     """TIẾNG VIỆT (TARGET): Bắt buộc giữ nguyên để Vocoder đọc đúng nhịp điệu"""
#     print(f"  -> COPY NGUYÊN BẢN (Raw) Tiếng Việt: {os.path.basename(input_path)}")
#     shutil.copy(input_path, output_path)

# def prepare_data():
#     temp_dir = "temp_corpus_asym"
#     os.makedirs(temp_dir, exist_ok=True)

#     print("\n--- BƯỚC 1: Xử lý Dữ liệu Bất Đối Xứng (Asymmetric) ---")
    
#     # Tập TRAIN
#     process_source_file(os.path.join(FEAT_DIR, "en/train_0_1.km"), os.path.join(temp_dir, "train.src"))
#     process_source_file(os.path.join(BASE_DATA, "kmean500/train_0_1.km"), os.path.join(temp_dir, "train.tgt"))
    
#     # Tập VALID
#     process_source_file(os.path.join(FEAT_DIR, "en/valid_0_1.km"), os.path.join(temp_dir, "valid.src"))
#     process_source_file(os.path.join(BASE_DATA, "kmean500/valid_0_1.km"), os.path.join(temp_dir, "valid.tgt"))
    
#     print("\n--- BƯỚC 2: Đang chạy fairseq-preprocess (Binarize) ---")
#     cmd = [
#         "fairseq-preprocess",
#         "--source-lang", "src",
#         "--target-lang", "tgt",
#         "--trainpref", os.path.join(temp_dir, "train"),
#         "--validpref", os.path.join(temp_dir, "valid"),
#         "--destdir", DATA_BIN,
#         "--workers", "4"
#     ]
#     subprocess.run(cmd, check=True)
    
#     shutil.rmtree(temp_dir) # Dọn rác
#     print(f"\nHoàn thành! Dữ liệu chuẩn đã lưu tại: {DATA_BIN}")

# if __name__ == "__main__":
#     prepare_data()




import os
import shutil
import subprocess
import itertools

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DATA = '/mnt/g/khanh'
BASE_TARGET = '/mnt/e/AI/khanh/checkpoints'
FEAT_DIR = os.path.join(BASE_DATA, "hubert_feats")

# Thư mục chứa data chuẩn (Dedup + Duration)
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit_Dedup_Dur") 

def deduplicate_source(line):
    """TIẾNG ANH: Chỉ cần cắt gọt, không cần quan tâm duration"""
    units = line.strip().split()
    if not units: return ""
    dedup_units = [k for k, g in itertools.groupby(units)]
    return " ".join(dedup_units)

def extract_dedup_and_duration_target(line):
    """TIẾNG VIỆT: Vừa cắt gọt, vừa đếm số lần lặp để làm nhãn Duration"""
    units = line.strip().split()
    if not units: return "", ""
    
    dedup_units = []
    durations = []
    
    # Hàm groupby siêu việt của Python sẽ gom các unit giống nhau đứng cạnh nhau
    for unit, group in itertools.groupby(units):
        dedup_units.append(unit)
        durations.append(str(len(list(group)))) # Đếm số lượng lặp
        
    return " ".join(dedup_units), " ".join(durations)

def process_source_file(input_path, output_path):
    print(f"  -> RÚT GỌN (Deduplicating) Tiếng Anh: {os.path.basename(input_path)}")
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(deduplicate_source(line) + "\n")

def process_target_file(input_path, output_tgt_path, output_dur_path):
    print(f"  -> RÚT GỌN & TRÍCH XUẤT DURATION Tiếng Việt: {os.path.basename(input_path)}")
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_tgt_path, 'w', encoding='utf-8') as ftgt, \
         open(output_dur_path, 'w', encoding='utf-8') as fdur:
        for line in fin:
            tgt_str, dur_str = extract_dedup_and_duration_target(line)
            ftgt.write(tgt_str + "\n")
            fdur.write(dur_str + "\n")

def prepare_data():
    temp_dir = "temp_corpus_dedup_dur"
    os.makedirs(temp_dir, exist_ok=True)

    print("\n--- BƯỚC 1: Xử lý Dữ liệu (Cắt gọt cả 2 chiều + Lấy Duration) ---")
    
    # Xử lý tập TRAIN
    process_source_file(os.path.join(FEAT_DIR, "en/train_0_1.km"), 
                        os.path.join(temp_dir, "train.src"))
    process_target_file(os.path.join(BASE_DATA, "kmean500/train_0_1.km"), 
                        os.path.join(temp_dir, "train.tgt"),
                        os.path.join(temp_dir, "train.dur")) # File nhãn độ dài
    
    # Xử lý tập VALID
    process_source_file(os.path.join(FEAT_DIR, "en/valid_0_1.km"), 
                        os.path.join(temp_dir, "valid.src"))
    process_target_file(os.path.join(BASE_DATA, "kmean500/valid_0_1.km"), 
                        os.path.join(temp_dir, "valid.tgt"),
                        os.path.join(temp_dir, "valid.dur")) # File nhãn độ dài
    
    print("\n--- BƯỚC 2: Đang chạy fairseq-preprocess (Binarize) ---")
    # Fairseq preprocess sẽ đóng gói file .src và .tgt thành định dạng nhị phân (.bin, .idx)
    cmd = [
        "fairseq-preprocess",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--trainpref", os.path.join(temp_dir, "train"),
        "--validpref", os.path.join(temp_dir, "valid"),
        "--destdir", DATA_BIN,
        "--workers", "4"
    ]
    subprocess.run(cmd, check=True)
    
    print("\n--- BƯỚC 3: Copy các file Duration vào thư mục DATA_BIN ---")
    # Fairseq không biết đóng gói file .dur, nên ta phải copy thủ công vào DATA_BIN
    # để lát nữa file custom_task.py của chúng ta tự đọc.
    shutil.copy(os.path.join(temp_dir, "train.dur"), os.path.join(DATA_BIN, "train.dur"))
    shutil.copy(os.path.join(temp_dir, "valid.dur"), os.path.join(DATA_BIN, "valid.dur"))
    print(f"  -> Đã copy train.dur và valid.dur vào {DATA_BIN}")
    
    shutil.rmtree(temp_dir) # Dọn rác
    print(f"\nHoàn thành! Dữ liệu chuẩn (có Duration) đã lưu tại: {DATA_BIN}")

if __name__ == "__main__":
    prepare_data()