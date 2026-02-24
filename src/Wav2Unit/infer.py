# import os
# import subprocess
# import argparse


# FAIRSEQ_DIR = "/mnt/e/AI/khanh/fairseq"
# BASE_DIR = '/mnt/e/AI/khanh'
# HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
# BASE_TARGET = os.path.join(BASE_DIR, "final", "wav2unit")
# def single_infer(lang_mode):
#     """
#         Biến 1 file .wav thành các Unit IDs
#     """
#     input = os.path.join(BASE_TARGET, lang_mode, "input/input.wav")
#     # 1. Tạo thư mục tạm để chứa manifest của 1 file này
#     temp_dir = os.path.join(BASE_TARGET, lang_mode, "temp_infer")
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # 2. Tạo manifest cho duy nhất 1 file âm thanh đó
#     temp_wav_folder = os.path.join(temp_dir, "wav_input")
#     os.makedirs(temp_wav_folder, exist_ok=True)
#     import shutil
#     shutil.copy(input, os.path.join(temp_wav_folder, "input.wav"))
    
#     print("--- 1. Tạo Manifest cho file đầu vào ---")
#     subprocess.run([
#         "python", os.path.join(FAIRSEQ_DIR, "examples/wav2vec/wav2vec_manifest.py"),
#         temp_wav_folder, "--dest", temp_dir, "--ext", "wav"
#     ], check=True)

#     # 3. Trích xuất Features
#     print("--- 2. Trích xuất Hubert Features ---")
#     subprocess.run([
#         "python", os.path.join(FAIRSEQ_DIR, "examples/hubert/simple_kmeans/dump_hubert_feature.py"),
#         temp_dir, "train", HUBERT_CKPT, "11", "1", "0", temp_dir
#     ], check=True)

#     if lang_mode == "source":
#         km_model = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")
#     else:
#         km_model = os.path.join(BASE_DIR, "kmeans/kmeans_vn_1000.bin")

#     # 4. Quantize sang Unit
#     print("--- 3. Chuyển đổi sang Unit IDs ---")
#     subprocess.run([
#         "python", os.path.join(FAIRSEQ_DIR, "examples/hubert/simple_kmeans/dump_km_label.py"),
#         temp_dir, "train", km_model, "1", "0", temp_dir
#     ], check=True)

#     # 5. Đọc kết quả
#     km_file = os.path.join(temp_dir, "train.km")
#     if not os.path.exists(km_file):
#         raise FileNotFoundError(f"File {km_file} không tồn tại!")
#     with open(km_file, 'r') as f:
#         units = f.read().strip()
    
#     print(f"\n[KẾT QUẢ] Unit IDs của file {input}:")
#     print(units)
    
#     # Dọn dẹp thư mục tạm (tùy chọn)
#     # shutil.rmtree(temp_dir)
#     return units

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lang", required=True, choices=["source", "target"], help="Ngôn ngữ đầu vào (source hoặc target)")
#     args = parser.parse_args()
    
#     single_infer( args.lang)



import os
import subprocess
import argparse
import shutil

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = '/mnt/e/AI/khanh'
# SỬA 1: Trỏ đúng vào thư mục code Fairseq
FAIRSEQ_CODE_DIR = os.path.join(BASE_DIR, "fairseq") 

HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
BASE_TARGET = os.path.join(BASE_DIR, "final", "wav2unit")

def single_infer(lang_mode):
    print(f"\n>>> BẮT ĐẦU: Chuyển đổi Wav sang Unit cho {lang_mode.upper()}")
    
    # Đường dẫn file input/output
    input_wav = os.path.join(BASE_TARGET, lang_mode, "input/input.wav")
    output_unit_file = os.path.join(BASE_TARGET, lang_mode, "predicted_unit.txt") # File kết quả để dùng sau này

    if not os.path.exists(input_wav):
        print(f"LỖI: Không tìm thấy file đầu vào tại {input_wav}")
        return

    # 1. Tạo môi trường tạm
    temp_dir = os.path.join(BASE_TARGET, lang_mode, "temp_infer")
    temp_wav_folder = os.path.join(temp_dir, "wav_input")
    
    # Xóa tạm cũ nếu có để tránh lẫn lộn
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_wav_folder, exist_ok=True)
    
    # Copy file wav vào folder tạm (đổi tên thành input.wav cho gọn)
    shutil.copy(input_wav, os.path.join(temp_wav_folder, "input.wav"))

    try:
        # 2. Tạo Manifest (wav2vec_manifest.py)
        print("--- Bước 1: Tạo Manifest ---")
        cmd_manifest = [
            "python", os.path.join(FAIRSEQ_CODE_DIR, "examples/wav2vec/wav2vec_manifest.py"),
            temp_wav_folder, 
            "--dest", temp_dir, 
            "--ext", "wav",
            "--valid-percent", "0" # Chỉ cần train set
        ]
        subprocess.run(cmd_manifest, check=True, stdout=subprocess.DEVNULL)

        # 3. Trích xuất Features (dump_hubert_feature.py)
        print("--- Bước 2: Trích xuất Hubert Features (Layer 11) ---")
        cmd_feature = [
            "python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_hubert_feature.py"),
            temp_dir,      # Thư mục chứa tsv
            "train",       # Split name (do manifest tạo ra train.tsv)
            HUBERT_CKPT,   # Checkpoint mHubert
            "11",          # Layer 11
            "1", "0",      # nshard, rank
            temp_dir       # Nơi lưu features (.npy)
        ]
        subprocess.run(cmd_feature, check=True) # Bỏ stdout=DEVNULL để xem tiến trình nếu cần

        # 4. Quantize sang Unit (dump_km_label.py)
        print("--- Bước 3: Áp dụng K-means để ra Unit IDs ---")
        
        # Chọn K-means model dựa trên ngôn ngữ
        if lang_mode == "source": # Tiếng Anh -> Dùng mHubert Kmeans
            km_model = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")
        else: # Tiếng Việt -> Dùng Kmeans tự train
            km_model = os.path.join(BASE_DIR, "kmeans/kmeans_vn_1000.bin")
            
        cmd_km = [
            "python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_km_label.py"),
            temp_dir,   # Thư mục chứa features (.npy)
            "train",    # Split name
            km_model,   # File Kmeans .bin
            "1", "0",   # nshard, rank
            temp_dir    # Nơi lưu kết quả (.km)
        ]
        subprocess.run(cmd_km, check=True)

        # 5. Đọc và Lưu kết quả
        km_output_file = os.path.join(temp_dir, "train_0_1.km")
        if os.path.exists(km_output_file):
            with open(km_output_file, 'r') as f:
                units = f.read().strip()
            
            # --- QUAN TRỌNG: Lưu ra file .txt để bước Dịch (S2UT) dùng ---
            with open(output_unit_file, 'w') as f_out:
                f_out.write(units)
                
            print(f"\n[THÀNH CÔNG] Đã lưu Unit IDs vào: {output_unit_file}")
            print(f"Sample: {units[:50]}...")
        else:
            print("LỖI: Không tìm thấy file output train.km")

    except subprocess.CalledProcessError as e:
        print(f"\n[LỖI] Quá trình chạy bị dừng lại tại một bước nào đó.")
        print(e)
    finally:
        # Dọn dẹp thư mục tạm để tiết kiệm chỗ (Mở lại comment nếu muốn xóa)
        # shutil.rmtree(temp_dir)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["source", "target"], help="Ngôn ngữ: source (Anh) hoặc target (Việt)")
    args = parser.parse_args()
    
    single_infer(args.lang)