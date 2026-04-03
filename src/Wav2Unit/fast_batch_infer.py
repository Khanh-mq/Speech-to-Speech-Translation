import os
import subprocess
import argparse
import shutil
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN TỔNG QUÁT ---
KMEAN= 500
BASE_DIR = '/mnt/e/AI/khanh'
FAIRSEQ_CODE_DIR = os.path.join(BASE_DIR, "fairseq") 
HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
BASE_TARGET = os.path.join(BASE_DIR, "final", "wav2unit")

def ultra_fast_batch_infer(lang_mode, input_wav_dir, out_npy_dir, out_km_dir, max_files=300):
    """
    PHIÊN BẢN TỐC ĐỘ BÀN THỜ (ULTRA-FAST BATCH INFERENCE)
    Gom 300 file wav chạy mHuBERT đúng 1 LẦN DUY NHẤT để tận dụng sức mạnh GPU.
    Sau đó tự động xẻ ngược cục ma trận tổng ra 300 file nhỏ để khớp hàm ABX.
    """
    print(f"\n========================================================")
    print(f"🚀 BẮT ĐẦU: ULTRA-FAST BATCH INFERENCE CHO {lang_mode.upper()} ({max_files} files)")
    print(f"========================================================")
    
    os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_km_dir, exist_ok=True)
    
    wav_files = [f for f in os.listdir(input_wav_dir) if f.endswith(".wav")][:max_files]
    if not wav_files:
        print(f"❌ Không có file .wav nào trong thư mục: {input_wav_dir}")
        return
        
    temp_dir = os.path.join(BASE_TARGET, lang_mode, "ultra_fast_temp")
    temp_wav_folder = os.path.join(temp_dir, "wav_input")
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_wav_folder, exist_ok=True)
    
    # -------------------------------------------------------------
    # BƯỚC 1: DỌN RÁC & GOM TOÀN BỘ VÀO 1 THƯ MỤC
    # -------------------------------------------------------------
    print(f"⏳ Cắt lọc Data rác chuẩn bị đưa vào máy ép mHuBERT...")
    valid_wav_names = []
    import librosa
    import soundfile as sf
    import warnings
    warnings.filterwarnings('ignore')
    
    for wav_name in wav_files:
        source_wav_path = os.path.join(input_wav_dir, wav_name)
        try:
            y, sr = librosa.load(source_wav_path, sr=16000)
            if 320 <= len(y) <= 16000 * 20: 
                sf.write(os.path.join(temp_wav_folder, wav_name), y, 16000, subtype='PCM_16')
                valid_wav_names.append(wav_name)
        except Exception: 
            pass
            
    print(f"✅ Gom thành công {len(valid_wav_names)}/{len(wav_files)} files Đạt chuẩn vào bệ phóng.")

    if not valid_wav_names:
        return

    # -------------------------------------------------------------
    # BƯỚC 2: KÍCH HOẠT FAIRSEQ GPU CHUẨN XỬ LÝ SONG SONG
    # -------------------------------------------------------------
    print("🔥 Đang khởi động lõi GPU...")
    try:
        # Tạo tsv
        subprocess.run(["python", os.path.join(FAIRSEQ_CODE_DIR, "examples/wav2vec/wav2vec_manifest.py"),
            temp_wav_folder, "--dest", temp_dir, "--ext", "wav", "--valid-percent", "0"
        ], check=True, stdout=subprocess.DEVNULL)
        
        # Múc Layer 11
        print("💥 Ép Feature mHuBERT Layer 11 (Tất cả File bay luôn trong 1 lượt)...")
        subprocess.run(["python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_hubert_feature.py"),
            temp_dir, "train", HUBERT_CKPT, "11", "1", "0", temp_dir
        ], check=True, stdout=subprocess.DEVNULL)
        
        # Múc K-Means
        print("💥 Định lượng Unit K-Means...")
        km_model = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin") if lang_mode == "source" else os.path.join(BASE_DIR, f"kmeans/kmeans_vn_{KMEAN}.bin")
        subprocess.run(["python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_km_label.py"),
            temp_dir, "train", km_model, "1", "0", temp_dir
        ], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"❌ Xảy ra điểm đứt tại FAIRSEQ. Mọi file không thể hoàn thành. Lỗi: {e}")
        return

    # -------------------------------------------------------------
    # BƯỚC 3: GIẢI NÉN MA TRẬN KHỔNG LỒ THÀNH CÁC FILE ĐỘC LẬP
    # -------------------------------------------------------------
    print("✂️ Đang tiến hành Phẫu thuật xẻ gập (Tensor Slicing) ra file cá nhân...")
    
    # 1. Đọc Danh sách thứ tự từ file TSV
    tsv_list = []
    with open(os.path.join(temp_dir, "train.tsv"), "r") as f:
        lines = f.read().strip().split('\n')
        # Bỏ dòng 1 (Root path), Tách cột 1 (Filename)
        tsv_list = [line.split('\t')[0] for line in lines[1:]] 

    # 2. Đọc độ dài Frame Array (.len)
    with open(os.path.join(temp_dir, "train_0_1.len"), "r") as f:
        frame_lengths = [int(l) for l in f.read().strip().split('\n')]
        
    # 3. Đọc file K-means khổng lồ (.km)
    with open(os.path.join(temp_dir, "train_0_1.km"), "r") as f:
        km_lines = f.read().strip().split('\n')

    # 4. Load ma trận Continuous Khổng lồ (.npy) - [Tổng số frames, 1024]
    big_matrix = np.load(os.path.join(temp_dir, "train_0_1.npy"))

    # ---> Tiếng hành "XẺ" (Slicing Index) theo thứ tự
    current_frame_idx = 0
    exported_count = 0
    
    for i, wav_basename in enumerate(tsv_list):
        n_frames = frame_lengths[i]
        unit_string = km_lines[i]
        file_id = os.path.splitext(wav_basename)[0]
        
        # Mảng con (Sub-matrix) (Kích thước: n_frames x 1024)
        sub_matrix = big_matrix[current_frame_idx : current_frame_idx + n_frames]
        current_frame_idx += n_frames
        
        # Lưu vào đích đến
        np.save(os.path.join(out_npy_dir, f"{file_id}.npy"), sub_matrix)
        with open(os.path.join(out_km_dir, f"{file_id}.km"), "w") as f:
            f.write(unit_string)
            
        exported_count += 1

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n🎯 HOÀN MỸ! Thay vì mất ~30 phút, quá trình hoàn tất trong chớp mắt!")
    print(f"- Đã tách bung thành công {exported_count} tệp âm thanh đơn lẻ.")
    print(f"- Files Layer 11 tại: {out_npy_dir}")
    print(f"- Files Units tại: {out_km_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["source", "target"], required=True, help="source (Anh) hoặc target (Việt)")
    parser.add_argument("--in_wav",  default= "/mnt/g/data_final/data/target/test" , required=False, help="Thư mục chứa file Audio.wav")
    parser.add_argument("--out_npy", default= f"/mnt/e/AI/khanh/abx_test_sample_{KMEAN}/npys" , required=False, help="Thư mục xuất NPY")
    parser.add_argument("--out_km", default= f"/mnt/e/AI/khanh/abx_test_sample_{KMEAN}/kms" , required=False, help="Thư mục xuất KM")
    parser.add_argument("--max_files", type=int, default=300, help="Số file tối đa xử lý")
    
    args = parser.parse_args()
    ultra_fast_batch_infer(args.lang, args.in_wav, args.out_npy, args.out_km, args.max_files)
