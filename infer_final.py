import os
import subprocess
import shutil
import re

# --- CẤU HÌNH ĐƯỜNG DẪN TỔNG ---
BASE_DIR = '/mnt/e/AI/khanh'
FAIRSEQ_DIR = os.path.join(BASE_DIR, "fairseq")
FAIRSEQ_CODE_DIR = os.path.join(BASE_DIR, "fairseq") # Thư mục chứa code python fairseq

# 1. Cấu hình Speech-to-Unit (Source: Tiếng Anh)
HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
KM_MODEL_SOURCE = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

# 2. Cấu hình Translation (Unit-to-Unit)
DATA_BIN_U2U = os.path.join(BASE_DIR, "checkpoints", "data_bin_unit2unit")
MODEL_U2U_PATH = os.path.join(BASE_DIR, "checkpoints/unit2unit/checkpoint_best.pt")

# 3. Cấu hình Vocoder (Target: Tiếng Việt)
VOCODER_CKPT = os.path.join(BASE_DIR, 'checkpoints/vocoder_target/g_00027000')
VOCODER_CONFIG = os.path.join(BASE_DIR, 'checkpoints/vocoder_target/config.json')


def step_1_speech_to_unit(input_wav_path):
    """Bước 1: Chuyển Audio Tiếng Anh -> Source Unit IDs"""
    print(f"\n>>> BƯỚC 1: Speech-to-Unit (Source)...")
    
    # Setup thư mục tạm
    temp_dir = os.path.join(BASE_DIR, "temp_pipeline_s2u")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(os.path.join(temp_dir, "wav_input"), exist_ok=True)
    
    # Copy file wav vào
    shutil.copy(input_wav_path, os.path.join(temp_dir, "wav_input/input.wav"))

    try:
        # 1.1 Manifest
        subprocess.run([
            "python", os.path.join(FAIRSEQ_CODE_DIR, "examples/wav2vec/wav2vec_manifest.py"),
            os.path.join(temp_dir, "wav_input"), "--dest", temp_dir, "--ext", "wav", "--valid-percent", "0"
        ], check=True, stdout=subprocess.DEVNULL)

        # 1.2 Hubert Feature
        subprocess.run([
            "python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_hubert_feature.py"),
            temp_dir, "train", HUBERT_CKPT, "11", "1", "0", temp_dir
        ], check=True, stdout=subprocess.DEVNULL)

        # 1.3 Quantize (K-means)
        subprocess.run([
            "python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_km_label.py"),
            temp_dir, "train", KM_MODEL_SOURCE, "1", "0", temp_dir
        ], check=True, stdout=subprocess.DEVNULL)

        # 1.4 Đọc kết quả
        with open(os.path.join(temp_dir, "train_0_1.km"), 'r') as f:
            source_units = f.read().strip()
        
        print(f"--> Source Units: {source_units[:50]}...")
        return source_units

    except Exception as e:
        print(f"Lỗi Bước 1: {e}")
        return None

def step_2_translation(source_units):
    """Bước 2: Dịch Source Units -> Target Units (Fairseq Interactive)"""
    print(f"\n>>> BƯỚC 2: Translating Unit-to-Unit...")
    
    # Lệnh chạy fairseq-interactive
    cmd = [
        "fairseq-interactive", DATA_BIN_U2U,
        "--path", MODEL_U2U_PATH,
        "--beam", "5",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--buffer-size", "1024",
        "--max-tokens", "4096" , 
        "--max-len-a", "1.4",  # Cho phép output dài gấp 1.2 lần input
        "--max-len-b", "500"
    ]

    # Truyền source_units vào qua stdin (giả lập việc gõ phím)
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    stdout, stderr = process.communicate(input=source_units)

    # Phân tích kết quả trả về để tìm dòng Hypothesis (H-0)
    target_units = ""
    for line in stdout.split('\n'):
        if line.startswith("H-0"):
            # Format: H-0  -0.4532  10 20 55...
            # Cắt bỏ phần đầu, lấy phần unit
            parts = line.split('\t')
            if len(parts) >= 3:
                target_units = parts[2].strip()
                break
    
    if target_units:
        print(f"--> Target Units: {target_units[:50]}...")
        return target_units
    else:
        print("Lỗi Bước 2: Không tìm thấy kết quả dịch.")
        print("STDERR:", stderr) # In lỗi nếu có
        return None

def step_3_vocoder(target_units, output_wav_path):
    """Bước 3: Chuyển Target Units -> Audio Tiếng Việt (Custom Vocoder)"""
    print(f"\n>>> BƯỚC 3: Vocoder (Unit-to-Speech)...")
    
    # Lưu target units ra file tạm
    temp_unit_file = "temp_target_units.txt"
    with open(temp_unit_file, 'w') as f:
        f.write(target_units)

    # Gọi custom infer.py 
    cmd = [
        "python", os.path.join(BASE_DIR, "speech-resynthesis/infer.py"),
        "--input_file", temp_unit_file,
        "--output_file", output_wav_path,
        "--checkpoint_file", VOCODER_CKPT,
        "--config", VOCODER_CONFIG
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"--> Đã lưu file âm thanh tại: {output_wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi Bước 3: {e}")
    finally:
        if os.path.exists(temp_unit_file):
            os.remove(temp_unit_file)

def run_pipeline(input_wav, output_wav):
    if not os.path.exists(input_wav):
        print("File đầu vào không tồn tại!")
        return

    # 1. Audio -> Source Units
    src_units = step_1_speech_to_unit(input_wav)
    if not src_units: return

    # 2. Translate -> Target Units
    tgt_units = step_2_translation(src_units)
    if not tgt_units: return

    # 3. Target Units -> Audio
    step_3_vocoder(tgt_units, output_wav)
    
    print("\n=== QUÁ TRÌNH DỊCH HOÀN TẤT ===")

if __name__ == "__main__":
    # FILE ĐẦU VÀO (Tiếng Anh)
    IN_WAV = "/mnt/e/AI/khanh/sentence_000012.wav" 
    
    # FILE ĐẦU RA (Tiếng Việt)
    OUT_WAV = "/mnt/e/AI/khanh/output_result_vi.wav"
    
   
    run_pipeline(IN_WAV, OUT_WAV)