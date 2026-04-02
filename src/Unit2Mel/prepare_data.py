import os
import glob
import torch
import librosa
import numpy as np
import pyworld as pw
from tqdm import tqdm
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- Configuration ---
WAV_DIR = '/mnt/g/data_final/data/target/train' # Thư mục chứa audio gốc tiếng Việt
TSV_PATH = '/mnt/g/khanh/manifest_temp/target/train.tsv' # File manifest
KM_PATH = '/mnt/g/khanh/kmean500/train_0_1.km' # File chứa Units gộp
OUT_DIR = '/mnt/g/khanh/Unit2Mel/processed_data'

# --- Audio Config (Tương thích HuBERT 50Hz) ---
SR = 16000 
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 320 # Khớp với HuBERT (50Hz)
WIN_LENGTH = 1024
FMIN = 0
FMAX = 8000

def get_mel_spectrogram(audio, sr):
    # Trích xuất Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, 
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    # Chuyển sang log scale
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T # Shape: (Time, n_mels)

def get_f0(audio, sr):
    # PyWorld trích xuất F0 tốt hơn cho giọng nói
    _f0, t = pw.dio(audio.astype(np.float64), sr, frame_period=HOP_LENGTH/(sr/1000))
    f0 = pw.stonemask(audio.astype(np.float64), _f0, t, sr)
    
    # Continuous F0 (Nội suy cho các khoảng unvoiced - bằng 0)
    nonzero_ids = np.where(f0 != 0)[0]
    if len(nonzero_ids) > 0:
        interp_fn = interp1d(
            nonzero_ids, f0[nonzero_ids], 
            fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]), 
            bounds_error=False
        )
        f0 = interp_fn(np.arange(0, len(f0)))
    return f0 # Shape: (Time,)

def get_energy(audio):
    # Tính Năng lượng bằng L2 norm qua các frame
    frames = librosa.util.frame(audio, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    energy = np.linalg.norm(frames, axis=0)
    return energy # Shape: (Time,)

def deduplicate_units(units, f0, energy):
    """ Deduplicate units and average f0/energy for each unit """
    dedup_units = []
    durations = []
    f0_avg = []
    energy_avg = []
    
    if len(units) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    curr_u = units[0]
    curr_dur = 0
    curr_f0_sum = 0
    curr_energy_sum = 0
    
    # Đảm bảo f0/energy có cùng độ dài với units
    min_len = min(len(units), len(f0), len(energy))
    units = units[:min_len]
    f0 = f0[:min_len]
    energy = energy[:min_len]
    
    for i in range(len(units)):
        if units[i] == curr_u:
            curr_dur += 1
            curr_f0_sum += f0[i]
            curr_energy_sum += energy[i]
        else:
            dedup_units.append(curr_u)
            durations.append(curr_dur)
            f0_avg.append(curr_f0_sum / curr_dur)
            energy_avg.append(curr_energy_sum / curr_dur)
            
            curr_u = units[i]
            curr_dur = 1
            curr_f0_sum = f0[i]
            curr_energy_sum = energy[i]
            
    # Add last unit
    dedup_units.append(curr_u)
    durations.append(curr_dur)
    f0_avg.append(curr_f0_sum / curr_dur)
    energy_avg.append(curr_energy_sum / curr_dur)
    
    return np.array(dedup_units), np.array(durations), np.array(f0_avg), np.array(energy_avg)

def process_one(data_pair):
    """ Hàm worker xử lý 1 file audio """
    wav_path, unit_str, base_id = data_pair
    
    try:
        # 1. Load Audio
        if not os.path.exists(wav_path):
            return None
        audio, _ = librosa.load(wav_path, sr=SR)
        
        # 2. Extract Features (Frame-level)
        mel = get_mel_spectrogram(audio, SR)
        f0 = get_f0(audio, SR)
        energy = get_energy(audio)
        units_raw = np.array([int(x) for x in unit_str.split()])
        
        # 3. Deduplicate and Average (Unit-level)
        # Đồng bộ độ dài thô trước khi dedup
        min_len = min(len(units_raw), len(f0), len(energy))
        units_raw = units_raw[:min_len]
        
        dedup_u, dur, f0_u, energy_u = deduplicate_units(units_raw, f0, energy)
        
        # Mel vẫn giữ nguyên độ dài frame-level để train decoder
        # Nhưng cần đảm bảo tổng duration = len(mel)
        mel = mel[:np.sum(dur), :]
        
        # 4. Save
        np.save(os.path.join(OUT_DIR, "mel", f"{base_id}.npy"), mel)
        np.save(os.path.join(OUT_DIR, "f0", f"{base_id}.npy"), f0_u) # Now unit-level
        np.save(os.path.join(OUT_DIR, "energy", f"{base_id}.npy"), energy_u) # Now unit-level
        np.save(os.path.join(OUT_DIR, "duration", f"{base_id}.npy"), dur)
        np.save(os.path.join(OUT_DIR, "unit", f"{base_id}.npy"), dedup_u) # Now deduped
        
        return base_id
    except Exception as e:
        # print(f"Error processing {base_id}: {e}")
        return None

def process_dataset():
    os.makedirs(os.path.join(OUT_DIR, "mel"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "f0"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "energy"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "duration"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "unit"), exist_ok=True)
    
    # 1. Đọc và ghép cặp dữ liệu TRƯỚC (để đảm bảo không bao giờ lệch)
    print(f"--> Đang nạp danh sách file và units...")
    data_pairs = []
    try:
        with open(TSV_PATH, 'r', encoding='utf-8') as f_tsv, \
             open(KM_PATH, 'r', encoding='utf-8') as f_km:
            
            f_tsv.readline() # Header
            for line_tsv, line_km in zip(f_tsv, f_km):
                filename = line_tsv.strip().split('\t')[0]
                base_id = filename.split('.')[0]
                wav_path = os.path.join(WAV_DIR, filename)
                unit_str = line_km.strip()
                
                if unit_str:
                    data_pairs.append((wav_path, unit_str, base_id))
    except Exception as e:
        print(f"[LỖI] Đọc file manifest: {e}")
        return

    print(f"--> Tìm thấy {len(data_pairs)} cặp hợp lệ. Bắt đầu chạy đa luồng...")
    
    # 2. Sử dụng Multi-processing
    processed_ids = []
    num_workers = min(multiprocessing.cpu_count(), 16) # Dùng tối đa 16 cores
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, pair) for pair in data_pairs]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Features"):
            res = future.result()
            if res:
                processed_ids.append(res)

    print(f"\nHoàn tất! Đã lưu {len(processed_ids)} files.")
    
    if len(processed_ids) > 0:
        import random
        random.seed(42)
        random.shuffle(processed_ids)
        train_size = int(len(processed_ids) * 0.95)
        
        with open(os.path.join(OUT_DIR, "train.txt"), 'w') as f:
            f.write("\n".join(processed_ids[:train_size]))
        with open(os.path.join(OUT_DIR, "val.txt"), 'w') as f:
            f.write("\n".join(processed_ids[train_size:]))
            
        print(f"Đã tạo danh sách: {train_size} train, {len(processed_ids)-train_size} validation.")
    else:
        print("[CẢNH BÁO] Không có file nào được xử lý thành công.")

if __name__ == "__main__":
    process_dataset()
