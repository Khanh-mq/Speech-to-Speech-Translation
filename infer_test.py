import os
import sys
import torch
import soundfile as sf
import joblib
import glob
import subprocess
from tqdm import tqdm
import time 
import shutil


# --- 1. FIX LỖI IMPORT FAIRSEQ ---
FAIRSEQ_PATH = "/mnt/e/AI/khanh/fairseq"
if FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, FAIRSEQ_PATH)
    sys.path.insert(0, os.path.join(FAIRSEQ_PATH, "fairseq"))

try:
    from fairseq.checkpoint_utils import load_model_ensemble_and_task
    from fairseq import utils
except ImportError:
    print("LỖI: Không import được Fairseq. Kiểm tra FAIRSEQ_PATH.")
    sys.exit(1)

# --- 2. FIX LỖI PYTORCH 2.x (Mask Type & SDP) ---
# Dòng này sẽ có hiệu lực cho cả Bước 1 và Bước 2 vì giờ chạy chung process
torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

# --- CẤU HÌNH ---
BASE_DIR = '/mnt/e/AI/khanh'
BASE_DIR_DATA = "/mnt/g/data_final/data/source"
BASE_DIR_DATA_SOURCE =  "mnt/g/data_final/data/target"
# Models
HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
KM_MODEL_SOURCE = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")
DATA_BIN_U2U = os.path.join(BASE_DIR, "checkpoints", "data_bin_unit2unit")
MODEL_U2U_PATH = os.path.join(BASE_DIR, "checkpoints/unit2unit/checkpoint.best_loss_2.8144.pt")
VOCODER_CKPT = os.path.join(BASE_DIR, 'checkpoints/vocoder_target/g_00027000')
VOCODER_CONFIG = os.path.join(BASE_DIR, 'checkpoints/vocoder_target/config.json')

# Output
TEST_INPUT_DIR = os.path.join(BASE_DIR_DATA, "test") 
TEST_INPUT_DIR_TARGET = os.path.join(BASE_DIR_DATA_SOURCE, "test") 
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "test_output_results") 

# --- CLASS 1: HUBERT (Giữ nguyên vì đã chạy tốt) ---
class FastUnitExtractor:
    def __init__(self, hubert_path, km_path):
        print(">>> [1/3] Loading HuBERT...")
        models, cfg, task = load_model_ensemble_and_task([hubert_path])
        self.model = models[0].cuda().eval()
        km_data = joblib.load(km_path)
        if isinstance(km_data, dict) and 'centroids' in km_data:
            self.centroids = torch.from_numpy(km_data['centroids']).cuda()
        elif hasattr(km_data, 'cluster_centers_'):
            self.centroids = torch.from_numpy(km_data.cluster_centers_).cuda()
        else:
            self.centroids = torch.from_numpy(km_data).cuda()

    def extract_units(self, wav_path):
        speech, sr = sf.read(wav_path)
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        input_values = torch.from_numpy(speech).float().cuda().unsqueeze(0)
        with torch.no_grad():
            padding_mask = torch.BoolTensor(input_values.shape).fill_(False).cuda()
            res = self.model.extract_features(input_values, padding_mask)
            feat = res[0] if isinstance(res, (tuple, list)) else res['x'] if isinstance(res, dict) else res
            if isinstance(feat, list): feat = feat[11] # Layer 11 safety check
        
        dist = ( (feat**2).sum(dim=-1, keepdim=True) 
                 - 2 * torch.matmul(feat, self.centroids.transpose(0, 1))
                 + (self.centroids**2).sum(dim=-1).unsqueeze(0).unsqueeze(0) )
        units = dist.argmin(dim=-1).squeeze(0).cpu().numpy()
        return " ".join(map(str, units))

# # --- CLASS 2: TRANSLATOR (THAY THẾ SUBPROCESS FAIRSEQ) ---
# class FastTranslator:
#     def __init__(self, model_path, data_bin):
#         print(">>> [2/3] Loading Translation Model...")
#         # Load model thủ công thay vì gọi lệnh shell
#         self.models, self.cfg, self.task = load_model_ensemble_and_task(
#             [model_path], 
#             arg_overrides={'data': data_bin, 'source_lang': 'src', 'target_lang': 'tgt'}
#         )

#         self.model = self.models[0].cuda().eval()
#         self.generator = self.task.build_generator(self.models, self.cfg.generation)
#         self.src_dict = self.task.source_dictionary
#         self.tgt_dict = self.task.target_dictionary

#     # def translate_batch(self, unit_strings, beam=5):
#     #     # 1. Convert text units -> tensor ids
#     #     tokens_list = [
#     #         self.src_dict.encode_line(u, add_if_not_exist=False).long() 
#     #         for u in unit_strings
#     #     ]
        
#     #     # 2. Batching (Collating)
#     #     lengths = torch.LongTensor([t.numel() for t in tokens_list])
#     #     # Pad inputs
#     #     max_len = lengths.max().item()
#     #     # Cắt nếu quá dài để tránh OOM (tương tự max-source-positions)
#     #     if max_len > 4000: max_len = 4000
            
#     #     src_tokens = torch.full((len(tokens_list), max_len), self.src_dict.pad(), dtype=torch.long)
#     #     for i, t in enumerate(tokens_list):
#     #         l = min(t.numel(), 4000)
#     #         src_tokens[i, :l] = t[:l]
            
#     #     src_tokens = src_tokens.cuda()
#     #     src_lengths = lengths.cuda().clamp(max=4000)

#     #     # 3. Generate Input Dict for Fairseq
#     #     sample = {
#     #         'net_input': {
#     #             'src_tokens': src_tokens,
#     #             'src_lengths': src_lengths,
#     #         }
#     #     }

#     #     # 4. Run Inference (TRỰC TIẾP TRONG PYTHON - FIX ĐƯỢC LỖI MASK)
#     #     with torch.no_grad():
#     #         translations = self.generator.generate(self.models, sample, prefix_tokens=None)

#     #     # 5. Decode outputs
#     #     results = []
#     #     for hypos in translations:
#     #         # Lấy giả thuyết tốt nhất (hypos[0])
#     #         best_hypo = hypos[0] 
#     #         # Chuyển tensor ids -> unit string
#     #         out_str = self.tgt_dict.string(best_hypo['tokens'])
#     #         # Loại bỏ các token đặc biệt nếu có
#     #         results.append(out_str)
            
#     #     return results

# --- CLASS 2: TRANSLATOR (ĐÃ SỬA LỖI ĐỘ DÀI OUTPUT) ---
class FastTranslator:
    def __init__(self, model_path, data_bin):
        print(">>> [2/3] Loading Translation Model...")
        self.models, self.cfg, self.task = load_model_ensemble_and_task(
            [model_path], 
            arg_overrides={'data': data_bin, 'source_lang': 'src', 'target_lang': 'tgt'}
        )
        
        # --- FIX QUAN TRỌNG: Cấu hình lại độ dài sinh mã ---
        # Mặc định nó chỉ cho phép sinh khoảng 200 token (~4 giây âm thanh)
        # Ta phải tăng lên để phù hợp với Speech Units
        self.cfg.generation.beam = 5
        self.cfg.generation.max_len_a = 1.2  # Output được phép dài gấp 1.2 lần Input
        self.cfg.generation.max_len_b = 500  # Cộng thêm 500 token đệm (~10 giây)
        self.cfg.generation.min_len = 10     # Độ dài tối thiểu
        
        # Build Generator với cấu hình mới
        self.model = self.models[0].cuda().eval()
        self.generator = self.task.build_generator(self.models, self.cfg.generation)
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

    def translate_batch(self, unit_strings):
        # 1. Convert text units -> tensor ids
        tokens_list = [
            self.src_dict.encode_line(u, add_if_not_exist=False).long() 
            for u in unit_strings
        ]
        
        # 2. Batching
        lengths = torch.LongTensor([t.numel() for t in tokens_list])
        max_len = lengths.max().item()
        if max_len > 4000: max_len = 4000
            
        src_tokens = torch.full((len(tokens_list), max_len), self.src_dict.pad(), dtype=torch.long)
        for i, t in enumerate(tokens_list):
            l = min(t.numel(), 4000)
            src_tokens[i, :l] = t[:l]
            
        src_tokens = src_tokens.cuda()
        src_lengths = lengths.cuda().clamp(max=4000)

        # 3. Generate Input Dict
        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            }
        }

        # 4. Inference
        with torch.no_grad():
            # Generator bây giờ đã nhận cấu hình max_len_a/b mới
            translations = self.generator.generate(self.models, sample, prefix_tokens=None)

        # 5. Decode outputs
        results = []
        for hypos in translations:
            best_hypo = hypos[0] 
            # In ra độ dài để kiểm tra (Debug)
            # print(f"DEBUG Length: {len(best_hypo['tokens'])}") 
            out_str = self.tgt_dict.string(best_hypo['tokens'])
            results.append(out_str)
            
        return results

# --- MAIN PIPELINE ---
def run_fast_pipeline(limit=200):
    if not os.path.exists(TEST_OUTPUT_DIR): os.makedirs(TEST_OUTPUT_DIR)

    #  thuwcj hiện copy số file vào bên trong  uest putput_dỉr để dể kiểm tra 

    # 1. EXTRACT
    extractor = FastUnitExtractor(HUBERT_CKPT, KM_MODEL_SOURCE)
    wav_files = sorted(glob.glob(os.path.join(TEST_INPUT_DIR, "*.wav")))[:limit]
    
    wav_files_source = sorted(glob.glob(os.path.join(TEST_INPUT_DIR_TARGET, "*.wav")))[:limit]
    print(f">>> Copy {len(wav_files_source)} file test sang thư mục output...")
    for wav_path in wav_files_source:
        filename = os.path.basename(wav_path)
        dest_path = os.path.join(TEST_OUTPUT_DIR, filename)
        shutil.copy2(wav_path, dest_path)

    # Lọc file, lưu index để map lại sau
    batch_data = [] # List tuple (index, original_path, source_unit_str)
    
    print(f"\n>>> BƯỚC 1: Trích xuất Units ({len(wav_files)} file)...")
    for i, wav_path in enumerate(tqdm(wav_files)):
        try:
            u = extractor.extract_units(wav_path)
            batch_data.append({'id': i, 'path': wav_path, 'src': u})
        except Exception as e:
            print(f"Lỗi extract {os.path.basename(wav_path)}: {e}")

    # Xóa extractor để giải phóng GPU
    del extractor
    torch.cuda.empty_cache()

    if not batch_data: return

    # 2. TRANSLATE
    translator = FastTranslator(MODEL_U2U_PATH, DATA_BIN_U2U)
    print(f"\n>>> BƯỚC 2: Dịch Unit-to-Unit (Trực tiếp trên GPU)...")
    
    # Batch size loop
    BATCH_SIZE = 16
    results_map = {} # map id -> target_unit_str
    
    # Gom tất cả src units thành list để xử lý theo batch
    all_src = [item['src'] for item in batch_data]
    
    for i in tqdm(range(0, len(all_src), BATCH_SIZE)):
        batch_src = all_src[i : i + BATCH_SIZE]
        try:
            batch_tgt = translator.translate_batch(batch_src)
            # Map kết quả lại vào dict
            for j, tgt_str in enumerate(batch_tgt):
                real_idx = i + j
                original_id = batch_data[real_idx]['id']
                results_map[original_id] = tgt_str
        except Exception as e:
            print(f"Lỗi dịch batch {i}: {e}")

    # Xóa translator
    del translator
    torch.cuda.empty_cache()

    # 3. VOCODER
    print(f"\n>>> BƯỚC 3: Tổng hợp âm thanh ({len(results_map)} câu)...")
    success_count = 0
    
    for item in tqdm(batch_data):
        oid = item['id']
        if oid in results_map:
            tgt_units = results_map[oid]
            filename = os.path.basename(item['path'])
            out_path = os.path.join(TEST_OUTPUT_DIR, f"final_{filename}")
            
            # Vocoder vẫn dùng subprocess vì code infer.py phức tạp
            temp_f = f"temp_{oid}.txt"
            with open(temp_f, 'w') as f: f.write(tgt_units)
            
            subprocess.run(["python", os.path.join(BASE_DIR, "speech-resynthesis/infer.py"),
                            "--input_file", temp_f, "--output_file", out_path,
                            "--checkpoint_file", VOCODER_CKPT, "--config", VOCODER_CONFIG], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Ẩn log rác
            
            if os.path.exists(temp_f): os.remove(temp_f)
            success_count += 1

    print(f"\n=== HOÀN TẤT! ===")
    print(f"Thành công: {success_count}/{limit}")
    print(f"Output: {TEST_OUTPUT_DIR}")

if __name__ == "__main__":
    run_fast_pipeline(15)
