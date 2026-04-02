import os
import sys
import torch
import soundfile as sf
import joblib
import subprocess
import time 

# --- CẤU HÌNH ĐƯỜNG DẪN TỔNG ---
BASE_DIR = '/mnt/e/AI/khanh'
FAIRSEQ_PATH = os.path.join(BASE_DIR, "fairseq")
if FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, FAIRSEQ_PATH)
    sys.path.insert(0, os.path.join(FAIRSEQ_PATH, "fairseq"))

try:
    from fairseq.checkpoint_utils import load_model_ensemble_and_task
except ImportError:
    print("LỖI: Không import được Fairseq. Kiểm tra FAIRSEQ_PATH.")
    sys.exit(1)

# Import model Duration Predictor
sys.path.insert(0, os.path.join(BASE_DIR, "src", "model_duration"))
try:
    from model import DurationPredictor
except ImportError:
    print("LỖI: Không import được DurationPredictor.")
    sys.exit(1)

# Tắt Flash Attention để tránh lỗi PyTorch 2.x
torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

# --- ĐƯỜNG DẪN MODEL & CHECKPOINT ---
# 1. HuBERT (Wav -> Source Unit)
HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
KM_MODEL_SOURCE = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

# 2. Translation (Source Unit -> Target Dedup Unit)
DATA_BIN_U2U = os.path.join(BASE_DIR, "checkpoints", "data_bin_unit2unit_Dedup_Dur")
MODEL_U2U_PATH = os.path.join(BASE_DIR, "checkpoints", "unit2unit_Dedup_Dur_v1", "checkpoint_best.pt")

# 3. Duration Predictor (Target Dedup Unit -> Target Dup Unit)
DUR_MODEL_CKPT = os.path.join(BASE_DIR, "checkpoints", "duration_predictor", "best.pt")

# 4. Vocoder (Target Dup Unit -> Wav)
VOCODER_CKPT = os.path.join(BASE_DIR, 'checkpoints/vocoder_target_kmean500/g_00110000')
VOCODER_CONFIG = os.path.join(BASE_DIR, 'checkpoints/vocoder_target_kmean500/config.json')


# --- CLASS 1: HUBERT ---
class FastUnitExtractor:
    def __init__(self, hubert_path, km_path):
        print(">>> [1/4] Loading HuBERT...")
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
            if isinstance(feat, list): feat = feat[11]
        
        dist = ( (feat**2).sum(dim=-1, keepdim=True) 
                 - 2 * torch.matmul(feat, self.centroids.transpose(0, 1))
                 + (self.centroids**2).sum(dim=-1).unsqueeze(0).unsqueeze(0) )
        units = dist.argmin(dim=-1).squeeze(0).cpu().numpy()
        return " ".join(map(str, units))

# --- CLASS 2: TRANSLATOR (DEDUP) ---
class FastTranslator:
    def __init__(self, model_path, data_bin):
        USER_DIR = os.path.join(BASE_DIR, 'custom_u2u')
        if os.path.dirname(USER_DIR) not in sys.path:
            sys.path.insert(0, os.path.dirname(USER_DIR))
        import custom_u2u  # Đăng ký custom task_with_duration

        print(">>> [2/4] Loading Translation Model (Dedup)...")
        self.models, self.cfg, self.task = load_model_ensemble_and_task(
            [model_path], 
            arg_overrides={
                'data': data_bin, 
                'source_lang': 'src', 
                'target_lang': 'tgt',
                'user_dir': USER_DIR
            }
        )
        
        self.cfg.generation.beam = 5
        self.cfg.generation.max_len_a = 1.2
        self.cfg.generation.max_len_b = 500
        self.cfg.generation.min_len = 10
        
        self.model = self.models[0].cuda().eval()
        self.generator = self.task.build_generator(self.models, self.cfg.generation)
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

    def translate(self, unit_string):
        tokens = self.src_dict.encode_line(unit_string, add_if_not_exist=False).long() 
        src_tokens = tokens.cuda().unsqueeze(0)
        src_lengths = torch.LongTensor([tokens.numel()]).cuda()

        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            }
        }

        with torch.no_grad():
            translations = self.generator.generate(self.models, sample, prefix_tokens=None)

        best_hypo = translations[0][0] 
        out_str = self.tgt_dict.string(best_hypo['tokens'])
        return out_str

# --- CLASS 3: DURATION PREDICTOR ---
class DurationPredictorWrapper:
    def __init__(self, checkpoint_path):
        print(">>> [3/4] Loading Duration Predictor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["config"]
        
        self.model = DurationPredictor(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            conv_channels=cfg["conv_channels"],
            conv_layers=cfg["conv_layers"],
            kernel_size=cfg["kernel_size"],
            dropout=0.0,
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict_and_expand(self, dedup_unit_str):
        unit_ids = list(map(int, dedup_unit_str.split()))
        if not unit_ids:
            return ""
        
        with torch.no_grad():
            units_tensor = torch.tensor(unit_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            log_dur = self.model(units_tensor)
            dur = torch.exp(log_dur).squeeze(0)
            durations = dur.round().long().clamp(min=1).tolist()
            
        dup_units = []
        for u, d in zip(unit_ids, durations):
            dup_units.extend([str(u)] * d)
            
        return " ".join(dup_units)

# --- PIPELINE END-TO-END ---
class S2STPipeline:
    def __init__(self):
        print("=== KHỞI TẠO PIPELINE (LOAD MODELS) ===")
        # Khởi tạo tất cả models 1 lần duy nhất
        self.extractor = FastUnitExtractor(HUBERT_CKPT, KM_MODEL_SOURCE)
        self.translator = FastTranslator(MODEL_U2U_PATH, DATA_BIN_U2U)
        self.dur_predictor = DurationPredictorWrapper(DUR_MODEL_CKPT)
        print("=== KHỞI TẠO HOÀN TẤT ===\n")

    def process_audio(self, input_wav, output_wav):
        print("\n" + "="*50)
        print("🚀 BẮT ĐẦU QUÁ TRÌNH INFERENCE (SPEECH-TO-SPEECH)")
        print(f"📥 Input : {input_wav}")
        print(f"📤 Output: {output_wav}")
        print("="*50 + "\n")

        try:
            # Bước 1: Wav -> Unit
            src_units = self.extractor.extract_units(input_wav)
            print(f"✅ [Step 1] Source Units: {src_units[:100]}...\n")

            # Bước 2: Source Unit -> Target Dedup Unit
            tgt_dedup_units = self.translator.translate(src_units)
            print(f"✅ [Step 2] Target Dedup Units: {tgt_dedup_units[:100]}...\n")

            # Bước 3: Target Dedup Unit -> Target Dup Unit
            tgt_dup_units = self.dur_predictor.predict_and_expand(tgt_dedup_units)
            print(f"✅ [Step 3] Target Dup Units (Expanded): {tgt_dup_units[:100]}...\n")

            # Bước 4: Vocoder (Target Dup Unit -> Wav)
            print(">>> [4/4] Generating Audio (Vocoder)...")
            os.makedirs(os.path.dirname(output_wav), exist_ok=True) if os.path.dirname(output_wav) else None
            
            temp_unit_file = f"temp_vocoder_input_{int(time.time())}.txt"
            with open(temp_unit_file, 'w') as f:
                f.write(tgt_dup_units)

            cmd = [
                "python", os.path.join(BASE_DIR, "speech-resynthesis/infer.py"),
                "--input_file", temp_unit_file,
                "--output_file", output_wav,
                "--checkpoint_file", VOCODER_CKPT,
                "--config", VOCODER_CONFIG
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_unit_file):
                os.remove(temp_unit_file)
                
            print("\n🎉 [HOÀN TẤT] Pipeline chạy thành công!")
            print(f"🎵 Âm thanh đã được lưu tại: {output_wav}")

        except Exception as e:
            print(f"\n❌ [LỖI] Pipeline gặp sự cố: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation Pipeline (Tiếng Anh -> Tiếng Việt)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Đường dẫn tới file wav tiếng Anh đầu vào")
    parser.add_argument("-o", "--output", type=str, default="output_vi.wav", help="Đường dẫn lưu file wav tiếng Việt đầu ra")
    args = parser.parse_args()

    IN_WAV = args.input
    OUT_WAV = args.output
    
    if not os.path.exists(IN_WAV):
        print(f"⚠️ LỖI: Không tìm thấy file thử nghiệm: {IN_WAV}")
        sys.exit(1)

    # 1. Load pipeline (Chỉ tốn thời gian 1 lần khi khởi động)
    pipeline = S2STPipeline()
    
    # 2. Chạy inference
    pipeline.process_audio(IN_WAV, OUT_WAV)
