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

# Unit2Mel Acoustic Model
sys.path.insert(0, os.path.join(BASE_DIR, "src", "Unit2Mel"))
try:
    from model import FastSpeech2AcousticModel
except ImportError:
    print("LỖI: Không import được FastSpeech2AcousticModel.")

torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

# --- ĐƯỜNG DẪN MODEL & CHECKPOINT ---
# 1. HuBERT (Wav -> Source Unit)
HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
KM_MODEL_SOURCE = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

# 2. Translation (Source Unit -> Target Dedup Unit)
DATA_BIN_U2U = os.path.join(BASE_DIR, "checkpoints", "data_bin_unit2unit_Dedup_Dur")
MODEL_U2U_PATH = os.path.join(BASE_DIR, "checkpoints", "unit2unit_Dedup_Dur_v1", "checkpoint_best.pt")

# 3. Acoustic Model (Target Dedup Unit -> Mel)
ACOUSTIC_CKPT = os.path.join(BASE_DIR, "checkpoints", "Unit2Mel", "fs2_epoch_1000.pt")

# 4. Universal Vocoder (Mel -> Wav)
VOCODER_CKPT = os.path.join(BASE_DIR, "checkpoints", "universal_hifigan", "g_02500000")

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
        import custom_u2u  

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

# --- CLASS 3: ACOUSTIC MODEL (UNIT -> MEL) ---
class FastSpeech2Wrapper:
    def __init__(self, checkpoint_path):
        print(">>> [3/4] Loading Acoustic Model (Unit2Mel)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FastSpeech2AcousticModel().to(self.device)
        
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
        else:
            print(f"Bỏ qua Load Checkpoint AM (Đang trong quá trình Dev/Train)")
            
        self.model.eval()

    def generate_mel(self, dedup_unit_str):
        unit_ids = [int(u) for u in dedup_unit_str.split()]
        if not unit_ids:
            return None
        
        with torch.no_grad():
            units_tensor = torch.tensor(unit_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            src_masks = (units_tensor == 0)
            
            mel_before, mel_after, log_dur, p, e, mel_masks = self.model(
                units=units_tensor, src_masks=src_masks
            )
            
        return mel_after.squeeze(0).cpu().numpy() # [Time, 80]

# --- PIPELINE END-TO-END ---
class S2STPipelineV2:
    def __init__(self):
        print("=== KHỞI TẠO PIPELINE V2 (UNIT -> MEL -> WAV) ===")
        self.extractor = FastUnitExtractor(HUBERT_CKPT, KM_MODEL_SOURCE)
        self.translator = FastTranslator(MODEL_U2U_PATH, DATA_BIN_U2U)
        self.acoustic_model = FastSpeech2Wrapper(ACOUSTIC_CKPT)
        # TODO: self.vocoder = UniversalHiFiGANWrapper(VOCODER_CKPT)
        print("=== KHỞI TẠO HOÀN TẤT ===\n")

    def process_audio(self, input_wav, output_wav):
        print("\n" + "="*50)
        print("🚀 BẮT ĐẦU QUÁ TRÌNH INFERENCE (SPEECH-TO-SPEECH V2)")
        print(f"📥 Input : {input_wav}")
        print("="*50 + "\n")

        try:
            # Bước 1: Wav -> Unit
            src_units = self.extractor.extract_units(input_wav)
            print(f"✅ [Step 1] Source Units: {src_units[:100]}...\n")

            # Bước 2: Source Unit -> Target Dedup Unit
            tgt_dedup_units = self.translator.translate(src_units)
            print(f"✅ [Step 2] Target Dedup Units: {tgt_dedup_units[:100]}...\n")

            # Bước 3: Target Dedup Unit -> Mel Spectrogram
            mel = self.acoustic_model.generate_mel(tgt_dedup_units)
            if mel is None: raise ValueError("Lỗi Mel generation")
            print(f"✅ [Step 3] Generated Mel Spectrogram shape: {mel.shape}\n")

            # Bước 4: Mel -> Wav (Universal HiFi-GAN)
            print(">>> [4/4] Generating Audio (Universal Vocoder)...")
            os.makedirs(os.path.dirname(output_wav), exist_ok=True) if os.path.dirname(output_wav) else None
            
            # TODO: audio = self.vocoder.infer(mel)
            # sf.write(output_wav, audio, 22050)
                
            print("\n🎉 [HOÀN TẤT] Pipeline chạy thành công!")
            print(f"🎵 Âm thanh dự kiến được lưu tại: {output_wav}")

        except Exception as e:
            print(f"\n❌ [LỖI] Pipeline gặp sự cố: {e}")

if __name__ == "__main__":
    IN_WAV = "/mnt/e/AI/khanh/sentence_001832.wav" 
    OUT_WAV = "/mnt/e/AI/khanh/output_pipeline_v2_vi.wav"
    
    pipeline = S2STPipelineV2()
    
    if os.path.exists(IN_WAV):
        pipeline.process_audio(IN_WAV, OUT_WAV)
    else:
        print(f"⚠️ Không tìm thấy file thử nghiệm: {IN_WAV}")
