import os
import numpy as np
import whisper
import librosa
from collections import defaultdict
from evaluate_abx import ABXEvaluator, generate_triplets_from_pseudo_labels



class WhisperPseudoLabeler:
    """
    Sử dụng Whisper để nghe và cắt nhãn thời gian (Timestamps) tự động cho Audio.
    Nhãn ở đây có thể là Từ (Words) hoặc Cụm âm thành phần.
    """
    def __init__(self, model_size="base"):
        print(f"⌛ Đang tải mô hình ASR Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)
    
    def generate_timestamps(self, audio_path):
        """
        Trả về danh sách các từ và mốc thời gian của nó.
        Format: [{'word': 'xin', 'start': 0.5, 'end': 0.8}, ...]
        """
        print(f"🎙️ Đang trích xuất Timestamp Nhãn giả cho: {os.path.basename(audio_path)}")
        
        # force_plugins="words" để ép Whisper nhả word_timestamps
        result = self.model.transcribe(audio_path, word_timestamps=True)
        
        segments_info = []
        for segment in result["segments"]:
            for word in segment["words"]:
                # Làm sạch nhãn (Loại khoảng trắng, dấu phẩy, đưa về chữ thường)
                clean_word = word['word'].strip().lower().replace(",","").replace(".","")
                if clean_word == "":
                    continue
                    
                segments_info.append({
                    "word": clean_word,
                    "start": word['start'],
                    "end": word['end']
                })
        return segments_info

class S2STFeatureExtractor:
    def __init__(self, frame_shift=0.02):
        # Đặc trưng của mHuBERT trích tại Layer 11 ứng với 20ms/frame (tức 50 Frames/giây)
        self.frame_shift = frame_shift  
        
    def time_to_frame(self, time_sec):
        return int(time_sec / self.frame_shift)
        
    def slice_features(self, npy_path, km_path, word_segments):
        """
        Cắt .npy và chuỗi .km dựa trên timestamp phân định từ Whisper
        """
        # Load Data
        features_layer11 = np.load(npy_path) # Ma trận Numpy (N, 1024)
        
        with open(km_path, 'r', encoding='utf-8') as f:
            units_discrete = [int(x) for x in f.read().strip().split()] # Chuỗi (N,)
            
        continuous_dict = defaultdict(list)
        discrete_dict = defaultdict(list)
        
        for seg in word_segments:
            word = seg["word"]
            start_frame = self.time_to_frame(seg["start"])
            end_frame = self.time_to_frame(seg["end"])
            
            # An toàn: Đảm bảo không Vượt quá biên mảng
            end_frame = min(end_frame, len(features_layer11))
            end_frame_unit = min(end_frame, len(units_discrete))
            
            # Nếu cụm từ quá ngắn (dưới 1 frame tức < 20ms) thì bỏ qua
            if start_frame >= end_frame:
                continue
                
            # Trích xuất dữ liệu ranh giới ngữ âm
            feat_slice = features_layer11[start_frame:end_frame]
            unit_slice = units_discrete[start_frame:end_frame_unit]
            
            continuous_dict[word].append(feat_slice)
            discrete_dict[word].append(unit_slice)
            
        return continuous_dict, discrete_dict

def run_batch_evaluation_pipeline(wav_dir, npy_dir, km_dir, max_files=300):
    print("="*60)
    print(f" BỘ TIỀN XỬ LÝ NHÃN GIẢ HÀNG LOẠT (BATCH MODE - {max_files} FILES)")
    print("="*60)
    
    whisper_labeler = WhisperPseudoLabeler(model_size="base")
    feature_extractor = S2STFeatureExtractor(frame_shift=0.02)
    abx_eval = ABXEvaluator()
    
    global_layer11 = defaultdict(list)
    global_unit = defaultdict(list)
    
    # 1. Quét toàn bộ file Wav trong thư mục
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    wav_files = wav_files[:max_files] # Giới hạn số lượng test
    
    print(f"📁 Tìm thấy {len(wav_files)} files. Đang bắt đầu quá trình trích xuất...")
    
    successful_files = 0
    for idx, wav_name in enumerate(wav_files):
        base_name = os.path.splitext(wav_name)[0]
        
        # Mapping tên file Wav sang Npy và Km tương ứng
        # (Ví dụ: sentence_01.wav -> sentence_01.npy, sentence_01.km)
        wav_path = os.path.join(wav_dir, wav_name)
        npy_path = os.path.join(npy_dir, f"{base_name}.npy")
        km_path  = os.path.join(km_dir, f"{base_name}.km")
        
        if not (os.path.exists(npy_path) and os.path.exists(km_path)):
            print(f"⚠️ Bỏ qua {wav_name}: Không tìm thấy file .npy hoặc .km tương ứng.")
            continue
            
        print(f"[{idx+1}/{len(wav_files)}] Xử lý: {wav_name} ...")
        
        try:
            # Thu thập nhãn và Thời gian
            word_segments = whisper_labeler.generate_timestamps(wav_path)
            
            # Cắt và dồn data vào bộ nhớ cục bộ
            dict_layer11, dict_unit = feature_extractor.slice_features(npy_path, km_path, word_segments)
            
            # Dồn vào ma trận TỔNG (Global)
            for word, feats in dict_layer11.items():
                global_layer11[word].extend(feats)
            for word, units in dict_unit.items():
                global_unit[word].extend(units)
                
            successful_files += 1
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {wav_name}: {e}")
            continue

    print(f"\\n🏁 Hoàn tất quét {successful_files} files âm thanh.")
    print(f"📚 Từ điển tổng thu thập được {len(global_layer11)} nhãn từ vựng khác nhau.")
    
    # 2. Bộ Lọc Data Tổng (Chỉ lấy Nhãn xuất hiện >= 2 lần)
    valid_labels = [w for w, feats in global_layer11.items() if len(feats) >= 2]
    if len(valid_labels) < 2:
        print("❌ Dataset vẫn quá ngắn! Không đủ từ vựng trùng lặp để sinh ABX. Hãy tăng max_files lên!")
        return
        
    filtered_layer11 = {w: global_layer11[w] for w in valid_labels}
    filtered_unit = {w: global_unit[w] for w in valid_labels}
    
    # 3. Ráp cặp và Chạy Test ABX cục bộ 
    num_triplets = min(len(valid_labels) * 10, 5000) # Tuỳ biến số cặp dựa theo Data
    print(f"Đang sinh {num_triplets} tổ hợp ABX ngẫu nhiên từ kho dữ liệu chéo...")
    
    cont_triplets = generate_triplets_from_pseudo_labels(filtered_layer11, num_triplets=num_triplets)
    disc_triplets = generate_triplets_from_pseudo_labels(filtered_unit, num_triplets=num_triplets)
    
    print("\\n🚀 BẮT ĐẦU CHẤM ĐIỂM (EVALUATION):")
    err_layer11 = abx_eval.evaluate_continuous_abx(cont_triplets)
    err_unit = abx_eval.evaluate_discrete_abx(disc_triplets)
    
    quantization_loss = err_unit - err_layer11
    
    print("="*60)
    print(f"📊 SUMMARY REPORT REPORT TRÊN {successful_files} FILES:")
    print(f"- Lỗi rễ (Layer 11 Baseline) : {err_layer11:.2f}%")
    print(f"- Lỗi ngọn (Wav2Unit K-Means): {err_unit:.2f}%")
    print(f"- Cán cân Hao hụt Loss       : +{quantization_loss:.2f}%")
    print("="*60)

if __name__ == "__main__":
    KMEAN  = 500
    # Đã tự động Configure theo thư mục 299 file bạn vừa trích xuất xong!
    wav_folder = f"/mnt/g/data_final/data/target/test"
    npy_folder = f"/mnt/e/AI/khanh/abx_test_sample_{KMEAN}/npys" 
    km_folder  = f"/mnt/e/AI/khanh/abx_test_sample_{KMEAN}/kms"
    
    run_batch_evaluation_pipeline(wav_folder, npy_folder, km_folder, max_files=300)
