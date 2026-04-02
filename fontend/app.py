import gradio as gr
import os
import subprocess
import shutil
import re

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TỔNG (Giữ nguyên của bạn)
# ==========================================
BASE_DIR = '/mnt/e/AI/khanh'
FAIRSEQ_DIR = os.path.join(BASE_DIR, "fairseq")
FAIRSEQ_CODE_DIR = os.path.join(BASE_DIR, "fairseq") 

# 1. Cấu hình Speech-to-Unit (Source: Tiếng Anh)
HUBERT_CKPT = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
KM_MODEL_SOURCE = os.path.join(BASE_DIR, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

# 2. Cấu hình Translation (Unit-to-Unit)
# CHÚ Ý: Tôi đã cập nhật thành data_bin_unit2unit_dedup và thư mục checkpoint mới của bạn
DATA_BIN_U2U = os.path.join(BASE_DIR, "checkpoints", "data_bin_unit2unit_Dedup_Dur") 
MODEL_U2U_PATH = os.path.join(BASE_DIR, "checkpoints/unit2unit_Dedup_Dur/checkpoint_best.pt")

# 3. Cấu hình Vocoder (Target: Tiếng Việt)
VOCODER_CKPT = os.path.join(BASE_DIR, 'checkpoints/vocoder_target/g_00027000')
VOCODER_CONFIG = os.path.join(BASE_DIR, 'checkpoints/vocoder_target/config.json')


# ==========================================
# CÁC HÀM XỬ LÝ (Từ code của bạn)
# ==========================================
def step_1_speech_to_unit(input_wav_path):
    print(f"\n>>> BƯỚC 1: Speech-to-Unit (Source)...")
    temp_dir = os.path.join(BASE_DIR, "temp_pipeline_s2u")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(os.path.join(temp_dir, "wav_input"), exist_ok=True)
    
    shutil.copy(input_wav_path, os.path.join(temp_dir, "wav_input/input.wav"))

    try:
        subprocess.run(["python", os.path.join(FAIRSEQ_CODE_DIR, "examples/wav2vec/wav2vec_manifest.py"), os.path.join(temp_dir, "wav_input"), "--dest", temp_dir, "--ext", "wav", "--valid-percent", "0"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_hubert_feature.py"), temp_dir, "train", HUBERT_CKPT, "11", "1", "0", temp_dir], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["python", os.path.join(FAIRSEQ_CODE_DIR, "examples/hubert/simple_kmeans/dump_km_label.py"), temp_dir, "train", KM_MODEL_SOURCE, "1", "0", temp_dir], check=True, stdout=subprocess.DEVNULL)

        with open(os.path.join(temp_dir, "train_0_1.km"), 'r') as f:
            source_units = f.read().strip()
        return source_units
    except Exception as e:
        return f"Lỗi Bước 1: {e}"

def step_2_translation(source_units):
    print(f"\n>>> BƯỚC 2: Translating Unit-to-Unit...")
    cmd = [
        "fairseq-interactive", DATA_BIN_U2U,
        "--path", MODEL_U2U_PATH,
        "--beam", "5",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--buffer-size", "1024",
        "--max-tokens", "4096", 
        "--max-len-a", "1.4", 
        "--max-len-b", "500"
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=source_units)

    target_units = ""
    for line in stdout.split('\n'):
        if line.startswith("H-0"):
            parts = line.split('\t')
            if len(parts) >= 3:
                target_units = parts[2].strip()
                break
    
    if target_units:
        return target_units
    else:
        return f"Lỗi Bước 2: STDERR: {stderr}"

def step_3_vocoder(target_units, output_wav_path):
    print(f"\n>>> BƯỚC 3: Vocoder (Unit-to-Speech)...")
    temp_unit_file = "temp_target_units.txt"
    with open(temp_unit_file, 'w') as f:
        f.write(target_units)

    cmd = [
        "python", os.path.join(BASE_DIR, "speech-resynthesis/infer.py"),
        "--input_file", temp_unit_file,
        "--output_file", output_wav_path,
        "--checkpoint_file", VOCODER_CKPT,
        "--config", VOCODER_CONFIG
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return f"Lỗi Bước 3: {e}"
    finally:
        if os.path.exists(temp_unit_file):
            os.remove(temp_unit_file)

# ==========================================
# CẦU NỐI VỚI FRONTEND GRADIO
# ==========================================
def gradio_s2ut_pipeline(audio_filepath):
    """
    Hàm này nhận file từ Gradio, chạy pipeline và trả về kết quả
    """
    if audio_filepath is None:
        return None, "⚠️ Vui lòng tải lên hoặc thu âm file audio tiếng Anh!"

    print(f"\n[Gradio] Bắt đầu xử lý file: {audio_filepath}")
    
    # Tạo đường dẫn file đầu ra cố định
    output_wav_path = os.path.join(BASE_DIR, "gradio_output_vi.wav")
    
    # Xóa file cũ nếu có
    if os.path.exists(output_wav_path):
        os.remove(output_wav_path)

    # --- Bước 1 ---
    src_units = step_1_speech_to_unit(audio_filepath)
    if src_units is None or "Lỗi" in src_units:
        return None, f"❌ Thất bại ở Bước 1 (Trích xuất Unit).\n{src_units}"

    # --- Bước 2 ---
    # CHÚ Ý QUAN TRỌNG: Cần Deduplicate chuỗi đầu vào trước khi cho vào dịch
    # Vì bạn đã train bằng data dedup
    src_units_dedup = " ".join([u for i, u in enumerate(src_units.split()) if i == 0 or u != src_units.split()[i-1]])
    
    tgt_units = step_2_translation(src_units_dedup)
    if tgt_units is None or "Lỗi" in tgt_units:
        return None, f"❌ Thất bại ở Bước 2 (Dịch Unit).\n{tgt_units}"

    # --- Bước 3 ---
    status = step_3_vocoder(tgt_units, output_wav_path)
    if status is not True:
        return None, f"❌ Thất bại ở Bước 3 (Sinh âm thanh).\n{status}"
    
    success_msg = f"✅ Dịch thành công!\n- Source Units: {src_units_dedup[:30]}...\n- Target Units: {tgt_units[:30]}..."
    return output_wav_path, success_msg

# ==========================================
# THIẾT KẾ GIAO DIỆN GRADIO
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🎙️ Hệ thống Dịch thuật Giọng nói Trực tiếp (Anh - Việt)</h1>")
    gr.Markdown("<p style='text-align: center;'>Dự án Speech-to-Unit Translation (S2UT) không qua văn bản trung gian.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🗣️ Giọng nói Tiếng Anh (Input)")
            translate_btn = gr.Button("🚀 Dịch sang Tiếng Việt", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="🎧 Giọng nói Tiếng Việt (Output)", interactive=False)
            output_text = gr.Textbox(label="Trạng thái & Log", lines=4)

    translate_btn.click(
        fn=gradio_s2ut_pipeline,
        inputs=input_audio,
        outputs=[output_audio, output_text]
    )

if __name__ == "__main__":
    # Cấu hình server để có thể truy cập từ Windows qua WSL
    print("\n🌐 Bật Server Frontend... Vui lòng bấm vào link http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)