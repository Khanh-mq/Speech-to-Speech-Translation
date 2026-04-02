"""
infer_v4_dur.py — Inference script cho model train_v4_dur.py

Model này (TransformerWithDuration) output 2 thứ:
  1. Chuỗi Unit IDs tiếng Việt (đã dedup)
  2. Duration của mỗi unit (số lần lặp) để expand lại thành chuỗi raw

Pipeline:
  Input: Chuỗi Unit IDs tiếng Anh (đã dedup)
       │
       ▼ Transformer + Duration Predictor
  Output 1: Unit IDs tiếng Việt (dedup)   ← ví dụ: "14 3 256 78 12"
  Output 2: Duration tương ứng            ← ví dụ: " 3  2   4  1  2"
       │
       ▼ Expand (lặp lại theo duration)
  Output cuối: Raw Unit IDs               ← "14 14 14 3 3 256 256 256 256 78 12 12"
       │
       ▼ Lưu vào file
  final/unit2wav/target/predicted_unit.txt (raw, để Unit2Wav dùng)
  final/unit2wav/target/predicted_unit_dedup.txt (dedup, để debug)
"""

import os
import torch
import sys

# --- CẤU HÌNH ---
BASE_DIR = '/mnt/e/AI/khanh'

# Thư mục data_bin (cần để load dict)
DATA_BIN = os.path.join(BASE_DIR, 'checkpoints/data_bin_unit2unit_Dedup_Dur')

# Checkpoint của model train_v4_dur
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints/unit2unit_Dedup_Dur_v1/checkpoint_best.pt')

# Thư mục custom code (chứa custom_task.py, custom_model.py, ...)
USER_DIR = os.path.join(BASE_DIR, 'custom_u2u')

# Thư mục output
OUTPUT_DIR = os.path.join(BASE_DIR, 'final/unit2wav/target')
OUTPUT_RAW_FILE = os.path.join(OUTPUT_DIR, 'predicted_unit.txt')             # Raw (expanded) — dùng cho Unit2Wav
OUTPUT_DEDUP_FILE = os.path.join(OUTPUT_DIR, 'predicted_unit_dedup.txt')     # Dedup — để debug
OUTPUT_DUR_FILE = os.path.join(OUTPUT_DIR, 'predicted_duration.txt')         # Duration — để debug


def expand_units_with_duration(unit_str: str, dur_str: str) -> str:
    """
    Expand chuỗi unit đã dedup về chuỗi raw bằng cách lặp mỗi unit theo duration.
    
    Ví dụ:
        unit_str = "14 3 256 78 12"
        dur_str  = "3 2 4 1 2"
        → "14 14 14 3 3 256 256 256 256 78 12 12"
    """
    units = unit_str.strip().split()
    durs = dur_str.strip().split()
    
    if len(units) != len(durs):
        print(f"[CẢNH BÁO] Số unit ({len(units)}) ≠ số duration ({len(durs)}). "
              f"Sẽ dùng min của 2 bên.")
    
    result = []
    for u, d in zip(units, durs):
        count = max(1, int(round(float(d))))  # Đảm bảo ít nhất 1 lần
        result.extend([u] * count)
    
    return " ".join(result)


def load_model():
    """
    Load model custom translation_with_duration từ checkpoint.
    Trả về: (task, model, generator, src_dict, tgt_dict)
    """
    print(f">>> Đang load model từ: {CHECKPOINT_PATH}")
    
    # Import fairseq sau khi đặt user_dir
    from fairseq import checkpoint_utils, options, tasks
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    
    # Thêm user_dir vào sys.path để fairseq tìm thấy custom code
    if USER_DIR not in sys.path:
        sys.path.insert(0, os.path.dirname(USER_DIR))
    
    # Import custom modules (đăng ký vào fairseq registry)
    sys.path.insert(0, BASE_DIR)
    import custom_u2u  # noqa: trigger __init__.py

    # Load model từ checkpoint
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        filenames=[CHECKPOINT_PATH],
        arg_overrides={
            "data": DATA_BIN,
            "user_dir": USER_DIR,
        }
    )
    
    model = models[0]
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print(">>> Dùng GPU để inference")
    else:
        print(">>> Dùng CPU để inference (chậm hơn)")
    
    # Tạo generator với beam search
    generator = task.build_generator(
        models,
        cfg.generation if hasattr(cfg, 'generation') else cfg,
    )
    
    return task, model, generator, task.source_dictionary, task.target_dictionary


def infer_single(unit_str_en: str, task, model, generator, src_dict, tgt_dict, beam: int = 5):
    """
    Dịch 1 chuỗi Unit IDs tiếng Anh (đã dedup) sang Unit IDs tiếng Việt.
    
    Args:
        unit_str_en: Chuỗi unit IDs tiếng Anh dedup, VD: "26 55 25 11 6 38"
        beam: Beam search width
    
    Returns:
        (dedup_units_vi, durations, raw_units_vi)
    """
    from fairseq import utils
    
    # --- 1. Tokenize input ---
    tokens = src_dict.encode_line(
        unit_str_en.strip(),
        add_if_not_exist=False,
        append_eos=True
    ).long()
    
    # Tạo batch (1 câu)
    sample = {
        'id': torch.LongTensor([0]),
        'net_input': {
            'src_tokens': tokens.unsqueeze(0),
            'src_lengths': torch.LongTensor([tokens.numel()]),
        }
    }
    
    if torch.cuda.is_available():
        sample = utils.move_to_cuda(sample)
    
    # --- 2. Generate (beam search) ---
    with torch.no_grad():
        # Bỏ qua prefix token nếu có
        hypos = task.inference_step(generator, [model], sample)
    
    # Lấy hypothesis tốt nhất (beam[0])
    best_hypo = hypos[0][0]
    
    # Decode Unit IDs tiếng Việt (bỏ EOS)
    tgt_tokens = best_hypo['tokens']
    if tgt_tokens[-1] == tgt_dict.eos():
        tgt_tokens = tgt_tokens[:-1]
    
    dedup_units_vi = tgt_dict.string(tgt_tokens, bpe_symbol=None)
    
    # --- 3. Lấy Duration từ model ---
    # Chạy lại forward pass để lấy extra["durations"]
    with torch.no_grad():
        encoder_out = model.encoder(
            sample['net_input']['src_tokens'],
            sample['net_input']['src_lengths']
        )
        
        # Dùng tgt_tokens làm prev_output_tokens (teacher-forcing 1 bước)
        prev_tokens = torch.cat([
            tgt_tokens.new_full((1,), tgt_dict.eos()),  # BOS = EOS token
            tgt_tokens
        ]).unsqueeze(0)
        
        if torch.cuda.is_available():
            prev_tokens = prev_tokens.cuda()
        
        _, extra = model.decoder(
            prev_output_tokens=prev_tokens,
            encoder_out=encoder_out,
        )
    
    # extra["durations"] shape: (1, seq_len) — đã qua softplus nên luôn dương
    dur_tensor = extra["durations"][0, :len(tgt_tokens)]  # Cắt đúng độ dài
    durations = [max(1, int(round(d.item()))) for d in dur_tensor]
    dur_str = " ".join(str(d) for d in durations)
    
    # --- 4. Expand units theo duration ---
    raw_units_vi = expand_units_with_duration(dedup_units_vi, dur_str)
    
    return dedup_units_vi, dur_str, raw_units_vi


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inference cho model Unit2Unit với Duration Predictor (train_v4_dur)")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Chuỗi Unit IDs tiếng Anh (đã dedup). Nếu không truyền, đọc từ stdin."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Đường dẫn file chứa Unit IDs tiếng Anh (đọc dòng đầu tiên)."
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=5,
        help="Beam search width (mặc định: 5)"
    )
    parser.add_argument(
        "--no_expand",
        action="store_true",
        help="Không expand duration — chỉ lưu units dedup (để test nhanh)"
    )
    args = parser.parse_args()

    # --- Lấy input ---
    if args.input:
        unit_str_en = args.input
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            unit_str_en = f.readline().strip()
    else:
        print("Nhập chuỗi Unit IDs tiếng Anh (đã dedup), nhấn Enter:")
        unit_str_en = input().strip()

    if not unit_str_en:
        print("[LỖI] Không có input!")
        return

    print(f"\n>>> Input (EN units dedup): {unit_str_en[:80]}...")
    print(f">>> Số units đầu vào: {len(unit_str_en.split())}")

    # --- Load model ---
    task, model, generator, src_dict, tgt_dict = load_model()

    # --- Inference ---
    print("\n>>> Đang dịch...")
    dedup_vi, dur_str, raw_vi = infer_single(
        unit_str_en, task, model, generator, src_dict, tgt_dict, beam=args.beam
    )

    # --- Hiển thị kết quả ---
    print(f"\n{'='*60}")
    print(f"[Dedup VI units] ({len(dedup_vi.split())} units): {dedup_vi[:80]}...")
    print(f"[Duration]       ({len(dur_str.split())} values): {dur_str[:80]}...")
    print(f"[Raw VI units]   ({len(raw_vi.split())} units):  {raw_vi[:80]}...")
    print(f"{'='*60}")

    # --- Lưu output ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not args.no_expand:
        with open(OUTPUT_RAW_FILE, 'w') as f:
            f.write(raw_vi + "\n")
        print(f"\n>>> Đã lưu Raw units (cho Unit2Wav): {OUTPUT_RAW_FILE}")
    
    with open(OUTPUT_DEDUP_FILE, 'w') as f:
        f.write(dedup_vi + "\n")
    
    with open(OUTPUT_DUR_FILE, 'w') as f:
        f.write(dur_str + "\n")
    
    print(f">>> Đã lưu Dedup units (debug):      {OUTPUT_DEDUP_FILE}")
    print(f">>> Đã lưu Duration (debug):          {OUTPUT_DUR_FILE}")
    print("\n[XONG] Bước tiếp theo: chạy Unit2Wav để tổng hợp giọng nói.")
    print(f"       python src/Unit2Wav/infer.py --lang target")


if __name__ == "__main__":
    main()
