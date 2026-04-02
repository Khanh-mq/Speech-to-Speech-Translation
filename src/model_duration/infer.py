"""
Inference script cho Duration Predictor.

Nhận chuỗi unit (de-dup hoặc raw), predict duration rồi expand thành full sequence.

Cách dùng:
    python src/model_duration/infer.py \
        --checkpoint checkpoints/duration_predictor/best.pt \
        --input final/unit2wav/target/predicted_unit_dedup.txt \
        --output final/unit2wav/target/predicted_duration.txt
"""

import argparse

import torch

from model import ConvTransformerDurationPredictor


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = ConvTransformerDurationPredictor(
        vocab_size           = cfg["vocab_size"],
        embedding_dim        = cfg["embedding_dim"],
        conv_channels        = cfg["conv_channels"],
        conv_layers          = cfg["conv_layers"],
        kernel_size          = cfg["kernel_size"],
        n_transformer_layers = cfg.get("n_transformer_layers", 4),
        n_heads              = cfg.get("n_heads", 4),
        ffn_dim              = cfg.get("ffn_dim", 1024),
        dropout              = 0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def predict_durations(model, unit_ids, device):
    """unit_ids: list[int] — de-duped, raw IDs (0–499). Will be shifted +1 internally."""
    with torch.no_grad():
        # Shift +1 giống như trong dataset.py
        shifted = [u + 1 for u in unit_ids]
        units = torch.tensor(shifted, dtype=torch.long).unsqueeze(0).to(device)
        padding_mask = (units == 0)
        log_dur = model(units, padding_mask=padding_mask)
        dur = torch.exp(log_dur).squeeze(0)
        durations = dur.round().long().clamp(min=1).tolist()
    return durations


def run_length_encode(ids):
    if not ids:
        return [], []
    dedup = [ids[0]]
    counts = [1]
    for x in ids[1:]:
        if x == dedup[-1]:
            counts[-1] += 1
        else:
            dedup.append(x)
            counts.append(1)
    return dedup, counts


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")
    print(f"[Infer] Loading checkpoint: {args.checkpoint}")

    model, cfg = load_model(args.checkpoint, device)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    all_durations = []

    for i, line in enumerate(lines):
        raw_ids = list(map(int, line.split()))

        if args.dedup:
            unit_ids, _ = run_length_encode(raw_ids)
        else:
            unit_ids = raw_ids

        durations = predict_durations(model, unit_ids, device)
        dur_str = " ".join(map(str, durations))
        all_durations.append(dur_str)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(lines)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for line in all_durations:
            f.write(line + "\n")

    print(f"\n[Done] Saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duration Predictor Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dedup", action="store_true",
                        help="Bat neu input la raw units (chua de-dup)")
    args = parser.parse_args()
    infer(args)
