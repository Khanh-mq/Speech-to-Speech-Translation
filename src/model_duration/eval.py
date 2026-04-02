"""
Evaluation script cho Duration Predictor.

Metrics:
  1. MAE      - mean absolute error trên repeat count (exp-domain)
  2. Acc@1    - % dự đoán đúng repeat count (sau round)
  3. Acc@2    - % dự đoán lệch <= 1
  4. MAE(log) - MSE trên log-domain (giống train loss)
  5. Distribution: histogram pred vs. ground truth

Cách chạy:
    python src/model_duration/eval.py \
        --checkpoint checkpoints/duration_predictor/best.pt \
        --manifest  src/Unit2Wav/processed_data/target/valid.manifest \
        --plot      results/duration_eval.png
"""

import argparse
import ast
import math
import os

import torch
import torch.nn.functional as F
import numpy as np

from model import ConvTransformerDurationPredictor
from dataset import DurationDataset


# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
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
    print(f"[Eval] Loaded checkpoint (epoch {ckpt.get('epoch', '?')}), "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return model, cfg


def run_length_encode(ids):
    if not ids:
        return [], []
    dedup, counts = [ids[0]], [1]
    for x in ids[1:]:
        if x == dedup[-1]:
            counts[-1] += 1
        else:
            dedup.append(x)
            counts.append(1)
    return dedup, counts


def evaluate(model, manifest_path, device, max_frames=1000):
    all_gt_log   = []   # ground-truth log duration
    all_pred_log = []   # predicted  log duration

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    skipped = 0
    for line in lines:
        try:
            item = ast.literal_eval(line)
        except Exception:
            skipped += 1
            continue

        hubert_str = item.get("hubert", "")
        if not hubert_str:
            skipped += 1
            continue

        unit_ids = list(map(int, hubert_str.split()))
        dedup, counts = run_length_encode(unit_ids)
        if len(dedup) == 0 or len(dedup) > max_frames:
            skipped += 1
            continue

        with torch.no_grad():
            units_t      = torch.tensor(dedup, dtype=torch.long).unsqueeze(0).to(device)
            padding_mask = (units_t == 0)
            pred_log = model(units_t, padding_mask=padding_mask).squeeze(0).cpu().numpy()  # (T,)

        gt_log = np.log(np.array(counts, dtype=np.float32).clip(min=1e-5))

        all_gt_log.extend(gt_log.tolist())
        all_pred_log.extend(pred_log.tolist())

    print(f"[Eval] Samples processed: {len(lines) - skipped}, skipped: {skipped}")
    return np.array(all_gt_log), np.array(all_pred_log)


def compute_metrics(gt_log, pred_log):
    # ── Log-domain ────────────────────────────────────────────────────────────
    mse_log = float(np.mean((gt_log - pred_log) ** 2))
    mae_log = float(np.mean(np.abs(gt_log - pred_log)))

    # ── Count-domain (exp) ────────────────────────────────────────────────────
    gt_count   = np.exp(gt_log)
    pred_count = np.exp(pred_log)

    mae_count  = float(np.mean(np.abs(gt_count - pred_count)))

    # Round to nearest integer
    gt_round   = np.round(gt_count).astype(int).clip(min=1)
    pred_round = np.round(pred_count).astype(int).clip(min=1)

    acc1 = float(np.mean(gt_round == pred_round))
    acc2 = float(np.mean(np.abs(gt_round - pred_round) <= 1))

    return {
        "MSE (log)":     mse_log,
        "MAE (log)":     mae_log,
        "MAE (count)":   mae_count,
        "Acc@0 (exact)": acc1,
        "Acc@1 (±1)":    acc2,
        "Total tokens":  len(gt_log),
    }


def plot_distribution(gt_log, pred_log, save_path):
    """Vẽ histogram pred vs. gt trên count-domain."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Eval] matplotlib not found, skipping plot.")
        return

    gt_count   = np.exp(gt_log).clip(0, 30)
    pred_count = np.exp(pred_log).clip(0, 30)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Histogram so sánh ────────────────────────────────────────────────────
    bins = np.arange(0.5, 31.5, 1)
    axes[0].hist(gt_count,   bins=bins, alpha=0.6, label="Ground Truth", color="steelblue")
    axes[0].hist(pred_count, bins=bins, alpha=0.6, label="Predicted",    color="tomato")
    axes[0].set_xlabel("Repeat count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution: GT vs. Predicted duration")
    axes[0].legend()

    # ── Scatter: GT vs. Pred ──────────────────────────────────────────────────
    # Subsample 5000 points for clarity
    idx = np.random.choice(len(gt_count), size=min(5000, len(gt_count)), replace=False)
    axes[1].scatter(gt_count[idx], pred_count[idx], alpha=0.3, s=5, color="steelblue")
    lim_max = max(gt_count.max(), pred_count.max())
    axes[1].plot([0, lim_max], [0, lim_max], "r--", linewidth=1, label="Perfect")
    axes[1].set_xlabel("Ground Truth count")
    axes[1].set_ylabel("Predicted count")
    axes[1].set_title("GT vs. Predicted (scatter, 5k pts)")
    axes[1].legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Duration Predictor")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/duration_predictor/best.pt")
    parser.add_argument("--manifest", type=str,
                        default="src/Unit2Wav/processed_data/target/valid.manifest")
    parser.add_argument("--plot", type=str, default="",
                        help="Path để lưu plot (để trống thì bỏ qua)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    model, cfg = load_model(args.checkpoint, device)

    print(f"[Eval] Evaluating on: {args.manifest}")
    gt_log, pred_log = evaluate(model, args.manifest, device)

    metrics = compute_metrics(gt_log, pred_log)

    print("\n" + "=" * 45)
    print("  Duration Predictor — Evaluation Results")
    print("=" * 45)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v:,}")
    print("=" * 45)

    if args.plot:
        plot_distribution(gt_log, pred_log, args.plot)


if __name__ == "__main__":
    main()
