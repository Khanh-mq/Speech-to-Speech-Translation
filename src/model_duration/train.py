"""
Training script nâng cấp cho Duration Predictor.

Cải tiến so với phiên bản cũ:
  - Huber loss + weighted loss (upweight token duration cao)
  - Cosine LR schedule với warmup
  - EMA (Exponential Moving Average) weights
  - Padding mask truyền đúng vào Transformer
  - Gradient clipping

Cách chạy:
    python src/model_duration/train.py --config src/model_duration/config.json
"""

import os
import json
import copy
import argparse
import random
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model import ConvTransformerDurationPredictor
from dataset import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average of model weights — cho inference tốt hơn."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def cosine_lr_with_warmup(optimizer, step: int, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    """
    Manually update lr:
      - Linear warmup in [0, warmup_steps]
      - Cosine decay in (warmup_steps, total_steps]
    """
    if step < warmup_steps:
        scale = float(step + 1) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    base_lr = optimizer.defaults["lr"]
    new_lr = max(min_lr, base_lr * scale)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    return new_lr


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def duration_loss(
    pred_log_dur: torch.Tensor,
    target_dur: torch.Tensor,
    lengths: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    weight_threshold: int = 3,
    weight_scale: float = 2.0,
) -> torch.Tensor:
    """
    Loss trên log-domain, chỉ tính trên vị trí hợp lệ (không padding).

    Weighted: token có duration >= weight_threshold được upweight thêm
    để counter imbalanced distribution (dur=1,2 chiếm đa số).

    Args:
        pred_log_dur    : (B, T) model output (log-scale, softplus nên >= 0)
        target_dur      : (B, T) ground-truth repeat count (raw int, float tensor)
        lengths         : (B,)   độ dài thực của mỗi sequence
        loss_type       : 'huber' hoặc 'mse'
        huber_delta     : delta cho Huber loss
        weight_threshold: ngưỡng để upweight
        weight_scale    : hệ số nhân weight cho token dur >= threshold
    """
    target_log_dur = torch.log(target_dur.clamp(min=1e-5))

    B, T = pred_log_dur.shape
    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)

    # Per-token weight: upweight token có duration cao
    weights = torch.ones_like(target_dur)
    weights[target_dur >= weight_threshold] = weight_scale
    weights = weights * mask.float()

    # Normalize weight
    norm = weights.sum().clamp(min=1.0)

    if loss_type == "huber":
        per_token_loss = F.huber_loss(pred_log_dur, target_log_dur, reduction="none", delta=huber_delta)
    else:
        per_token_loss = F.mse_loss(pred_log_dur, target_log_dur, reduction="none")

    loss = (per_token_loss * weights).sum() / norm
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: torch.nn.Module, loader, device, cfg: dict) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            units     = batch["units"].to(device)
            durations = batch["durations"].to(device)
            lengths   = batch["lengths"].to(device)

            padding_mask = (units == 0)
            pred = model(units, padding_mask=padding_mask)

            loss = duration_loss(
                pred, durations, lengths,
                loss_type=cfg.get("loss_type", "huber"),
                huber_delta=cfg.get("huber_delta", 1.0),
                weight_threshold=cfg.get("dur_weight_threshold", 3),
                weight_scale=cfg.get("dur_weight_scale", 2.0),
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, valid_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = steps_per_epoch * cfg.get("warmup_epochs", 5)

    print(f"[Train] Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConvTransformerDurationPredictor(
        vocab_size           = cfg["vocab_size"],
        embedding_dim        = cfg["embedding_dim"],
        conv_channels        = cfg["conv_channels"],
        conv_layers          = cfg["conv_layers"],
        kernel_size          = cfg["kernel_size"],
        n_transformer_layers = cfg.get("n_transformer_layers", 4),
        n_heads              = cfg.get("n_heads", 4),
        ffn_dim              = cfg.get("ffn_dim", 1024),
        dropout              = cfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Tổng params: {total_params:,}")

    # ── EMA ───────────────────────────────────────────────────────────────────
    use_ema = cfg.get("use_ema", True)
    ema = EMA(model, decay=cfg.get("ema_decay", 0.999)) if use_ema else None

    # ── Optimizer (AdamW tốt hơn Adam) ────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-2)

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0
    latest_ckpt = os.path.join(cfg["checkpoint_dir"], "latest.pt")

    if os.path.exists(latest_ckpt):
        print(f"[Train] Tiếp tục từ checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt.get("global_step", start_epoch * steps_per_epoch)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        print(f"[Train] Resumed tại epoch {start_epoch}, step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            units     = batch["units"].to(device)
            durations = batch["durations"].to(device)
            lengths   = batch["lengths"].to(device)

            # LR schedule: warmup + cosine (update mỗi step)
            current_lr = cosine_lr_with_warmup(optimizer, global_step, warmup_steps, total_steps)

            optimizer.zero_grad()

            padding_mask = (units == 0)
            pred = model(units, padding_mask=padding_mask)

            loss = duration_loss(
                pred, durations, lengths,
                loss_type        = cfg.get("loss_type", "huber"),
                huber_delta      = cfg.get("huber_delta", 1.0),
                weight_threshold = cfg.get("dur_weight_threshold", 3),
                weight_scale     = cfg.get("dur_weight_scale", 2.0),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if ema:
                ema.update(model)

            total_train_loss += loss.item()
            n_batches += 1
            global_step += 1

            if (step + 1) % cfg.get("log_interval", 100) == 0:
                avg = total_train_loss / n_batches
                print(
                    f"  [Epoch {epoch+1:03d}] Step {step+1:5d}/{steps_per_epoch} | "
                    f"loss={avg:.4f} | lr={current_lr:.2e}"
                )

        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Eval trên EMA model nếu có, otherwise model thường
        eval_model = ema.shadow if ema else model
        val_loss = evaluate(eval_model, valid_loader, device, cfg)

        print(
            f"\n[Epoch {epoch+1:03d}/{cfg['num_epochs']:03d}] "
            f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}\n"
        )

        # ── Save checkpoint ────────────────────────────────────────────────────
        ckpt_data = {
            "epoch":        epoch,
            "global_step":  global_step,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_loss":     val_loss,
            "best_val_loss": best_val_loss,
            "config":       cfg,
        }
        if ema:
            ckpt_data["ema"] = ema.state_dict()

        # Latest (luôn lưu)
        torch.save(ckpt_data, latest_ckpt)

        # Periodic
        if (epoch + 1) % cfg.get("save_interval", 10) == 0:
            periodic_path = os.path.join(cfg["checkpoint_dir"], f"dur_pred_epoch_{epoch+1:04d}.pt")
            torch.save(ckpt_data, periodic_path)
            print(f"  → Checkpoint: {periodic_path}")

        # Best (dùng EMA model weights nếu có)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_data = dict(ckpt_data)
            if ema:
                # Lưu EMA weights làm "model" trong best checkpoint
                # → infer/eval sẽ load trực tiếp như model thường
                best_data["model"] = ema.state_dict()
            best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
            torch.save(best_data, best_path)
            print(f"  ★ Best model saved (val_loss={val_loss:.4f})")

    print("\n✓ Training hoàn tất!")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Advanced Duration Predictor")
    parser.add_argument(
        "--config",
        type=str,
        default="src/model_duration/config.json",
        help="Path to config JSON",
    )
    args = parser.parse_args()
    train(args.config)
