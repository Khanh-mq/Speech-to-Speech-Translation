import os

# Optimize CUDA memory allocation to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging

from model import FastSpeech2AcousticModel

# --- CONFIG ---
DATA_DIR        = '/mnt/g/khanh/Unit2Mel/processed_data'
CKPT_DIR        = "/mnt/e/AI/khanh/checkpoints/Unit2Mel"
BATCH_SIZE      = 4
GRAD_ACCUM_STEPS= 8       # Effective batch size = 32
LEARNING_RATE   = 1e-4
WARMUP_STEPS    = 4000    # Transformer warmup
EPOCHS          = 1000
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP         = torch.cuda.is_available()
MAX_GRAD_NORM   = 1.0
MAX_MEL_LEN     = 1200    # Filter sequences longer than this (chống OOM)
SAVE_EVERY      = 10      # Lưu checkpoint mỗi N epoch
LOG_EVERY       = 10      # Log lên TensorBoard mỗi N step


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    # FIX: lấy root logger và xoá handler cũ trước khi add mới
    # (basicConfig bị ignore nếu logger đã có handler)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh  = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Unit2MelDataset(Dataset):
    def __init__(self, data_dir, split="train", max_mel_len=MAX_MEL_LEN):
        self.data_dir = data_dir
        print(f"[{split}] Đang load dataset...", flush=True)

        split_file = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"  Không tìm thấy {split_file}. Load toàn bộ file trong thư mục mel.")
            all_files = glob.glob(os.path.join(data_dir, "mel", "*.npy"))
        else:
            with open(split_file, "r") as f:
                base_ids = [l.strip() for l in f if l.strip()]
            all_files = [os.path.join(data_dir, "mel", f"{b}.npy") for b in base_ids]

        # Lọc file không tồn tại
        all_files = [f for f in all_files if os.path.exists(f)]
        print(f"  Tìm thấy {len(all_files)} files. Đang lọc theo độ dài...", flush=True)

        # FIX: cache kết quả filter để lần sau không scan lại 73k file
        cache_path = os.path.join(data_dir, f".filter_cache_{split}_{max_mel_len}.txt")
        self.mel_files = self._filter_by_length(all_files, max_mel_len, cache_path)

        n_removed = len(all_files) - len(self.mel_files)
        print(f"  [{split}] Lọc xong: {len(self.mel_files)} files dùng được"
              + (f" ({n_removed} bị loại do > {max_mel_len} frames)" if n_removed else ""),
              flush=True)

    @staticmethod
    def _filter_by_length(files, max_len, cache_path=None):
        # Dùng cache nếu có — tránh scan lại 73k file mỗi lần chạy
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = set(l.strip() for l in f if l.strip())
            valid = [f for f in files if f in cached]
            print(f"  Dùng cache filter ({len(valid)} files). Xoá {cache_path} để scan lại.",
                  flush=True)
            return valid

        valid = []
        for f in tqdm(files, desc="  Scanning mel length", ncols=80, leave=True):
            try:
                mel = np.load(f, mmap_mode='r')
                if mel.shape[0] <= max_len:
                    valid.append(f)
            except Exception:
                pass

        # Lưu cache
        if cache_path:
            with open(cache_path, 'w') as cf:
                cf.write('\n'.join(valid))
            print(f"  Cache đã lưu: {cache_path}", flush=True)
        return valid

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        base_name = os.path.basename(self.mel_files[idx])
        d = self.data_dir

        mel      = np.load(os.path.join(d, "mel",      base_name))
        f0       = np.load(os.path.join(d, "f0",       base_name))
        energy   = np.load(os.path.join(d, "energy",   base_name))
        units    = np.load(os.path.join(d, "unit",     base_name))
        duration = np.load(os.path.join(d, "duration", base_name))

        return {
            "mel":      torch.FloatTensor(mel),
            "f0":       torch.FloatTensor(f0),
            "energy":   torch.FloatTensor(energy),
            "units":    torch.LongTensor(units),
            "duration": torch.LongTensor(duration),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
    pad = nn.utils.rnn.pad_sequence

    mels      = [x["mel"]      for x in batch]
    units     = [x["units"]    for x in batch]
    f0s       = [x["f0"]       for x in batch]
    energies  = [x["energy"]   for x in batch]
    durations = [x["duration"] for x in batch]

    mel_padded      = pad(mels,      batch_first=True, padding_value=0.0)
    units_padded    = pad(units,     batch_first=True, padding_value=0)
    f0_padded       = pad(f0s,       batch_first=True, padding_value=0.0)
    energy_padded   = pad(energies,  batch_first=True, padding_value=0.0)
    duration_padded = pad(durations, batch_first=True, padding_value=0)

    # True = padding position
    src_masks = (units_padded == 0)
    mel_masks = torch.zeros(mel_padded.shape[:2], dtype=torch.bool)
    for i, m in enumerate(mels):
        mel_masks[i, m.shape[0]:] = True   # mark padding

    return {
        "mel":       mel_padded,
        "units":     units_padded,
        "f0":        f0_padded,
        "energy":    energy_padded,
        "duration":  duration_padded,
        "src_masks": src_masks,
        "mel_masks": mel_masks,   # ← thêm mới, dùng để mask mel loss
    }


# ---------------------------------------------------------------------------
# Loss  (có mask padding)
# ---------------------------------------------------------------------------
class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _masked_l1(pred, target, mask):
        """
        pred, target : (B, T, D)  hoặc (B, T)
        mask         : (B, T)  True = padding
        """
        loss = F.l1_loss(pred, target, reduction='none')
        if loss.dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(loss)
        loss = loss.masked_fill(mask, 0.0)
        n_valid = (~mask).sum().clamp(min=1)
        return loss.sum() / n_valid

    @staticmethod
    def _masked_mse(pred, target, mask):
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.masked_fill(mask, 0.0)
        n_valid = (~mask).sum().clamp(min=1)
        return loss.sum() / n_valid

    def forward(self, mel_before, mel_after,
                log_dur_pred, p_pred, e_pred,
                target_mel, target_dur, target_p, target_e,
                src_masks, mel_masks):
        """
        src_masks : (B, T_src)  — mask cho encoder (unit positions)
        mel_masks : (B, T_mel)  — mask cho decoder (mel positions)
        """
        # --- Mel loss: mask vùng padding của mel ---
        mel_loss = (self._masked_l1(mel_before, target_mel, mel_masks)
                  + self._masked_l1(mel_after,  target_mel, mel_masks))

        # --- Duration loss: mask vùng padding của unit ---
        target_log_dur = torch.log(target_dur.float() + 1)
        dur_loss = self._masked_mse(log_dur_pred, target_log_dur, src_masks)

        # --- Pitch & Energy loss: mask vùng padding của unit ---
        p_loss = self._masked_mse(p_pred, target_p, src_masks)
        e_loss = self._masked_mse(e_pred, target_e, src_masks)

        total = mel_loss + dur_loss + p_loss + e_loss
        return total, mel_loss, dur_loss, p_loss, e_loss


# ---------------------------------------------------------------------------
# LR Scheduler: Transformer warmup (theo paper gốc)
# ---------------------------------------------------------------------------
class TransformerLRScheduler:
    """lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    def __init__(self, optimizer, d_model=256, warmup_steps=WARMUP_STEPS):
        self.optimizer   = optimizer
        self.d_model     = d_model
        self.warmup      = warmup_steps
        self._step       = 0

    def step(self):
        self._step += 1
        lr = (self.d_model ** -0.5) * min(
            self._step ** -0.5,
            self._step * (self.warmup ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def find_latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "fs2_epoch_*.pt")))
    return ckpts[-1] if ckpts else None

def save_checkpoint(path, epoch, step, model, optimizer, scheduler, val_loss):
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": scheduler._step,
        "val_loss":  val_loss,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, logger):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler._step = ckpt.get("scheduler_step", 0)
    logger.info(f"Resumed from: {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"] + 1, ckpt["step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    logger  = setup_logger(CKPT_DIR)
    writer  = SummaryWriter(os.path.join(CKPT_DIR, "runs"))

    # --- Data ---
    train_dataset = Unit2MelDataset(DATA_DIR, split="train")
    val_dataset   = Unit2MelDataset(DATA_DIR, split="val")

    if len(train_dataset) == 0:
        logger.error(f"Không tìm thấy dữ liệu trong {DATA_DIR}!")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )

    # --- Model, Optimizer, Loss ---
    model     = FastSpeech2AcousticModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0,  # lr do scheduler quản lý
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    criterion = FastSpeech2Loss()
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    scheduler = TransformerLRScheduler(optimizer, d_model=256, warmup_steps=WARMUP_STEPS)

    # --- Resume ---
    start_epoch = 1
    step        = 0
    best_val    = float('inf')

    latest = find_latest_checkpoint(CKPT_DIR)
    if latest:
        start_epoch, step = load_checkpoint(latest, model, optimizer, scheduler, logger)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params/1e6:.2f}M  |  Device: {DEVICE}")
    logger.info(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")
    logger.info(f"Start epoch: {start_epoch}  |  Step: {step}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, batch in enumerate(pbar):
            mel       = batch["mel"].to(DEVICE, non_blocking=True)
            units     = batch["units"].to(DEVICE, non_blocking=True)
            f0        = batch["f0"].to(DEVICE, non_blocking=True)
            energy    = batch["energy"].to(DEVICE, non_blocking=True)
            duration  = batch["duration"].to(DEVICE, non_blocking=True)
            src_masks = batch["src_masks"].to(DEVICE, non_blocking=True)
            mel_masks = batch["mel_masks"].to(DEVICE, non_blocking=True)

            try:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    mel_b, mel_a, log_dur, p, e, mel_masks_model = model(
                        units=units, src_masks=src_masks,
                        target_durations=duration,
                        target_pitch=f0,
                        target_energy=energy,
                        mel_masks=mel_masks,
                    )

                    loss, l_mel, l_dur, l_p, l_e = criterion(
                        mel_b, mel_a, log_dur, p, e,
                        mel, duration, f0, energy,
                        src_masks, mel_masks,
                    )
                    loss = loss / GRAD_ACCUM_STEPS

            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    logger.warning(f"OOM tại batch {i}, epoch {epoch}. Bỏ qua.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Loss={loss.item():.4f} tại step {step}. Bỏ qua batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                lr = scheduler.step()
                step += 1

                if step % LOG_EVERY == 0:
                    writer.add_scalar("Loss/train_total",    loss.item() * GRAD_ACCUM_STEPS, step)
                    writer.add_scalar("Loss/train_mel",      l_mel.item(),  step)
                    writer.add_scalar("Loss/train_duration", l_dur.item(),  step)
                    writer.add_scalar("Loss/train_pitch",    l_p.item(),    step)
                    writer.add_scalar("Loss/train_energy",   l_e.item(),    step)
                    writer.add_scalar("Train/lr",            lr,            step)
                    writer.add_scalar("Train/grad_norm",     grad_norm,     step)

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({
                "loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}",
                "mel":  f"{l_mel.item():.4f}",
                "lr":   f"{scheduler.optimizer.param_groups[0]['lr']:.2e}",
            })

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mel       = batch["mel"].to(DEVICE, non_blocking=True)
                units     = batch["units"].to(DEVICE, non_blocking=True)
                f0        = batch["f0"].to(DEVICE, non_blocking=True)
                energy    = batch["energy"].to(DEVICE, non_blocking=True)
                duration  = batch["duration"].to(DEVICE, non_blocking=True)
                src_masks = batch["src_masks"].to(DEVICE, non_blocking=True)
                mel_masks = batch["mel_masks"].to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    mel_b, mel_a, log_dur, p, e, _ = model(
                        units=units, src_masks=src_masks,
                        target_durations=duration,
                        target_pitch=f0,
                        target_energy=energy,
                        mel_masks=mel_masks,
                    )
                    loss, _, _, _, _ = criterion(
                        mel_b, mel_a, log_dur, p, e,
                        mel, duration, f0, energy,
                        src_masks, mel_masks,
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        logger.info(
            f"Epoch {epoch:4d} | Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | Step: {step}"
        )

        # --- Lưu checkpoint ---
        if epoch % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"fs2_epoch_{epoch:04d}.pt")
            save_checkpoint(path, epoch, step, model, optimizer, scheduler, avg_val_loss)
            logger.info(f"Saved: {path}")

        # --- Lưu best model ---
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_path = os.path.join(CKPT_DIR, "fs2_best.pt")
            save_checkpoint(best_path, epoch, step, model, optimizer, scheduler, avg_val_loss)
            logger.info(f"★ Best model: val_loss={best_val:.4f} → {best_path}")

    writer.close()
    logger.info("Huấn luyện hoàn tất!")


if __name__ == "__main__":
    train()