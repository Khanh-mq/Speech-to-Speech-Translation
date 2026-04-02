import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding (không cần học, generalize tốt hơn Embedding)
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding — không có learnable param."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Transformer Layer  (FFT-style: MHA + Conv1d FFN)
# ---------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Conv-based FFN: kernel=9 để capture local context tốt hơn Linear
        self.conv1 = nn.Conv1d(d_model, dim_feedforward, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                src_mask=None,
                src_key_padding_mask=None) -> torch.Tensor:
        # --- Self-attention (Pre-LN style: norm trước attention) ---
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # --- Conv FFN (Pre-LN) ---
        src2 = self.norm2(src)
        src2 = src2.transpose(1, 2)                   # (B, D, T)
        src2 = F.relu(self.conv1(src2))
        src2 = self.conv2(src2)
        src2 = src2.transpose(1, 2)                   # (B, T, D)

        src = src + self.dropout2(src2)
        return src


# ---------------------------------------------------------------------------
# Variance Predictor  (Pitch / Energy / Duration)
# ---------------------------------------------------------------------------
class VariancePredictor(nn.Module):
    """Dự đoán pitch, energy, hoặc log-duration từ hidden states (B, T, H)."""
    def __init__(self, hidden_dim: int = 256, filter_size: int = 256,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1   = nn.Conv1d(hidden_dim, filter_size, kernel_size, padding=pad)
        self.conv2   = nn.Conv1d(filter_size, filter_size, kernel_size, padding=pad)
        self.norm1   = nn.LayerNorm(filter_size)
        self.norm2   = nn.LayerNorm(filter_size)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(filter_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        x = x.transpose(1, 2)                               # (B, H, T)

        x = F.relu(self.conv1(x))
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = x.transpose(1, 2)                               # (B, T, H)
        return self.linear(x).squeeze(-1)                   # (B, T)


# ---------------------------------------------------------------------------
# Length Regulator
# ---------------------------------------------------------------------------
class LengthRegulator(nn.Module):
    """Expand encoder hidden states theo duration để match mel-frame space."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor):
        """
        x         : (B, T_src, H)
        durations : (B, T_src)  int — số frame mỗi unit
        returns   : expanded (B, T_mel, H), mel_masks (B, T_mel) True=pad
        """
        outs = []
        for i in range(x.size(0)):
            # repeat_interleave chạy trên GPU nếu x trên GPU
            out = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            outs.append(out)

        max_len = max(o.size(0) for o in outs)
        H = x.size(2)
        out_padded = x.new_zeros(len(outs), max_len, H)
        mel_masks  = torch.ones(len(outs), max_len, device=x.device, dtype=torch.bool)

        for i, o in enumerate(outs):
            L = o.size(0)
            out_padded[i, :L] = o
            mel_masks[i, :L]  = False        # False = valid frame

        return out_padded, mel_masks


# ---------------------------------------------------------------------------
# FastSpeech2 Acoustic Model
# ---------------------------------------------------------------------------
class FastSpeech2AcousticModel(nn.Module):
    def __init__(self,
                 vocab_size      : int   = 1000,
                 encoder_dim     : int   = 256,
                 encoder_layers  : int   = 4,
                 encoder_heads   : int   = 2,
                 decoder_dim     : int   = 256,
                 decoder_layers  : int   = 6,
                 decoder_heads   : int   = 2,
                 n_mels          : int   = 80,
                 dropout         : float = 0.1):
        super().__init__()

        # ── ENCODER ────────────────────────────────────────────────────────
        self.unit_emb   = nn.Embedding(vocab_size, encoder_dim, padding_idx=0)
        self.encoder_pe = SinusoidalPE(encoder_dim)          # FIX: sinusoidal, bỏ nn.Embedding
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(encoder_dim, encoder_heads, dropout=dropout)
            for _ in range(encoder_layers)
        ])

        # ── VARIANCE ADAPTOR ───────────────────────────────────────────────
        self.duration_predictor = VariancePredictor(encoder_dim)
        self.pitch_predictor    = VariancePredictor(encoder_dim)
        self.energy_predictor   = VariancePredictor(encoder_dim)

        # Pitch & energy embedding: frame-level (add SAU length regulator)
        self.pitch_emb  = nn.Conv1d(1, decoder_dim, kernel_size=3, padding=1)
        self.energy_emb = nn.Conv1d(1, decoder_dim, kernel_size=3, padding=1)

        self.length_regulator = LengthRegulator()

        # ── DECODER ────────────────────────────────────────────────────────
        self.decoder_pe = SinusoidalPE(decoder_dim)          # FIX: PE riêng cho decoder
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(decoder_dim, decoder_heads, dropout=dropout)
            for _ in range(decoder_layers)
        ])

        # ── OUTPUT ─────────────────────────────────────────────────────────
        self.mel_linear = nn.Linear(decoder_dim, n_mels)

        # FIX: Postnet đầy đủ 5 block theo paper gốc
        # Block 1-4: Conv → BN → Tanh → Dropout(0.5)
        # Block 5  : Conv → Dropout(0.5)   (không Tanh)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),

            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),

            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),

            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),

            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),   # block 5
            nn.Dropout(0.5),
        )

        # Weight init
        self._init_weights()

    # ── Weight initialization ───────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self,
                units            : torch.Tensor,
                src_masks        : torch.Tensor,
                target_durations : torch.Tensor | None = None,
                target_pitch     : torch.Tensor | None = None,
                target_energy    : torch.Tensor | None = None,
                mel_masks        : torch.Tensor | None = None):
        """
        units            : (B, T_src)     LongTensor
        src_masks        : (B, T_src)     BoolTensor  True=padding
        target_durations : (B, T_src)     LongTensor  (training only)
        target_pitch     : (B, T_mel)     FloatTensor (training only)
        target_energy    : (B, T_mel)     FloatTensor (training only)
        mel_masks        : (B, T_mel)     BoolTensor  True=padding (optional)

        returns:
            mel_before   : (B, T_mel, n_mels)
            mel_after    : (B, T_mel, n_mels)
            log_dur_pred : (B, T_src)
            p_pred       : (B, T_src)
            e_pred       : (B, T_src)
            mel_masks    : (B, T_mel)
        """

        # ── 1. ENCODER ──────────────────────────────────────────────────────
        x = self.encoder_pe(self.unit_emb(units))          # (B, T_src, D)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_masks)

        # ── 2. VARIANCE PREDICTOR (unit-level) ──────────────────────────────
        log_dur_pred = self.duration_predictor(x)          # (B, T_src)
        p_pred       = self.pitch_predictor(x)             # (B, T_src)
        e_pred       = self.energy_predictor(x)            # (B, T_src)

        # Chọn duration để dùng
        if target_durations is not None:
            dur_to_use = target_durations                  # training: dùng ground-truth
        else:
            # FIX: min=1 tránh unit bị xoá hoàn toàn
            dur_to_use = torch.clamp(
                torch.round(torch.exp(log_dur_pred) - 1), min=1
            ).long()

        # # ── 3. LENGTH REGULATOR ─────────────────────────────────────────────
        # # FIX: expand TRƯỚC khi add pitch/energy embed
        # x_exp, mel_masks_inf = self.length_regulator(x, dur_to_use)
        # # x_exp: (B, T_mel, D)

        # if mel_masks is None:
        #     mel_masks = mel_masks_inf

        # # ── 4. PITCH & ENERGY EMBED (frame-level, sau expand) ───────────────
        # # FIX: pitch/energy là frame-level feature → phải add sau LR
        # if target_durations is not None:
        #     pitch_to_use  = target_pitch    # (B, T_mel)
        #     energy_to_use = target_energy   # (B, T_mel)
        # else:
        #     pitch_to_use  = p_pred          # inference: dùng prediction (B, T_src)
        #     # Expand pitch/energy prediction lên T_mel bằng cách repeat theo dur
        #     pitch_to_use  = torch.repeat_interleave(
        #         p_pred.view(-1), dur_to_use.view(-1)
        #     ).view(x.size(0), -1)[:, :x_exp.size(1)]
        #     energy_to_use = torch.repeat_interleave(
        #         e_pred.view(-1), dur_to_use.view(-1)
        #     ).view(x.size(0), -1)[:, :x_exp.size(1)]

        # p_emb = self.pitch_emb(pitch_to_use.unsqueeze(1)).transpose(1, 2)   # (B, T_mel, D)
        # e_emb = self.energy_emb(energy_to_use.unsqueeze(1)).transpose(1, 2) # (B, T_mel, D)

        # x_exp = x_exp + p_emb + e_emb

        # ── 3. LENGTH REGULATOR ─────────────────────────────────────────────
        x_exp, mel_masks_inf = self.length_regulator(x, dur_to_use)

        # ── CĂN CHỈNH CHIỀU DÀI ───────────────────────────────────
        if target_durations is not None and mel_masks is not None:
            # SỬA LỖI: Lấy chiều dài chuẩn từ mel_masks (T_mel) thay vì target_pitch
            target_len = mel_masks.size(1) 
        else:
            target_len = x_exp.size(1)

        def align_tensor(tensor, max_len):
            curr_len = tensor.size(1)
            if curr_len > max_len:
                return tensor[:, :max_len, :]  # Cắt bớt phần thừa
            elif curr_len < max_len:
                return F.pad(tensor, (0, 0, 0, max_len - curr_len)) # Pad thêm 0
            return tensor

        # Ép x_exp về đúng chiều dài Mel
        x_exp = align_tensor(x_exp, target_len)

        # Cập nhật lại mel_masks (cho Inference)
        if mel_masks is None:
            if target_durations is not None:
                mel_masks = torch.zeros(x_exp.size(0), target_len, dtype=torch.bool, device=x.device)
            else:
                mel_masks = F.pad(mel_masks_inf, (0, max(0, target_len - mel_masks_inf.size(1))), value=True)[:, :target_len]

        # ── 4. PITCH & ENERGY EMBED (Unit-level -> Frame-level) ─────────────
        # Lấy giá trị đầu vào (Dài bằng T_src)
        pitch_val  = target_pitch  if target_durations is not None else p_pred
        energy_val = target_energy if target_durations is not None else e_pred

        # Cả Train và Inference đều phải expand Pitch/Energy từ T_src lên T_mel
        pitch_expanded = []
        energy_expanded = []
        for i in range(x.size(0)):
            p_item = torch.repeat_interleave(pitch_val[i], dur_to_use[i])
            e_item = torch.repeat_interleave(energy_val[i], dur_to_use[i])
            pitch_expanded.append(p_item)
            energy_expanded.append(e_item)
            
        # Pad lại thành Tensor (B, T_mel_approx)
        pitch_expanded = torch.nn.utils.rnn.pad_sequence(pitch_expanded, batch_first=True)
        energy_expanded = torch.nn.utils.rnn.pad_sequence(energy_expanded, batch_first=True)

        # Đưa qua Conv1d Embed
        p_emb = self.pitch_emb(pitch_expanded.unsqueeze(1)).transpose(1, 2)
        e_emb = self.energy_emb(energy_expanded.unsqueeze(1)).transpose(1, 2)

        # Ép p_emb và e_emb về đúng chiều dài chuẩn T_mel
        p_emb = align_tensor(p_emb, target_len)
        e_emb = align_tensor(e_emb, target_len)

        # Cộng lại an toàn 100%
        x_exp = x_exp + p_emb + e_emb

        # ── 5. DECODER ──────────────────────────────────────────────────────
        x_exp = self.decoder_pe(x_exp)                     # FIX: dùng PE riêng

        for layer in self.decoder_layers:
            x_exp = layer(x_exp, src_key_padding_mask=mel_masks)

        # ── 6. OUTPUT ────────────────────────────────────────────────────────
        mel_before = self.mel_linear(x_exp)                # (B, T_mel, n_mels)
        mel_after  = mel_before + self.postnet(
            mel_before.transpose(1, 2)
        ).transpose(1, 2)                                  # residual connection

        return mel_before, mel_after, log_dur_pred, p_pred, e_pred, mel_masks


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, T_src, T_mel = 2, 30, 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FastSpeech2AcousticModel().to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total/1e6:.2f}M")

    units    = torch.randint(1, 100, (B, T_src)).to(device)
    src_mask = torch.zeros(B, T_src, dtype=torch.bool).to(device)
    duration = torch.ones(B, T_src, dtype=torch.long).to(device) * (T_mel // T_src)
    pitch    = torch.randn(B, T_mel).to(device)
    energy   = torch.randn(B, T_mel).to(device)

    with torch.no_grad():
        mel_b, mel_a, log_dur, p, e, mm = model(
            units, src_mask,
            target_durations=duration,
            target_pitch=pitch,
            target_energy=energy,
        )

    print(f"mel_before : {mel_b.shape}")   # (B, T_mel, 80)
    print(f"mel_after  : {mel_a.shape}")   # (B, T_mel, 80)
    print(f"log_dur    : {log_dur.shape}") # (B, T_src)
    print(f"pitch_pred : {p.shape}")       # (B, T_src)
    print(f"mel_masks  : {mm.shape}")      # (B, T_mel)
    print("OK")