"""
ConvTransformerDurationPredictor
─────────────────────────────────
Kiến trúc nâng cấp cho Duration Predictor:

  [Unit IDs]
      │
  Embedding(500, 256)
      │
  PositionalEncoding (sinusoidal)
      │
  3× ConvBlock (local feature, kernel=5, GELU)
      │
  4× TransformerEncoderLayer (4 heads, ffn=1024) ← hiểu context dài
      │
  Projection Head: Linear(256→128) → GELU → Linear(128→1)
      │
  Softplus ← buộc log_duration ≥ 0, tức exp(out) ≥ 1 (duration tối thiểu = 1)
      │
  [log_duration: (B, T)]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding (sinusoidal, không học)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Local Conv Block (inductive bias cho âm thanh)
# ─────────────────────────────────────────────────────────────────────────────

class LocalConvBlock(nn.Module):
    """
    Conv1D → LayerNorm → GELU → Dropout
    Residual connection nếu in_channels == out_channels
    """
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)"""
        # Conv expects (B, C, T)
        residual = x
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + residual  # residual


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class ConvTransformerDurationPredictor(nn.Module):
    """
    Thay thế DurationPredictor cũ (Conv1D only).

    Args:
        vocab_size           : Số lượng unit (default 500 cho K-Means K=500)
        embedding_dim        : Chiều embedding
        conv_channels        : Số channels của local conv block
        conv_layers          : Số lớp conv local (feature extraction)
        kernel_size          : Kernel size conv local
        n_transformer_layers : Số lớp Transformer
        n_heads              : Số attention heads
        ffn_dim              : Chiều FFN bên trong Transformer
        dropout              : Dropout chung
    """

    def __init__(
        self,
        vocab_size: int = 500,
        embedding_dim: int = 256,
        conv_channels: int = 256,
        conv_layers: int = 3,
        kernel_size: int = 5,
        n_transformer_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Embedding
        # vocab_size+1 vì unit IDs đã shift +1 (0–499 → 1–500).
        # Index 0 là padding, không bao giờ là unit thật.
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)

        # 2. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(embedding_dim, dropout=dropout)

        # 3. Input projection (embedding_dim → conv_channels)
        if embedding_dim != conv_channels:
            self.input_proj = nn.Linear(embedding_dim, conv_channels)
        else:
            self.input_proj = nn.Identity()

        # 4. Local conv blocks (extract local acoustic patterns)
        self.conv_blocks = nn.ModuleList([
            LocalConvBlock(conv_channels, kernel_size, dropout)
            for _ in range(conv_layers)
        ])

        # 5. Transformer encoder (understand long-range context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, C) — quan trọng!
            norm_first=True,   # Pre-norm → ổn định hơn khi train
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False,
        )

        # 6. Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(conv_channels, conv_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo weights theo Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, units: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            units        : (B, T) — unit IDs (đã de-dup, có thể padded bằng 0)
            padding_mask : (B, T) bool — True tại vị trí padding (optional)

        Returns:
            log_durations: (B, T) — predicted log duration, tất cả >= 0 (softplus)
                           → exp(output) = predicted repeat count >= 1
        """
        # Padding mask từ units nếu không truyền vào
        if padding_mask is None:
            padding_mask = (units == 0)  # padding_idx = 0

        # 1. Embedding + PE
        x = self.embedding(units)           # (B, T, embedding_dim)
        x = self.pos_enc(x)                 # (B, T, embedding_dim)
        x = self.input_proj(x)              # (B, T, conv_channels)

        # 2. Local conv blocks
        for block in self.conv_blocks:
            x = block(x)                    # (B, T, conv_channels)

        # 3. Transformer (batch_first=True, mask=padding)
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, T, conv_channels)

        # 4. Project → scalar
        out = self.proj_head(x).squeeze(-1)  # (B, T)

        # 5. Softplus: buộc output ≥ 0 → exp(out) ≥ 1 (duration không bao giờ < 1)
        log_durations = F.softplus(out)      # (B, T)

        return log_durations


# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Alias
# ─────────────────────────────────────────────────────────────────────────────

# Giữ tên cũ để eval.py / infer.py cũ không bị lỗi import
DurationPredictor = ConvTransformerDurationPredictor
