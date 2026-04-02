import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerDecoder

@register_model("transformer_with_duration")
class TransformerWithDurationModel(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # Sử dụng Decoder đã được "độ" lại của chúng ta
        return TransformerDecoderWithDuration(
            args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False)
        )

class TransformerDecoderWithDuration(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        # --- MẠNG DỰ BÁO ĐỘ DÀI (DURATION PREDICTOR) ---
        embed_dim = self.output_embed_dim
        self.duration_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),  # NÂNG CẤP 1: Thêm LayerNorm để ổn định Gradient
            nn.ReLU(),
            nn.Dropout(p=getattr(args, 'dropout', 0.1)),
            nn.Linear(256, 1)
        )

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, features_only=False, **kwargs):
        
        # Lọc bỏ các tham số thừa mà Fairseq Task tự động truyền vào
        kwargs.pop("src_lengths", None)
        kwargs.pop("return_all_hiddens", None)

        # 1. Chạy quá trình giải mã gốc của Transformer để lấy hidden states (x)
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **kwargs,
        )
        
        # 2. Dự đoán xác suất của các mã Unit (từ vựng)
        if not features_only:
            x_logits = self.output_layer(x)
        else:
            x_logits = x
            
        # 3. DỰ ĐOÁN ĐỘ DÀI
        predicted_durations = self.duration_predictor(x)
        
        # NÂNG CẤP 2: Ép kiểu về .float() trước khi dùng softplus để chống lỗi fp16/bf16
        predicted_durations = nn.functional.softplus(predicted_durations.float()).squeeze(-1)
        
        # Nhét kết quả dự đoán độ dài vào biến extra để truyền ra ngoài cho hàm Loss
        if extra is None:
            extra = {}
        extra["durations"] = predicted_durations
        
        return x_logits, extra


# --- KIẾN TRÚC 1: BIG (Dễ bị Overfitting nếu data ít) ---
@register_model_architecture("transformer_with_duration", "transformer_big_with_duration")
def transformer_big_with_duration(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

# --- NÂNG CẤP 3: KIẾN TRÚC BASE (Chống Overfitting cực tốt cho dataset < 100k câu) ---
@register_model_architecture("transformer_with_duration", "transformer_base_with_duration")
def transformer_base_with_duration(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)