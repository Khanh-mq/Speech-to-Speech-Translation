import math
import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("unit_and_duration_loss")
class UnitAndDurationLoss(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.lambda_dur = 1.0 

    # --- ĐÂY LÀ PHẦN SỬA LỖI (Báo cho Fairseq biết về --label-smoothing) ---
    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)

    @classmethod
    def build_criterion(cls, args, task):
        return cls(
            task,
            args.sentence_avg,
            args.label_smoothing,
            ignore_prefix_size=getattr(args, "ignore_prefix_size", 0),
            report_accuracy=getattr(args, "report_accuracy", False),
        )
    # -----------------------------------------------------------------------

    
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss_unit, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        pred_durations = net_output[1].get("durations", None)
        
        if pred_durations is not None and "target_durations" in sample:
            # 1. Chuyển target_durations sang GPU
            target_durations = sample["target_durations"].to(pred_durations.device)
            pad_mask = sample["target"] != self.padding_idx
            
            # 2. Đồng bộ kích thước Tensor
            seq_len_target = sample["target"].size(1) 
            seq_len_dur = target_durations.size(1)    
            
            if seq_len_dur < seq_len_target:
                diff = seq_len_target - seq_len_dur
                target_durations = F.pad(target_durations, (0, diff), value=0)
            elif seq_len_dur > seq_len_target:
                target_durations = target_durations[:, :seq_len_target]
            
            # --- SỬA LỖI FP16 Ở ĐÂY ---
            # Ép kiểu target_durations cho giống y hệt kiểu của pred_durations (Half hoặc Float)
            target_durations_matched = target_durations[pad_mask].to(dtype=pred_durations.dtype)
            
            # Tính Log
            log_pred = torch.log(pred_durations[pad_mask] + 1.0)
            log_target = torch.log(target_durations_matched + 1.0)
            
            # Tính Loss MSE
            loss_dur = F.mse_loss(log_pred, log_target, reduction="sum")
            # ---------------------------
        else:
            loss_dur = torch.tensor(0.0, device=loss_unit.device, dtype=loss_unit.dtype)
        
        loss_total = loss_unit + (self.lambda_dur * loss_dur)

        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        
        logging_output = {
            "loss": loss_total.data,
            "loss_unit": loss_unit.data,
            "loss_dur": loss_dur.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss_total, sample_size, logging_output