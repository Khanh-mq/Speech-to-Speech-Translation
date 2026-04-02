import os
import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import FairseqDataset  # <-- IMPORT BẮT BUỘC

@register_task("translation_with_duration")
class TranslationWithDurationTask(TranslationTask):
    
    @classmethod
    def add_args(cls, parser):
        TranslationTask.add_args(parser)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return super(TranslationWithDurationTask, cls).setup_task(args, **kwargs)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch, combine, **kwargs)
        
        # Lấy config an toàn (hỗ trợ cả self.cfg và self.args)
        config = getattr(self, "cfg", getattr(self, "args", None))
        data_path = config.data
        if isinstance(data_path, list):
            data_path = data_path[0]
        
        dur_path = os.path.join(data_path, f"{split}.dur")
        
        durations_list = []
        if os.path.exists(dur_path):
            with open(dur_path, "r", encoding="utf-8") as f:
                for line in f:
                    dur_vals = [int(x) for x in line.strip().split()]
                    durations_list.append(torch.LongTensor(dur_vals))
        else:
            print(f"WARNING: Không tìm thấy file {dur_path}. Đảm bảo bạn đã copy file .dur vào DATA_BIN.")
        
        original_dataset = self.datasets[split]
        self.datasets[split] = CustomDurationDataset(original_dataset, durations_list)

# --- SỬA LỖI Ở ĐÂY: Kế thừa FairseqDataset ---
class CustomDurationDataset(FairseqDataset):
    def __init__(self, original_dataset, durations_list):
        self.original_dataset = original_dataset
        self.durations_list = durations_list

    def __getitem__(self, index):
        item = self.original_dataset[index]
        if index < len(self.durations_list):
            item["target_durations"] = self.durations_list[index]
        return item

    def __len__(self):
        return len(self.original_dataset)

    def collater(self, samples):
        batch = self.original_dataset.collater(samples)
        if len(samples) == 0 or batch is None:
            return batch
            
        target_durations = [s["target_durations"] for s in samples if "target_durations" in s]
        
        if target_durations and len(target_durations) == len(samples):
            max_len = max(len(d) for d in target_durations)
            padded_durations = torch.zeros((len(samples), max_len), dtype=torch.long)
            for i, dur in enumerate(target_durations):
                if len(dur) > 0:
                    padded_durations[i, :len(dur)] = dur
                
            batch["target_durations"] = padded_durations
            
        return batch

    # --- THÊM CÁC HÀM BẮT BUỘC CỦA FAIRSEQ DATASET (Ủy quyền cho dataset gốc) ---
    def num_tokens(self, index):
        return self.original_dataset.num_tokens(index)

    def size(self, index):
        return self.original_dataset.size(index)

    @property
    def sizes(self):
        return self.original_dataset.sizes

    def ordered_indices(self):
        return self.original_dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.original_dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        if hasattr(self.original_dataset, "prefetch"):
            self.original_dataset.prefetch(indices)
            