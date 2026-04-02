"""
DurationDataset:
- Đọc file manifest (mỗi dòng là dict có 'hubert' và 'duration')
- Tính per-unit duration từ số lần lặp của unit liên tiếp
- Label: log(n_repeats) cho mỗi unit trong chuỗi de-dup

Ví dụ:
  hubert = "71 71 71 354 354 36"
  -> de-dup units  = [71, 354, 36]
  -> per-unit reps = [3, 2, 1]
  -> label         = log([3, 2, 1])
"""

import ast
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class DurationDataset(Dataset):
    def __init__(self, manifest_path, max_frames=1000):
        self.samples = []
        self.max_frames = max_frames

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = ast.literal_eval(line)
                except Exception:
                    continue

                hubert_str = item.get("hubert", "")
                if not hubert_str:
                    continue

                unit_ids = list(map(int, hubert_str.split()))

                # De-dup và tính số lần lặp
                dedup_units, durations = self._run_length_encode(unit_ids)

                if len(dedup_units) > max_frames or len(dedup_units) == 0:
                    continue

                # Shift unit IDs lên +1 để tránh xung đột với padding_idx=0
                # K-Means K=500 → IDs gốc: 0–499 → sau shift: 1–500
                # Padding token = 0 (không bao giờ là unit thật)
                shifted_units = [u + 1 for u in dedup_units]

                self.samples.append({
                    "units": torch.tensor(shifted_units, dtype=torch.long),
                    "durations": torch.tensor(durations, dtype=torch.float),
                })

        print(f"[Dataset] Loaded {len(self.samples)} samples from {manifest_path}")

    def _run_length_encode(self, ids):
        """
        Trả về (de-dup list, per-unit repeat count)
        """
        if not ids:
            return [], []

        dedup = [ids[0]]
        counts = [1]

        for idx in ids[1:]:
            if idx == dedup[-1]:
                counts[-1] += 1
            else:
                dedup.append(idx)
                counts.append(1)

        return dedup, counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Pad sequences về cùng độ dài
    """
    units = [item["units"] for item in batch]
    durations = [item["durations"] for item in batch]

    # Độ dài thực của mỗi sample (để tính loss chỉ trên vùng hợp lệ)
    lengths = torch.tensor([len(u) for u in units], dtype=torch.long)

    units_padded = pad_sequence(units, batch_first=True, padding_value=0)
    durations_padded = pad_sequence(durations, batch_first=True, padding_value=0.0)

    return {
        "units": units_padded,
        "durations": durations_padded,
        "lengths": lengths,
    }


def build_dataloaders(config):
    train_dataset = DurationDataset(config["train_manifest"])
    valid_dataset = DurationDataset(config["valid_manifest"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, valid_loader
