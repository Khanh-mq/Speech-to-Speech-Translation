import os
import shutil
import random

BASE = "/mnt/g/data_final"

folder_A = "wav_en"
folder_B = "wav_vi"

out_A = "source"
out_B = "target"

train_ratio = 0.8
valid_ratio = 0.1
seed = 42

def get_id(f):
    return os.path.splitext(f)[0].split("_")[-1]

dir_A = f"{BASE}/{folder_A}"
dir_B = f"{BASE}/{folder_B}"

files_A = {get_id(f): f for f in os.listdir(dir_A) if f.endswith(".wav")}
files_B = {get_id(f): f for f in os.listdir(dir_B) if f.endswith(".wav")}

common_ids = sorted(set(files_A) & set(files_B))
print(f"Tổng cặp hợp lệ: {len(common_ids)}")

random.seed(seed)
random.shuffle(common_ids)

n = len(common_ids)
n_train = int(n * train_ratio)
n_valid = int(n * valid_ratio)

splits = {
    "train": common_ids[:n_train],
    "valid": common_ids[n_train:n_train + n_valid],
    "test":  common_ids[n_train + n_valid:]
}

for split, ids in splits.items():
    os.makedirs(f"{BASE}/data/{out_A}/{split}", exist_ok=True)
    os.makedirs(f"{BASE}/data/{out_B}/{split}", exist_ok=True)

    for i in ids:
        shutil.copy(
            os.path.join(dir_A, files_A[i]),
            os.path.join(BASE, "data", out_A, split, files_A[i])
        )
        shutil.copy(
            os.path.join(dir_B, files_B[i]),
            os.path.join(BASE, "data", out_B, split, files_B[i])
        )

    print(f"{split}: {len(ids)} cặp")
print("Hoàn tất việc chia dữ liệu.")