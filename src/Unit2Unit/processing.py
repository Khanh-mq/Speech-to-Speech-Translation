import os
import shutil
import subprocess

# CẤU HÌNH ĐƯỜNG DẪN
BASE_DATA = '/mnt/g/khanh'
BASE_TARGET = os.path.join('/mnt/e/AI/khanh/checkpoints')
FEAT_DIR = os.path.join(BASE_DATA, "hubert_feats")
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit") #  lưu dữ liệu đã đóng gói

def prepare_data():
    """
    Bước 1: Đổi tên file .km thành dạng cặp song ngữ (train.src - train.tgt)
    để fairseq-preprocess hiểu được.
    """
    temp_dir = "temp_corpus"
    os.makedirs(temp_dir, exist_ok=True)

    print("--- Đang chuẩn bị dữ liệu thô... ---")
    
    # Copy và đổi tên file TRAIN
    shutil.copy(os.path.join(FEAT_DIR, "en/train_0_1.km"), os.path.join(temp_dir, "train.src"))
    shutil.copy(os.path.join(FEAT_DIR, "vn/train_0_1.km"), os.path.join(temp_dir, "train.tgt"))
    
    # Copy và đổi tên file VALID
    shutil.copy(os.path.join(FEAT_DIR, "en/valid_0_1.km"), os.path.join(temp_dir, "valid.src"))
    shutil.copy(os.path.join(FEAT_DIR, "vn/valid_0_1.km"), os.path.join(temp_dir, "valid.tgt"))
    print("--- Đang chạy fairseq-preprocess (Binarize)... ---")
    
    # Lệnh đóng gói dữ liệu
    cmd = [
        "fairseq-preprocess",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--trainpref", os.path.join(temp_dir, "train"),
        "--validpref", os.path.join(temp_dir, "valid"),
        "--destdir", DATA_BIN,
        "--workers", "4"
    ]
    
    subprocess.run(cmd, check=True)
    
    # Dọn dẹp
    shutil.rmtree(temp_dir)
    print(f"Done! Dữ liệu đã sẵn sàng tại: {DATA_BIN}")

if __name__ == "__main__":
    prepare_data()