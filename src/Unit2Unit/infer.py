import os
import subprocess

BASE_TARGET = '/mnt/e/AI/khanh'
DATA_BIN = os.path.join(BASE_TARGET, "data_bin_unit2unit")
MODEL_PATH = os.path.join(BASE_TARGET, "checkpoints/unit2unit_en_vi/checkpoint_best.pt")

def infer_interactive():
    """
    Chế độ tương tác:  chuỗi unit EN, máy trả về unit VN
    Ví dụ nhập: 10 20 55 102
    """
    print("--- Đang khởi động chế độ dịch Unit2Unit ---")
    print("Hãy nhập chuỗi Unit Source ")
    
    cmd = [
        "fairseq-interactive", DATA_BIN,
        "--path", MODEL_PATH,
        "--beam", "5",
        "--source-lang", "src",
        "--target-lang", "tgt"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    infer_interactive()