import os
import shutil
import subprocess
import argparse


BASE_DIR = "/mnt/g/data_final/data"      # Nơi chứa folder wav source/target
FAIRSEQ_DIR = "/mnt/e/AI/khanh/fairseq"          # Nơi cài đặt fairseq
BASE_TARGET = '/mnt/g/khanh'          # Nơi lưu kết quả đầu ra
MANIFEST_ROOT = os.path.join(BASE_TARGET, "manifest_temp")
BASE_ROOT =  '/mnt/e/AI/khanh'



kmean =  500

# Model checkpoints
HUBERT_CKPT = os.path.join(BASE_ROOT, "checkpoints/mhubert_base_vp_en_es_fr_it3.pt")
EN_KMEANS_MODEL = os.path.join(BASE_ROOT, "checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")
VN_KMEANS_MODEL = os.path.join(BASE_ROOT, f"kmeans/kmeans_vn_{kmean}.bin")

def generate_manifest(mode ,  target_split):
    """ Tạo file .tsv từ các file .wav """
    print(f"\n>>> Tạo Manifest cho {mode} -  split {target_split}")
    
    splits = ["train", "valid" , "test"] if target_split == "all"   else [target_split]

    lang_suffix = "en" if mode == "source" else "vn"
    
    for split in splits:
        data_path = os.path.join(BASE_DIR, mode, split)
        dest_path = os.path.join(MANIFEST_ROOT, f"{split}_{lang_suffix}")
        os.makedirs(dest_path, exist_ok=True)
        
        cmd = [
            "python", os.path.join(FAIRSEQ_DIR, "examples/wav2vec/wav2vec_manifest.py"),
            data_path, "--dest", dest_path, "--ext", "wav", "--valid-percent", "0"
        ]
        subprocess.run(cmd, check=True)
    
    # Gom file về thư mục chuẩn
    final_dir = os.path.join(MANIFEST_ROOT, mode)
    os.makedirs(final_dir, exist_ok=True)

    for split in splits:
        src_tsv =  os.path.join(MANIFEST_ROOT  ,f"{split}_{lang_suffix}")
        if os.path.exists(src_tsv): # Copy về thư mục chung
             shutil.copy(os.path.join(MANIFEST_ROOT, f"{split}_{lang_suffix}", "train.tsv") , os.path.join(final_dir, f"{split}.tsv"))
    # shutil.copy(os.path.join(MANIFEST_ROOT, f"train_{lang_suffix}", "train.tsv"), os.path.join(final_dir, "train.tsv"))
    # shutil.copy(os.path.join(MANIFEST_ROOT, f"valid_{lang_suffix}", "train.tsv"), os.path.join(final_dir, "valid.tsv"))
    print(f"--- Done Manifest: Files located in {final_dir} ---")

def extract_features(mode ,  target_split):
    """  Wav -> Hubert Features (.npy) """
    print(f"\n>>> Trích xuất Hubert Features cho {mode} - split {target_split}...")
    
    tsv_dir = os.path.join(MANIFEST_ROOT, mode)
    feat_out = os.path.join(BASE_TARGET, "hubert_feats", "en" if mode == "source" else "vn")
    os.makedirs(feat_out, exist_ok=True)

    splits = ['train' , 'valid' , 'test']  if target_split == "all"  else  [target_split]

    for split in splits:
        cmd = [
            "python", os.path.join(FAIRSEQ_DIR, "examples/hubert/simple_kmeans/dump_hubert_feature.py"),
            tsv_dir, split, HUBERT_CKPT, "11", "1", "0", feat_out, "--max_chunk", "1600000"
        ]
        subprocess.run(cmd, check=True)
    print(f"--- Done Features: Saved to {feat_out} ---")

def run_kmeans_training():
    """ Huấn luyện K-means (Chỉ cho Target) """
    print(f"\n>>> Huấn luyện K-means cho Target...")
    
    feat_dir = os.path.join(BASE_TARGET, "hubert_feats/vn")
    os.makedirs(os.path.dirname(VN_KMEANS_MODEL), exist_ok=True)
    
    cmd = [
        "python", os.path.join(FAIRSEQ_DIR, "examples/hubert/simple_kmeans/learn_kmeans.py"),
        feat_dir, "train", "1", VN_KMEANS_MODEL, f'{kmean}', "--percent", "0.1"
    ]
    subprocess.run(cmd, check=True)
    print(f"--- Done K-means: Model saved at {VN_KMEANS_MODEL} ---")

def quantize(mode ,  target_split):
    """  Features -> Unit IDs (.km) """
    print(f"\n>>> Quantization (Tạo Unit IDs) cho {mode}...")
    
    feat_dir = os.path.join(BASE_TARGET, "hubert_feats", "en" if mode == "source" else "vn")
    # km_model = EN_KMEANS_MODEL if mode == "source" else VN_KMEANS_MODEL
    
    if mode ==  'source':
        km_model =  EN_KMEANS_MODEL 
        out_dir =  os.path.join(BASE_TARGET , 'kmean_1000')
    else :
        km_model =  VN_KMEANS_MODEL
        out_dir =  os.path.join(BASE_TARGET  , f'kmean{kmean}')
    os.makedirs(out_dir, exist_ok=True)

    splits = ['train' , 'valid' , 'test'] if target_split == 'all' else [target_split]
 
    for split in splits:
        cmd = [
            "python", os.path.join(FAIRSEQ_DIR, "examples/hubert/simple_kmeans/dump_km_label.py"),
            feat_dir, split, km_model, "1", "0", out_dir
        ]
        subprocess.run(cmd, check=True)
    print(f"--- Done Quantize: Files .km generated in {out_dir} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Wav2Unit ")
    parser.add_argument("--lang", choices=["source", "target"], help="Chọn nguồn (en) hoặc đích (vn)")
    
    parser.add_argument("--split"  , default='all' ,  choices=['train' , 'valid' , 'test' , 'all'] ,  help="chon tập dữ liệu để để chạy , mặc định là chạy hết")


    # Các cờ hiệu để chạy lẻ từng bước
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ pipeline")
    parser.add_argument("--manifest", action="store_true", help="Chỉ chạy tạo Manifest")
    parser.add_argument("--feature", action="store_true", help="Chỉ chạy trích xuất Features")
    parser.add_argument("--kmeans", action="store_true", help="Chỉ huấn luyện K-means (cho target)")
    parser.add_argument("--quantize", action="store_true", help="Chỉ chạy Quantization")

    args = parser.parse_args()

    # Thực thi dựa trên flags
    if args.all or args.manifest:
        generate_manifest(args. lang , args.split)

    if args.all or args.feature:
        extract_features(args.lang ,  args.split)

    if (args.all or args.kmeans) and args.lang == "target":
        if args.split ==  'test':
            print("skip : chỉ huấn luyên trên tập train laoij bo viws tập test " )
        else :
            run_kmeans_training()

    if args.all or args.quantize:
        quantize(args.lang  ,args.split)

    print("\n[FINISH] Hoàn thành các tác vụ được yêu cầu.")