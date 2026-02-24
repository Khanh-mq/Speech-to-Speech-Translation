#  file để kiêm tra dữ liệu 2 folder có giống nhau không
#  kiểm tra xem dữ liệu có bị  lệch nhau hay không 



import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 

base_root = '/mnt/g'

# # --- CẤU HÌNH ---
# folder_A = os.path.join(base_root, 'data_final/wav_en')
# folder_B = os.path.join(base_root, 'data_final/wav_vi')


folder_A = os.path.join(base_root, 'data_final/data/source/train')
folder_B = os.path.join(base_root, 'data_final/data/target/train')



dst_folder_A = os.path.join(base_root, 'data_final/wav_en')
dst_folder_B = os.path.join(base_root, 'data_final/wav_vi')

os.makedirs(dst_folder_A, exist_ok=True)
os.makedirs(dst_folder_B, exist_ok=True)

# --- HÀM LẤY ID ---
def get_id(filename):
    return os.path.splitext(filename)[0].split('_')[-1]

# --- 1. QUÉT FILE  ---
print("Đang quét danh sách file...")
dict_A = {get_id(f): f for f in os.listdir(folder_A) if f.endswith('.wav')} if os.path.exists(folder_A) else {}
dict_B = {get_id(f): f for f in os.listdir(folder_B) if f.endswith('.wav')} if os.path.exists(folder_B) else {}

print(f"-> Thư mục A có {len(dict_A)} file .wav")
print(f"-> Thư mục B có {len(dict_B)} file .wav")

# --- 2. TÌM KHỚP ---
common_ids = sorted(list(set(dict_A.keys()) & set(dict_B.keys())))
print(f"-> Tìm thấy {len(common_ids)} cặp file khớp nhau.")
if len(common_ids) == 0:
    print("Không tìm thấy cặp file khớp nhau. Kết thúc chương trình.")
    exit()
