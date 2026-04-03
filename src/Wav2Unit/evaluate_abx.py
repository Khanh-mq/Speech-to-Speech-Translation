import numpy as np
import Levenshtein
from scipy.spatial.distance import cosine
import itertools
import random
import os

try:
    from fastdtw import fastdtw
except ImportError:
    print("Vui lòng cài đặt fastdtw: pip install fastdtw")

class ABXEvaluator:
    def __init__(self):
        pass

    # ==========================================
    # 1. ĐO LƯỜNG CONTINUOUS (mHuBERT Layer 11)
    # ==========================================
    def continuous_distance(self, feat1, feat2):
        """ 
        Tính DTW + Cosine Distance giữa 2 ma trận đặc trưng số thực.
        feat1, feat2 có shape (Số_Frames, 1024)
        """
        # fastdtw yêu cầu hàm khoảng cách, ta dùng hàm cosine của scipy
        distance, path = fastdtw(feat1, feat2, dist=cosine)
        return distance

    def evaluate_continuous_abx(self, triplets):
        """
        triplets: list các tuple (feat_A, feat_B, feat_X)
        feat_A và feat_X cùng 1 gốc ngữ âm nhưng khác bối cảnh/người nói.
        feat_B khác ngữ âm hoàn toàn.
        """
        print("Đang chạy ABX trên mHuBERT Continuous Features (Dùng DTW + Cosine)...")
        errors = 0
        total = len(triplets)
        
        for idx, (A, B, X) in enumerate(triplets):
            dist_AX = self.continuous_distance(A, X)
            dist_BX = self.continuous_distance(B, X)
            
            if dist_AX > dist_BX:
                errors += 1
                
        error_rate = (errors / total) * 100
        print(f"-> mHuBERT Layer 11 ABX Error: {error_rate:.2f}%")
        return error_rate

    # ==========================================
    # 2. ĐO LƯỜNG DISCRETE (Wav2Unit K-Means)
    # ==========================================
    def discrete_distance(self, seq1, seq2):
        """ 
        Dùng khoảng cách chỉnh sửa Levenshtein cho chuỗi Unit.
        Khắc phục được DTW tự nhiên do đặc tính so khớp chuỗi.
        """
        # Ép kiểu danh sách [1, 25, 400] thành chuỗi String nối nhau kiểu "1_25_400"
        # Hoặc convert qua ký tự theo hàm chr() nếu K < 1114111 (Unicode limit) 
        # Cách chuẩn: dùng ký tự chr(unit + 1000)
        str1 = "".join([chr(u + 1000) for u in seq1])
        str2 = "".join([chr(u + 1000) for u in seq2])
        return Levenshtein.distance(str1, str2)

    def evaluate_discrete_abx(self, triplets):
        """
        triplets: list các tuple (unit_A, unit_B, unit_X)
        unit: mảng 1 chiều chứa các số Unit IDs.
        """
        print("Đang chạy ABX trên Wav2Unit Discrete Units (Dùng Levenshtein)...")
        errors = 0
        total = len(triplets)
        
        for idx, (A, B, X) in enumerate(triplets):
            dist_AX = self.discrete_distance(A, X)
            dist_BX = self.discrete_distance(B, X)
            
            if dist_AX > dist_BX:
                errors += 1
                
        error_rate = (errors / total) * 100
        print(f"-> Wav2Unit Discrete ABX Error: {error_rate:.2f}%")
        return error_rate

# =================================================================================
# MODULE LIÊN LẠC (API) VỚI DATA CỦA BẠN 
# =================================================================================
def generate_triplets_from_pseudo_labels(label_dict, num_triplets=1000):
    """
    Hàm này tự động ráp cặp ABX từ Dictionary nhãn giả.
    label_dict có cấu trúc:
    {
       "âm_a": [feature_tensor_1, feature_tensor_2, ...],
       "âm_e": [feature_tensor_3, feature_tensor_4, ...]
    }
    """
    triplets = []
    labels = list(label_dict.keys())
    
    if len(labels) < 2:
        raise ValueError("Cần ít nhất 2 nhãn ngữ âm khác nhau để tính ABX!")
        
    for _ in range(num_triplets):
        # Lấy nhãn mục tiêu (A và X) và nhãn đối chứng (B)
        label_target, label_other = random.sample(labels, 2)
        
        # Phải có ít nhất 2 mẫu của nhãn mục tiêu để bốc ra A và X
        if len(label_dict[label_target]) < 2:
            continue
            
        # Bốc ngẫu nhiên A, X từ tập target và B từ tập other
        A, X = random.sample(label_dict[label_target], 2)
        B = random.choice(label_dict[label_other])
        
        triplets.append((A, B, X))
        
    return triplets

if __name__ == "__main__":
    # Lưu ý: Cài đặt thư viện trước khi chạy: 
    # pip install numpy scipy fastdtw python-Levenshtein
    
    evaluator = ABXEvaluator()
    print("="*60)
    print(" KHỞI CHẠY QUY TRÌNH ĐO LƯỜNG ABX TEST CHUẨN KHOA HỌC")
    print("="*60)
    
    # -------------------------------------------------------------
    # BƯỚC CỦA BẠN: HÃY LOAD FILE .npy VÀ .km LÊN VÀ PHÂN ĐOẠN 
    # THEO TIMESTAMP NHÃN GIẢ (VÍ DỤ TỪ GIÂY THỨ 1.0 ĐẾN 1.5 LÀ ÂM 'A')
    # -------------------------------------------------------------
    # (Dưới đây là Mock Data mô phỏng luồng hoạt động)
    
    # Giả lập: Cắt ma trận Layer 11 Continuous
    mock_continuous_data = {
        "phoneme_a": [np.random.rand(10, 1024), np.random.rand(12, 1024), np.random.rand(11, 1024)],
        "phoneme_e": [np.random.rand(9, 1024), np.random.rand(13, 1024), np.random.rand(10, 1024)]
    }
    
    # Giả lập: Lấy chuỗi Discrete Units tương ứng
    mock_discrete_data = {
        "phoneme_a": [[10, 10, 11], [10, 12, 11, 10], [9, 10, 11]],
        "phoneme_e": [[400, 405], [401, 401, 402], [403, 405, 406]]
    }
    
    # Tạo danh sách Triplet (A, B, X)
    cont_triplets = generate_triplets_from_pseudo_labels(mock_continuous_data, num_triplets=200)
    disc_triplets = generate_triplets_from_pseudo_labels(mock_discrete_data, num_triplets=200)
    
    if len(cont_triplets) > 0 and len(disc_triplets) > 0:
        # Chạy Khảo sát Thực tế
        err_layer11 = evaluator.evaluate_continuous_abx(cont_triplets)
        err_unit = evaluator.evaluate_discrete_abx(disc_triplets)
        
        # Tính Quantization Loss
        loss = err_unit - err_layer11
        print("-" * 60)
        print(f"🎯 KẾT QUẢ: Quantization Loss = +{loss:.2f}%")
        if loss < 5.0:
            print("=> Xếp loại: XUẤT SẮC (K-Means giữ lại gần như trọn vẹn ngữ âm)")
        elif loss < 10.0:
            print("=> Xếp loại: KHÁ (Mức tiêu chuẩn ổn định cho dịch thuật S2ST)")
        else:
            print("=> Xếp loại: RỦI RO (Mô hình bị vỡ âm, hãy tăng data hoặc tăng K)")
    else:
        print("Lỗi: Không ráp được Triplets. Hãy kiểm tra lại Pseudo-labels!")
