import numpy as np
from collections import Counter
from scipy.stats import entropy

def evaluate_kmeans_health(unit_file_path, k_clusters=500):
    """
    Tiêu chí 1: Đánh giá "Sức khỏe" thuật toán K-Means (Codebook Utilization)
    Kiểm tra hai tiêu chuẩn: Active Units (Chống sụp đổ) và Entropy (Độ đồng đều)
    """
    # 1. Quét thống kê toàn bộ tệp kết quả dự đoán
    with open(unit_file_path, 'r', encoding='utf-8') as file:
        units = file.read().strip().split()
    
    # 2. Xây dựng phân phối tần suất của từng Unit
    frequency_map = Counter(units)
    active_units_count = len(frequency_map)
    total_tokens = len(units)
    
    # Lấy ra Unit xuất hiện áp đảo nhất (Max Frequency Unit)
    most_common_unit, max_freq = frequency_map.most_common(1)[0]
    max_percentage = (max_freq / total_tokens) * 100
    
    # 3. Tính toán độ bất định tĩnh (Entropy)
    probabilities = np.array(list(frequency_map.values())) / total_tokens
    dataset_entropy = entropy(probabilities, base=2)
    max_possible_entropy = np.log2(k_clusters)
    
    # 4. Hiển thị thông số báo cáo
    print(f"--- K-Means Health Checklist ---")
    print(f"Total Predicted Tokens : {total_tokens}")
    print(f"Active Units (Alive)   : {active_units_count} / {k_clusters} clusters")
    print(f"Most Frequent Unit     : Unit ID {most_common_unit} ({max_percentage:.2f}%)")
    print(f"System Entropy Score   : {dataset_entropy:.2f} / {max_possible_entropy:.2f}")
    
    # Đánh giá cảnh báo sụp đổ
    if active_units_count < (k_clusters * 0.9):
         print("⚠️ WARNING: Codebook đang có dấu hiệu Collapse (Sụp đổ cụm).")
         
    return active_units_count, dataset_entropy

import Levenshtein
import numpy as np

def calculate_discrete_abx(unit_seq_A, unit_seq_B, unit_seq_X):
    """
    Tiêu chí 2: Đánh giá ABX Test trên chuỗi Lượng tử rời rạc (Discrete Units)
    Sử dụng khoảng cách chuẩn Levenshtein, vượt qua ranh giới dịch thời gian (DTW)
    
    - unit_seq_A, B: Chuỗi đặc trưng String rời rạc tựa nhãn (Ví dụ âm /a/ và /e/)
    - unit_seq_X: Chuỗi đặc trưng âm thanh đầu vào giả định cần thử nghiệm
    """
    # Ép kiểu cấu trúc dãy số sang String nối liền để tính Levenshtein
    str_A = "".join(map(str, unit_seq_A))
    str_B = "".join(map(str, unit_seq_B))
    str_X = "".join(map(str, unit_seq_X))
    
    # Tính khoảng cách chỉnh sửa giữa chuỗi thử nghiệm X đối với gốc A và B
    lev_distance_AX = Levenshtein.distance(str_A, str_X)
    lev_distance_BX = Levenshtein.distance(str_B, str_X)
    
    # Nếu X cách xa đặc trưng cấu trúc của A hơn B => Lượng tử hóa sinh lỗi
    token_error = 1 if lev_distance_AX > lev_distance_BX else 0
    return token_error

def evaluate_abx_phonetics(pseudo_labelled_triplet_data):
    """ Đánh giá toàn cục tập nhãn giả (Pseudo-labels Dataset) """
    total_errors = 0
    total_samples = len(pseudo_labelled_triplet_data)
    
    for triplet in pseudo_labelled_triplet_data:
        # Lấy dữ liệu Unit Sequences
        seq_A = triplet['units_A'] 
        seq_B = triplet['units_B']
        seq_X = triplet['units_X']
        
        total_errors += calculate_discrete_abx(seq_A, seq_B, seq_X)
        
    abx_error_rate = (total_errors / total_samples) * 100
    print(f"=> Discrete Phonetic ABX Error Limit: {abx_error_rate:.2f}%")
    
    return abx_error_rate

if __name__ == "__main__":
    import os
    import random
    
    print("="*50)
    print("1. TEST CHẶNG 1: K-MEANS HEALTH CHECK (MOCK DATA)")
    print("="*50)
    # Tự động tạo 1 file .km giả lập chứa 5000 ký tự unit 
    dummy_km_path = "dummy_test.km"
    mock_units = [str(random.randint(0, 480)) for _ in range(5000)] 
    with open(dummy_km_path, "w", encoding="utf-8") as f:
        f.write(" ".join(mock_units))
    
    evaluate_kmeans_health(dummy_km_path, k_clusters=500)
    
    if os.path.exists(dummy_km_path):
        os.remove(dummy_km_path)

    print("\n" + "="*50)
    print("2. TEST CHẶNG 2: PHONETIC ABX TEST (MOCK DATA)")
    print("="*50)
    
    mock_triplet_data = [
        {"units_A": [12, 14, 15], "units_B": [400, 401, 405], "units_X": [12, 14, 15, 15]},
        {"units_A": [50, 51], "units_B": [120, 120, 121], "units_X": [50, 55]},
        {"units_A": [10, 10, 11], "units_B": [90, 92], "units_X": [90, 92, 92]} 
    ]
    
    evaluate_abx_phonetics(mock_triplet_data)
