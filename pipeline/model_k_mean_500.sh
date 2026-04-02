#  thực hiện trian pipline model từ đầu đến cuối với kmean = 500 
# thực hiện train model từ đầu đến cuối với k_mean  =  500 và thay đổi vocoder của u2w sau khi train xong mô kmean 
#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
set -e

# Tên file python của bạn
PYTHON_SCRIPT="src/Wav2Unit/train.py"

# Các file đánh dấu trạng thái
MARKER_KMEANS=".done_kmeans_target"
MARKER_QUANTIZE=".done_quantize_target"


echo "BẮT ĐẦU PIPELINE K-MEANS & QUANTIZE CHO TARGET (VN)"


# ---------------------------------------------------------
# BƯỚC 1: HUẤN LUYỆN K-MEANS
# ---------------------------------------------------------
if [ -f "$MARKER_KMEANS" ]; then
    echo "Bước K-means đã hoàn thành trước đó. Bỏ qua..."
else
    echo "Bắt đầu huấn luyện K-means trên tập Train..."
    
    # Chạy lệnh python
    python "$PYTHON_SCRIPT" --lang target --split train --kmeans
    
    # Nếu lệnh trên chạy thành công (không bị lỗi ngắt ngang), tạo file đánh dấu
    touch "$MARKER_KMEANS"
    echo "Đã hoàn thành huấn luyện K-means!"
fi

echo "--------------------------------------------------------"

# ---------------------------------------------------------
# BƯỚC 2: LƯỢNG TỬ HOÁ (QUANTIZE)
# ---------------------------------------------------------
if [ -f "$MARKER_QUANTIZE" ]; then
    echo "  Bước Quantize đã hoàn thành trước đó. Bỏ qua..."
else
    echo "Bắt đầu lượng tử hoá (Quantize) cho tất cả các tập..."
    
    # Chạy lệnh python
    python "$PYTHON_SCRIPT" --lang target --split all --quantize
    
    # Nếu lệnh trên chạy thành công, tạo file đánh dấu
    touch "$MARKER_QUANTIZE"
    echo "[DONE] Đã hoàn thành Quantize!"
fi

echo "TẤT CẢ CÁC BƯỚC ĐÃ HOÀN TẤT THÀNH CÔNG!"
