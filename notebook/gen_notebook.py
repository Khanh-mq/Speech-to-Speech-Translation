import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Đánh giá Chặng 1: \"Khám sức khỏe\" thuật toán K-Means\n",
    "\n",
    "Phần này nhằm mục đích phân tích chất lượng của bộ lượng tử hóa (Quantizer) dựa trên thuật toán K-Means.\n",
    "Hai tiêu chí quan trọng để khẳng định **Wav2Unit** đang hoạt động đúng hướng là:\n",
    "1. **Active Units (Chống sụp đổ):** Tối thiểu phải có 90% cụm (cluster) tham gia hoạt động (ví dụ: > 450/500 units).\n",
    "2. **Phân phối đồng đều (Entropy):** Phân phối phải tương đối trải đều. Unit được dự đoán không được quá chênh lệch, chiếm tới 30-40% tổng thể. Entropy phải đạt mức tương đối cao để chứng minh tính đa dạng của hệ thống biểu diễn.\n",
    "---\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "import random\n",
    "from collections import Counter\n",
    "from scipy.stats import entropy\n",
    "\n",
    "# Thiết lập phong cách cho biểu đồ báo cáo khoa học\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({'font.size': 12})\n",
    "\n",
    "# ==================================================================\n",
    "# GỌI ĐƯỜNG DẪN TỚI FILE .km THỰC TẾ CỦA BẠN VÀO ĐÂY\n",
    "# Ví dụ: \"/mnt/e/AI/khanh/kmeans/kmeans500/train_0_1.km\"\n",
    "# ==================================================================\n",
    "unit_file_path = \"/mnt/e/AI/khanh/src/Wav2Unit/dummy_test.km\"\n",
    "\n",
    "# [Chức năng Mock Data giả lập] \n",
    "# Trong trường hợp file chưa kịp sinh, ta tự động tạo file Mock 20000 Unit có phân phối dạng chuẩn (Gaussian) \n",
    "# để giả lập mô hình có một lượng \"Đỉnh Unit\" nhưng vẫn bị thiếu số lượng cụm (Dead Units)\n",
    "if not os.path.exists(unit_file_path):\n",
    "    print(\"⚙️ Chưa tìm thấy file Unit gốc. Đang dùng hệ thống Mock Data để vẽ đồ thị mẫu...\")\n",
    "    # Tạo phân bố chuẩn tập trung ở cụm 250, lệch chuẩn 100, và giới hạn ở 500 cụm\n",
    "    mock_units = [str(int(random.gauss(250, 100)) % 500) for _ in range(20000)]\n",
    "    \n",
    "    # Giả lập tình huống: Mô hình bị \"rỗng\" mất một số Unit do train data chưa kỹ\n",
    "    # Lọc bỏ ngẫu nhiên 30 cụm từ danh sách để báo cáo đo lường chân thực hơn\n",
    "    dead_mock_units = set([str(random.randint(0,499)) for _ in range(30)])\n",
    "    mock_units = [u for u in mock_units if u not in dead_mock_units]\n",
    "    \n",
    "    with open(unit_file_path, \"w\", encoding=\"utf-8\") as f:\n",
    "        f.write(\" \".join(mock_units))\n",
    "\n",
    "print(\"✅ Dữ liệu đã sẵn sàng để kiểm tra 'Sức khoẻ' K-Means!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --------- XỬ LÝ TOÁN HỌC VÀ LOGIC --------------\n",
    "k_clusters = 500\n",
    "with open(unit_file_path, 'r', encoding='utf-8') as f:\n",
    "    units = f.read().strip().split()\n",
    "    \n",
    "frequency_map = Counter(units)\n",
    "active_units_count = len(frequency_map)\n",
    "total_tokens = len(units)\n",
    "\n",
    "most_common_unit, max_freq = frequency_map.most_common(1)[0]\n",
    "max_percentage = (max_freq / total_tokens) * 100\n",
    "\n",
    "probabilities = np.array(list(frequency_map.values())) / total_tokens\n",
    "dataset_entropy = entropy(probabilities, base=2)\n",
    "max_possible_entropy = np.log2(k_clusters)\n",
    "\n",
    "# === BÁO CÁO KẾT QUẢ IN RA MÀN HÌNH ===\n",
    "print(\"=\"*50)\n",
    "print(\"           BÁO CÁO SỨC KHỎE K-MEANS           \")\n",
    "print(\"=\"*50)\n",
    "print(f\"🔹 Tổng số Tokens đã trích xuất: {total_tokens:,} frames\")\n",
    "print(f\"🔹 Độ phủ đơn vị (Active Units): {active_units_count} / {k_clusters} cụm\")\n",
    "print(f\"🔹 Đơn vị chiếm chóp tỷ trọng   : Unit ID [{most_common_unit}] ({max_percentage:.2f}%)\")\n",
    "print(f\"🔹 Cấp độ Entropy Hệ thống     : {dataset_entropy:.2f} / {max_possible_entropy:.2f}\")\n",
    "print(\"-\"*50)\n",
    "if active_units_count < (k_clusters * 0.9):\n",
    "    print(\"⚠️ CẢNH BÁO: Codebook đang có tỷ lệ Unit Chết (Dead Units) cao. Nguy cơ Codebook Collapse!\")\n",
    "else:\n",
    "    print(\"👍 HOÀN HẢO: Mô hình lượng tử phủ đều không gian đặc trưng. Thuật toán phân cụm thành công.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --------- TRỰC QUAN HÓA (VISUALIZATION) ĐỂ CHỤP ẢNH ĐƯA LÊN LUẬN VĂN --------------\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})\n",
    "fig.suptitle(\"Đánh giá Chất lượng Phân cụm Lượng tử (K-Means Quantization Health)\", fontsize=16, fontweight=\"bold\", y=1.05)\n",
    "\n",
    "# 👉 Bảng 1: Biểu đồ phân phối Unit (Frequency Distribution Bar Chart)\n",
    "top_k = 60 # Trích xuất top 60 Units xuất hiện nhiều nhất để vẽ tránh quá rối mắt\n",
    "top_units = frequency_map.most_common(top_k)\n",
    "x_labels = [u[0] for u in top_units]\n",
    "y_counts = [u[1] for u in top_units]\n",
    "\n",
    "sns.barplot(x=x_labels, y=y_counts, ax=ax1, color=\"#3498db\")\n",
    "ax1.set_title(f\"Biểu đồ Phân phối Tần suất (Top {top_k} Active Units)\", fontsize=14, fontweight=\"bold\")\n",
    "ax1.set_xlabel(\"Mã Unit ID (Discrete Tokens)\", fontsize=12)\n",
    "ax1.set_ylabel(\"Số lần xuất hiện (Token Count)\", fontsize=12)\n",
    "ax1.tick_params(axis='x', rotation=90, labelsize=9)\n",
    "\n",
    "# Thêm đường trung bình (Mean Line) lên phổ tần suất\n",
    "mean_freq = total_tokens / active_units_count\n",
    "ax1.axhline(mean_freq, color='r', linestyle='--', linewidth=1.5, label='Trung bình phân phối lý tưởng')\n",
    "ax1.legend()\n",
    "\n",
    "# 👉 Bảng 2: Biểu đồ phần trăm Active Units (Pie Chart)\n",
    "dead_units = k_clusters - active_units_count\n",
    "sizes = [active_units_count, dead_units]\n",
    "labels = [f'Active Units\\n({sizes[0]})', f'Dead Units\\n({sizes[1]})']\n",
    "colors = ['#2ecc71', '#e74c3c']\n",
    "explode = (0.1, 0) if dead_units > 0 else (0, 0)\n",
    "\n",
    "ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',\n",
    "        shadow=True, startangle=140, textprops={'fontsize': 12, 'fontweight': 'bold'})\n",
    "ax2.set_title(\"Tỷ lệ Kích hoạt Codebook (Utilization)\", fontsize=14, fontweight=\"bold\")\n",
    "\n",
    "# Lưu ảnh đồ thị xuất ra .png\n",
    "plt.tight_layout()\n",
    "export_path = '/mnt/e/AI/khanh/notebook/kmeans_evaluation_plot.png'\n",
    "plt.savefig(export_path, dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f\"📸 Đã xuất bản đồ thị với chất lượng 300dpi vào file:\\n=> {export_path}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('/mnt/e/AI/khanh/notebook/metrics_wav2unit.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Đã tạo Notebook metrics_wav2unit.ipynb thành công!")
