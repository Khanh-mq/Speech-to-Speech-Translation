#!/bin/bash
set -e

# ===== CẤU HÌNH =====
# Đổi tên này thành tên thư mục bạn muốn chứa môi trường (ví dụ ./env)
ENV_PATH=./ai310_env
PYTHON_VERSION=3.10

echo "Kiểm tra conda..."
if ! command -v conda &> /dev/null; then
    echo "Conda chưa được cài."
    exit 1
fi
echo "Conda OK"

# Load conda để dùng được lệnh 'conda activate' trong script
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Tạo môi trường tại $ENV_PATH (Python $PYTHON_VERSION)"


conda create -p $ENV_PATH python=$PYTHON_VERSION -y

echo "Kích hoạt môi trường..."
conda activate $ENV_PATH

echo "Fix pip version"
pip install -U "pip<24.1"

# --- GPU ---
echo "Cài PyTorch (GPU CUDA 12.1)"
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo "Cài thư viện phụ trợ"
pip install \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  soundfile \
  numpy==1.23.5 \
  npy-append-array \
  pandas \
  sentencepiece \
  tensorboardX \
  scikit-learn

echo "Clone fairseq (version ổn định)"
# Kiểm tra nếu folder fairseq đã có thì không clone lại
if [ ! -d "fairseq" ]; then
  git clone https://github.com/facebookresearch/fairseq.git
  cd fairseq
  git checkout v0.12.2
else
  echo "Folder fairseq đã tồn tại, bỏ qua clone."
  cd fairseq
fi

echo "Cài fairseq"
pip install -e .

echo "=== HOÀN TẤT ==="
echo "Để kích hoạt lần sau, hãy chạy lệnh:"
echo "   conda activate $ENV_PATH"