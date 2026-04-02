# Unit2Wav — Chuyển đổi Unit IDs → Waveform (Vocoder)

Module này chịu trách nhiệm huấn luyện và chạy **Vocoder** (dựa trên kiến trúc **HiFi-GAN**) để chuyển đổi chuỗi **Unit IDs (Discrete Speech Units)** sang file âm thanh **`.wav`** có thể nghe được.

Module hỗ trợ cả 2 ngôn ngữ trong pipeline Speech-to-Speech:
- **`source`** — Tiếng Anh (sử dụng Vocoder từ Fairseq gốc, đã được pretrained)
- **`target`** — Tiếng Việt (train lại Vocoder từ đầu bằng data tiếng Việt)

---

## 📁 Cấu trúc thư mục

```
src/Unit2Wav/
├── config.json          # Cấu hình kiến trúc mô hình HiFi-GAN & đường dẫn dữ liệu
├── processing.py        # Chuẩn bị dữ liệu: từ file TSV + KM → manifest
├── train.py             # Huấn luyện Vocoder (HiFi-GAN)
├── infer.py             # Sinh file âm thanh từ Unit IDs đã dự đoán
└── processed_data/      # Thư mục chứa file manifest (tự tạo khi chạy processing.py)
    ├── source/
    │   ├── train.manifest
    │   └── valid.manifest
    └── target/
        ├── train.manifest
        └── valid.manifest
```

---

## 🔄 Tổng quan Pipeline

```
[File .wav gốc]
       │
       ▼
[HuBERT Encoder + K-Means] ──► [File .km] (Unit IDs)
       │
       ▼
[File TSV Manifest]  +  [File .km]
       │
       ▼
  [processing.py]  ──────────────► [train.manifest / valid.manifest]
       │
       ▼
  [train.py] + [config.json] ──► [HiFi-GAN Checkpoint: g_xxxxxxxx, do_xxxxxxxx]
       │
       ▼
  [infer.py] + [predicted_unit.txt] ──► [File .wav kết quả]
```

---

## ⚙️ Yêu cầu trước khi chạy

### Thư viện bên ngoài
- **Framework HiFi-GAN (Speech Resynthesis)**: Phải clone repo về thư mục `speech-resynthesis/` ở thư mục gốc project.
  ```bash
  git clone https://github.com/jik876/hifi-gan speech-resynthesis
  ```
- **Fairseq** (chỉ cần cho `source` — Tiếng Anh): Phải có thư mục `fairseq/` ở thư mục gốc project (đã clone sẵn).
- **Python** >= 3.8, **PyTorch** >= 1.9, **CUDA** (khuyến nghị khi train)

### Dữ liệu đầu vào cần có
Trước khi chạy `processing.py`, cần chuẩn bị đủ:

| File | Mô tả | Ví dụ đường dẫn |
|---|---|---|
| **`train.tsv`** | File danh sách âm thanh (header + tên file .wav mỗi dòng) | `/mnt/g/khanh/manifest_temp/target/train.tsv` |
| **`train.km`** | File chứa Unit IDs từ K-Means (mỗi dòng = chuỗi số nguyên cách nhau bằng space) | `/mnt/g/khanh/kmean500/train_0_1.km` |
| **Thư mục `.wav`** | Thư mục chứa các file âm thanh gốc tương ứng | `/mnt/g/data_final/data/target/train/` |

> **Lưu ý**: File `.tsv` và `.km` phải **khớp thứ tự dòng với nhau** — dòng 1 của `.tsv` tương ứng với dòng 1 của `.km`.

### 📄 Định dạng chi tiết từng file đầu vào

**File `train.tsv`** — danh sách audio, dòng đầu là root path, các dòng sau là `tên_file.wav\tnum_frames`:
```
/mnt/g/data_final/data/target/train
sentence_000001.wav	130560
sentence_000002.wav	98304
sentence_000037.wav	45056
...
```

**File `train.km`** — mỗi dòng là chuỗi Unit IDs (số nguyên cách nhau bằng dấu cách) tương ứng với từng dòng trong `.tsv`:
```
26 55 25 11 11 6 11 38 21 28 14 12 30 11 56 42 11 32 20 33 ...
34 46 11 34 67 37 6 11 27 26 19 11 34 48 11 34 75 11 24 39 ...
56 64 33 11 32 21 45 26 11 14 20 53 26 19 11 32 50 21 11 26 ...
...
```

> File `.km` được tạo ra bởi bước **HuBERT Encoder + K-Means clustering** trước đó (xem module `S2U`).

### 📄 Định dạng file đầu ra sau `processing.py` — `train.manifest`

Mỗi dòng là một **Python dict string** (đọc bằng `ast.literal_eval`), ví dụ thực tế:

```python
{'audio': '/mnt/g/data_final/data/target/train/sentence_057076.wav', 'hubert': '26 55 25 11 11 6 11 38 21 28 14 12 30 ...', 'duration': 4.08}
{'audio': '/mnt/g/data_final/data/target/train/sentence_060386.wav', 'hubert': '34 46 11 34 67 37 6 11 27 26 19 11 34 ...', 'duration': 3.94}
{'audio': '/mnt/g/data_final/data/target/train/sentence_054992.wav', 'hubert': '34 39 11 32 50 21 11 56 42 11 32 98 ...', 'duration': 1.98}
```

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `audio` | `string` | Đường dẫn tuyệt đối tới file `.wav` gốc |
| `hubert` | `string` | Chuỗi Unit IDs cách nhau bằng dấu cách (output của K-Means) |
| `duration` | `float` | Thời lượng ước tính = `số_unit × 0.02 giây` |

---

## 🛠️ Bước 1: Chuẩn bị dữ liệu (`processing.py`)

Script này đọc file `.tsv` và `.km`, ghép cặp `(tên file wav ↔ chuỗi Unit IDs)`, xáo trộn, chia tập train/valid, và ghi ra file **manifest** chuẩn cho HiFi-GAN.

### Đầu vào (Input)

| Tham số | Loại | Mô tả |
|---|---|---|
| `tsv` | File `.tsv` | Danh sách file âm thanh, dòng đầu là đường dẫn gốc (root), các dòng sau là `tên_file.wav\t số_frame` |
| `km` | File `.km` | Mỗi dòng là chuỗi Unit IDs tương ứng, VD: `14 3 256 78 12 ...` |
| `wav_root` | Thư mục | Thư mục chứa toàn bộ file `.wav` gốc |

> Cấu hình đường dẫn được đặt trực tiếp trong biến `DATASETS` trong file `processing.py`:
> ```python
> DATASETS = {
>     "target": {   # Tiếng Việt
>         "tsv": '/mnt/g/khanh/manifest_temp/target/train.tsv',
>         "km":  '/mnt/g/khanh/kmean500/train_0_1.km',
>         "wav_root": "/mnt/g/data_final/data/target/train",
>     },
>     "source": {   # Tiếng Anh
>         "tsv": '/mnt/g/khanh/manifest_temp/source/train.tsv',
>         "km":  '/mnt/g/khanh/hubert_feats/en/train_0_1.km',
>         "wav_root": "/mnt/g/data_final/data/source/train",
>     }
> }
> ```

### Lệnh chạy

```bash
# Chỉ xử lý tiếng Việt (target)
python src/Unit2Wav/processing.py --lang target

# Chỉ xử lý tiếng Anh (source)
python src/Unit2Wav/processing.py --lang source

# Xử lý cả hai
python src/Unit2Wav/processing.py --lang both
```

### Đầu ra (Output)

Tự động tạo ra 2 file manifest trong thư mục `processed_data/[lang]/`:

```
src/Unit2Wav/processed_data/
└── target/
    ├── train.manifest   # 30.000 mẫu (đã xáo trộn ngẫu nhiên, seed=42)
    └── valid.manifest   # 1.000 mẫu tiếp theo
```

**Định dạng mỗi dòng trong file manifest** (Python dict string — đọc bằng `ast.literal_eval`):
```python
{'audio': '/mnt/g/data_final/data/target/train/sentence_000001.wav', 'hubert': '14 3 256 78 12 99 ...', 'duration': 3.4}
```

| Trường | Ý nghĩa |
|---|---|
| `audio` | Đường dẫn tuyệt đối tới file `.wav` gốc |
| `hubert` | Chuỗi Unit IDs dạng text (số nguyên cách nhau bằng space) |
| `duration` | Thời lượng ước tính = `số_frames × 0.02 giây` |

---

## 🏋️ Bước 2: Cấu hình mô hình (`config.json`)

Trước khi train, cập nhật 2 dòng cuối trong `config.json` để trỏ đúng vào file manifest vừa tạo:

**Cho tiếng Việt (`target`)**:
```json
{
  "input_training_file":   "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/target/train.manifest",
  "input_validation_file": "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/target/valid.manifest"
}
```

**Cho tiếng Anh (`source`)**:
```json
{
  "input_training_file":   "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/source/train.manifest",
  "input_validation_file": "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/source/valid.manifest"
}
```

### Các tham số quan trọng trong `config.json`

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `sampling_rate` | `16000` | Tần số mẫu âm thanh (Hz) |
| `num_mels` | `80` | Số mel filter banks |
| `n_fft` | `1024` | Kích thước FFT |
| `hop_size` | `256` | Hop size của STFT |
| `segment_size` | `8960` | Kích thước đoạn khi train (= 35 frames × 256) |
| `num_embeddings` | `1000` | Tổng số cụm K-Means (vocab size của Unit IDs) |
| `embedding_dim` | `512` | Chiều embedding cho mỗi unit |
| `code_hop_size` | `320` | Số sample mỗi unit chiếm (≈ 20ms ở 16kHz) |
| `batch_size` | `16` | Số mẫu mỗi batch khi train |
| `learning_rate` | `0.0002` | Tốc độ học ban đầu (Adam) |
| `upsample_rates` | `[5,4,4,2,2]` | Hệ số upsample qua các lớp Generator |
| `resblock` | `"1"` | Kiểu ResBlock của HiFi-GAN (1 hoặc 2) |

---

## 🚀 Bước 3: Huấn luyện Vocoder (`train.py`)

Script gọi `speech-resynthesis/train.py` với các tham số từ `config.json`.

### Đầu vào (Input)

| Tham số | Mô tả |
|---|---|
| `--lang source/target` | Chọn ngôn ngữ để train |
| `config.json` | Cấu hình kiến trúc và đường dẫn dữ liệu |
| `train.manifest`, `valid.manifest` | File dữ liệu đã tạo ở Bước 1 |

### Lệnh chạy

```bash
# Huấn luyện cho tiếng Việt (target)
python src/Unit2Wav/train.py --lang target

# Huấn luyện cho tiếng Anh (source)
python src/Unit2Wav/train.py --lang source
```

### Tham số train cố định (trong `train.py`)
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--training_epochs` | `300` | Tổng số epoch |
| `--checkpoint_interval` | `5000` | Lưu checkpoint mỗi 5000 steps |
| `--validation_interval` | `5000` | Validation mỗi 5000 steps |
| `--stdout_interval` | `50` | In log loss mỗi 50 steps |

> **Bật/tắt Duration Prediction**: Trong `train.py`, comment/uncomment dòng `"--dur_prediction"` để bật tính năng dự đoán duration (hiện đang **tắt**).

### Đầu ra (Output)

```
checkpoints/
└── vocoder_target_kmean500/          # Thư mục checkpoint
    ├── g_00005000                    # Generator checkpoint (dùng để infer)
    ├── do_00005000                   # Discriminator + Optimizer (dùng để fine-tune tiếp)
    ├── g_00010000
    ├── do_00010000
    ├── ...
    └── train_vocoder_target.log      # Toàn bộ log huấn luyện
```

| File | Ý nghĩa |
|---|---|
| `g_xxxxxxxx` | **Generator** — file checkpoint để sử dụng khi **Infer** (sinh wav) |
| `do_xxxxxxxx` | **Discriminator + Optimizer state** — dùng để **resume training** hoặc **fine-tune** |
| `train_vocoder_[lang].log` | Log toàn bộ quá trình train (loss, step, epoch) |

---

## 🔊 Bước 4: Sinh âm thanh (Inference) (`infer.py`)

Chạy sau khi đã có checkpoint để chuyển chuỗi Unit IDs thành file `.wav`.

### Chuẩn bị file input

Tạo/cập nhật file text chứa chuỗi Unit IDs cần chuyển đổi:

- **Target (Tiếng Việt)**: `final/unit2wav/target/predicted_unit.txt`
- **Source (Tiếng Anh)**: `final/unit2wav/source/predicted_unit.txt`

**Định dạng file `predicted_unit.txt`** — mỗi dòng là một chuỗi unit IDs:
```
14 3 256 78 12 99 45 200 ...
```
> File này thường là **output của module Unit2Unit** (bước dịch Unit Anh → Unit Việt).

### Đầu vào (Input)

| Tham số | Mô tả |
|---|---|
| `--lang target` | Dùng checkpoint custom tại `checkpoints/vocoder_target_kmean500/g_00020000` |
| `--lang source` | Dùng checkpoint Fairseq tại `checkpoints/vocoder_source/g_00500000` |

### Lệnh chạy

```bash
# Sinh wav tiếng Việt (target)
python src/Unit2Wav/infer.py --lang target

# Sinh wav tiếng Anh (source)
python src/Unit2Wav/infer.py --lang source
```

### Đầu ra (Output)

| Ngôn ngữ | Đường dẫn file wav kết quả |
|---|---|
| `target` (Tiếng Việt) | `final/unit2wav/target/predicted_wav/result_vn.wav` |
| `source` (Tiếng Anh) | `final/unit2wav/source/predicted_wav/result_en.wav` |

---

## 🔁 Sơ đồ luồng đầy đủ (End-to-End)

```
Bước 0: Chuẩn bị dữ liệu thô
──────────────────────────────────────────
  train.tsv  +  train.km  +  folder .wav
            │
            ▼
Bước 1: Tạo Manifest
──────────────────────────────────────────
  $ python src/Unit2Wav/processing.py --lang target
            │
            ▼
  processed_data/target/train.manifest
  processed_data/target/valid.manifest

Bước 2: Cập nhật config.json
──────────────────────────────────────────
  Sửa "input_training_file" và "input_validation_file"

Bước 3: Huấn luyện Vocoder
──────────────────────────────────────────
  $ python src/Unit2Wav/train.py --lang target
            │
            ▼
  checkpoints/vocoder_target_kmean500/g_00020000 ✅

Bước 4: Inference
──────────────────────────────────────────
  Tạo: final/unit2wav/target/predicted_unit.txt
  $ python src/Unit2Wav/infer.py --lang target
            │
            ▼
  final/unit2wav/target/predicted_wav/result_vn.wav 🔊
```

---

## 📌 Ghi chú & Lưu ý

- **Train bao lâu?** Với 30k mẫu, 300 epoch, batch size 16, thường mất **10–30 giờ** tùy GPU. Checkpoint tốt thường bắt đầu từ step **20000 trở lên**.
- **Chọn checkpoint nào để infer?** Dùng file `g_xxxxxxxx` có số step lớn nhất (và validation loss thấp nhất theo log). Hiện tại `infer.py` mặc định dùng `g_00020000`.
- **Source (Tiếng Anh)** không cần train — đã dùng checkpoint pretrained của Fairseq (`g_00500000`).
- **Bật Duration Prediction**: Bỏ comment dòng `# "--dur_prediction"` trong `train.py` nếu muốn mô hình học thêm cả thời lượng của từng unit.
- **Số K-Means cụm**: Dự án dùng **K=500** (kmean500). Nếu thay K khác, phải cập nhật `num_embeddings` trong `config.json`.
