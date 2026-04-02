# Wav2Unit — Chuyển đổi Audio (.wav) → Unit IDs

Module này chịu trách nhiệm biến đổi dữ liệu âm thanh thô (`.wav`) thành chuỗi **Unit IDs (Discrete Speech Units)** — là đầu vào cho các module dịch thuật tiếp theo (`Unit2Unit`, `Unit2Wav`).

Pipeline gồm **4 bước tuần tự**, hỗ trợ cả 2 ngôn ngữ:
- **`source`** — Tiếng Anh (dùng K-Means pretrained của mHuBERT, K=1000)
- **`target`** — Tiếng Việt (tự train K-Means từ dữ liệu tiếng Việt, K=500)

---

## 📁 Cấu trúc thư mục

```
src/Wav2Unit/
├── train.py     # Pipeline 4 bước xử lý toàn bộ dataset (offline)
└── infer.py     # Chuyển đổi 1 file .wav đơn lẻ sang Unit IDs (inference)
```

---

## 🔄 Tổng quan Pipeline

```
[Thư mục .wav gốc]
        │
        ▼  Bước 1: generate_manifest()
[File .tsv]  ──────────────────────────────────────── train.tsv / valid.tsv / test.tsv
        │
        ▼  Bước 2: extract_features()
[File .npy + .len]  ─────────────────────────────── HuBERT features (Layer 11)
        │
        ▼  Bước 3: run_kmeans_training()  (CHỈ target)
[File .bin]  ────────────────────────────────────── K-Means model tiếng Việt
        │
        ▼  Bước 4: quantize()
[File .km]  ─────────────────────────────────────── Unit IDs (chuỗi số nguyên)
```

---

## ⚙️ Yêu cầu trước khi chạy

### Checkpoints cần có sẵn

| File | Mô tả | Đường dẫn |
|---|---|---|
| `mhubert_base_vp_en_es_fr_it3.pt` | mHuBERT pretrained (dùng cho cả source lẫn target) | `checkpoints/mhubert_base_vp_en_es_fr_it3.pt` |
| `mhubert_base_vp_en_es_fr_it3_L11_km1000.bin` | K-Means pretrained cho tiếng Anh (K=1000) | `checkpoints/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin` |
| `kmeans_vn_500.bin` | K-Means tiếng Việt — **tự train** ở Bước 3 | `kmeans/kmeans_vn_500.bin` |

### Cấu trúc dữ liệu âm thanh cần có

```
/mnt/g/data_final/data/
├── source/          # Tiếng Anh
│   ├── train/       # Thư mục chứa file .wav
│   │   ├── sentence_000001.wav
│   │   ├── sentence_000002.wav
│   │   └── ...
│   ├── valid/
│   └── test/
└── target/          # Tiếng Việt
    ├── train/
    ├── valid/
    └── test/
```

> ⚠️ Tất cả file `.wav` phải được chuẩn hóa về **16kHz, mono** trước khi xử lý.

---

## 🛠️ Bước 1: Tạo Manifest (`--manifest`)

Quét toàn bộ thư mục `.wav` và tạo file `.tsv` lập chỉ mục danh sách âm thanh.

### Đầu vào (Input)
| Thứ | Mô tả |
|---|---|
| Thư mục `.wav` | VD: `/mnt/g/data_final/data/target/train/` |
| `--lang` | `source` hoặc `target` |
| `--split` | `train`, `valid`, `test`, hoặc `all` |

### Đầu ra (Output) — File `train.tsv`

Mỗi dòng: `tên_file.wav\t<số_sample>`, dòng đầu là root path:

```
/mnt/g/data_final/data/target/train
sentence_000001.wav	130560
sentence_000002.wav	98304
sentence_000037.wav	45056
...
```

| Cột | Ý nghĩa |
|---|---|
| Cột 1 | Tên file `.wav` (tương đối so với root) |
| Cột 2 | Số lượng audio samples (= thời lượng × sample_rate) |

Kết quả lưu tại:
```
/mnt/g/khanh/manifest_temp/
├── source/
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
└── target/
    ├── train.tsv
    ├── valid.tsv
    └── test.tsv
```

---

## 🧠 Bước 2: Trích xuất Features HuBERT (`--feature`)

Đưa audio qua mô hình **mHuBERT** và lấy vector đặc trưng tại **Layer 11**.

### Đầu vào (Input)
| Thứ | Mô tả |
|---|---|
| File `.tsv` | Từ Bước 1 |
| `mhubert_base_vp_en_es_fr_it3.pt` | Checkpoint mHuBERT pretrained |
| Layer | Cố định **Layer 11** (embedded trong code) |

### Đầu ra (Output)

Với mỗi split, sinh ra 2 file trong thư mục `hubert_feats/[en|vn]/`:

```
/mnt/g/khanh/hubert_feats/
├── en/               # Tiếng Anh
│   ├── train_0_1.npy    # Ma trận features shape (N, 1024) — float32
│   ├── train_0_1.len    # Số frame của từng câu (1 số nguyên/dòng)
│   ├── valid_0_1.npy
│   └── valid_0_1.len
└── vn/               # Tiếng Việt
    ├── train_0_1.npy
    ├── train_0_1.len
    └── ...
```

| File | Ý nghĩa |
|---|---|
| `.npy` | Ma trận numpy, mỗi hàng là vector 1024 chiều của 1 frame (20ms) |
| `.len` | Mỗi dòng = số frame của 1 câu audio, dùng để tách ranh giới câu |

---

## 📊 Bước 3: Huấn luyện K-Means (`--kmeans`) — CHỈ cho `target`

Học bộ **K=500 tâm cụm** từ các HuBERT features tiếng Việt để tạo ra "từ vựng" Unit.

> **Chỉ cần chạy 1 lần duy nhất cho tiếng Việt.** Tiếng Anh đã có K-Means pretrained sẵn từ mHuBERT (`km1000.bin`).

### Đầu vào (Input)
| Thứ | Mô tả |
|---|---|
| `train_0_1.npy` | Features tiếng Việt từ Bước 2 |
| K | Số cụm = **500** (biến `kmean` trong code) |
| `--percent 0.1` | Chỉ dùng 10% data để train K-Means (đủ nhanh, đủ tốt) |

### Đầu ra (Output)

```
/mnt/e/AI/khanh/kmeans/
└── kmeans_vn_500.bin     # File K-Means model đã train (dùng cho Bước 4)
```

---

## 🔢 Bước 4: Quantization — Features → Unit IDs (`--quantize`)

Áp dụng K-Means đã train để phân loại mỗi frame HuBERT vào 1 trong K cụm, tạo ra chuỗi số nguyên (Unit IDs).

### Đầu vào (Input)
| Thứ | Mô tả |
|---|---|
| `.npy` | Features từ Bước 2 |
| `.bin` | K-Means model (source: `km1000.bin`, target: `kmeans_vn_500.bin`) |

### Đầu ra (Output) — File `.km`

Mỗi dòng là chuỗi Unit IDs của 1 câu audio, các số cách nhau bằng dấu cách:

```
26 55 25 11 11 6 11 38 21 28 14 12 30 11 56 42 11 32 20 33 ...
34 46 11 34 67 37 6 11 27 26 19 11 34 48 11 34 75 11 24 39 ...
56 64 33 11 32 21 45 26 11 14 20 53 26 19 11 32 50 21 11 26 ...
```

Kết quả lưu tại:
```
/mnt/g/khanh/
├── kmean1000/        # Tiếng Anh (K=1000)
│   ├── train_0_1.km
│   ├── valid_0_1.km
│   └── test_0_1.km
└── kmean500/         # Tiếng Việt (K=500)
    ├── train_0_1.km
    ├── valid_0_1.km
    └── test_0_1.km
```

> File `.km` này chính là **đầu vào cho module `Unit2Unit`** (bước dịch) và **`Unit2Wav`** (bước tổng hợp âm thanh).

---

## 🚀 Cách chạy `train.py` — Xử lý toàn bộ dataset

### Lệnh chạy từng bước lẻ

```bash
# Bước 1: Tạo manifest
python src/Wav2Unit/train.py --lang target --split all --manifest

# Bước 2: Trích xuất features
python src/Wav2Unit/train.py --lang target --split all --feature

# Bước 3: Train K-means (CHỈ target)
python src/Wav2Unit/train.py --lang target --kmeans

# Bước 4: Quantize sang Unit IDs
python src/Wav2Unit/train.py --lang target --split all --quantize
```

### Lệnh chạy toàn bộ pipeline 1 lần

```bash
# Xử lý tiếng Việt (target) — toàn bộ 4 bước
python src/Wav2Unit/train.py --lang target --split all --all

# Xử lý tiếng Anh (source) — bỏ qua bước train K-means
python src/Wav2Unit/train.py --lang source --split all --all
```

### Tóm tắt các flag

| Flag | Ý nghĩa |
|---|---|
| `--lang source/target` | Chọn ngôn ngữ |
| `--split train/valid/test/all` | Chọn tập dữ liệu (mặc định: `all`) |
| `--all` | Chạy toàn bộ 4 bước |
| `--manifest` | Chỉ chạy Bước 1 |
| `--feature` | Chỉ chạy Bước 2 |
| `--kmeans` | Chỉ chạy Bước 3 (chỉ có tác dụng với `--lang target`) |
| `--quantize` | Chỉ chạy Bước 4 |

---

## 🔊 Inference — Chuyển 1 file `.wav` đơn lẻ sang Unit IDs (`infer.py`)

Dùng khi cần chuyển đổi 1 file wav lúc inference (không phải batch training).

### Chuẩn bị input

Đặt file `.wav` cần xử lý vào đúng đường dẫn:

| Ngôn ngữ | Đường dẫn input |
|---|---|
| Tiếng Anh (`source`) | `final/wav2unit/source/input/input.wav` |
| Tiếng Việt (`target`) | `final/wav2unit/target/input/input.wav` |

> File phải đặt tên đúng là **`input.wav`**.

### Lệnh chạy

```bash
# Inference tiếng Anh
python src/Wav2Unit/infer.py --lang source

# Inference tiếng Việt
python src/Wav2Unit/infer.py --lang target
```

### Quá trình xử lý bên trong (`infer.py`)

`infer.py` thực hiện đầy đủ 3 bước tự động với 1 file wav duy nhất:

```
input.wav
    │
    ▼  Bước 1: wav2vec_manifest.py  → temp_infer/train.tsv
    │
    ▼  Bước 2: dump_hubert_feature.py (Layer 11) → temp_infer/train_0_1.npy
    │
    ▼  Bước 3: dump_km_label.py + kmeans.bin → temp_infer/train_0_1.km
    │
    ▼  Đọc & lưu kết quả
    │
    ▼
predicted_unit.txt  ──────►  (đầu vào cho Unit2Unit / Unit2Wav)
```

### Đầu ra (Output)

| Ngôn ngữ | Đường dẫn file kết quả |
|---|---|
| `source` | `final/wav2unit/source/predicted_unit.txt` |
| `target` | `final/wav2unit/target/predicted_unit.txt` |

**Định dạng file `predicted_unit.txt`** — 1 dòng duy nhất chứa toàn bộ Unit IDs:
```
26 55 25 11 11 6 11 38 21 28 14 12 30 11 56 42 ...
```

---

## 📌 Ghi chú & Lưu ý

- **K-Means cho source đã có sẵn**: Tiếng Anh dùng `km1000.bin` của mHuBERT gốc — **không cần train lại**. Chỉ tiếng Việt mới cần chạy `--kmeans`.
- **Layer 11**: Thực nghiệm cho thấy Layer 11 của mHuBERT cho features tốt nhất cho bài toán discrete unit.
- **`--percent 0.1`** trong K-Means training: Chỉ dùng 10% data để train, vừa đủ nhanh vừa đủ tốt với lượng data lớn.
- **Thứ tự file phải khớp**: File `.tsv` và `.km` luôn phải khớp thứ tự dòng — đây là yêu cầu bắt buộc khi dùng với `Unit2Wav/processing.py`.
- **Số K cụm**: Source dùng K=1000, Target dùng K=500. Nếu thay đổi, nhớ cập nhật biến `kmean` trong `train.py` và `num_embeddings` trong `Unit2Wav/config.json`.
