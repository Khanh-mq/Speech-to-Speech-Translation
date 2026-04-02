# Unit2Unit — Dịch Unit IDs Tiếng Anh → Unit IDs Tiếng Việt

Module này chịu trách nhiệm huấn luyện mô hình **Sequence-to-Sequence Transformer** (dùng Fairseq) để **dịch chuỗi Unit IDs tiếng Anh sang chuỗi Unit IDs tiếng Việt** — đây là bước "dịch thuật" trung tâm trong pipeline Speech-to-Speech Translation (S2ST).

```
Unit IDs (Anh):  "26 55 25 11 11 6 38 21 ..."
        │
        ▼  [Transformer Seq2Seq]
Unit IDs (Việt): "34 46 11 27 26 19 34 48 ..."
```

---

## 📁 Cấu trúc thư mục

```
src/Unit2Unit/
├── processing.py   # Chuẩn bị & binarize dữ liệu (dedup + duration)
├── train.py        # Fine-tune Transformer từ checkpoint sẵn có
├── train_v2.py     # Train Transformer BIG từ đầu (from scratch)
└── infer.py        # Dịch chuỗi Unit Anh → Unit Việt (interactive)
```

> **Lưu ý về các phiên bản train**:
> - `train.py` — Fine-tune từ checkpoint (Transformer base, lr=1e-4)
> - `train_v2.py` — Train from scratch Transformer BIG (kiến trúc `transformer_wmt_en_de_big`, lr=5e-4)

---

## 🔄 Tổng quan Pipeline

```
[File .km Tiếng Anh]   [File .km Tiếng Việt]
(kmean1000/train_0_1.km) (kmean500/train_0_1.km)
        │                        │
        ▼                        ▼
Bước 1: processing.py  ──── Dedup + Duration
        │
        ├── train.src  (EN dedup)
        ├── train.tgt  (VI dedup)
        └── train.dur  (VI duration labels)
        │
        ▼  fairseq-preprocess (Binarize)
[data_bin_unit2unit_Dedup_Dur/]
        │
        ▼  train.py / train_v2.py (fairseq-train)
[checkpoint_best.pt]
        │
        ▼  infer.py (fairseq-interactive)
[Chuỗi Unit IDs Tiếng Việt] ──► Unit2Wav
```

---

## ⚙️ Yêu cầu trước khi chạy

### Dữ liệu đầu vào cần có (output từ module `Wav2Unit`)

| File | Mô tả | Đường dẫn |
|---|---|---|
| `train_0_1.km` (Anh) | Unit IDs tiếng Anh, K=1000 | `/mnt/g/khanh/hubert_feats/en/train_0_1.km` |
| `valid_0_1.km` (Anh) | Unit IDs tiếng Anh tập valid | `/mnt/g/khanh/hubert_feats/en/valid_0_1.km` |
| `train_0_1.km` (Việt) | Unit IDs tiếng Việt, K=500 | `/mnt/g/khanh/kmean500/train_0_1.km` |
| `valid_0_1.km` (Việt) | Unit IDs tiếng Việt tập valid | `/mnt/g/khanh/kmean500/valid_0_1.km` |

---

## 🛠️ Bước 1: Chuẩn bị dữ liệu (`processing.py`)

Script thực hiện 3 việc:
1. **Dedup** (cắt gọt) — loại bỏ các unit lặp liên tiếp ở **cả 2 ngôn ngữ**
2. **Duration extraction** — với tiếng Việt, đếm số lần lặp của mỗi unit để tạo nhãn `duration`
3. **Binarize** — dùng `fairseq-preprocess` đóng gói dữ liệu thành định dạng nhị phân `.bin`/`.idx`

### Tại sao phải Dedup?

Unit IDs thô từ HuBERT thường lặp rất nhiều (1 âm tiết có thể chiếm 10–20 frame liên tiếp cùng unit):
```
Raw:   "11 11 11 26 26 55 55 55 55 25 25 11 ..."   ← dài, nhiều lặp
Dedup: "11 26 55 25 11 ..."                         ← gọn, Transformer dễ học hơn
```

### Tại sao cần file Duration?

Sau khi dedup, model cần biết **mỗi unit kéo dài bao lâu** để tái tạo âm thanh đúng nhịp. File `.dur` lưu số lần lặp ban đầu của mỗi unit:
```
Raw target:   "11 11 11 26 26 55 55 55 55 25 25 11"
Dedup target: "11 26 55 25 11"
Duration:     "3  2  4  2  1"     ← unit "11" lặp 3 lần, "26" lặp 2 lần, ...
```

### Đầu vào (Input)

| File | Ngôn ngữ | Mô tả |
|---|---|---|
| `en/train_0_1.km` | Source (Anh) | Unit IDs thô K=1000 |
| `en/valid_0_1.km` | Source (Anh) | Tập valid |
| `kmean500/train_0_1.km` | Target (Việt) | Unit IDs thô K=500 |
| `kmean500/valid_0_1.km` | Target (Việt) | Tập valid |

### Định dạng file `.km` đầu vào

Mỗi dòng = chuỗi Unit IDs của 1 câu (số nguyên cách nhau bằng dấu cách):
```
26 55 25 11 11 6 11 38 21 28 14 12 30 11 56 42 ...
34 46 11 34 67 37 6 11 27 26 19 11 34 48 11 34 ...
```

### Lệnh chạy

```bash
python src/Unit2Unit/processing.py
```

> Không có tham số dòng lệnh — tất cả đường dẫn được cấu hình trực tiếp trong file.

### Đầu ra (Output)

**Bước trung gian** (trong thư mục `temp_corpus_dedup_dur/`):
```
temp_corpus_dedup_dur/
├── train.src    # Tiếng Anh đã dedup
├── train.tgt    # Tiếng Việt đã dedup
├── train.dur    # Nhãn duration tiếng Việt
├── valid.src
├── valid.tgt
└── valid.dur
```

**Định dạng các file trung gian:**

`train.src` — Tiếng Anh sau dedup (mỗi dòng 1 câu):
```
26 55 25 11 6 38 21 28 14 12 30 56 42 32 20 33 ...
34 46 34 67 37 6 27 26 19 34 48 34 75 24 39 ...
```

`train.tgt` — Tiếng Việt sau dedup:
```
14 3 256 78 12 99 45 200 7 34 ...
56 42 32 98 13 84 25 33 85 21 ...
```

`train.dur` — Duration tiếng Việt (tương ứng từng unit trong `.tgt`):
```
3 2 4 1 2 1 3 2 1 1 ...
2 1 3 2 1 4 2 1 1 2 ...
```

**Kết quả cuối** — Dữ liệu binarize tại `checkpoints/data_bin_unit2unit_Dedup_Dur/`:
```
checkpoints/data_bin_unit2unit_Dedup_Dur/
├── train.src-tgt.src.bin    # Dữ liệu nhị phân nguồn
├── train.src-tgt.src.idx    # Index tìm kiếm nhanh
├── train.src-tgt.tgt.bin    # Dữ liệu nhị phân đích
├── train.src-tgt.tgt.idx
├── valid.src-tgt.src.bin
├── valid.src-tgt.src.idx
├── valid.src-tgt.tgt.bin
├── valid.src-tgt.tgt.idx
├── dict.src.txt             # Từ điển Unit IDs nguồn
├── dict.tgt.txt             # Từ điển Unit IDs đích
├── train.dur                # File duration (copy thủ công từ temp)
└── valid.dur
```

---

## 🏋️ Bước 2: Huấn luyện mô hình

Có **2 script train** với 2 chiến lược khác nhau:

### `train_v2.py` — Train from Scratch (Transformer BIG)

Dùng khi chưa có checkpoint nào, huấn luyện model hoàn toàn từ đầu với kiến trúc **Transformer BIG** (lớn hơn, mạnh hơn).

```bash
python src/Unit2Unit/train_v2.py
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--arch` | `transformer_wmt_en_de_big` | Kiến trúc Transformer lớn (6 lớp encoder/decoder, d_model=1024) |
| `--lr` | `5e-4` | Learning rate ban đầu |
| `--warmup-updates` | `8000` | Số bước warmup |
| `--max-tokens` | `1536` | Số token tối đa mỗi batch |
| `--update-freq` | `16` | Gradient accumulation (tương đương batch × 16) |
| `--max-epoch` | `200` | Số epoch tối đa |
| `--patience` | `30` | Dừng sớm nếu không cải thiện sau 30 epoch |
| `--max-source-positions` | `1536` | Độ dài chuỗi source tối đa |
| `--fp16` | ✓ | Mixed precision (tiết kiệm VRAM) |

### `train.py` — Fine-tune từ checkpoint

Dùng khi đã có `checkpoint_last.pt`, tiếp tục fine-tune với learning rate nhỏ hơn.

```bash
python src/Unit2Unit/train.py
```

| Tham số | Giá trị | So với train_v2 |
|---|---|---|
| `--arch` | `transformer` | Transformer base (nhỏ hơn) |
| `--lr` | `1e-4` | Nhỏ hơn 5× để tránh phá vỡ weights đã học |
| `--warmup-updates` | `1000` | Ít warmup hơn |
| `--finetune-from-model` | `checkpoint_last.pt` | Load weights nhưng reset optimizer |
| `--max-tokens` | `4096` | Batch lớn hơn vì model nhỏ hơn |
| `--update-freq` | `4` | Gradient accumulation nhỏ hơn |
| `--patience` | `50` | Kiên nhẫn hơn khi fine-tune |

### Đầu ra (Output)

```
checkpoints/unit2unit_from_scratch_BIG/   # train_v2.py
checkpoints/unit2unit/                    # train.py
├── checkpoint_best.pt       # Checkpoint tốt nhất (theo validation loss)
├── checkpoint_last.pt       # Checkpoint mới nhất
├── checkpoint10.pt          # Checkpoint mỗi N epoch
├── ...
├── train_scratch.log        # Log toàn bộ quá trình train
└── tb_logs/                 # TensorBoard logs (chỉ train_v2)
```

| Metric theo dõi | Ý nghĩa |
|---|---|
| `train_loss` | Cross-entropy loss trên tập train (càng thấp càng tốt) |
| `valid_loss` | Loss trên tập validation (dùng để lưu `checkpoint_best`) |
| `bleu` | BLEU score trên tập valid (chỉ `train.py`) |

---

## 🔊 Bước 3: Dịch Unit IDs (`infer.py`)

Dùng `fairseq-interactive` để nhập chuỗi Unit IDs tiếng Anh và nhận kết quả Unit IDs tiếng Việt theo thời gian thực.

### Lệnh chạy

```bash
python src/Unit2Unit/infer.py
```

Sau khi khởi động, nhập chuỗi Unit IDs tiếng Anh (dedup) vào terminal:
```
Hãy nhập chuỗi Unit Source:
> 26 55 25 11 6 38 21 28 14 12 30

H-0     14 3 256 78 12 99 45 200 7 34    ← Unit IDs tiếng Việt dịch ra
D-0     -0.432                            ← Log-probability
P-0     -0.12 -0.45 -0.23 ...            ← Token probabilities
```

### Tham số cấu hình (trong `infer.py`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--beam` | `5` | Beam search width |
| `--source-lang` | `src` | Ngôn ngữ nguồn (Anh) |
| `--target-lang` | `tgt` | Ngôn ngữ đích (Việt) |
| `MODEL_PATH` | `checkpoints/unit2unit_en_vi/checkpoint_best.pt` | Checkpoint tốt nhất |

### Kết quả đầu ra

Output (Unit IDs tiếng Việt) được lưu vào:
```
final/unit2wav/target/predicted_unit.txt
```

**Định dạng `predicted_unit.txt`** — 1 dòng duy nhất chứa chuỗi Unit IDs tiếng Việt (đã dedup):
```
14 3 256 78 12 99 45 200 7 34 56 42 32 98 ...
```

> File này là **đầu vào trực tiếp cho module `Unit2Wav`** (bước cuối tổng hợp giọng nói tiếng Việt).

---

## 📊 Toàn bộ luồng dữ liệu

```
INPUT (.km files từ Wav2Unit)
───────────────────────────────────────────────────────────
en/train_0_1.km       kmean500/train_0_1.km
"26 55 55 25 11 11…"  "14 14 3 256 78 78 12…"
        │                        │
        ▼ Dedup                  ▼ Dedup + Đếm duration
"26 55 25 11…"        "14 3 256 78 12…"  +  "2 1 1 2 1…"
        │                        │                │
        ▼ (train.src)       (train.tgt)      (train.dur)
        └──────────┬─────────────┘
                   ▼
           fairseq-preprocess
                   │
                   ▼
         data_bin_unit2unit_Dedup_Dur/
                   │
                   ▼
        fairseq-train (Transformer)
                   │
                   ▼
         checkpoint_best.pt ✅

INFERENCE
───────────────────────────────────────────────────────────
"26 55 25 11 6 38 21…"   ← Input Unit EN (đã dedup)
        │
        ▼ fairseq-interactive + checkpoint_best.pt
"14 3 256 78 12 99 45…"  ← Output Unit VI 🔊
        │
        ▼
final/unit2wav/target/predicted_unit.txt
```

---

## 📌 Ghi chú & Lưu ý

- **Thứ tự file bắt buộc**: File `.km` tiếng Anh và tiếng Việt phải **khớp dòng với nhau** — dòng N của Anh là bản dịch của dòng N tiếng Việt (parallel corpus).
- **Nên dùng `train_v2.py` hay `train.py`?** Với data mới từ đầu, dùng `train_v2.py` (Transformer BIG). Khi đã có checkpoint ổn định, dùng `train.py` để fine-tune tiếp.
- **Input infer phải dedup**: Khi infer, chuỗi Unit IDs tiếng Anh đưa vào phải đã qua **dedup** (giống format `train.src`), nếu không kết quả sẽ sai.
- **File duration trong bước train**: Hiện tại file `train.dur` / `valid.dur` được copy vào `DATA_BIN` nhưng cần custom task của Fairseq để sử dụng (xem `custom_u2u/`).
- **Checkpoint tốt nhất**: Lưu tại `checkpoint_best.pt` theo metric `valid_loss`. BLEU không phải metric chính vì Unit IDs không phải ngôn ngữ tự nhiên.
