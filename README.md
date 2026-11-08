# [Bài Tập Lớn Học Sâu - Nhóm 18 - Đề Tài 06] Phân Tích Cảm Xúc Văn Bản Tiếng Việt

Tài liệu này là **Đặc tả Kỹ thuật & Hướng dẫn Thực hiện (Single Source of Truth)** dành cho các thành viên Nhóm 18 và các công cụ AI hỗ trợ lập trình (Cursor, GitHub Copilot, Claude, ChatGPT...). Mọi thành viên khi viết code hoặc soạn báo cáo chuẩn hội nghị ACL bắt buộc phải tuân thủ thống nhất các quy chuẩn dưới đây.

---

## 1. Tổng Quan Bài Toán & Dữ Liệu

* **Tên đề tài:** Đề tài 6 – Phân tích cảm xúc văn bản tiếng Việt (Vietnamese Sentiment Analysis).
* **Đầu vào (Input):** Một câu hoặc đoạn bình luận tiếng Việt (phản hồi sinh viên từ bộ **UIT-VSFC** hoặc đánh giá sản phẩm từ **UIT-ViSFD**).
* **Đầu ra (Output):**
  * **Bài toán chính (Polarity Classification):** Nhãn cảm xúc gồm 3 lớp (`0: Negative / Tiêu cực`, `1: Neutral / Trung tính`, `2: Positive / Tích cực`) kèm vector phân phối xác suất $\mathbf{p} \in \mathbb{R}^3$.
  * **Bài toán mở rộng (Aspect-Based Sentiment Analysis - ABSA):** Các cặp `(Khía cạnh, Cảm xúc)` thông qua kiến trúc học đa nhiệm (Multi-task Learning) với $N$ nhánh phân loại tương ứng với $N$ khía cạnh, mỗi nhánh dự đoán 4 trạng thái (`0: None`, `1: Negative`, `2: Neutral`, `3: Positive`).
* **Chỉ số đánh giá bắt buộc:** `Accuracy`, `Macro-Precision`, `Macro-Recall`, và trọng tâm là **`Macro-F1`**.
* **Công cụ & Framework:** `PyTorch`, `Scikit-learn`, `HuggingFace Transformers`, `PyVi`, `Weights & Biases (wandb)`, và `Gradio`.

---

## 2. Cấu Trúc Thư Mục Chuẩn Của Repo

```text
dl-nhom18-vietnamese-sentiment-analysis/
├── configs/
│   └── config.yaml              # Lưu siêu tham số & đường dẫn chung
├── data/                        # KHÔNG commit dữ liệu nặng (.csv, .bin, .vec) lên Git
│   ├── prepare_dataset.py       # Script tải và xuất 3 file train/val/test cố định
│   ├── raw/                     # Chứa dữ liệu gốc tải về (đã chặn trong .gitignore)
│   ├── processed/               # Chứa train.csv, val.csv, test.csv đồng nhất cho 3 thành viên
│   └── embeddings/              # Chứa PhoW2V / fastText tiếng Việt (đã chặn trong .gitignore)
├── models/
│   └── checkpoints/             # Lưu trọng số mô hình tốt nhất (đã chặn trong .gitignore)
├── logs/                        # Lưu file dự đoán <model>_test_preds.csv để phân tích lỗi
├── notebooks/
│   ├── 01_eda.ipynb             # Thống kê dữ liệu, vẽ biểu đồ phân bố nhãn & độ dài câu
│   └── 02_error_analysis.ipynb  # Vẽ Confusion Matrix, lọc câu mỉa mai / đa nghĩa
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py           # Pipeline tiền xử lý tiếng Việt chuẩn hóa v2.0
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_lr.py       # Nhánh 1: TF-IDF + Logistic Regression
│   │   ├── bilstm.py            # Nhánh 2: BiLSTM + Pretrained Word2Vec/fastText
│   │   ├── phobert.py           # Nhánh 3: Fine-tuning vinai/phobert-base-v2
│   │   └── absa_model.py        # Mở rộng: Phân tích cảm xúc theo khía cạnh (ABSA)
│   └── utils/
│       ├── __init__.py
│       ├── seed.py              # Hàm cố định Random Seed (seed=42)
│       └── metrics.py           # Tính Accuracy, Macro-F1 & xuất file CSV dự đoán
├── app.py                       # Giao diện web demo thời gian thực bằng Gradio
├── train.py                     # Script huấn luyện mô hình qua CLI
├── evaluate.py                  # Script tái tạo toàn bộ bảng kết quả trong báo cáo ACL
├── requirements.txt             # Danh sách thư viện cài đặt
├── .gitignore                   # Chặn file rác, dữ liệu lớn và trọng số mô hình
└── GUIDELINES.md                # Tài liệu đặc tả kỹ thuật của nhóm
```

---

## 3. Quy Tắc Kỹ Thuật Cốt Lõi (Bắt Buộc Tuân Thủ)

### 3.1. Cố định Luồng Dữ Liệu & Random Seed (Chống lệch kết quả)
1. **Single Source of Truth cho dữ liệu:** Thành viên 1 chạy script tiền xử lý **1 lần duy nhất**, xuất ra 3 file cố định tại `data/processed/train.csv`, `data/processed/val.csv`, và `data/processed/test.csv`. Thành viên 2 (BiLSTM) và Thành viên 3 (PhoBERT) bắt buộc đọc trực tiếp từ 3 file này, tuyệt đối không tự chia lại dữ liệu.
2. **Các cột bắt buộc trong file CSV đã xử lý:**
   * `raw_text`: Văn bản gốc.
   * `clean_word_nopunct`: Đã chuẩn hóa teencode/emoji, chữ thường, xóa dấu câu, đã tách từ bằng `PyVi` (Dùng cho **TF-IDF** và **BiLSTM**).
   * `clean_word_keep_punct`: Đã chuẩn hóa teencode/emoji, giữ chữ hoa và dấu câu biểu cảm (`!?,.;:`), đã tách từ bằng `PyVi` (Dùng cho **PhoBERT**).
   * `clean_syllable`: Văn bản sạch nhưng **không tách từ** (Dùng cho **Ablation 1: Word vs. Syllable**).
   * `clean_no_accent`: Văn bản đã lược bỏ dấu tiếng Việt (Dùng cho **Ablation 2: Văn bản không dấu**).
   * `label`: Nhãn số nguyên (`0`, `1`, `2`).
3. **Cố định Random Seed (`seed = 42`):** Mọi script huấn luyện phải gọi hàm `set_seed(42)` ở đầu chương trình:
   ```python
   import os
   import random
   import numpy as np
   import torch

   def set_seed(seed: int = 42) -> None:
       random.seed(seed)
       os.environ["PYTHONHASHSEED"] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```

### 3.2. Quy Chuẩn Tiền Xử Lý Văn Bản Tiếng Việt (`cleaner.py`)
1. **Thứ tự thực hiện 7 bước:**
   * Bước 1: Chuẩn hóa bảng mã Unicode về chuẩn dựng sẵn `NFC`.
   * Bước 2: Ánh xạ Emoji & Emoticon (`:))`, `:(`, `colonsmile`, `colonsad`, `😭`...) sang từ cảm xúc (`tích_cực`, `tiêu_cực`) bằng regex không phân biệt hoa thường (`re.IGNORECASE`) **trước khi** xóa ký tự đặc biệt.
   * Bước 3: Chuẩn hóa Teencode & Từ viết tắt **trước khi** xóa dấu câu; luôn ép `token.lower()` khi tra từ điển để bắt được cả từ viết hoa như `"KO"`, `"OK!"`, `"VKL"`.
   * Bước 4: Chuyển chữ thường (tùy theo cờ `lowercase`).
   * Bước 5: Xóa URL, HTML tags và xử lý dấu câu theo cờ `keep_punctuation` (`False` cho TF-IDF/BiLSTM, `True` cho PhoBERT).
   * Bước 6: Xóa dấu tiếng Việt nếu bật cờ thí nghiệm `remove_accents=True`.
   * Bước 7: Đồng nhất dùng duy nhất thư viện **`PyVi` (`ViTokenizer.tokenize`)** cho toàn bộ dự án để khớp quy tắc nối dấu `_` với dữ liệu tiền huấn luyện của PhoBERT.
2. **Cấm xóa Stopwords bừa bãi:** Tuyệt đối không dùng danh sách stopwords mặc định xóa mất các từ phủ định và từ chỉ mức độ (*không, chẳng, chưa, rất, quá, nhưng, tuy nhiên*).

### 3.3. Thiết Kế 3 Mô Hình & Xử Lý Lệch Lớp Công Bằng
Để xử lý việc lớp `Neutral` có số mẫu ít hơn hai lớp còn lại, **cả 3 mô hình** đều phải sử dụng trọng số lớp (`class_weights`) được tính trên tập `Train`:
$$\text{class\_weights} = \text{compute\_class\_weight}(\text{'balanced'}, \text{classes}=[0, 1, 2], y=y_{\text{train}})$$

* **Mô hình 1 (`baseline_lr.py`):** Dùng `TfidfVectorizer(ngram_range=(1, 2), max_features=10000)` chỉ `fit` trên tập `Train`. Huấn luyện với `LogisticRegression(C=1.0, max_iter=250, class_weight="balanced", random_state=42)`.
* **Mô hình 2 (`bilstm.py`):**
  * Xây dựng từ điển `vocab` từ tập `Train` (`0: <pad>`, `1: <unk>`), chuyển câu thành chuỗi chỉ số nguyên 2D `(batch_size, max_len)` kiểu `torch.long` (không tạo tensor 3D `float32` ngoài RAM).
  * Nạp trọng số Pretrained Word2Vec/fastText tiếng Việt vào `nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)`.
  * Kiến trúc 2 tầng `BiLSTM` (`bidirectional=True`), có xử lý **Masking** bỏ qua các vị trí `<pad>` (`input_ids != 0`) khi Pooling/Attention.
  * Hàm mất mát: `nn.CrossEntropyLoss(weight=class_weights_tensor)`.
* **Mô hình 3 (`phobert.py`):**
  * Fine-tune `vinai/phobert-base-v2` với `max_length=128`, `batch_size=16` hoặc `32`, `AdamW(lr=2e-5, weight_decay=0.01)`, `linear warmup`.
  * Sử dụng `nn.CrossEntropyLoss(weight=class_weights_tensor)` để đối sánh công bằng với Mô hình 1 và 2.

### 3.4. Xuất File Dự Đoán Phục Vụ Phân Tích Lỗi (Error Analysis)
Mọi mô hình khi đánh giá trên tập `Test` bắt buộc xuất ra file CSV tại `logs/<model_name>_test_preds.csv` gồm đúng các cột:
`sentence, clean_sentence, true_label, pred_label, prob_neg, prob_neu, prob_pos, is_correct`

---

## 4. Toàn Bộ Code Khởi Tạo Cần Tạo Ngay Trong Repo

### 4.1. File `.gitignore`
```text
__pycache__/
*.py[cod]
*$py.class
.env
.venv/
venv/
ENV/
.ipynb_checkpoints/
data/raw/
data/processed/
data/embeddings/
models/checkpoints/
logs/*.csv
wandb/
runs/
*.pt
*.pth
*.bin
*.vec
*.onnx
.DS_Store
```

### 4.2. File `requirements.txt`
```text
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
pyvi>=0.1.1
gensim>=4.3.0
pyyaml>=6.0
wandb>=0.16.0
gradio>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### 4.3. File `configs/config.yaml`
```yaml
project:
  name: "dl-nhom18-vietnamese-sentiment-analysis"
  seed: 42
  device: "cuda" # Tự động chuyển về 'cpu' nếu máy không có GPU

paths:
  raw_dir: "data/raw"
  train_data: "data/processed/train.csv"
  val_data: "data/processed/val.csv"
  test_data: "data/processed/test.csv"
  pretrained_w2v: "data/embeddings/word2vec_vi_words_300dims.txt"
  checkpoints_dir: "models/checkpoints"
  logs_dir: "logs"

preprocessing:
  normalize_unicode: true
  normalize_teencode: true
  normalize_emoji: true
  use_word_segmentation: true  # Ablation 1: true (Word-level) vs false (Syllable-level)
  remove_diacritics: false     # Ablation 2: true (Không dấu) vs false (Có dấu)

models:
  baseline_tfidf:
    max_features: 10000
    ngram_range: [1, 2]
    c_param: 1.0
    class_weight: "balanced"

  bilstm:
    embedding_dim: 300
    hidden_dim: 128
    num_layers: 2
    dropout: 0.3
    bidirectional: true
    freeze_embeddings: false
    lr: 1.0e-3
    max_length: 100
    vocab_size: 15000

  phobert:
    pretrained_name: "vinai/phobert-base-v2"
    max_length: 128
    learning_rate: 2.0e-5
    weight_decay: 0.01
    warmup_ratio: 0.1

training:
  batch_size: 32
  epochs: 5
  early_stopping_patience: 2
  wandb_project: "dl-nhom18-topic6-sentiment"
```

### 4.4. File `src/preprocessing/cleaner.py`
```python
import re
import unicodedata
from typing import Dict, Optional
from pyvi import ViTokenizer


class VietnameseTextCleaner:
    """
    Bộ tiền xử lý văn bản tiếng Việt chuẩn hóa cho Đề tài 6 (Nhóm 18 - v2.0).
    - Cố định công cụ tách từ PyVi (tương thích chuẩn RDRSegmenter của PhoBERT).
    - Hỗ trợ tra từ điển Teencode/Emoji không phân biệt chữ hoa/thường.
    - Hỗ trợ cờ keep_punctuation=True (cho PhoBERT) và False (cho TF-IDF/BiLSTM).
    """

    DEFAULT_TEENCODE_DICT: Dict[str, str] = {
        "ko": "không", "k": "không", "khong": "không", "kg": "không", "koff": "không",
        "hong": "không", "hông": "không", "kh": "không", "đc": "được", "dc": "được",
        "ok": "tốt", "okey": "tốt", "oke": "tốt", "okie": "tốt",
        "vs": "với", "j": "gì", "gi": "gì", "z": "vậy", "zậy": "vậy", "v": "vậy",
        "t": "tôi", "m": "mày", "b": "bạn", "ng": "người", "mn": "mọi người",
        "sv": "sinh viên", "gv": "giảng viên", "đk": "đăng ký", "dk": "đăng ký",
        "trc": "trước", "vkl": "rất", "vl": "rất", "thik": "thích", "iu": "yêu",
        "bt": "bình thường", "bth": "bình thường", "ntn": "như thế nào",
    }

    DEFAULT_EMOJI_DICT: Dict[str, str] = {
        ":))": " tích_cực ", "=))": " tích_cực ", ":)": " tích_cực ", "<3": " tích_cực ",
        "colonsmile": " tích_cực ", "colonlovelove": " tích_cực ", "colonhihi": " tích_cực ",
        ":((": " tiêu_cực ", ":(": " tiêu_cực ", "=(": " tiêu_cực ",
        "colonsad": " tiêu_cực ", "coloncontemn": " tiêu_cực ",
        "👍": " tích_cực ", "❤️": " tích_cực ", "😊": " tích_cực ", "😍": " tích_cực ",
        "😢": " tiêu_cực ", "😭": " tiêu_cực ", "😡": " tiêu_cực ", "👎": " tiêu_cực ",
    }

    def __init__(
        self,
        teencode_dict: Optional[Dict[str, str]] = None,
        emoji_dict: Optional[Dict[str, str]] = None,
        use_word_segmentation: bool = True,
        lowercase: bool = True,
        normalize_teencode: bool = True,
        normalize_emoji: bool = True,
        keep_punctuation: bool = False,
    ):
        self.teencode_dict = {
            k.lower(): v for k, v in (teencode_dict or self.DEFAULT_TEENCODE_DICT).items()
        }
        self.emoji_dict = emoji_dict or self.DEFAULT_EMOJI_DICT
        self.use_word_segmentation = use_word_segmentation
        self.lowercase = lowercase
        self.normalize_teencode = normalize_teencode
        self.normalize_emoji = normalize_emoji
        self.keep_punctuation = keep_punctuation

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def replace_emojis(self, text: str) -> str:
        for emo, replacement in self.emoji_dict.items():
            pattern = re.compile(re.escape(emo), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def replace_teencode(self, text: str) -> str:
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        normalized_tokens = [
            self.teencode_dict.get(tok.lower(), tok) for tok in tokens
        ]
        return " ".join(normalized_tokens)

    def clean_special_characters(self, text: str) -> str:
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        if self.keep_punctuation:
            text = re.sub(r"[^\w\s\d_!?,.;:]", " ", text)
        else:
            text = re.sub(r"[^\w\s\d_]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def remove_diacritics(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = re.sub(r"[\u0300-\u036f]", "", text)
        text = text.replace("đ", "d").replace("Đ", "D")
        return unicodedata.normalize("NFC", text)

    def word_segment(self, text: str) -> str:
        if self.use_word_segmentation:
            return ViTokenizer.tokenize(text)
        return text

    def clean(self, text: str, remove_accents: bool = False) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.normalize_unicode(text)
        if self.normalize_emoji:
            text = self.replace_emojis(text)
        if self.normalize_teencode:
            text = self.replace_teencode(text)
        if self.lowercase:
            text = text.lower()
        text = self.clean_special_characters(text)
        if remove_accents:
            text = self.remove_diacritics(text)
        text = self.word_segment(text)

        return re.sub(r"\s+", " ", text).strip()
```

---

## 5. Phân Công Công Việc (3 Thành Viên)

| Thành viên | Họ và Tên | MSSV | Nhiệm vụ phụ trách chính | Tỉ lệ dự kiến |
| :--- | :--- | :--- | :--- | :---: |
| **Thành viên 1** | *(Điền tên)* | *(MSSV)* | • Xây dựng pipeline tiền xử lý (`cleaner.py`), từ điển Teencode/Emoji.<br>• Xuất bộ dữ liệu chuẩn (`train.csv`, `val.csv`, `test.csv`) & làm EDA.<br>• Cài đặt và huấn luyện **TF-IDF + Logistic Regression** (`baseline_lr.py`).<br>• Phụ trách thí nghiệm **Ablation 2: Ảnh hưởng của văn bản không dấu**. | 33.3% |
| **Thành viên 2** | *(Điền tên)* | *(MSSV)* | • Xây dựng Vocabulary & nạp Pretrained **Word2Vec / fastText** tiếng Việt.<br>• Tự cài đặt mô hình **BiLSTM** (có Masking & Pooling/Attention) bằng PyTorch.<br>• Phụ trách thí nghiệm **Ablation 1: Ảnh hưởng của tách từ (Word vs. Syllable)** trên cả 3 mô hình. | 33.3% |
| **Thành viên 3** | *(Điền tên)* | *(MSSV)* | • Fine-tune mô hình **PhoBERT (`vinai/phobert-base-v2`)** và quản lý log W&B.<br>• Cài đặt phần mở rộng **Phân tích cảm xúc theo khía cạnh (ABSA)**.<br>• Thực hiện **Phân tích lỗi (Error Analysis)** trên câu mỉa mai & làm Demo **Gradio**. | 33.3% |

---

## 6. Lộ Trình Thực Hiện 6 Tuần & Các Mốc Nộp Bài

| Tuần | Nội dung công việc | Sản phẩm bắt buộc |
| :---: | :--- | :--- |
| **Tuần 1** | • Khởi tạo GitHub repo, mời giảng viên và các thành viên.<br>• Đọc 3 bài báo chính (PhoBERT, UIT-VSFC/ViSFD, bài báo đối sánh).<br>• Chạy EDA trên tập dữ liệu và viết Proposal. | **Proposal** (Tối đa 2 trang chuẩn ACL) |
| **Tuần 2** | • Hoàn thiện script `prepare_dataset.py` xuất 3 file CSV cố định.<br>• Chạy xong Baseline **TF-IDF + Logistic Regression**.<br>• Viết xong Dataset/DataLoader cho BiLSTM và PhoBERT. | Commit code & Log W&B đều đặn |
| **Tuần 3** | • Huấn luyện xong **BiLSTM (+Word2Vec/fastText)** và **PhoBERT-base-v2**.<br>• Tổng hợp bảng so sánh kết quả sơ bộ của 3 mô hình. | **Milestone Report** (Tối đa 4 trang chuẩn ACL) |
| **Tuần 4** | • Chạy toàn bộ thí nghiệm **Ablation** (Tách từ vs. Âm tiết; Teencode; Không dấu).<br>• Tinh chỉnh siêu tham số (Hyperparameter tuning). | Bảng kết quả Ablation & Đồ thị W&B |
| **Tuần 5** | • Hoàn thành mô hình mở rộng **ABSA**.<br>• Xuất file dự đoán Test, vẽ Confusion Matrix, lọc câu mỉa mai để phân tích lỗi. | **Bản nháp Báo cáo cuối** (Không chấm điểm) |
| **Tuần 6** | • Đóng gói ứng dụng demo bằng **Gradio** (`app.py`).<br>• Hoàn thiện báo cáo cuối, làm slide và ôn tập vấn đáp chéo code của nhau. | **Báo cáo cuối `Nhom18_DeTai06.pdf`** (Tối đa 8 trang), Code, Slide & Demo |