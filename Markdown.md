# TECHNICAL SPECIFICATION & CODING GUIDELINES: ĐỀ TÀI 6 - PHÂN TÍCH CẢM XÚC VĂN BẢN TIẾNG VIỆT (NHÓM 18)

Bạn là kỹ sư Deep Learning / NLP hỗ trợ Nhóm 18 thực hiện Bài tập lớn môn Học sâu (Đề tài 6: Phân tích cảm xúc văn bản tiếng Việt). Khi viết code hoặc nội dung báo cáo (chuẩn hội nghị ACL), bạn BẮT BUỘC phải tuân thủ nghiêm ngặt các quy tắc kỹ thuật được đúc kết từ tài liệu tham khảo dưới đây:

## 1. YÊU CẦU BÀI TOÁN & DỮ LIỆU
- **Đầu vào:** Câu bình luận tiếng Việt (phản hồi sinh viên UIT-VSFC hoặc đánh giá sản phẩm UIT-ViSFD).
- **Đầu ra:** Phân loại 3 lớp cảm xúc (`0: Negative/Tiêu cực`, `1: Neutral/Trung tính`, `2: Positive/Tích cực`) kèm phân phối xác suất của từng lớp. Phần mở rộng: Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA).
- **Framework:** BẮT BUỘC dùng `PyTorch`, `Scikit-learn`, và `HuggingFace Transformers` (KHÔNG dùng TensorFlow/Keras).
- **Quản lý dữ liệu:** Chia tập `Train / Val / Test` có phân tầng (`stratified split`, `seed=42`) hoặc dùng đúng tập chia sẵn của UIT-VSFC/UIT-ViSFD. KHÔNG commit dữ liệu lớn hoặc file trọng số (`.pt`, `.bin`) lên GitHub.

### 1.1. Nguồn tải & Script chuẩn bị dữ liệu tự động (`data/download_data.py`)
- **Nguồn tải trực tiếp (Dataset Sources):**
  - **Tập UIT-VSFC (Khuyên dùng cho Đề tài 6):**
    - **HuggingFace:** `uitnlp/vietnamese_students_feedback` (Đã chia sẵn 3 tập train / validation / test).
    - **GitHub chính thức của UIT:** Repo `UIT-NLP/Vietnamese-Students-Feedback-Corpus`.
- **Đoạn mã chuẩn bị dữ liệu (`data/download_data.py`):**
```python
import os
import pandas as pd
from datasets import load_dataset

def download_and_prepare_vsfc(output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    print("⏳ Đang tải tập dữ liệu UIT-VSFC từ HuggingFace...")
    
    # Tải dataset chính thức từ HuggingFace
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")
    
    # Chuyển đổi sang Pandas DataFrame và lưu thành 3 file CSV cố định
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        # Đổi tên tập validation -> val cho đồng nhất với config.yaml
        split_name = "val" if split == "validation" else split
        save_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ Đã xuất {split_name}.csv: {len(df)} mẫu -> {save_path}")

if __name__ == "__main__":
    download_and_prepare_vsfc()
```

## 2. PIPELINE TIỀN XỬ LÝ VĂN BẢN TIẾNG VIỆT (`src/preprocessing/cleaner.py`)
Phải thực hiện đúng thứ tự 7 bước sau để tránh lỗi logic và mất tín hiệu cảm xúc:
1. **Chuẩn hóa Unicode:** Đưa toàn bộ văn bản về chuẩn dựng sẵn `NFC` (`unicodedata.normalize("NFC", text)`).
2. **Ánh xạ Emoji & Emoticon TRƯỚC KHI xóa ký tự đặc biệt:** Thay thế các emoji/emoticon phổ biến (`:))`, `:(`, `colonsmile`, `colonsad`, `😭`, `😡`, `❤️`...) thành các token mang sắc thái cảm xúc (`tích_cực`, `tiêu_cực`, `trung_tính`).
3. **Chuẩn hóa chữ thường & Số:** Chuyển về lowercase (có cờ bật/tắt riêng cho PhoBERT) và chuẩn hóa các con số thành token `[number]` nếu cần giảm nhiễu.
4. **Chuẩn hóa Teencode & Viết tắt TRƯỚC KHI xóa dấu câu:** Dùng regex tách từ và dấu câu (`re.findall(r"\w+|[^\w\s]", text)`) rồi mới tra từ điển Teencode (`"ko"` -> `"không"`, `"ok!"` -> `"tốt"`, `"sv"` -> `"sinh viên"`, `"gv"` -> `"giảng viên"`...) để không bị sót các từ dính liền dấu câu.
5. **Làm sạch ký tự đặc biệt:** Xóa URL, HTML tags và ký tự lạ nhưng PHẢI GIỮ LẠI chữ cái tiếng Việt có dấu và dấu gạch dưới `_` (`re.sub(r"[^\w\s\d_]", " ", text)`).
6. **TUYỆT ĐỐI KHÔNG xóa Stopwords bừa bãi:** Không dùng bộ stopwords mặc định vì sẽ xóa mất các từ phủ định và từ chỉ mức độ quan trọng trong tiếng Việt (*không, chẳng, chưa, rất, quá, nhưng, tuy nhiên*).
7. **Tách từ (Word Segmentation) & Cờ Ablation Study:**
   - Hỗ trợ bật/tắt tách từ bằng `pyvi` (`ViTokenizer.tokenize`) hoặc `underthesea` / `VnCoreNLP` để nối từ ghép bằng dấu `_` (ví dụ: `học_sâu`, `giảng_viên`).
   - Hỗ trợ cờ `remove_diacritics` (xóa dấu tiếng Việt) để phục vụ thí nghiệm đánh giá độ bền vững trên văn bản không dấu.

## 3. NGUYÊN TẮC THIẾT KẾ 3 MÔ HÌNH ĐỐI SÁNH (`src/models/`)
### A. Mô hình 1: TF-IDF + Logistic Regression (`baseline_lr.py`)
- **Chống Data Leakage:** Chỉ gọi `tfidf.fit_transform()` trên tập `Train`, và chỉ gọi `tfidf.transform()` trên tập `Val` và `Test`.
- **Cấu hình:** `ngram_range=(1, 2)`, `max_features=10000`, `LogisticRegression(C=1.0, max_iter=250, class_weight="balanced")` để xử lý mất cân bằng ở lớp `Neutral`.

### B. Mô hình 2: BiLSTM + Pretrained Word2Vec/fastText tiếng Việt (`bilstm.py`)
- **Quản lý bộ nhớ & Vocabulary (KHÔNG làm theo cách tạo mảng 3D float32 ngoài RAM của notebook IMDB):**
  - Xây dựng bộ từ điển `vocab` chỉ từ tập `Train` với `0: <pad>` và `1: <unk>`.
  - Biểu diễn câu dưới dạng chuỗi số nguyên 2D `(batch_size, max_len)` kiểu `torch.long`.
  - Tải trọng số Pretrained tiếng Việt (PhoW2V hoặc fastText `cc.vi.300`) nạp vào ma trận `embedding_matrix` kích thước `(vocab_size, embedding_dim)`, sử dụng `nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)`.
- **Kiến trúc mạng & Masking (Chuẩn bị cho vấn đáp cá nhân):**
  - Sử dụng `LSTM(..., bidirectional=True, num_layers=2, dropout=0.3)`.
  - Bắt buộc xử lý Padding bằng `pack_padded_sequence` / `pad_packed_sequence` hoặc truyền `attention_mask = (input_ids != 0)` khi thực hiện Mean/Max Pooling hoặc Attention để các token `<pad>` ở cuối câu ngắn không làm nhiễu vector biểu diễn câu.
  - Đầu ra là tầng `nn.Linear(hidden_dim * 2, 3)` kết hợp `nn.CrossEntropyLoss(weight=class_weights)`.

### C. Mô hình 3: Pretrained Transformer - PhoBERT (`phobert.py`)
- **Model checkpoint:** `vinai/phobert-base-v2` (yêu cầu đầu vào ĐÃ được tách từ bằng dấu `_`).
- **Siêu tham số chuẩn (Tối ưu cho GPU T4):** `max_length=128`, `batch_size=16` hoặc `32`, Optimizer `AdamW(lr=2e-5, weight_decay=0.01)`, `linear warmup scheduler`, huấn luyện `3-5 epochs` kèm `EarlyStopping` dựa trên `val_macro_f1`.

## 4. ĐÁNH GIÁ, LOGGING & XUẤT FILE PHÂN TÍCH LỖI (`src/utils/`)
- **Thước đo bắt buộc:** Luôn tính và in ra đồng thời **Accuracy**, **Macro-Precision**, **Macro-Recall**, và **Macro-F1** (`f1_score(y_true, y_pred, average='macro')`). Không dùng `RandomOverSampler` nhân bản văn bản gây overfitting/data leakage; ưu tiên xử lý lệch lớp bằng `class_weight`.
- **Logging:** Tích hợp `Weights & Biases (wandb)` để log `train_loss`, `val_loss`, `val_acc`, `val_macro_f1` sau mỗi epoch.
- **Xuất file dự đoán phục vụ Error Analysis (Học từ Mục 8 Notebook IMDB):**
  - Mọi mô hình sau khi đánh giá trên tập Test phải xuất ra một file CSV (`logs/<model_name>_test_preds.csv`) gồm các cột:
    `sentence, clean_sentence, true_label, pred_label, prob_neg, prob_neu, prob_pos, is_correct`
  - Dùng các file CSV này để vẽ `Confusion Matrix` và lọc tự động các ca dự đoán sai kinh điển: (1) Câu mỉa mai (sarcasm), (2) Câu chứa 2 vế khen-chê đối lập (thường thuộc lớp Neutral nhưng bị đoán nhầm), (3) Câu chứa teencode/không dấu.
- **Demo:** Xây dựng giao diện `app.py` bằng `Gradio` cho phép nhập 1 câu tiếng Việt và hiển thị trực quan nhãn + thanh xác suất dự đoán của cả 3 mô hình cùng lúc.