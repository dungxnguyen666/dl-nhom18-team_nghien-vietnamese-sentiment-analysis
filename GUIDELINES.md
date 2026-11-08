# GUIDELINES: ĐẶC TẢ KỸ THUẬT & HƯỚNG DẪN THỰC HIỆN (NHÓM 18)

Tài liệu này tổng hợp toàn bộ quy tắc kỹ thuật từ `README.md` và `Markdown.md` phục vụ việc thực hiện Bài tập lớn Học Sâu (Đề tài 6: Phân tích cảm xúc văn bản tiếng Việt).

## 1. Yêu cầu Bài toán
- **Đầu vào:** Phản hồi sinh viên UIT-VSFC hoặc đánh giá sản phẩm UIT-ViSFD.
- **Đầu ra:** Phân loại 3 nhãn (`0: Negative`, `1: Neutral`, `2: Positive`) kèm phân phối xác suất. Mở rộng: ABSA đa nhiệm.
- **Framework:** PyTorch, Scikit-learn, HuggingFace Transformers, PyVi, Weights & Biases, Gradio.
- **Độ đo bắt buộc:** Accuracy, Macro-Precision, Macro-Recall, và trọng tâm là **Macro-F1**.

## 2. Quy tắc Dữ liệu & Tiền xử lý
- **Single Source of Truth:** Chạy `python data/prepare_dataset.py` để xuất 3 file cố định tại `data/processed/{train,val,test}.csv` với 6 cột chuẩn:
  - `raw_text`: Văn bản gốc
  - `clean_word_nopunct`: TF-IDF & BiLSTM (lowercase, xóa dấu câu, tách từ PyVi)
  - `clean_word_keep_punct`: PhoBERT (giữ hoa/thường, giữ !?,.;:, tách từ PyVi)
  - `clean_syllable`: Ablation 1 (không tách từ ghép)
  - `clean_no_accent`: Ablation 2 (văn bản không dấu)
  - `label`: Nhãn số nguyên (0, 1, 2)
- **Cố định Random Seed:** `set_seed(42)` ở đầu mọi script huấn luyện.

## 3. Quy tắc 3 Nhánh Mô hình
- **Mô hình 1 (`baseline_lr.py`):** TF-IDF n-gram (1, 2), max_features=10000. Chỉ fit trên Train. LogisticRegression `class_weight='balanced'`.
- **Mô hình 2 (`bilstm.py`):** Xây dựng vocab từ Train (0: `<pad>`, 1: `<unk>`). Nạp pretrained PhoW2V/fastText. BiLSTM 2 layers, dropout 0.3, có Masking chống nhiễu token `<pad>`.
- **Mô hình 3 (`phobert.py`):** `vinai/phobert-base-v2`, max_length=128, lr=2e-5, warmup, AdamW, early stopping theo `val_macro_f1`.
- **Đồng nhất Loss:** Cả 3 mô hình xử lý lệch lớp bằng `class_weights` tính trên tập Train.

## 4. Đánh giá & Error Analysis
- Mọi mô hình khi test phải xuất file CSV tại `logs/<model>_test_preds.csv` gồm:
  `sentence, clean_sentence, true_label, pred_label, prob_neg, prob_neu, prob_pos, is_correct`
- Phân tích sâu các ca lỗi kinh điển: câu mỉa mai (sarcasm), câu 2 vế khen-chê đối lập, teencode/không dấu.

