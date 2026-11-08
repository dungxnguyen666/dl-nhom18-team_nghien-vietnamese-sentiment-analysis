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

---

## 5. QUY CHUẨN QUẢN LÝ MÃ NGUỒN & HƯỚNG DẪN GIT COMMIT

Để tránh xung đột code giữa 3 thành viên (merge conflict) và tuyệt đối không làm nặng GitHub repo dẫn đến lỗi push/pull, mọi thành viên bắt buộc phải tuân thủ nghiêm ngặt các quy tắc dưới đây:

### 5.1. Nguyên tắc cốt lõi (Core Principles)
1. **Tuyệt đối KHÔNG commit dữ liệu và trọng số:** 
   - Không commit các thư mục `data/raw/`, `data/processed/`, `data/embeddings/`.
   - Không commit file nhị phân checkpoint mô hình (`.pt`, `.bin`, `.joblib`, `.onnx`).
   - Không commit logs đánh giá test (`logs/*.csv`) và thư mục môi trường ảo (`.venv/`, `venv/`).
2. **Luôn kiểm tra trạng thái trước khi commit:**
   - Trước khi `git add`, luôn chạy `git status` để đảm bảo không có file nặng nào lọt vào staging area.

### 5.2. Phân loại File NÊN và CẤM Commit

| Trạng thái | Danh sách file / thư mục |
| :--- | :--- |
| **ĐƯỢC PHÉP COMMIT** | • Mã nguồn (`src/`, `train.py`, `evaluate.py`, `app.py`).<br>• Cấu hình & Quản trị (`configs/config.yaml`, `requirements.txt`, `.gitignore`).<br>• Tài liệu hướng dẫn (`README.md`, `GUIDELINES.md`).<br>• Notebooks (chỉ lưu code, **phải Clear Outputs** trước khi commit).<br>• Các file `.gitkeep` để giữ cấu trúc thư mục rỗng. |
| **CẤM COMMIT** *(Đã chặn bằng `.gitignore`)* | • Dữ liệu: `data/processed/*.csv`, `data/raw/*`.<br>• Embeddings: `data/embeddings/*` (PhoW2V, fastText).<br>• Trọng số mô hình: `models/checkpoints/*.pt`, `*.bin`, `*.joblib`.<br>• Logs chi tiết: `logs/*.csv`, `wandb/`, `runs/`.<br>• Môi trường: `venv/`, `.env`, `__pycache__/`. |

---

### 5.3. Quy chuẩn viết Commit Message (Conventional Commits)
Thông điệp commit phải ngắn gọn, rõ ràng theo cấu trúc: `<loại_thay_đổi>: <mô tả ngắn bằng tiếng Anh hoặc tiếng Việt>`

* `feat:` Thêm tính năng mới hoặc mô hình mới (VD: `feat: implement bilstm with masked mean pooling`).
* `fix:` Sửa lỗi logic hoặc lỗi xử lý văn bản (VD: `fix: handle edge cases in teencode normalization`).
* `refactor:` Tối ưu hóa code mà không thay đổi chức năng (VD: `refactor: optimize DataLoader memory footprint`).
* `docs:` Cập nhật tài liệu (VD: `docs: update guidelines with git commit rules`).
* `chore:` Cập nhật dependencies, cấu hình yaml hoặc gitignore (VD: `chore: add gitkeep and update requirements.txt`).

---

### 5.4. Quy trình thực hiện Git Commit chuẩn từng bước (CLI / PowerShell)

Mỗi khi hoàn thành một chức năng, thành viên thực hiện đúng 5 bước sau:

#### Bước 1: Kéo code mới nhất từ nhánh chính về
Tránh bị lệch code so với các thành viên khác:
```bash
git pull origin main