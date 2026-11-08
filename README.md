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


## 3. Phân Công Công Việc (3 Thành Viên)

| Thành viên | Họ và Tên | MSSV | Nhiệm vụ phụ trách chính | Tỉ lệ dự kiến |
| :--- | :--- | :--- | :--- | :---: |
| **Thành viên 1** | *(Điền tên)* | *(MSSV)* | • Xây dựng pipeline tiền xử lý (`cleaner.py`), từ điển Teencode/Emoji.<br>• Xuất bộ dữ liệu chuẩn (`train.csv`, `val.csv`, `test.csv`) & làm EDA.<br>• Cài đặt và huấn luyện **TF-IDF + Logistic Regression** (`baseline_lr.py`).<br>• Phụ trách thí nghiệm **Ablation 2: Ảnh hưởng của văn bản không dấu**. | 33.3% |
| **Thành viên 2** | *(Điền tên)* | *(MSSV)* | • Xây dựng Vocabulary & nạp Pretrained **Word2Vec / fastText** tiếng Việt.<br>• Tự cài đặt mô hình **BiLSTM** (có Masking & Pooling/Attention) bằng PyTorch.<br>• Phụ trách thí nghiệm **Ablation 1: Ảnh hưởng của tách từ (Word vs. Syllable)** trên cả 3 mô hình. | 33.3% |
| **Thành viên 3** | *(Điền tên)* | *(MSSV)* | • Fine-tune mô hình **PhoBERT (`vinai/phobert-base-v2`)** và quản lý log W&B.<br>• Cài đặt phần mở rộng **Phân tích cảm xúc theo khía cạnh (ABSA)**.<br>• Thực hiện **Phân tích lỗi (Error Analysis)** trên câu mỉa mai & làm Demo **Gradio**. | 33.3% |

---

## 4. Lộ Trình Thực Hiện 6 Tuần & Các Mốc Nộp Bài

| Tuần | Nội dung công việc | Sản phẩm bắt buộc |
| :---: | :--- | :--- |
| **Tuần 1** | • Khởi tạo GitHub repo, mời giảng viên và các thành viên.<br>• Đọc 3 bài báo chính (PhoBERT, UIT-VSFC/ViSFD, bài báo đối sánh).<br>• Chạy EDA trên tập dữ liệu và viết Proposal. | **Proposal** (Tối đa 2 trang chuẩn ACL) |
| **Tuần 2** | • Hoàn thiện script `prepare_dataset.py` xuất 3 file CSV cố định.<br>• Chạy xong Baseline **TF-IDF + Logistic Regression**.<br>• Viết xong Dataset/DataLoader cho BiLSTM và PhoBERT. | Commit code & Log W&B đều đặn |
| **Tuần 3** | • Huấn luyện xong **BiLSTM (+Word2Vec/fastText)** và **PhoBERT-base-v2**.<br>• Tổng hợp bảng so sánh kết quả sơ bộ của 3 mô hình. | **Milestone Report** (Tối đa 4 trang chuẩn ACL) |
| **Tuần 4** | • Chạy toàn bộ thí nghiệm **Ablation** (Tách từ vs. Âm tiết; Teencode; Không dấu).<br>• Tinh chỉnh siêu tham số (Hyperparameter tuning). | Bảng kết quả Ablation & Đồ thị W&B |
| **Tuần 5** | • Hoàn thành mô hình mở rộng **ABSA**.<br>• Xuất file dự đoán Test, vẽ Confusion Matrix, lọc câu mỉa mai để phân tích lỗi. | **Bản nháp Báo cáo cuối** (Không chấm điểm) |
| **Tuần 6** | • Đóng gói ứng dụng demo bằng **Gradio** (`app.py`).<br>• Hoàn thiện báo cáo cuối, làm slide và ôn tập vấn đáp chéo code của nhau. | **Báo cáo cuối `Nhom18_DeTai06.pdf`** (Tối đa 8 trang), Code, Slide & Demo |