"""
Nhánh 1 ─ Baseline: TF-IDF + Logistic Regression
==================================================
Quy tắc kỹ thuật (GUIDELINES §3-A):
  • Chống Data Leakage: fit_transform() CHỈ trên tập Train; Val/Test chỉ transform().
  • Cấu hình chuẩn: ngram_range=(1,2), max_features=10 000, class_weight='balanced'.
  • Dữ liệu đầu vào đã được tiền xử lý bởi prepare_dataset.py → KHÔNG clean lại.
"""

import argparse
import os
import sys
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Đảm bảo import được khi chạy trực tiếp: python src/models/baseline_lr.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.metrics import (
    compute_metrics,
    export_error_analysis_csv,
    print_metrics_summary,
)
from src.utils.seed import set_seed


# ──────────────────────────────────────────────
# 1. MODEL CLASS
# ──────────────────────────────────────────────
class BaselineTFIDFLogisticRegression:
    """
    TF-IDF (1,2)-gram + Logistic Regression cho phân loại cảm xúc 3 lớp.

    Lưu ý quan trọng:
      Các phương thức fit / predict / evaluate nhận **văn bản đã tiền xử lý**
      (cột clean_word_nopunct từ data/processed/). KHÔNG tự gọi cleaner bên
      trong để tránh lỗi double-cleaning.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: Tuple[int, int] = (1, 2),
        c_param: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # Dùng 1 + log(TF) → giảm ảnh hưởng từ lặp quá nhiều
        )
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=500,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs",           # Thuật toán tối ưu chuẩn cho multiclass
        )
        self.is_fitted = False

    # ── Huấn luyện ───────────────────────────
    def fit(
        self,
        train_texts: List[str],
        train_labels: Union[List[int], np.ndarray],
    ) -> "BaselineTFIDFLogisticRegression":
        """
        Huấn luyện TF-IDF và Logistic Regression trên tập Train.
        CHỈ gọi fit_transform() ở đây để chống rò rỉ dữ liệu (Data Leakage).
        """
        x_train = self.vectorizer.fit_transform(train_texts)
        self.classifier.fit(x_train, train_labels)
        self.is_fitted = True
        return self

    # ── Dự đoán ──────────────────────────────
    def predict(self, texts: List[str]) -> np.ndarray:
        """Dự đoán nhãn (0, 1, 2) cho danh sách câu đã tiền xử lý."""
        self._check_fitted()
        x = self.vectorizer.transform(texts)
        return self.classifier.predict(x)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất [prob_neg, prob_neu, prob_pos] cho danh sách câu."""
        self._check_fitted()
        x = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(x)

    # ── Đánh giá ─────────────────────────────
    def evaluate(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        raw_texts: Optional[List[str]] = None,
        export_csv_path: Optional[str] = None,
        split_name: str = "Test",
    ) -> Dict[str, float]:
        """
        Đánh giá mô hình trên một tập dữ liệu bất kỳ (Val hoặc Test).

        Parameters
        ----------
        texts : danh sách câu đã tiền xử lý (clean)
        labels : nhãn thật (0, 1, 2)
        raw_texts : câu gốc chưa xử lý (để ghi vào CSV phân tích lỗi)
        export_csv_path : đường dẫn xuất CSV phân tích lỗi (tùy chọn)
        split_name : tên tập (hiển thị khi in kết quả)
        """
        y_pred = self.predict(texts)
        probabilities = self.predict_proba(texts)

        metrics = compute_metrics(labels, y_pred)
        print_metrics_summary(
            y_true=labels,
            y_pred=y_pred,
            model_name=f"Baseline TF-IDF + LR ({split_name})",
        )

        if export_csv_path:
            export_error_analysis_csv(
                sentences=raw_texts if raw_texts else texts,
                clean_sentences=texts,
                true_labels=list(labels),
                pred_labels=list(y_pred),
                probabilities=probabilities,
                output_path=export_csv_path,
            )

        return metrics

    # ── Lưu / Tải ────────────────────────────
    def save(self, filepath: str) -> None:
        """Lưu pipeline (vectorizer + classifier) xuống đĩa."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "classifier": self.classifier}, f
            )
        print(f"💾 Đã lưu mô hình Baseline → {filepath}")

    def load(self, filepath: str) -> "BaselineTFIDFLogisticRegression":
        """Tải mô hình đã lưu từ đĩa."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.is_fitted = True
        print(f"📂 Đã tải mô hình Baseline ← {filepath}")
        return self

    # ── Helper ────────────────────────────────
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit! Hãy gọi fit() trước.")


# ──────────────────────────────────────────────
# 2. TRAINING SCRIPT
# ──────────────────────────────────────────────
# Mapping cột tiền xử lý cho các thí nghiệm
TEXT_COL_OPTIONS = {
    "word":     "clean_word_nopunct",   # Mặc định: tách từ, bỏ dấu câu
    "syllable": "clean_syllable",       # Ablation 1: không tách từ ghép
    "no_accent": "clean_no_accent",     # Ablation 2: văn bản không dấu
}


def run_baseline_training(
    config_path: str = "configs/config.yaml",
    text_mode: str = "word",
) -> Dict[str, float]:
    """
    Huấn luyện và đánh giá Baseline TF-IDF + LR.

    Parameters
    ----------
    config_path : đường dẫn tới config.yaml
    text_mode : chọn cột tiền xử lý ('word' | 'syllable' | 'no_accent')
                phục vụ Ablation Study.
    """
    # ── 1. Đọc config & cố định seed ────────
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"].get("seed", 42)
    set_seed(seed)

    # ── 2. Đọc dữ liệu đã tiền xử lý ──────
    train_df = pd.read_csv(cfg["paths"]["train_data"])
    val_df   = pd.read_csv(cfg["paths"]["val_data"])
    test_df  = pd.read_csv(cfg["paths"]["test_data"])

    text_col = TEXT_COL_OPTIONS.get(text_mode, "clean_word_nopunct")
    if text_col not in train_df.columns:
        raise ValueError(
            f"Cột '{text_col}' không tồn tại trong dữ liệu. "
            f"Các cột hiện có: {list(train_df.columns)}"
        )

    label_col = "label"
    raw_text_col = "raw_text"

    # Chuyển NaN thành chuỗi rỗng để tránh lỗi
    for df in [train_df, val_df, test_df]:
        df[text_col] = df[text_col].fillna("").astype(str)

    print(f"\n{'='*60}")
    print(f"  BASELINE TF-IDF + LOGISTIC REGRESSION")
    print(f"  Text mode : {text_mode} → cột '{text_col}'")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"{'='*60}\n")

    # ── 3. Phân phối nhãn ───────────────────
    print("📊 Phân phối nhãn tập Train:")
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for lbl, name in label_names.items():
        count = (train_df[label_col] == lbl).sum()
        pct = count / len(train_df) * 100
        print(f"   {name} ({lbl}): {count:>5d} mẫu ({pct:.1f}%)")
    print()

    # ── 4. Khởi tạo mô hình từ config ──────
    model_cfg = cfg["models"]["baseline_tfidf"]
    model = BaselineTFIDFLogisticRegression(
        max_features=model_cfg.get("max_features", 10_000),
        ngram_range=tuple(model_cfg.get("ngram_range", [1, 2])),
        c_param=model_cfg.get("c_param", 1.0),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=seed,
    )

    # ── 5. Huấn luyện trên tập Train ───────
    print("⏳ Đang huấn luyện...")
    t_start = time.time()

    model.fit(
        train_texts=train_df[text_col].tolist(),
        train_labels=train_df[label_col].values,
    )

    t_elapsed = time.time() - t_start
    print(f"✅ Huấn luyện xong trong {t_elapsed:.2f} giây\n")

    # ── 6. Đánh giá trên Validation ────────
    print("── Validation ──")
    val_metrics = model.evaluate(
        texts=val_df[text_col].tolist(),
        labels=val_df[label_col].values,
        split_name="Validation",
    )

    # ── 7. Đánh giá trên Test & Xuất CSV ───
    logs_dir = cfg["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    test_csv_path = os.path.join(logs_dir, f"baseline_tfidf_{text_mode}_test_preds.csv")

    print("\n── Test ──")
    test_metrics = model.evaluate(
        texts=test_df[text_col].tolist(),
        labels=test_df[label_col].values,
        raw_texts=test_df[raw_text_col].astype(str).tolist(),
        export_csv_path=test_csv_path,
        split_name="Test",
    )

    # ── 8. Lưu mô hình ────────────────────
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"baseline_tfidf_{text_mode}.pkl")
    model.save(save_path)

    # ── 9. Tóm tắt kết quả ────────────────
    print(f"\n{'='*60}")
    print(f"  TÓM TẮT KẾT QUẢ ({text_mode})")
    print(f"  Val  Macro-F1 : {val_metrics['macro_f1']:.4f}")
    print(f"  Test Macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Checkpoint    : {save_path}")
    print(f"  Error Analysis: {test_csv_path}")
    print(f"{'='*60}\n")

    return test_metrics


# ──────────────────────────────────────────────
# 3. CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện Baseline TF-IDF + Logistic Regression"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Đường dẫn file config.yaml",
    )
    parser.add_argument(
        "--text-mode", type=str, default="word",
        choices=list(TEXT_COL_OPTIONS.keys()),
        help="Chọn cột tiền xử lý: word (mặc định) | syllable (Ablation 1) | no_accent (Ablation 2)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Không tìm thấy file cấu hình: {args.config}")
    else:
        run_baseline_training(config_path=args.config, text_mode=args.text_mode)
