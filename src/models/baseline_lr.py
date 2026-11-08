import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing.cleaner import VietnameseTextCleaner
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics, export_error_analysis_csv, print_metrics_summary


class BaselineTFIDFLogisticRegression:
    """
    Nhánh 1: TF-IDF + Logistic Regression.
    
    Quy tắc kỹ thuật bắt buộc (Mục 3.A Markdown.md):
    - Chống Data Leakage: fit_transform() CHỈ trên tập Train, Val/Test chỉ transform().
    - Cấu hình chuẩn: ngram_range=(1, 2), max_features=10000, class_weight='balanced'.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        c_param: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = 42,
        cleaner: Optional[VietnameseTextCleaner] = None,
    ):
        self.cleaner = cleaner or VietnameseTextCleaner()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=250,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs",
        )
        self.is_fitted = False

    def clean_texts(self, texts: List[str]) -> List[str]:
        """Làm sạch văn bản bằng VietnameseTextCleaner."""
        return [self.cleaner.clean(t) for t in texts]

    def fit(self, train_texts: List[str], train_labels: Union[List[int], np.ndarray]) -> "BaselineTFIDFLogisticRegression":
        """
        Huấn luyện TF-IDF và Logistic Regression trên tập Train.
        CHỈ gọi fit trên tập Train để chống rò rỉ dữ liệu (Data Leakage).
        """
        clean_train = self.clean_texts(train_texts)
        x_train = self.vectorizer.fit_transform(clean_train)
        self.classifier.fit(x_train, train_labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Dự đoán nhãn (0, 1, 2) cho các câu văn bản."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit! Hãy gọi fit() trước.")
        clean_texts = self.clean_texts(texts)
        x = self.vectorizer.transform(clean_texts)
        return self.classifier.predict(x)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất cho 3 lớp [prob_neg, prob_neu, prob_pos]."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit! Hãy gọi fit() trước.")
        clean_texts = self.clean_texts(texts)
        x = self.vectorizer.transform(clean_texts)
        return self.classifier.predict_proba(x)

    def evaluate(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        export_csv_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Đánh giá mô hình và tùy chọn xuất file CSV phục vụ phân tích lỗi."""
        clean_texts = self.clean_texts(texts)
        x = self.vectorizer.transform(clean_texts)
        y_pred = self.classifier.predict(x)
        probabilities = self.classifier.predict_proba(x)

        metrics = compute_metrics(labels, y_pred)
        print_metrics_summary(metrics, model_name="Baseline TF-IDF + Logistic Regression")

        if export_csv_path:
            label_names = {0: "negative", 1: "neutral", 2: "positive"}
            export_error_analysis_csv(
                output_path=export_csv_path,
                sentences=texts,
                clean_sentences=clean_texts,
                y_true=labels,
                y_pred=y_pred,
                probabilities=probabilities,
                label_names=label_names,
            )

        return metrics

    def save(self, filepath: str) -> None:
        """Lưu toàn bộ pipeline (vectorizer + classifier) xuống đĩa."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, f)
        print(f"💾 Đã lưu mô hình Baseline vào: {filepath}")

    def load(self, filepath: str) -> "BaselineTFIDFLogisticRegression":
        """Tải mô hình đã lưu từ đĩa."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.is_fitted = True
        print(f"📂 Đã tải mô hình Baseline từ: {filepath}")
        return self


def run_baseline_training(config_path: str = "configs/config.yaml"):
    """Script huấn luyện độc lập cho mô hình Baseline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Cố định Random Seed theo Mục 3.1
    seed = cfg["project"].get("seed", 42)
    set_seed(seed)

    # 2. Đọc dữ liệu từ data/processed/ (Single Source of Truth)
    train_df = pd.read_csv(cfg["paths"]["train_data"])
    val_df = pd.read_csv(cfg["paths"]["val_data"])
    test_df = pd.read_csv(cfg["paths"]["test_data"])

    # Ưu tiên cột đã tiền xử lý chuẩn clean_word_nopunct cho TF-IDF
    if "clean_word_nopunct" in train_df.columns:
        text_col = "clean_word_nopunct"
    elif "sentence" in train_df.columns:
        text_col = "sentence"
    else:
        text_col = "Sentence"

    label_col = "label" if "label" in train_df.columns else (
        "sentiment" if "sentiment" in train_df.columns else "Encoded_sentiment"
    )

    raw_text_col = "raw_text" if "raw_text" in test_df.columns else text_col

    print(f"📊 Dữ liệu: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"📌 Cột văn bản: '{text_col}', Cột nhãn: '{label_col}'")

    # Khởi tạo mô hình
    model_cfg = cfg["models"]["baseline_tfidf"]
    model = BaselineTFIDFLogisticRegression(
        max_features=model_cfg.get("max_features", 10000),
        ngram_range=tuple(model_cfg.get("ngram_range", [1, 2])),
        c_param=model_cfg.get("c_param", 1.0),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=cfg["project"].get("seed", 42),
    )

    # Fit trên tập Train
    print("⏳ Đang huấn luyện TF-IDF + Logistic Regression trên tập Train...")
    model.fit(train_df[text_col].astype(str).tolist(), train_df[label_col].values)

    # Đánh giá trên Validation
    print("🧪 Đánh giá trên tập Validation:")
    model.evaluate(val_df[text_col].astype(str).tolist(), val_df[label_col].values)

    # Đánh giá trên Test & Xuất file Error Analysis
    test_csv_path = os.path.join(cfg["paths"]["logs_dir"], "baseline_tfidf_test_preds.csv")
    print("🧪 Đánh giá trên tập Test & Xuất file Error Analysis:")
    clean_texts = model.clean_texts(test_df[text_col].astype(str).tolist())
    x_test = model.vectorizer.transform(clean_texts)
    y_pred = model.classifier.predict(x_test)
    probabilities = model.classifier.predict_proba(x_test)
    labels = test_df[label_col].values

    metrics = compute_metrics(labels, y_pred)
    print_metrics_summary(metrics, model_name="Baseline TF-IDF + Logistic Regression")

    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    export_error_analysis_csv(
        output_path=test_csv_path,
        sentences=test_df[raw_text_col].astype(str).tolist(),
        clean_sentences=clean_texts,
        y_true=labels,
        y_pred=y_pred,
        probabilities=probabilities,
        label_names=label_names,
    )

    # Lưu mô hình
    save_path = os.path.join(cfg["paths"]["checkpoints_dir"], "baseline_tfidf_lr.pkl")
    model.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file config.yaml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        run_baseline_training(args.config)
    else:
        print(f"Không tìm thấy file cấu hình {args.config}")
