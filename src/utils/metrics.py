import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> Dict[str, float]:
    """Tính toán 4 chỉ số đo lường bắt buộc theo chuẩn báo cáo ACL."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def print_evaluation_report(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    model_name: str = "Model",
) -> None:
    """In chi tiết báo cáo đánh giá và classification report."""
    metrics = compute_metrics(y_true, y_pred)
    target_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    print(f"\n{'=' * 20} KẾT QUẢ ĐÁNH GIÁ: {model_name} {'=' * 20}")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Macro-Precision : {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall    : {metrics['macro_recall']:.4f}")
    print(f"Macro-F1 (Core) : {metrics['macro_f1']:.4f}")
    print("\nChi tiết từng lớp:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))


def export_test_predictions(
    sentences: List[str],
    clean_sentences: List[str],
    true_labels: List[int],
    pred_labels: List[int],
    probabilities: np.ndarray,
    output_path: str,
) -> None:
    """
    Xuất file CSV chứa dự đoán và xác suất theo đúng định dạng 8 cột phục vụ Error Analysis:
    sentence, clean_sentence, true_label, pred_label, prob_neg, prob_neu, prob_pos, is_correct
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({
        "sentence": sentences,
        "clean_sentence": clean_sentences,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "prob_neg": probabilities[:, 0],
        "prob_neu": probabilities[:, 1],
        "prob_pos": probabilities[:, 2],
        "is_correct": np.array(true_labels) == np.array(pred_labels),
    })

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"💾 Đã lưu kết quả phân tích lỗi -> {output_path} ({len(df)} dòng)")


# Alias tương thích ngược
print_metrics_summary = print_evaluation_report
export_error_analysis_csv = export_test_predictions
