import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Thử import các thư viện tách từ tiếng Việt phổ biến
try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

try:
    from underthesea import word_tokenize as underthesea_tokenize
except ImportError:
    underthesea_tokenize = None


class VietnameseWordTokenizer:
    """
    Module phân tách từ tiếng Việt (Word Segmentation).
    
    Hỗ trợ:
    - 'pyvi': Nối từ ghép bằng dấu gạch dưới (VD: 'học sâu' -> 'học_sâu')
    - 'underthesea': Tách từ theo cú pháp Underthesea
    - 'none' / 'syllable': Giữ nguyên cấp độ âm tiết (không nối từ ghép) cho Ablation Study
    """

    AVAILABLE_ENGINES = ["pyvi", "underthesea", "none"]

    def __init__(self, engine: str = "pyvi"):
        self.engine = engine.lower()
        if self.engine not in self.AVAILABLE_ENGINES:
            raise ValueError(
                f"Engine '{engine}' không hợp lệ. Chọn một trong: {self.AVAILABLE_ENGINES}"
            )

        if self.engine == "pyvi" and ViTokenizer is None:
            logger.warning(
                "Thư viện 'pyvi' chưa được cài đặt. Sẽ fallback sang không tách từ ('none')."
            )
        elif self.engine == "underthesea" and underthesea_tokenize is None:
            logger.warning(
                "Thư viện 'underthesea' chưa được cài đặt. Sẽ fallback sang không tách từ ('none')."
            )

    def tokenize(self, text: str) -> str:
        """Tách từ ghép cho một câu văn bản."""
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.engine == "pyvi" and ViTokenizer is not None:
            return ViTokenizer.tokenize(text)
        elif self.engine == "underthesea" and underthesea_tokenize is not None:
            return underthesea_tokenize(text, format="text")
        else:
            # Không tách từ ghép (syllable-level)
            return text

    def tokenize_batch(self, texts: List[str]) -> List[str]:
        """Tách từ ghép cho danh sách các câu văn bản."""
        return [self.tokenize(text) for text in texts]


if __name__ == "__main__":
    sample_text = "Thầy cô giảng dạy môn học sâu rất nhiệt tình."
    print("Văn bản gốc:", sample_text)

    # Thử nghiệm PyVi
    tok_pyvi = VietnameseWordTokenizer(engine="pyvi")
    print("PyVi:", tok_pyvi.tokenize(sample_text))

    # Thử nghiệm Underthesea
    tok_uts = VietnameseWordTokenizer(engine="underthesea")
    print("Underthesea:", tok_uts.tokenize(sample_text))

    # Thử nghiệm Không tách từ (Ablation)
    tok_none = VietnameseWordTokenizer(engine="none")
    print("Không tách từ (Syllable-level):", tok_none.tokenize(sample_text))

