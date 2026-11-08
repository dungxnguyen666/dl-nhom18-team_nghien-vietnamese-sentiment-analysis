import re
import unicodedata
from typing import Dict, Optional

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


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
        if self.use_word_segmentation and ViTokenizer is not None:
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


if __name__ == "__main__":
    cleaner = VietnameseTextCleaner()
    sample = "Thầy dạy vkl hay, ok! Ko uổng công đk môn học sâu :))"
    print("📌 Văn bản gốc:", sample)
    print("✅ Sau tiền xử lý (Chuẩn):", cleaner.clean(sample))
    print("🧪 Thí nghiệm không dấu:", cleaner.clean(sample, remove_accents=True))
