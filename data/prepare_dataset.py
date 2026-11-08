import os
import sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Thêm đường dẫn gốc để import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.cleaner import VietnameseTextCleaner


def prepare_vsfc_dataset(raw_dir: str = "data/raw", output_dir: str = "data/processed") -> None:
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print("⏳ Đang tải UIT-VSFC từ HuggingFace (repo Parquet chuẩn: tridm/UIT-VSFC)...")
    
    # tridm/UIT-VSFC chứa đúng nguyên bản UIT-VSFC dạng Parquet, tương thích datasets >= 3.0
    try:
        raw_dataset = load_dataset("tridm/UIT-VSFC")
    except Exception as e:
        print(f"⚠️ Thử tải từ uitnlp/vietnamese_students_feedback (trust_remote_code=True)... ({e})")
        raw_dataset = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)

    # Khởi tạo các bộ clean theo từng cấu hình
    cleaner_nopunct = VietnameseTextCleaner(lowercase=True, keep_punctuation=False, use_word_segmentation=True)
    cleaner_keep_punct = VietnameseTextCleaner(lowercase=False, keep_punctuation=True, use_word_segmentation=True)
    cleaner_syllable = VietnameseTextCleaner(lowercase=True, keep_punctuation=False, use_word_segmentation=False)
    cleaner_no_accent = VietnameseTextCleaner(lowercase=True, keep_punctuation=False, use_word_segmentation=True)

    splits = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }

    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    for hf_split, out_filename in splits.items():
        print(f"\n🔄 Đang xử lý tập: {hf_split} -> {out_filename}")
        data = raw_dataset[hf_split]

        # 1. Lưu bản raw nguyên gốc vào data/raw/
        raw_csv_path = os.path.join(raw_dir, out_filename)
        pd.DataFrame(data).to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
        print(f"📁 Đã lưu dữ liệu thô -> {raw_csv_path}")

        # Lấy tên cột văn bản và nhãn tương ứng
        text_col = "sentence" if "sentence" in data.column_names else "Sentence"
        raw_texts = data[text_col]

        if "Encoded_sentiment" in data.column_names:
            labels = data["Encoded_sentiment"]
        elif "sentiment" in data.column_names:
            first_label = data["sentiment"][0]
            if isinstance(first_label, str):
                labels = [label_map.get(str(s).lower().strip(), 1) for s in data["sentiment"]]
            else:
                labels = data["sentiment"]
        else:
            labels = data["label"]

        clean_word_nopunct = []
        clean_word_keep_punct = []
        clean_syllable = []
        clean_no_accent = []

        for text in tqdm(raw_texts, desc=f"Tiền xử lý {hf_split}"):
            text_str = str(text) if text is not None else ""
            clean_word_nopunct.append(cleaner_nopunct.clean(text_str))
            clean_word_keep_punct.append(cleaner_keep_punct.clean(text_str))
            clean_syllable.append(cleaner_syllable.clean(text_str))
            clean_no_accent.append(cleaner_no_accent.clean(text_str, remove_accents=True))

        df_dict = {
            "raw_text": raw_texts,
            "clean_word_nopunct": clean_word_nopunct,
            "clean_word_keep_punct": clean_word_keep_punct,
            "clean_syllable": clean_syllable,
            "clean_no_accent": clean_no_accent,
            "label": labels,
        }

        # Lưu thêm thông tin topic cho bài toán mở rộng ABSA nếu có
        if "Topic" in data.column_names:
            df_dict["topic"] = data["Topic"]
        elif "topic" in data.column_names:
            df_dict["topic"] = data["topic"]

        if "Encoded_topic" in data.column_names:
            df_dict["encoded_topic"] = data["Encoded_topic"]

        df = pd.DataFrame(df_dict)

        # Lọc bỏ các dòng có text rỗng sau khi làm sạch
        initial_len = len(df)
        df = df[df["clean_word_nopunct"].str.len() > 0].reset_index(drop=True)
        if len(df) < initial_len:
            print(f"⚠️ Đã loại bỏ {initial_len - len(df)} mẫu rỗng.")

        out_path = os.path.join(output_dir, out_filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ Đã xuất {out_path} ({len(df)} mẫu)")


if __name__ == "__main__":
    prepare_vsfc_dataset()
