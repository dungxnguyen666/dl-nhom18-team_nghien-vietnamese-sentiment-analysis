import argparse
import os
import pandas as pd
from datasets import load_dataset


def download_and_prepare_vsfc(output_dir: str = "data/raw"):
    """
    Tải tập dữ liệu UIT-VSFC từ HuggingFace và xuất thành 3 file CSV cố định (train, val, test).
    Đồng bộ với cấu hình trong configs/config.yaml.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"⏳ Đang tải tập dữ liệu UIT-VSFC từ HuggingFace (repo Parquet chuẩn: tridm/UIT-VSFC)...")

    try:
        dataset = load_dataset("tridm/UIT-VSFC")
    except Exception as e:
        print(f"⚠️ Fallback uitnlp/vietnamese_students_feedback... ({e})")
        dataset = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        split_name = "val" if split == "validation" else split
        save_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ Đã xuất {split_name}.csv: {len(df)} mẫu -> {save_path}")

    print("\n🎉 Hoàn thành tải và chuẩn bị dữ liệu UIT-VSFC!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải tập dữ liệu UIT-VSFC từ HuggingFace")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Thư mục lưu các file CSV (mặc định: data/raw)",
    )
    args = parser.parse_args()

    download_and_prepare_vsfc(output_dir=args.output_dir)
