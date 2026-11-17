import argparse
import os

from datasets import load_dataset

PRIMARY_REPO = "tridm/UIT-VSFC"
FALLBACK_REPO = "uitnlp/vietnamese_students_feedback"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def load_vsfc():
    try:
        return load_dataset(PRIMARY_REPO)
    except Exception as e:
        print(f"⚠️ Không tải được {PRIMARY_REPO} ({e}). Thử {FALLBACK_REPO}...")
        return load_dataset(FALLBACK_REPO)


def download_and_prepare_vsfc(output_dir: str = "data/raw", force: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    targets = {s: os.path.join(output_dir, f"{n}.csv") for s, n in SPLIT_MAP.items()}
    if not force and all(os.path.exists(p) for p in targets.values()):
        print("✅ Dữ liệu đã có sẵn, bỏ qua (dùng --force để tải lại).")
        return

    print("⏳ Đang tải UIT-VSFC từ HuggingFace...")
    dataset = load_vsfc()

    missing = [s for s in SPLIT_MAP if s not in dataset]
    if missing:
        raise KeyError(f"Thiếu split {missing}; các split hiện có: {list(dataset.keys())}")

    for split, path in targets.items():
        df = dataset[split].to_pandas()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ {os.path.basename(path)}: {len(df)} mẫu -> {path}")

    print("\n🎉 Hoàn thành tải và chuẩn bị dữ liệu UIT-VSFC!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải tập dữ liệu UIT-VSFC từ HuggingFace")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Thư mục lưu các file CSV (mặc định: data/raw)")
    parser.add_argument("--force", action="store_true", help="Tải lại dù file đã tồn tại")
    args = parser.parse_args()

    download_and_prepare_vsfc(output_dir=args.output_dir, force=args.force)