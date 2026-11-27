import os
import json
import random
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

CSV_PATH = "inscriptions_dataset.csv"
AUG_DIR = "augmented_images"
os.makedirs(AUG_DIR, exist_ok=True)


def simulate_real_photo(img: Image.Image) -> Image.Image:
    """Transforms a clean rubbing into a simulated field photo (applied to TRAIN only)."""
    img = img.convert("RGB")
    angle = random.uniform(-6, 6)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(120, 115, 108))

    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * random.uniform(0.85, 1.10))))
    g = g.point(lambda i: min(255, int(i * random.uniform(0.85, 1.05))))
    b = b.point(lambda i: min(255, int(i * random.uniform(0.75, 0.98))))
    img = Image.merge("RGB", (r, g, b))

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.60, 1.30))

    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.1)))
    return img


def make_record(row, img_path: str, is_augmented: bool) -> dict:
    target_payload = {
        "transliteration": str(row["Transliteration"]).strip(),
        "translation": str(row["Translation"]).strip(),
        "footnotes": str(row["Footnotes"]).strip(),
        "period": str(row["Period"]).strip(),
        "language": str(row["Language"]).strip(),
        "sinhala_prakrit": str(row["Sinhala Prakrit"]).strip(),
        "script": str(row["Script"]).strip(),
        "likely_site": str(row["Site"]).strip(),
        "reference_publication": str(row["Publication"]).strip(),
    }
    return {
        "inscriptionNo": str(row["inscriptionNo"]),
        "image_path": img_path,
        "is_augmented": is_augmented,
        "target_json": json.dumps(target_payload, ensure_ascii=False),
    }


def save_jsonl(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_80_10_10_splits(seed: int = 42):
    df = pd.read_csv(CSV_PATH).fillna("")
    valid_rows = [
        row for _, row in df.iterrows()
        if str(row["locallink"]).strip() and os.path.exists(str(row["locallink"]).strip())
    ]

    random.seed(seed)
    random.shuffle(valid_rows)

    total = len(valid_rows)
    train_end = int(total * 0.80)
    val_end = int(total * 0.90)

    train_rows = valid_rows[:train_end]
    val_rows = valid_rows[train_end:val_end]
    test_rows = valid_rows[val_end:]

    # 1. Build TRAIN set (real images + 2 synthetic augmentations each)
    train_records = []
    for row in train_rows:
        img_path = str(row["locallink"]).strip()
        train_records.append(make_record(row, img_path, is_augmented=False))
        try:
            orig_img = Image.open(img_path)
            for aug_idx in range(2):
                aug_path = os.path.join(AUG_DIR, f"{row['inscriptionNo']}_aug{aug_idx}.jpg")
                simulate_real_photo(orig_img).save(aug_path, quality=88)
                train_records.append(make_record(row, aug_path, is_augmented=True))
        except Exception as e:
            print(f"Skipping augmentation for {img_path}: {e}")

    random.shuffle(train_records)

    # 2. Build VAL & TEST sets (100% real, un-augmented images only)
    val_records = [make_record(row, str(row["locallink"]).strip(), is_augmented=False) for row in val_rows]
    test_records = [make_record(row, str(row["locallink"]).strip(), is_augmented=False) for row in test_rows]

    save_jsonl(train_records, "train.jsonl")
    save_jsonl(val_records, "val.jsonl")
    save_jsonl(test_records, "test.jsonl")

    print(f"Total unique real inscriptions: {total}")
    print(f"  Train split (80%): {len(train_rows)} real -> {len(train_records)} total samples (with aug)")
    print(f"  Val split   (10%): {len(val_records)} real-only samples")
    print(f"  Test split  (10%): {len(test_records)} real-only samples (held out for final evaluation)")


if __name__ == "__main__":
    build_80_10_10_splits()