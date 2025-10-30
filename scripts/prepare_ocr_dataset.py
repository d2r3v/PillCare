"""
Prepare and validate OCR dataset for CRNN training.

Tasks:
- Ensure each label file has a corresponding image (and vice versa)
- Detect and optionally move low-contrast/blank images to a quarantine folder
- Compute character/label length distributions and dataset stats
- Save stats to logs/ocr_dataset_stats.json

Usage:
  python scripts/prepare_ocr_dataset.py [--apply] [--data_dir ocr_dataset_epillid]

Notes:
- Non-destructive by default. Use --apply to move low-quality samples to '_filtered_out/'.
"""
import os
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def is_low_contrast(img_gray: np.ndarray, std_threshold: float = 10.0) -> bool:
    """Simple low-contrast heuristic using pixel standard deviation."""
    return float(img_gray.std()) < std_threshold


def is_blank(img_gray: np.ndarray, white_ratio_threshold: float = 0.98) -> bool:
    """Detect near-blank images (almost all-white after binarization)."""
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = (th == 255).mean()
    return white_ratio > white_ratio_threshold


def scan_dataset(data_dir: str = "ocr_dataset_epillid"):
    images_dir = Path(data_dir) / "images"
    labels_dir = Path(data_dir) / "labels"
    assert images_dir.exists(), f"Images dir not found: {images_dir}"
    assert labels_dir.exists(), f"Labels dir not found: {labels_dir}"

    image_paths = sorted([p for p in images_dir.glob("*.jpg")])
    label_paths = sorted([p for p in labels_dir.glob("*.txt")])

    # Map by stem
    images_by_stem = {p.stem: p for p in image_paths}
    labels_by_stem = {p.stem: p for p in label_paths}

    stems_images = set(images_by_stem.keys())
    stems_labels = set(labels_by_stem.keys())

    missing_labels = sorted(stems_images - stems_labels)
    missing_images = sorted(stems_labels - stems_images)

    pairs = []
    for stem in sorted(stems_images & stems_labels):
        pairs.append((images_by_stem[stem], labels_by_stem[stem]))

    return pairs, missing_labels, missing_images


def compute_stats(pairs, characters: str):
    char_freq = Counter()
    label_len_dist = Counter()
    num_blank_or_low = 0

    for img_path, lbl_path in pairs:
        text = Path(lbl_path).read_text(encoding="utf-8", errors="ignore").strip().upper()
        char_freq.update([c for c in text if c in characters])
        label_len_dist[len(text)] += 1

    total_pairs = len(pairs)
    total_chars = sum(char_freq.values())
    avg_label_len = (total_chars / total_pairs) if total_pairs else 0.0

    return {
        "total_pairs": total_pairs,
        "total_chars": total_chars,
        "avg_label_len": avg_label_len,
        "label_len_dist": dict(sorted(label_len_dist.items())),
        "char_freq": dict(sorted(char_freq.items()))
    }


def filter_low_quality(pairs, apply=False, quarantine_dir: Path | None = None, std_th=10.0):
    low_quality = []
    kept = []

    if quarantine_dir is None:
        quarantine_dir = Path("ocr_dataset_epillid/_filtered_out")
    (quarantine_dir / "images").mkdir(parents=True, exist_ok=True)
    (quarantine_dir / "labels").mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            low_quality.append((img_path, lbl_path, "unreadable"))
            continue
        low_contrast = is_low_contrast(img, std_threshold=std_th)
        blank = is_blank(img)
        if low_contrast or blank:
            low_quality.append((img_path, lbl_path, "low_contrast" if low_contrast else "blank"))
            if apply:
                # Move to quarantine
                q_img = quarantine_dir / "images" / img_path.name
                q_lbl = quarantine_dir / "labels" / lbl_path.name
                img_path.replace(q_img)
                lbl_path.replace(q_lbl)
        else:
            kept.append((img_path, lbl_path))

    return kept, low_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ocr_dataset_epillid")
    parser.add_argument("--apply", action="store_true", help="Move low-quality samples to _filtered_out/")
    parser.add_argument("--std_threshold", type=float, default=10.0, help="Stddev threshold for low contrast")
    parser.add_argument("--characters", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ;:/-", help="Allowed characters")
    args = parser.parse_args()

    print("🔎 Scanning dataset...")
    pairs, missing_labels, missing_images = scan_dataset(args.data_dir)

    print(f"✅ Found {len(pairs)} image/label pairs")
    if missing_labels:
        print(f"⚠️  {len(missing_labels)} images missing labels (samples: {missing_labels[:5]})")
    if missing_images:
        print(f"⚠️  {len(missing_images)} labels missing images (samples: {missing_images[:5]})")

    print("🧹 Checking for low-contrast / blank images...")
    kept, low_quality = filter_low_quality(pairs, apply=args.apply, std_th=args.std_threshold)
    print(f"   Kept: {len(kept)} | Filtered: {len(low_quality)}")

    print("📊 Computing stats...")
    stats = compute_stats(kept, args.characters)
    stats.update({
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "low_quality_count": len(low_quality),
        "low_quality_examples": [
            {"image": str(p[0].name), "label": str(p[1].name), "reason": p[2]}
            for p in low_quality[:10]
        ]
    })

    os.makedirs("logs", exist_ok=True)
    out_path = Path("logs/ocr_dataset_stats.json")
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"✅ Saved stats to {out_path}")


if __name__ == "__main__":
    main()
