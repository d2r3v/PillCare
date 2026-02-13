"""
Expand the pill dataset by mining ePillID segmented images.

Uses openFDA drug/label API to find NDC product codes for each drug,
then matches those codes to filenames in segmented_nih_pills_224/.

Usage:
    python scripts/expand_dataset.py            # expand dataset
    python scripts/expand_dataset.py --dry-run  # preview without copying
"""
import os
import sys
import json
import shutil
import random
import argparse
import urllib.request
import urllib.parse
from collections import defaultdict

# ──────────────────────── Config ────────────────────────
SEGMENTED_DIR = "data/ePillID_data/classification_data/segmented_nih_pills_224"
DATASET_DIR   = "pill_dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Target drugs — ordered by expected image count in ePillID
TARGET_DRUGS = [
    # Existing 5
    "diltiazem hydrochloride",
    "gabapentin",
    "hydrocodone bitartrate and acetaminophen",
    "lisinopril",
    "metformin hydrochloride",
    # Best new classes from ePillID (20+ images each)
    "simvastatin",                   # ~72 images
    "warfarin sodium",               # ~46 images
    "prednisone",                    # ~28 images
    "hydrochlorothiazide",           # ~28 images
    "levothyroxine sodium",          # ~28 images
    "metoprolol tartrate",           # ~26 images
    "amlodipine besylate",           # ~24 images
    "losartan potassium",            # ~24 images
    "pantoprazole sodium",           # ~22 images
    "carvedilol",                    # ~22 images
    "amoxicillin",                   # ~18 images
]

random.seed(42)


def query_openfda_labels(drug_name: str, limit: int = 100) -> set:
    """Query openFDA drug/label API for NDC product codes."""
    query = urllib.parse.quote(f'openfda.generic_name:"{drug_name}"')
    url = f"https://api.fda.gov/drug/label.json?search={query}&limit={limit}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PillCare/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    API error: {e}")
        return set()

    product_codes = set()
    for result in data.get("results", []):
        openfda = result.get("openfda", {})
        for ndc in openfda.get("product_ndc", []):
            # NDC format varies: "0093-7248" or "71335-1093" etc.
            # Normalize to match segmented filenames (labeler-product)
            parts = ndc.split("-")
            if len(parts) >= 2:
                # Pad labeler code to 4 or 5 digits as needed
                labeler = parts[0].zfill(4)
                product = parts[1].zfill(4)
                product_codes.add(f"{labeler}-{product}")
                # Also try without zero-padding
                product_codes.add(ndc.rsplit("-", 1)[0] if len(parts) > 2 else ndc)

    return product_codes


def find_images_for_codes(ndc_codes: set, seg_dir: str) -> list:
    """Find segmented images matching any of the NDC codes."""
    matched = []
    all_files = os.listdir(seg_dir)

    for ndc in ndc_codes:
        prefix = ndc + "_"
        for fname in all_files:
            if fname.startswith(prefix) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                matched.append(os.path.join(seg_dir, fname))

    return sorted(set(matched))


def get_existing_basenames(dataset_dir: str, drug_name: str) -> set:
    """Get basenames of images already in the dataset for this drug."""
    existing = set()
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split, drug_name)
        if os.path.isdir(split_dir):
            existing.update(os.listdir(split_dir))
    return existing


def split_and_copy(images: list, drug_name: str, dataset_dir: str, dry_run: bool):
    """Split images into train/val/test and copy them."""
    random.shuffle(images)
    n = len(images)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        dest_dir = os.path.join(dataset_dir, split_name, drug_name)
        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
        for src in split_images:
            dest = os.path.join(dest_dir, os.path.basename(src))
            if not os.path.exists(dest) and not dry_run:
                shutil.copy2(src, dest)

    return {s: len(v) for s, v in splits.items()}


def count_dataset(dataset_dir: str):
    """Count images per class per split."""
    counts = defaultdict(lambda: defaultdict(int))
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                counts[cls][split] = len([
                    f for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drugs", nargs="+", default=None)
    args = parser.parse_args()

    drugs = args.drugs or TARGET_DRUGS

    # Pre-index segmented directory for fast lookup
    print("Indexing segmented images directory...")
    seg_files = os.listdir(SEGMENTED_DIR)
    # Extract unique NDC prefixes from filenames
    seg_prefixes = defaultdict(list)
    for f in seg_files:
        # Format: "0093-7248_0_0.jpg" -> prefix "0093-7248"
        parts = f.split("_", 1)
        if len(parts) >= 1:
            seg_prefixes[parts[0]].append(f)
    print(f"  Found {len(seg_files)} files, {len(seg_prefixes)} unique NDC prefixes\n")

    # Show current state
    print("=" * 70)
    print("BEFORE:")
    print("=" * 70)
    before = count_dataset(DATASET_DIR)
    for cls in sorted(before):
        c = before[cls]
        t = sum(c.values())
        print(f"  {cls:50s}  {t:4d} (train={c['train']}, val={c['val']}, test={c['test']})")

    print(f"\n{'=' * 70}")
    print(f"Expanding for {len(drugs)} drugs...")
    print(f"{'=' * 70}")

    for drug in drugs:
        print(f"\n🔍 {drug}")

        # Query openFDA
        print("  Querying openFDA...")
        ndc_codes = query_openfda_labels(drug)
        print(f"  Got {len(ndc_codes)} NDC codes")

        if not ndc_codes:
            print("  ⚠️  No NDC codes — skipping")
            continue

        # Match against segmented files
        matched_files = []
        for ndc in ndc_codes:
            if ndc in seg_prefixes:
                for fname in seg_prefixes[ndc]:
                    matched_files.append(os.path.join(SEGMENTED_DIR, fname))

        matched_files = sorted(set(matched_files))
        print(f"  Matched {len(matched_files)} segmented images")

        if not matched_files:
            # Show what NDC codes we tried
            sample = list(ndc_codes)[:5]
            print(f"  Tried codes like: {sample}")
            print("  No matches in segmented directory — skipping")
            continue

        # Filter out existing
        existing = get_existing_basenames(DATASET_DIR, drug)
        new_files = [f for f in matched_files if os.path.basename(f) not in existing]
        print(f"  New images: {len(new_files)} (already have {len(existing)})")

        if new_files:
            counts = split_and_copy(new_files, drug, DATASET_DIR, args.dry_run)
            verb = "Would add" if args.dry_run else "Added"
            print(f"  ✅ {verb}: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("AFTER:")
    print(f"{'=' * 70}")
    after = count_dataset(DATASET_DIR)
    total = 0
    for cls in sorted(after):
        c = after[cls]
        t = sum(c.values())
        total += t
        was = sum(before.get(cls, {}).values())
        delta = f" (+{t - was})" if t > was else " (new!)" if cls not in before else ""
        print(f"  {cls:50s}  {t:4d}{delta}")

    print(f"\n  Total: {total} images across {len(after)} classes")

    # Update label map
    if not args.dry_run:
        label_map = {str(i): cls for i, cls in enumerate(sorted(after.keys()))}
        with open("data/label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"  ✅ Updated data/label_map.json")

    if args.dry_run:
        print("\n⚠️  DRY RUN — remove --dry-run to copy files")


if __name__ == "__main__":
    main()
