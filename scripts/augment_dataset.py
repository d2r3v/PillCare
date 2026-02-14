"""
Offline data augmentation for the pill dataset.

Generates augmented copies of training images to:
  1. Increase total dataset size (3-5x)
  2. Balance classes (underrepresented classes get more augmentations)

Augmented images are saved directly into pill_dataset_split/train/
with an "aug_" prefix so they can be identified later.

Usage:
    python scripts/augment_dataset.py                  # default target
    python scripts/augment_dataset.py --target 200     # 200 images per class
    python scripts/augment_dataset.py --dry-run        # preview without saving
    python scripts/augment_dataset.py --clean          # remove all augmented images
"""
import os
import sys
import random
import argparse
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

DATASET_DIR = "pill_dataset_split"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
AUG_PREFIX = "aug_"

random.seed(42)
np.random.seed(42)


# ──────────────────────── Augmentation Functions (OpenCV) ────────────────────────

def random_rotation(img):
    """Rotate by a random angle between -20 and 20 degrees."""
    h, w = img.shape[:2]
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_flip(img):
    """Random horizontal and/or vertical flip."""
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # vertical
    return img


def random_brightness(img):
    """Adjust brightness randomly."""
    factor = random.uniform(0.7, 1.4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_contrast(img):
    """Adjust contrast randomly."""
    factor = random.uniform(0.7, 1.4)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_saturation(img):
    """Adjust color saturation randomly."""
    factor = random.uniform(0.6, 1.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_sharpness(img):
    """Randomly sharpen or blur the image."""
    if random.random() > 0.5:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * random.uniform(0.1, 0.3)
        kernel[1, 1] += 1 - kernel.sum()
        return cv2.filter2D(img, -1, kernel)
    else:
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


def random_crop_and_resize(img):
    """Crop 80-95% of the image and resize back to original size."""
    h, w = img.shape[:2]
    crop_frac = random.uniform(0.80, 0.95)
    new_w, new_h = int(w * crop_frac), int(h * crop_frac)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(img):
    """Add slight Gaussian noise."""
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_blur(img):
    """Apply slight Gaussian blur."""
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(img, (ksize, ksize), random.uniform(0.3, 1.5))


def random_perspective(img):
    """Slight perspective warp."""
    h, w = img.shape[:2]
    mag = random.uniform(0.02, 0.08)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.uniform(0, w * mag), random.uniform(0, h * mag)],
        [w - random.uniform(0, w * mag), random.uniform(0, h * mag)],
        [w - random.uniform(0, w * mag), h - random.uniform(0, h * mag)],
        [random.uniform(0, w * mag), h - random.uniform(0, h * mag)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


# All available transforms
ALL_TRANSFORMS = [
    random_rotation,
    random_flip,
    random_brightness,
    random_contrast,
    random_saturation,
    random_sharpness,
    random_crop_and_resize,
    add_gaussian_noise,
    random_blur,
    random_perspective,
]


def augment_image(img):
    """Apply a random combination of 3-5 transforms to an image."""
    num_transforms = random.randint(3, 5)
    transforms = random.sample(ALL_TRANSFORMS, num_transforms)
    for t in transforms:
        try:
            img = t(img)
        except Exception:
            pass  # Skip failed transforms gracefully
    return img


# ──────────────────────── Main Logic ────────────────────────

def get_class_counts():
    """Count original (non-augmented) and augmented images per class."""
    counts = {}
    for cls_name in sorted(os.listdir(TRAIN_DIR)):
        cls_dir = os.path.join(TRAIN_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        original = [f for f in files if not f.startswith(AUG_PREFIX)]
        augmented = [f for f in files if f.startswith(AUG_PREFIX)]
        counts[cls_name] = {
            "original": len(original),
            "augmented": len(augmented),
            "total": len(files),
        }
    return counts


def clean_augmented():
    """Remove all augmented images."""
    removed = 0
    for cls_name in os.listdir(TRAIN_DIR):
        cls_dir = os.path.join(TRAIN_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.startswith(AUG_PREFIX):
                os.remove(os.path.join(cls_dir, f))
                removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="Offline data augmentation")
    parser.add_argument("--target", type=int, default=150,
                        help="Target number of TOTAL images per class (default: 150)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview augmentation plan without saving")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all augmented images and exit")
    args = parser.parse_args()

    if args.clean:
        removed = clean_augmented()
        print(f"✅ Removed {removed} augmented images")
        return

    print("=" * 65)
    print("OFFLINE DATA AUGMENTATION")
    print("=" * 65)

    counts = get_class_counts()
    target = args.target

    print(f"\nTarget: {target} images per class")
    print(f"{'Class':50s}  {'Orig':>5s}  {'Aug':>5s}  {'Total':>5s}  {'Need':>5s}")
    print("-" * 75)

    plan = {}
    for cls_name in sorted(counts.keys()):
        c = counts[cls_name]
        need = max(0, target - c["total"])
        plan[cls_name] = need
        print(f"{cls_name:50s}  {c['original']:5d}  {c['augmented']:5d}  {c['total']:5d}  {need:5d}")

    total_to_generate = sum(plan.values())
    print(f"\nTotal images to generate: {total_to_generate}")

    if args.dry_run:
        print("\n⚠️  DRY RUN — no images saved. Remove --dry-run to generate.")
        return

    if total_to_generate == 0:
        print("✅ All classes already at or above target. Nothing to do.")
        return

    # Generate augmented images
    print(f"\nGenerating augmented images...")
    generated_total = 0

    for cls_name, need in sorted(plan.items()):
        if need == 0:
            continue

        cls_dir = os.path.join(TRAIN_DIR, cls_name)
        # Only use original (non-augmented) images as sources
        originals = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith(AUG_PREFIX)
        ]

        if not originals:
            print(f"  ⚠️  {cls_name}: no original images found, skipping")
            continue

        generated = 0
        attempt = 0
        max_attempts = need * 3  # safety valve

        while generated < need and attempt < max_attempts:
            # Pick a random source image
            src_name = random.choice(originals)
            src_path = os.path.join(cls_dir, src_name)

            try:
                img = cv2.imread(src_path)
                if img is None:
                    attempt += 1
                    continue
                aug_img = augment_image(img)

                # Save with unique name
                aug_name = f"{AUG_PREFIX}{generated:04d}_{src_name}"
                aug_path = os.path.join(cls_dir, aug_name)

                if not os.path.exists(aug_path):
                    cv2.imwrite(aug_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    generated += 1
                    generated_total += 1
            except Exception as e:
                pass  # Skip corrupted images

            attempt += 1

        print(f"  ✅ {cls_name:50s}  +{generated} augmented images")

    # Final summary
    print(f"\n{'=' * 65}")
    print("FINAL DATASET:")
    print(f"{'=' * 65}")
    final_counts = get_class_counts()
    grand_total = 0
    for cls_name in sorted(final_counts.keys()):
        c = final_counts[cls_name]
        grand_total += c["total"]
        print(f"  {cls_name:50s}  {c['total']:4d}  (orig={c['original']}, aug={c['augmented']})")

    print(f"\n  Grand total: {grand_total} training images")
    print(f"  Generated: {generated_total} new augmented images")
    print(f"\n✅ Done! Now retrain with: python3 models/train_v2.py")


if __name__ == "__main__":
    main()
