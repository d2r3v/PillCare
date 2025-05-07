import os
import shutil
import random

SRC_DIR = "pill_dataset_resized"
DEST_DIR = "pill_dataset_split"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

random.seed(SEED)
os.makedirs(DEST_DIR, exist_ok=True)

# Check classes
classes = os.listdir(SRC_DIR)
print(f"[INFO] Found classes: {classes}")

# Make subdirectories
for split in SPLIT_RATIOS:
    for class_name in classes:
        os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

# Go class by class
for class_name in classes:
    class_path = os.path.join(SRC_DIR, class_name)
    images = os.listdir(class_path)
    print(f"[INFO] Processing {class_name}: {len(images)} images")

    if not images:
        continue

    random.shuffle(images)
    total = len(images)
    n_train = int(total * SPLIT_RATIOS["train"])
    n_val = int(total * SPLIT_RATIOS["val"])
    n_test = total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(DEST_DIR, split, class_name, file)
            shutil.copy(src, dst)

print("\n Dataset split complete!")
print(f" Output: {DEST_DIR}")
