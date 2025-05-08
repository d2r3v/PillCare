import pandas as pd
import os
import shutil

# --- CONFIG ---
CSV_PATH = "Pillbox.csv"
IMG_DIR = "pillbox_images"
OUT_DIR = "pill_dataset"
TOP_N_CLASSES = 5

# --- Load and clean ---
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df[(df["has_image"] == True) & df["medicine_name"].notna() & df["ndc9"].notna()]

df["class"] = df["medicine_name"].str.strip().str.lower()
df["filename"] = df["ndc9"].astype(str).str.strip() + ".jpg"

# Match to available files
available_images = set(os.listdir(IMG_DIR))
df = df[df["filename"].isin(available_images)]

print(f"[INFO] Matched {len(df)} metadata rows to actual images.")

# Pick top N classes
top_classes = df["class"].value_counts().head(TOP_N_CLASSES).index.tolist()
df = df[df["class"].isin(top_classes)]

print("\n[TOP CLASSES]")
print(df["class"].value_counts())

# --- Copy images into class folders ---
copied = 0
for _, row in df.iterrows():
    class_name = row["class"]
    filename = row["filename"]
    src = os.path.join(IMG_DIR, filename)
    dst_dir = os.path.join(OUT_DIR, class_name)
    dst = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, dst)
    copied += 1

print(f"\n Copied {copied} images.")
print(f" Dataset ready at: {OUT_DIR}")
