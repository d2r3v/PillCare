from PIL import Image
import os

INPUT_DIR = "pill_dataset"
OUTPUT_DIR = "pill_dataset_resized"
IMAGE_SIZE = (224, 224)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

count = 0
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    out_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class_path, exist_ok=True)

    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        out_path = os.path.join(out_class_path, filename)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(out_path)
            count += 1
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")

print(f"\n Resized and saved {count} images to: {OUTPUT_DIR}")
