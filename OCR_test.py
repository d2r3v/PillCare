import cv2
import pytesseract
import os
import glob
from matplotlib import pyplot as plt

# === CONFIGURATION ===
IMAGE_DIR = "test_images"        # Folder with your pill images
EXTENSIONS = ('*.jpg', '*.png')  # Image types to include
TESSERACT_CONFIG = r'--oem 3 --psm 7'  # Assume single text line

# === OCR FUNCTION ===
def extract_imprint(image_path, show=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert if background is black (optional)
    if gray.mean() < 127:
        gray = cv2.bitwise_not(gray)

    # Crop top half of the pill (imprint region)
    h, w = gray.shape
    top_half = gray[:h // 2, :]

    # OCR
    text = pytesseract.image_to_string(top_half, config=TESSERACT_CONFIG).strip()

    if show:
        plt.imshow(top_half, cmap="gray")
        plt.title(f"OCR: {text}")
        plt.axis('off')
        plt.show()

    return text

# === MAIN LOOP ===
if __name__ == "__main__":
    print(f"📁 Reading images from: {IMAGE_DIR}")
    image_paths = []
    for ext in EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    for path in sorted(image_paths):
        imprint = extract_imprint(path, show=True)
        print(f"[{os.path.basename(path)}] → OCR: {imprint}")
