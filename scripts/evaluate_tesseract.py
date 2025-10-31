import os
import glob
import pytesseract
from jiwer import cer, wer
from tqdm import tqdm
import cv2

# --- Config ---
DATA_DIR = "ocr_dataset_epillid"
TESSERACT_CONFIG = r'--oem 3 --psm 7'

# --- Main Evaluation ---
def evaluate_tesseract():
    """
    Evaluates Tesseract OCR performance on the validation set.
    """
    print(f"--- Loading Validation Data from {DATA_DIR} ---")
    label_paths = glob.glob(os.path.join(DATA_DIR, 'labels', '*.txt'))

    from sklearn.model_selection import train_test_split
    train_paths, val_paths = train_test_split(label_paths, test_size=0.2, random_state=42)

    val_labels = []
    val_images = []

    for path in val_paths:
        img_path = path.replace('labels', 'images').replace('.txt', '.jpg')
        if os.path.exists(img_path):
            with open(path, 'r') as f:
                val_labels.append(f.read().strip().upper())
            val_images.append(img_path)

    print(f"Found {len(val_images)} validation samples.")

    predictions = []
    ground_truth = []

    print("--- Running Tesseract OCR ---")
    for i in tqdm(range(len(val_images))):
        img = cv2.imread(val_images[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config=TESSERACT_CONFIG).strip().upper()

        predictions.append(text)
        ground_truth.append(val_labels[i])

    # --- Calculate Metrics ---
    print("\n--- Tesseract Evaluation Results ---")
    if not ground_truth:
        print("No validation data found to evaluate.")
        return

    character_error_rate = cer(ground_truth, predictions)
    word_error_rate = wer(ground_truth, predictions)

    print(f"Character Error Rate (CER): {character_error_rate:.4f}")
    print(f"Word Error Rate (WER):      {word_error_rate:.4f}")

    # --- Show some examples ---
    print("\n--- Sample Predictions ---")
    for i in range(min(10, len(predictions))):
        print(f"GT:       '{ground_truth[i]}'")
        print(f"Predicted: '{predictions[i]}'\n")

if __name__ == "__main__":
    evaluate_tesseract()
