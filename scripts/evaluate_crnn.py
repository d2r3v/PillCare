import os
import glob
import numpy as np
import tensorflow as tf
from jiwer import cer, wer
from tqdm import tqdm
import cv2

# --- Add project root to path ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess import preprocess_for_ocr
from scripts.ocr_crnn import decode_prediction, CHARACTERS

# --- Config ---
MODEL_PATH = "models/crnn_epillid.h5"
DATA_DIR = "ocr_dataset_epillid"
PARTITION = "val" # or "test"

# --- Main Evaluation ---
def evaluate():
    """
    Loads the trained CRNN model and evaluates it on the validation set.
    """
    print("--- Loading Model ---")
    try:
        # Load the inference model (assuming it's saved separately)
        from scripts.ocr_crnn import load_model
        model = load_model(MODEL_PATH)
        model.compile()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("A dummy model file was created. Evaluation will not be possible.")
        return

    print(f"--- Loading Validation Data from {DATA_DIR} ---")
    label_paths = glob.glob(os.path.join(DATA_DIR, 'labels', '*.txt'))

    # Simple split for now, should match train_test_split from training
    # A more robust solution would save the split indices
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

    print("--- Running Inference ---")
    for i in tqdm(range(len(val_images))):
        # Load and preprocess image
        img = cv2.imread(val_images[i])
        processed_img = preprocess_for_ocr(img)

        # Expand dims to create a batch of 1
        processed_img = np.expand_dims(processed_img, axis=0)

        # Get prediction
        pred = model.predict(processed_img, verbose=0)
        decoded_pred = decode_prediction(pred)

        predictions.append(decoded_pred)
        ground_truth.append(val_labels[i])

    # --- Calculate Metrics ---
    print("\n--- Evaluation Results ---")
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
    evaluate()
