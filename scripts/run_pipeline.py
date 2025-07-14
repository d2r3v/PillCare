import cv2
import os
from scripts.classify import classify_pill
from scripts.ocr_crnn import predict_text

CLASS_LABELS = ["diltiazem hydrochloride", "gabapentin", "hydrocodone...", "lisinopril", "metformin hydrochloride"]

def run(image_path):
    image = cv2.imread(image_path)

    # Classification
    class_idx, confidence = classify_pill(image)
    predicted_class = CLASS_LABELS[class_idx]
    
    # OCR
    imprint_text = predict_text(image)

    # Fusion (basic)
    print(f"Predicted Class: {predicted_class} ({confidence:.2f})")
    print(f"OCR Text: {imprint_text}")

    # Optional: match imprint with known imprints in dataset
    # You can add fuzzy match or query logic here

if __name__ == "__main__":
    for file in os.listdir("data/test_images"):
        print(f"\n Processing {file}")
        run(f"data/test_images/{file}")
