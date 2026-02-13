"""
Evaluate the trained Fusion Model on the test set.
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import cv2

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train_fusion import FusionDataGenerator, CTCLayer

# Config
DATA_DIR = "pill_dataset_split"
MODEL_PATH = "models/fusion_model.h5"
BATCH_SIZE = 1  # Small batch to handle tiny test sets


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run train_fusion.py first.")
        return

    indices_path = "models/fusion_class_indices.json"
    if not os.path.exists(indices_path):
        print("Error: Class indices not found. Run training first.")
        return

    with open(indices_path, "r") as f:
        class_indices = json.load(f)

    print("Loading Fusion Model...")
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"CTCLayer": CTCLayer}
    )

    print("Preparing Test Data...")
    test_gen = FusionDataGenerator(
        os.path.join(DATA_DIR, "test"),
        BATCH_SIZE,
        shuffle=False,
        class_indices=class_indices,
    )

    print("Evaluating...")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\nFusion Test Accuracy: {acc:.4f}")

    # Predictions
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # True labels (shuffle=False so order matches)
    num_samples = len(test_gen) * BATCH_SIZE
    y_true = np.array(test_gen.labels[:num_samples])

    # Ensure lengths match (predict may give exactly num_samples)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    report = classification_report(
        y_true, y_pred, target_names=list(class_indices.keys())
    )
    print("\nClassification Report:\n")
    print(report)

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/fusion_evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
    print("Saved report to logs/fusion_evaluation.txt")


if __name__ == "__main__":
    evaluate()
