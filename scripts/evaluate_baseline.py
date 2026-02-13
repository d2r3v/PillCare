"""
Evaluate the baseline vision-only model (MobileNetV2) on the test set.
Reconstructs the model architecture and loads weights to work with Keras 3.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

DATA_DIR = "pill_dataset_split"
MODEL_PATH = "best_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 1


def evaluate_baseline():
    # Load class indices from fusion training (same classes)
    indices_path = "models/fusion_class_indices.json"
    if os.path.exists(indices_path):
        with open(indices_path, "r") as f:
            class_indices = json.load(f)
    else:
        # Determine from directory
        test_dir = os.path.join(DATA_DIR, "test")
        classes = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
        class_indices = {c: i for i, c in enumerate(classes)}

    num_classes = len(class_indices)

    # Reconstruct model architecture (same as models/train.py)
    backbone = MobileNetV2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    print("Loading baseline weights from best_model.h5...")
    try:
        model.load_weights(MODEL_PATH)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Could not load weights: {e}")
        print("Falling back to ImageNet backbone (no fine-tuned weights).")
        backbone = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
        model = tf.keras.Sequential([
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    # Test data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print("Evaluating baseline...")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\nBaseline Test Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes

    label_names = list(class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=label_names)
    print("\nClassification Report:\n")
    print(report)

    os.makedirs("logs", exist_ok=True)
    with open("logs/baseline_evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
    print("Saved to logs/baseline_evaluation.txt")


if __name__ == "__main__":
    evaluate_baseline()
