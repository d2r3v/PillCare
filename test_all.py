import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
MODEL_PATH = "pillcare_model.tflite"
TEST_DIR = "pill_dataset_split/test"
IMG_SIZE = (224, 224)
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# === LOAD MODEL ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === CLASS LABELS ===
class_names = sorted([d.name for d in Path(TEST_DIR).iterdir() if d.is_dir()])

# === IMAGE PREPROCESSING ===
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# === BATCH INFERENCE ===
y_true = []
y_pred = []

for label in class_names:
    label_dir = Path(TEST_DIR) / label
    for img_file in label_dir.iterdir():
        if img_file.suffix.lower() not in VALID_EXTENSIONS:
            continue
        image = preprocess_image(img_file)
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred_index = np.argmax(output)

        y_true.append(label)
        y_pred.append(class_names[pred_index])

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred, labels=class_names)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("TFLite Model Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_tflite.png")
print("Saved confusion matrix as 'confusion_matrix_tflite.png'")

# === CLASSIFICATION REPORT ===
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print("Classification Report:\n")
print(report)
