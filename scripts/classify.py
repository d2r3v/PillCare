import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.preprocess import preprocess_for_classification


model = tf.keras.models.load_model("models/classifier_model.h5")

def classify_pill(image):
    processed = preprocess_for_classification(image)
    preds = model.predict(processed)[0]
    class_idx = preds.argmax()
    confidence = preds[class_idx]
    return class_idx, confidence


