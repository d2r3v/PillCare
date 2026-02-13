import os
import argparse
import json
import numpy as np
import tensorflow as tf
import cv2

# Import preprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.preprocess import preprocess_for_classification, preprocess_for_ocr

# Config
FUSION_MODEL_PATH = "models/fusion_model.h5"
INDICES_PATH = "models/fusion_class_indices.json"
IMG_SIZE = (224, 224)
OCR_HEIGHT = 32

def load_fusion_model():
    if not os.path.exists(FUSION_MODEL_PATH):
        print(f"Error: Model not found at {FUSION_MODEL_PATH}")
        return None

    # Load class indices
    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Invert to int -> class
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Custom object needed?
    class CTCLayer(tf.keras.layers.Layer):
        def __init__(self, name=None, blank_index=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.blank_index = blank_index
        def call(self, inputs): return inputs[1]
        def get_config(self): return super().get_config()
        
    model = tf.keras.models.load_model(FUSION_MODEL_PATH, custom_objects={'CTCLayer': CTCLayer})
    return model, idx_to_class

def predict_fusion(model, image, idx_to_class):
    # Preprocess Vision
    vis_input = preprocess_for_classification(image, size=IMG_SIZE) # (1, 224, 224, 3)
    
    # Preprocess OCR
    ocr_img = preprocess_for_ocr(image, height=OCR_HEIGHT) # (32, W, 1)
    ocr_input = np.expand_dims(ocr_img, axis=0) # (1, 32, W, 1)
    
    # Predict
    # Inputs must be a list or dict depending on model definition.
    # We defined inputs=[vision_input, ocr_input] in build_fusion_model.
    # But usually passed as list.
    
    preds = model.predict([vis_input, ocr_input], verbose=0)
    
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]
    
    class_name = idx_to_class.get(pred_idx, "Unknown")
    return class_name, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to pill image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print("Image not found.")
        return
        
    img = cv2.imread(args.image_path)
    if img is None:
        print("Could not read image.")
        return
        
    print(f"Loading model from {FUSION_MODEL_PATH}...")
    model, idx_to_class = load_fusion_model()
    if model is None:
        return
        
    print(f"Processing {args.image_path}...")
    class_name, conf = predict_fusion(model, img, idx_to_class)
    
    print("\n" + "="*40)
    print(f"💊 Fusion Prediction: {class_name}")
    print(f"📊 Confidence: {conf:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
