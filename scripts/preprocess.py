import cv2
import numpy as np

def preprocess_for_classification(img, size=(224, 224)):
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_for_ocr(img, height=32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to handle lighting variations and enhance text
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

    h, w = gray.shape
    new_w = int((w / h) * height)
    resized = cv2.resize(gray, (new_w, height))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=-1)
