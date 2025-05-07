# preprocess.py
import cv2
import os

def preprocess_image(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255.0


