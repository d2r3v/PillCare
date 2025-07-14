import tensorflow as tf
import numpy as np

crnn_model = tf.keras.models.load_model("models/crnn_ocr_model.h5")

# You need to define the character list and decoding function
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def decode_prediction(pred):
    out = tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0]
    return "".join([CHARACTERS[i] for i in out[0].numpy() if i < len(CHARACTERS)])

def predict_text(image):
    from scripts.preprocess import preprocess_for_ocr
    processed = preprocess_for_ocr(image)
    processed = np.expand_dims(processed, axis=0)
    preds = crnn_model(processed)
    return decode_prediction(preds)
