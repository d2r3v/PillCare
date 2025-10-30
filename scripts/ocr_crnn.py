import tensorflow as tf
import numpy as np

# Keep CHARACTERS in sync with training
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ;:/-"


# Minimal CTCLayer to enable model loading; in inference, it simply passes y_pred through
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        # inputs = [y_true, y_pred] during training; at inference we'll receive y_pred directly
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return inputs[1]
        return inputs
    
    def get_config(self):
        return super().get_config()


def _load_model(model_path: str = "models/crnn_epillid.h5"):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"CTCLayer": CTCLayer}
    )


def decode_prediction(pred: np.ndarray, characters: str = CHARACTERS) -> str:
    # pred: (batch, time_steps, num_classes+1)
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    seq = decoded[0].numpy()[0]
    return "".join([characters[i] for i in seq if 0 <= i < len(characters)])


def predict_text(image) -> str:
    from scripts.preprocess import preprocess_for_ocr
    processed = preprocess_for_ocr(image)  # (H, W, 1)
    processed = np.expand_dims(processed, axis=0)  # (1, H, W, 1)
    model = _load_model()
    preds = model(processed)
    return decode_prediction(preds.numpy())
