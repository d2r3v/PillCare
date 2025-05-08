import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("best_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: quantize for smaller size
tflite_model = converter.convert()

# Save the TFLite model
with open("pillcare_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as pillcare_model.tflite")
