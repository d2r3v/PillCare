import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# --- Paths ---
DATASET_DIR = "pill_dataset_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
FINE_TUNE_EPOCHS = 5

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --- Model ---
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- Callbacks ---
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy"),
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

# --- Initial Training ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# --- Fine-tuning ---
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(" Fine-tuning last 20 layers of MobileNetV2...")
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# --- Evaluate ---
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Test accuracy: {acc:.4f}")
print(f"✅ Test loss: {loss:.4f}")

# --- Save Evaluation Metrics ---
metrics = {
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "num_classes": int(train_generator.num_classes),
    "class_labels": {str(k): v for k, v in train_generator.class_indices.items()},
    "total_training_epochs": EPOCHS + FINE_TUNE_EPOCHS,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE
}

# Get per-class predictions for additional metrics
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate per-class accuracy
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys()), output_dict=True)
metrics["classification_report"] = report
metrics["overall_accuracy"] = float(report["accuracy"])

os.makedirs("logs", exist_ok=True)
with open("logs/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Metrics saved to logs/training_metrics.json")

# --- Plot Training ---
def plot_history(histories, labels):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    for hist, label in zip(histories, labels):
        plt.plot(hist.history['accuracy'], label=f'{label} Train')
        plt.plot(hist.history['val_accuracy'], label=f'{label} Val')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    for hist, label in zip(histories, labels):
        plt.plot(hist.history['loss'], label=f'{label} Train')
        plt.plot(hist.history['val_loss'], label=f'{label} Val')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_finetuned_plots.png", dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved to plots/training_finetuned_plots.png")
    plt.show()

plot_history([history, history_fine], ["Initial", "Fine-tune"])
