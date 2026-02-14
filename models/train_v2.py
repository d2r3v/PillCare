"""
Re-train the Vision model (MobileNetV2) in Keras 3.
Saves weights in a format compatible with the fusion pipeline.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# --- Config ---
DATASET_DIR = "pill_dataset_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_FROZEN = 15
EPOCHS_FINETUNE = 15
SAVE_PATH = "models/vision_model_v2.keras"  # Keras 3 native format

np.random.seed(42)
tf.random.set_seed(42)


def main():
    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print("WARNING: No GPU detected.")

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = train_gen.num_classes
    print(f"Classes: {num_classes}  |  Train: {train_gen.samples}  |  Val: {val_gen.samples}  |  Test: {test_gen.samples}")

    # --- Build Model (Functional API for Keras 3 compatibility) ---
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="vision_input")
    backbone = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    backbone.trainable = False  # Freeze for Phase 1

    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    callbacks = [
        ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    # --- Phase 1: Train head ---
    print("\n--- Phase 1: Training head (backbone frozen) ---")
    steps_phase1 = (train_gen.samples // BATCH_SIZE) * EPOCHS_FROZEN
    lr_schedule_1 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=steps_phase1, alpha=1e-5 / 1e-3
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule_1),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    model.summary()

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=train_gen.classes
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights computed for {num_classes} classes")

    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_FROZEN, callbacks=callbacks,
                   class_weight=class_weight_dict)

    # --- Phase 2: Fine-tune last 30 layers ---
    print("\n--- Phase 2: Fine-tuning last 30 layers of MobileNetV2 ---")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    num_trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Unfrozen: {num_trainable}/{len(backbone.layers)} backbone layers")

    steps_phase2 = (train_gen.samples // BATCH_SIZE) * EPOCHS_FINETUNE
    lr_schedule_2 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5, decay_steps=steps_phase2, alpha=1e-7 / 1e-5
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule_2),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])

    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE,
                   initial_epoch=EPOCHS_FROZEN,
                   callbacks=callbacks,
                   class_weight=class_weight_dict)

    # --- Evaluate ---
    print("\n--- Evaluation ---")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    label_names = list(train_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=label_names)
    print("\n" + report)

    # Save metrics
    os.makedirs("logs", exist_ok=True)
    metrics = {
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "num_classes": num_classes,
        "class_labels": train_gen.class_indices,
    }
    with open("logs/vision_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("logs/vision_v2_evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    all_acc = h1.history["accuracy"] + h2.history["accuracy"]
    all_val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    all_loss = h1.history["loss"] + h2.history["loss"]
    all_val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    ax1.plot(all_acc, label="Train")
    ax1.plot(all_val_acc, label="Val")
    ax1.axvline(x=len(h1.history["accuracy"]) - 0.5, ls="--", c="grey", label="Unfreeze")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(all_loss, label="Train")
    ax2.plot(all_val_loss, label="Val")
    ax2.axvline(x=len(h1.history["loss"]) - 0.5, ls="--", c="grey", label="Unfreeze")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/vision_v2_training.png", dpi=150)
    print(f"Saved plot to plots/vision_v2_training.png")
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
