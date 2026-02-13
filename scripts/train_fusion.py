"""
Fusion Model: MobileNetV2 (Vision) + CRNN Conv Layers (OCR) -> Classification
"""
import os
import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2

np.random.seed(42)
tf.random.set_seed(42)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.preprocess import preprocess_for_classification, preprocess_for_ocr

# --- Config ---
DATA_DIR = "pill_dataset_split"
OCR_MODEL_PATH = "models/crnn_epillid.h5"
FUSION_MODEL_PATH = "models/fusion_model.h5"

IMG_SIZE = (224, 224)
OCR_HEIGHT = 32
OCR_WIDTH = 128  # Fixed width for OCR input
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
LEARNING_RATE = 1e-4

# --- CTC stub for loading the CRNN .h5 ---
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, blank_index=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.blank_index = blank_index
    def call(self, inputs):
        return inputs[1]
    def get_config(self):
        cfg = super().get_config()
        cfg["blank_index"] = self.blank_index
        return cfg

# --- Custom Data Generator ---
class FusionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, shuffle=True, class_indices=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_paths = []
        self.labels = []

        classes = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.class_indices = class_indices or {c: i for i, c in enumerate(classes)}
        self.num_classes = len(self.class_indices)

        for cls in classes:
            if cls not in self.class_indices:
                continue
            cls_dir = os.path.join(data_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cls_dir, f))
                    self.labels.append(self.class_indices[cls])

        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        paths = [self.image_paths[k] for k in idxs]
        labs  = [self.labels[k] for k in idxs]
        return self._generate(paths, labs)

    def _generate(self, paths, labs):
        vis_batch = []
        ocr_batch = np.zeros((len(paths), OCR_HEIGHT, OCR_WIDTH, 1), dtype=np.float32)

        for i, p in enumerate(paths):
            img = cv2.imread(p)
            # Vision
            vis_batch.append(preprocess_for_classification(img, size=IMG_SIZE)[0])
            # OCR – resize to fixed (H, W) then normalise
            ocr_img = preprocess_for_ocr(img, height=OCR_HEIGHT)   # (32, ?, 1)
            w = min(ocr_img.shape[1], OCR_WIDTH)
            ocr_batch[i, :, :w, :] = ocr_img[:, :w, :]

        return (
            {"vision_input": np.array(vis_batch), "ocr_input": ocr_batch},
            tf.keras.utils.to_categorical(labs, self.num_classes),
        )


# --- Build the OCR feature extractor from conv layers only ---
def _build_ocr_branch(ocr_model_path):
    """
    Loads the trained CRNN and extracts features from the LAST CONV block
    (dropout_3, index 13) – everything BEFORE the Reshape layer that is
    incompatible with Keras 3.  A GlobalAveragePooling2D collapses the
    spatial dims to a fixed-length vector.
    """
    full_crnn = tf.keras.models.load_model(
        ocr_model_path, custom_objects={"CTCLayer": CTCLayer}
    )

    # Layer 13 = dropout_3 (last layer before Reshape)
    last_conv_layer = full_crnn.layers[13]  # dropout_3
    print(f"  OCR cutoff layer: {last_conv_layer.name}")

    # image_input is at index 0
    img_input = full_crnn.input[0] if isinstance(full_crnn.input, list) else full_crnn.input

    conv_extractor = Model(inputs=img_input, outputs=last_conv_layer.output, name="crnn_conv")

    # Build new branch with fixed input size
    ocr_input = Input(shape=(OCR_HEIGHT, OCR_WIDTH, 1), name="ocr_input")
    x = conv_extractor(ocr_input)          # (batch, H', W', 256)
    ocr_features = GlobalAveragePooling2D()(x)  # (batch, 256)

    return ocr_input, ocr_features, conv_extractor


# --- Build full fusion model ---
def build_fusion_model(ocr_model_path, num_classes):
    from tensorflow.keras.applications import MobileNetV2

    # ---- Vision branch ----
    print("Initializing Vision branch (MobileNetV2 + ImageNet)...")
    vision_input = Input(shape=IMG_SIZE + (3,), name="vision_input")
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=IMG_SIZE + (3,))
    x = backbone(vision_input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    vision_features = Dropout(0.3)(x)

    # ---- OCR branch ----
    print("Initializing OCR branch (CRNN conv layers)...")
    ocr_input, ocr_features, _ = _build_ocr_branch(ocr_model_path)

    # ---- Fusion head ----
    combined = Concatenate()([vision_features, ocr_features])
    x = Dense(256, activation="relu")(combined)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax", name="fusion_output")(x)

    model = Model(inputs=[vision_input, ocr_input], outputs=output)
    return model


# --- Training ---
def train():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("fusion_train")

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU detected: {gpus}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        logger.warning("No GPU detected – training on CPU (will be slow).")

    # Data generators
    logger.info("Setting up data generators...")
    train_gen = FusionDataGenerator(os.path.join(DATA_DIR, "train"), BATCH_SIZE)
    val_gen   = FusionDataGenerator(os.path.join(DATA_DIR, "val"),   BATCH_SIZE,
                                    shuffle=False, class_indices=train_gen.class_indices)
    logger.info(f"Classes: {train_gen.num_classes}  |  "
                f"Train batches: {len(train_gen)}  |  Val batches: {len(val_gen)}")

    # Build model
    logger.info("Building Fusion Model...")
    model = build_fusion_model(OCR_MODEL_PATH, train_gen.num_classes)

    # Phase 1 – freeze backbones, train head only
    for layer in model.layers:
        layer.trainable = False
    # Unfreeze only the fusion head layers
    model.get_layer("fusion_output").trainable = True
    for layer in model.layers:
        if "dense" in layer.name or "dropout" in layer.name or "concatenate" in layer.name:
            layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(FUSION_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    logger.info("--- Phase 1: Training Fusion Head (backbones frozen) ---")
    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_HEAD, callbacks=callbacks)

    # Phase 2 – unfreeze everything, low LR
    logger.info("--- Phase 2: Fine-tuning entire model ---")
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_FINE, callbacks=callbacks)

    logger.info("Training complete.")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    acc     = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.axvline(x=len(h1.history["accuracy"]) - 0.5, ls="--", c="grey", label="Unfreeze")
    plt.title("Fusion Model Training")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fusion_history.png", dpi=150)
    logger.info("Saved plot to plots/fusion_history.png")

    # Save class indices
    with open("models/fusion_class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)


if __name__ == "__main__":
    train()
