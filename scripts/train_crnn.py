import os
import glob
import json
import argparse
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional,
    GRU, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.preprocess import preprocess_for_ocr

# --- Config ---
DATA_DIR = "ocr_dataset_epillid"
MODEL_PATH = "models/crnn_epillid.h5"
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ;:/-"
IMG_HEIGHT = 32
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

char_to_num = {char: i for i, char in enumerate(CHARACTERS)}
num_to_char = {i: char for i, char in enumerate(CHARACTERS)}

# --- Data Loader ---
def load_and_split_data(data_dir):
    print("Searching for images and labels...")
    image_paths = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
    labels = []
    valid_paths = []
    downsample_factor = 2 # Minimal downsampling

    for img_path in image_paths:
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                text = f.read().strip().upper()
                if all(c in CHARACTERS for c in text) and text:
                    # Check width AFTER preprocessing (height-normalized) to ensure enough time steps
                    img = cv2.imread(img_path)
                    proc = preprocess_for_ocr(img, height=IMG_HEIGHT)
                    pre_w = proc.shape[1]
                    if pre_w > (len(text) + 2):  # small margin
                        labels.append(text)
                        valid_paths.append(img_path)

    print(f"Found {len(valid_paths)} valid samples.")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        valid_paths, labels, test_size=0.2, random_state=42
    )
    return train_paths, train_labels, val_paths, val_labels

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Precompute global max dimensions to ensure consistent batch shapes
        self._precompute_dims()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        return self.__data_generation(batch_paths, batch_labels)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths, batch_labels):
        images = []
        processed_labels = []
        max_width = self.max_width
        max_label_len = self.max_label_len

        for path, label in zip(batch_paths, batch_labels):
            img = cv2.imread(path)
            img = preprocess_for_ocr(img, height=IMG_HEIGHT)
            images.append(img)
            max_width = max(max_width, img.shape[1])
            
            # Encode and pad labels to max_label_len
            encoded_label = self.char_to_num(list(label))
            processed_labels.append(encoded_label)

        # Pad images
        padded_images = np.zeros((len(batch_paths), IMG_HEIGHT, max_width, 1), dtype=np.float32)
        for i, img in enumerate(images):
            padded_images[i, :, :img.shape[1], :] = img

        # Pad labels to max length in batch
        padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
            processed_labels,
            maxlen=max_label_len,
            padding='post',
            value=len(CHARACTERS)
        )

        # For training, feed a dict mapping input layer names to arrays
        inputs = {
            'image_input': padded_images,
            'label_input': padded_labels
        }
        outputs = np.zeros(len(batch_labels))
        return inputs, outputs

    def char_to_num(self, label):
        return [char_to_num[c] for c in label]

    def _precompute_dims(self):
        # Compute the maximum label length and image width after preprocessing
        self.max_label_len = max((len(l) for l in self.labels), default=1)
        max_w = 1
        for p in self.image_paths:
            img = cv2.imread(p)
            proc = preprocess_for_ocr(img, height=IMG_HEIGHT)
            max_w = max(max_w, proc.shape[1])
        self.max_width = int(max_w)

# --- Model Definition ---
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, blank_index: int | None = None):
        super().__init__(name=name)
        self.avg_input_len_metric = tf.keras.metrics.Mean(name='avg_input_len')
        self.avg_label_len_metric = tf.keras.metrics.Mean(name='avg_label_len')
        self.blank_index = -1 if blank_index is None else int(blank_index)

    def call(self, inputs):
        y_true, y_pred = inputs
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        
        # Calculate true label length per sample (count non-blank tokens)
        if self.blank_index >= 0:
            non_blank = tf.cast(tf.not_equal(y_true, self.blank_index), tf.int64)
            label_length_vec = tf.reduce_sum(non_blank, axis=1)
        else:
            # Fallback: assume all timesteps are label tokens
            label_length_vec = tf.cast(tf.shape(y_true)[1]) * tf.ones((batch_len,), dtype=tf.int64)
        
        # Get the sequence length from the input (time steps)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        # Tile input_length for each sample and expand dims
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = tf.expand_dims(label_length_vec, axis=1)
        
        # Calculate CTC loss
        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        
        # Update metrics
        self.avg_input_len_metric.update_state(tf.reduce_mean(input_length))
        self.avg_label_len_metric.update_state(tf.reduce_mean(label_length))
        
        return y_pred
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def compute_output_shape(self, input_shape):
        # Return the shape of y_pred (the second input)
        return input_shape[1]

def build_crnn(input_shape, num_classes):
    # Input layers
    image_input = Input(name='image_input', shape=input_shape, dtype='float32')
    label_input = Input(name='label_input', shape=[None], dtype='int32')

    # First conv block - no downsampling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Second conv block - no downsampling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Third conv block with minimal downsampling (reduce height, keep width)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)  # Reduce height only, preserve width/time steps
    x = Dropout(0.3)(x)
    
    # Fourth conv block - no downsampling
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    conv_shape = x.shape
    x = tf.keras.layers.Reshape(target_shape=(conv_shape[2], conv_shape[1] * conv_shape[3]))(x)
    # RNN layers with more units to capture longer sequences
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.3))(x)
    
    # Output layer
    y_pred = Dense(num_classes + 1, activation='softmax', name='output')(x)
    
    # Add CTC layer for calculating loss
    output = CTCLayer(name='ctc_loss', blank_index=num_classes)([label_input, y_pred])
    
    # Define the model
    model = tf.keras.Model(
        inputs=[image_input, label_input],
        outputs=output
    )
    
    # Compile the model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)
    
    # Print model summary
    model.summary()
    
    return model

# --- Main Training Function ---
def train():
    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional limit for number of training samples (for quick runs)")
    parser.add_argument("--model_path", default=MODEL_PATH)
    args = parser.parse_args()

    # Logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("logs/ocr_training.log"),
            logging.StreamHandler()
        ],
    )
    logger = logging.getLogger("ocr_train")
    # Add warning filter to help debug any remaining issues
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Set memory growth to prevent OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
    
    logger.info("--- Stage 1: Loading and Splitting Data ---")
    train_paths, train_labels, val_paths, val_labels = load_and_split_data(args.data_dir)

    # Optional limit for quick sanity runs
    if args.max_samples and len(train_paths) > args.max_samples:
        train_paths = train_paths[:args.max_samples]
        train_labels = train_labels[:args.max_samples]

    logger.info(f"Data split: {len(train_paths)} training, {len(val_paths)} validation.")

    if not train_paths or not val_paths:
        logger.error("No data to train on. Please check the dataset.")
        return

    logger.info("--- Stage 2: Creating Data Generators ---")
    train_generator = DataGenerator(train_paths, train_labels, args.batch_size)
    val_generator = DataGenerator(val_paths, val_labels, args.batch_size, shuffle=False)
    logger.info("Data generators created.")

    logger.info("--- Stage 3: Building and Compiling Model ---")
    model = build_crnn(input_shape=(IMG_HEIGHT, None, 1), num_classes=len(CHARACTERS))
    model.compile(optimizer=Adam(learning_rate=args.learning_rate))
    logger.info("Model built and compiled.")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(args.model_path, monitor='val_loss', save_best_only=True)
    ]

    logger.info("--- Stage 4: Starting Training ---")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )
    logger.info("--- Training Complete ---")

    # Save split for reproducibility
    split = {
        "train": [{"image": os.path.basename(p), "label": l} for p, l in zip(train_paths, train_labels)],
        "val":   [{"image": os.path.basename(p), "label": l} for p, l in zip(val_paths, val_labels)]
    }
    with open("logs/ocr_split.json", "w") as f:
        json.dump(split, f, indent=2)
    logger.info("Saved split to logs/ocr_split.json")

    # Plot history
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title('CRNN Training History')
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/crnn_history.png", dpi=300)
    logger.info("Saved training plot to plots/crnn_history.png")

if __name__ == "__main__":
    train()
