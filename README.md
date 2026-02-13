# PillCare — Intelligent Pill Identifier with Visual and Text Recognition

**PillCare** is a deep learning-powered application designed to identify pills using both visual appearance (shape, color) and text imprints. It's built for accessibility and safety, especially for elderly users or those managing multiple medications.

## Features

### Visual Recognition
- **Pill recognition** from camera input using MobileNetV2
- Transfer learning with fine-tuning
- Predicts among multiple medication classes
- Converted to **TensorFlow Lite** for mobile/edge deployment

### Text Recognition (OCR)
- **CRNN (Convolutional Recurrent Neural Network)** for pill imprint recognition
- Handles variable-length text sequences
- Preprocessing pipeline for pill images
- CTC (Connectionist Temporal Classification) loss for sequence learning

### Fusion Model (Vision + OCR)
- **Late Fusion** architecture combining MobileNetV2 visual features with CRNN text features
- Two-phase training: frozen-backbone head training, then full fine-tuning
- Extracts learned convolutional features from the CRNN (before RNN layers)
- Achieves **80% test accuracy** with only ImageNet initialization

## Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | TensorFlow / Keras |
| Model Architectures | MobileNetV2 (visual), CRNN (text), Fusion (combined) |
| Image Preprocessing | OpenCV, NumPy |
| Text Processing | TensorFlow Text, Regular Expressions |
| Data Augmentation | TensorFlow Image |
| Model Deployment | TensorFlow Lite |

## Dataset

### Visual Recognition Dataset
- Images extracted from the [Pillbox dataset](https://www.fda.gov/drugs/pillbox)
- Preprocessed to 224x224 resolution
- Multiple classes of medications

### OCR Dataset
- Processed pill images with corresponding text imprints
- Character set includes alphanumeric characters and common symbols
- Preprocessed using adaptive thresholding and resizing while maintaining aspect ratio
- Located in `ocr_dataset_epillid/` directory

## Model Architectures

### CRNN OCR Pipeline

#### Data Preparation
1. **Image Preprocessing** (`preprocess.py`)
   - Converts images to grayscale
   - Applies adaptive thresholding for better text visibility
   - Resizes images to fixed height while maintaining aspect ratio
   - Normalizes pixel values to [0, 1] range

2. **Data Generation** (`create_ocr_dataset.py`)
   - Processes raw images and labels
   - Generates character-to-index mappings
   - Creates train/validation/test splits
   - Handles variable-length sequences with padding

#### Architecture
```
Input → Conv2D → BatchNorm → ReLU → MaxPool2D → Dropout →
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool2D → Dropout →
Conv2D → BatchNorm → ReLU → Dropout → Reshape →
Bidirectional(GRU) → BatchNorm → Dropout →
Bidirectional(GRU) → Dense → Softmax → CTCLoss
```

### Fusion Model

#### Architecture
```
Vision Branch:  Input(224x224x3) → MobileNetV2 → GAP → Dense(128) → Dropout(0.3)
OCR Branch:     Input(32x128x1)  → CRNN Conv Layers → GAP → (256-dim features)
                                    ↓
Fusion:         Concatenate(vision_features, ocr_features) → Dense(256) → Dropout(0.4) → Softmax
```

#### Training Strategy
- **Phase 1**: Freeze both backbones, train only the fusion head (10 epochs, LR=1e-4)
- **Phase 2**: Unfreeze all layers, fine-tune end-to-end (20 epochs, LR=1e-5)

## Results

### Visual Recognition (Baseline)
| Metric | Score |
|--------|-------|
| Test Accuracy | 90% (TFLite, original model) |
| Classes | 5 |

### Fusion Model (Vision + OCR)
| Metric | Score |
|--------|-------|
| Test Accuracy | **80%** |
| Macro F1 | 0.80 |
| Classes | 5 |

#### Per-Class Performance
| Pill | Precision | Recall | F1 |
|------|-----------|--------|----|
| Diltiazem HCl | 1.00 | 0.67 | 0.80 |
| Gabapentin | 0.86 | 0.86 | 0.86 |
| Hydrocodone/APAP | 0.83 | 0.83 | 0.83 |
| Lisinopril | 0.62 | 0.83 | 0.71 |
| Metformin HCl | 0.80 | 0.80 | 0.80 |

> **Note**: The fusion model currently uses ImageNet-initialized MobileNetV2 weights (the original `best_model.h5` is incompatible with Keras 3). Re-training the vision model in the same environment should push accuracy past 90%.

## Getting Started

### Prerequisites
- Python 3.10–3.12
- TensorFlow 2.16+
- OpenCV
- NumPy
- Matplotlib (for visualization)
- WSL2 recommended for GPU training on Windows

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/PillCare.git
cd PillCare

# Install dependencies
pip install -r requirements.txt
```

### Training

#### Vision Model (Baseline)
```bash
python models/train.py
```

#### OCR Model (CRNN)
```bash
python scripts/train_crnn.py --data_dir=ocr_dataset_epillid --epochs=100 --batch_size=32
```

#### Fusion Model (Vision + OCR)
```bash
python scripts/train_fusion.py
```

### Evaluation
```bash
# Evaluate baseline
python scripts/evaluate_baseline.py

# Evaluate fusion model
python scripts/evaluate_fusion.py
```

### Inference
```bash
# Run the full pipeline on an image
python scripts/run_fusion_pipeline.py --image path_to_pill_image.jpg
```

## Project Structure
```
PillCare/
├── pill_dataset_split/          # Visual dataset (train/val/test)
├── ocr_dataset_epillid/         # OCR dataset (images + labels)
├── models/
│   ├── train.py                 # Vision model training
│   ├── crnn_epillid.h5          # Trained CRNN weights
│   └── fusion_model.h5          # Trained fusion model
├── scripts/
│   ├── train_fusion.py          # Fusion model training
│   ├── train_crnn.py            # CRNN training script
│   ├── evaluate_fusion.py       # Fusion evaluation
│   ├── evaluate_baseline.py     # Baseline evaluation
│   ├── run_fusion_pipeline.py   # Fusion inference
│   ├── run_pipeline.py          # Original pipeline
│   ├── preprocess.py            # Image preprocessing
│   ├── classify.py              # Vision classification
│   └── ocr_crnn.py              # OCR prediction
├── logs/                        # Training logs & evaluation reports
├── plots/                       # Training history plots
├── best_model.h5                # Original vision model (Keras 2)
└── README.md
```

## Future Work
- [x] Combine visual and text recognition for more accurate identification
- [ ] Re-train vision model in Keras 3 for full weight transfer to fusion
- [ ] Expand dataset with more pill types and images
- [ ] Improve OCR accuracy with more training data
- [ ] Tune fusion hyperparameters (OCR width, architecture, augmentation)
- [ ] Develop mobile application with TFLite deployment
- [ ] Implement real-time inference on mobile devices

## Author Notes

This project explores:
- Transfer learning for visual recognition
- Sequence learning with CRNN and CTC loss
- **Multi-modal fusion** of visual + text features for pill identification
- Model optimization for edge devices

## License
[Your License Here]
