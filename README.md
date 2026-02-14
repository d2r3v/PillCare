# PillCare — Intelligent Pill Identifier with Visual and Text Recognition

**PillCare** is a deep learning-powered application designed to identify pills using both visual appearance (shape, color) and text imprints. It's built for accessibility and safety, especially for elderly users or those managing multiple medications.

## Features

### Visual Recognition
- **Pill recognition** from camera input using MobileNetV2
- Transfer learning with fine-tuning
- Predicts among **16 medication classes**
- Converted to **TensorFlow Lite** for mobile/edge deployment

### Text Recognition (OCR)
- **CRNN (Convolutional Recurrent Neural Network)** for pill imprint recognition
- Handles variable-length text sequences
- Preprocessing pipeline for pill images
- CTC (Connectionist Temporal Classification) loss for sequence learning

### Fusion Model (Vision + OCR)
- **Gated Fusion** architecture combining MobileNetV2 visual features with CRNN text features
- Confidence gating mechanism that learns when to trust/ignore OCR features
- Class-weighted training for imbalanced datasets
- Achieves **81% test accuracy** across 16 drug classes

## Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | TensorFlow / Keras |
| Model Architectures | MobileNetV2 (visual), CRNN (text), Gated Fusion (combined) |
| Image Preprocessing | OpenCV, NumPy |
| Text Processing | TensorFlow Text, Regular Expressions |
| Data Augmentation | OpenCV (offline), TensorFlow (online) |
| Dataset Expansion | openFDA API, ePillID |
| Model Deployment | TensorFlow Lite |

## Dataset

### Visual Recognition Dataset
- **707 images** across **16 drug classes**, expanded from [ePillID](https://github.com/usuyama/ePillID-benchmark) segmented NIH pill images
- Augmented offline with rotation, flip, brightness, contrast, perspective, noise, and crop transforms
- Preprocessed to 224x224 resolution
- Balanced via class-weighted training + targeted augmentation

#### Supported Drug Classes
| Drug | Train Images | Drug | Train Images |
|------|-------------|------|-------------|
| Diltiazem HCl | 70 | Simvastatin | 50 |
| Lisinopril | 67 | Gabapentin | 52 |
| Metformin HCl | 49 | Hydrocodone/APAP | 43 |
| Warfarin Sodium | 32 | Prednisone | 19 |
| Hydrochlorothiazide | 19 | Levothyroxine Sodium | 19 |
| Metoprolol Tartrate | 18 | Amlodipine Besylate | 16 |
| Losartan Potassium | 16 | Carvedilol | 15 |
| Pantoprazole Sodium | 15 | Amoxicillin | 12 |

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
Gate:           OCR features → Dense(64) → Sigmoid → scale OCR features
                                    ↓
Fusion:         Concatenate(vision_features, gated_ocr) → Dense(256) → Dropout(0.4) → Softmax
```

#### Training Strategy
- **Phase 1**: Freeze both backbones, train fusion head + gate (25 epochs, LR=1e-4, class weights)
- **Phase 2**: Disabled — causes catastrophic forgetting with small datasets
- Fine-tuned MobileNetV2 weights loaded from separately trained vision model

## Results

### Vision-Only Model (MobileNetV2)
| Metric | Score |
|--------|-------|
| Test Accuracy | **80%** |
| Macro F1 | 0.83 |
| Classes | 16 |

### Fusion Model (Vision + Gated OCR)
| Metric | Score |
|--------|-------|
| Test Accuracy | **84%** |
| Macro F1 | 0.85 |
| Classes | 16 |

With increased augmentation (200 samples per class), the late-fusion model outperformed the vision-only baseline by +3.6%, demonstrating the benefit of multi-modal integration when sufficient data is available.

#### Per-Class Performance (Fusion)
| Pill | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Amlodipine Besylate | 0.67 | 0.80 | 0.73 |
| Amoxicillin | 0.57 | 1.00 | 0.73 |
| Carvedilol | 1.00 | 1.00 | 1.00 |
| Diltiazem HCl | 0.73 | 0.57 | 0.64 |
| Gabapentin | 1.00 | 0.78 | 0.88 |
| Hydrochlorothiazide | 0.50 | 0.60 | 0.55 |
| Hydrocodone/APAP | 0.86 | 1.00 | 0.92 |
| Levothyroxine Sodium | 0.83 | 1.00 | 0.91 |
| Lisinopril | 0.88 | 0.58 | 0.70 |
| Losartan Potassium | 1.00 | 1.00 | 1.00 |
| Metformin HCl | 1.00 | 0.88 | 0.93 |
| Metoprolol Tartrate | 0.71 | 1.00 | 0.83 |
| Pantoprazole Sodium | 0.80 | 1.00 | 0.89 |
| Prednisone | 1.00 | 1.00 | 1.00 |
| Simvastatin | 1.00 | 0.92 | 0.96 |
| Warfarin Sodium | 0.89 | 1.00 | 0.94 |

## Getting Started

### Prerequisites
- Python 3.10–3.12
- TensorFlow 2.16+
- OpenCV
- NumPy, scikit-learn
- Matplotlib (for visualization)
- WSL2 recommended for GPU training on Windows

### Installation
```bash
# Clone the repository
git clone https://github.com/d2r3v/PillCare.git
cd PillCare

# Install dependencies
pip install -r requirements.txt
```

### Training

#### Vision Model
```bash
python models/train_v2.py
```

#### OCR Model (CRNN)
```bash
python scripts/train_crnn.py --data_dir=ocr_dataset_epillid --epochs=100 --batch_size=32
```

#### Fusion Model (Vision + OCR)
```bash
python scripts/train_fusion.py
```

#### Dataset Expansion
```bash
# Expand dataset from ePillID source images
python scripts/expand_dataset.py

# Offline data augmentation (3-5x more training images)
python scripts/augment_dataset.py --target 150
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
├── pill_dataset_split/          # Visual dataset (train/val/test, 16 classes)
├── ocr_dataset_epillid/         # OCR dataset (images + labels)
├── data/
│   ├── ePillID_data/            # Source ePillID images
│   └── label_map.json           # Class index → drug name mapping
├── models/
│   ├── train.py                 # Original vision training (Keras 2)
│   ├── train_v2.py              # Vision training (Keras 3, class weights)
│   ├── vision_model_v2.keras    # Trained vision model
│   ├── crnn_epillid.h5          # Trained CRNN weights
│   └── fusion_model.h5          # Trained fusion model
├── scripts/
│   ├── train_fusion.py          # Fusion model training (gated fusion)
│   ├── train_crnn.py            # CRNN training script
│   ├── evaluate_fusion.py       # Fusion evaluation
│   ├── evaluate_baseline.py     # Baseline evaluation
│   ├── expand_dataset.py        # Dataset expansion via openFDA API
│   ├── augment_dataset.py       # Offline data augmentation
│   ├── run_fusion_pipeline.py   # Fusion inference
│   ├── preprocess.py            # Image preprocessing
│   ├── classify.py              # Vision classification
│   └── ocr_crnn.py              # OCR prediction
├── logs/                        # Training logs & evaluation reports
├── plots/                       # Training history plots
└── README.md
```

## Future Work
- [x] Combine visual and text recognition for more accurate identification
- [x] Re-train vision model in Keras 3 for full weight transfer to fusion
- [x] Expand dataset to 16 drug classes (707 images)
- [x] Implement confidence gating for OCR branch
- [ ] Further augmentation to 250+ images per class
- [ ] Late fusion (ensemble) approach for combining models
- [ ] Improve weakest classes (hydrochlorothiazide, lisinopril)
- [ ] Develop mobile application with TFLite deployment
- [ ] Implement real-time inference on mobile devices

## Development Journey & Lessons Learned

This project went through several iterations. Here are the key challenges encountered and how they were resolved:

### 1. Catastrophic Forgetting During Fine-Tuning
**Problem**: Unfreezing the full MobileNetV2 backbone in Phase 2 caused train accuracy to drop 30-40% immediately, destroying all progress from Phase 1.

**What was tried**:
- Full unfreeze with low LR (1e-5) — accuracy crashed
- Partial unfreeze (last 30 layers) with LR 1e-6 — still dropped ~10%
- Various learning rate schedules

**Solution**: Disabled Phase 2 entirely. With small datasets (<1000 images), training only the fusion head on top of frozen pretrained features is more stable and performant.

### 2. OCR Branch Hurting Instead of Helping
**Problem**: The fusion model (53%) initially performed *worse* than vision-only (77%). The CRNN text features added noise for pills without readable imprints.

| Approach | Accuracy | Outcome |
|----------|----------|---------|
| Vision-only | 77% | Best standalone |
| Naive fusion (concat) | 53% | OCR noise dominated |
| Fusion + class weights | 72% | Improved but still behind |
| Fusion + confidence gating | 68% | Gate added too many params |
| Fusion + gating + augmentation | **81%** | Finally competitive |

**Key insight**: The confidence gating *concept* was right (learn when to ignore OCR), but it only worked once we had enough data for the gate to learn meaningful patterns. Architecture improvements without data are ineffective.

### 3. Data Quantity > Model Complexity
**Problem**: Every architecture trick gave diminishing returns. The real bottleneck was always data.

| Change | Impact |
|--------|--------|
| Confidence gating | -4% (hurt with small data) |
| Class weights | +19% (53% → 72%) |
| Disable Phase 2 | +3% |
| **Data augmentation (3x)** | **+9%** (biggest single gain) |

**Lesson**: With small datasets, simple approaches (frozen backbone + class weights + more data) consistently outperform complex ones (gating, multi-phase fine-tuning).

### 4. Class Imbalance Kills Minority Classes
**Problem**: After expanding to 16 classes, 6 drugs had 0% precision/recall — the model simply never predicted them.

**Root cause**: Amoxicillin had 12 training images while diltiazem had 70. The cross-entropy loss optimized for the majority classes.

**Solution**: `sklearn.utils.class_weight.compute_class_weight("balanced")` upweighted rare classes proportionally. Combined with targeted augmentation (generating more images for underrepresented classes), all 16 classes achieved non-zero performance.

### 5. Keras 2 → Keras 3 Migration Pain
**Problem**: The original vision model (`best_model.h5`) couldn't be loaded in TensorFlow 2.16+ (Keras 3) due to `Sequential` model format changes and the deprecated `.h5` save format.

**Solution**: Rewrote training with the Functional API and `.keras` save format. Pre-trained the vision model separately, then loaded its MobileNetV2 weights into the fusion model.

### 6. Regularization Isn't Always Free
**Problem**: Label smoothing (0.1) + cosine LR decay reduced test accuracy (fusion 83.78% → 82.88%, vision 80% → 78%).

**Key insight**: With small, noisy datasets, the model benefits more from stronger supervision (hard labels) than softer targets. Label smoothing dilutes the learning signal when every training example counts. Cosine decay reduced the LR too aggressively before the model fully converged. Reverted both changes — the simple constant LR + hard cross-entropy remained the best configuration.

## Author Notes

This project explores:
- Transfer learning for visual recognition
- Sequence learning with CRNN and CTC loss
- **Multi-modal fusion** with confidence gating for pill identification
- Class-balanced training for imbalanced medical datasets
- Automated dataset expansion using FDA drug databases
- Model optimization for edge devices

## License
[Your License Here]
