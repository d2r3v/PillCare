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

## Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | TensorFlow / Keras |
| Model Architectures | MobileNetV2 (visual), CRNN (text) |
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

## CRNN OCR Pipeline

### Data Preparation
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

### Model Architecture

```
Input → Conv2D → BatchNorm → ReLU → MaxPool2D → Dropout →
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool2D → Dropout →
Conv2D → BatchNorm → ReLU → Dropout → Reshape →
Bidirectional(GRU) → BatchNorm → Dropout →
Bidirectional(GRU) → Dense → Softmax → CTCLoss
```

### Training
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: CTC Loss for sequence learning
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20%

### Data Augmentation
- Random rotation (±5 degrees)
- Random brightness/contrast adjustments
- Small translations

## Results

### Visual Recognition
| Metric | Score |
|--------|-------|
| Test Accuracy | 90% (TFLite) |
| Classes | 5+ |
| Model Size | ~10 MB |
| Inference Time | ~50ms (mobile) |

### OCR Pipeline
| Metric | Target |
|--------|--------|
| Character Set | A-Z, 0-9, common symbols |
| Input Height | 32px (variable width) |
| Sequence Length | Variable (handled by CTC) |
| Character Error Rate (CER) | TBD (in progress) |

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib (for visualization)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/PillCare.git
cd PillCare

# Install dependencies
pip install -r requirements.txt
```

### Training the OCR Model
```bash
python scripts/train_crnn.py --data_dir=ocr_dataset_epillid --epochs=100 --batch_size=32
```

### Using the Trained Model
```python
from scripts.inference import predict_pill_text

# Load and preprocess image
image_path = "path_to_pill_image.jpg"
predicted_text = predict_pill_text(image_path)
print(f"Predicted text: {predicted_text}")
```

## Project Structure
```
PillCare/
├── data/                    # Raw and processed datasets
│   ├── pill_dataset/        # Visual recognition dataset
│   └── ocr_dataset_epillid/ # OCR dataset (images + labels)
├── models/                  # Saved models
├── scripts/
│   ├── train_crnn.py        # CRNN training script
│   ├── preprocess.py        # Image preprocessing utilities
│   ├── create_ocr_dataset.py # Dataset preparation
│   └── inference.py         # Model inference utilities
├── notebooks/               # Jupyter notebooks for exploration
└── README.md                # This file
```

## Future Work
- [ ] Improve OCR accuracy with more training data
- [ ] Combine visual and text recognition for more accurate identification
- [ ] Develop mobile application with TFLite deployment
- [ ] Add support for more pill types and imprints
- [ ] Implement real-time inference on mobile devices

## Author Notes

This project explores:
- Transfer learning for visual recognition
- Sequence learning with CRNN and CTC loss
- Multi-modal pill identification (visual + text)
- Model optimization for edge devices

## License
[Your License Here]
