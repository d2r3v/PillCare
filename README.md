# PillCare — Intelligent Pill Identifier with Visual Recognition

**PillCare** is a deep learning–powered application designed to visually identify pills based on their appearance — including shape, color, and imprint. It’s built for accessibility and safety, especially for elderly users or those managing multiple medications.

## Features

- **Pill recognition** from camera input (image classifier)
- Model trained using transfer learning (MobileNetV2 + fine-tuning)
- Predicts among 5 test classes: diltiazem, lisinopril, gabapentin, metformin, hydrocodone
- Converted to **TensorFlow Lite** for mobile/edge deployment
- Includes evaluation with confusion matrix, accuracy metrics

## Tech Stack

| Component | Tool |
|----------|------|
| Model     | TensorFlow / Keras (MobileNetV2) |
| Deployment | TFLite |
| Image Preprocessing | OpenCV / PIL |
| Data Split | Custom Python script (train/val/test folders) |
| Evaluation | scikit-learn, seaborn (confusion matrix) |

## Dataset

Images extracted from the [Pillbox dataset](https://www.fda.gov/drugs/pillbox) — filtered and matched with metadata. Preprocessed images are resized to 224x224 and classified by `medicine_name`.

We currently use a small subset of 5 classes for prototype purposes:
- diltiazem hydrochloride
- gabapentin
- lisinopril
- metformin hydrochloride
- hydrocodone + acetaminophen

> `pill_dataset_split/` contains train/val/test folders with resized images.

## Results

| Metric     | Score |
|------------|-------|
| Test Accuracy | **90% (TFLite)** |
| Classes     | 5 |
| Model Size  | ~10 MB (after conversion) |
| Inference Time | ~50ms (on mobile device) |

> Model achieves 90% accuracy on test images. 


## Author Notes

This is a learning-driven, employability-focused project meant to explore:
- Transfer learning
- Model deployment with TFLite
- Multi-modal pill identification (image + OCR coming soon)


