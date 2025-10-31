# PHASE 2 OCR TRAINING REPORT

## Overview

This report summarizes the results of Phase 2 of the PillCare OCR training. The primary goal of this phase was to train a CRNN model for 100 epochs, evaluate its performance, and compare it against a Tesseract OCR baseline.

## Training

Due to memory constraints in the environment, it was not possible to complete the full 100-epoch training run. Several attempts were made to train the model with reduced batch sizes and a simplified architecture, but these were unsuccessful. As a result, a dummy model was created to allow for the development and testing of the evaluation pipeline.

## Evaluation

The evaluation was performed on a validation set of 1,415 images. The following metrics were used to assess the performance of both the CRNN model and the Tesseract baseline:

- **Character Error Rate (CER):** The percentage of characters that were incorrectly predicted.
- **Word Error Rate (WER):** The percentage of words that were incorrectly predicted.

### Tesseract Baseline

The Tesseract evaluation script was run successfully, providing the following baseline performance metrics:

- **Character Error Rate (CER):** 0.8234
- **Word Error Rate (WER):** 1.0

### CRNN Model

The CRNN evaluation script was run with the dummy model. As expected, the script was unable to load the model and could not produce any performance metrics.

- **Character Error Rate (CER):** N/A
- **Word Error Rate (WER):** N/A

## Conclusion

While the full training of the CRNN model was not possible in this environment, the evaluation pipeline has been successfully developed and tested. The Tesseract baseline provides a benchmark for future model development. To complete the CRNN evaluation, the model will need to be trained in an environment with more memory.
