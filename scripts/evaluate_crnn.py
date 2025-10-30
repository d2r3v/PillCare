"""
Evaluate CRNN OCR model.

Computes:
- CER (Character Error Rate)
- Word Accuracy
- Character-level confusion counts (approximate)

Saves metrics to logs/ocr_metrics.json
"""
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.ocr_crnn import CHARACTERS, CTCLayer, decode_prediction
from scripts.preprocess import preprocess_for_ocr


def load_model(model_path: str = "models/crnn_epillid.h5"):
    # Import CTCLayer from ocr_crnn to ensure consistency
    from scripts.ocr_crnn import CTCLayer
    return tf.keras.models.load_model(model_path, custom_objects={"CTCLayer": CTCLayer})


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance (edit distance)."""
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1):
        dp[i, 0] = i
    for j in range(len(b) + 1):
        dp[0, j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[len(a), len(b)])


def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, gt) / max(1, len(gt))


def approximate_char_confusions(pred: str, gt: str) -> Dict[Tuple[str, str], int]:
    # Approximate by positional comparison (not edit-aligned)
    conf = {}
    L = min(len(pred), len(gt))
    for i in range(L):
        if pred[i] != gt[i]:
            key = (gt[i], pred[i])
            conf[key] = conf.get(key, 0) + 1
    return conf


def evaluate_on_split(split_json: str = "logs/ocr_split.json", data_dir: str = "ocr_dataset_epillid", subset: str = "val", model_path: str = "models/crnn_epillid.h5"):
    assert os.path.exists(split_json), f"Split JSON not found: {split_json}. Run training first."
    with open(split_json, "r") as f:
        split = json.load(f)
    samples = split.get(subset, [])
    assert samples, f"No samples found for subset '{subset}'"

    model = load_model(model_path)

    cer_list = []
    word_correct = 0
    confusions = {}
    examples = []

    for item in samples:
        img_name = item["image"]
        gt_text = item["label"].upper()
        img_path = Path(data_dir) / "images" / img_name
        img = cv2.imread(str(img_path))
        proc = preprocess_for_ocr(img)
        proc = np.expand_dims(proc, axis=0)
        y_pred = model(proc).numpy()
        pred_text = decode_prediction(y_pred)

        c = cer(pred_text, gt_text)
        cer_list.append(c)
        if pred_text == gt_text:
            word_correct += 1

        # accumulate rough confusions
        cdict = approximate_char_confusions(pred_text, gt_text)
        for k, v in cdict.items():
            confusions[k] = confusions.get(k, 0) + v

        if len(examples) < 5:
            examples.append({"image": img_name, "gt": gt_text, "pred": pred_text, "cer": c})

    metrics = {
        "count": len(samples),
        "cer_mean": float(np.mean(cer_list) if cer_list else 0.0),
        "cer_median": float(np.median(cer_list) if cer_list else 0.0),
        "word_accuracy": float(word_correct / len(samples) if samples else 0.0),
        "top_confusions": [
            {"gt": k[0], "pred": k[1], "count": v}
            for k, v in sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:20]
        ],
        "examples": examples,
    }

    os.makedirs("logs", exist_ok=True)
    out_path = "logs/ocr_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {out_path}")
    print(json.dumps(metrics, indent=2)[:800] + ("..." if len(json.dumps(metrics)) > 800 else ""))


if __name__ == "__main__":
    evaluate_on_split()
