"""
Evaluate Tesseract OCR baseline on the same split as CRNN.

Outputs logs/ocr_baseline.json with CER, word accuracy, and examples.
"""
import os
import json
from pathlib import Path
from typing import Dict

import numpy as np
import cv2
import pytesseract
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.preprocess import preprocess_for_ocr


WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ;:/-"
TESS_CONFIG = f"--oem 3 --psm 7 -c tessedit_char_whitelist={WHITELIST}"


def levenshtein(a: str, b: str) -> int:
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


def evaluate(split_json: str = "logs/ocr_split.json", data_dir: str = "ocr_dataset_epillid"):
    assert os.path.exists(split_json), "Missing logs/ocr_split.json; train CRNN first to create split."
    with open(split_json, "r") as f:
        split = json.load(f)
    samples = split.get("val", [])
    assert samples, "No validation samples found."

    cer_list = []
    word_correct = 0
    examples = []

    for item in samples:
        img_name = item["image"]
        gt_text = item["label"].upper()

        img_path = Path(data_dir) / "images" / img_name
        img = cv2.imread(str(img_path))
        proc = preprocess_for_ocr(img)  # (H, W, 1)
        proc_u8 = (proc.squeeze() * 255).astype(np.uint8)

        pred = pytesseract.image_to_string(proc_u8, config=TESS_CONFIG).strip().upper()

        c = cer(pred, gt_text)
        cer_list.append(c)
        if pred == gt_text:
            word_correct += 1
        if len(examples) < 5:
            examples.append({"image": img_name, "gt": gt_text, "pred": pred, "cer": c})

    metrics = {
        "count": len(samples),
        "cer_mean": float(np.mean(cer_list) if cer_list else 0.0),
        "cer_median": float(np.median(cer_list) if cer_list else 0.0),
        "word_accuracy": float(word_correct / len(samples) if samples else 0.0),
        "examples": examples,
    }

    os.makedirs("logs", exist_ok=True)
    out_path = "logs/ocr_baseline.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved baseline metrics to {out_path}")
    print(json.dumps(metrics, indent=2)[:800] + ("..." if len(json.dumps(metrics)) > 800 else ""))


if __name__ == "__main__":
    evaluate()
