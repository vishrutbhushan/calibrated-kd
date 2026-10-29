"""
cache_teacher_targets_emotion_1200.py
-------------------------------------
Extracts Teacher (google/gemma-3-4b-it BF16) logits on Emotion 1,200 train items
and 300 balanced test items.
Computes:
  1) Raw Teacher logits and probabilities
  2) Static Calibrated logits and probabilities (z - W_null)
Evaluates Teacher Direct Inference on the test set.
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEACHER_DIR = DATA_DIR / "teacher_targets"
TEACHER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
VERBALIZERS = ["A", "B", "C", "D", "E", "F"]
CLASS_NAMES = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

PROMPT_TEMPLATE = (
    "Classify the emotional sentiment of the text into one of six categories:\n"
    "A: Sadness\n"
    "B: Joy\n"
    "C: Love\n"
    "D: Anger\n"
    "E: Fear\n"
    "F: Surprise\n\n"
    "Text: {text}\n"
    "Respond with ONLY the single letter corresponding to the correct category.\n"
    "Answer:"
)

def build_prompt(text: str):
    return PROMPT_TEMPLATE.format(text=text.strip()[:600])

def main():
    print(f"Loading Teacher from {MODEL_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True
    )
    model.eval()

    # 1. Null Vector
    null_prompt = build_prompt("N/A")
    inp_null = tokenizer(null_prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()
    
    print("\n" + "="*75, flush=True)
    print("STEP 1: EMOTION TEACHER NULL BIAS EXISTENCE CHECK", flush=True)
    print("="*75, flush=True)
    null_probs = np.exp(null_logits - np.max(null_logits)) / np.sum(np.exp(null_logits - np.max(null_logits)))
    for v, l, p, c in zip(VERBALIZERS, null_logits, null_probs, CLASS_NAMES):
        print(f"  Option {v} [{c:10s}]: Logit = {l:6.2f} | Prior Prob = {p*100:6.2f}%", flush=True)
    null_gap = float(np.max(null_logits) - np.min(null_logits))
    print(f"-> Null Logit Skew Gap: {null_gap:.2f} logits", flush=True)
    print(f"-> Favored: Option {VERBALIZERS[np.argmax(null_logits)]} ({CLASS_NAMES[np.argmax(null_logits)]}) with {np.max(null_probs)*100:.2f}%", flush=True)
    print(f"-> Prior Imbalance Ratio: {np.max(null_probs)/np.min(null_probs):.1f}x", flush=True)
    print("="*75, flush=True)

    # 2. Extract Train Targets (1,200 samples)
    train_csv = DATA_DIR / "external_benchmarks" / "emotion_train_1200.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    train_prompts = [build_prompt(r["text"]) for r in train_rows]
    train_sids = [r["sample_id"] for r in train_rows]
    train_golds = [int(r["label"]) for r in train_rows]

    print(f"\nExtracting Teacher logits on {len(train_prompts)} training samples...", flush=True)
    train_logits = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(train_prompts), batch_size):
            b = train_prompts[i : i + batch_size]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            train_logits.append(logits)
    train_logits = np.concatenate(train_logits, axis=0)

    # 3. Extract Test Targets (300 samples)
    test_csv = DATA_DIR / "external_benchmarks" / "emotion_test_300.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    test_prompts = [build_prompt(r["text"]) for r in test_rows]
    test_sids = [r["sample_id"] for r in test_rows]
    test_golds = [int(r["label"]) for r in test_rows]

    print(f"Extracting Teacher logits on {len(test_prompts)} test samples...", flush=True)
    test_logits = []
    with torch.no_grad():
        for i in range(0, len(test_prompts), batch_size):
            b = test_prompts[i : i + batch_size]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            test_logits.append(logits)
    test_logits = np.concatenate(test_logits, axis=0)

    # Compute Soft Targets: Raw and Calibrated
    train_probs_raw = np.exp(train_logits - np.max(train_logits, axis=-1, keepdims=True))
    train_probs_raw = train_probs_raw / np.sum(train_probs_raw, axis=-1, keepdims=True)

    test_probs_raw = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
    test_probs_raw = test_probs_raw / np.sum(test_probs_raw, axis=-1, keepdims=True)

    train_cal_logits = train_logits - null_logits
    train_probs_cal = np.exp(train_cal_logits - np.max(train_cal_logits, axis=-1, keepdims=True))
    train_probs_cal = train_probs_cal / np.sum(train_probs_cal, axis=-1, keepdims=True)

    test_cal_logits = test_logits - null_logits
    test_probs_cal = np.exp(test_cal_logits - np.max(test_cal_logits, axis=-1, keepdims=True))
    test_probs_cal = test_probs_cal / np.sum(test_probs_cal, axis=-1, keepdims=True)

    # Metrics on Test
    raw_preds = np.argmax(test_logits, axis=-1)
    cal_preds = np.argmax(test_cal_logits, axis=-1)

    raw_acc = accuracy_score(test_golds, raw_preds)
    raw_f1 = f1_score(test_golds, raw_preds, average="macro")
    raw_counts = np.bincount(raw_preds, minlength=6).tolist()

    cal_acc = accuracy_score(test_golds, cal_preds)
    cal_f1 = f1_score(test_golds, cal_preds, average="macro")
    cal_counts = np.bincount(cal_preds, minlength=6).tolist()

    print("\n" + "="*75, flush=True)
    print("STEP 2: TEACHER INFERENCE EVALUATION (N=300 BALANCED TEST SAMPLES)", flush=True)
    print("="*75, flush=True)
    print(f"Ground Truth Distribution (N=300): 50 per class across 6 emotions")
    print(f"\n1. Raw Teacher:")
    print(f"   Accuracy: {raw_acc*100:.2f}% | Macro-F1: {raw_f1*100:.2f}%")
    print(f"   Predicted Counts: {raw_counts} {list(zip(CLASS_NAMES, raw_counts))}")

    print(f"\n2. Calibrated Teacher (z - W_null):")
    print(f"   Accuracy: {cal_acc*100:.2f}% | Macro-F1: {cal_f1*100:.2f}%")
    print(f"   Predicted Counts: {cal_counts} {list(zip(CLASS_NAMES, cal_counts))}")

    print("\n" + "-"*75, flush=True)
    print(f"DELTAS ON 300 TEST SAMPLES:")
    print(f"  Calibrated vs Raw: Acc Shift = {(cal_acc - raw_acc)*100:+.2f}% | F1 Shift = {(cal_f1 - raw_f1)*100:+.2f}%")
    print("="*75, flush=True)

    out_file = TEACHER_DIR / "teacher_targets_emotion_1200_gemma3_4b.npz"
    np.savez_compressed(
        out_file,
        train_sids=train_sids,
        train_golds=train_golds,
        train_logits_raw=train_logits,
        train_probs_raw=train_probs_raw,
        train_logits_cal=train_cal_logits,
        train_probs_cal=train_probs_cal,
        test_sids=test_sids,
        test_golds=test_golds,
        test_logits_raw=test_logits,
        test_probs_raw=test_probs_raw,
        test_logits_cal=test_cal_logits,
        test_probs_cal=test_probs_cal,
        null_logits=null_logits,
        verbalizers=VERBALIZERS,
        class_names=CLASS_NAMES,
    )
    print(f"\n[✓] Targets successfully cached to {out_file}", flush=True)

if __name__ == "__main__":
    main()

