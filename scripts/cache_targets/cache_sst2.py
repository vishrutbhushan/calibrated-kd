"""
cache_teacher_targets_sst2.py
------------------------------
Extracts Teacher (google/gemma-3-4b-it BF16) logits on SST-2 (1,000 train items
and 200 test items). Computes the prompt-specific null vector on "Sentence: N/A\nAnswer:"
and saves both raw and calibrated soft targets.
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEACHER_DIR = DATA_DIR / "teacher_targets"
TEACHER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
VERBALIZERS = ["A", "B"]
CLASS_NAMES = ["Negative", "Positive"]

SYSTEM_INSTRUCTION = (
    "Classify the sentiment of the following sentence into one of two categories:\n"
    "A: Negative\nB: Positive\n\n"
    "Respond with ONLY the single letter corresponding to the correct category."
)

def build_prompt(tokenizer, sentence_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Sentence: {sentence_text.strip()[:1000]}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def main():
    print(f"Loading Teacher from {MODEL_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True
    )
    model.eval()

    # 1. Null Vector
    null_prompt = build_prompt(tokenizer, "N/A")
    inp_null = tokenizer(null_prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()

    print("\n--- Teacher Null Vector ---", flush=True)
    for v, l, c in zip(VERBALIZERS, null_logits, CLASS_NAMES):
        print(f"  Token '{v}' ({c}): logit = {l:.2f}", flush=True)
    null_gap = np.max(null_logits) - np.min(null_logits)
    print(f"Null Logit Skew Gap: {null_gap:.2f}", flush=True)

    # 2. Extract Train Targets (1,000 samples)
    train_csv = DATA_DIR / "external_benchmarks" / "sst2_train_1000.csv"
    train_rows = []
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    train_prompts = [build_prompt(tokenizer, r["text"]) for r in train_rows]
    train_sids = [r["sample_id"] for r in train_rows]
    train_golds = [int(r["label"]) for r in train_rows]

    print(f"\nExtracting Teacher logits on {len(train_prompts)} training samples...", flush=True)
    train_logits = []
    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(train_prompts), batch_size):
            b = train_prompts[i : i + batch_size]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            train_logits.append(logits)
    train_logits = np.concatenate(train_logits, axis=0)

    # 3. Extract Test Targets (200 samples)
    test_csv = DATA_DIR / "external_benchmarks" / "sst2_test_200.csv"
    test_rows = []
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    test_prompts = [build_prompt(tokenizer, r["text"]) for r in test_rows]
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

    # Compute raw & calibrated probabilities
    # Train
    train_probs_raw = np.exp(train_logits - np.max(train_logits, axis=-1, keepdims=True))
    train_probs_raw = train_probs_raw / np.sum(train_probs_raw, axis=-1, keepdims=True)

    train_cal_logits = train_logits - null_logits
    train_probs_cal = np.exp(train_cal_logits - np.max(train_cal_logits, axis=-1, keepdims=True))
    train_probs_cal = train_probs_cal / np.sum(train_probs_cal, axis=-1, keepdims=True)

    # Test
    test_probs_raw = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
    test_probs_raw = test_probs_raw / np.sum(test_probs_raw, axis=-1, keepdims=True)

    test_cal_logits = test_logits - null_logits
    test_probs_cal = np.exp(test_cal_logits - np.max(test_cal_logits, axis=-1, keepdims=True))
    test_probs_cal = test_probs_cal / np.sum(test_probs_cal, axis=-1, keepdims=True)

    # Teacher Metrics on Test
    raw_preds = np.argmax(test_logits, axis=-1)
    cal_preds = np.argmax(test_cal_logits, axis=-1)

    raw_acc = accuracy_score(test_golds, raw_preds)
    raw_f1 = f1_score(test_golds, raw_preds, average="macro")
    cal_acc = accuracy_score(test_golds, cal_preds)
    cal_f1 = f1_score(test_golds, cal_preds, average="macro")

    print(f"\n--- Teacher Test Performance (N=200) ---", flush=True)
    print(f"Raw Teacher:        Acc = {raw_acc*100:.2f}% | Macro-F1 = {raw_f1*100:.2f}%", flush=True)
    print(f"Calibrated Teacher: Acc = {cal_acc*100:.2f}% | Macro-F1 = {cal_f1*100:.2f}%", flush=True)
    print(f"Raw Pred Counts:        {[int(np.sum(raw_preds == c)) for c in range(2)]}", flush=True)
    print(f"Calibrated Pred Counts: {[int(np.sum(cal_preds == c)) for c in range(2)]}", flush=True)

    out_file = TEACHER_DIR / "teacher_targets_sst2_gemma3_4b.npz"
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

