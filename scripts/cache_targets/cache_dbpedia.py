"""
cache_teacher_targets_dbpedia_1400.py
-------------------------------------
Extracts Teacher (google/gemma-3-4b-it BF16) logits on DBpedia 1,400 train items
and 700 balanced test items.
Computes:
  1) Raw Teacher logits and probabilities
  2) Static Calibrated logits and probabilities (z - W_null)
Evaluates Teacher Direct Inference on the test set.
Saves targets to:
  - data/teacher_targets/teacher_targets_dbpedia_1400_gemma3_4b.npz
  - /home/vishrut/calibrated-kd/data/teacher_targets/teacher_targets_dbpedia_1400_gemma3_4b.npz
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEACHER_DIR = DATA_DIR / "teacher_targets"
TEACHER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]
VERBALIZERS = [chr(65 + i) for i in range(14)] # A through N

def format_prompt(text):
    c_str = "\n".join([f"{v}: {c}" for v, c in zip(VERBALIZERS, CLASSES)])
    return (
        "Classify the text into one of fourteen categories:\n"
        f"{c_str}\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\n"
        "Answer:"
    )

def main():
    print(f"Loading Teacher from {MODEL_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers token mapping: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True
    )
    model.eval()

    # 1. Null Vector
    null_prompt = format_prompt("N/A")
    inp_null = tokenizer(null_prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()

    null_probs = np.exp(null_logits - np.max(null_logits)) / np.sum(np.exp(null_logits - np.max(null_logits)))
    null_gap = float(np.max(null_logits) - np.min(null_logits))

    print("\n" + "="*75, flush=True)
    print("STEP 1: DBPEDIA-14 NULL BIAS EXISTENCE CHECK", flush=True)
    print("="*75, flush=True)
    for v, c, l, p in zip(VERBALIZERS, CLASSES, null_logits, null_probs):
        print(f"  {v} [{c:22s}]: Logit = {l:6.2f} | Prior Prob = {p*100:6.2f}%", flush=True)
    print(f"-> Null Skew Gap (Max - Min Logit): {null_gap:.2f} logits", flush=True)
    print(f"-> Favored: Option {VERBALIZERS[np.argmax(null_logits)]} ({CLASSES[np.argmax(null_logits)]}) with {np.max(null_probs)*100:.2f}%", flush=True)
    print(f"-> Imbalance Ratio: {np.max(null_probs)/np.min(null_probs):.1f}x", flush=True)
    print("="*75, flush=True)

    # 2. Extract Train Targets (1,400 samples)
    train_csv = DATA_DIR / "external_benchmarks" / "dbpedia_train_1400.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    train_prompts = [format_prompt(r["text"]) for r in train_rows]
    train_texts = [r["text"] for r in train_rows]
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
            del inp, out
            if (i // batch_size) % 50 == 0:
                print(f"  Train: Processed {min(i + batch_size, len(train_prompts))}/{len(train_prompts)}...", flush=True)
    train_logits = np.concatenate(train_logits, axis=0)

    # 3. Extract Test Targets (700 samples)
    test_csv = DATA_DIR / "external_benchmarks" / "dbpedia_test_700.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    test_prompts = [format_prompt(r["text"]) for r in test_rows]
    test_texts = [r["text"] for r in test_rows]
    test_golds = [int(r["label"]) for r in test_rows]

    print(f"\nExtracting Teacher logits on {len(test_prompts)} test samples...", flush=True)
    test_logits = []
    with torch.no_grad():
        for i in range(0, len(test_prompts), batch_size):
            b = test_prompts[i : i + batch_size]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            test_logits.append(logits)
            del inp, out
            if (i // batch_size) % 30 == 0:
                print(f"  Test: Processed {min(i + batch_size, len(test_prompts))}/{len(test_prompts)}...", flush=True)
    test_logits = np.concatenate(test_logits, axis=0)

    # 4. Compute Softmax Probabilities (Raw vs Calibrated)
    def softmax_logits(l):
        e = np.exp(l - np.max(l, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    train_probs_raw = softmax_logits(train_logits)
    train_logits_cal = train_logits - null_logits
    train_probs_cal = softmax_logits(train_logits_cal)

    test_probs_raw = softmax_logits(test_logits)
    test_logits_cal = test_logits - null_logits
    test_probs_cal = softmax_logits(test_logits_cal)

    # 5. Evaluate Teacher Direct Inference on Test Set (N=700)
    raw_preds = np.argmax(test_logits, axis=-1)
    raw_acc = float(accuracy_score(test_golds, raw_preds))
    raw_f1 = float(f1_score(test_golds, raw_preds, average="macro"))
    raw_counts = np.bincount(raw_preds, minlength=14).tolist()

    cal_preds = np.argmax(test_logits_cal, axis=-1)
    cal_acc = float(accuracy_score(test_golds, cal_preds))
    cal_f1 = float(f1_score(test_golds, cal_preds, average="macro"))
    cal_counts = np.bincount(cal_preds, minlength=14).tolist()

    print("\n" + "="*75, flush=True)
    print("STEP 2: TEACHER DIRECT INFERENCE RESULTS (N=700 BALANCED TEST SET)", flush=True)
    print("="*75, flush=True)
    print(f"Raw Teacher:        Acc = {raw_acc*100:.2f}% | Macro-F1 = {raw_f1*100:.2f}%", flush=True)
    print(f"Calibrated Teacher: Acc = {cal_acc*100:.2f}% | Macro-F1 = {cal_f1*100:.2f}%", flush=True)
    print(f"Delta:              Acc Shift = {(cal_acc - raw_acc)*100:+.2f}% | F1 Shift = {(cal_f1 - raw_f1)*100:+.2f}%", flush=True)
    print("\nPer-class prediction counts (Target = 50 each):", flush=True)
    for v, c, cr, cc in zip(VERBALIZERS, CLASSES, raw_counts, cal_counts):
        print(f"  {v} [{c:22s}]: Raw = {cr:2d} | Cal = {cc:2d}", flush=True)
    print("="*75, flush=True)

    # 6. Save Targets
    npz_data = {
        "train_texts": np.array(train_texts),
        "train_golds": np.array(train_golds),
        "train_logits_raw": train_logits,
        "train_probs_raw": train_probs_raw,
        "train_logits_cal": train_logits_cal,
        "train_probs_cal": train_probs_cal,
        "test_texts": np.array(test_texts),
        "test_golds": np.array(test_golds),
        "test_logits_raw": test_logits,
        "test_probs_raw": test_probs_raw,
        "test_logits_cal": test_logits_cal,
        "test_probs_cal": test_probs_cal,
        "null_logits": null_logits,
        "null_probs": null_probs,
        "verbalizers": np.array(VERBALIZERS),
        "class_names": np.array(CLASSES)
    }

    out_paths = [
        DATA_DIR / "teacher_targets" / "teacher_targets_dbpedia_1400_gemma3_4b.npz",
        Path("/home/vishrut/calibrated-kd/data/teacher_targets/teacher_targets_dbpedia_1400_gemma3_4b.npz")
    ]

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **npz_data)
        print(f"[✓] Saved teacher targets to {p}", flush=True)

if __name__ == "__main__":
    main()
