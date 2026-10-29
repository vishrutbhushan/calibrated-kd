"""
cache_teacher_targets_agnews_1000.py
------------------------------------
Extracts Teacher (google/gemma-3-4b-it BF16) logits on AG News 1,000 train items
and 1,000 balanced test items.
Computes:
  1) Raw Teacher logits and probabilities
  2) Static Calibrated logits and probabilities (z - W_null)
  3) Entropy-Adaptive Calibrated logits and probabilities (z - alpha(x) * W_null)
Evaluates Teacher Direct Inference on the full 1,000 test set.
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
VERBALIZERS = ["A", "B", "C", "D"]
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

SYSTEM_INSTRUCTION = (
    "Classify the news article into one of four topics:\n"
    "A: World\nB: Sports\nC: Business\nD: Science and Technology\n\n"
    "Respond with ONLY the single letter corresponding to the correct category."
)

def build_prompt(tokenizer, article_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Article: {article_text.strip()[:1000]}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def compute_entropy_adaptive_targets(logits, null_logits):
    # 1. Softmax on raw logits
    raw_probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    raw_probs = raw_probs / np.sum(raw_probs, axis=-1, keepdims=True)

    # 2. Shannon entropy H(p) = - sum(p * log(p))
    eps = 1e-12
    safe_probs = np.clip(raw_probs, eps, 1.0)
    entropy = -np.sum(safe_probs * np.log(safe_probs), axis=-1) # shape: (N,)
    max_entropy = np.log(len(VERBALIZERS)) # log(4)

    # 3. Alpha(x) = H(p) / log(K)
    alpha = np.clip(entropy / max_entropy, 0.0, 1.0)[:, np.newaxis] # shape: (N, 1)

    # 4. Adaptive logits: z - alpha(x) * W_null
    ada_logits = logits - alpha * null_logits
    ada_probs = np.exp(ada_logits - np.max(ada_logits, axis=-1, keepdims=True))
    ada_probs = ada_probs / np.sum(ada_probs, axis=-1, keepdims=True)

    return ada_logits, ada_probs, alpha.flatten()

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
    
    print("\n" + "="*75, flush=True)
    print("STEP 1: AG NEWS TEACHER NULL BIAS EXISTENCE CHECK", flush=True)
    print("="*75, flush=True)
    null_probs = np.exp(null_logits - np.max(null_logits)) / np.sum(np.exp(null_logits - np.max(null_logits)))
    for v, l, p, c in zip(VERBALIZERS, null_logits, null_probs, CLASS_NAMES):
        print(f"  Option {v} [{c:10s}]: Logit = {l:6.2f} | Prior Prob = {p*100:6.2f}%", flush=True)
    null_gap = float(np.max(null_logits) - np.min(null_logits))
    print(f"-> Null Logit Skew Gap: {null_gap:.2f} logits", flush=True)
    print(f"-> Favored: Option {VERBALIZERS[np.argmax(null_logits)]} ({CLASS_NAMES[np.argmax(null_logits)]})", flush=True)
    print(f"-> Prior Imbalance Ratio: {np.max(null_probs)/np.min(null_probs):.1f}x", flush=True)
    print("="*75, flush=True)

    # 2. Extract Train Targets (1,000 samples)
    train_csv = DATA_DIR / "external_benchmarks" / "ag_news_train_1000.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    train_prompts = [build_prompt(tokenizer, r["text"]) for r in train_rows]
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

    # 3. Extract Test Targets (1,000 samples)
    test_csv = DATA_DIR / "external_benchmarks" / "ag_news_test_1000.csv"
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

    # Compute Soft Targets: Raw, Static, and Entropy-Adaptive
    # 1. Raw
    train_probs_raw = np.exp(train_logits - np.max(train_logits, axis=-1, keepdims=True))
    train_probs_raw = train_probs_raw / np.sum(train_probs_raw, axis=-1, keepdims=True)

    test_probs_raw = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
    test_probs_raw = test_probs_raw / np.sum(test_probs_raw, axis=-1, keepdims=True)

    # 2. Static Calibrated (z - W_null)
    train_cal_logits = train_logits - null_logits
    train_probs_cal = np.exp(train_cal_logits - np.max(train_cal_logits, axis=-1, keepdims=True))
    train_probs_cal = train_probs_cal / np.sum(train_probs_cal, axis=-1, keepdims=True)

    test_cal_logits = test_logits - null_logits
    test_probs_cal = np.exp(test_cal_logits - np.max(test_cal_logits, axis=-1, keepdims=True))
    test_probs_cal = test_probs_cal / np.sum(test_probs_cal, axis=-1, keepdims=True)

    # 3. Entropy-Adaptive Calibrated (z - alpha(x) * W_null)
    train_ada_logits, train_probs_ada, train_alphas = compute_entropy_adaptive_targets(train_logits, null_logits)
    test_ada_logits, test_probs_ada, test_alphas = compute_entropy_adaptive_targets(test_logits, null_logits)

    # Teacher Metrics on Full 1,000 Test Set
    raw_preds = np.argmax(test_logits, axis=-1)
    static_preds = np.argmax(test_cal_logits, axis=-1)
    ada_preds = np.argmax(test_ada_logits, axis=-1)

    raw_acc = accuracy_score(test_golds, raw_preds)
    raw_f1 = f1_score(test_golds, raw_preds, average="macro")
    raw_counts = np.bincount(raw_preds, minlength=4).tolist()

    static_acc = accuracy_score(test_golds, static_preds)
    static_f1 = f1_score(test_golds, static_preds, average="macro")
    static_counts = np.bincount(static_preds, minlength=4).tolist()

    ada_acc = accuracy_score(test_golds, ada_preds)
    ada_f1 = f1_score(test_golds, ada_preds, average="macro")
    ada_counts = np.bincount(ada_preds, minlength=4).tolist()

    print("\n" + "="*75, flush=True)
    print("STEP 2: FULL TEACHER INFERENCE EVALUATION (N=1,000 TEST SAMPLES)", flush=True)
    print("="*75, flush=True)
    print(f"Ground Truth Distribution (N=1,000): 250 World, 250 Sports, 250 Business, 250 Sci/Tech")
    print(f"\n1. Raw Teacher:")
    print(f"   Accuracy: {raw_acc*100:.2f}% | Macro-F1: {raw_f1*100:.2f}%")
    print(f"   Predicted Counts: {raw_counts} [World: {raw_counts[0]}, Sports: {raw_counts[1]}, Business: {raw_counts[2]}, Sci/Tech: {raw_counts[3]}]")

    print(f"\n2. Static Calibrated Teacher (z - W_null):")
    print(f"   Accuracy: {static_acc*100:.2f}% | Macro-F1: {static_f1*100:.2f}%")
    print(f"   Predicted Counts: {static_counts} [World: {static_counts[0]}, Sports: {static_counts[1]}, Business: {static_counts[2]}, Sci/Tech: {static_counts[3]}]")

    print(f"\n3. Entropy-Adaptive Teacher (z - alpha(x) * W_null):")
    print(f"   Accuracy: {ada_acc*100:.2f}% | Macro-F1: {ada_f1*100:.2f}%")
    print(f"   Predicted Counts: {ada_counts} [World: {ada_counts[0]}, Sports: {ada_counts[1]}, Business: {ada_counts[2]}, Sci/Tech: {ada_counts[3]}]")
    print(f"   Average Alpha (Uncertainty Gating): {np.mean(test_alphas):.3f}")

    print("\n" + "-"*75, flush=True)
    print(f"DELTAS ON FULL 1,000 TEST SET:")
    print(f"  Static vs Raw:      Acc Shift = {(static_acc - raw_acc)*100:+.2f}% | F1 Shift = {(static_f1 - raw_f1)*100:+.2f}%")
    print(f"  Adaptive vs Raw:    Acc Shift = {(ada_acc - raw_acc)*100:+.2f}% | F1 Shift = {(ada_f1 - raw_f1)*100:+.2f}%")
    print(f"  Adaptive vs Static: Acc Shift = {(ada_acc - static_acc)*100:+.2f}% | F1 Shift = {(ada_f1 - static_f1)*100:+.2f}%")
    print("="*75, flush=True)

    out_file = TEACHER_DIR / "teacher_targets_agnews_1000_gemma3_4b.npz"
    np.savez_compressed(
        out_file,
        train_sids=train_sids,
        train_golds=train_golds,
        train_logits_raw=train_logits,
        train_probs_raw=train_probs_raw,
        train_logits_static=train_cal_logits,
        train_probs_static=train_probs_cal,
        train_logits_ada=train_ada_logits,
        train_probs_ada=train_probs_ada,
        train_alphas=train_alphas,
        test_sids=test_sids,
        test_golds=test_golds,
        test_logits_raw=test_logits,
        test_probs_raw=test_probs_raw,
        test_logits_static=test_cal_logits,
        test_probs_static=test_probs_cal,
        test_logits_ada=test_ada_logits,
        test_probs_ada=test_probs_ada,
        test_alphas=test_alphas,
        null_logits=null_logits,
        verbalizers=VERBALIZERS,
        class_names=CLASS_NAMES,
    )
    print(f"\n[✓] Targets successfully cached to {out_file}", flush=True)

if __name__ == "__main__":
    main()

