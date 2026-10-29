"""
Step 1 & Step 2 Evaluation on TREC-6 and Emotion
Teacher: google/gemma-3-4b-it (BF16)
Tests null bias existence and direct inference delta before and after calibration.
"""

import os
import json
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
TEACHER_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_trec_eval(model, tokenizer):
    print("\n" + "="*75)
    print("EVALUATING BENCHMARK: TREC-6 (QUESTION CLASSIFICATION, 6 CLASSES)")
    print("="*75)
    verbalizers = ["A", "B", "C", "D", "E", "F"]
    classes = ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]
    v_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]

    prompt_template = (
        "Classify the question into one of six categories:\n"
        "A: Abbreviation (e.g., what does an acronym stand for)\n"
        "B: Entity (e.g., animals, inventions, terms)\n"
        "C: Description (e.g., definitions, explanations, reasons)\n"
        "D: Human (e.g., individual person, group, organization)\n"
        "E: Location (e.g., city, country, mountain, place)\n"
        "F: Numeric (e.g., dates, counts, money, distance)\n\n"
        "Question: {question}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\n"
        "Answer:"
    )

    null_prompt = prompt_template.format(question="N/A")

    # Step 1: Null Vector
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, v_ids].float().cpu().numpy()
    
    null_probs = np.exp(null_logits - np.max(null_logits))
    null_probs = null_probs / np.sum(null_probs)
    null_gap = np.max(null_logits) - np.min(null_logits)

    print("Step 1: Null Bias Check:")
    for v, c, l, p in zip(verbalizers, classes, null_logits, null_probs):
        print(f"  {v} [{c:12s}]: Logit = {l:6.2f} | Prior Prob = {p*100:6.2f}%")
    print(f"-> Null Logit Skew Gap: {null_gap:.2f} logits | Imbalance: {np.max(null_probs)/np.min(null_probs):.1f}x")

    # Load TREC test samples (50 per class = 300 samples)
    cache_dir = Path("/home/vishrut/.cache/huggingface/hub")
    # find cached parquet file
    p_files = list(cache_dir.glob("**/data/test-00000-of-00001-9a0214b5dca0caf8.parquet"))
    if not p_files:
        raise FileNotFoundError("TREC cached parquet not found")
    table = pq.read_table(p_files[0])
    rows = table.to_pylist()

    # Balanced subset
    balanced_rows = []
    class_counts = {i: 0 for i in range(6)}
    for r in rows:
        lbl = r["label"]
        if lbl in class_counts and class_counts[lbl] < 50:
            balanced_rows.append(r)
            class_counts[lbl] += 1
        if all(c >= 50 for c in class_counts.values()):
            break

    print(f"\nEvaluating on {len(balanced_rows)} balanced test samples (50 per class)...")
    prompts = [prompt_template.format(question=r["text"]) for r in balanced_rows]
    golds = [r["label"] for r in balanced_rows]

    all_logits = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            b = prompts[i:i+batch_size]
            enc = tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            all_logits.append(out.logits[:, -1, v_ids].float().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # Raw Inference
    preds_raw = np.argmax(all_logits, axis=-1)
    acc_raw = accuracy_score(golds, preds_raw)
    f1_raw = f1_score(golds, preds_raw, average="macro")
    counts_raw = np.bincount(preds_raw, minlength=6).tolist()

    # Calibrated Inference
    cal_logits = all_logits - null_logits
    preds_cal = np.argmax(cal_logits, axis=-1)
    acc_cal = accuracy_score(golds, preds_cal)
    f1_cal = f1_score(golds, preds_cal, average="macro")
    counts_cal = np.bincount(preds_cal, minlength=6).tolist()

    print("\nStep 2: Direct Teacher Inference Check:")
    print(f"  Raw Teacher:        Acc = {acc_raw*100:.2f}% | Macro-F1 = {f1_raw*100:.2f}% | Counts = {counts_raw}")
    print(f"  Calibrated Teacher: Acc = {acc_cal*100:.2f}% | Macro-F1 = {f1_cal*100:.2f}% | Counts = {counts_cal}")
    print(f"  Delta:              Acc Shift = {(acc_cal - acc_raw)*100:+.2f}% | F1 Shift = {(f1_cal - f1_raw)*100:+.2f}%")
    return {
        "dataset": "TREC-6",
        "null_gap": float(null_gap),
        "raw_acc": float(acc_raw*100),
        "cal_acc": float(acc_cal*100),
        "raw_f1": float(f1_raw*100),
        "cal_f1": float(f1_cal*100),
        "counts_raw": counts_raw,
        "counts_cal": counts_cal
    }

def run_emotion_eval(model, tokenizer):
    print("\n" + "="*75)
    print("EVALUATING BENCHMARK: EMOTION (HUMAN AFFECT, 6 CLASSES)")
    print("="*75)
    verbalizers = ["A", "B", "C", "D", "E", "F"]
    classes = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    v_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]

    prompt_template = (
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

    null_prompt = prompt_template.format(text="N/A")

    # Step 1: Null Vector
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, v_ids].float().cpu().numpy()
    
    null_probs = np.exp(null_logits - np.max(null_logits))
    null_probs = null_probs / np.sum(null_probs)
    null_gap = np.max(null_logits) - np.min(null_logits)

    print("Step 1: Null Bias Check:")
    for v, c, l, p in zip(verbalizers, classes, null_logits, null_probs):
        print(f"  {v} [{c:12s}]: Logit = {l:6.2f} | Prior Prob = {p*100:6.2f}%")
    print(f"-> Null Logit Skew Gap: {null_gap:.2f} logits | Imbalance: {np.max(null_probs)/np.min(null_probs):.1f}x")

    # Load Emotion test samples (50 per class = 300 samples)
    cache_dir = Path("/home/vishrut/.cache/huggingface/hub")
    p_files = list(cache_dir.glob("**/split/test-00000-of-00001.parquet"))
    if not p_files:
        raise FileNotFoundError("Emotion cached parquet not found")
    table = pq.read_table(p_files[0])
    rows = table.to_pylist()

    # Balanced subset (50 per class)
    balanced_rows = []
    class_counts = {i: 0 for i in range(6)}
    for r in rows:
        lbl = r["label"]
        if lbl in class_counts and class_counts[lbl] < 50:
            balanced_rows.append(r)
            class_counts[lbl] += 1
        if all(c >= 50 for c in class_counts.values()):
            break

    print(f"\nEvaluating on {len(balanced_rows)} balanced test samples (50 per class)...")
    prompts = [prompt_template.format(text=r["text"]) for r in balanced_rows]
    golds = [r["label"] for r in balanced_rows]

    all_logits = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            b = prompts[i:i+batch_size]
            enc = tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            all_logits.append(out.logits[:, -1, v_ids].float().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # Raw Inference
    preds_raw = np.argmax(all_logits, axis=-1)
    acc_raw = accuracy_score(golds, preds_raw)
    f1_raw = f1_score(golds, preds_raw, average="macro")
    counts_raw = np.bincount(preds_raw, minlength=6).tolist()

    # Calibrated Inference
    cal_logits = all_logits - null_logits
    preds_cal = np.argmax(cal_logits, axis=-1)
    acc_cal = accuracy_score(golds, preds_cal)
    f1_cal = f1_score(golds, preds_cal, average="macro")
    counts_cal = np.bincount(preds_cal, minlength=6).tolist()

    print("\nStep 2: Direct Teacher Inference Check:")
    print(f"  Raw Teacher:        Acc = {acc_raw*100:.2f}% | Macro-F1 = {f1_raw*100:.2f}% | Counts = {counts_raw}")
    print(f"  Calibrated Teacher: Acc = {acc_cal*100:.2f}% | Macro-F1 = {f1_cal*100:.2f}% | Counts = {counts_cal}")
    print(f"  Delta:              Acc Shift = {(acc_cal - acc_raw)*100:+.2f}% | F1 Shift = {(f1_cal - f1_raw)*100:+.2f}%")
    return {
        "dataset": "Emotion",
        "null_gap": float(null_gap),
        "raw_acc": float(acc_raw*100),
        "cal_acc": float(acc_cal*100),
        "raw_f1": float(f1_raw*100),
        "cal_f1": float(f1_cal*100),
        "counts_raw": counts_raw,
        "counts_cal": counts_cal
    }

def main():
    print("Loading Teacher model (Gemma-3-4B-it)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True
    )
    model.eval()

    res_trec = run_trec_eval(model, tokenizer)
    res_emo = run_emotion_eval(model, tokenizer)

    out_path = Path("results/new_candidate_benchmarks_step1_step2.json")
    with open(out_path, "w") as f:
        json.dump({"trec": res_trec, "emotion": res_emo}, f, indent=2)
    print(f"\n[✓] Results saved to {out_path}")

if __name__ == "__main__":
    main()

