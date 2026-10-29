"""
Cache Teacher Targets for NLI (3 Classes: Entailment, Neutral, Contradiction)
Teacher: google/gemma-3-4b-it (BF16)
Outputs: data/teacher_targets/teacher_targets_nli_gemma3_4b.npz
"""

import csv
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score

TEACHER_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
VERBALIZERS = ["A", "B", "C"]
CLASS_NAMES = ["Entailment", "Neutral", "Contradiction"]

def format_prompt(premise, hypothesis):
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Does the premise entail, remain neutral, or contradict the hypothesis?\n"
        "Options:\n"
        "A. Entailment\n"
        "B. Neutral\n"
        "C. Contradiction\n"
        "Answer:"
    )

NULL_PROMPT = (
    "Premise: N/A\n"
    "Hypothesis: N/A\n"
    "Does the premise entail, remain neutral, or contradict the hypothesis?\n"
    "Options:\n"
    "A. Entailment\n"
    "B. Neutral\n"
    "C. Contradiction\n"
    "Answer:"
)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers token mapping: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    # 1. Null Vector
    null_inputs = tokenizer(NULL_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        null_out = model(**null_inputs)
        null_logits = null_out.logits[0, -1, verbalizer_ids].float().cpu().numpy()

    null_probs = np.exp(null_logits - np.max(null_logits))
    null_probs = null_probs / np.sum(null_probs)
    null_gap = float(np.max(null_logits) - np.min(null_logits))

    print(f"\nNull Vector: {null_logits} | Gap: {null_gap:.2f}")

    # 2. Extract Train Targets (900 samples)
    train_csv = Path("data/external_benchmarks/nli_train_900.csv")
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    train_prompts = [format_prompt(r["premise"], r["hypothesis"]) for r in train_rows]
    train_golds = [int(r["label"]) for r in train_rows]

    print(f"Extracting Teacher logits on {len(train_prompts)} train samples...", flush=True)
    batch_size = 16
    train_logits = []
    with torch.no_grad():
        for i in range(0, len(train_prompts), batch_size):
            b = train_prompts[i : i + batch_size]
            enc = tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = model(**enc)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            train_logits.append(logits)
    train_logits = np.concatenate(train_logits, axis=0)

    # 3. Extract Test Targets (300 samples)
    test_csv = Path("data/external_benchmarks/nli_test_300.csv")
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    test_prompts = [format_prompt(r["premise"], r["hypothesis"]) for r in test_rows]
    test_golds = [int(r["label"]) for r in test_rows]

    print(f"Extracting Teacher logits on {len(test_prompts)} test samples...", flush=True)
    test_logits = []
    with torch.no_grad():
        for i in range(0, len(test_prompts), batch_size):
            b = test_prompts[i : i + batch_size]
            enc = tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = model(**enc)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            test_logits.append(logits)
    test_logits = np.concatenate(test_logits, axis=0)

    # Compute soft target probabilities (Raw and Calibrated)
    # Train
    train_probs_raw = np.exp(train_logits - np.max(train_logits, axis=-1, keepdims=True))
    train_probs_raw = train_probs_raw / np.sum(train_probs_raw, axis=-1, keepdims=True)

    train_logits_cal = train_logits - null_logits
    train_probs_cal = np.exp(train_logits_cal - np.max(train_logits_cal, axis=-1, keepdims=True))
    train_probs_cal = train_probs_cal / np.sum(train_probs_cal, axis=-1, keepdims=True)

    # Test
    test_probs_raw = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
    test_probs_raw = test_probs_raw / np.sum(test_probs_raw, axis=-1, keepdims=True)

    test_logits_cal = test_logits - null_logits
    test_probs_cal = np.exp(test_logits_cal - np.max(test_logits_cal, axis=-1, keepdims=True))
    test_probs_cal = test_probs_cal / np.sum(test_probs_cal, axis=-1, keepdims=True)

    out_file = Path("data/teacher_targets/teacher_targets_nli_gemma3_4b.npz")
    np.savez_compressed(
        out_file,
        train_golds=np.array(train_golds),
        train_logits_raw=train_logits,
        train_probs_raw=train_probs_raw,
        train_logits_cal=train_logits_cal,
        train_probs_cal=train_probs_cal,
        test_golds=np.array(test_golds),
        test_logits_raw=test_logits,
        test_probs_raw=test_probs_raw,
        test_logits_cal=test_logits_cal,
        test_probs_cal=test_probs_cal,
        null_logits=null_logits,
        verbalizers=VERBALIZERS,
        class_names=CLASS_NAMES
    )
    print(f"[✓] Saved cached targets to {out_file}", flush=True)

if __name__ == "__main__":
    main()

