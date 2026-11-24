"""
scripts/train/train_cross_family_qwen_roberta_sst2.py
----------------------------------------------------
Ablation 1 (Config 2): Cross-Family Distillation on SST-2 (Balanced Binary Sentiment):
  Teacher: Qwen/Qwen3-4B (4B Causal Decoder)
  Student: roberta-base (125M Masked Bidirectional Encoder, Full Fine-Tuning)

Uses correct leading-space verbalizers [' A', ' B'].
Evaluates:
  1) Teacher Raw Zero-Shot
  2) Teacher Null-Calibrated Zero-Shot
  3) Student Standard KD
  4) Student Calibrated KD (CKD)

Saves to results/ablations/1_cross_architecture/cross_family_qwen_to_roberta_sst2.json
"""

import os
import csv
import json
import random
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
SST2_DIR = ROOT / "data" / "datasets" / "sst2"
OUT_DIR = ROOT / "outputs" / "ablations" / "1_cross_architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_PATH = "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
STUDENT_NAME = "roberta-base"

PROMPT_TEMPLATE = (
    "Sentence: {text}\n"
    "Question: Is the sentiment of this sentence Negative or Positive?\n"
    "A: Negative\n"
    "B: Positive\n"
    "Answer:"
)
LETTERS = ["A", "B"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_qwen_targets():
    print(f"\n[Step 1/3] Loading Teacher Qwen3-4B from {TEACHER_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use leading-space verbalizers
    spaced_verbalizer_ids = [tokenizer.encode(" " + l, add_special_tokens=False)[-1] for l in LETTERS]
    print(f"Spaced verbalizer IDs for [' A', ' B']: {list(zip(LETTERS, spaced_verbalizer_ids))}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True
    )
    model.eval()

    # 1. Null probe
    null_prompt = PROMPT_TEMPLATE.format(text="N/A")
    inputs = tokenizer(null_prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model(**inputs)
        null_logits = out.logits[0, -1, spaced_verbalizer_ids].float().cpu().numpy()
    null_gap = float(abs(null_logits[0] - null_logits[1]))
    print(f"Qwen3-4B SST-2 Null Logits [A, B]: {null_logits.tolist()} | Gap: {null_gap:.2f} logits", flush=True)

    # Load data
    with open(SST2_DIR / "sst2_train_1000.csv") as f:
        train_rows = list(csv.DictReader(f))
    with open(SST2_DIR / "sst2_test_200.csv") as f:
        test_rows = list(csv.DictReader(f))

    train_texts = [r["text"] for r in train_rows]
    train_golds = [int(r["label"]) for r in train_rows]
    test_texts = [r["text"] for r in test_rows]
    test_golds = [int(r["label"]) for r in test_rows]

    def get_logits(texts):
        all_logits = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            prompts = [PROMPT_TEMPLATE.format(text=t.strip()[:500]) for t in chunk]
            enc = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model(**enc)
                logits = out.logits[:, -1, spaced_verbalizer_ids].float().cpu().numpy()
                all_logits.append(logits)
        return np.vstack(all_logits)

    print("Extracting train logits...", flush=True)
    train_logits = get_logits(train_texts)
    print("Extracting test logits...", flush=True)
    test_logits = get_logits(test_texts)

    # Unload Qwen teacher
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Teacher Qwen3-4B unloaded from GPU.", flush=True)

    return {
        "null_logits": null_logits,
        "train_texts": train_texts,
        "train_logits": train_logits,
        "test_texts": test_texts,
        "test_logits": test_logits,
        "test_golds": test_golds
    }

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

def train_roberta_student(train_texts, train_probs, test_texts, test_golds, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_NAME, num_labels=2, local_files_only=True
    ).to(device)

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    dataset = KDDataset(train_texts, train_probs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(epochs):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    all_preds = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_texts), bs):
            chunk = test_texts[i:i+bs]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**enc)
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

    del model, optimizer, scheduler, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    acc = float(accuracy_score(test_golds, all_preds) * 100)
    macro_f1 = float(f1_score(test_golds, all_preds, average="macro") * 100)
    per_class = f1_score(test_golds, all_preds, average=None) * 100
    worst_f1 = float(np.min(per_class))
    counts = np.bincount(all_preds, minlength=2).tolist()

    return {"acc": acc, "macro_f1": macro_f1, "worst_f1": worst_f1, "counts": counts}

def main():
    set_seed(42)
    data = extract_qwen_targets()

    null_logits = data["null_logits"]
    train_texts = data["train_texts"]
    train_logits = data["train_logits"]
    test_texts = data["test_texts"]
    test_logits = data["test_logits"]
    test_golds = data["test_golds"]

    # Teacher raw
    t_raw_preds = np.argmax(test_logits, axis=-1)
    t_raw_acc = float(accuracy_score(test_golds, t_raw_preds) * 100)
    t_raw_f1 = float(f1_score(test_golds, t_raw_preds, average="macro") * 100)
    print(f"\nTeacher Qwen3-4B Raw Zero-Shot: Acc: {t_raw_acc:.2f}% | F1: {t_raw_f1:.2f}% | Counts: {np.bincount(t_raw_preds, minlength=2).tolist()}", flush=True)

    # Teacher calibrated
    t_cal_preds = np.argmax(test_logits - null_logits, axis=-1)
    t_cal_acc = float(accuracy_score(test_golds, t_cal_preds) * 100)
    t_cal_f1 = float(f1_score(test_golds, t_cal_preds, average="macro") * 100)
    print(f"Teacher Qwen3-4B Null-Calibrated: Acc: {t_cal_acc:.2f}% | F1: {t_cal_f1:.2f}% | Counts: {np.bincount(t_cal_preds, minlength=2).tolist()}", flush=True)

    # Compute training probabilities
    # 1. Standard KD (raw softmax)
    exp_raw = np.exp(train_logits - train_logits.max(axis=-1, keepdims=True))
    probs_raw = exp_raw / exp_raw.sum(axis=-1, keepdims=True)

    # 2. Calibrated KD (z - b_null)
    cal_logits = train_logits - null_logits
    exp_cal = np.exp(cal_logits - cal_logits.max(axis=-1, keepdims=True))
    probs_ckd = exp_cal / exp_cal.sum(axis=-1, keepdims=True)

    print("\n[Step 2/3] Training RoBERTa-base with Standard KD (Qwen3-4B Raw Targets)...", flush=True)
    std_res = train_roberta_student(train_texts, probs_raw, test_texts, test_golds)
    print(f"Standard KD -> Acc: {std_res['acc']:.2f}% | F1: {std_res['macro_f1']:.2f}% [Worst: {std_res['worst_f1']:.2f}%] | Counts: {std_res['counts']}", flush=True)

    print("\n[Step 3/3] Training RoBERTa-base with Calibrated KD (Qwen3-4B CKD Targets)...", flush=True)
    ckd_res = train_roberta_student(train_texts, probs_ckd, test_texts, test_golds)
    print(f"Calibrated KD -> Acc: {ckd_res['acc']:.2f}% | F1: {ckd_res['macro_f1']:.2f}% [Worst: {ckd_res['worst_f1']:.2f}%] | Counts: {ckd_res['counts']}", flush=True)

    out = {
        "teacher": "Qwen/Qwen3-4B",
        "student": "roberta-base",
        "dataset": "SST-2 (Balanced Binary Sentiment)",
        "null_logits": null_logits.tolist(),
        "teacher_raw": {"acc": t_raw_acc, "f1": t_raw_f1},
        "teacher_calibrated": {"acc": t_cal_acc, "f1": t_cal_f1},
        "standard_kd": std_res,
        "calibrated_kd": ckd_res,
        "delta_macro_f1": ckd_res["macro_f1"] - std_res["macro_f1"]
    }

    out_path = OUT_DIR / "cross_family_qwen_to_roberta_sst2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
