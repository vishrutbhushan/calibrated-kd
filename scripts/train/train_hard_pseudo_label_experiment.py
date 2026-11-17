#!/usr/bin/env python3
"""
scripts/train/train_hard_pseudo_label_experiment.py
---------------------------------------------------
Compares:
1. Hard Label Distillation (Raw Teacher Top-1 Argmax Pseudo-Label) with Cross-Entropy Loss
2. Hard Label Distillation (Calibrated Teacher Top-1 Argmax Pseudo-Label) with Cross-Entropy Loss
3. Soft Label Distillation (Standard KD) with KL Loss
4. Soft Label Distillation (Calibrated KD / CKD) with KL Loss

Dataset: AG News Skewed (1,000 train, 1,000 test)
Student: RoBERTa-base (125M Encoder)
Teacher: Gemma-3-4B-IT
"""

import os
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TARGETS_NPZ = DATA_DIR / "teacher_targets" / "skewed" / "teacher_targets_agnews_skewed_1000_gemma3_4b.npz"
TRAIN_CSV = DATA_DIR / "benchmarks" / "skewed" / "agnews_skewed_train_1000.csv"
TEST_CSV = DATA_DIR / "benchmarks" / "ag_news_test_1000.csv"
OUT_DIR = ROOT / "results" / "hard_pseudo_label_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-base"
NUM_CLASSES = 4
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HardDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

class SoftDataset(Dataset):
    def __init__(self, texts, probs, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.probs = torch.tensor(probs, dtype=torch.float32)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "probs": self.probs[idx]
        }

def get_eval_logits(model, tokenizer, texts, device, batch_size=32):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i+batch_size]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**enc)
            all_logits.append(out.logits.cpu().numpy())
    return np.vstack(all_logits)

def train_hard_model(train_texts, train_hard_labels, test_texts, test_golds, device, tokenizer):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(device)
    dataset = HardDataset(train_texts, train_hard_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(3):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = criterion(out.logits, b["labels"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    test_logits = get_eval_logits(model, tokenizer, test_texts, device)
    preds = test_logits.argmax(axis=-1)
    del model
    torch.cuda.empty_cache()
    
    acc = float(accuracy_score(test_golds, preds) * 100)
    f1 = float(f1_score(test_golds, preds, average="macro") * 100)
    counts = np.bincount(preds, minlength=NUM_CLASSES).tolist()
    return {"acc": acc, "f1": f1, "counts": counts}

def train_soft_model(train_texts, train_probs, test_texts, test_golds, device, tokenizer):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(device)
    dataset = SoftDataset(train_texts, train_probs, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    model.train()
    for epoch in range(3):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            log_p = torch.log_softmax(out.logits, dim=-1)
            loss = criterion(log_p, b["probs"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    test_logits = get_eval_logits(model, tokenizer, test_texts, device)
    preds = test_logits.argmax(axis=-1)
    del model
    torch.cuda.empty_cache()
    
    acc = float(accuracy_score(test_golds, preds) * 100)
    f1 = float(f1_score(test_golds, preds, average="macro") * 100)
    counts = np.bincount(preds, minlength=NUM_CLASSES).tolist()
    return {"acc": acc, "f1": f1, "counts": counts}

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Load targets
    npz = np.load(TARGETS_NPZ)
    train_logits_raw = npz["train_logits_raw"]
    train_logits_ckd = npz["train_logits_ckd"]
    train_probs_raw  = npz["train_probs_raw"]
    train_probs_ckd  = npz["train_probs_ckd"]
    
    # Argmax hard pseudo-labels
    raw_hard_labels = np.argmax(train_logits_raw, axis=-1)
    ckd_hard_labels = np.argmax(train_logits_ckd, axis=-1)
    
    # Load texts
    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        train_texts = [r["text"] for r in reader]
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_rows = list(reader)
        test_texts = [r["text"] for r in test_rows]
        test_golds = [int(r["label"]) for r in test_rows]
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("--- 1. Training on Raw Teacher Hard Pseudo-Labels (Argmax, Cross-Entropy) ---", flush=True)
    raw_hard_res = train_hard_model(train_texts, raw_hard_labels, test_texts, test_golds, device, tokenizer)
    print(f"Raw Hard Pseudo-Labels -> Acc: {raw_hard_res['acc']:.2f}% | Macro-F1: {raw_hard_res['f1']:.2f}% | Counts: {raw_hard_res['counts']}", flush=True)

    print("--- 2. Training on Calibrated Teacher Hard Pseudo-Labels (CKD Argmax, Cross-Entropy) ---", flush=True)
    ckd_hard_res = train_hard_model(train_texts, ckd_hard_labels, test_texts, test_golds, device, tokenizer)
    print(f"CKD Hard Pseudo-Labels -> Acc: {ckd_hard_res['acc']:.2f}% | Macro-F1: {ckd_hard_res['f1']:.2f}% | Counts: {ckd_hard_res['counts']}", flush=True)

    print("--- 3. Training on Raw Teacher Soft Probs (Standard KD, KL-Divergence) ---", flush=True)
    raw_soft_res = train_soft_model(train_texts, train_probs_raw, test_texts, test_golds, device, tokenizer)
    print(f"Standard Soft KD -> Acc: {raw_soft_res['acc']:.2f}% | Macro-F1: {raw_soft_res['f1']:.2f}% | Counts: {raw_soft_res['counts']}", flush=True)

    print("--- 4. Training on Calibrated Teacher Soft Probs (Calibrated KD / CKD, KL-Divergence) ---", flush=True)
    ckd_soft_res = train_soft_model(train_texts, train_probs_ckd, test_texts, test_golds, device, tokenizer)
    print(f"Calibrated Soft KD -> Acc: {ckd_soft_res['acc']:.2f}% | Macro-F1: {ckd_soft_res['f1']:.2f}% | Counts: {ckd_soft_res['counts']}", flush=True)

    results = {
        "raw_hard_pseudo_labels": raw_hard_res,
        "ckd_hard_pseudo_labels": ckd_hard_res,
        "standard_soft_kd": raw_soft_res,
        "calibrated_soft_kd": ckd_soft_res,
        "pseudo_label_counts_train": {
            "raw_teacher_argmax": np.bincount(raw_hard_labels, minlength=NUM_CLASSES).tolist(),
            "ckd_teacher_argmax": np.bincount(ckd_hard_labels, minlength=NUM_CLASSES).tolist()
        }
    }
    
    out_file = OUT_DIR / "hard_vs_soft_pseudo_label_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll finished! Saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
