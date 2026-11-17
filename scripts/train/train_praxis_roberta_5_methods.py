"""
scripts/train/train_praxis_roberta_5_methods.py
----------------------------------------------
Evaluates RoBERTa-base (125M) on PRAXIS English (5-class pragmatic intent, balanced):
  1) Teacher Raw Zero-Shot
  2) Teacher Null-Calibrated Zero-Shot
  3) Student Base (Zero-Shot RoBERTa untrained head)
  4) Student Hard Pseudo-Labels (Argmax, Cross-Entropy)
  5) Student Standard KD (Raw soft targets, KL-Div)
  6) Student Post-Hoc Output Calibration (z^S - E[z^S])
  7) Student Batch-Calibrated KD (z^T - log p_batch)
  8) Student Calibrated KD / CKD (z^T - b_null)
  9) Student Oracle Prior CKD (identical to CKD in balanced setting)

Saves results to results/main_results/praxis/praxis_roberta_5_methods.json
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
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
PRAXIS_DIR = ROOT / "data" / "datasets" / "praxis"
OUT_DIR = ROOT / "results" / "main_results" / "praxis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-base"
CANONICAL_CLASSES = ["direct_imperative", "modal_deontic", "optative", "conditional", "interrogative"]
NUM_CLASSES = len(CANONICAL_CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SoftKDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

class HardDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def evaluate_model(model, test_texts, tokenizer, device):
    model.eval()
    all_logits = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_texts), bs):
            chunk = test_texts[i:i+bs]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**enc)
            all_logits.append(out.logits.detach().cpu().numpy())
    return np.vstack(all_logits)

def train_soft_kd(train_texts, train_probs, test_texts, test_golds, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, local_files_only=True
    ).to(device)

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    dataset = SoftKDDataset(train_texts, train_probs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["probs"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            log_p_student = torch.log_softmax(out.logits, dim=-1)
            loss = kl_loss_fn(log_p_student, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    test_logits = evaluate_model(model, test_texts, tokenizer, device)
    
    # Clean memory
    del model, optimizer, scheduler, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_logits

def train_hard_ce(train_texts, train_labels, test_texts, test_golds, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, local_files_only=True
    ).to(device)

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        labs = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["labels"] = labs
        return enc

    dataset = HardDataset(train_texts, train_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    test_logits = evaluate_model(model, test_texts, tokenizer, device)

    del model, optimizer, scheduler, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return test_logits

def main():
    set_seed(42)
    print("=======================================================", flush=True)
    print("  PRAXIS ENGLISH (5-Class Balanced) - RoBERTa-base 5 Methods", flush=True)
    print("=======================================================", flush=True)

    # Load data
    with open(PRAXIS_DIR / "praxis_train_1000.csv") as f:
        train_rows = list(csv.DictReader(f))
    with open(PRAXIS_DIR / "praxis_test_150.csv") as f:
        test_rows = list(csv.DictReader(f))

    train_texts = [r["review_en"] for r in train_rows]
    test_texts = [r["review_en"] for r in test_rows]
    test_golds = [CLASS2IDX[r["tier1_form"]] for r in test_rows]

    # Load teacher targets
    npz = np.load(PRAXIS_DIR / "teacher_targets_praxis_1000_gemma3_4b.npz")
    # First 150 are test, last 1000 are train
    test_logits_teacher = npz["logits_en"][:150]
    train_logits_raw = npz["logits_en"][150:]
    train_probs_raw = npz["probs_en"][150:]

    with open(ROOT / "data" / "teacher_targets" / "null_vectors_gemma3_4b.json") as f:
        null_vecs = json.load(f)
    b_null = np.array(null_vecs["en"]["logits"])
    print(f"Teacher Null Logits: {b_null.tolist()}")
    print(f"Null gap (max - min): {b_null.max() - b_null.min():.2f} logits\n")

    summary = {}

    def compute_metrics(preds):
        acc = float(accuracy_score(test_golds, preds) * 100)
        macro_f1 = float(f1_score(test_golds, preds, average="macro") * 100)
        per_class = f1_score(test_golds, preds, average=None) * 100
        worst_f1 = float(np.min(per_class))
        counts = np.bincount(preds, minlength=NUM_CLASSES).tolist()
        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "worst_f1": worst_f1,
            "per_class_f1": [float(x) for x in per_class],
            "counts": counts
        }

    # a. Teacher Raw Zero-Shot
    t_raw_preds = np.argmax(test_logits_teacher, axis=-1)
    summary["teacher_raw"] = compute_metrics(t_raw_preds)
    print(f"a. Teacher (Raw Zero-Shot): Acc: {summary['teacher_raw']['acc']:.2f}% | F1: {summary['teacher_raw']['macro_f1']:.2f}% [Worst: {summary['teacher_raw']['worst_f1']:.2f}%] | Counts: {summary['teacher_raw']['counts']}")

    # b. Teacher Null-Calibrated
    t_cal_preds = np.argmax(test_logits_teacher - b_null, axis=-1)
    summary["teacher_calibrated"] = compute_metrics(t_cal_preds)
    print(f"b. Teacher (Null-Calibrated): Acc: {summary['teacher_calibrated']['acc']:.2f}% | F1: {summary['teacher_calibrated']['macro_f1']:.2f}% [Worst: {summary['teacher_calibrated']['worst_f1']:.2f}%] | Counts: {summary['teacher_calibrated']['counts']}")

    # c. Student Base (Zero-Shot RoBERTa untrained classification head)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, local_files_only=True
    ).to(device)
    base_logits = evaluate_model(base_model, test_texts, tokenizer, device)
    base_preds = base_logits.argmax(axis=-1)
    summary["student_base_zero_shot"] = compute_metrics(base_preds)
    print(f"c. Student Base (Zero-Shot): Acc: {summary['student_base_zero_shot']['acc']:.2f}% | F1: {summary['student_base_zero_shot']['macro_f1']:.2f}% [Worst: {summary['student_base_zero_shot']['worst_f1']:.2f}%] | Counts: {summary['student_base_zero_shot']['counts']}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # d. Student Hard Pseudo-Labels (Argmax)
    print("\nTraining d. Student: Hard Pseudo-Labels (Argmax)...")
    hard_labels = np.argmax(train_logits_raw, axis=-1)
    d_logits = train_hard_ce(train_texts, hard_labels, test_texts, test_golds)
    d_preds = d_logits.argmax(axis=-1)
    summary["hard_pseudo_labels"] = compute_metrics(d_preds)
    print(f"d. Student Hard Pseudo-Labels: Acc: {summary['hard_pseudo_labels']['acc']:.2f}% | F1: {summary['hard_pseudo_labels']['macro_f1']:.2f}% [Worst: {summary['hard_pseudo_labels']['worst_f1']:.2f}%] | Counts: {summary['hard_pseudo_labels']['counts']}")

    # e. Student Standard KD (Raw Soft Targets)
    print("\nTraining e. Student: Standard KD (Raw Soft)...")
    e_logits = train_soft_kd(train_texts, train_probs_raw, test_texts, test_golds)
    e_preds = e_logits.argmax(axis=-1)
    summary["standard_kd"] = compute_metrics(e_preds)
    print(f"e. Student Standard KD: Acc: {summary['standard_kd']['acc']:.2f}% | F1: {summary['standard_kd']['macro_f1']:.2f}% [Worst: {summary['standard_kd']['worst_f1']:.2f}%] | Counts: {summary['standard_kd']['counts']}")

    # f. Student Post-Hoc Output Calibration (applied to Standard KD logits)
    print("\nEvaluating f. Student: Post-Hoc Output Calibration...")
    post_logits = e_logits - e_logits.mean(axis=0, keepdims=True)
    post_preds = post_logits.argmax(axis=-1)
    summary["post_hoc_output_cal"] = compute_metrics(post_preds)
    print(f"f. Student Post-Hoc Output Cal: Acc: {summary['post_hoc_output_cal']['acc']:.2f}% | F1: {summary['post_hoc_output_cal']['macro_f1']:.2f}% [Worst: {summary['post_hoc_output_cal']['worst_f1']:.2f}%] | Counts: {summary['post_hoc_output_cal']['counts']}")

    # g. Student Batch-Calibrated KD
    print("\nTraining g. Student: Batch-Calibrated KD...")
    p_batch = train_probs_raw.mean(axis=0, keepdims=True)
    p_batch = np.clip(p_batch, 1e-12, 1.0)
    p_batch /= p_batch.sum(axis=-1, keepdims=True)
    logits_batch_cal = train_logits_raw - np.log(p_batch)
    logits_batch_cal -= logits_batch_cal.max(axis=-1, keepdims=True)
    exp_batch = np.exp(logits_batch_cal)
    train_probs_batch = exp_batch / exp_batch.sum(axis=-1, keepdims=True)

    g_logits = train_soft_kd(train_texts, train_probs_batch, test_texts, test_golds)
    g_preds = g_logits.argmax(axis=-1)
    summary["batch_cal_kd"] = compute_metrics(g_preds)
    print(f"g. Student Batch-Calibrated KD: Acc: {summary['batch_cal_kd']['acc']:.2f}% | F1: {summary['batch_cal_kd']['macro_f1']:.2f}% [Worst: {summary['batch_cal_kd']['worst_f1']:.2f}%] | Counts: {summary['batch_cal_kd']['counts']}")

    # h. Student Calibrated KD (CKD, Ours)
    print("\nTraining h. Student: Calibrated KD (CKD)...")
    logits_ckd = train_logits_raw - b_null
    logits_ckd -= logits_ckd.max(axis=-1, keepdims=True)
    exp_ckd = np.exp(logits_ckd)
    train_probs_ckd = exp_ckd / exp_ckd.sum(axis=-1, keepdims=True)

    h_logits = train_soft_kd(train_texts, train_probs_ckd, test_texts, test_golds)
    h_preds = h_logits.argmax(axis=-1)
    summary["calibrated_kd"] = compute_metrics(h_preds)
    print(f"h. Student Calibrated KD (CKD): Acc: {summary['calibrated_kd']['acc']:.2f}% | F1: {summary['calibrated_kd']['macro_f1']:.2f}% [Worst: {summary['calibrated_kd']['worst_f1']:.2f}%] | Counts: {summary['calibrated_kd']['counts']}")

    # i. Student Oracle Prior CKD
    print("\nEvaluating i. Student: Oracle Prior CKD (identical to CKD in balanced setting)...")
    summary["oracle_prior_ckd"] = summary["calibrated_kd"].copy()

    # Save to disk
    out_file = OUT_DIR / "praxis_roberta_5_methods.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {out_file}")

if __name__ == "__main__":
    main()
