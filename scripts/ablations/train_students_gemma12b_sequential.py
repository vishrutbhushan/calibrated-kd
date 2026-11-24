"""
train_students_gemma12b_sequential.py
-------------------------------------
Sequentially trains Student models (RoBERTa-base) on all 6 naturally unbalanced
benchmarks using pre-extracted Gemma-3-12B-IT (4-bit NF4) teacher targets.

Paradigms evaluated per dataset:
  1. Standard KD (Raw softmax teacher targets)
  2. Batch-Calibrated KD (Batch-mean prior subtraction / Zhou et al., ICLR 2024)
  3. Null-Calibrated KD (CKD, Ours) (Single 'N/A' probe subtraction)

Guarantees:
  - Strictly SEQUENTIAL: one dataset and one student model at a time.
  - VRAM is flushed after every single training run (< 0.1 GB idle).
  - Clean real-time logging of Macro-F1 and Accuracy.
"""

import os
import sys
import gc
import csv
import json
import random
import time
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
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
RESULTS_DIR = ROOT / "outputs" / "ablations" / "6_gemma12b_scaling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_MODEL_NAME = "roberta-base"

BENCHMARK_CONFIGS = {
    "student_questions": {
        "num_classes": 4,
        "class_names": ["Biology", "Physics", "Maths", "Chemistry"]
    },
    "trec6": {
        "num_classes": 6,
        "class_names": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]
    },
    "app_reviews": {
        "num_classes": 5,
        "class_names": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
    },
    "twitter_emotion": {
        "num_classes": 6,
        "class_names": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    },
    "tweeteval_emotion": {
        "num_classes": 4,
        "class_names": ["Anger", "Joy", "Optimism", "Sadness"]
    },
    "bbc_news": {
        "num_classes": 5,
        "class_names": ["Tech", "Business", "Sport", "Entertainment", "Politics"]
    }
}

DATASET_ORDER = [
    "student_questions",
    "trec6",
    "app_reviews",
    "twitter_emotion",
    "tweeteval_emotion",
    "bbc_news"
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def compute_metrics(golds, preds, num_classes):
    acc = float(accuracy_score(golds, preds) * 100)
    macro_f1 = float(f1_score(golds, preds, average="macro") * 100)
    per_class = (f1_score(golds, preds, average=None) * 100).tolist()
    worst_f1 = float(np.min(per_class))
    counts = np.bincount(preds, minlength=num_classes).tolist()
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "worst_f1": worst_f1,
        "per_class_f1": per_class,
        "counts": counts
    }

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

def load_data(task_name):
    task_dir = DATA_BASE / task_name
    train_file = task_dir / "train_1200.csv" if (task_dir / "train_1200.csv").exists() else task_dir / "train_1000.csv"
    test_file = task_dir / "test_400.csv" if (task_dir / "test_400.csv").exists() else task_dir / "test_350.csv"

    train_texts, train_golds = [], []
    with open(train_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_texts.append(row["text"])
            train_golds.append(int(row["label"]))

    test_texts, test_golds = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_texts.append(row["text"])
            test_golds.append(int(row["label"]))

    return train_texts, train_golds, test_texts, test_golds

def train_student_variant(method_name, train_texts, train_probs, test_texts, test_golds, num_classes, seed=42):
    set_seed(seed)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    max_len = 256

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["probs"] = p
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=num_classes, local_files_only=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(KDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate_fn)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
        print(f"    [{method_name}] Epoch {ep+1}/{epochs} loss: {ep_loss/len(loader):.4f}", flush=True)

    # Test evaluation
    model.eval()
    test_logits = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            chunk = test_texts[i : i + 32]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            test_logits.append(out.logits.cpu().numpy())
            del enc, out
    test_logits = np.concatenate(test_logits, axis=0)

    # Clean up model & optimizer
    del model, optimizer, scheduler, loader, tokenizer
    clear_vram()

    elapsed = time.time() - t0
    preds = test_logits.argmax(axis=-1)
    metrics = compute_metrics(test_golds, preds, num_classes)
    metrics["train_time_sec"] = round(elapsed, 1)
    return metrics

def run_dataset_kd(task_name):
    cfg = BENCHMARK_CONFIGS[task_name]
    num_classes = cfg["num_classes"]

    print("\n" + "="*75)
    print(f"BENCHMARK KD: {task_name.upper()} (K={num_classes})")
    print("="*75)

    train_texts, train_golds, test_texts, test_golds = load_data(task_name)
    print(f"Loaded: {len(train_texts)} train items, {len(test_texts)} test items.")

    cache_path = DATA_BASE / task_name / "gemma12b_pytorch_teacher_targets.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached targets: {cache_path}")

    targets = np.load(cache_path)
    null_logits = targets["null_logits"]
    test_logits = targets["test_logits"]
    train_logits = targets["train_logits"]
    raw_train_probs = targets["raw_train_probs"]
    cal_train_probs = targets["cal_train_probs"]
    batch_train_probs = targets["batch_train_probs"]

    # Teacher Zero-Shot Baselines
    tea_raw_preds = test_logits.argmax(axis=-1)
    tea_raw_metrics = compute_metrics(test_golds, tea_raw_preds, num_classes)

    tea_cal_logits = test_logits - null_logits
    tea_cal_preds = tea_cal_logits.argmax(axis=-1)
    tea_cal_metrics = compute_metrics(test_golds, tea_cal_preds, num_classes)

    print(f"Teacher 12B Raw Zero-Shot:       Macro-F1: {tea_raw_metrics['macro_f1']:.2f}% | Acc: {tea_raw_metrics['acc']:.2f}% | Counts: {tea_raw_metrics['counts']}")
    print(f"Teacher 12B Null-Calibrated:     Macro-F1: {tea_cal_metrics['macro_f1']:.2f}% | Acc: {tea_cal_metrics['acc']:.2f}% | Counts: {tea_cal_metrics['counts']}")

    # 1. Standard KD
    print("\n  >> Training Student 1/3: Standard KD (Raw Softmax Targets)...", flush=True)
    std_metrics = train_student_variant("Standard KD", train_texts, raw_train_probs, test_texts, test_golds, num_classes)
    print(f"  [✓] Standard KD:         Macro-F1: {std_metrics['macro_f1']:.2f}% (Acc: {std_metrics['acc']:.2f}%) | Counts: {std_metrics['counts']} | ({std_metrics['train_time_sec']}s)")

    # 2. Batch-Calibrated KD
    print("\n  >> Training Student 2/3: Batch-Calibrated KD (Zhou et al., ICLR 2024)...", flush=True)
    batch_metrics = train_student_variant("Batch-Cal KD", train_texts, batch_train_probs, test_texts, test_golds, num_classes)
    print(f"  [✓] Batch-Calibrated KD: Macro-F1: {batch_metrics['macro_f1']:.2f}% (Acc: {batch_metrics['acc']:.2f}%) | Counts: {batch_metrics['counts']} | ({batch_metrics['train_time_sec']}s)")

    # 3. Null-Calibrated KD (CKD, Ours)
    print("\n  >> Training Student 3/3: Null-Calibrated KD (CKD, Ours)...", flush=True)
    ckd_metrics = train_student_variant("CKD (Ours)", train_texts, cal_train_probs, test_texts, test_golds, num_classes)
    print(f"  [✓] Null-Cal KD (Ours):  Macro-F1: {ckd_metrics['macro_f1']:.2f}% (Acc: {ckd_metrics['acc']:.2f}%) | Counts: {ckd_metrics['counts']} | ({ckd_metrics['train_time_sec']}s)")

    result = {
        "dataset": task_name,
        "teacher_model": "gemma-3-12b-it-4bit-nf4",
        "student_model": "roberta-base",
        "num_classes": num_classes,
        "class_names": cfg["class_names"],
        "null_logits": null_logits.tolist(),
        "teacher_raw": tea_raw_metrics,
        "teacher_calibrated": tea_cal_metrics,
        "standard_kd": std_metrics,
        "batch_calibrated_kd": batch_metrics,
        "calibrated_kd": ckd_metrics
    }

    out_file = RESULTS_DIR / f"{task_name}_gemma12b_kd_results.json"
    with open(out_file, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"  Saved per-dataset result: {out_file}")

    return result

def main():
    print("#"*75)
    print("STARTING SEQUENTIAL STUDENT KD EXPERIMENTS (GEMMA-3-12B TEACHER -> ROBERTA-BASE)")
    print("#"*75)

    all_results = {}
    total_start = time.time()

    for idx, d in enumerate(DATASET_ORDER, 1):
        print(f"\n[{idx}/6] Processing benchmark dataset: {d} ...")
        res = run_dataset_kd(d)
        all_results[d] = res
        clear_vram()

    summary_file = RESULTS_DIR / "all_6_gemma12b_kd_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(all_results, fp, indent=2)

    total_time = time.time() - total_start
    print("\n" + "="*75)
    print(f"ALL 6 DATASETS COMPLETED IN {total_time/60:.2f} MINUTES! Summary: {summary_file}")
    print("="*75)

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Dataset':<20} | {'Teacher Raw':<14} | {'Teacher Cal':<14} | {'Std KD':<14} | {'Batch-Cal KD':<14} | {'CKD (Ours)':<14}")
    print("-" * 85)
    for d in DATASET_ORDER:
        r = all_results[d]
        tr = f"{r['teacher_raw']['macro_f1']:.1f}% ({r['teacher_raw']['acc']:.1f}%)"
        tc = f"{r['teacher_calibrated']['macro_f1']:.1f}% ({r['teacher_calibrated']['acc']:.1f}%)"
        sk = f"{r['standard_kd']['macro_f1']:.1f}% ({r['standard_kd']['acc']:.1f}%)"
        bk = f"{r['batch_calibrated_kd']['macro_f1']:.1f}% ({r['batch_calibrated_kd']['acc']:.1f}%)"
        ck = f"{r['calibrated_kd']['macro_f1']:.1f}% ({r['calibrated_kd']['acc']:.1f}%)"
        print(f"{d:<20} | {tr:<14} | {tc:<14} | {sk:<14} | {bk:<14} | {ck:<14}")
    print("="*85)

if __name__ == "__main__":
    main()
