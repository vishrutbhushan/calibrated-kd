"""
train_roberta_benchmarks.py
---------------------------
Distills Gemma-3-4B-it teacher targets into a lightweight encoder student (roberta-base, 125M)
across all 4 canonical benchmarks:
  1) AG News (4 classes)
  2) Emotion (6 classes)
  3) TREC-6 (6 classes)
  4) SST-2 (2 classes - Control)

For each benchmark, evaluates both:
  - Standard KD (Raw teacher soft targets)
  - Calibrated KD (Static CKD targets: z - W_null)

Saves full metrics and per-class prediction distributions to results/roberta/.
"""

import os
import csv
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "roberta"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-base"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TextKDDataset(Dataset):
    def __init__(self, texts, probs, golds, tokenizer, max_length=256):
        self.texts = texts
        self.probs = probs
        self.golds = golds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "prob": torch.tensor(self.probs[idx], dtype=torch.float32),
            "gold": int(self.golds[idx])
        }

def collate_fn(batch, tokenizer, max_length=256):
    texts = [b["text"] for b in batch]
    probs = torch.stack([b["prob"] for b in batch])
    golds = torch.tensor([b["gold"] for b in batch], dtype=torch.long)
    inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {
        "input_ids": inp["input_ids"],
        "attention_mask": inp["attention_mask"],
        "probs": probs,
        "golds": golds
    }

def evaluate_test(model, tokenizer, test_texts, test_golds, num_classes, device, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            b_texts = test_texts[i : i + batch_size]
            inp = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model(**inp)
            top = torch.argmax(out.logits, dim=-1).cpu().tolist()
            preds.extend(top)

    acc = float(accuracy_score(test_golds, preds) * 100)
    f1 = float(f1_score(test_golds, preds, average="macro") * 100)
    pred_counts = [int(np.sum(np.array(preds) == c)) for c in range(num_classes)]
    return acc, f1, pred_counts

def train_and_eval(dataset_name, mode, train_texts, train_probs, train_golds,
                   test_texts, test_golds, num_classes, class_names,
                   epochs=3, batch_size=16, lr=2e-5, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[{dataset_name}] Starting {mode.upper()} Distillation (Epochs={epochs}, LR={lr}, Batch={batch_size})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        local_files_only=True
    ).to(device)

    dataset = TextKDDataset(train_texts, train_probs, train_golds, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["probs"].to(device) # [B, K]

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits # [B, K]
            log_probs = torch.log_softmax(logits, dim=-1)

            loss = kl_loss_fn(log_probs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        acc, f1, counts = evaluate_test(model, tokenizer, test_texts, test_golds, num_classes, device)
        print(f"  Epoch {ep}/{epochs} | KD Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}% | Test F1: {f1:.2f}% | Counts: {counts}", flush=True)

    result = {
        "dataset": dataset_name,
        "mode": mode,
        "student": MODEL_NAME,
        "epochs": epochs,
        "test_acc": acc,
        "test_f1": f1,
        "pred_counts": counts,
        "class_names": class_names
    }
    out_file = RESULTS_DIR / f"{dataset_name.lower().replace('-', '')}_{mode}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[✓] Saved {out_file}", flush=True)
    return result

def load_data(dataset_key):
    benchmarks_dir = DATA_DIR / "benchmarks"
    targets_dir = DATA_DIR / "teacher_targets"

    if dataset_key == "agnews":
        npz = np.load(targets_dir / "teacher_targets_agnews_1000_gemma3_4b.npz")
        raw_probs = npz["train_probs_raw"]
        cal_probs = npz["train_probs_static"]
        with open(benchmarks_dir / "ag_news_train_1000.csv") as f:
            tr_rows = list(csv.DictReader(f))
        with open(benchmarks_dir / "ag_news_test_1000.csv") as f:
            te_rows = list(csv.DictReader(f))
        classes = ["World", "Sports", "Business", "Sci/Tech"]
        return "AGNews", [r["text"] for r in tr_rows], raw_probs, cal_probs, [int(r["label"]) for r in tr_rows], [r["text"] for r in te_rows], [int(r["label"]) for r in te_rows], 4, classes

    elif dataset_key == "emotion":
        npz = np.load(targets_dir / "teacher_targets_emotion_1200_gemma3_4b.npz")
        raw_probs = npz["train_probs_raw"]
        cal_probs = npz["train_probs_cal"]
        with open(benchmarks_dir / "emotion_train_1200.csv") as f:
            tr_rows = list(csv.DictReader(f))
        with open(benchmarks_dir / "emotion_test_300.csv") as f:
            te_rows = list(csv.DictReader(f))
        classes = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        return "Emotion", [r["text"] for r in tr_rows], raw_probs, cal_probs, [int(r["label"]) for r in tr_rows], [r["text"] for r in te_rows], [int(r["label"]) for r in te_rows], 6, classes

    elif dataset_key == "trec":
        npz = np.load(targets_dir / "teacher_targets_trec_1200_gemma3_4b.npz")
        raw_probs = npz["train_probs_raw"]
        cal_probs = npz["train_probs_cal"]
        with open(benchmarks_dir / "trec_train_1200.csv") as f:
            tr_rows = list(csv.DictReader(f))
        with open(benchmarks_dir / "trec_test_500.csv") as f:
            te_rows = list(csv.DictReader(f))
        classes = ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]
        return "TREC-6", [r["text"] for r in tr_rows], raw_probs, cal_probs, [int(r["label"]) for r in tr_rows], [r["text"] for r in te_rows], [int(r["label"]) for r in te_rows], 6, classes

    elif dataset_key == "sst2":
        npz = np.load(targets_dir / "teacher_targets_sst2_gemma3_4b.npz")
        raw_probs = npz["train_probs_raw"]
        cal_probs = npz["train_probs_cal"]
        with open(benchmarks_dir / "sst2_train_1000.csv") as f:
            tr_rows = list(csv.DictReader(f))
        with open(benchmarks_dir / "sst2_test_200.csv") as f:
            te_rows = list(csv.DictReader(f))
        classes = ["Negative", "Positive"]
        return "SST-2", [r["text"] for r in tr_rows], raw_probs, cal_probs, [int(r["label"]) for r in tr_rows], [r["text"] for r in te_rows], [int(r["label"]) for r in te_rows], 2, classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "agnews", "emotion", "trec", "sst2"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    datasets = ["emotion", "agnews", "trec", "sst2"] if args.dataset == "all" else [args.dataset]
    summary = []

    for d_key in datasets:
        name, tr_x, raw_p, cal_p, tr_y, te_x, te_y, K, classes = load_data(d_key)
        # 1. Standard KD
        res_raw = train_and_eval(name, "standard_kd", tr_x, raw_p, tr_y, te_x, te_y, K, classes, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        # 2. Calibrated KD
        res_cal = train_and_eval(name, "calibrated_kd", tr_x, cal_p, tr_y, te_x, te_y, K, classes, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

        summary.append({
            "dataset": name,
            "classes": K,
            "std_acc": res_raw["test_acc"],
            "cal_acc": res_cal["test_acc"],
            "delta_acc": res_cal["test_acc"] - res_raw["test_acc"],
            "std_f1": res_raw["test_f1"],
            "cal_f1": res_cal["test_f1"],
            "delta_f1": res_cal["test_f1"] - res_raw["test_f1"],
            "std_counts": res_raw["pred_counts"],
            "cal_counts": res_cal["pred_counts"]
        })

    print("\n" + "="*80)
    print("ROBERTA-BASE DISTILLATION SUMMARY ACROSS ALL DATASETS")
    print("="*80)
    print(f"{'Dataset':<10} | {'Std Acc':<9} | {'Cal Acc':<9} | {'Delta Acc':<10} | {'Std F1':<9} | {'Cal F1':<9} | {'Delta F1':<10}")
    print("-" * 80)
    for s in summary:
        print(f"{s['dataset']:<10} | {s['std_acc']:6.2f}%   | {s['cal_acc']:6.2f}%   | {s['delta_acc']:+6.2f}%    | {s['std_f1']:6.2f}%   | {s['cal_f1']:6.2f}%   | {s['delta_f1']:+6.2f}%")
    print("="*80)

    with open(RESULTS_DIR / "roberta_master_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
