"""
train_roberta_audit_experiments.py
----------------------------------
Executes the lean high-ROI audit experiments on RoBERTa-base:
1. Calibration Strength (beta) sweep: beta in {0.0, 0.5, 1.0, 1.5}
   Target: Softmax(z - beta * W_null)
2. Multi-seed validation: seeds 42, 43, 44 for Standard KD vs Calibrated KD (beta=1.0)
   Dataset: AG News Skewed (10% Sci/Tech minority)
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
TARGETS_DIR = DATA_DIR / "teacher_targets" / "skewed"
RESULTS_DIR = ROOT / "results" / "audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "roberta-base"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = torch.tensor(probs, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "probs": self.probs[idx]}

def train_eval_roberta(train_texts, target_probs, test_texts, test_golds, num_classes, seed=42, epochs=3, batch_size=16, lr=2e-5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=num_classes, local_files_only=True
    ).to(device)

    def collate_fn(batch):
        txts = [b["text"] for b in batch]
        p = torch.stack([b["probs"] for b in batch])
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    train_loader = DataLoader(KDDataset(train_texts, target_probs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_enc = tokenizer(test_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(epochs):
        for b in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    with torch.no_grad():
        out = model(input_ids=test_enc["input_ids"].to(device), attention_mask=test_enc["attention_mask"].to(device))
        preds = out.logits.argmax(dim=-1).cpu().numpy()

    acc = accuracy_score(test_golds, preds) * 100
    macro_f1 = f1_score(test_golds, preds, average="macro") * 100
    counts = np.bincount(preds, minlength=num_classes).tolist()

    del model, tokenizer
    torch.cuda.empty_cache()
    return acc, macro_f1, counts

def main():
    print(f"{'='*75}\nSTARTING ROBERTA AUDIT EXPERIMENTS: BETA SWEEP & MULTI-SEED VALIDATION\n{'='*75}")

    # Load AG News Skewed data
    npz = np.load(TARGETS_DIR / "teacher_targets_agnews_skewed_1000_gemma3_4b.npz")
    train_logits_raw = npz["train_logits_raw"]
    null_logits = npz["null_logits"]
    num_classes = 4

    with open(BENCHMARK_DIR / "skewed" / "agnews_skewed_train_1000.csv") as f:
        tr_rows = list(csv.DictReader(f))
    with open(BENCHMARK_DIR / "ag_news_test_1000.csv") as f:
        te_rows = list(csv.DictReader(f))

    train_texts = [r["text"] for r in tr_rows]
    test_texts = [r["text"] for r in te_rows]
    test_golds = [int(r["label"]) for r in te_rows]

    results = {
        "beta_sweep": {},
        "multi_seed": {
            "standard_kd": {},
            "calibrated_kd": {}
        }
    }

    # 1. Calibration Strength Sweep (beta in 0.0, 0.5, 1.0, 1.5) on seed 42
    print("\n--- Part 1: Calibration Strength (beta) Sweep on AG News Skewed (Seed 42) ---")
    betas = [0.0, 0.5, 1.0, 1.5]
    for b in betas:
        print(f"\nTraining with beta = {b} ...", flush=True)
        # Compute targets: Softmax(z - beta * W_null)
        scaled_logits = train_logits_raw - b * null_logits
        scaled_probs = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        scaled_probs /= scaled_probs.sum(axis=-1, keepdims=True)

        acc, f1, counts = train_eval_roberta(train_texts, scaled_probs, test_texts, test_golds, num_classes, seed=42)
        print(f"  Result beta={b}: Macro-F1 = {f1:.2f}%, Acc = {acc:.2f}%, Sci/Tech (Class 3) = {counts[3]}", flush=True)
        results["beta_sweep"][str(b)] = {
            "beta": b,
            "macro_f1": f1,
            "acc": acc,
            "counts": counts,
            "scitech_preds": counts[3]
        }

    # 2. Multi-Seed Validation: Seeds 42, 43, 44 for Standard KD and Calibrated KD
    print("\n--- Part 2: Multi-Seed Validation (Seeds 42, 43, 44) on AG News Skewed ---")
    seeds = [42, 43, 44]
    std_probs = results["beta_sweep"]["0.0"] # already ran seed 42
    ckd_probs = results["beta_sweep"]["1.0"] # already ran seed 42

    results["multi_seed"]["standard_kd"]["42"] = results["beta_sweep"]["0.0"]
    results["multi_seed"]["calibrated_kd"]["42"] = results["beta_sweep"]["1.0"]

    # Compute raw & calibrated probability targets
    raw_p = np.exp(train_logits_raw - np.max(train_logits_raw, axis=-1, keepdims=True))
    raw_p /= raw_p.sum(axis=-1, keepdims=True)

    ckd_l = train_logits_raw - null_logits
    ckd_p = np.exp(ckd_l - np.max(ckd_l, axis=-1, keepdims=True))
    ckd_p /= ckd_p.sum(axis=-1, keepdims=True)

    for s in [43, 44]:
        print(f"\nRunning Seed {s} - Standard KD ...", flush=True)
        acc_std, f1_std, counts_std = train_eval_roberta(train_texts, raw_p, test_texts, test_golds, num_classes, seed=s)
        print(f"  Seed {s} Std KD: Macro-F1 = {f1_std:.2f}%, Acc = {acc_std:.2f}%, Sci/Tech = {counts_std[3]}")
        results["multi_seed"]["standard_kd"][str(s)] = {
            "seed": s, "macro_f1": f1_std, "acc": acc_std, "counts": counts_std, "scitech_preds": counts_std[3]
        }

        print(f"Running Seed {s} - Calibrated KD (CKD) ...", flush=True)
        acc_ckd, f1_ckd, counts_ckd = train_eval_roberta(train_texts, ckd_p, test_texts, test_golds, num_classes, seed=s)
        print(f"  Seed {s} CKD:    Macro-F1 = {f1_ckd:.2f}%, Acc = {acc_ckd:.2f}%, Sci/Tech = {counts_ckd[3]}")
        results["multi_seed"]["calibrated_kd"][str(s)] = {
            "seed": s, "macro_f1": f1_ckd, "acc": acc_ckd, "counts": counts_ckd, "scitech_preds": counts_ckd[3]
        }

    # Aggregate Statistics
    std_f1s = [results["multi_seed"]["standard_kd"][str(s)]["macro_f1"] for s in seeds]
    std_accs = [results["multi_seed"]["standard_kd"][str(s)]["acc"] for s in seeds]
    std_scis = [results["multi_seed"]["standard_kd"][str(s)]["scitech_preds"] for s in seeds]

    ckd_f1s = [results["multi_seed"]["calibrated_kd"][str(s)]["macro_f1"] for s in seeds]
    ckd_accs = [results["multi_seed"]["calibrated_kd"][str(s)]["acc"] for s in seeds]
    ckd_scis = [results["multi_seed"]["calibrated_kd"][str(s)]["scitech_preds"] for s in seeds]

    results["summary_statistics"] = {
        "standard_kd": {
            "macro_f1_mean": float(np.mean(std_f1s)),
            "macro_f1_std": float(np.std(std_f1s)),
            "acc_mean": float(np.mean(std_accs)),
            "acc_std": float(np.std(std_accs)),
            "scitech_mean": float(np.mean(std_scis)),
            "scitech_std": float(np.std(std_scis))
        },
        "calibrated_kd": {
            "macro_f1_mean": float(np.mean(ckd_f1s)),
            "macro_f1_std": float(np.std(ckd_f1s)),
            "acc_mean": float(np.mean(ckd_accs)),
            "acc_std": float(np.std(ckd_accs)),
            "scitech_mean": float(np.mean(ckd_scis)),
            "scitech_std": float(np.std(ckd_scis))
        },
        "delta": {
            "f1_gain": float(np.mean(ckd_f1s) - np.mean(std_f1s)),
            "acc_gain": float(np.mean(ckd_accs) - np.mean(std_accs))
        }
    }

    print("\n" + "="*75)
    print("FINAL 3-SEED VALIDATION SUMMARY (AG News Skewed)")
    print("="*75)
    print(f"Standard KD:   Macro-F1 = {np.mean(std_f1s):.2f} ± {np.std(std_f1s):.2f}% | Acc = {np.mean(std_accs):.2f} ± {np.std(std_accs):.2f}% | Sci/Tech = {np.mean(std_scis):.1f} ± {np.std(std_scis):.1f}")
    print(f"Calibrated KD: Macro-F1 = {np.mean(ckd_f1s):.2f} ± {np.std(ckd_f1s):.2f}% | Acc = {np.mean(ckd_accs):.2f} ± {np.std(ckd_accs):.2f}% | Sci/Tech = {np.mean(ckd_scis):.1f} ± {np.std(ckd_scis):.1f}")
    print(f"Net CKD Gain:  +{np.mean(ckd_f1s) - np.mean(std_f1s):.2f} pp F1 | +{np.mean(ckd_accs) - np.mean(std_accs):.2f} pp Acc")
    print("="*75)

    out_file = RESULTS_DIR / "roberta_audit_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved all audit results to {out_file}")

if __name__ == "__main__":
    main()

