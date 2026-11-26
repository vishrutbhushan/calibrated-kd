"""
ablation3_hard_vs_soft.py
-------------------------
Ablation 3: Hard Label Fine-Tuning (argmax) vs. Soft Label Distillation (KL)
evaluated strictly on Main 6 benchmarks (BBC News & TweetEval Emotion).

Outputs saved natively to outputs/ablations/3_hard_vs_soft/hard_vs_soft_results.json.
"""

import os
import gc
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
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
CACHE_DIR = ROOT / "outputs" / "main_experiments" / "cache"
OUT_DIR = ROOT / "outputs" / "ablations" / "3_hard_vs_soft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_NAME = "roberta-base"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


class TargetDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]


def train_student(
    train_texts, targets, test_texts, test_golds, num_classes: int, is_hard: bool, label: str
) -> dict:
    """Trains RoBERTa-base with Cross-Entropy (hard) or KL-Divergence (soft)."""
    set_seed(SEED)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, local_files_only=True)

    def collate_fn(batch):
        texts = [b[0] for b in batch]
        t_vals = [b[1] for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc["targets"] = (
            torch.tensor(t_vals, dtype=torch.long)
            if is_hard
            else torch.tensor(np.array(t_vals), dtype=torch.float32)
        )
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_NAME, num_labels=num_classes, local_files_only=True
    ).to(device)
    loader = DataLoader(TargetDataset(train_texts, targets), batch_size=16, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss() if is_hard else nn.KLDivLoss(reduction="batchmean")

    model.train()
    for _ in range(epochs):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            if is_hard:
                loss = criterion(out.logits, b["targets"].to(device))
            else:
                loss = criterion(torch.log_softmax(out.logits, dim=-1), b["targets"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            enc = tokenizer(test_texts[i : i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            del enc, out

    del model, optimizer, scheduler, loader, tokenizer
    clear_vram()

    macro_f1 = float(f1_score(test_golds, preds, average="macro") * 100.0)
    acc = float(accuracy_score(test_golds, preds) * 100.0)
    print(f"  [{label}] Macro-F1: {macro_f1:.2f}% | Acc: {acc:.2f}%", flush=True)
    return {"macro_f1": macro_f1, "acc": acc}


def eval_dataset(task_name: str, num_classes: int) -> dict:
    task_dir = DATA_BASE / task_name
    tr_txt, te_txt, te_y = [], [], []
    with open(task_dir / "train_1200.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tr_txt.append(r["text"])
    with open(task_dir / "test_400.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            te_txt.append(r["text"])
            te_y.append(int(r["label"]))

    cached = np.load(CACHE_DIR / f"{task_name}_gemma4b_targets.npz")
    raw_logits = cached["train_logits"]
    cal_logits = raw_logits - cached["null_logits"]

    print(f"\n>>> Hard vs Soft Evaluation on {task_name.upper()} (K={num_classes}) <<<")
    return {
        "raw_hard": train_student(tr_txt, raw_logits.argmax(axis=-1).tolist(), te_txt, te_y, num_classes, True, "Raw Hard"),
        "calibrated_hard": train_student(tr_txt, cal_logits.argmax(axis=-1).tolist(), te_txt, te_y, num_classes, True, "Calibrated Hard"),
        "standard_soft_kd": train_student(tr_txt, softmax(raw_logits), te_txt, te_y, num_classes, False, "Standard Soft KD"),
        "calibrated_soft_kd": train_student(tr_txt, softmax(cal_logits), te_txt, te_y, num_classes, False, "CKD Soft (Ours)"),
    }


def main() -> None:
    res = {
        "bbc_news": eval_dataset("bbc_news", 5),
        "tweeteval_emotion": eval_dataset("tweeteval_emotion", 4),
    }
    out_file = OUT_DIR / "hard_vs_soft_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2)
    print(f"\n[✓] Saved Ablation 3 results to: {out_file}")


if __name__ == "__main__":
    main()
