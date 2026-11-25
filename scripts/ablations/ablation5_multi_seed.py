"""
ablation5_multi_seed.py
-----------------------
Ablation 5: Multi-Seed Statistical Significance Validation (Seeds 42, 43, 44)
evaluated strictly on the Main 6 benchmarks (RoBERTa-base with cached Gemma-3-4B targets).

Outputs saved natively to outputs/ablations/5_multi_seed/multi_seed_summary.json.
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
OUT_DIR = ROOT / "outputs" / "ablations" / "5_multi_seed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_NAME = "roberta-base"
DATASETS = [
    ("student_questions", 4),
    ("trec6", 6),
    ("app_reviews", 5),
    ("twitter_emotion", 6),
    ("tweeteval_emotion", 4),
    ("bbc_news", 5),
]
SEEDS = [42, 43, 44]


def set_seed(seed: int) -> None:
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


class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]


def train_student(train_texts, train_probs, test_texts, test_golds, num_classes: int, seed: int) -> dict:
    set_seed(seed)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, local_files_only=True)

    def collate_fn(batch):
        texts = [b[0] for b in batch]
        probs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = probs
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_NAME, num_labels=num_classes, local_files_only=True).to(device)
    loader = DataLoader(KDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for _ in range(epochs):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            enc = tokenizer(test_texts[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            del enc, out

    del model, optimizer, scheduler, loader, tokenizer
    clear_vram()

    return {
        "macro_f1": float(f1_score(test_golds, preds, average="macro") * 100.0),
        "acc": float(accuracy_score(test_golds, preds) * 100.0),
    }


def main() -> None:
    summary = {}
    for task_name, num_classes in DATASETS:
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
        p_std = softmax(cached["train_logits"], axis=-1)
        p_ckd = softmax(cached["train_logits"] - cached["null_logits"], axis=-1)

        task_res = {"standard_kd": [], "calibrated_kd": []}
        for seed in SEEDS:
            r_std = train_student(tr_txt, p_std, te_txt, te_y, num_classes, seed)
            r_ckd = train_student(tr_txt, p_ckd, te_txt, te_y, num_classes, seed)
            task_res["standard_kd"].append(r_std)
            task_res["calibrated_kd"].append(r_ckd)
            print(f"  [{task_name}] Seed {seed}: Std={r_std['macro_f1']:.2f}% ({r_std['acc']:.1f}%) | CKD={r_ckd['macro_f1']:.2f}% ({r_ckd['acc']:.1f}%)", flush=True)

        agg = {}
        for m in ["standard_kd", "calibrated_kd"]:
            f1s = [x["macro_f1"] for x in task_res[m]]
            accs = [x["acc"] for x in task_res[m]]
            agg[m] = {
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "seeds": task_res[m],
            }
        agg["delta_f1_mean"] = agg["calibrated_kd"]["f1_mean"] - agg["standard_kd"]["f1_mean"]
        summary[task_name] = agg

    out_file = OUT_DIR / "multi_seed_summary.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[✓] Saved Ablation 5 multi-seed summary to: {out_file}")


if __name__ == "__main__":
    main()
