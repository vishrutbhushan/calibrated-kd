"""
exp_combine_ckd_posthoc.py
--------------------------
Empirical investigation: What happens when we combine Calibrated Knowledge Distillation (CKD)
with Post-Hoc Output Calibration?

Evaluates 4 paradigms:
  1. Standard KD (Raw teacher soft targets)
  2. Standard KD + Post-Hoc Output Calibration
  3. Calibrated KD (CKD: teacher targets calibrated via null probe)
  4. CKD + Post-Hoc Output Calibration (CKD student with student-bias shift at test time)

Tested across 4 diverse benchmarks:
  - 20 Newsgroups Balanced (4 classes)
  - 20 Newsgroups Naturally Skewed (4 classes)
  - DBpedia-14 Naturally Skewed (14 classes)
  - AG News Skewed (4 classes)

Model: roberta-base (125M)
Teacher targets: google/gemma-3-4b-it (BF16)
Outputs saved to: outputs/results/ckd_plus_posthoc_combination_study.json
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
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_MODEL_NAME = "roberta-base"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

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

def train_and_eval_student(train_texts, train_probs, test_texts, test_golds, num_classes, seed=42):
    set_seed(seed)
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

    # Inference on test set
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

    # Compute student output mean bias on train subset for post-hoc calibration
    with torch.no_grad():
        sub_texts = train_texts[:300]
        enc_sub = tokenizer(sub_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out_sub = model(input_ids=enc_sub["input_ids"], attention_mask=enc_sub["attention_mask"])
        mean_bias = out_sub.logits.mean(dim=0).cpu().numpy()
        del enc_sub, out_sub

    del model, optimizer, scheduler, loader, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Base predictions
    base_preds = test_logits.argmax(axis=-1)
    base_metrics = compute_metrics(test_golds, base_preds, num_classes)

    # Post-hoc calibrated predictions
    post_logits = test_logits - mean_bias
    post_preds = post_logits.argmax(axis=-1)
    post_metrics = compute_metrics(test_golds, post_preds, num_classes)

    return base_metrics, post_metrics

def run_experiment():
    print("="*75)
    print("EMPIRICAL STUDY: COMBINING CALIBRATED KD (CKD) WITH POST-HOC CALIBRATION")
    print("="*75)

    all_results = {}

    # -------------------------------------------------------------------------
    # 1. 20 Newsgroups (4 classes) - Balanced & Naturally Skewed
    # -------------------------------------------------------------------------
    print("\n[Dataset 1 & 2] Loading 20 Newsgroups...", flush=True)
    raw_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    raw_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    category_map = {'comp': 0, 'rec': 1, 'sci': 2, 'talk': 3}
    def get_coarse_label(target_name):
        prefix = target_name.split('.')[0]
        if prefix in category_map:
            return category_map[prefix]
        elif target_name in ['alt.atheism', 'soc.religion.christian']:
            return 3
        return None

    train_pool = {0: [], 1: [], 2: [], 3: []}
    for text, t_idx in zip(raw_train.data, raw_train.target):
        t_name = raw_train.target_names[t_idx]
        c = get_coarse_label(t_name)
        if c is not None and len(text.strip()) >= 50:
            train_pool[c].append(text.strip())

    test_pool = {0: [], 1: [], 2: [], 3: []}
    for text, t_idx in zip(raw_test.data, raw_test.target):
        t_name = raw_test.target_names[t_idx]
        c = get_coarse_label(t_name)
        if c is not None and len(text.strip()) >= 50:
            test_pool[c].append(text.strip())

    test_texts_20 = []
    test_golds_20 = []
    for c in range(4):
        items = test_pool[c][:100]
        test_texts_20.extend(items)
        test_golds_20.extend([c] * len(items))

    bal_train_texts_20 = []
    for c in range(4):
        bal_train_texts_20.extend(train_pool[c][:250])

    skew_counts = [500, 250, 150, 100]
    skew_train_texts_20 = []
    for c in range(4):
        skew_train_texts_20.extend(train_pool[c][:skew_counts[c]])

    cached_20 = np.load(ROOT / "data" / "datasets" / "20newsgroups" / "teacher_targets_20newsgroups_gemma3_4b.npz")
    null_logits_20 = cached_20["null_logits"]
    bal_logits_20 = cached_20["bal_train_logits"]
    skew_logits_20 = cached_20["skew_train_logits"]

    # 1A. 20 Newsgroups Balanced
    print("\n--- 20 Newsgroups Balanced (N=1,000, 4 classes) ---", flush=True)
    p_std_bal = softmax(bal_logits_20)
    p_ckd_bal = softmax(bal_logits_20 - null_logits_20)

    print("  Training Standard KD on 20 Newsgroups Balanced...", flush=True)
    m_std_bal, m_std_post_bal = train_and_eval_student(bal_train_texts_20, p_std_bal, test_texts_20, test_golds_20, 4)
    print(f"    Standard KD:          Acc={m_std_bal['acc']:.2f}% | F1={m_std_bal['macro_f1']:.2f}% [{m_std_bal['worst_f1']:.2f}%]")
    print(f"    Standard KD + PostHoc: Acc={m_std_post_bal['acc']:.2f}% | F1={m_std_post_bal['macro_f1']:.2f}% [{m_std_post_bal['worst_f1']:.2f}%]")

    print("  Training Calibrated KD (CKD) on 20 Newsgroups Balanced...", flush=True)
    m_ckd_bal, m_ckd_post_bal = train_and_eval_student(bal_train_texts_20, p_ckd_bal, test_texts_20, test_golds_20, 4)
    print(f"    Calibrated KD (CKD):  Acc={m_ckd_bal['acc']:.2f}% | F1={m_ckd_bal['macro_f1']:.2f}% [{m_ckd_bal['worst_f1']:.2f}%]")
    print(f"    CKD + Post-Hoc:       Acc={m_ckd_post_bal['acc']:.2f}% | F1={m_ckd_post_bal['macro_f1']:.2f}% [{m_ckd_post_bal['worst_f1']:.2f}%]")

    all_results["20newsgroups_balanced"] = {
        "standard_kd": m_std_bal,
        "standard_kd_posthoc": m_std_post_bal,
        "calibrated_kd": m_ckd_bal,
        "ckd_posthoc": m_ckd_post_bal
    }

    # 1B. 20 Newsgroups Naturally Skewed
    print("\n--- 20 Newsgroups Skewed (N=1,000, 4 classes: [500, 250, 150, 100]) ---", flush=True)
    p_std_skew = softmax(skew_logits_20)
    p_ckd_skew = softmax(skew_logits_20 - null_logits_20)

    print("  Training Standard KD on 20 Newsgroups Skewed...", flush=True)
    m_std_skew, m_std_post_skew = train_and_eval_student(skew_train_texts_20, p_std_skew, test_texts_20, test_golds_20, 4)
    print(f"    Standard KD:          Acc={m_std_skew['acc']:.2f}% | F1={m_std_skew['macro_f1']:.2f}% [{m_std_skew['worst_f1']:.2f}%]")
    print(f"    Standard KD + PostHoc: Acc={m_std_post_skew['acc']:.2f}% | F1={m_std_post_skew['macro_f1']:.2f}% [{m_std_post_skew['worst_f1']:.2f}%]")

    print("  Training Calibrated KD (CKD) on 20 Newsgroups Skewed...", flush=True)
    m_ckd_skew, m_ckd_post_skew = train_and_eval_student(skew_train_texts_20, p_ckd_skew, test_texts_20, test_golds_20, 4)
    print(f"    Calibrated KD (CKD):  Acc={m_ckd_skew['acc']:.2f}% | F1={m_ckd_skew['macro_f1']:.2f}% [{m_ckd_skew['worst_f1']:.2f}%]")
    print(f"    CKD + Post-Hoc:       Acc={m_ckd_post_skew['acc']:.2f}% | F1={m_ckd_post_skew['macro_f1']:.2f}% [{m_ckd_post_skew['worst_f1']:.2f}%]")

    all_results["20newsgroups_skewed"] = {
        "standard_kd": m_std_skew,
        "standard_kd_posthoc": m_std_post_skew,
        "calibrated_kd": m_ckd_skew,
        "ckd_posthoc": m_ckd_post_skew
    }

    # -------------------------------------------------------------------------
    # 2. DBpedia-14 Naturally Skewed (14 classes, N=1,400)
    # -------------------------------------------------------------------------
    print("\n--- DBpedia-14 Naturally Skewed (N=1,400, 14 classes) ---", flush=True)
    dbp_train_csv = ROOT / "data" / "datasets" / "dbpedia14" / "dbpedia_natural_train_1400.csv"
    dbp_train_texts = []
    with open(dbp_train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dbp_train_texts.append(row["text"])

    cached_dbp_base = np.load(ROOT / "data" / "datasets" / "dbpedia14" / "teacher_targets_dbpedia_1400_gemma3_4b.npz")
    cached_dbp_nat = np.load(ROOT / "data" / "datasets" / "dbpedia14" / "teacher_targets_dbpedia_natural_1400_gemma3_4b.npz")

    dbp_test_texts = [str(t) for t in cached_dbp_base["test_texts"]]
    dbp_test_golds = [int(g) for g in cached_dbp_base["test_golds"]]
    dbp_null_logits = cached_dbp_base["null_logits"]
    dbp_train_logits = cached_dbp_nat["train_logits_raw"]

    p_std_dbp = softmax(dbp_train_logits)
    p_ckd_dbp = softmax(dbp_train_logits - dbp_null_logits)

    print("  Training Standard KD on DBpedia-14 Natural...", flush=True)
    m_std_dbp, m_std_post_dbp = train_and_eval_student(dbp_train_texts, p_std_dbp, dbp_test_texts, dbp_test_golds, 14)
    print(f"    Standard KD:          Acc={m_std_dbp['acc']:.2f}% | F1={m_std_dbp['macro_f1']:.2f}% [{m_std_dbp['worst_f1']:.2f}%]")
    print(f"    Standard KD + PostHoc: Acc={m_std_post_dbp['acc']:.2f}% | F1={m_std_post_dbp['macro_f1']:.2f}% [{m_std_post_dbp['worst_f1']:.2f}%]")

    print("  Training Calibrated KD (CKD) on DBpedia-14 Natural...", flush=True)
    m_ckd_dbp, m_ckd_post_dbp = train_and_eval_student(dbp_train_texts, p_ckd_dbp, dbp_test_texts, dbp_test_golds, 14)
    print(f"    Calibrated KD (CKD):  Acc={m_ckd_dbp['acc']:.2f}% | F1={m_ckd_dbp['macro_f1']:.2f}% [{m_ckd_dbp['worst_f1']:.2f}%]")
    print(f"    CKD + Post-Hoc:       Acc={m_ckd_post_dbp['acc']:.2f}% | F1={m_ckd_post_dbp['macro_f1']:.2f}% [{m_ckd_post_dbp['worst_f1']:.2f}%]")

    all_results["dbpedia14_natural"] = {
        "standard_kd": m_std_dbp,
        "standard_kd_posthoc": m_std_post_dbp,
        "calibrated_kd": m_ckd_dbp,
        "ckd_posthoc": m_ckd_post_dbp
    }

    # -------------------------------------------------------------------------
    # 3. AG News Skewed (4 classes, N=1,000)
    # -------------------------------------------------------------------------
    print("\n--- AG News Skewed (N=1,000, 4 classes) ---", flush=True)
    ag_train_csv = ROOT / "data" / "benchmarks" / "skewed" / "agnews_skewed_train_1000.csv"
    ag_test_csv = ROOT / "data" / "benchmarks" / "ag_news_test_1000.csv"

    ag_train_texts = []
    with open(ag_train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ag_train_texts.append(row["text"])

    ag_test_texts = []
    ag_test_golds = []
    with open(ag_test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ag_test_texts.append(row["text"])
            ag_test_golds.append(int(row["label"]))

    cached_ag = np.load(ROOT / "data" / "datasets" / "agnews" / "teacher_targets_agnews_skewed_1000_gemma3_4b.npz")
    p_std_ag = cached_ag["train_probs_raw"]
    p_ckd_ag = cached_ag["train_probs_ckd"]

    print("  Training Standard KD on AG News Skewed...", flush=True)
    m_std_ag, m_std_post_ag = train_and_eval_student(ag_train_texts, p_std_ag, ag_test_texts, ag_test_golds, 4)
    print(f"    Standard KD:          Acc={m_std_ag['acc']:.2f}% | F1={m_std_ag['macro_f1']:.2f}% [{m_std_ag['worst_f1']:.2f}%]")
    print(f"    Standard KD + PostHoc: Acc={m_std_post_ag['acc']:.2f}% | F1={m_std_post_ag['macro_f1']:.2f}% [{m_std_post_ag['worst_f1']:.2f}%]")

    print("  Training Calibrated KD (CKD) on AG News Skewed...", flush=True)
    m_ckd_ag, m_ckd_post_ag = train_and_eval_student(ag_train_texts, p_ckd_ag, ag_test_texts, ag_test_golds, 4)
    print(f"    Calibrated KD (CKD):  Acc={m_ckd_ag['acc']:.2f}% | F1={m_ckd_ag['macro_f1']:.2f}% [{m_ckd_ag['worst_f1']:.2f}%]")
    print(f"    CKD + Post-Hoc:       Acc={m_ckd_post_ag['acc']:.2f}% | F1={m_ckd_post_ag['macro_f1']:.2f}% [{m_ckd_post_ag['worst_f1']:.2f}%]")

    all_results["agnews_skewed"] = {
        "standard_kd": m_std_ag,
        "standard_kd_posthoc": m_std_post_ag,
        "calibrated_kd": m_ckd_ag,
        "ckd_posthoc": m_ckd_post_ag
    }

    # Save to disk
    out_file = OUT_DIR / "ckd_plus_posthoc_combination_study.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*75)
    print(f"[✓] Experiment Complete! All results saved to {out_file}")
    print("="*75)

if __name__ == "__main__":
    run_experiment()
