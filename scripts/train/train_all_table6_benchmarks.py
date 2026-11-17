"""
scripts/train/train_all_table6_benchmarks.py
-------------------------------------------
Executes complete, rigorous, and verifiable evaluation of all 5 distillation paradigms
for RoBERTa-base across the 5 remaining benchmarks in Table 6:
  1) SST-2 (Balanced, 2 classes)
  2) DBpedia-14 (Balanced, 14 classes)
  3) AG News (Skewed, 4 classes)
  4) Emotion (Naturally Skewed, 6 classes)
  5) TREC-6 (Naturally Skewed, 6 classes)

For EACH benchmark, computes and stores:
  - Teacher Raw Zero-Shot
  - Teacher Null-Calibrated Zero-Shot
  - Standard KD (Raw soft targets)
  - Post-Hoc Output Calibration
  - Batch-Calibrated KD (Global training set mean prior)
  - Calibrated KD (CKD: z - b_null)

Metrics saved per method:
  - Accuracy (%)
  - Macro-F1 (%)
  - Worst-Class F1 / F1_min (%)
  - Per-Class F1 (%)
  - Prediction counts per class

Outputs saved directly to results/main_results/<benchmark>/roberta_5_methods.json
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
RESULTS_ROOT = ROOT / "results" / "main_results"
MODEL_NAME = "roberta-base"

BENCHMARKS = [
    {
        "name": "sst2",
        "num_classes": 2,
        "classes": ["Negative", "Positive"],
        "max_len": 256,
        "train_csv": ROOT / "data" / "datasets" / "sst2" / "sst2_train_1000.csv",
        "test_csv": ROOT / "data" / "datasets" / "sst2" / "sst2_test_200.csv",
        "npz": ROOT / "data" / "datasets" / "sst2" / "teacher_targets_sst2_gemma3_4b.npz",
    },
    {
        "name": "agnews",
        "num_classes": 4,
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "max_len": 256,
        "train_csv": ROOT / "data" / "datasets" / "agnews" / "agnews_skewed_train_1000.csv",
        "test_csv": ROOT / "data" / "datasets" / "agnews" / "ag_news_test_1000.csv",
        "npz": ROOT / "data" / "datasets" / "agnews" / "teacher_targets_agnews_skewed_1000_gemma3_4b.npz",
    },
    {
        "name": "emotion",
        "num_classes": 6,
        "classes": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "max_len": 256,
        "train_csv": ROOT / "data" / "datasets" / "emotion" / "emotion_natural_train_1200.csv",
        "test_csv": ROOT / "data" / "datasets" / "emotion" / "emotion_natural_test_500.csv",
        "npz": ROOT / "data" / "datasets" / "emotion" / "teacher_targets_emotion_natural_1200_gemma3_4b.npz",
    },
    {
        "name": "trec6",
        "num_classes": 6,
        "classes": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "max_len": 256,
        "train_csv": ROOT / "data" / "datasets" / "trec6" / "trec_natural_train_1200.csv",
        "test_csv": ROOT / "data" / "datasets" / "trec6" / "trec_test_500.csv",
        "npz": ROOT / "data" / "datasets" / "trec6" / "teacher_targets_trec_natural_1200_gemma3_4b.npz",
    },
    {
        "name": "dbpedia14",
        "num_classes": 14,
        "classes": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
        ],
        "max_len": 512,
        "train_csv": ROOT / "data" / "datasets" / "dbpedia14" / "dbpedia_train_1400.csv",
        "test_csv": ROOT / "data" / "datasets" / "dbpedia14" / "dbpedia_test_700.csv",
        "npz": ROOT / "data" / "datasets" / "dbpedia14" / "teacher_targets_dbpedia_1400_gemma3_4b.npz",
    }
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

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

def run_benchmark(cfg, seed=42):
    name = cfg["name"]
    num_classes = cfg["num_classes"]
    max_len = cfg["max_len"]
    print(f"\n==================================================================", flush=True)
    print(f"  RUNNING BENCHMARK: {name.upper()} ({num_classes} classes, max_len={max_len})", flush=True)
    print(f"==================================================================", flush=True)

    # 1. Load data
    with open(cfg["train_csv"]) as f:
        tr_rows = list(csv.DictReader(f))
    with open(cfg["test_csv"]) as f:
        te_rows = list(csv.DictReader(f))

    train_texts = [r["text"] for r in tr_rows]
    test_texts = [r["text"] for r in te_rows]
    test_golds = [int(r["label"]) for r in te_rows]

    # 2. Load teacher targets
    npz = np.load(cfg["npz"])
    train_logits_raw = npz["train_logits_raw"]
    train_probs_raw = npz["train_probs_raw"]
    test_logits_raw = npz["test_logits_raw"]
    null_logits = npz["null_logits"]

    # Resolve CKD targets
    if "train_probs_ckd" in npz:
        train_probs_ckd = npz["train_probs_ckd"]
    elif "train_probs_cal" in npz:
        train_probs_ckd = npz["train_probs_cal"]
    else:
        cal_l = train_logits_raw - null_logits
        cal_l -= cal_l.max(axis=-1, keepdims=True)
        exp_cal = np.exp(cal_l)
        train_probs_ckd = exp_cal / exp_cal.sum(axis=-1, keepdims=True)

    # Resolve Batch-Calibrated targets
    if "train_probs_batch" in npz:
        train_probs_batch = npz["train_probs_batch"]
    else:
        p_batch = train_probs_raw.mean(axis=0, keepdims=True)
        p_batch = np.clip(p_batch, 1e-12, 1.0)
        p_batch /= p_batch.sum(axis=-1, keepdims=True)
        b_logits = train_logits_raw - np.log(p_batch)
        b_logits -= b_logits.max(axis=-1, keepdims=True)
        exp_b = np.exp(b_logits)
        train_probs_batch = exp_b / exp_b.sum(axis=-1, keepdims=True)

    # Resolve Test logits cal
    if "test_logits_ckd" in npz:
        test_logits_cal = npz["test_logits_ckd"]
    elif "test_logits_cal" in npz:
        test_logits_cal = npz["test_logits_cal"]
    else:
        test_logits_cal = test_logits_raw - null_logits

    res = {
        "benchmark": name,
        "num_classes": num_classes,
        "classes": cfg["classes"],
        "num_train": len(train_texts),
        "num_test": len(test_texts),
        "null_gap": float(null_logits.max() - null_logits.min())
    }

    # 3. Teacher Baselines
    t_raw_preds = test_logits_raw.argmax(axis=-1)
    res["teacher_raw"] = compute_metrics(test_golds, t_raw_preds, num_classes)
    print(f"Teacher Raw: Acc={res['teacher_raw']['acc']:.2f}% | F1={res['teacher_raw']['macro_f1']:.2f}% [{res['teacher_raw']['worst_f1']:.2f}%]", flush=True)

    t_cal_preds = test_logits_cal.argmax(axis=-1)
    res["teacher_calibrated"] = compute_metrics(test_golds, t_cal_preds, num_classes)
    print(f"Teacher Cal: Acc={res['teacher_calibrated']['acc']:.2f}% | F1={res['teacher_calibrated']['macro_f1']:.2f}% [{res['teacher_calibrated']['worst_f1']:.2f}%]", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["probs"] = p
        return enc

    def get_eval_logits(model):
        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(test_texts), 32):
                chunk = test_texts[i:i+32]
                enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                all_logits.append(out.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    def train_student_variant(probs_arr, return_model=False):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_classes, local_files_only=True
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        loader = DataLoader(KDDataset(train_texts, probs_arr), batch_size=16, shuffle=True, collate_fn=collate_fn)
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

        logits = get_eval_logits(model)
        if return_model:
            return model, logits
        else:
            del model, optimizer, scheduler, loader
            gc.collect()
            torch.cuda.empty_cache()
            return None, logits

    # Method 1: Standard KD
    print("Training 1/4: Standard KD...", flush=True)
    m_std, std_logits = train_student_variant(train_probs_raw, return_model=True)
    std_preds = std_logits.argmax(axis=-1)
    res["standard_kd"] = compute_metrics(test_golds, std_preds, num_classes)
    print(f"  Standard KD -> Acc: {res['standard_kd']['acc']:.2f}% | F1: {res['standard_kd']['macro_f1']:.2f}% [{res['standard_kd']['worst_f1']:.2f}%] | Counts: {res['standard_kd']['counts']}", flush=True)

    # Method 2: Post-Hoc Output Calibration
    print("Evaluating 2/4: Post-Hoc Output Calibration...", flush=True)
    with torch.no_grad():
        sample_subset = train_texts[:300]
        enc_sub = tokenizer(sample_subset, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out_sub = m_std(input_ids=enc_sub["input_ids"], attention_mask=enc_sub["attention_mask"])
        student_mean_logits = out_sub.logits.mean(dim=0).cpu().numpy()
        del enc_sub, out_sub
    del m_std
    gc.collect()
    torch.cuda.empty_cache()

    post_logits = std_logits - student_mean_logits
    post_preds = post_logits.argmax(axis=-1)
    res["post_hoc_output_cal"] = compute_metrics(test_golds, post_preds, num_classes)
    print(f"  Post-Hoc Cal -> Acc: {res['post_hoc_output_cal']['acc']:.2f}% | F1: {res['post_hoc_output_cal']['macro_f1']:.2f}% [{res['post_hoc_output_cal']['worst_f1']:.2f}%] | Counts: {res['post_hoc_output_cal']['counts']}", flush=True)

    # Method 3: Batch-Calibrated KD
    print("Training 3/4: Batch-Calibrated KD...", flush=True)
    _, batch_logits = train_student_variant(train_probs_batch, return_model=False)
    batch_preds = batch_logits.argmax(axis=-1)
    res["batch_cal_kd"] = compute_metrics(test_golds, batch_preds, num_classes)
    print(f"  Batch-Cal KD -> Acc: {res['batch_cal_kd']['acc']:.2f}% | F1: {res['batch_cal_kd']['macro_f1']:.2f}% [{res['batch_cal_kd']['worst_f1']:.2f}%] | Counts: {res['batch_cal_kd']['counts']}", flush=True)

    # Method 4: Calibrated KD (CKD)
    print("Training 4/4: Calibrated KD (CKD)...", flush=True)
    _, ckd_logits = train_student_variant(train_probs_ckd, return_model=False)
    ckd_preds = ckd_logits.argmax(axis=-1)
    res["calibrated_kd"] = compute_metrics(test_golds, ckd_preds, num_classes)
    print(f"  Calibrated KD -> Acc: {res['calibrated_kd']['acc']:.2f}% | F1: {res['calibrated_kd']['macro_f1']:.2f}% [{res['calibrated_kd']['worst_f1']:.2f}%] | Counts: {res['calibrated_kd']['counts']}", flush=True)

    # Save to disk
    out_dir = RESULTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "roberta_5_methods.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved verified results to {out_file}", flush=True)
    return res

def main():
    set_seed(42)
    all_results = {}
    for cfg in BENCHMARKS:
        all_results[cfg["name"]] = run_benchmark(cfg)

    summary_file = RESULTS_ROOT / "all_table6_benchmarks_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=======================================================", flush=True)
    print(f"  ALL BENCHMARKS COMPLETED! Summary saved to {summary_file}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == "__main__":
    main()
