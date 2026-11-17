"""
scripts/train/train_balanced_5_methods.py
-----------------------------------------
Evaluates knowledge distillation across balanced dataset distributions for RoBERTa-base:
  1) Emotion (6 classes, balanced test)
  2) TREC-6 (6 classes, balanced test)
  3) AG News (4 classes, balanced test)
  4) SST-2 (2 classes, balanced test)

For EACH benchmark, trains and evaluates all 5 distillation configurations:
  1) Standard KD (Raw teacher soft targets)
  2) Standard KD + Post-Hoc Output Calibration (z^S - E[z^S])
  3) Batch Calibrated KD (Data-mean prior: z^T - log p_batch)
  4) Calibrated KD / CKD (Static null calibration: z^T - W_null)
  5) Oracle Prior CKD (z^T - W_null + log P_prior(y))

Saves results to results/1_balanced_benchmarks/ and results/roberta/.
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "1_balanced_benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-base"

BENCHMARKS = {
    "emotion": {
        "npz": "teacher_targets_emotion_1200_gemma3_4b.npz",
        "train_csv": "data/benchmarks/emotion_train_1200.csv",
        "test_csv": "data/benchmarks/emotion_test_300.csv",
        "num_classes": 6,
        "classes": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    },
    "trec": {
        "npz": "teacher_targets_trec_1200_gemma3_4b.npz",
        "train_csv": "data/benchmarks/trec_train_1200.csv",
        "test_csv": "data/benchmarks/trec_test_500.csv",
        "num_classes": 6,
        "classes": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]
    },
    "agnews": {
        "npz": "teacher_targets_agnews_1000_gemma3_4b.npz",
        "train_csv": "data/benchmarks/ag_news_train_1000.csv",
        "test_csv": "data/benchmarks/ag_news_test_1000.csv",
        "num_classes": 4,
        "classes": ["World", "Sports", "Business", "Sci/Tech"]
    },
    "sst2": {
        "npz": "teacher_targets_sst2_gemma3_4b.npz",
        "train_csv": "data/benchmarks/sst2_train_1000.csv",
        "test_csv": "data/benchmarks/sst2_test_200.csv",
        "num_classes": 2,
        "classes": ["Negative", "Positive"]
    },
    "dbpedia": {
        "npz": "teacher_targets_dbpedia_1400_gemma3_4b.npz",
        "train_csv": "data/benchmarks/dbpedia_train_1400.csv",
        "test_csv": "data/benchmarks/dbpedia_test_700.csv",
        "num_classes": 14,
        "classes": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
        ]
    }
}

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

def train_eval_roberta_5_methods(dname, train_texts, test_texts, test_golds, targets_dict, num_classes, class_names, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def collate(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def get_eval_logits(model):
        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(test_texts), 32):
                b_texts = test_texts[i : i + 32]
                enc = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                all_logits.append(out.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    results = {}

    def run_one_variant(probs_key, return_model=False):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_classes, local_files_only=True
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        loader = DataLoader(KDDataset(train_texts, targets_dict[probs_key]), batch_size=16, shuffle=True, collate_fn=collate)
        total_steps = len(loader) * 3
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        model.train()
        for ep in range(3):
            for b in loader:
                optimizer.zero_grad()
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                loss = kl_loss_fn(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        logits = get_eval_logits(model)
        if return_model:
            return model, logits
        else:
            del model
            torch.cuda.empty_cache()
            return None, logits

    # 1. Standard KD
    print(f"\n--- [{dname}] Training 1/5: Standard KD ---", flush=True)
    m_std, std_logits = run_one_variant("train_probs_raw", return_model=True)
    std_preds = std_logits.argmax(axis=-1)
    results["standard_kd"] = {
        "acc": float(accuracy_score(test_golds, std_preds) * 100),
        "f1": float(f1_score(test_golds, std_preds, average="macro") * 100),
        "counts": np.bincount(std_preds, minlength=num_classes).tolist()
    }
    print(f"  Standard KD -> Acc: {results['standard_kd']['acc']:.2f}% | F1: {results['standard_kd']['f1']:.2f}% | Counts: {results['standard_kd']['counts']}", flush=True)

    # 2. Post-Hoc Output Calibration
    print(f"--- [{dname}] Evaluating 2/5: Post-Hoc Output Calibration ---", flush=True)
    with torch.no_grad():
        tr_sample_enc = tokenizer(train_texts[:300], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        tr_sample_out = m_std(input_ids=tr_sample_enc["input_ids"], attention_mask=tr_sample_enc["attention_mask"])
        student_mean_logits = tr_sample_out.logits.mean(dim=0).cpu().numpy()
        del tr_sample_enc, tr_sample_out
    del m_std
    torch.cuda.empty_cache()

    post_logits = std_logits - student_mean_logits
    post_preds = post_logits.argmax(axis=-1)
    results["post_hoc_output_cal"] = {
        "acc": float(accuracy_score(test_golds, post_preds) * 100),
        "f1": float(f1_score(test_golds, post_preds, average="macro") * 100),
        "counts": np.bincount(post_preds, minlength=num_classes).tolist()
    }
    print(f"  Post-Hoc Output Cal -> Acc: {results['post_hoc_output_cal']['acc']:.2f}% | F1: {results['post_hoc_output_cal']['f1']:.2f}% | Counts: {results['post_hoc_output_cal']['counts']}", flush=True)

    # 3. Batch Calibrated KD (Data-Mean Prior)
    print(f"--- [{dname}] Training 3/5: Batch Calibrated KD ---", flush=True)
    _, batch_logits = run_one_variant("train_probs_batch", return_model=False)
    batch_preds = batch_logits.argmax(axis=-1)
    results["batch_cal_kd"] = {
        "acc": float(accuracy_score(test_golds, batch_preds) * 100),
        "f1": float(f1_score(test_golds, batch_preds, average="macro") * 100),
        "counts": np.bincount(batch_preds, minlength=num_classes).tolist()
    }
    print(f"  Batch Cal KD -> Acc: {results['batch_cal_kd']['acc']:.2f}% | F1: {results['batch_cal_kd']['f1']:.2f}% | Counts: {results['batch_cal_kd']['counts']}", flush=True)

    # 4. Calibrated KD (CKD)
    print(f"--- [{dname}] Training 4/5: Calibrated KD (CKD) ---", flush=True)
    _, ckd_logits = run_one_variant("train_probs_ckd", return_model=False)
    ckd_preds = ckd_logits.argmax(axis=-1)
    results["calibrated_kd"] = {
        "acc": float(accuracy_score(test_golds, ckd_preds) * 100),
        "f1": float(f1_score(test_golds, ckd_preds, average="macro") * 100),
        "counts": np.bincount(ckd_preds, minlength=num_classes).tolist()
    }
    print(f"  Calibrated KD (CKD) -> Acc: {results['calibrated_kd']['acc']:.2f}% | F1: {results['calibrated_kd']['f1']:.2f}% | Counts: {results['calibrated_kd']['counts']}", flush=True)

    # 5. Prior-Aware CKD
    print(f"--- [{dname}] Training 5/5: Oracle Prior CKD ---", flush=True)
    _, prior_logits = run_one_variant("train_probs_prior", return_model=False)
    prior_preds = prior_logits.argmax(axis=-1)
    results["prior_aware_ckd"] = {
        "acc": float(accuracy_score(test_golds, prior_preds) * 100),
        "f1": float(f1_score(test_golds, prior_preds, average="macro") * 100),
        "counts": np.bincount(prior_preds, minlength=num_classes).tolist()
    }
    print(f"  Oracle Prior CKD -> Acc: {results['prior_aware_ckd']['acc']:.2f}% | F1: {results['prior_aware_ckd']['f1']:.2f}% | Counts: {results['prior_aware_ckd']['counts']}", flush=True)
    torch.cuda.empty_cache()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "emotion", "trec", "agnews", "sst2", "dbpedia"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmarks_to_run = BENCHMARKS.keys() if args.benchmark == "all" else [args.benchmark]
    master_summary = {}

    for b_key in benchmarks_to_run:
        b_info = BENCHMARKS[b_key]
        print(f"\n=======================================================", flush=True)
        print(f"  BENCHMARK: {b_key.upper()} (Balanced Suite, 5 Methods)", flush=True)
        print(f"=======================================================", flush=True)

        # Load texts & golds
        with open(ROOT / b_info["train_csv"]) as f:
            tr_rows = list(csv.DictReader(f))
        with open(ROOT / b_info["test_csv"]) as f:
            te_rows = list(csv.DictReader(f))

        train_texts = [r["text"] for r in tr_rows]
        test_texts = [r["text"] for r in te_rows]
        test_golds = [int(r["label"]) for r in te_rows]

        # Load teacher targets
        npz_file = ROOT / "data" / "teacher_targets" / b_info["npz"]
        npz = np.load(npz_file)

        train_probs_raw = npz["train_probs_raw"]
        
        # CKD probs
        if "train_probs_static" in npz:
            train_probs_ckd = npz["train_probs_static"]
        elif "train_probs_cal" in npz:
            train_probs_ckd = npz["train_probs_cal"]
        else:
            raise KeyError(f"No cal/static probs in {npz_file}")

        # Compute Batch-Calibrated probs: z^T - log(p_batch)
        train_logits_raw = npz["train_logits_raw"]
        # p_batch is average teacher probability across training set
        p_batch = train_probs_raw.mean(axis=0, keepdims=True)
        p_batch = np.clip(p_batch, 1e-12, 1.0)
        p_batch /= p_batch.sum(axis=-1, keepdims=True)
        
        logits_batch_cal = train_logits_raw - np.log(p_batch)
        # Stable softmax
        logits_batch_cal -= logits_batch_cal.max(axis=-1, keepdims=True)
        exp_batch = np.exp(logits_batch_cal)
        train_probs_batch = exp_batch / exp_batch.sum(axis=-1, keepdims=True)

        # Compute Prior-Aware CKD probs:
        # In balanced dataset, prior P(y) = 1/K for all classes, so log P(y) is constant.
        # Target distribution is mathematically identical to CKD targets.
        train_probs_prior = train_probs_ckd.copy()

        targets_dict = {
            "train_probs_raw": train_probs_raw,
            "train_probs_batch": train_probs_batch,
            "train_probs_ckd": train_probs_ckd,
            "train_probs_prior": train_probs_prior
        }

        b_results = train_eval_roberta_5_methods(
            b_key, train_texts, test_texts, test_golds, targets_dict,
            b_info["num_classes"], b_info["classes"], seed=args.seed
        )

        master_summary[b_key] = b_results

        # Save individual json
        ind_file = RESULTS_DIR / f"{b_key}_balanced_5_methods.json"
        with open(ind_file, "w") as f:
            json.dump(b_results, f, indent=2)
        print(f"[✓] Saved individual results to {ind_file}", flush=True)

        # Also copy to results/main_results/<b_key>/balanced_5_methods.json
        main_dest = ROOT / "results" / "main_results" / (b_key if b_key != "dbpedia" else "dbpedia14") / "balanced_5_methods.json"
        main_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(main_dest, "w") as f:
            json.dump(b_results, f, indent=2)
        print(f"[✓] Saved structured results to {main_dest}", flush=True)

    # Save/update master summary
    out_file = RESULTS_DIR / "balanced_roberta_5_methods_summary.json"
    existing_master = {}
    if out_file.exists():
        try:
            with open(out_file) as f:
                existing_master = json.load(f)
        except Exception:
            pass
    existing_master.update(master_summary)
    with open(out_file, "w") as f:
        json.dump(existing_master, f, indent=2)
    print(f"\n[✓] ALL DONE! Master summary saved to: {out_file}", flush=True)

if __name__ == "__main__":
    main()
