"""
train_skewed_benchmarks.py
--------------------------
Evaluates knowledge distillation across natural and skewed dataset distributions:
  1) Emotion Natural
  2) TREC-6 Natural
  3) AG News Skewed
  4) SST-2 Skewed

For EACH benchmark, trains and evaluates 5 configurations:
  1) Standard KD (Raw teacher soft targets)
  2) Standard KD + Post-Hoc Output Calibration (z^S - log(p^S))
  3) Batch Calibrated KD (Data-mean prior: z - log(p_batch))
  4) Calibrated KD / CKD (Static null calibration: z - W_null)
  5) Prior-Aware CKD (z - W_null + log P_prior(y))

Saves results cleanly to results/skewed/.
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
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TARGETS_DIR = DATA_DIR / "teacher_targets" / "skewed"
RESULTS_DIR = ROOT / "results" / "skewed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "roberta": "roberta-base",
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-4B"
}

BENCHMARKS = {
    "emotion": {
        "npz": "teacher_targets_emotion_natural_1200_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/emotion_natural_train_1200.csv",
        "test_csv": "data/benchmarks/skewed/emotion_natural_test_500.csv",
        "num_classes": 6
    },
    "trec": {
        "npz": "teacher_targets_trec_natural_1200_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/trec_natural_train_1200.csv",
        "test_csv": "data/benchmarks/trec_test_500.csv",
        "num_classes": 6
    },
    "agnews": {
        "npz": "teacher_targets_agnews_skewed_1000_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/agnews_skewed_train_1000.csv",
        "test_csv": "data/benchmarks/ag_news_test_1000.csv",
        "num_classes": 4
    },
    "sst2": {
        "npz": "teacher_targets_sst2_skewed_1000_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/sst2_skewed_train_1000.csv",
        "test_csv": "data/benchmarks/sst2_test_200.csv",
        "num_classes": 2
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

def train_eval_roberta(dname, train_texts, test_texts, test_golds, targets_dict, num_classes, class_names, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_PATHS["roberta"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def collate(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    test_enc = tokenizer(test_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    results = {}

    # Helper function to train a single RoBERTa model on given target probabilities
    def run_one_variant(probs_key):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_classes, local_files_only=True
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        loader = DataLoader(KDDataset(train_texts, targets_dict[probs_key]), batch_size=16, shuffle=True, collate_fn=collate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(loader)*3*0.1), num_training_steps=len(loader)*3)

        model.train()
        for ep in range(3):
            for b in loader:
                optimizer.zero_grad()
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                loss = kl_loss_fn(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            out = model(input_ids=test_enc["input_ids"].to(device), attention_mask=test_enc["attention_mask"].to(device))
            logits = out.logits.cpu().numpy()
        return model, logits

    print(f"\n--- Training Standard KD ---", flush=True)
    m_std, std_logits = run_one_variant("train_probs_raw")
    std_preds = std_logits.argmax(axis=-1)
    results["standard_kd"] = {
        "acc": accuracy_score(test_golds, std_preds) * 100,
        "f1": f1_score(test_golds, std_preds, average="macro") * 100,
        "counts": np.bincount(std_preds, minlength=num_classes).tolist()
    }

    # Post-Hoc Output Calibration: compute empirical training logit mean from Standard KD student
    with torch.no_grad():
        tr_sample_enc = tokenizer(train_texts[:300], padding=True, truncation=True, max_length=256, return_tensors="pt")
        tr_sample_out = m_std(input_ids=tr_sample_enc["input_ids"].to(device), attention_mask=tr_sample_enc["attention_mask"].to(device))
        student_mean_logits = tr_sample_out.logits.mean(dim=0).cpu().numpy()
        del tr_sample_enc, tr_sample_out
    del m_std
    torch.cuda.empty_cache()

    post_logits = std_logits - student_mean_logits
    post_preds = post_logits.argmax(axis=-1)
    results["post_hoc_output_cal"] = {
        "acc": accuracy_score(test_golds, post_preds) * 100,
        "f1": f1_score(test_golds, post_preds, average="macro") * 100,
        "counts": np.bincount(post_preds, minlength=num_classes).tolist()
    }

    print(f"--- Training Batch Calibrated KD (Data-Mean Prior) ---", flush=True)
    _, batch_logits = run_one_variant("train_probs_batch")
    batch_preds = batch_logits.argmax(axis=-1)
    results["batch_cal_kd"] = {
        "acc": accuracy_score(test_golds, batch_preds) * 100,
        "f1": f1_score(test_golds, batch_preds, average="macro") * 100,
        "counts": np.bincount(batch_preds, minlength=num_classes).tolist()
    }
    torch.cuda.empty_cache()

    print(f"--- Training Calibrated KD (CKD, Null Prompt) ---", flush=True)
    _, ckd_logits = run_one_variant("train_probs_ckd")
    ckd_preds = ckd_logits.argmax(axis=-1)
    results["calibrated_kd"] = {
        "acc": accuracy_score(test_golds, ckd_preds) * 100,
        "f1": f1_score(test_golds, ckd_preds, average="macro") * 100,
        "counts": np.bincount(ckd_preds, minlength=num_classes).tolist()
    }
    torch.cuda.empty_cache()

    print(f"--- Training Prior-Aware CKD (Null + log P(y)) ---", flush=True)
    _, prior_logits = run_one_variant("train_probs_prior")
    prior_preds = prior_logits.argmax(axis=-1)
    results["prior_aware_ckd"] = {
        "acc": accuracy_score(test_golds, prior_preds) * 100,
        "f1": f1_score(test_golds, prior_preds, average="macro") * 100,
        "counts": np.bincount(prior_preds, minlength=num_classes).tolist()
    }
    torch.cuda.empty_cache()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="roberta", choices=["roberta", "qwen05", "qwen3"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    master_summary = {}

    for bname, bcfg in BENCHMARKS.items():
        print(f"\n{'='*75}\nEVALUATING SKEWED BENCHMARK: {bname.upper()} (Student: {args.student}, Seed: {args.seed})\n{'='*75}", flush=True)
        npz_path = TARGETS_DIR / bcfg["npz"]
        if not npz_path.exists():
            print(f"Target file {npz_path} does not exist yet. Skipping {bname}.")
            continue
        data = np.load(npz_path)

        with open(bcfg["train_csv"]) as f:
            tr_rows = list(csv.DictReader(f))
        with open(bcfg["test_csv"]) as f:
            te_rows = list(csv.DictReader(f))

        train_texts = [r["text"] for r in tr_rows]
        test_texts = [r["text"] for r in te_rows]
        test_golds = [int(r["label"]) for r in te_rows]
        class_names = data["class_names"].tolist()

        targets_dict = {
            "train_probs_raw": data["train_probs_raw"],
            "train_probs_ckd": data["train_probs_ckd"],
            "train_probs_batch": data["train_probs_batch"],
            "train_probs_prior": data["train_probs_prior"]
        }

        if args.student == "roberta":
            res = train_eval_roberta(bname, train_texts, test_texts, test_golds, targets_dict, bcfg["num_classes"], class_names, seed=args.seed)
        else:
            raise NotImplementedError(f"Student {args.student} not implemented in this runner.")

        print(f"\nRESULTS FOR {bname.upper()}:")
        for k, v in res.items():
            print(f"  {k:25s} | Acc: {v['acc']:.2f}% | F1: {v['f1']:.2f}% | Counts: {v['counts']}")

        master_summary[bname] = res
        out_file = RESULTS_DIR / f"{bname}_skewed_{args.student}_results.json"
        with open(out_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[✓] Saved {out_file}", flush=True)

    summary_file = RESULTS_DIR / f"master_skewed_{args.student}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\n[✓] Saved complete skewed summary to {summary_file}", flush=True)

if __name__ == "__main__":
    main()

