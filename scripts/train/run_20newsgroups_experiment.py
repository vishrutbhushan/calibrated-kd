"""
run_20newsgroups_experiment.py
------------------------------
Evaluates 20 Newsgroups (4 classes: Computers, Recreation, Science, Politics)
under BOTH:
  1) Balanced Setup (250/class = 1,000 train items)
  2) Naturally Skewed Setup (500, 250, 150, 100 = 1,000 train items)

Evaluates on a held-out balanced test set (100/class = 400 test items).
Primary Focus: Comparing Calibrated KD (CKD) vs. Post-Hoc Output Calibration
(with Standard KD included as baseline reference).

Models:
  Teacher: google/gemma-3-4b-it (BF16)
  Student: roberta-base (125M)
"""

import os
import csv
import json
import random
import gc
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
DATA_DIR = ROOT / "data" / "datasets" / "20newsgroups"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_MODEL_NAME = "roberta-base"

CLASSES = ["Computers", "Recreation", "Science", "Politics"]
VERBALIZERS = ["A", "B", "C", "D"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_prompt(text):
    return (
        "Classify the newsgroup text into one of four categories:\n"
        "A: Computers\n"
        "B: Recreation\n"
        "C: Science\n"
        "D: Politics\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )

def compute_metrics(golds, preds, num_classes=4):
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

def main():
    set_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Step 1: Prepare 20 Newsgroups Splits (Balanced & Skewed)
    # =========================================================================
    print("Loading raw 20 Newsgroups corpus...", flush=True)
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

    # Filter and clean texts
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

    # Build balanced test set: 100 per class = 400 test items
    test_texts = []
    test_golds = []
    for c in range(4):
        items = test_pool[c][:100]
        test_texts.extend(items)
        test_golds.extend([c] * len(items))

    # Build balanced train set: 250 per class = 1,000 train items
    bal_train_texts = []
    bal_train_golds = []
    for c in range(4):
        items = train_pool[c][:250]
        bal_train_texts.extend(items)
        bal_train_golds.extend([c] * len(items))

    # Build skewed train set: 500, 250, 150, 100 = 1,000 train items
    skew_counts = [500, 250, 150, 100]
    skew_train_texts = []
    skew_train_golds = []
    for c in range(4):
        items = train_pool[c][:skew_counts[c]]
        skew_train_texts.extend(items)
        skew_train_golds.extend([c] * len(items))

    print(f"[✓] Data splits prepared:")
    print(f"  Balanced Train: {len(bal_train_texts)} items (Counts: 250 each)")
    print(f"  Skewed Train:   {len(skew_train_texts)} items (Counts: {skew_counts})")
    print(f"  Balanced Test:  {len(test_texts)} items (Counts: 100 each)")

    # =========================================================================
    # Step 2: Extract Teacher Logits via Gemma-3-4B-IT
    # =========================================================================
    # Check if target cache exists
    cache_file = DATA_DIR / "teacher_targets_20newsgroups_gemma3_4b.npz"

    if cache_file.exists():
        print(f"[✓] Loading existing teacher targets from {cache_file}...", flush=True)
        cached = np.load(cache_file)
        null_logits = cached["null_logits"]
        test_logits = cached["test_logits"]
        bal_train_logits = cached["bal_train_logits"]
        skew_train_logits = cached["skew_train_logits"]
    else:
        print(f"Loading Teacher from {TEACHER_MODEL_PATH} in BF16...", flush=True)
        t_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, local_files_only=True)
        t_tokenizer.padding_side = "left"
        t_tokenizer.truncation_side = "left"
        if t_tokenizer.pad_token is None:
            t_tokenizer.pad_token = t_tokenizer.eos_token
        verbalizer_ids = [t_tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]

        t_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            local_files_only=True
        )
        t_model.eval()

        # 1. Null vector
        null_prompt = format_prompt("N/A")
        inp_null = t_tokenizer(null_prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            null_logits = t_model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()

        def extract_logits(texts_list, desc=""):
            prompts = [format_prompt(t) for t in texts_list]
            res_logits = []
            batch_size = 8
            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    b = prompts[i : i + batch_size]
                    inp = t_tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda:0")
                    out = t_model(**inp)
                    logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                    res_logits.append(logits)
                    del inp, out
                    if (i // batch_size) % 25 == 0:
                        print(f"  {desc}: {min(i + batch_size, len(prompts))}/{len(prompts)}...", flush=True)
            return np.concatenate(res_logits, axis=0)

        print("\nExtracting Teacher logits on Test Set (N=400)...", flush=True)
        test_logits = extract_logits(test_texts, "Test")

        print("\nExtracting Teacher logits on Balanced Train Set (N=1,000)...", flush=True)
        bal_train_logits = extract_logits(bal_train_texts, "Balanced Train")

        print("\nExtracting Teacher logits on Skewed Train Set (N=1,000)...", flush=True)
        skew_train_logits = extract_logits(skew_train_texts, "Skewed Train")

        np.savez_compressed(
            cache_file,
            null_logits=null_logits,
            test_logits=test_logits,
            bal_train_logits=bal_train_logits,
            skew_train_logits=skew_train_logits
        )
        print(f"[✓] Saved teacher targets to {cache_file}", flush=True)

        del t_model, t_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Softmax conversion
    def softmax_logits(l):
        e = np.exp(l - np.max(l, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    test_probs = softmax_logits(test_logits)
    test_logits_cal = test_logits - null_logits
    test_probs_cal = softmax_logits(test_logits_cal)

    null_probs = np.exp(null_logits - np.max(null_logits)) / np.sum(np.exp(null_logits - np.max(null_logits)))
    print("\n" + "="*70)
    print("TEACHER NULL BIAS ON 20 NEWSGROUPS (4 CLASSES):")
    for v, c, l, p in zip(VERBALIZERS, CLASSES, null_logits, null_probs):
        print(f"  {v} [{c:12s}]: Logit = {l:6.2f} | Prior Prob = {p*100:6.2f}%")
    print(f"Null Logit Gap: {null_logits.max() - null_logits.min():.2f} logits")
    print("="*70)

    # Teacher Zero-Shot Baselines
    t_raw_preds = test_logits.argmax(axis=-1)
    t_raw_metrics = compute_metrics(test_golds, t_raw_preds, 4)
    t_cal_preds = test_logits_cal.argmax(axis=-1)
    t_cal_metrics = compute_metrics(test_golds, t_cal_preds, 4)

    print(f"\nTEACHER ZERO-SHOT on N=400 Test Items:")
    print(f"  Raw Teacher:        Acc = {t_raw_metrics['acc']:.2f}% | Macro-F1 = {t_raw_metrics['macro_f1']:.2f}% [{t_raw_metrics['worst_f1']:.2f}%] | Counts = {t_raw_metrics['counts']}")
    print(f"  Calibrated Teacher: Acc = {t_cal_metrics['acc']:.2f}% | Macro-F1 = {t_cal_metrics['macro_f1']:.2f}% [{t_cal_metrics['worst_f1']:.2f}%] | Counts = {t_cal_metrics['counts']}")

    # =========================================================================
    # Step 3: Student Training & Comparison (RoBERTa-base)
    # =========================================================================
    s_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    max_len = 256

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = s_tokenizer(txts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["probs"] = p
        return enc

    def eval_model(model):
        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(test_texts), 32):
                chunk = test_texts[i : i + 32]
                enc = s_tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                all_logits.append(out.logits.cpu().numpy())
                del enc, out
        return np.concatenate(all_logits, axis=0)

    def train_student(train_texts, probs_arr, return_model=False):
        set_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(
            STUDENT_MODEL_NAME, num_labels=4, local_files_only=True
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

        logits = eval_model(model)
        if return_model:
            return model, logits
        else:
            del model, optimizer, scheduler, loader
            gc.collect()
            torch.cuda.empty_cache()
            return None, logits

    master_results = {
        "teacher_raw": t_raw_metrics,
        "teacher_calibrated": t_cal_metrics,
        "balanced": {},
        "skewed": {}
    }

    # -------------------------------------------------------------------------
    # PART A: BALANCED SETUP
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART A: BALANCED SETUP (250 items per class = 1,000 train items)")
    print("="*70)

    bal_probs_raw = softmax_logits(bal_train_logits)
    bal_probs_ckd = softmax_logits(bal_train_logits - null_logits)

    # 1. Standard KD
    print("Training Standard KD (Balanced)...", flush=True)
    m_std_bal, std_bal_logits = train_student(bal_train_texts, bal_probs_raw, return_model=True)
    std_bal_metrics = compute_metrics(test_golds, std_bal_logits.argmax(axis=-1), 4)
    print(f"  Standard KD: Acc={std_bal_metrics['acc']:.2f}% | F1={std_bal_metrics['macro_f1']:.2f}% [{std_bal_metrics['worst_f1']:.2f}%] | Counts={std_bal_metrics['counts']}")

    # 2. Post-Hoc Output Calibration
    print("Evaluating Post-Hoc Output Calibration (Balanced)...", flush=True)
    with torch.no_grad():
        sub_texts = bal_train_texts[:300]
        enc_sub = s_tokenizer(sub_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out_sub = m_std_bal(input_ids=enc_sub["input_ids"], attention_mask=enc_sub["attention_mask"])
        mean_bias_bal = out_sub.logits.mean(dim=0).cpu().numpy()
        del enc_sub, out_sub
    del m_std_bal
    gc.collect()
    torch.cuda.empty_cache()

    post_bal_logits = std_bal_logits - mean_bias_bal
    post_bal_metrics = compute_metrics(test_golds, post_bal_logits.argmax(axis=-1), 4)
    print(f"  Post-Hoc:    Acc={post_bal_metrics['acc']:.2f}% | F1={post_bal_metrics['macro_f1']:.2f}% [{post_bal_metrics['worst_f1']:.2f}%] | Counts={post_bal_metrics['counts']}")

    # 3. Calibrated KD (CKD)
    print("Training Calibrated KD (CKD, Balanced)...", flush=True)
    _, ckd_bal_logits = train_student(bal_train_texts, bal_probs_ckd, return_model=False)
    ckd_bal_metrics = compute_metrics(test_golds, ckd_bal_logits.argmax(axis=-1), 4)
    print(f"  Calibrated KD: Acc={ckd_bal_metrics['acc']:.2f}% | F1={ckd_bal_metrics['macro_f1']:.2f}% [{ckd_bal_metrics['worst_f1']:.2f}%] | Counts={ckd_bal_metrics['counts']}")

    master_results["balanced"] = {
        "standard_kd": std_bal_metrics,
        "post_hoc": post_bal_metrics,
        "calibrated_kd": ckd_bal_metrics
    }

    # -------------------------------------------------------------------------
    # PART B: NATURALLY SKEWED SETUP
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART B: NATURALLY SKEWED SETUP (500, 250, 150, 100 = 1,000 train items)")
    print("="*70)

    skew_probs_raw = softmax_logits(skew_train_logits)
    skew_probs_ckd = softmax_logits(skew_train_logits - null_logits)

    # 1. Standard KD
    print("Training Standard KD (Skewed)...", flush=True)
    m_std_skew, std_skew_logits = train_student(skew_train_texts, skew_probs_raw, return_model=True)
    std_skew_metrics = compute_metrics(test_golds, std_skew_logits.argmax(axis=-1), 4)
    print(f"  Standard KD: Acc={std_skew_metrics['acc']:.2f}% | F1={std_skew_metrics['macro_f1']:.2f}% [{std_skew_metrics['worst_f1']:.2f}%] | Counts={std_skew_metrics['counts']}")

    # 2. Post-Hoc Output Calibration
    print("Evaluating Post-Hoc Output Calibration (Skewed)...", flush=True)
    with torch.no_grad():
        sub_texts = skew_train_texts[:300]
        enc_sub = s_tokenizer(sub_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out_sub = m_std_skew(input_ids=enc_sub["input_ids"], attention_mask=enc_sub["attention_mask"])
        mean_bias_skew = out_sub.logits.mean(dim=0).cpu().numpy()
        del enc_sub, out_sub
    del m_std_skew
    gc.collect()
    torch.cuda.empty_cache()

    post_skew_logits = std_skew_logits - mean_bias_skew
    post_skew_metrics = compute_metrics(test_golds, post_skew_logits.argmax(axis=-1), 4)
    print(f"  Post-Hoc:    Acc={post_skew_metrics['acc']:.2f}% | F1={post_skew_metrics['macro_f1']:.2f}% [{post_skew_metrics['worst_f1']:.2f}%] | Counts={post_skew_metrics['counts']}")

    # 3. Calibrated KD (CKD)
    print("Training Calibrated KD (CKD, Skewed)...", flush=True)
    _, ckd_skew_logits = train_student(skew_train_texts, skew_probs_ckd, return_model=False)
    ckd_skew_metrics = compute_metrics(test_golds, ckd_skew_logits.argmax(axis=-1), 4)
    print(f"  Calibrated KD: Acc={ckd_skew_metrics['acc']:.2f}% | F1={ckd_skew_metrics['macro_f1']:.2f}% [{ckd_skew_metrics['worst_f1']:.2f}%] | Counts={ckd_skew_metrics['counts']}")

    master_results["skewed"] = {
        "standard_kd": std_skew_metrics,
        "post_hoc": post_skew_metrics,
        "calibrated_kd": ckd_skew_metrics
    }

    # Save to disk
    out_file = OUT_DIR / "20newsgroups_balanced_and_skewed_ckd_vs_posthoc.json"
    with open(out_file, "w") as f:
        json.dump(master_results, f, indent=2)

    print("\n" + "="*70)
    print(f"[✓] Completed 20 Newsgroups evaluation! Saved to {out_file}")
    print("="*70)

if __name__ == "__main__":
    main()
