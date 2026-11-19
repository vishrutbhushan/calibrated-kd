"""
prepare_natural_benchmarks.py
-----------------------------
Prepares 5 authentic naturally skewed benchmarks with:
  - Standardized 75% Train (N=1,200) : 25% Test (N=400) split
  - Stratified random sampling (seed=42)
  - Strict preservation of raw real-world class distributions
  - Zero artificial balancing or downsampling
  - Moderate classes (4 to 6 classes), No "Neutral" class

Output directory: data/datasets/natural_benchmarks/<benchmark>/
"""

import os
import csv
import random
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
OUT_BASE.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def save_csv(path, samples):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "text", "label"])
        writer.writeheader()
        for s in samples:
            writer.writerow(s)

def stratified_sample_split(texts, labels, task_name, n_train=1200, n_test=400):
    total_needed = n_train + n_test
    assert len(texts) >= total_needed, f"Not enough samples in {task_name}: {len(texts)} < {total_needed}"

    # If dataset has more than needed, do stratified sub-sample of total_needed first
    if len(texts) > total_needed:
        texts_sub, _, labels_sub, _ = train_test_split(
            texts, labels, train_size=total_needed, stratify=labels, random_state=SEED
        )
    else:
        texts_sub, labels_sub = texts, labels

    # Now split into train and test
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        texts_sub, labels_sub, train_size=n_train, test_size=n_test, stratify=labels_sub, random_state=SEED
    )

    tr_samples = [{"sample_id": f"{task_name}_tr_{i:05d}", "text": t, "label": l} for i, (t, l) in enumerate(zip(tr_txt, tr_lbl))]
    te_samples = [{"sample_id": f"{task_name}_te_{i:05d}", "text": t, "label": l} for i, (t, l) in enumerate(zip(te_txt, te_lbl))]

    task_dir = OUT_BASE / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    save_csv(task_dir / "train_1200.csv", tr_samples)
    save_csv(task_dir / "test_400.csv", te_samples)

    print(f"\n[✓] Prepared {task_name}:")
    tr_dist = Counter(tr_lbl)
    te_dist = Counter(te_lbl)
    print(f"  Train (N={len(tr_samples)}): {dict(sorted(tr_dist.items()))}")
    print(f"  Test  (N={len(te_samples)}): {dict(sorted(te_dist.items()))}")

# 1. TweetEval Emotion (4 classes: 0: anger, 1: joy, 2: optimism, 3: sadness)
print("Processing 1/5: TweetEval Emotion...", flush=True)
ds_te = load_dataset('cardiffnlp/tweet_eval', 'emotion')
texts_te = list(ds_te['train']['text']) + list(ds_te['test']['text'])
labels_te = list(ds_te['train']['label']) + list(ds_te['test']['label'])
stratified_sample_split(texts_te, labels_te, "tweeteval_emotion", 1200, 400)

# 2. BBC News (5 classes: business, entertainment, politics, sport, tech)
print("\nProcessing 2/5: BBC News...", flush=True)
ds_bbc = load_dataset('SetFit/bbc-news')
texts_bbc = list(ds_bbc['train']['text']) + list(ds_bbc['test']['text'])
labels_bbc = list(ds_bbc['train']['label']) + list(ds_bbc['test']['label'])
stratified_sample_split(texts_bbc, labels_bbc, "bbc_news", 1200, 400)

# 3. Twitter Emotion (6 classes: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise)
print("\nProcessing 3/5: Twitter Emotion (Saravia)...", flush=True)
ds_emo = load_dataset('dair-ai/emotion', 'split')
texts_emo = list(ds_emo['train']['text']) + list(ds_emo['test']['text'])
labels_emo = list(ds_emo['train']['label']) + list(ds_emo['test']['label'])
stratified_sample_split(texts_emo, labels_emo, "twitter_emotion", 1200, 400)

# 4. TREC-6 (6 classes: 0: ABBR, 1: ENTY, 2: DESC, 3: HUM, 4: LOC, 5: NUM)
print("\nProcessing 4/5: TREC-6...", flush=True)
trec_train_file = ROOT / "data" / "benchmarks" / "skewed" / "trec_natural_train_1200.csv"
trec_test_file = ROOT / "data" / "benchmarks" / "trec_test_500.csv"
texts_trec, labels_trec = [], []
with open(trec_train_file, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        texts_trec.append(row["text"])
        labels_trec.append(int(row["label"]))
with open(trec_test_file, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        texts_trec.append(row["text"])
        labels_trec.append(int(row["label"]))
stratified_sample_split(texts_trec, labels_trec, "trec6", 1200, 400)

# 5. PRAXIS (5 classes: conditional, direct_imperative, interrogative, modal_deontic, optative)
print("\nProcessing 5/5: PRAXIS...", flush=True)
praxis_train_file = ROOT / "data" / "datasets" / "praxis" / "praxis_train_1000.csv"
praxis_test_file = ROOT / "data" / "datasets" / "praxis" / "praxis_golden_test_300.csv"
praxis_extra_file = ROOT / "data" / "datasets" / "praxis" / "praxis_test_150.csv"

praxis_raw = []
for p in [praxis_train_file, praxis_test_file, praxis_extra_file]:
    with open(p, "r", encoding="utf-8") as f:
        praxis_raw.extend(list(csv.DictReader(f)))

class_map = {
    "direct_imperative": 0,
    "interrogative": 1,
    "modal_deontic": 2,
    "conditional": 3,
    "optative": 4
}
texts_praxis = [r["review_en"] for r in praxis_raw]
labels_praxis = [class_map[r["tier1_form"]] for r in praxis_raw]
# PRAXIS total is 1450, we use 1000 train / 350 test (74:26 ratio)
stratified_sample_split(texts_praxis, labels_praxis, "praxis", 1000, 350)

print("\n" + "="*70)
print("[✓] All 5 naturally skewed benchmarks prepared successfully!")
print("="*70)
