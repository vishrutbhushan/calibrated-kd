"""
prepare_3_new_benchmarks.py
---------------------------
Prepares 3 NEW authentic naturally skewed benchmarks (5 to 6 classes, no neutral):
1. pubmed_rct: 5 classes (Methods, Results, Conclusions, Background, Objective)
2. citation_intent: 6 classes (Background, Uses, CompareOrContrast, Motivation, Extends, Future)
3. app_reviews: 5 classes (1 Star, 2 Stars, 3 Stars, 4 Stars, 5 Stars)

Standardized format:
- Exactly 75% train (1,200) : 25% test (400)
- Stratified sampling with seed=42
- Saved to data/datasets/natural_benchmarks/<name>/
"""

import os
import csv
import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"

SEED = 42

def save_csv(path, texts, labels, prefix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "text", "label"])
        for idx, (t, l) in enumerate(zip(texts, labels)):
            clean_text = " ".join(t.strip().split())[:600]
            writer.writerow([f"{prefix}_{idx:05d}", clean_text, l])
    print(f"[✓] Saved {len(texts)} samples to {path}")
    counts = Counter(labels)
    print(f"    Class distribution: {dict(sorted(counts.items()))}")

def prepare_pubmed_rct():
    print("\n" + "="*50)
    print("Preparing PubMed RCT (5 classes)...")
    ds = load_dataset("armanc/pubmed-rct20k", split="train")
    # label mapping:
    # 0: background, 1: objective, 2: methods, 3: results, 4: conclusions
    label_map = {"background": 0, "objective": 1, "methods": 2, "results": 3, "conclusions": 4}
    
    texts, labels = [], []
    for item in ds:
        lbl = item["label"].lower().strip()
        if lbl in label_map and len(item["text"].strip()) > 10:
            texts.append(item["text"].strip())
            labels.append(label_map[lbl])

    # Stratified 1,600 samples
    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    # 1,200 train : 400 test
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    out_dir = DATA_BASE / "pubmed_rct"
    save_csv(out_dir / "train_1200.csv", tr_txt, tr_lbl, "pubmed_tr")
    save_csv(out_dir / "test_400.csv", te_txt, te_lbl, "pubmed_te")

def prepare_citation_intent():
    print("\n" + "="*50)
    print("Preparing Citation Intent (6 classes)...")
    ds = load_dataset("zapsdcn/citation_intent", split="train")
    label_map = {
        "Background": 0,
        "Uses": 1,
        "CompareOrContrast": 2,
        "Motivation": 3,
        "Extends": 4,
        "Future": 5
    }
    texts, labels = [], []
    for item in ds:
        lbl = item["label"].strip()
        if lbl in label_map and len(item["text"].strip()) > 10:
            texts.append(item["text"].strip())
            labels.append(label_map[lbl])

    # 1600 samples
    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    out_dir = DATA_BASE / "citation_intent"
    save_csv(out_dir / "train_1200.csv", tr_txt, tr_lbl, "cite_tr")
    save_csv(out_dir / "test_400.csv", te_txt, te_lbl, "cite_te")

def prepare_app_reviews():
    print("\n" + "="*50)
    print("Preparing App Reviews (5 classes)...")
    ds = load_dataset("sealuzh/app_reviews", split="train")
    # star ratings: 1, 2, 3, 4, 5 -> map to 0, 1, 2, 3, 4
    texts, labels = [], []
    for item in ds:
        star = item["star"]
        rev = item["review"].strip()
        if star in [1, 2, 3, 4, 5] and len(rev) > 15:
            texts.append(rev)
            labels.append(star - 1)  # 0 to 4

    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    out_dir = DATA_BASE / "app_reviews"
    save_csv(out_dir / "train_1200.csv", tr_txt, tr_lbl, "app_tr")
    save_csv(out_dir / "test_400.csv", te_txt, te_lbl, "app_te")

def main():
    prepare_pubmed_rct()
    prepare_citation_intent()
    prepare_app_reviews()
    print("\n[✓] All 3 new benchmarks successfully prepared!")

if __name__ == "__main__":
    main()
