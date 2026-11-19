"""
prepare_3_additional_benchmarks.py
----------------------------------
Prepares 3 NEW naturally skewed, authentic NLP classification benchmarks (3 to 6 classes, no neutral):
1. go_emotions_ekman: 6 classes (Anger, Disgust, Fear, Joy, Sadness, Surprise) from Reddit comments
2. sst4_finegrained: 4 classes (Very Negative, Negative, Positive, Very Positive) from SST-5 (Neutral excluded)
3. student_questions: 4 classes (Biology, Physics, Maths, Chemistry) STEM academic questions

Standardized constraints:
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

def prepare_go_emotions_ekman():
    print("\n" + "="*50)
    print("Preparing GoEmotions Ekman (6 classes)...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    label_names = ds.features["labels"].feature.names

    ekman_mapping = {
        "Anger": ["anger", "annoyance", "disapproval"],
        "Disgust": ["disgust"],
        "Fear": ["fear", "nervousness"],
        "Joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
        "Sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
        "Surprise": ["surprise", "realization", "confusion", "curiosity"]
    }
    ekman_order = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    name_to_id = {k: idx for idx, k in enumerate(ekman_order)}

    sub_to_ekman = {}
    for ek, sublist in ekman_mapping.items():
        for sub in sublist:
            sub_to_ekman[sub] = ek

    texts, labels = [], []
    for row in ds:
        row_labels = [label_names[idx] for idx in row["labels"]]
        ek_labels = set(sub_to_ekman[l] for l in row_labels if l in sub_to_ekman)
        if len(ek_labels) == 1:
            ek_name = list(ek_labels)[0]
            txt = row["text"].strip()
            if len(txt) > 10:
                texts.append(txt)
                labels.append(name_to_id[ek_name])

    print(f"Total single-label Ekman pool: {len(texts)}")
    print(f"Raw pool distribution: {dict(sorted(Counter(labels).items()))}")

    # Stratified 1,600 samples
    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    out_dir = DATA_BASE / "go_emotions_ekman"
    save_csv(out_dir / "train_1200.csv", tr_txt, tr_lbl, "goemo_tr")
    save_csv(out_dir / "test_400.csv", te_txt, te_lbl, "goemo_te")

def prepare_sst4_finegrained():
    print("\n" + "="*50)
    print("Preparing SST-4 Fine-Grained (4 classes, Neutral excluded)...")
    ds = load_dataset("SetFit/sst5", split="train")
    # SST-5 original labels:
    # 0: very negative -> 0
    # 1: negative      -> 1
    # 2: neutral       -> EXCLUDE
    # 3: positive      -> 2
    # 4: very positive -> 3
    sst4_map = {0: 0, 1: 1, 3: 2, 4: 3}

    texts, labels = [], []
    for item in ds:
        lbl = item["label"]
        txt = item["text"].strip()
        if lbl in sst4_map and len(txt) > 10:
            texts.append(txt)
            labels.append(sst4_map[lbl])

    print(f"Total SST-4 pool: {len(texts)}")
    print(f"Raw pool distribution: {dict(sorted(Counter(labels).items()))}")

    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    out_dir = DATA_BASE / "sst4_finegrained"
    save_csv(out_dir / "train_1200.csv", tr_txt, tr_lbl, "sst4_tr")
    save_csv(out_dir / "test_400.csv", te_txt, te_lbl, "sst4_te")

def prepare_student_questions():
    print("\n" + "="*50)
    print("Preparing Student Question Categories (4 classes: Biology, Physics, Maths, Chemistry)...")
    ds = load_dataset("SetFit/student-question-categories", split="train")
    # label mapping:
    # 0: Chemistry, 1: Physics, 2: Maths, 3: Biology
    # Let's standardize:
    # 0: Biology, 1: Physics, 2: Maths, 3: Chemistry
    orig_to_new = {3: 0, 1: 1, 2: 2, 0: 3}

    texts, labels = [], []
    for item in ds:
        lbl = item["label"]
        txt = item["text"].strip()
        if lbl in orig_to_new and len(txt) > 15:
            texts.append(txt)
            labels.append(orig_to_new[lbl])

    print(f"Total Student Questions pool: {len(texts)}")
    print(f"Raw pool distribution: {dict(sorted(Counter(labels).items()))}")

    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    out_dir = DATA_BASE / "student_questions"
    save_csv(out_dir / "train_1200.csv", tr_txt, tr_lbl, "stud_tr")
    save_csv(out_dir / "test_400.csv", te_txt, te_lbl, "stud_te")

def main():
    prepare_go_emotions_ekman()
    prepare_sst4_finegrained()
    prepare_student_questions()
    print("\n[✓] All 3 additional benchmarks successfully prepared!")

if __name__ == "__main__":
    main()
