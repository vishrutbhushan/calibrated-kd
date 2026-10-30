"""
create_trec_splits.py
Downloads the official TREC Question Classification dataset and creates:
1) trec_train_1200.csv (balanced training set: 200 per class across 6 coarse categories)
2) trec_test_500.csv (canonical test set of 500 questions)
3) trec_test_balanced_300.csv (balanced test set of 300 questions, 50 per class, sampled from held-out items)

Classes:
0: Abbreviation (ABBR)
1: Entity (ENTY)
2: Description (DESC)
3: Human (HUM)
4: Location (LOC)
5: Numeric (NUM)
"""

import os
import csv
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

COARSE_MAP = {
    "ABBR": 0,
    "ENTY": 1,
    "DESC": 2,
    "HUM": 3,
    "LOC": 4,
    "NUM": 5
}
CLASS_NAMES = ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]

TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
TEST_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"

def download_and_parse(url):
    req = urllib.request.urlopen(url)
    lines = req.read().decode("utf-8", errors="replace").splitlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        coarse = parts[0].split(":")[0]
        question = parts[1].strip()
        data.append({"label": COARSE_MAP[coarse], "text": question})
    return data

def main():
    print("Downloading TREC dataset from CogComp repository...", flush=True)
    raw_train = download_and_parse(TRAIN_URL)
    raw_test = download_and_parse(TEST_URL)

    print(f"Total raw train: {len(raw_train)}, raw test: {len(raw_test)}")

    # 1. Canonical Test Set (500 samples)
    test_500_path = BENCHMARK_DIR / "trec_test_500.csv"
    with open(test_500_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "text", "label"])
        writer.writeheader()
        for idx, item in enumerate(raw_test):
            writer.writerow({
                "sample_id": f"trec_test_{idx:04d}",
                "text": item["text"],
                "label": item["label"]
            })
    print(f"[✓] Created canonical test set: {test_500_path} (N={len(raw_test)})")

    # Group train data by class
    by_class = defaultdict(list)
    for item in raw_train:
        by_class[item["label"]].append(item["text"])

    print("\nRaw Train Class Distribution:")
    for c_id in range(6):
        print(f"  Class {c_id} [{CLASS_NAMES[c_id]:12s}]: {len(by_class[c_id])} items")

    # 2. Balanced Train Set (1,200 samples = 200 per class)
    # For classes with >= 200 items, take first 200. For ABBR (86 items), sample 200 with replacement.
    balanced_train = []
    for c_id in range(6):
        items = by_class[c_id]
        if len(items) >= 200:
            chosen = items[:200]
        else:
            # deterministic oversampling
            rng = random.Random(42 + c_id)
            chosen = [rng.choice(items) for _ in range(200)]
        for text in chosen:
            balanced_train.append({
                "text": text,
                "label": c_id
            })

    # Shuffle training set
    random.Random(42).shuffle(balanced_train)

    train_1200_path = BENCHMARK_DIR / "trec_train_1200.csv"
    with open(train_1200_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "text", "label"])
        writer.writeheader()
        for idx, item in enumerate(balanced_train):
            writer.writerow({
                "sample_id": f"trec_train_{idx:04d}",
                "text": item["text"],
                "label": item["label"]
            })
    print(f"[✓] Created balanced train set: {train_1200_path} (N={len(balanced_train)}, 200/class)")

if __name__ == "__main__":
    main()

