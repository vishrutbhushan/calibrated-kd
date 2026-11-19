"""
prepare_sst2_natural.py
-----------------------
Prepares standardized natural SST-2 benchmark:
- 1,200 Train (75%) : 400 Test (25%)
- Stratified sampling with seed=42
- Saved to data/datasets/natural_benchmarks/sst2/
"""

import csv
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "datasets" / "natural_benchmarks" / "sst2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def save_csv(path, texts, labels, prefix):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "text", "label"])
        for idx, (t, l) in enumerate(zip(texts, labels)):
            clean_text = " ".join(t.strip().split())[:600]
            writer.writerow([f"{prefix}_{idx:05d}", clean_text, l])
    print(f"[✓] Saved {len(texts)} samples to {path}")
    counts = Counter(labels)
    print(f"    Class distribution: {dict(sorted(counts.items()))}")

def main():
    print("Preparing natural SST-2 benchmark...")
    ds = load_dataset("SetFit/sst2", split="train")
    texts, labels = [], []
    for item in ds:
        txt = item["text"].strip()
        lbl = item["label"]
        if lbl in [0, 1] and len(txt) > 10:
            texts.append(txt)
            labels.append(lbl)

    print(f"Total SST-2 pool: {len(texts)}")
    print(f"Raw pool distribution: {dict(sorted(Counter(labels).items()))}")

    # Stratified sample 1,600 samples
    sub_texts, _, sub_labels, _ = train_test_split(
        texts, labels, train_size=1600, stratify=labels, random_state=SEED
    )
    # 1,200 train : 400 test
    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        sub_texts, sub_labels, train_size=1200, test_size=400, stratify=sub_labels, random_state=SEED
    )

    save_csv(OUT_DIR / "train_1200.csv", tr_txt, tr_lbl, "sst2_tr")
    save_csv(OUT_DIR / "test_400.csv", te_txt, te_lbl, "sst2_te")
    print("[✓] SST-2 natural benchmark successfully prepared!")

if __name__ == "__main__":
    main()
