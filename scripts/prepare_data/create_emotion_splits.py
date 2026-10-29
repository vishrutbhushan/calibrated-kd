"""
create_emotion_splits.py
Creates balanced training (1,200 samples, 200/class) and test splits (300 samples, 50/class)
for the 6-class Emotion benchmark from the cached parquet files.
"""

import csv
from pathlib import Path
import pyarrow.parquet as pq

DATA_DIR = Path("data/external_benchmarks")
CACHE_DIR = Path("/home/vishrut/.cache/huggingface/hub")

# 1. Load train parquet
train_p = list(CACHE_DIR.glob("**/split/train-00000-of-00001.parquet"))[0]
table_train = pq.read_table(train_p)
rows_train = table_train.to_pylist()

class_counts_train = {i: 0 for i in range(6)}
balanced_train = []
for idx, r in enumerate(rows_train):
    lbl = r["label"]
    if class_counts_train[lbl] < 200:
        balanced_train.append({
            "sample_id": f"emotion_train_{idx:05d}",
            "text": r["text"],
            "label": lbl
        })
        class_counts_train[lbl] += 1
    if all(c >= 200 for c in class_counts_train.values()):
        break

train_csv = DATA_DIR / "emotion_train_1200.csv"
with open(train_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "text", "label"])
    writer.writeheader()
    writer.writerows(balanced_train)
print(f"[✓] Created {train_csv} with {len(balanced_train)} samples: {class_counts_train}")

# 2. Load test parquet
test_p = list(CACHE_DIR.glob("**/split/test-00000-of-00001.parquet"))[0]
table_test = pq.read_table(test_p)
rows_test = table_test.to_pylist()

class_counts_test = {i: 0 for i in range(6)}
balanced_test = []
for idx, r in enumerate(rows_test):
    lbl = r["label"]
    if class_counts_test[lbl] < 50:
        balanced_test.append({
            "sample_id": f"emotion_test_{idx:05d}",
            "text": r["text"],
            "label": lbl
        })
        class_counts_test[lbl] += 1
    if all(c >= 50 for c in class_counts_test.values()):
        break

test_csv = DATA_DIR / "emotion_test_300.csv"
with open(test_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "text", "label"])
    writer.writeheader()
    writer.writerows(balanced_test)
print(f"[✓] Created {test_csv} with {len(balanced_test)} samples: {class_counts_test}")

