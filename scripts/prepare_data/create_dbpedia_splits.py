"""
create_dbpedia_splits.py
------------------------
Creates balanced train and test splits for DBpedia-14:
  - 100 samples per class x 14 classes = 1,400 train items
  - 50 samples per class x 14 classes = 700 test items
Saves to data/external_benchmarks/ and /home/vishrut/calibrated-kd/data/benchmarks/
"""

import csv
import pyarrow.parquet as pq
from pathlib import Path
from collections import Counter

CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]

cache_dir = Path("/home/vishrut/.cache/huggingface/hub/datasets--fancyzhx--dbpedia_14/snapshots/9abd46cf7fc8b4c64290f26993c540b92aa145ac/dbpedia_14")

# 1. Train split
train_table = pq.read_table(cache_dir / "train-00000-of-00001.parquet")
train_rows = train_table.to_pylist()

balanced_train = []
counts_train = {i: 0 for i in range(14)}
for r in train_rows:
    lbl = r["label"]
    if counts_train[lbl] < 100:
        balanced_train.append({
            "text": f"{r['title']}: {r['content'].strip()[:600]}",
            "label": lbl,
            "class_name": CLASSES[lbl]
        })
        counts_train[lbl] += 1
    if all(c >= 100 for c in counts_train.values()):
        break

# 2. Test split
test_table = pq.read_table(cache_dir / "test-00000-of-00001.parquet")
test_rows = test_table.to_pylist()

balanced_test = []
counts_test = {i: 0 for i in range(14)}
for r in test_rows:
    lbl = r["label"]
    if counts_test[lbl] < 50:
        balanced_test.append({
            "text": f"{r['title']}: {r['content'].strip()[:600]}",
            "label": lbl,
            "class_name": CLASSES[lbl]
        })
        counts_test[lbl] += 1
    if all(c >= 50 for c in counts_test.values()):
        break

print(f"Train balanced items: {len(balanced_train)} ({Counter(r['label'] for r in balanced_train)})")
print(f"Test balanced items:  {len(balanced_test)} ({Counter(r['label'] for r in balanced_test)})")

# Save outputs
dest_dirs = [
    Path("data/external_benchmarks"),
    Path("/home/vishrut/calibrated-kd/data/benchmarks")
]

for d in dest_dirs:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "dbpedia_train_1400.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "class_name"])
        writer.writeheader()
        writer.writerows(balanced_train)

    with open(d / "dbpedia_test_700.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "class_name"])
        writer.writeheader()
        writer.writerows(balanced_test)

    print(f"[✓] Saved splits to {d}")

