"""
create_agnews_1000_splits.py
Creates a balanced test split of exactly 1,000 samples (250 per class) from ag_news_test.parquet.
"""

import csv
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path("data/external_benchmarks")
table = pq.read_table(DATA_DIR / "ag_news_test.parquet")
rows = table.to_pylist()

class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
balanced_test = []

for idx, r in enumerate(rows):
    lbl = r["label"]
    if class_counts[lbl] < 250:
        balanced_test.append({
            "sample_id": f"agnews_test_{idx:05d}",
            "text": r["text"],
            "label": lbl
        })
        class_counts[lbl] += 1
    if all(c >= 250 for c in class_counts.values()):
        break

out_path = DATA_DIR / "ag_news_test_1000.csv"
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "text", "label"])
    writer.writeheader()
    writer.writerows(balanced_test)

print(f"[✓] Created {out_path} with {len(balanced_test)} samples (Counts: {class_counts})")

