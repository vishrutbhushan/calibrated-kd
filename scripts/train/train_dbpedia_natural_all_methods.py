"""
train_dbpedia_natural_all_methods.py
------------------------------------
Executes complete evaluation of all distillation paradigms on DBpedia-14 under
a naturally skewed / exponential distribution (rho = 10, N=1,400):
  Counts: [248, 208, 174, 146, 122, 102, 86, 72, 60, 50, 42, 35, 30, 25]

Steps:
  1. Creates dbpedia_natural_train_1400.csv from raw DBpedia parquet.
  2. Extracts Teacher (google/gemma-3-4b-it BF16) logits on the 1,400 natural train samples.
     (Reuses cached test targets & null vector from dbpedia_1400_gemma3_4b.npz).
  3. Unloads teacher from GPU memory.
  4. Trains RoBERTa-base across 4 student distillation paradigms:
     - Standard KD (Raw soft targets)
     - Post-Hoc Output Calibration
     - Batch-Calibrated KD (Global training set prior)
     - Calibrated KD (CKD: z - b_null)
  5. Evaluates Teacher Raw, Teacher Calibrated, and all 4 students on the 700 test set items.
  6. Saves full metrics to outputs/results/dbpedia14_natural_5_methods.json
"""

import os
import csv
import json
import random
import gc
from pathlib import Path
import pyarrow.parquet as pq
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
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "datasets" / "dbpedia14"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_MODEL_NAME = "roberta-base"

CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]
VERBALIZERS = [chr(65 + i) for i in range(14)] # A through N

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_prompt(text):
    c_str = "\n".join([f"{v}: {c}" for v, c in zip(VERBALIZERS, CLASSES)])
    return (
        "Classify the text into one of fourteen categories:\n"
        f"{c_str}\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\n"
        "Answer:"
    )

def compute_metrics(golds, preds, num_classes=14):
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
    # Step 1: Create Natural Train Split (N=1,400 with rho=10 decay)
    # =========================================================================
    train_csv_path = DATA_DIR / "dbpedia_natural_train_1400.csv"
    test_csv_path = DATA_DIR / "dbpedia_test_700.csv"
    cached_npz_path = DATA_DIR / "teacher_targets_dbpedia_1400_gemma3_4b.npz"

    target_counts = [248, 208, 174, 146, 122, 102, 86, 72, 60, 50, 42, 35, 30, 25]
    print(f"Target Natural Distribution across 14 classes: {target_counts} (Total={sum(target_counts)})")

    if not train_csv_path.exists():
        print(f"Creating natural train split from raw DBpedia parquet...")
        cache_dir = Path("/home/vishrut/.cache/huggingface/hub/datasets--fancyzhx--dbpedia_14/snapshots/9abd46cf7fc8b4c64290f26993c540b92aa145ac/dbpedia_14")
        table = pq.read_table(cache_dir / "train-00000-of-00001.parquet")
        sampled_rows = []
        curr_counts = {i: 0 for i in range(14)}
        for r in table.to_pylist():
            lbl = r["label"]
            if curr_counts[lbl] < target_counts[lbl]:
                sampled_rows.append({
                    "text": f"{r['title']}: {r['content'].strip()[:600]}",
                    "label": lbl,
                    "class_name": CLASSES[lbl]
                })
                curr_counts[lbl] += 1
            if all(curr_counts[i] >= target_counts[i] for i in range(14)):
                break

        with open(train_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label", "class_name"])
            writer.writeheader()
            writer.writerows(sampled_rows)
        print(f"[✓] Created {train_csv_path} with {len(sampled_rows)} samples.")
    else:
        print(f"[✓] Using existing natural train split: {train_csv_path}")

    with open(train_csv_path) as f:
        train_rows = list(csv.DictReader(f))
    with open(test_csv_path) as f:
        test_rows = list(csv.DictReader(f))

    train_texts = [r["text"] for r in train_rows]
    test_texts = [r["text"] for r in test_rows]
    test_golds = [int(r["label"]) for r in test_rows]

    # =========================================================================
    # Step 2: Extract / Load Teacher Targets
    # =========================================================================
    cached_npz = np.load(cached_npz_path)
    null_logits = cached_npz["null_logits"]
    test_logits_raw = cached_npz["test_logits_raw"]

    natural_npz_path = DATA_DIR / "teacher_targets_dbpedia_natural_1400_gemma3_4b.npz"

    if natural_npz_path.exists():
        print(f"[✓] Loading existing natural teacher targets from {natural_npz_path}...")
        nat_npz = np.load(natural_npz_path)
        train_logits_raw = nat_npz["train_logits_raw"]
    else:
        print(f"Loading Teacher from {TEACHER_MODEL_PATH} in BF16 for natural training targets...", flush=True)
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

        train_prompts = [format_prompt(t) for t in train_texts]
        train_logits_list = []
        batch_size = 4
        print(f"Extracting teacher logits on {len(train_prompts)} natural train items...", flush=True)
        with torch.no_grad():
            for i in range(0, len(train_prompts), batch_size):
                b = train_prompts[i : i + batch_size]
                inp = t_tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
                out = t_model(**inp)
                logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                train_logits_list.append(logits)
                del inp, out
                if (i // batch_size) % 50 == 0:
                    print(f"  Processed {min(i + batch_size, len(train_prompts))}/{len(train_prompts)}...", flush=True)
        train_logits_raw = np.concatenate(train_logits_list, axis=0)

        # Save targets
        np.savez_compressed(natural_npz_path, train_logits_raw=train_logits_raw)
        print(f"[✓] Saved natural teacher targets to {natural_npz_path}")

        del t_model, t_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Softmax conversion
    def softmax_logits(l):
        e = np.exp(l - np.max(l, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    train_probs_raw = softmax_logits(train_logits_raw)

    # CKD Soft Targets: z - b_null
    train_logits_ckd = train_logits_raw - null_logits
    train_probs_ckd = softmax_logits(train_logits_ckd)

    # Batch-Cal Soft Targets: z - log(p_batch) over full train set
    p_batch = train_probs_raw.mean(axis=0, keepdims=True)
    p_batch = np.clip(p_batch, 1e-12, 1.0)
    p_batch /= p_batch.sum(axis=-1, keepdims=True)
    train_logits_batch = train_logits_raw - np.log(p_batch)
    train_probs_batch = softmax_logits(train_logits_batch)

    # Test targets
    test_logits_cal = test_logits_raw - null_logits

    results = {
        "benchmark": "dbpedia14_natural",
        "distribution": "exponential_rho10",
        "train_counts": target_counts,
        "num_classes": 14,
        "classes": CLASSES,
        "num_train": len(train_texts),
        "num_test": len(test_texts),
        "null_gap": float(null_logits.max() - null_logits.min())
    }

    # =========================================================================
    # Step 3: Teacher Baselines on Test Set (N=700)
    # =========================================================================
    t_raw_preds = test_logits_raw.argmax(axis=-1)
    results["teacher_raw"] = compute_metrics(test_golds, t_raw_preds, 14)
    print("\n" + "="*70)
    print(f"TEACHER RAW: Acc={results['teacher_raw']['acc']:.2f}% | Macro-F1={results['teacher_raw']['macro_f1']:.2f}% [{results['teacher_raw']['worst_f1']:.2f}%]")

    t_cal_preds = test_logits_cal.argmax(axis=-1)
    results["teacher_calibrated"] = compute_metrics(test_golds, t_cal_preds, 14)
    print(f"TEACHER CAL: Acc={results['teacher_calibrated']['acc']:.2f}% | Macro-F1={results['teacher_calibrated']['macro_f1']:.2f}% [{results['teacher_calibrated']['worst_f1']:.2f}%]")
    print("="*70)

    # =========================================================================
    # Step 4: Student Distillation on RoBERTa-base
    # =========================================================================
    s_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    max_len = 512

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = s_tokenizer(txts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["probs"] = p
        return enc

    def get_eval_logits(model):
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

    def train_student_variant(probs_arr, return_model=False):
        set_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(
            STUDENT_MODEL_NAME, num_labels=14, local_files_only=True
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

        logits = get_eval_logits(model)
        if return_model:
            return model, logits
        else:
            del model, optimizer, scheduler, loader
            gc.collect()
            torch.cuda.empty_cache()
            return None, logits

    # Method 1: Standard KD
    print("\nTraining 1/4: Standard KD...", flush=True)
    m_std, std_logits = train_student_variant(train_probs_raw, return_model=True)
    std_preds = std_logits.argmax(axis=-1)
    results["standard_kd"] = compute_metrics(test_golds, std_preds, 14)
    print(f"  Standard KD -> Acc: {results['standard_kd']['acc']:.2f}% | F1: {results['standard_kd']['macro_f1']:.2f}% [{results['standard_kd']['worst_f1']:.2f}%]")
    print(f"  Standard KD Pred Counts: {results['standard_kd']['counts']}", flush=True)

    # Method 2: Post-Hoc Output Calibration
    print("\nEvaluating 2/4: Post-Hoc Output Calibration...", flush=True)
    with torch.no_grad():
        sample_subset = train_texts[:300]
        enc_sub = s_tokenizer(sample_subset, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out_sub = m_std(input_ids=enc_sub["input_ids"], attention_mask=enc_sub["attention_mask"])
        student_mean_logits = out_sub.logits.mean(dim=0).cpu().numpy()
        del enc_sub, out_sub
    del m_std
    gc.collect()
    torch.cuda.empty_cache()

    post_logits = std_logits - student_mean_logits
    post_preds = post_logits.argmax(axis=-1)
    results["post_hoc_output_cal"] = compute_metrics(test_golds, post_preds, 14)
    print(f"  Post-Hoc Cal -> Acc: {results['post_hoc_output_cal']['acc']:.2f}% | F1: {results['post_hoc_output_cal']['macro_f1']:.2f}% [{results['post_hoc_output_cal']['worst_f1']:.2f}%]")
    print(f"  Post-Hoc Pred Counts: {results['post_hoc_output_cal']['counts']}", flush=True)

    # Method 3: Batch-Calibrated KD
    print("\nTraining 3/4: Batch-Calibrated KD...", flush=True)
    _, batch_logits = train_student_variant(train_probs_batch, return_model=False)
    batch_preds = batch_logits.argmax(axis=-1)
    results["batch_cal_kd"] = compute_metrics(test_golds, batch_preds, 14)
    print(f"  Batch-Cal KD -> Acc: {results['batch_cal_kd']['acc']:.2f}% | F1: {results['batch_cal_kd']['macro_f1']:.2f}% [{results['batch_cal_kd']['worst_f1']:.2f}%]")
    print(f"  Batch-Cal Pred Counts: {results['batch_cal_kd']['counts']}", flush=True)

    # Method 4: Calibrated KD (CKD)
    print("\nTraining 4/4: Calibrated KD (CKD)...", flush=True)
    _, ckd_logits = train_student_variant(train_probs_ckd, return_model=False)
    ckd_preds = ckd_logits.argmax(axis=-1)
    results["calibrated_kd"] = compute_metrics(test_golds, ckd_preds, 14)
    print(f"  Calibrated KD -> Acc: {results['calibrated_kd']['acc']:.2f}% | F1: {results['calibrated_kd']['macro_f1']:.2f}% [{results['calibrated_kd']['worst_f1']:.2f}%]")
    print(f"  Calibrated KD Pred Counts: {results['calibrated_kd']['counts']}", flush=True)

    # =========================================================================
    # Step 5: Save Summary
    # =========================================================================
    out_file = OUT_DIR / "dbpedia14_natural_5_methods.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*70)
    print(f"[✓] Completed DBpedia-14 Natural run across all methods! Saved to {out_file}")
    print("="*70)

if __name__ == "__main__":
    main()
