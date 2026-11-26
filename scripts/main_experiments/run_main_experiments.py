"""
run_main_experiments.py
-----------------------
Unified, strictly sequential runner for all 6 main benchmark experiments in
Section 6 of the Calibrated Knowledge Distillation (CKD) paper.

Benchmarks evaluated (in order):
  1. student_questions (K=4) - SetFit/student-question-categories
  2. trec6             (K=6) - CogComp/trec
  3. app_reviews       (K=5) - sealuzh/app_reviews
  4. twitter_emotion   (K=6) - dair-ai/emotion
  5. tweeteval_emotion (K=4) - cardiffnlp/tweet_eval (emotion)
  6. bbc_news          (K=5) - SetFit/bbc-news

Methods evaluated per benchmark:
  - Teacher Base (Gemma-3-4B-IT Raw Zero-Shot)
  - Teacher Null-Calibrated (Gemma-3-4B-IT with "N/A" probe subtraction)
  - Student: Standard KD (RoBERTa-base on raw teacher soft targets)
  - Student: Batch-Calibrated KD (RoBERTa-base with empirical data-mean centering)
  - Student: Null Calibrated KD (Ours) (RoBERTa-base with "N/A" probe centering)

Execution & Safety Guarantees:
  - Single command execution across all 6 main datasets (no ablations).
  - Verifies local dataset splits and downloads from HuggingFace if missing.
  - Strictly sequential: only one model (Teacher or Student) in GPU memory at a time.
  - Single timestamped log file in logs/main_experiments_<timestamp>.log.
  - Single output directory in outputs/main_experiments/.
"""

import os
import sys
import gc
import csv
import json
import time
import random
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Prevent memory fragmentation on single-GPU workstations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
OUTPUT_DIR = ROOT / "outputs" / "main_experiments"
LOGS_DIR = ROOT / "logs"

TEACHER_MODEL_PATH = (
    "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/"
    "snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
)
STUDENT_MODEL_NAME = "roberta-base"

SEED = 42

DATASET_ORDER = [
    "student_questions",
    "trec6",
    "app_reviews",
    "twitter_emotion",
    "tweeteval_emotion",
    "bbc_news",
]

BENCHMARK_CONFIGS = {
    "student_questions": {
        "num_classes": 4,
        "class_names": ["Biology", "Physics", "Maths", "Chemistry"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": (
            "Classify the academic question into the correct subject:\n"
            "A: Biology\nB: Physics\nC: Maths\nD: Chemistry"
        ),
        "hf_source": "SetFit/student-question-categories",
    },
    "trec6": {
        "num_classes": 6,
        "class_names": [
            "Abbreviation",
            "Entity",
            "Description",
            "Human",
            "Location",
            "Numeric",
        ],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": (
            "Classify the question into one of six question types:\n"
            "A: Abbreviation\nB: Entity\nC: Description\n"
            "D: Human\nE: Location\nF: Numeric"
        ),
        "hf_source": "CogComp/trec",
    },
    "app_reviews": {
        "num_classes": 5,
        "class_names": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": (
            "Classify the customer app review into one of five star ratings:\n"
            "A: 1 Star\nB: 2 Stars\nC: 3 Stars\nD: 4 Stars\nE: 5 Stars"
        ),
        "hf_source": "sealuzh/app_reviews",
    },
    "twitter_emotion": {
        "num_classes": 6,
        "class_names": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": (
            "Classify the tweet into one of six emotions:\n"
            "A: Sadness\nB: Joy\nC: Love\nD: Anger\nE: Fear\nF: Surprise"
        ),
        "hf_source": "dair-ai/emotion",
    },
    "tweeteval_emotion": {
        "num_classes": 4,
        "class_names": ["Anger", "Joy", "Optimism", "Sadness"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": (
            "Classify the tweet into one of four emotions:\n"
            "A: Anger\nB: Joy\nC: Optimism\nD: Sadness"
        ),
        "hf_source": "cardiffnlp/tweet_eval (emotion)",
    },
    "bbc_news": {
        "num_classes": 5,
        "class_names": ["Tech", "Business", "Sport", "Entertainment", "Politics"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": (
            "Classify the news article into one of five categories:\n"
            "A: Tech\nB: Business\nC: Sport\nD: Entertainment\nE: Politics"
        ),
        "hf_source": "SetFit/bbc-news",
    },
}


class TeeLogger:
    """Duplicates stdout and stderr to both the console and a timestamped log file."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self.terminal.fileno()

    def close(self):
        self.log_file.close()


class KDDataset(Dataset):
    """PyTorch Dataset wrapping input texts and teacher soft probability targets."""

    def __init__(self, texts: list[str], probs: np.ndarray):
        self.texts = texts
        self.probs = probs

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.probs[idx]


def set_seed(seed: int = SEED) -> None:
    """Fixes Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_vram() -> None:
    """Runs garbage collection and clears the PyTorch CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes numerically stable softmax probabilities from raw logits."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def compute_metrics(golds: list[int], preds: np.ndarray, num_classes: int) -> dict:
    """
    Computes evaluation metrics: Macro-F1 (primary), Accuracy (secondary),
    worst-class F1, per-class F1, and predicted class distribution counts.
    """
    acc = float(accuracy_score(golds, preds) * 100.0)
    macro_f1 = float(f1_score(golds, preds, average="macro") * 100.0)
    per_class = (f1_score(golds, preds, average=None) * 100.0).tolist()
    worst_f1 = float(np.min(per_class))
    counts = np.bincount(preds, minlength=num_classes).tolist()
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "worst_f1": worst_f1,
        "per_class_f1": per_class,
        "counts": counts,
    }


def save_split_csv(path: Path, texts: list[str], labels: list[int], prefix: str) -> None:
    """Writes a stratified dataset split (sample_id, text, label) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "text", "label"])
        for idx, (text, label) in enumerate(zip(texts, labels)):
            writer.writerow([f"{prefix}_{idx:05d}", text, label])


def create_stratified_splits(
    texts: list[str], labels: list[int], task_name: str, n_train: int = 1200, n_test: int = 400
) -> None:
    """Performs stratified 1,200/400 sampling preserving the natural class distribution."""
    total_needed = n_train + n_test
    if len(texts) > total_needed:
        texts_sub, _, labels_sub, _ = train_test_split(
            texts, labels, train_size=total_needed, stratify=labels, random_state=SEED
        )
    else:
        texts_sub, labels_sub = texts, labels

    tr_txt, te_txt, tr_lbl, te_lbl = train_test_split(
        texts_sub,
        labels_sub,
        train_size=n_train,
        test_size=n_test,
        stratify=labels_sub,
        random_state=SEED,
    )

    task_dir = DATA_BASE / task_name
    save_split_csv(task_dir / "train_1200.csv", tr_txt, tr_lbl, f"{task_name}_tr")
    save_split_csv(task_dir / "test_400.csv", te_txt, te_lbl, f"{task_name}_te")
    print(f"  [✓] Saved stratified splits for {task_name} (Train={len(tr_txt)}, Test={len(te_txt)}).")


def download_and_prepare_dataset(task_name: str) -> None:
    """Downloads a missing benchmark from HuggingFace and prepares stratified 1200/400 CSVs."""
    from datasets import load_dataset

    print(f"  [↓] Downloading '{task_name}' from HuggingFace ({BENCHMARK_CONFIGS[task_name]['hf_source']})...")

    if task_name == "student_questions":
        ds = load_dataset("SetFit/student-question-categories", split="train")
        orig_to_new = {3: 0, 1: 1, 2: 2, 0: 3}  # Biology, Physics, Maths, Chemistry
        texts, labels = [], []
        for item in ds:
            txt = item["text"].strip()
            if item["label"] in orig_to_new and len(txt) > 15:
                texts.append(txt)
                labels.append(orig_to_new[item["label"]])

    elif task_name == "trec6":
        ds = load_dataset("CogComp/trec", revision="refs/convert/parquet")
        # CogComp/trec coarse_label: 0:ABBR, 1:ENTY, 2:DESC, 3:HUM, 4:LOC, 5:NUM
        texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
        labels = list(ds["train"]["coarse_label"]) + list(ds["test"]["coarse_label"])

    elif task_name == "app_reviews":
        ds = load_dataset("sealuzh/app_reviews", split="train")
        texts, labels = [], []
        for item in ds:
            star = item["star"]
            rev = item["review"].strip()
            if star in [1, 2, 3, 4, 5] and len(rev) > 15:
                texts.append(rev)
                labels.append(star - 1)

    elif task_name == "twitter_emotion":
        ds = load_dataset("dair-ai/emotion", "split")
        texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
        labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    elif task_name == "tweeteval_emotion":
        ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
        texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
        labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    elif task_name == "bbc_news":
        ds = load_dataset("SetFit/bbc-news")
        texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
        labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    else:
        raise ValueError(f"Unknown dataset: {task_name}")

    create_stratified_splits(texts, labels, task_name)


def verify_all_datasets(force_download: bool = False) -> None:
    """
    Verifies that all 6 main benchmark datasets exist locally with 1,200 train
    and 400 test samples. Downloads and prepares any missing datasets from HuggingFace.
    """
    print("=" * 75)
    print("STEP 0: VERIFYING & PREPARING ALL 6 MAIN BENCHMARK DATASETS")
    print("=" * 75)

    for task_name in DATASET_ORDER:
        task_dir = DATA_BASE / task_name
        train_file = task_dir / "train_1200.csv"
        test_file = task_dir / "test_400.csv"

        if force_download or not (train_file.exists() and test_file.exists()):
            download_and_prepare_dataset(task_name)
        else:
            with open(train_file, "r", encoding="utf-8") as f:
                n_tr = sum(1 for _ in csv.DictReader(f))
            with open(test_file, "r", encoding="utf-8") as f:
                n_te = sum(1 for _ in csv.DictReader(f))
            print(f"  [✓] Verified {task_name:<18} (Train={n_tr}, Test={n_te})")


def load_dataset_splits(task_name: str) -> tuple[list[str], list[int], list[str], list[int]]:
    """Loads train and test texts and gold labels for a given benchmark."""
    task_dir = DATA_BASE / task_name
    train_file = task_dir / "train_1200.csv"
    test_file = task_dir / "test_400.csv"

    train_texts, train_golds = [], []
    with open(train_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_texts.append(row["text"])
            train_golds.append(int(row["label"]))

    test_texts, test_golds = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_texts.append(row["text"])
            test_golds.append(int(row["label"]))

    return train_texts, train_golds, test_texts, test_golds


def format_teacher_prompt(text: str, config: dict) -> str:
    """Formats an input text into the multiple-choice single-letter teacher prompt."""
    return (
        f"{config['task_desc']}\n\n"
        f"Text: {text.strip()}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )


def extract_teacher_logits(
    task_name: str, config: dict, train_texts: list[str], test_texts: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads Gemma-3-4B-IT (BF16), extracts the single offline null probe logits ("N/A"),
    test set logits, and train set logits, then completely unloads the teacher from GPU.
    """
    print("\n" + "-" * 75)
    print(f"STAGE 1: TEACHER EXTRACTION ({task_name.upper()}) via Gemma-3-4B-IT (BF16)")
    print("-" * 75)

    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [
        tokenizer.encode(v, add_special_tokens=False)[-1] for v in config["verbalizers"]
    ]
    print(f"  Verbalizers {config['verbalizers']} -> Token IDs {verbalizer_ids}")

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True,
    )
    model.eval()
    print(f"  Gemma-3-4B loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Single offline null probe: z_T("N/A")
    null_prompt = format_teacher_prompt("N/A", config)
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_null = model(**inp_null)
        null_logits = out_null.logits[0, -1, verbalizer_ids].float().cpu().numpy()
        del inp_null, out_null
    print(f"  Null Probe Logits: {np.round(null_logits, 3).tolist()}")

    def run_batches(texts_list: list[str], desc: str) -> np.ndarray:
        prompts = [format_teacher_prompt(t, config) for t in texts_list]
        # Sort by prompt length and batch within a character budget to avoid truncation or OOM
        order = sorted(range(len(prompts)), key=lambda idx: len(prompts[idx]))
        batches = []
        cur_batch = []
        max_len_in_batch = 0
        for idx in order:
            plen = len(prompts[idx])
            cand_max = max(max_len_in_batch, plen)
            if cur_batch and (len(cur_batch) >= 8 or cand_max * (len(cur_batch) + 1) > 8000):
                batches.append(cur_batch)
                cur_batch = [idx]
                max_len_in_batch = plen
            else:
                cur_batch.append(idx)
                max_len_in_batch = cand_max
        if cur_batch:
            batches.append(cur_batch)

        logits_out = np.zeros((len(prompts), len(verbalizer_ids)), dtype=np.float32)
        processed = 0
        with torch.no_grad():
            for b_idx, batch_indices in enumerate(batches):
                batch_prompts = [prompts[i] for i in batch_indices]
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(device)
                out = model(**enc)
                batch_logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                for row_i, orig_i in enumerate(batch_indices):
                    logits_out[orig_i] = batch_logits[row_i]
                del enc, out
                processed += len(batch_indices)
                if b_idx % 50 == 0 or processed == len(prompts):
                    print(
                        f"    [{desc}] {processed}/{len(prompts)}",
                        flush=True,
                    )
        return logits_out

    test_logits = run_batches(test_texts, "Test")
    train_logits = run_batches(train_texts, "Train")

    # Unload teacher completely before student training
    del model, tokenizer
    clear_vram()
    print(f"  [✓] Teacher unloaded. Remaining VRAM: {torch.cuda.memory_allocated() / 1024**3:.4f} GB")

    return null_logits, test_logits, train_logits


def train_and_evaluate_student(
    method_name: str,
    train_texts: list[str],
    train_probs: np.ndarray,
    test_texts: list[str],
    test_golds: list[int],
    num_classes: int,
    seed: int = SEED,
) -> dict:
    """
    Trains a compact RoBERTa-base student for 3 epochs using KL-divergence against
    the provided soft teacher targets, evaluates on the test set, and frees VRAM.
    """
    set_seed(seed)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    max_len = 512

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        probs = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.float32)
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        enc["probs"] = probs
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=num_classes, local_files_only=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(
        KDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), batch["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(
            f"    [{method_name}] Epoch {epoch + 1}/{epochs} Loss: {epoch_loss / len(loader):.4f}",
            flush=True,
        )

    # Evaluate on test set
    model.eval()
    test_logits_chunks = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            chunk = test_texts[i : i + 32]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            ).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            test_logits_chunks.append(out.logits.cpu().numpy())
            del enc, out

    test_logits = np.concatenate(test_logits_chunks, axis=0)
    preds = test_logits.argmax(axis=-1)

    del model, optimizer, scheduler, loader, tokenizer
    clear_vram()

    metrics = compute_metrics(test_golds, preds, num_classes)
    metrics["train_time_sec"] = round(time.time() - t0, 1)
    return metrics


def run_benchmark(task_name: str, cache_dir: Path) -> dict:
    """Executes the full sequential pipeline (Teacher + 3 Student KD paradigms) for one dataset."""
    config = BENCHMARK_CONFIGS[task_name]
    num_classes = config["num_classes"]

    print("\n" + "=" * 75)
    print(f"RUNNING BENCHMARK: {task_name.upper()} (K={num_classes})")
    print("=" * 75)

    train_texts, _, test_texts, test_golds = load_dataset_splits(task_name)
    print(f"  Loaded Train={len(train_texts)}, Test={len(test_texts)}")

    # 1. Extract and cache teacher targets (or reuse if already cached in this output dir)
    cache_path = cache_dir / f"{task_name}_gemma4b_targets.npz"
    if cache_path.exists():
        print(f"  [INFO] Loading cached teacher targets from {cache_path}")
        cached = np.load(cache_path)
        null_logits = cached["null_logits"]
        test_logits = cached["test_logits"]
        train_logits = cached["train_logits"]
    else:
        null_logits, test_logits, train_logits = extract_teacher_logits(
            task_name, config, train_texts, test_texts
        )
        np.savez_compressed(
            cache_path,
            null_logits=null_logits,
            test_logits=test_logits,
            train_logits=train_logits,
        )

    # 2. Evaluate Teacher Zero-Shot Baselines
    teacher_raw_metrics = compute_metrics(test_golds, test_logits.argmax(axis=-1), num_classes)
    teacher_cal_metrics = compute_metrics(
        test_golds, (test_logits - null_logits).argmax(axis=-1), num_classes
    )
    print(
        f"  [Teacher Base Raw]       F1: {teacher_raw_metrics['macro_f1']:.2f}% "
        f"(Acc: {teacher_raw_metrics['acc']:.2f}%) | Counts: {teacher_raw_metrics['counts']}"
    )
    print(
        f"  [Teacher Null-Cal]       F1: {teacher_cal_metrics['macro_f1']:.2f}% "
        f"(Acc: {teacher_cal_metrics['acc']:.2f}%) | Counts: {teacher_cal_metrics['counts']}"
    )

    # 3. Compute soft probability targets for the 3 KD paradigms
    probs_standard = softmax(train_logits, axis=-1)

    p_batch = np.clip(probs_standard.mean(axis=0, keepdims=True), 1e-12, 1.0)
    p_batch /= p_batch.sum(axis=-1, keepdims=True)
    probs_batch = softmax(train_logits - np.log(p_batch), axis=-1)

    probs_ckd = softmax(train_logits - null_logits, axis=-1)

    # 4. Train Student models sequentially
    print("\n  STAGE 2: SEQUENTIAL STUDENT TRAINING (RoBERTa-base)")
    print("  --------------------------------------------------")

    print("  >> 1/3 Training Standard KD...")
    std_metrics = train_and_evaluate_student(
        "Standard KD", train_texts, probs_standard, test_texts, test_golds, num_classes
    )
    print(
        f"  [✓] Standard KD:         F1: {std_metrics['macro_f1']:.2f}% "
        f"(Acc: {std_metrics['acc']:.2f}%) | Counts: {std_metrics['counts']}"
    )

    print("  >> 2/3 Training Batch-Calibrated KD...")
    batch_metrics = train_and_evaluate_student(
        "Batch-Cal KD", train_texts, probs_batch, test_texts, test_golds, num_classes
    )
    print(
        f"  [✓] Batch-Calibrated KD: F1: {batch_metrics['macro_f1']:.2f}% "
        f"(Acc: {batch_metrics['acc']:.2f}%) | Counts: {batch_metrics['counts']}"
    )

    print("  >> 3/3 Training Null Calibrated KD (Ours)...")
    ckd_metrics = train_and_evaluate_student(
        "CKD (Ours)", train_texts, probs_ckd, test_texts, test_golds, num_classes
    )
    print(
        f"  [✓] Null Cal KD (Ours):  F1: {ckd_metrics['macro_f1']:.2f}% "
        f"(Acc: {ckd_metrics['acc']:.2f}%) | Counts: {ckd_metrics['counts']}"
    )

    result = {
        "dataset": task_name,
        "num_classes": num_classes,
        "class_names": config["class_names"],
        "teacher_model": "google/gemma-3-4b-it",
        "student_model": STUDENT_MODEL_NAME,
        "null_logits": null_logits.tolist(),
        "teacher_raw": teacher_raw_metrics,
        "teacher_calibrated": teacher_cal_metrics,
        "standard_kd": std_metrics,
        "batch_calibrated_kd": batch_metrics,
        "calibrated_kd": ckd_metrics,
    }

    out_file = OUTPUT_DIR / f"{task_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    print(f"  [✓] Saved result file: {out_file}")

    return result


def print_final_summary(all_results: dict) -> None:
    """Prints a formatted comparison table of all 6 benchmarks across all methods."""
    print("\n" + "=" * 95)
    print("MAIN BENCHMARK EVALUATION SUMMARY: Macro-F1 (%) with Accuracy (%) in parentheses")
    print("=" * 95)
    header = (
        f"{'Dataset':<20} | {'Teacher Raw':<14} | {'Teacher Cal':<14} | "
        f"{'Standard KD':<14} | {'Batch-Cal KD':<14} | {'CKD (Ours)':<14}"
    )
    print(header)
    print("-" * 95)
    for task_name in DATASET_ORDER:
        r = all_results[task_name]
        tr = f"{r['teacher_raw']['macro_f1']:.2f}% ({r['teacher_raw']['acc']:.1f}%)"
        tc = f"{r['teacher_calibrated']['macro_f1']:.2f}% ({r['teacher_calibrated']['acc']:.1f}%)"
        sk = f"{r['standard_kd']['macro_f1']:.2f}% ({r['standard_kd']['acc']:.1f}%)"
        bk = f"{r['batch_calibrated_kd']['macro_f1']:.2f}% ({r['batch_calibrated_kd']['acc']:.1f}%)"
        ck = f"{r['calibrated_kd']['macro_f1']:.2f}% ({r['calibrated_kd']['acc']:.1f}%)"
        print(f"{task_name:<20} | {tr:<14} | {tc:<14} | {sk:<14} | {bk:<14} | {ck:<14}")
    print("=" * 95)


def main() -> None:
    """Main entry point: configures logging, verifies datasets, and runs all 6 experiments."""
    parser = argparse.ArgumentParser(
        description="Run all 6 main Calibrated KD benchmark experiments sequentially."
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-downloading and re-splitting all 6 datasets from HuggingFace.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"main_experiments_{timestamp}.log"
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    start_time = time.time()
    print(f"Log started: {log_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        # Step 0: Verify and download datasets if missing
        verify_all_datasets(force_download=args.force_download)

        # Enable offline mode for local HuggingFace model weights once datasets are verified
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Step 1-6: Sequentially execute all 6 main benchmarks
        all_results = {}
        for idx, task_name in enumerate(DATASET_ORDER, 1):
            print(f"\n>>> BENCHMARK {idx}/{len(DATASET_ORDER)}: {task_name.upper()} <<<")
            all_results[task_name] = run_benchmark(task_name, cache_dir)
            clear_vram()

        summary_file = OUTPUT_DIR / "all_6_main_benchmarks_summary.json"
        with open(summary_file, "w", encoding="utf-8") as fp:
            json.dump(all_results, fp, indent=2)

        print_final_summary(all_results)
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"\n[✓] All 6 main experiments completed in {elapsed_min:.2f} minutes.")
        print(f"[✓] Summary saved to: {summary_file}")

        # Generate the 3 publication figures from the newly saved summary
        print("\n" + "=" * 75)
        print("STEP 3: GENERATING PUBLICATION FIGURES")
        print("=" * 75)
        plot_scripts = [
            ROOT / "scripts" / "plotting" / "generate_blog_table_image.py",
            ROOT / "scripts" / "plotting" / "generate_benchmark_barchart.py",
            ROOT / "scripts" / "plotting" / "plot_3_key_class_distributions.py",
        ]
        for script_path in plot_scripts:
            subprocess.run([sys.executable, str(script_path)], check=True)
    finally:
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
