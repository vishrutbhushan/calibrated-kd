"""
run_single_benchmark.py
-----------------------
Runs a complete, self-contained distillation evaluation on a single naturally skewed benchmark.
Enforces:
  1. STRICTLY ONE MODEL LOADED AT A TIME:
     - Teacher extracts logits -> unloads completely from GPU memory -> VRAM freed.
     - Student (RoBERTa-base) loads and trains sequentially -> unloads between methods.
  2. Zero data leakage:
     - Post-hoc calibration uses a random stratified 300-sample slice from training data.
  3. Evaluates all 5 distillation paradigms:
     - Teacher Raw Zero-Shot
     - Teacher Null-Calibrated
     - Student: Standard KD
     - Student: Post-Hoc Output Cal
     - Student: Batch-Calibrated KD
     - Student: Calibrated KD (CKD, Ours)
     - Student: Combined (CKD + Post-Hoc)

Usage:
  python run_single_benchmark.py --dataset tweeteval_emotion
"""

import os
import sys
import csv
import json
import random
import argparse
import gc
from pathlib import Path
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
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
RESULTS_DIR = ROOT / "outputs" / "results" / "main_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_MODEL_NAME = "roberta-base"

BENCHMARK_CONFIGS = {
    "tweeteval_emotion": {
        "num_classes": 4,
        "class_names": ["Anger", "Joy", "Optimism", "Sadness"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": "Classify the tweet into one of four emotions:\nA: Anger\nB: Joy\nC: Optimism\nD: Sadness"
    },
    "bbc_news": {
        "num_classes": 5,
        "class_names": ["Tech", "Business", "Sport", "Entertainment", "Politics"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": "Classify the news article into one of five categories:\nA: Tech\nB: Business\nC: Sport\nD: Entertainment\nE: Politics"
    },
    "praxis": {
        "num_classes": 5,
        "class_names": ["Imperative", "Interrogative", "Modal", "Conditional", "Optative"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": "Classify the dialogue utterance into one of five speech acts:\nA: Imperative\nB: Interrogative\nC: Modal\nD: Conditional\nE: Optative"
    },
    "twitter_emotion": {
        "num_classes": 6,
        "class_names": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": "Classify the tweet into one of six emotions:\nA: Sadness\nB: Joy\nC: Love\nD: Anger\nE: Fear\nF: Surprise"
    },
    "trec6": {
        "num_classes": 6,
        "class_names": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": "Classify the question into one of six question types:\nA: Abbreviation\nB: Entity\nC: Description\nD: Human\nE: Location\nF: Numeric"
    },
    "pubmed_rct": {
        "num_classes": 5,
        "class_names": ["Background", "Objective", "Methods", "Results", "Conclusions"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": "Classify the sentence from a medical abstract into one of five sections:\nA: Background\nB: Objective\nC: Methods\nD: Results\nE: Conclusions"
    },
    "citation_intent": {
        "num_classes": 6,
        "class_names": ["Background", "Uses", "CompareOrContrast", "Motivation", "Extends", "Future"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": "Classify the citation sentence from a scientific paper into one of six citation functions:\nA: Background\nB: Uses\nC: CompareOrContrast\nD: Motivation\nE: Extends\nF: Future"
    },
    "app_reviews": {
        "num_classes": 5,
        "class_names": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": "Classify the customer app review into one of five star ratings:\nA: 1 Star\nB: 2 Stars\nC: 3 Stars\nD: 4 Stars\nE: 5 Stars"
    },
    "go_emotions_ekman": {
        "num_classes": 6,
        "class_names": ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": "Classify the emotion expressed in the text into one of six categories:\nA: Anger\nB: Disgust\nC: Fear\nD: Joy\nE: Sadness\nF: Surprise"
    },
    "sst4_finegrained": {
        "num_classes": 4,
        "class_names": ["Very Negative", "Negative", "Positive", "Very Positive"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": "Classify the sentiment of the movie review into one of four fine-grained categories:\nA: Very Negative\nB: Negative\nC: Positive\nD: Very Positive"
    },
    "student_questions": {
        "num_classes": 4,
        "class_names": ["Biology", "Physics", "Maths", "Chemistry"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": "Classify the academic question into the correct subject:\nA: Biology\nB: Physics\nC: Maths\nD: Chemistry"
    },
    "sst2": {
        "num_classes": 2,
        "class_names": ["Negative", "Positive"],
        "verbalizers": ["A", "B"],
        "task_desc": "Classify the sentiment of the movie review into one of two categories:\nA: Negative\nB: Positive"
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def compute_metrics(golds, preds, num_classes):
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

def load_data(task_name):
    task_dir = DATA_BASE / task_name
    train_file = task_dir / "train_1200.csv" if (task_dir / "train_1200.csv").exists() else task_dir / "train_1000.csv"
    test_file = task_dir / "test_400.csv" if (task_dir / "test_400.csv").exists() else task_dir / "test_350.csv"

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

def format_prompt(text, config):
    return (
        f"{config['task_desc']}\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )

def extract_teacher_targets(dataset_name, config, train_texts, test_texts):
    print("\n" + "="*70)
    print(f"STAGE 1: EXTRACTING TEACHER TARGETS ({dataset_name}) via Gemma-3-4B-IT (BF16)")
    print("="*70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in config["verbalizers"]]

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True
    )
    model.eval()

    # 1. Null probe vector
    null_prompt = format_prompt("N/A", config)
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()

    print("Teacher Null Probe Logits:", null_logits)

    def run_batches(texts_list, desc=""):
        prompts = [format_prompt(t, config) for t in texts_list]
        res = []
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                b = prompts[i : i + batch_size]
                inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                out = model(**inp)
                l = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                res.append(l)
                del inp, out
                if (i // batch_size) % 25 == 0:
                    print(f"  {desc}: {min(i + batch_size, len(prompts))}/{len(prompts)}...", flush=True)
        return np.concatenate(res, axis=0)

    print("\nExtracting Teacher logits on Test Set...", flush=True)
    test_logits = run_batches(test_texts, "Test")

    print("\nExtracting Teacher logits on Train Set...", flush=True)
    train_logits = run_batches(train_texts, "Train")

    # Complete unload of teacher
    print("\nUnloading Teacher model from GPU memory...", flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("[✓] Teacher successfully unloaded. VRAM freed.")

    return null_logits, test_logits, train_logits

def train_student_model(train_texts, train_probs, test_texts, test_golds, num_classes, seed=42):
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    max_len = 256

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["probs"] = p
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=num_classes, local_files_only=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(KDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate_fn)
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

    # Inference on test set
    model.eval()
    test_logits = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            chunk = test_texts[i : i + 32]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            test_logits.append(out.logits.cpu().numpy())
            del enc, out
    test_logits = np.concatenate(test_logits, axis=0)

    # Compute student output mean bias on stratified 300 random training items
    with torch.no_grad():
        random.seed(seed)
        sub_indices = random.sample(range(len(train_texts)), min(300, len(train_texts)))
        sub_texts = [train_texts[idx] for idx in sub_indices]
        enc_sub = tokenizer(sub_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out_sub = model(input_ids=enc_sub["input_ids"], attention_mask=enc_sub["attention_mask"])
        mean_bias = out_sub.logits.mean(dim=0).cpu().numpy()
        del enc_sub, out_sub

    # Cleanup student
    del model, optimizer, scheduler, loader, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Base and post-hoc predictions
    base_preds = test_logits.argmax(axis=-1)
    base_metrics = compute_metrics(test_golds, base_preds, num_classes)

    post_logits = test_logits - mean_bias
    post_preds = post_logits.argmax(axis=-1)
    post_metrics = compute_metrics(test_golds, post_preds, num_classes)

    return base_metrics, post_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(BENCHMARK_CONFIGS.keys()))
    args = parser.parse_args()

    dataset_name = args.dataset
    config = BENCHMARK_CONFIGS[dataset_name]
    num_classes = config["num_classes"]

    print(f"\n>>> STARTING EXPERIMENT RUN FOR: {dataset_name} ({num_classes} classes)")

    train_texts, train_golds, test_texts, test_golds = load_data(dataset_name)
    print(f"Loaded {len(train_texts)} train items, {len(test_texts)} test items.")

    # 1. Teacher Target Extraction
    null_logits, test_logits_raw, train_logits_raw = extract_teacher_targets(
        dataset_name, config, train_texts, test_texts
    )

    # Teacher Metrics
    test_logits_cal = test_logits_raw - null_logits
    teacher_raw_metrics = compute_metrics(test_golds, test_logits_raw.argmax(axis=-1), num_classes)
    teacher_cal_metrics = compute_metrics(test_golds, test_logits_cal.argmax(axis=-1), num_classes)

    print(f"\nTEACHER EVALUATION:")
    print(f"  Teacher Raw:        Acc={teacher_raw_metrics['acc']:.2f}% | F1={teacher_raw_metrics['macro_f1']:.2f}% [{teacher_raw_metrics['worst_f1']:.2f}%]")
    print(f"  Teacher Calibrated: Acc={teacher_cal_metrics['acc']:.2f}% | F1={teacher_cal_metrics['macro_f1']:.2f}% [{teacher_cal_metrics['worst_f1']:.2f}%]")

    # 2. Student Distillation Runs (RoBERTa-base)
    print("\n" + "="*70)
    print(f"STAGE 2: STUDENT DISTILLATION (RoBERTa-base, 125M) on {dataset_name}")
    print("="*70)

    # Probability targets
    p_raw = softmax(train_logits_raw)
    p_ckd = softmax(train_logits_raw - null_logits)

    # Global data prior for Batch-Cal
    p_batch_prior = np.mean(p_raw, axis=0)
    p_batch = softmax(train_logits_raw - np.log(p_batch_prior + 1e-12))

    # A. Standard KD + Post-Hoc Output Cal
    print("\n[Method 1 & 2] Training Student with Standard KD...", flush=True)
    m_std, m_posthoc = train_student_model(train_texts, p_raw, test_texts, test_golds, num_classes)
    print(f"  Standard KD:          Acc={m_std['acc']:.2f}% | F1={m_std['macro_f1']:.2f}% [{m_std['worst_f1']:.2f}%] | Counts={m_std['counts']}")
    print(f"  Post-Hoc Output Cal:  Acc={m_posthoc['acc']:.2f}% | F1={m_posthoc['macro_f1']:.2f}% [{m_posthoc['worst_f1']:.2f}%] | Counts={m_posthoc['counts']}")

    # B. Batch-Calibrated KD
    print("\n[Method 3] Training Student with Batch-Calibrated KD...", flush=True)
    m_batch, _ = train_student_model(train_texts, p_batch, test_texts, test_golds, num_classes)
    print(f"  Batch-Calibrated KD:  Acc={m_batch['acc']:.2f}% | F1={m_batch['macro_f1']:.2f}% [{m_batch['worst_f1']:.2f}%] | Counts={m_batch['counts']}")

    # C. Calibrated KD (CKD) + Combined (CKD + Post-Hoc)
    print("\n[Method 4 & 5] Training Student with Calibrated KD (CKD)...", flush=True)
    m_ckd, m_ckd_posthoc = train_student_model(train_texts, p_ckd, test_texts, test_golds, num_classes)
    print(f"  Calibrated KD (CKD):  Acc={m_ckd['acc']:.2f}% | F1={m_ckd['macro_f1']:.2f}% [{m_ckd['worst_f1']:.2f}%] | Counts={m_ckd['counts']}")
    print(f"  CKD + Post-Hoc:       Acc={m_ckd_posthoc['acc']:.2f}% | F1={m_ckd_posthoc['macro_f1']:.2f}% [{m_ckd_posthoc['worst_f1']:.2f}%] | Counts={m_ckd_posthoc['counts']}")

    results = {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "class_names": config["class_names"],
        "null_logits": null_logits.tolist(),
        "teacher_raw": teacher_raw_metrics,
        "teacher_calibrated": teacher_cal_metrics,
        "standard_kd": m_std,
        "post_hoc_output_cal": m_posthoc,
        "batch_calibrated_kd": m_batch,
        "calibrated_kd": m_ckd,
        "ckd_posthoc_combined": m_ckd_posthoc
    }

    out_file = RESULTS_DIR / f"{dataset_name}_all_methods.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"[✓] EXPERIMENT COMPLETE: {dataset_name}. Saved to {out_file}")
    print("="*70)

if __name__ == "__main__":
    main()
