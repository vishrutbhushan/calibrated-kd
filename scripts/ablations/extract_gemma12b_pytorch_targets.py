"""
extract_gemma12b_pytorch_targets.py
-----------------------------------
Sequentially extracts and caches teacher targets using Gemma-3-12B-IT in PyTorch (4-bit NF4).
DOES NOT run Student KD yet (pure extraction & caching first).

Guarantees:
  1. Safe dedicated cache path:
     data/datasets/natural_benchmarks/<dataset>/gemma12b_pytorch_teacher_targets.npz
     (NEVER overwrites existing 4B caches).
  2. Full vocabulary unclipped floating-point logits for all candidate verbalizers.
  3. Strictly sequential execution across the 6 natural datasets.
  4. Safe batch size (batch_size=4) with no VRAM spikes.
  5. Computes and logs Teacher Raw vs. Teacher Null-Calibrated zero-shot performance.
"""

import os
import sys
import gc
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
RESULTS_DIR = ROOT / "outputs" / "ablations" / "6_gemma12b_scaling" / "extraction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"

BENCHMARK_CONFIGS = {
    "student_questions": {
        "num_classes": 4,
        "class_names": ["Biology", "Physics", "Maths", "Chemistry"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": "Classify the academic question into the correct subject:\nA: Biology\nB: Physics\nC: Maths\nD: Chemistry"
    },
    "trec6": {
        "num_classes": 6,
        "class_names": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": "Classify the question into one of six question types:\nA: Abbreviation\nB: Entity\nC: Description\nD: Human\nE: Location\nF: Numeric"
    },
    "app_reviews": {
        "num_classes": 5,
        "class_names": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": "Classify the customer app review into one of five star ratings:\nA: 1 Star\nB: 2 Stars\nC: 3 Stars\nD: 4 Stars\nE: 5 Stars"
    },
    "twitter_emotion": {
        "num_classes": 6,
        "class_names": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "task_desc": "Classify the tweet into one of six emotions:\nA: Sadness\nB: Joy\nC: Love\nD: Anger\nE: Fear\nF: Surprise"
    },
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
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

def extract_dataset(model, tokenizer, task_name, config):
    print("\n" + "="*70)
    print(f"EXTRACTING TARGETS: {task_name.upper()} (K={config['num_classes']})")
    print("="*70)

    cache_path = DATA_BASE / task_name / "gemma12b_pytorch_teacher_targets.npz"
    train_texts, train_golds, test_texts, test_golds = load_data(task_name)
    num_classes = config["num_classes"]
    verbalizers = config["verbalizers"]
    v_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]
    print(f"Verbalizers: {verbalizers} -> Token IDs: {v_ids}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1. Null Probe
    null_prompt = format_prompt("N/A", config)
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_null = model(**inp_null)
        null_logits = out_null.logits[0, -1, v_ids].float().cpu().numpy()
        del inp_null, out_null
    print("Exact Null Logits:", null_logits)

    # Helper for batch inference
    def run_batches(texts_list, desc=""):
        prompts = [format_prompt(t, config) for t in texts_list]
        res = []
        batch_size = 4
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                b = prompts[i : i + batch_size]
                inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                out = model(**inp)
                l = out.logits[:, -1, v_ids].float().cpu().numpy()
                res.append(l)
                del inp, out
                if (i // batch_size) % 25 == 0 or (i + batch_size >= len(prompts)):
                    print(f"  [{desc}] Processed {min(i + batch_size, len(prompts))}/{len(prompts)}...", flush=True)
        return np.concatenate(res, axis=0)

    # 2. Test Set (400 items)
    print("\nExtracting Test set logits (400 items)...")
    test_logits = run_batches(test_texts, "Test")

    # 3. Train Set (1200 items)
    print("\nExtracting Train set logits (1200 items)...")
    train_logits = run_batches(train_texts, "Train")

    # Evaluate Teacher zero-shot on test set
    tea_raw_preds = test_logits.argmax(axis=-1)
    tea_raw_metrics = compute_metrics(test_golds, tea_raw_preds, num_classes)
    print(f"\n[Teacher 12B Raw Zero-Shot] Acc: {tea_raw_metrics['acc']:.2f}% | Macro-F1: {tea_raw_metrics['macro_f1']:.2f}% | Counts: {tea_raw_metrics['counts']}")

    tea_cal_logits = test_logits - null_logits
    tea_cal_preds = tea_cal_logits.argmax(axis=-1)
    tea_cal_metrics = compute_metrics(test_golds, tea_cal_preds, num_classes)
    print(f"[Teacher 12B Null-Calibrated] Acc: {tea_cal_metrics['acc']:.2f}% | Macro-F1: {tea_cal_metrics['macro_f1']:.2f}% | Counts: {tea_cal_metrics['counts']}")

    # Compute soft target probabilities
    # Raw train probs
    exp_raw = np.exp(train_logits - train_logits.max(axis=-1, keepdims=True))
    raw_train_probs = exp_raw / exp_raw.sum(axis=-1, keepdims=True)

    # Calibrated train probs
    cal_train_logits = train_logits - null_logits
    exp_cal = np.exp(cal_train_logits - cal_train_logits.max(axis=-1, keepdims=True))
    cal_train_probs = exp_cal / exp_cal.sum(axis=-1, keepdims=True)

    # Batch-calibrated train probs
    p_batch = raw_train_probs.mean(axis=0, keepdims=True)
    p_batch = np.clip(p_batch, 1e-12, 1.0)
    p_batch /= p_batch.sum(axis=-1, keepdims=True)
    batch_train_logits = train_logits - np.log(p_batch)
    exp_batch = np.exp(batch_train_logits - batch_train_logits.max(axis=-1, keepdims=True))
    batch_train_probs = exp_batch / exp_batch.sum(axis=-1, keepdims=True)

    # Save to dedicated new npz cache
    np.savez_compressed(
        cache_path,
        null_logits=null_logits,
        test_logits=test_logits,
        train_logits=train_logits,
        raw_train_probs=raw_train_probs,
        cal_train_probs=cal_train_probs,
        batch_train_probs=batch_train_probs,
        test_golds=test_golds,
        train_golds=train_golds
    )
    print(f"[✓] Saved targets to: {cache_path}")

    return {
        "dataset": task_name,
        "teacher_raw": tea_raw_metrics,
        "teacher_calibrated": tea_cal_metrics,
        "null_logits": null_logits.tolist()
    }

def main():
    set_seed(42)
    clear_vram()

    print("="*70)
    print("LOADING GEMMA-3-12B-IT IN 4-BIT (NF4) VIA PYTORCH")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        local_files_only=True
    )
    model.eval()
    print(f"Gemma-12B loaded in {time.time()-t0:.1f}s. VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / 12.0 GB")

    datasets = [
        "student_questions",
        "trec6",
        "app_reviews",
        "twitter_emotion",
        "tweeteval_emotion",
        "bbc_news"
    ]

    summaries = {}
    for d in datasets:
        cfg = BENCHMARK_CONFIGS[d]
        res = extract_dataset(model, tokenizer, d, cfg)
        summaries[d] = res
        clear_vram()

    # Unload 12B model completely
    print("\n" + "="*70)
    print("UNLOADING GEMMA-12B MODEL FROM GPU MEMORY")
    print("="*70)
    del model, tokenizer
    clear_vram()
    print(f"Remaining VRAM: {torch.cuda.memory_allocated() / 1024**3:.4f} GB")

    summary_file = RESULTS_DIR / "all_6_gemma12b_teacher_zero_shot_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(summaries, fp, indent=2)

    print("\n" + "#"*70)
    print(f"ALL 6 DATASETS EXTRACTED & CACHED SUCCESSFULLY! Summary: {summary_file}")
    print("#"*70)

if __name__ == "__main__":
    main()
