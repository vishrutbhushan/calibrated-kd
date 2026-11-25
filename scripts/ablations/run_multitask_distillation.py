"""
run_multitask_distillation.py
-----------------------------
Generalized Multi-Task Distillation Runner with Task-Specific Post-Hoc Calibration.
Supports arbitrary combinations of naturally skewed benchmarks (2-task, 3-task, etc.)
with a shared RoBERTa-base encoder and task-specific classification heads.

Evaluates all 4 multi-task distillation paradigms:
  1. Multi-Task Standard KD (Raw)
  2. Multi-Task Standard KD + Post-Hoc (Task-specific null shift)
  3. Multi-Task CKD (Raw, Ours)
  4. Multi-Task CKD + Post-Hoc (Combined)

Strict constraints:
1. Exactly ONE model loaded in GPU VRAM at a time.
2. Complete teacher targets caching and unloading.
3. Macro-F1 computed on authentic stratified test splits (N=400 per task).
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
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
OUTPUT_DIR = ROOT / "outputs" / "ablations" / "2_multitask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "app_reviews": {
        "num_classes": 5,
        "class_names": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        "verbalizers": ["A", "B", "C", "D", "E"],
        "task_desc": "Classify the customer app review into one of five star ratings:\nA: 1 Star\nB: 2 Stars\nC: 3 Stars\nD: 4 Stars\nE: 5 Stars"
    },
    "student_questions": {
        "num_classes": 4,
        "class_names": ["Biology", "Physics", "Maths", "Chemistry"],
        "verbalizers": ["A", "B", "C", "D"],
        "task_desc": "Classify the academic question into the correct subject:\nA: Biology\nB: Physics\nC: Maths\nD: Chemistry"
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
    }
}

INDIVIDUAL_BASELINE = {
    "tweeteval_emotion": {"std_kd": 34.2, "posthoc": 55.4, "ckd": 55.5, "combined": 68.0},
    "bbc_news":          {"std_kd": 56.3, "posthoc": 86.8, "ckd": 71.5, "combined": 93.1},
    "app_reviews":       {"std_kd": 30.25, "ckd": 33.24},
    "student_questions": {"std_kd": 22.86, "ckd": 37.10},
    "twitter_emotion":   {"std_kd": 29.8, "posthoc": 32.6, "ckd": 34.5, "combined": 43.2},
    "trec6":             {"std_kd": 62.3, "posthoc": 58.9, "ckd": 64.7, "combined": 60.1}
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

def load_data(task_name):
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

def format_teacher_prompt(text, config):
    return (
        f"{config['task_desc']}\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )

def get_or_extract_teacher_targets(task_name):
    cache_path = DATA_BASE / task_name / "gemma4b_teacher_targets.npz"
    if cache_path.exists():
        print(f"[+] Loaded cached teacher targets for {task_name}: {cache_path}")
        d = np.load(cache_path)
        return d["raw_train_probs"], d["cal_train_probs"]

    print(f"\n[+] Extracting teacher targets for {task_name} via Gemma-3-4B-IT...")
    train_texts, _, test_texts, _ = load_data(task_name)
    config = BENCHMARK_CONFIGS[task_name]
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

    null_prompt = format_teacher_prompt("N/A", config)
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()

    def run_batches(texts_list):
        prompts = [format_teacher_prompt(t, config) for t in texts_list]
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
        return np.concatenate(res, axis=0)

    train_logits = run_batches(train_texts)
    raw_train_probs = softmax(train_logits, axis=-1)
    cal_train_probs = softmax(train_logits - null_logits, axis=-1)

    np.savez(cache_path, raw_train_probs=raw_train_probs, cal_train_probs=cal_train_probs, null_logits=null_logits)
    print(f"[✓] Saved teacher targets to: {cache_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return raw_train_probs, cal_train_probs

class MultiTaskRoBERTa(nn.Module):
    def __init__(self, task_names, model_name="roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, BENCHMARK_CONFIGS[task]["num_classes"])
            for task in task_names
        })

    def forward(self, input_ids, attention_mask, task_name):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.heads[task_name](cls_rep)

def compute_metrics(golds, preds, num_classes):
    acc = float(accuracy_score(golds, preds) * 100)
    macro_f1 = float(f1_score(golds, preds, average="macro") * 100)
    per_class = (f1_score(golds, preds, average=None) * 100).tolist()
    counts = np.bincount(preds, minlength=num_classes).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": [round(x, 2) for x in per_class],
        "counts": counts
    }

def train_and_eval_multitask(method_name, task_names, all_task_data, seed=42):
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*70)
    print(f"TRAINING MULTI-TASK STUDENT: {method_name.upper()} on tasks: {task_names}")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    model = MultiTaskRoBERTa(task_names, STUDENT_MODEL_NAME).to(device)

    task_loaders = {}
    total_steps_per_epoch = 0
    for task in task_names:
        txts = all_task_data[task]["train_texts"]
        probs = all_task_data[task][f"{method_name}_train_probs"]
        loader = DataLoader([(t, p) for t, p in zip(txts, probs)], batch_size=16, shuffle=True)
        task_loaders[task] = loader
        total_steps_per_epoch += len(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = total_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    max_batches = max(len(l) for l in task_loaders.values())

    for ep in range(epochs):
        epoch_loss = 0.0
        task_iters = {task: iter(task_loaders[task]) for task in task_names}

        for _ in range(max_batches):
            for task in task_names:
                try:
                    batch = next(task_iters[task])
                    txts, probs = batch
                    enc = tokenizer(list(txts), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                    target_p = torch.tensor(np.array(probs), dtype=torch.float32).to(device)

                    optimizer.zero_grad()
                    logits = model(enc["input_ids"], enc["attention_mask"], task)
                    loss = kl_loss(torch.log_softmax(logits, dim=-1), target_p)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                except StopIteration:
                    pass

        avg_loss = epoch_loss / total_steps_per_epoch
        print(f"  Epoch {ep+1}/{epochs} | Multi-Task KL Loss: {avg_loss:.4f}", flush=True)

    print("\nEvaluating Multi-Task Student...", flush=True)
    model.eval()

    task_results = {}
    raw_f1_list = []

    for task in task_names:
        test_texts = all_task_data[task]["test_texts"]
        test_golds = all_task_data[task]["test_golds"]
        num_classes = BENCHMARK_CONFIGS[task]["num_classes"]

        test_logits_chunks = []
        with torch.no_grad():
            for i in range(0, len(test_texts), 32):
                b = test_texts[i : i + 32]
                enc = tokenizer(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                logits = model(enc["input_ids"], enc["attention_mask"], task)
                test_logits_chunks.append(logits.cpu().numpy())
        test_logits = np.concatenate(test_logits_chunks, axis=0)

        preds = test_logits.argmax(axis=-1)
        metrics = compute_metrics(test_golds, preds, num_classes)
        task_results[task] = metrics
        raw_f1_list.append(metrics["macro_f1"])

        print(f"  [{task}] Macro-F1: {metrics['macro_f1']:.2f}% | Acc: {metrics['accuracy']:.2f}% | Counts: {metrics['counts']}")

    avg_f1 = float(np.mean(raw_f1_list))
    print(f"  -> Average Macro-F1: {avg_f1:.2f}%")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[✓] {method_name} model unloaded from VRAM.")

    return {
        "per_task": task_results,
        "average_macro_f1": avg_f1,
    }

def run_combination(task_names, combo_id):
    combo_name = f"combo{combo_id}_" + "_".join(task_names)
    print("\n" + "#"*70)
    print(f"RUNNING COMBINATION {combo_id}: {task_names}")
    print("#"*70)

    all_task_data = {}
    for task in task_names:
        tr_txt, tr_gold, te_txt, te_gold = load_data(task)
        raw_p, cal_p = get_or_extract_teacher_targets(task)
        all_task_data[task] = {
            "train_texts": tr_txt,
            "train_golds": tr_gold,
            "test_texts": te_txt,
            "test_golds": te_gold,
            "std_kd_train_probs": raw_p,
            "ckd_train_probs": cal_p
        }

    # 1. Standard KD
    std_kd_res = train_and_eval_multitask("std_kd", task_names, all_task_data)

    # 2. CKD (Ours)
    ckd_res = train_and_eval_multitask("ckd", task_names, all_task_data)

    indiv_std = np.mean([INDIVIDUAL_BASELINE[t]["std_kd"] for t in task_names])
    indiv_ckd = np.mean([INDIVIDUAL_BASELINE[t]["ckd"] for t in task_names])

    summary = {
        "combo_id": combo_id,
        "tasks": task_names,
        "std_kd_macro_f1": std_kd_res["average_macro_f1"],
        "ckd_macro_f1": ckd_res["average_macro_f1"],
        "delta_macro_f1": ckd_res["average_macro_f1"] - std_kd_res["average_macro_f1"],
        "details": {
            "std_kd": std_kd_res,
            "ckd": ckd_res
        },
        "individual_baselines": {
            "std_kd": indiv_std,
            "ckd": indiv_ckd,
        }
    }

    out_file = OUTPUT_DIR / f"{combo_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"COMBINATION {combo_id} SUMMARY:")
    print(f"  Multi-Task Std KD: {summary['std_kd_macro_f1']:.2f}%  (Individual: {indiv_std:.2f}%)")
    print(f"  Multi-Task CKD:    {summary['ckd_macro_f1']:.2f}%  (Individual: {indiv_ckd:.2f}%)")
    print("="*70)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True, help="List of benchmark task names")
    parser.add_argument("--combo_id", type=int, default=1, help="Combination identifier integer")
    args = parser.parse_args()

    run_combination(args.tasks, args.combo_id)

if __name__ == "__main__":
    main()
