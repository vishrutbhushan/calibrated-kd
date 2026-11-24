"""
run_multitask_distillation_ablation.py
--------------------------------------
Multi-Task Distillation Ablation Study on Shared Internal Representations.

Datasets Chosen:
1. TweetEval Emotion (4 classes: Anger, Joy, Optimism, Sadness)
   - Domain: Short, informal social media tweets
2. BBC News (5 classes: Tech, Business, Sport, Entertainment, Politics)
   - Domain: Long, structured formal journalistic articles

Core Research Questions:
1. Representation Interference: Does Standard KD cause destructive gradient interference
   in a shared student backbone due to conflicting teacher prior biases?
2. CKD Multi-Task Generalization: Does calibrating the teacher distributions allow a SINGLE
   shared student to learn both disparate domains effectively without class collapse?
3. Multi-Task vs. Individual: How close does the single multi-task CKD student get to
   individually fine-tuned students?

Student Model: RoBERTa-base (125M) with shared transformer encoder and dual task heads.
Teacher: Gemma-3-4B-IT (BF16)
"""

import os
import sys
import csv
import json
import random
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
OUTPUT_DIR = ROOT / "outputs" / "results" / "multitask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_MODEL_NAME = "roberta-base"

CONFIGS = {
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

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def load_dataset_splits(task_name):
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
        print(f"[+] Loading cached teacher targets for {task_name}: {cache_path}")
        d = np.load(cache_path)
        return d["raw_train_probs"], d["cal_train_probs"]

    print(f"\n[+] Extracting teacher targets for {task_name} via Gemma-3-4B-IT...")
    train_texts, _, test_texts, _ = load_dataset_splits(task_name)
    config = CONFIGS[task_name]
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
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_tweeteval = nn.Linear(hidden_size, 4)
        self.head_bbc = nn.Linear(hidden_size, 5)

    def forward(self, input_ids, attention_mask, task):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])
        if task == "tweeteval_emotion":
            return self.head_tweeteval(cls_rep)
        elif task == "bbc_news":
            return self.head_bbc(cls_rep)
        else:
            raise ValueError(f"Unknown task: {task}")

class CombinedMultiTaskDataset(Dataset):
    def __init__(self, items):
        # items: list of tuples (text, target_probs, task_name)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def train_and_eval_multitask(method_name, data_tweeteval, data_bbc, seed=42):
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*70)
    print(f"TRAINING MULTI-TASK STUDENT: {method_name} (Shared RoBERTa-base)")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    model = MultiTaskRoBERTa(STUDENT_MODEL_NAME).to(device)

    # Prepare combined training dataset
    train_items = []
    # Interleave TweetEval and BBC News
    tt_texts, tt_probs = data_tweeteval["train_texts"], data_tweeteval[f"{method_name}_train_probs"]
    bbc_texts, bbc_probs = data_bbc["train_texts"], data_bbc[f"{method_name}_train_probs"]

    for t, p in zip(tt_texts, tt_probs):
        train_items.append((t, p, "tweeteval_emotion"))
    for t, p in zip(bbc_texts, bbc_probs):
        train_items.append((t, p, "bbc_news"))

    random.seed(seed)
    random.shuffle(train_items)
    print(f"Total multi-task training instances: {len(train_items)} (1,200 TweetEval + 1,200 BBC News)")

    # Custom collate function grouping by task within batches or separate loaders
    # To keep task heads clean, let's create two parallel loaders or task-split mini-batches
    tt_loader = DataLoader(
        [(t, p) for t, p in zip(tt_texts, tt_probs)],
        batch_size=16,
        shuffle=True
    )
    bbc_loader = DataLoader(
        [(t, p) for t, p in zip(bbc_texts, bbc_probs)],
        batch_size=16,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = (len(tt_loader) + len(bbc_loader)) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        # Alternate batches between TweetEval and BBC
        tt_iter = iter(tt_loader)
        bbc_iter = iter(bbc_loader)
        steps_in_epoch = len(tt_loader) + len(bbc_loader)

        for s in range(max(len(tt_loader), len(bbc_loader))):
            # 1. TweetEval Step
            try:
                batch_tt = next(tt_iter)
                txts, probs = batch_tt
                enc = tokenizer(list(txts), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                target_p = torch.tensor(np.array(probs), dtype=torch.float32).to(device)

                optimizer.zero_grad()
                logits = model(enc["input_ids"], enc["attention_mask"], "tweeteval_emotion")
                loss = kl_loss(torch.log_softmax(logits, dim=-1), target_p)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            except StopIteration:
                pass

            # 2. BBC Step
            try:
                batch_bbc = next(bbc_iter)
                txts, probs = batch_bbc
                enc = tokenizer(list(txts), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                target_p = torch.tensor(np.array(probs), dtype=torch.float32).to(device)

                optimizer.zero_grad()
                logits = model(enc["input_ids"], enc["attention_mask"], "bbc_news")
                loss = kl_loss(torch.log_softmax(logits, dim=-1), target_p)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            except StopIteration:
                pass

        avg_loss = epoch_loss / steps_in_epoch
        print(f"  Epoch {ep+1}/{epochs} | Multi-Task KL Loss: {avg_loss:.4f}", flush=True)

    print("\nTraining complete. Evaluating multi-task student on both test sets...", flush=True)
    model.eval()

    def eval_task(test_texts, test_golds, task_name, num_classes):
        preds = []
        with torch.no_grad():
            for i in range(0, len(test_texts), 32):
                b = test_texts[i : i + 32]
                enc = tokenizer(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                logits = model(enc["input_ids"], enc["attention_mask"], task_name)
                p = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                preds.extend(p)

        acc = float(accuracy_score(test_golds, preds) * 100)
        macro_f1 = float(f1_score(test_golds, preds, average="macro") * 100)
        per_class = (f1_score(test_golds, preds, average=None) * 100).tolist()
        counts = np.bincount(preds, minlength=num_classes).tolist()
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": [round(x, 2) for x in per_class],
            "counts": counts
        }

    tt_eval = eval_task(data_tweeteval["test_texts"], data_tweeteval["test_golds"], "tweeteval_emotion", 4)
    bbc_eval = eval_task(data_bbc["test_texts"], data_bbc["test_golds"], "bbc_news", 5)
    avg_f1 = (tt_eval["macro_f1"] + bbc_eval["macro_f1"]) / 2.0

    print(f"\n[{method_name}] Evaluation Results:")
    print(f"  TweetEval Emotion Macro-F1: {tt_eval['macro_f1']:.2f}% | Counts: {tt_eval['counts']}")
    print(f"  BBC News Macro-F1:          {bbc_eval['macro_f1']:.2f}% | Counts: {bbc_eval['counts']}")
    print(f"  Average Multi-Task Macro-F1: {avg_f1:.2f}%")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[✓] {method_name} multi-task model unloaded. VRAM freed.")

    return {
        "tweeteval_emotion": tt_eval,
        "bbc_news": bbc_eval,
        "average_macro_f1": avg_f1
    }

def main():
    print("="*70)
    print("MULTI-TASK DISTILLATION ABLATION STUDY")
    print("Datasets: TweetEval Emotion (4 cls) + BBC News (5 cls)")
    print("Shared Backbone: RoBERTa-base (125M) | Teacher: Gemma-3-4B-IT")
    print("="*70)

    # 1. Load Data
    tt_train, tt_golds_tr, tt_test, tt_golds_te = load_dataset_splits("tweeteval_emotion")
    bbc_train, bbc_golds_tr, bbc_test, bbc_golds_te = load_dataset_splits("bbc_news")

    # 2. Get Teacher Targets
    tt_raw_probs, tt_cal_probs = get_or_extract_teacher_targets("tweeteval_emotion")
    bbc_raw_probs, bbc_cal_probs = get_or_extract_teacher_targets("bbc_news")

    data_tt = {
        "train_texts": tt_train,
        "test_texts": tt_test,
        "test_golds": tt_golds_te,
        "std_kd_train_probs": tt_raw_probs,
        "ckd_train_probs": tt_cal_probs
    }

    data_bbc = {
        "train_texts": bbc_train,
        "test_texts": bbc_test,
        "test_golds": bbc_golds_te,
        "std_kd_train_probs": bbc_raw_probs,
        "ckd_train_probs": bbc_cal_probs
    }

    # 3. Train Multi-Task Standard KD Student
    std_kd_results = train_and_eval_multitask("std_kd", data_tt, data_bbc)

    # 4. Train Multi-Task CKD Student
    ckd_results = train_and_eval_multitask("ckd", data_tt, data_bbc)

    # 5. Summary and Comparison against Individual Models
    # Individual models from Table 6:
    # TweetEval: Std KD = 34.2%, CKD = 55.5%
    # BBC News:  Std KD = 56.3%, CKD = 71.5%
    individual_baseline = {
        "std_kd": {"tweeteval": 34.2, "bbc": 56.3, "avg": 45.25},
        "ckd":    {"tweeteval": 55.5, "bbc": 71.5, "avg": 63.50}
    }

    summary = {
        "individual_models_baseline": individual_baseline,
        "multitask_std_kd": std_kd_results,
        "multitask_ckd": ckd_results,
        "delta_ckd_over_std_kd": {
            "tweeteval": ckd_results["tweeteval_emotion"]["macro_f1"] - std_kd_results["tweeteval_emotion"]["macro_f1"],
            "bbc_news": ckd_results["bbc_news"]["macro_f1"] - std_kd_results["bbc_news"]["macro_f1"],
            "average": ckd_results["average_macro_f1"] - std_kd_results["average_macro_f1"]
        }
    }

    out_file = OUTPUT_DIR / "multitask_tweeteval_bbc_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("MULTI-TASK STUDY COMPLETE!")
    print(f"Summary written to: {out_file}")
    print(f"Individual Models Baseline:  Std KD Avg = 45.25% | CKD Avg = 63.50%")
    print(f"Multi-Task Std KD Student:   TweetEval = {std_kd_results['tweeteval_emotion']['macro_f1']:.2f}% | BBC = {std_kd_results['bbc_news']['macro_f1']:.2f}% | Avg = {std_kd_results['average_macro_f1']:.2f}%")
    print(f"Multi-Task CKD Student:      TweetEval = {ckd_results['tweeteval_emotion']['macro_f1']:.2f}% | BBC = {ckd_results['bbc_news']['macro_f1']:.2f}% | Avg = {ckd_results['average_macro_f1']:.2f}%")
    print(f"--> CKD Multi-Task Advantage: {summary['delta_ckd_over_std_kd']['average']:+.2f} pp")
    print("="*70)

if __name__ == "__main__":
    main()
