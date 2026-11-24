"""
train_skewed_causal.py
----------------------
Distills Gemma-3-4B-it teacher targets into Causal LM students (Qwen2.5-0.5B and Qwen3-4B)
across the natural and skewed benchmarks:
  1) Emotion Natural
  2) TREC-6 Natural
  3) AG News Skewed
  4) SST-2 Skewed

Evaluates both:
  - Standard KD (Raw teacher soft targets)
  - Calibrated KD (Static CKD targets: z - W_null)
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "outputs" / "ablations" / "1_cross_architecture"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "qwen05": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "batch_size": 4,
        "grad_accum": 8,
        "lr": 5e-5,
        "epochs": 2
    },
    "qwen3": {
        "name": "Qwen/Qwen3-4B",
        "batch_size": 2,
        "grad_accum": 16,
        "lr": 5e-5,
        "epochs": 2
    }
}

TEMPLATES = {
    "agnews": {
        "name": "AGNews",
        "verbalizers": ["A", "B", "C", "D"],
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "system": (
            "Classify the news article into one of four topics:\n"
            "A: World\nB: Sports\nC: Business\nD: Science and Technology\n\n"
            "Respond with ONLY the single letter corresponding to the correct category."
        ),
        "build_user": lambda text: f"Article: {text.strip()[:1000]}\nAnswer:"
    },
    "emotion": {
        "name": "Emotion",
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "classes": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "system": None,
        "template": (
            "Classify the emotion expressed in the text into exactly one of the following categories:\n"
            "A: Sadness\nB: Joy\nC: Love\nD: Anger\nE: Fear\nF: Surprise\n\n"
            "Text: \"{text}\"\n\n"
            "Respond with ONLY the single letter (A, B, C, D, E, or F) corresponding to the correct category."
        )
    },
    "trec": {
        "name": "TREC-6",
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "classes": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "system": None,
        "template": (
            "Classify the given question into exactly one of the following 6 question types:\n"
            "A: Abbreviation (abbreviation, expression)\n"
            "B: Entity (animal, body, color, creation, currency, disease, event, food, instrument, lang, letter, other, plant, product, religion, sport, substance, symbol, technique, term, vehicle, word)\n"
            "C: Description (definition, description, manner, reason)\n"
            "D: Human (group, individual, title, description)\n"
            "E: Location (city, country, mountain, other, state)\n"
            "F: Numeric (code, count, date, distance, money, order, other, period, percent, speed, temp, size, weight)\n\n"
            "Question: \"{text}\"\n\n"
            "Respond with ONLY the single letter (A, B, C, D, E, or F) corresponding to the correct question type."
        )
    },
    "sst2": {
        "name": "SST-2",
        "verbalizers": ["A", "B"],
        "classes": ["Negative", "Positive"],
        "system": (
            "Classify the sentiment of the sentence into one of two categories:\n"
            "A: Negative\nB: Positive\n\n"
            "Respond with ONLY the single letter corresponding to the correct category."
        ),
        "build_user": lambda text: f"Sentence: {text.strip()[:1000]}\nAnswer:"
    }
}

BENCHMARKS = {
    "emotion": {
        "npz": "teacher_targets_emotion_natural_1200_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/emotion_natural_train_1200.csv",
        "test_csv": "data/benchmarks/skewed/emotion_natural_test_500.csv",
        "num_classes": 6
    },
    "trec": {
        "npz": "teacher_targets_trec_natural_1200_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/trec_natural_train_1200.csv",
        "test_csv": "data/benchmarks/trec_test_500.csv",
        "num_classes": 6
    },
    "agnews": {
        "npz": "teacher_targets_agnews_skewed_1000_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/agnews_skewed_train_1000.csv",
        "test_csv": "data/benchmarks/ag_news_test_1000.csv",
        "num_classes": 4
    },
    "sst2": {
        "npz": "teacher_targets_sst2_skewed_1000_gemma3_4b.npz",
        "train_csv": "data/benchmarks/skewed/sst2_skewed_train_1000.csv",
        "test_csv": "data/benchmarks/sst2_test_200.csv",
        "num_classes": 2
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_prompt(tokenizer, bname, text):
    cfg = TEMPLATES[bname]
    if cfg["system"] is not None:
        msgs = [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": cfg["build_user"](text)}
        ]
    else:
        msgs = [
            {"role": "user", "content": cfg["template"].format(text=text.strip()[:600])}
        ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

class CausalKDDataset(Dataset):
    def __init__(self, texts, probs, golds, tokenizer, bname, max_length=256):
        self.texts = texts
        self.probs = probs
        self.golds = golds
        self.prompts = [build_prompt(tokenizer, bname, t) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "prob": torch.tensor(self.probs[idx], dtype=torch.float32),
            "gold": int(self.golds[idx])
        }

def collate_fn(batch, tokenizer, max_length=256):
    prompts = [b["prompt"] for b in batch]
    probs = torch.stack([b["prob"] for b in batch])
    golds = torch.tensor([b["gold"] for b in batch], dtype=torch.long)
    inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {
        "input_ids": inp["input_ids"],
        "attention_mask": inp["attention_mask"],
        "probs": probs,
        "golds": golds
    }

def evaluate_test(model, tokenizer, bname, test_texts, test_golds, verbalizer_ids, num_classes, device, batch_size=4):
    model.eval()
    prompts = [build_prompt(tokenizer, bname, t) for t in test_texts]
    preds = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            b_prompts = prompts[i : i + batch_size]
            inp = tokenizer(b_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float()
            top = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(top)

    acc = float(accuracy_score(test_golds, preds) * 100)
    f1 = float(f1_score(test_golds, preds, average="macro") * 100)
    pred_counts = [int(np.sum(np.array(preds) == c)) for c in range(num_classes)]
    return acc, f1, pred_counts

def train_and_eval(model_key, bname, mode, train_texts, train_probs, train_golds,
                   test_texts, test_golds, num_classes, seed=42):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mcfg = MODEL_CONFIGS[model_key]
    model_name = mcfg["name"]
    batch_size = mcfg["batch_size"]
    grad_accum = mcfg["grad_accum"]
    epochs = mcfg["epochs"]
    lr = mcfg["lr"]

    print(f"\n[{bname.upper()}] Training {mode.upper()} ({model_key}) | Batch={batch_size}x{grad_accum}, Epochs={epochs}, LR={lr}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizers = TEMPLATES[bname]["verbalizers"]
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True
    )
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

    dataset = CausalKDDataset(train_texts, train_probs, train_golds, tokenizer, bname)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["probs"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, verbalizer_ids].float()
            log_probs = torch.log_softmax(logits, dim=-1)

            loss = kl_loss_fn(log_probs, targets) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        test_acc, test_f1, pred_counts = evaluate_test(model, tokenizer, bname, test_texts, test_golds, verbalizer_ids, num_classes, device, batch_size=batch_size*2)
        print(f"  Epoch {ep}/{epochs} | Loss={total_loss/len(loader):.4f} | Acc={test_acc:.2f}% | F1={test_f1:.2f}% | Counts={pred_counts}", flush=True)

    del model, base_model, optimizer, scheduler, loader, dataset
    torch.cuda.empty_cache()

    return {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "pred_counts": pred_counts
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen05", choices=["qwen05", "qwen3"])
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "emotion", "trec", "agnews", "sst2"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmarks_to_run = [args.dataset] if args.dataset != "all" else ["emotion", "trec", "agnews", "sst2"]
    summary = {}

    for bname in benchmarks_to_run:
        bcfg = BENCHMARKS[bname]
        npz_path = DATA_DIR / "teacher_targets" / "skewed" / bcfg["npz"]
        data = np.load(npz_path)

        with open(ROOT / bcfg["train_csv"]) as f:
            tr_rows = list(csv.DictReader(f))
        with open(ROOT / bcfg["test_csv"]) as f:
            te_rows = list(csv.DictReader(f))

        train_texts = [r["text"] for r in tr_rows]
        train_golds = [int(r["label"]) for r in tr_rows]
        test_texts = [r["text"] for r in te_rows]
        test_golds = [int(r["label"]) for r in te_rows]

        res_std = train_and_eval(args.model, bname, "standard_kd", train_texts, data["train_probs_raw"], train_golds,
                                 test_texts, test_golds, bcfg["num_classes"], seed=args.seed)
        res_ckd = train_and_eval(args.model, bname, "calibrated_kd", train_texts, data["train_probs_ckd"], train_golds,
                                 test_texts, test_golds, bcfg["num_classes"], seed=args.seed)

        summary[bname] = {
            "standard_kd": res_std,
            "calibrated_kd": res_ckd,
            "delta_acc": res_ckd["test_acc"] - res_std["test_acc"],
            "delta_f1": res_ckd["test_f1"] - res_std["test_f1"]
        }

        print(f"\n[{bname.upper()} SUMMARY - {args.model.upper()}]")
        print(f"  Standard KD   : Acc={res_std['test_acc']:.2f}% | F1={res_std['test_f1']:.2f}% | Counts={res_std['pred_counts']}")
        print(f"  Calibrated KD : Acc={res_ckd['test_acc']:.2f}% | F1={res_ckd['test_f1']:.2f}% | Counts={res_ckd['pred_counts']}")
        print(f"  Delta         : ΔAcc={summary[bname]['delta_acc']:+.2f} pp | ΔF1={summary[bname]['delta_f1']:+.2f} pp")

    out_file = RESULTS_DIR / f"skewed_{args.model}_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Saved causal summary to {out_file}", flush=True)

if __name__ == "__main__":
    main()

