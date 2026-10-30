"""
train_trec_single_run.py
------------------------
Trains student model (Qwen/Qwen3-4B BF16 + LoRA) on TREC-6 Question Classification
for knowledge distillation variants:
  1) raw        : Standard KD (Raw teacher soft targets)
  2) static_ckd : Calibrated KD (z - W_null soft targets)

Runs as an isolated process with strict 85% VRAM cap.
Evaluates on the canonical 500 test questions across 6 categories:
  [Abbreviation, Entity, Description, Human, Location, Numeric]
Saves logs to results/trec/.
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
BENCHMARK_DIR = DATA_DIR / "benchmarks"
RESULTS_DIR = ROOT / "results" / "trec"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
VERBALIZERS = ["A", "B", "C", "D", "E", "F"]
CLASS_NAMES = ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]

PROMPT_TEMPLATE = (
    "Question: {text}\n"
    "What type of answer is the question asking for?\n"
    "Options:\n"
    "A. Abbreviation\n"
    "B. Entity\n"
    "C. Description\n"
    "D. Human\n"
    "E. Location\n"
    "F. Numeric\n"
    "Answer:"
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_prompt(tokenizer, text: str):
    msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text.strip()[:600])}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

class TRECKDDataset(Dataset):
    def __init__(self, texts, probs, golds, tokenizer, max_length=512):
        self.texts = texts
        self.probs = probs
        self.golds = golds
        self.prompts = [build_prompt(tokenizer, t) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "prob": torch.tensor(self.probs[idx], dtype=torch.float32),
            "gold": self.golds[idx]
        }

def collate_fn(batch, tokenizer, max_length=512):
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

def evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device, batch_size=4):
    model.eval()
    torch.cuda.empty_cache()
    prompts = [build_prompt(tokenizer, r["text"]) for r in test_rows]
    golds = [int(r["label"]) for r in test_rows]
    preds = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            b = prompts[i : i + batch_size]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float()
            top = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(top)

    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    pred_counts = [int(np.sum(np.array(preds) == c)) for c in range(6)]
    return float(acc), float(f1), pred_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["raw", "static_ckd"],
                        help="Distillation variant to train: raw or static_ckd")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target_npz = DATA_DIR / "teacher_targets" / "teacher_targets_trec_1200_gemma3_4b.npz"
    if not target_npz.exists():
        raise FileNotFoundError(f"Missing teacher targets file: {target_npz}")

    npz = np.load(target_npz)
    if args.variant == "raw":
        train_probs = npz["train_probs_raw"]
    elif args.variant == "static_ckd":
        train_probs = npz["train_probs_cal"]
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    train_csv = BENCHMARK_DIR / "trec_train_1200.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    test_csv = BENCHMARK_DIR / "trec_test_500.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print("\n" + "="*75, flush=True)
    print(f"RUNNING TREC-6 DISTILLATION | VARIANT: {args.variant.upper()} (Student: Qwen3-4B LoRA)", flush=True)
    print(f"Train size: {len(train_rows)} | Test size: {len(test_rows)} | Epochs: {args.epochs} | LR: {args.lr}", flush=True)
    print("="*75, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers token mapping: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    dataset = TRECKDDataset([r["text"] for r in train_rows], train_probs, [int(r["label"]) for r in train_rows], tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True
    )
    base_model.gradient_checkpointing_enable()

    # Pre-training zero-shot evaluation
    print("\nEvaluating Student Zero-Shot Baseline on 500 Test Questions...", flush=True)
    zero_acc, zero_f1, zero_counts = evaluate_test(base_model, tokenizer, test_rows, verbalizer_ids, device)
    print(f"-> Zero-Shot Student: Acc={zero_acc*100:.2f}% | Macro-F1={zero_f1*100:.2f}% | Counts={zero_counts} {list(zip(CLASS_NAMES, zero_counts))}", flush=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    history = []
    final_acc, final_f1, final_counts = 0.0, 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_probs = batch["probs"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = out.logits[:, -1, verbalizer_ids].float()
            student_log_probs = torch.log_softmax(student_logits, dim=-1)

            kd_loss = kl_loss_fn(student_log_probs, teacher_probs)
            total_loss = kd_loss / args.grad_accum
            total_loss.backward()
            epoch_loss += kd_loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        test_acc, test_f1, test_counts = evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device)
        final_acc, final_f1, final_counts = test_acc, test_f1, test_counts
        history.append({
            "epoch": epoch,
            "train_loss": float(epoch_loss / len(loader)),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "pred_counts": test_counts
        })
        print(f"Epoch {epoch}/{args.epochs} | KL Loss = {epoch_loss/len(loader):.4f} | Test Acc = {test_acc*100:.2f}% | Macro-F1 = {test_f1*100:.2f}% | Counts = {test_counts}", flush=True)

    # Save results
    out_file = RESULTS_DIR / f"trec_student_{args.variant}_results.json"
    result_data = {
        "mode": args.variant,
        "dataset": "TREC-6",
        "num_classes": 6,
        "class_names": CLASS_NAMES,
        "zero_shot": {
            "acc": zero_acc,
            "f1": zero_f1,
            "pred_counts": zero_counts
        },
        "final_epoch": {
            "epoch": args.epochs,
            "test_acc": final_acc,
            "test_f1": final_f1,
            "pred_counts": final_counts
        },
        "history": history
    }
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)

    # Also save standard summary format
    summary_file = RESULTS_DIR / f"trec_distillation_{args.variant}_results.json"
    with open(summary_file, "w") as f:
        json.dump({
            "mode": args.variant,
            "dataset": "TREC-6",
            "test_acc": final_acc * 100,
            "test_f1": final_f1 * 100,
            "test_counts": final_counts,
            "class_names": CLASS_NAMES
        }, f, indent=2)

    print(f"\n[✓] Finished {args.variant.upper()}: Acc={final_acc*100:.2f}% | F1={final_f1*100:.2f}% | Counts={final_counts}", flush=True)
    print(f"[✓] Saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()

