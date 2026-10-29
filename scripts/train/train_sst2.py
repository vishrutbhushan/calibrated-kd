"""
train_student_lora_sst2.py
--------------------------
Trains Student (Qwen/Qwen3-4B BF16 + LoRA) on SST-2 1,000 items
using either:
  1) Raw Teacher Soft Targets (Standard KD)
  2) Calibrated Teacher Soft Targets (Calibrated Knowledge Distillation - CKD)

Evaluates on the 200 balanced test items (100 Negative, 100 Positive) and logs:
- Test Accuracy
- Test Macro-F1
- Per-class prediction counts
"""

import os
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
VERBALIZERS = ["A", "B"]
CLASS_NAMES = ["Negative", "Positive"]

SYSTEM_INSTRUCTION = (
    "Classify the sentiment of the following sentence into one of two categories:\n"
    "A: Negative\nB: Positive\n\n"
    "Respond with ONLY the single letter corresponding to the correct category."
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_prompt(tokenizer, sentence_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Sentence: {sentence_text.strip()[:1000]}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

class SST2KDDataset(Dataset):
    def __init__(self, texts, probs, golds, tokenizer, max_length=512):
        self.texts = texts
        self.probs = probs
        self.golds = golds
        self.tokenizer = tokenizer
        self.max_length = max_length
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
    pred_counts = [int(np.sum(np.array(preds) == c)) for c in range(2)]
    return float(acc), float(f1), pred_counts, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["raw", "calibrated"], required=True,
                        help="Train with raw teacher soft targets (Standard KD) or calibrated (CKD)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    npz_path = DATA_DIR / "teacher_targets" / "teacher_targets_sst2_gemma3_4b.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Teacher targets not found at {npz_path}")

    npz = np.load(npz_path)
    train_probs = npz["train_probs_raw"] if args.mode == "raw" else npz["train_probs_cal"]
    train_golds = npz["train_golds"]

    train_csv = DATA_DIR / "external_benchmarks" / "sst2_train_1000.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    train_texts = [r["text"] for r in train_rows]

    test_csv = DATA_DIR / "external_benchmarks" / "sst2_test_200.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print(f"\n=======================================================", flush=True)
    print(f"Distilling Qwen3-4B on SST-2 | Mode: {args.mode.upper()}", flush=True)
    print(f"=======================================================", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers token mapping: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    dataset = SST2KDDataset(train_texts, train_probs, train_golds, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True
    )
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Pre-train zero-shot evaluation
    acc_0, f1_0, counts_0, _ = evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device)
    print(f"\n[Zero-Shot Student Pre-Train] Acc: {acc_0*100:.2f}% | Macro-F1: {f1_0*100:.2f}% | Preds: {counts_0}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=total_steps)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    output_dir = MODELS_DIR / "checkpoints" / f"student_sst2_{args.mode}_kd_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_probs = batch["probs"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, verbalizer_ids].float()
            student_log_probs = torch.log_softmax(logits, dim=-1)

            loss = kl_loss_fn(student_log_probs, target_probs) / args.grad_accum
            loss.backward()
            step_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_loss += step_loss
                if global_step % 10 == 0:
                    print(f"  Step {global_step}/{total_steps} | Step Loss: {step_loss:.4f}", flush=True)
                step_loss = 0.0

        avg_loss = epoch_loss / max(1, len(loader) // args.grad_accum)
        val_acc, val_f1, val_counts, val_preds = evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {val_acc*100:.2f}% | Test F1: {val_f1*100:.2f}% | Preds: {val_counts}", flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "test_acc": val_acc,
            "test_f1": val_f1,
            "pred_counts": val_counts
        })

    # Save checkpoint
    model.save_pretrained(output_dir)
    print(f"[✓] Saved {args.mode.upper()} student checkpoint to {output_dir}", flush=True)

    # Save summary results
    res_path = RESULTS_DIR / f"sst2_student_{args.mode}_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "zero_shot": {"acc": acc_0, "f1": f1_0, "pred_counts": counts_0},
            "final_epoch": history[-1],
            "history": history
        }, f, indent=2)
    print(f"[✓] Saved results to {res_path}", flush=True)

if __name__ == "__main__":
    main()

