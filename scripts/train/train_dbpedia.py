"""
train_dbpedia_single_run.py
---------------------------
Trains Student (Qwen/Qwen3-4B BF16 + LoRA) on DBpedia 1,400 items
using either:
  1) Raw Teacher Soft Targets (Standard KD)
  2) Calibrated Teacher Soft Targets (Static CKD: z - W_null)

Evaluates on the 700 balanced test items (50 per class across 14 classes).
Runs as an isolated process with 85% VRAM cap to protect host DWM.
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]
VERBALIZERS = [chr(65 + i) for i in range(14)] # A through N

def format_prompt(tokenizer, text: str):
    c_str = "\n".join([f"{v}: {c}" for v, c in zip(VERBALIZERS, CLASSES)])
    content = (
        "Classify the text into one of fourteen categories:\n"
        f"{c_str}\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\n"
        "Answer:"
    )
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DBpediaKDDataset(Dataset):
    def __init__(self, texts, probs, golds, tokenizer):
        self.texts = texts
        self.probs = probs
        self.golds = golds
        self.tokenizer = tokenizer
        self.prompts = [format_prompt(tokenizer, t) for t in texts]

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
    prompts = [format_prompt(tokenizer, r["text"]) for r in test_rows]
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
            del inp, out

    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    pred_counts = [int(np.sum(np.array(preds) == c)) for c in range(14)]
    return float(acc), float(f1), pred_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["raw", "calibrated"],
                        help="Train with raw teacher soft targets (Standard KD) or calibrated (Static CKD)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    npz_path = DATA_DIR / "teacher_targets" / "teacher_targets_dbpedia_1400_gemma3_4b.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing targets at {npz_path}")
    npz = np.load(npz_path)

    if args.mode == "raw":
        train_probs = npz["train_probs_raw"]
    else:
        train_probs = npz["train_probs_cal"]

    train_csv = DATA_DIR / "external_benchmarks" / "dbpedia_train_1400.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    test_csv = DATA_DIR / "external_benchmarks" / "dbpedia_test_700.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print("\n" + "="*75, flush=True)
    print(f"DISTILLING QWEN3-4B ON DBPEDIA-14 | MODE: {args.mode.upper()}", flush=True)
    print("="*75, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers token mapping: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    dataset = DBpediaKDDataset([r["text"] for r in train_rows], train_probs, [int(r["label"]) for r in train_rows], tokenizer)
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

            loss = kl_loss_fn(student_log_probs, teacher_probs) / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        test_acc, test_f1, test_counts = evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device)
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} | KL Loss = {avg_loss:.4f} | Test Acc = {test_acc*100:.2f}% | Macro-F1 = {test_f1*100:.2f}%", flush=True)
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "test_acc": test_acc * 100,
            "test_f1": test_f1 * 100,
            "test_counts": test_counts
        })

    result_data = {
        "mode": args.mode,
        "dataset": "DBpedia-14",
        "num_classes": 14,
        "class_names": CLASSES,
        "test_acc": test_acc * 100,
        "test_f1": test_f1 * 100,
        "test_counts": test_counts,
        "history": history
    }

    out_file1 = RESULTS_DIR / f"dbpedia_distillation_{args.mode}_results.json"
    with open(out_file1, "w") as f:
        json.dump(result_data, f, indent=2)

    cal_kd_file = Path(f"/home/vishrut/calibrated-kd/results/dbpedia/dbpedia_distillation_{args.mode}_results.json")
    cal_kd_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_kd_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[✓] Finished {args.mode.upper()}: Acc={test_acc*100:.2f}% | F1={test_f1*100:.2f}%")
    print(f"[✓] Saved to {out_file1} and {cal_kd_file}", flush=True)

if __name__ == "__main__":
    main()
