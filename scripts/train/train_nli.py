"""
Train Student Model on NLI via Knowledge Distillation
Student: google/gemma-3-4b-it (LoRA r=16, alpha=32)
Teacher targets generated from: Qwen/Qwen3-4B
Modes:
  --mode raw         : Standard KD using raw teacher soft targets
  --mode calibrated  : Calibrated KD (CKD) using calibrated teacher soft targets
"""

import argparse
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
STUDENT_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

VERBALIZERS = ["A", "B", "C"]

def format_prompt(premise, hypothesis):
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Does the premise entail, remain neutral, or contradict the hypothesis?\n"
        "Options:\n"
        "A. Entailment\n"
        "B. Neutral\n"
        "C. Contradiction\n"
        "Answer:"
    )

class NLIKDDataset(Dataset):
    def __init__(self, rows, probs, golds, tokenizer, max_length=512):
        self.rows = rows
        self.probs = probs
        self.golds = golds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = format_prompt(row["premise"], row["hypothesis"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "probs": self.probs[idx],
            "gold": self.golds[idx]
        }

def collate_fn(batch, tokenizer):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_masks, probs, golds = [], [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append([tokenizer.pad_token_id] * pad_len + b["input_ids"])
        attention_masks.append([0] * pad_len + b["attention_mask"])
        probs.append(b["probs"])
        golds.append(b["gold"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "probs": torch.tensor(np.array(probs), dtype=torch.float),
        "golds": torch.tensor(golds, dtype=torch.long)
    }

def evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device, batch_size=16):
    model.eval()
    all_preds = []
    all_golds = [int(r["label"]) for r in test_rows]
    prompts = [format_prompt(r["premise"], r["hypothesis"]) for r in test_rows]

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            out = model(**enc)
            logits = out.logits[:, -1, verbalizer_ids].float()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)

    acc = accuracy_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds, average="macro")
    counts = np.bincount(all_preds, minlength=3).tolist()
    return acc, f1, counts, all_preds

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["raw", "calibrated"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    npz_path = DATA_DIR / "teacher_targets" / "teacher_targets_nli_qwen3_4b.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Teacher targets not found at {npz_path}")

    npz = np.load(npz_path)
    train_probs = npz["train_probs_raw"] if args.mode == "raw" else npz["train_probs_cal"]
    train_golds = npz["train_golds"]

    train_csv = DATA_DIR / "external_benchmarks" / "nli_train_900.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    test_csv = DATA_DIR / "external_benchmarks" / "nli_test_300.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print(f"\n=======================================================", flush=True)
    print(f"Distilling Gemma-3-4B on NLI | Mode: {args.mode.upper()}", flush=True)
    print(f"Teacher: Qwen3-4B | Student: Gemma-3-4B (LoRA)", flush=True)
    print(f"=======================================================", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
    print(f"Verbalizers token mapping: {list(zip(VERBALIZERS, verbalizer_ids))}", flush=True)

    dataset = NLIKDDataset(train_rows, train_probs, train_golds, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    # Load Base Student in BF16
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True
    )
    base_model.eval()

    # Pre-train Zero-Shot evaluation
    acc_0, f1_0, counts_0, _ = evaluate_test(base_model, tokenizer, test_rows, verbalizer_ids, device)
    print(f"\n[Zero-Shot Student Pre-Train] Acc: {acc_0*100:.2f}% | Macro-F1: {f1_0*100:.2f}% | Preds: {counts_0}", flush=True)

    # Attach LoRA to Student
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    output_dir = Path("models/checkpoints") / f"student_nli_{args.mode}_kd_lora"
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
                if global_step % 5 == 0 or global_step == 1:
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RESULTS_DIR / f"nli_student_{args.mode}_results.json"
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

