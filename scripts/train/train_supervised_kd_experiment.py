"""
train_supervised_kd_experiment.py
---------------------------------
Investigates the critical research question:
Does Null-Conditioned Verbalizer Calibration provide value when knowledge
distillation is combined with standard Cross-Entropy supervision (CE + KD)?

Regimes tested:
  1) sft            : Supervised Fine-Tuning only (CE loss on ground truth, lambda_kd = 0.0)
  2) standard_kd    : Standard Supervised KD (0.5 * CE + 0.5 * Raw Teacher KD)
  3) calibrated_kd  : Calibrated Supervised KD (0.5 * CE + 0.5 * Calibrated Teacher KD)

Evaluates on the 1,000 balanced AG News test set (250/class).
Strict VRAM headroom: 85% cap.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "supervised_kd"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
VERBALIZERS = ["A", "B", "C", "D"]
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

SYSTEM_INSTRUCTION = (
    "Classify the news article into one of four topics:\n"
    "A: World\nB: Sports\nC: Business\nD: Science and Technology\n\n"
    "Respond with ONLY the single letter corresponding to the correct category."
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_prompt(tokenizer, article_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Article: {article_text.strip()[:1000]}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

class AGNewsSupervisedKDDataset(Dataset):
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
    pred_counts = [int(np.sum(np.array(preds) == c)) for c in range(4)]
    report = classification_report(golds, preds, target_names=CLASS_NAMES, output_dict=True)
    return float(acc), float(f1), pred_counts, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, required=True, 
                        choices=["sft", "standard_kd", "calibrated_kd"],
                        help="Training regime: sft, standard_kd, or calibrated_kd")
    parser.add_argument("--lambda_kd", type=float, default=0.5,
                        help="Weight for KD loss in combined objective (default 0.5)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    npz_path = DATA_DIR / "teacher_targets" / "teacher_targets_agnews_1000_gemma3_4b.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing targets at {npz_path}")
    npz = np.load(npz_path)

    if args.regime == "sft":
        train_probs = np.zeros_like(npz["train_probs_static"]) # unused in SFT
    elif args.regime == "standard_kd":
        train_probs = npz["train_probs_raw"]
    elif args.regime == "calibrated_kd":
        train_probs = npz["train_probs_static"]

    train_csv = DATA_DIR / "benchmarks" / "ag_news_train_1000.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    test_csv = DATA_DIR / "benchmarks" / "ag_news_test_1000.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print("\n" + "="*80, flush=True)
    print(f"RUNNING SUPERVISED DISTILLATION EXPERIMENT: {args.regime.upper()}", flush=True)
    print(f"Dataset: AG News (1,000 Train, 1,000 Test) | Lambda KD: {args.lambda_kd if args.regime != 'sft' else 0.0}", flush=True)
    print("="*80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]

    dataset = AGNewsSupervisedKDDataset([r["text"] for r in train_rows], train_probs, [int(r["label"]) for r in train_rows], tokenizer)
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
    
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    history = []
    final_acc, final_f1, final_counts, final_report = 0.0, 0.0, [], {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_ce = 0.0
        epoch_kd = 0.0
        epoch_total = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_probs = batch["probs"].to(device)
            golds = batch["golds"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = out.logits[:, -1, verbalizer_ids].float()

            ce_loss = ce_loss_fn(student_logits, golds)

            if args.regime == "sft":
                loss = ce_loss
                kd_loss_val = 0.0
            else:
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                kd_loss = kl_loss_fn(student_log_probs, teacher_probs)
                loss = (1.0 - args.lambda_kd) * ce_loss + args.lambda_kd * kd_loss
                kd_loss_val = kd_loss.item()

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            epoch_ce += ce_loss.item()
            epoch_kd += kd_loss_val
            epoch_total += loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        test_acc, test_f1, test_counts, report = evaluate_test(model, tokenizer, test_rows, verbalizer_ids, device)
        final_acc, final_f1, final_counts, final_report = test_acc, test_f1, test_counts, report
        history.append({
            "epoch": epoch,
            "ce_loss": float(epoch_ce / len(loader)),
            "kd_loss": float(epoch_kd / len(loader)),
            "total_loss": float(epoch_total / len(loader)),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "pred_counts": test_counts
        })
        print(f"Epoch {epoch}/{args.epochs} | CE Loss = {epoch_ce/len(loader):.4f} | KD Loss = {epoch_kd/len(loader):.4f} | Test Acc = {test_acc*100:.2f}% | Macro-F1 = {test_f1*100:.2f}% | Counts = {test_counts}", flush=True)

    out_file = RESULTS_DIR / f"agnews_{args.regime}_results.json"
    result_data = {
        "regime": args.regime,
        "dataset": "AG News",
        "lambda_kd": args.lambda_kd if args.regime != "sft" else 0.0,
        "final_epoch": {
            "epoch": args.epochs,
            "test_acc": final_acc * 100,
            "test_f1": final_f1 * 100,
            "test_counts": final_counts,
            "scitech_count": final_counts[3]
        },
        "classification_report": final_report,
        "history": history
    }
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[✓] Finished {args.regime.upper()}: Acc={final_acc*100:.2f}% | F1={final_f1*100:.2f}% | Sci/Tech={final_counts[3]}/250")
    print(f"[✓] Saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()

