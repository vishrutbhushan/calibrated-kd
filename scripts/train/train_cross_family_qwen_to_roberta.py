"""
scripts/train/train_cross_family_qwen_to_roberta.py
---------------------------------------------------
Ablation 1 (Config 2): Cross-Family Distillation:
  Teacher: Qwen/Qwen3-4B (4B Causal Decoder)
  Student: roberta-base (125M Masked Bidirectional Encoder, Full Fine-Tuning)
Evaluated across:
  1) AG News Balanced (1,000 train, 1,000 test; 4 classes, 250/class)
  2) AG News Skewed (1,000 train, 1,000 test; 4 classes, 10% Sci/Tech minority)

Extracts Qwen3-4B teacher targets (raw & calibrated via b_null = z("N/A")),
unloads teacher from GPU, trains RoBERTa-base with Standard KD and CKD,
and saves results to results/ablations/1_cross_architecture/.
"""

import os
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
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
AGNEWS_DIR = ROOT / "data" / "datasets" / "agnews"
OUT_DIR = ROOT / "results" / "ablations" / "1_cross_architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_PATH = "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
STUDENT_NAME = "roberta-base"

SYSTEM_INSTRUCTION = (
    "Classify the news article into one of four topics:\n"
    "A: World\nB: Sports\nC: Business\nD: Science and Technology\n\n"
    "Respond with ONLY the single letter corresponding to the correct category."
)
LETTERS = ["A", "B", "C", "D"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_prompt(tokenizer, text: str):
    msgs = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Article: {text.strip()[:1000]}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "<think>\n</think>\n"

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

def extract_qwen_targets(texts_dict):
    print(f"\n[Step 1/3] Loading Teacher Qwen3-4B from {TEACHER_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(l, add_special_tokens=False)[-1] for l in LETTERS]
    print(f"Verbalizer IDs for {LETTERS}: {verbalizer_ids}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True
    )
    model.eval()

    # Context-free null probe
    null_prompt = build_prompt(tokenizer, "N/A")
    enc_null = tokenizer([null_prompt], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out_null = model(**enc_null)
        null_logits = out_null.logits[0, -1, verbalizer_ids].float().cpu().numpy()
    print(f"Context-free null logits (W_null): {null_logits} (Gap: {null_logits.max() - null_logits.min():.2f} L)", flush=True)

    targets_by_split = {}
    for split_name, texts in texts_dict.items():
        print(f"  Extracting Qwen3-4B targets for '{split_name}' ({len(texts)} samples)...", flush=True)
        prompts = [build_prompt(tokenizer, t) for t in texts]
        
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(prompts), 16):
                b = prompts[i : i + 16]
                enc = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
                out = model(**enc)
                logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                all_logits.append(logits)
                del enc, out

        logits = np.concatenate(all_logits, axis=0) # shape: (N, 4)
        
        # Raw probs
        shift_raw = logits - logits.max(axis=-1, keepdims=True)
        exp_raw = np.exp(shift_raw)
        probs_raw = exp_raw / exp_raw.sum(axis=-1, keepdims=True)

        # Calibrated probs: z - W_null
        cal_logits = logits - null_logits
        shift_cal = cal_logits - cal_logits.max(axis=-1, keepdims=True)
        exp_cal = np.exp(shift_cal)
        probs_cal = exp_cal / exp_cal.sum(axis=-1, keepdims=True)

        targets_by_split[split_name] = {
            "raw": probs_raw,
            "cal": probs_cal
        }

    # Free teacher GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Qwen3-4B teacher unloaded completely from GPU.", flush=True)
    return targets_by_split, null_logits

def train_eval_roberta(train_texts, train_probs, test_texts, test_golds, split_name, mode_name, seed=42):
    print(f"  Training RoBERTa-base on [{split_name} | {mode_name}]...", flush=True)
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, local_files_only=True)

    def collate(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_NAME, num_labels=4, local_files_only=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(KDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate)
    total_steps = len(loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(3):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss_fn(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            b_texts = test_texts[i : i + 32]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            all_logits.append(out.logits.cpu().numpy())
            del enc, out

    test_logits = np.concatenate(all_logits, axis=0)
    preds = test_logits.argmax(axis=-1)

    acc = float(accuracy_score(test_golds, preds) * 100)
    macro_f1 = float(f1_score(test_golds, preds, average="macro") * 100)
    per_class_f1 = f1_score(test_golds, preds, average=None) * 100
    worst_f1 = float(np.min(per_class_f1))
    counts = np.bincount(preds, minlength=4).tolist()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"    [{split_name} | {mode_name}] -> Acc: {acc:.2f}% | Macro-F1: {macro_f1:.2f}% | Worst-F1: {worst_f1:.2f}% | Counts: {counts}", flush=True)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "worst_f1": worst_f1,
        "per_class_f1": per_class_f1.tolist(),
        "counts": counts
    }

def main():
    set_seed(42)

    # 1. Load splits
    balanced_train_csv = AGNEWS_DIR / "ag_news_train_1000.csv"
    skewed_train_csv = AGNEWS_DIR / "agnews_skewed_train_1000.csv"
    test_csv = AGNEWS_DIR / "ag_news_test_1000.csv"

    with open(balanced_train_csv) as f:
        bal_texts = [r["text"] for r in csv.DictReader(f)]
    with open(skewed_train_csv) as f:
        skew_texts = [r["text"] for r in csv.DictReader(f)]
    with open(test_csv) as f:
        te_rows = list(csv.DictReader(f))
        test_texts = [r["text"] for r in te_rows]
        test_golds = [int(r["label"]) for r in te_rows]

    print(f"Loaded Balanced Train: {len(bal_texts)} | Skewed Train: {len(skew_texts)} | Test: {len(test_texts)}")

    # 2. Extract Qwen3-4B targets
    texts_dict = {
        "balanced": bal_texts,
        "skewed": skew_texts
    }
    targets, null_logits = extract_qwen_targets(texts_dict)

    # 3. Train and evaluate RoBERTa-base students
    print("\n[Step 2/3] Distilling Qwen3-4B Teacher into RoBERTa-base...", flush=True)
    results = {
        "config": "Cross-Family Distillation (Teacher: Qwen3-4B -> Student: RoBERTa-base 125M)",
        "teacher": "Qwen/Qwen3-4B",
        "student": "roberta-base",
        "null_logits": null_logits.tolist(),
        "balanced": {
            "standard_kd": train_eval_roberta(bal_texts, targets["balanced"]["raw"], test_texts, test_golds, "Balanced", "Standard KD"),
            "calibrated_kd": train_eval_roberta(bal_texts, targets["balanced"]["cal"], test_texts, test_golds, "Balanced", "Calibrated KD")
        },
        "skewed": {
            "standard_kd": train_eval_roberta(skew_texts, targets["skewed"]["raw"], test_texts, test_golds, "Skewed", "Standard KD"),
            "calibrated_kd": train_eval_roberta(skew_texts, targets["skewed"]["cal"], test_texts, test_golds, "Skewed", "Calibrated KD")
        }
    }

    # Summary table
    bal_std = results["balanced"]["standard_kd"]
    bal_ckd = results["balanced"]["calibrated_kd"]
    skew_std = results["skewed"]["standard_kd"]
    skew_ckd = results["skewed"]["calibrated_kd"]

    print("\n=======================================================", flush=True)
    print("  ABLATION 1 (CONFIG 2): CROSS-FAMILY DISTILLATION", flush=True)
    print("  Teacher: Qwen3-4B (4B Causal) -> Student: RoBERTa-base (125M)", flush=True)
    print("=======================================================", flush=True)
    print(f"  AG News Balanced Macro-F1: {bal_std['macro_f1']:.2f}% [{bal_std['worst_f1']:.2f}%] -> {bal_ckd['macro_f1']:.2f}% [{bal_ckd['worst_f1']:.2f}%]", flush=True)
    print(f"  AG News Skewed Macro-F1:   {skew_std['macro_f1']:.2f}% [{skew_std['worst_f1']:.2f}%] -> {skew_ckd['macro_f1']:.2f}% [{skew_ckd['worst_f1']:.2f}%]", flush=True)
    print(f"  Minority Sci/Tech Recovery: {skew_std['counts'][3]} -> {skew_ckd['counts'][3]} (Gold=250)", flush=True)
    print("=======================================================", flush=True)

    out_file = OUT_DIR / "cross_family_qwen_to_roberta_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Results saved successfully to {out_file}", flush=True)

if __name__ == "__main__":
    main()
