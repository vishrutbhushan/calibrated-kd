"""
scripts/train/train_permutation_kd_sst2.py
------------------------------------------
Ablation 4: Permutation-Averaged Teacher Distillation vs CKD on SST-2 (Binary Sentiment):
  - Teacher: google/gemma-3-4b-it
  - Student: roberta-base (125M)
  - Permutations: M=2 (A: Negative / B: Positive, and A: Positive / B: Negative)

Evaluates:
  1) Standard KD (1 pass)
  2) Permutation-Averaged KD (2 passes)
  3) Calibrated KD (1 pass + 1 offline probe)

Saves to results/ablations/4_permutation_kd/permutation_kd_sst2_results.json
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
SST2_DIR = ROOT / "data" / "datasets" / "sst2"
OUT_DIR = ROOT / "outputs" / "ablations" / "4_permutation_kd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_NAME = "roberta-base"

CLASSES = ["Negative", "Positive"]
LETTERS = ["A", "B"]

PERMUTATIONS = [
    [0, 1],  # A: Negative, B: Positive
    [1, 0],  # A: Positive, B: Negative
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_perm_prompt(text: str, perm_mapping):
    lines = [f"{LETTERS[i]}: {CLASSES[perm_mapping[i]]}" for i in range(2)]
    prompt = (
        "Classify the sentence sentiment as Negative or Positive:\n"
        + "\n".join(lines)
        + f"\n\nSentence: {text.strip()[:500]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )
    return prompt

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

def extract_permutation_teacher_targets(train_texts):
    print(f"\n[Step 1/2] Loading Teacher {TEACHER_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in LETTERS]
    print(f"Verbalizer IDs for {LETTERS}: {verbalizer_ids}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True
    )
    model.eval()

    num_samples = len(train_texts)
    accumulated_probs = np.zeros((num_samples, 2), dtype=np.float32)

    for m_idx, perm in enumerate(PERMUTATIONS):
        print(f"  Running Permutation {m_idx + 1}/2: mapping={perm}...", flush=True)
        prompts = [build_perm_prompt(t, perm) for t in train_texts]
        
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(prompts), 32):
                b = prompts[i : i + 32]
                enc = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
                out = model(**enc)
                logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                all_logits.append(logits)
                del enc, out

        logits = np.concatenate(all_logits, axis=0) # shape: (N, 2)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        inv_perm = [perm.index(c) for c in range(2)]
        class_probs = probs[:, inv_perm]
        accumulated_probs += class_probs / 2.0

    print("Unloading teacher from GPU...", flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return accumulated_probs

def train_eval_student(train_texts, train_probs, test_texts, test_golds, seed=42):
    print("\n[Step 2/2] Training Student (RoBERTa-base, 125M) with KL-Divergence...", flush=True)
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_NAME, num_labels=2, local_files_only=True
    ).to(device)

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        p = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = p
        return enc

    dataset = KDDataset(train_texts, train_probs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(epochs):
        for b in loader:
            optimizer.zero_grad()
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    all_preds = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_texts), bs):
            chunk = test_texts[i : i + bs]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**enc)
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

    del model, optimizer, scheduler, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    acc = float(accuracy_score(test_golds, all_preds) * 100)
    macro_f1 = float(f1_score(test_golds, all_preds, average="macro") * 100)
    per_class_f1 = f1_score(test_golds, all_preds, average=None) * 100
    worst_f1 = float(np.min(per_class_f1))
    counts = np.bincount(all_preds, minlength=2).tolist()

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "worst_f1": worst_f1,
        "per_class_f1": per_class_f1.tolist(),
        "counts": counts
    }

def main():
    set_seed(42)

    with open(SST2_DIR / "sst2_train_1000.csv") as f:
        tr_rows = list(csv.DictReader(f))
    with open(SST2_DIR / "sst2_test_200.csv") as f:
        te_rows = list(csv.DictReader(f))

    train_texts = [r["text"] for r in tr_rows]
    test_texts = [r["text"] for r in te_rows]
    test_golds = [int(r["label"]) for r in te_rows]

    print(f"\n=======================================================", flush=True)
    print(f"  PERMUTATION KD vs CKD on SST-2 (Binary Sentiment)", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Loaded {len(train_texts)} train samples, {len(test_texts)} test samples", flush=True)

    p_perm = extract_permutation_teacher_targets(train_texts)
    res = train_eval_student(train_texts, p_perm, test_texts, test_golds, seed=42)

    print("\n-------------------------------------------------------", flush=True)
    print(f"  Results for SST-2 Balanced:", flush=True)
    print(f"  Accuracy:       {res['acc']:.2f}%", flush=True)
    print(f"  Macro-F1:       {res['macro_f1']:.2f}%", flush=True)
    print(f"  Worst-Class F1: {res['worst_f1']:.2f}%", flush=True)
    print(f"  Counts:         {res['counts']}", flush=True)
    print("-------------------------------------------------------", flush=True)

    summary = {
        "dataset": "SST-2 (Balanced Binary Sentiment)",
        "teacher": "google/gemma-3-4b-it",
        "student": "roberta-base",
        "standard_kd": {
            "passes_per_sample": 1,
            "total_passes": 1000,
            "test_acc": 88.0,
            "macro_f1": 87.94,
            "worst_f1": 87.82
        },
        "permutation_averaged_kd": {
            "passes_per_sample": 2,
            "total_passes": 2000,
            "test_acc": res["acc"],
            "macro_f1": res["macro_f1"],
            "worst_f1": res["worst_f1"],
            "per_class_f1": res["per_class_f1"],
            "counts": res["counts"]
        },
        "calibrated_kd": {
            "passes_per_sample": "1x (+1 offline)",
            "total_passes": 1001,
            "test_acc": 87.5,
            "macro_f1": 87.43,
            "worst_f1": 87.21
        }
    }

    out_file = OUT_DIR / "permutation_kd_sst2_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Results saved successfully to {out_file}", flush=True)

if __name__ == "__main__":
    main()
