"""
ablation4_permutation_kd.py
---------------------------
Ablation 4: Permutation-Averaged Teacher Distillation (Liusie et al., ACL 2024)
vs. Standard KD and CKD evaluated strictly on Main 6 benchmarks:
  - TweetEval Emotion (K=4, 4x cyclic permutations)
  - Student Questions (K=4, 4x cyclic permutations)

Outputs saved natively to outputs/ablations/4_permutation_kd/permutation_kd_results.json.
"""

import os
import gc
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
MAIN_SUMMARY = ROOT / "outputs" / "main_experiments" / "all_6_main_benchmarks_summary.json"
OUT_DIR = ROOT / "outputs" / "ablations" / "4_permutation_kd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_PATH = (
    "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/"
    "snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
)
STUDENT_NAME = "roberta-base"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]


def run_perm_for_dataset(task_name: str, header_desc: str, class_names: list[str], main_summary: dict) -> dict:
    """Runs 4x cyclic permutation extraction with Gemma-3-4B and distills into RoBERTa-base."""
    print(f"\n>>> PERMUTATION KD ON {task_name.upper()} (4x Passes) <<<")
    task_dir = DATA_BASE / task_name
    tr_txt, te_txt, te_y = [], [], []
    with open(task_dir / "train_1200.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tr_txt.append(r["text"])
    with open(task_dir / "test_400.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            te_txt.append(r["text"])
            te_y.append(int(r["label"]))

    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    verbalizers = ["A", "B", "C", "D"]
    perms = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    v_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}, local_files_only=True
    )
    model.eval()

    probs_sum = np.zeros((len(tr_txt), 4), dtype=np.float64)
    for p_idx, perm in enumerate(perms, 1):
        print(f"  Permutation {p_idx}/4: {perm}...", flush=True)
        opt_lines = "\n".join(f"{verbalizers[j]}: {class_names[perm[j]]}" for j in range(4))
        prompts = [
            f"{header_desc}\n{opt_lines}\n\nText: {t.strip()[:600]}\n"
            "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
            for t in tr_txt
        ]
        perm_logits = []
        with torch.no_grad():
            for i in range(0, len(prompts), 8):
                enc = tokenizer(prompts[i : i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                out = model(**enc)
                perm_logits.append(out.logits[:, -1, v_ids].float().cpu().numpy())
                del enc, out
        perm_probs = softmax(np.concatenate(perm_logits, axis=0), axis=-1)
        unpermed = np.zeros_like(perm_probs)
        for j, orig_cls in enumerate(perm):
            unpermed[:, orig_cls] = perm_probs[:, j]
        probs_sum += unpermed

    p_perm = (probs_sum / len(perms)).astype(np.float32)
    del model, tokenizer
    clear_vram()

    # Train RoBERTa-base student
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(STUDENT_NAME, local_files_only=True)

    def collate_fn(batch):
        texts = [b[0] for b in batch]
        probs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = probs
        return enc

    stu = AutoModelForSequenceClassification.from_pretrained(STUDENT_NAME, num_labels=4, local_files_only=True).to(device)
    loader = DataLoader(KDDataset(tr_txt, p_perm), batch_size=16, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(stu.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    stu.train()
    for _ in range(epochs):
        for b in loader:
            optimizer.zero_grad()
            out = stu(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = kl_loss(torch.log_softmax(out.logits, dim=-1), b["probs"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stu.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    stu.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_txt), 32):
            enc = tok(te_txt[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = stu(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            del enc, out

    del stu, optimizer, scheduler, loader, tok
    clear_vram()

    perm_f1 = float(f1_score(te_y, preds, average="macro") * 100.0)
    perm_acc = float(accuracy_score(te_y, preds) * 100.0)
    print(f"  [✓] {task_name} Permutation KD: F1={perm_f1:.2f}% | Acc={perm_acc:.2f}%", flush=True)

    return {
        "standard_kd": {
            "macro_f1": main_summary[task_name]["standard_kd"]["macro_f1"],
            "acc": main_summary[task_name]["standard_kd"]["acc"],
        },
        "permutation_averaged_kd": {"passes": 4, "macro_f1": perm_f1, "acc": perm_acc},
        "calibrated_kd": {
            "macro_f1": main_summary[task_name]["calibrated_kd"]["macro_f1"],
            "acc": main_summary[task_name]["calibrated_kd"]["acc"],
        },
    }


def main() -> None:
    with open(MAIN_SUMMARY, "r", encoding="utf-8") as fp:
        main_summary = json.load(fp)

    res = {
        "tweeteval_emotion": run_perm_for_dataset(
            "tweeteval_emotion", "Classify the tweet into one of four emotions:", ["Anger", "Joy", "Optimism", "Sadness"], main_summary
        ),
        "student_questions": run_perm_for_dataset(
            "student_questions", "Classify the academic question into the correct subject:", ["Biology", "Physics", "Maths", "Chemistry"], main_summary
        ),
    }

    out_file = OUT_DIR / "permutation_kd_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2)
    print(f"\n[✓] Saved Ablation 4 results to: {out_file}")


if __name__ == "__main__":
    main()
