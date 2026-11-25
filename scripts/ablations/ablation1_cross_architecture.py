"""
ablation1_cross_architecture.py
-------------------------------
Ablation 1: Cross-Architecture Triad evaluated strictly on the Main 6 benchmarks.

Configurations:
  1. Same-Size Distillation (Gemma-3-4B -> Qwen3-4B LoRA) on BBC News (K=5)
  2. Cross-Family Distillation (Qwen3-4B -> RoBERTa-base) on TweetEval Emotion (K=4)
  3. Small SLM Compression (Gemma-3-4B -> Qwen2.5-0.5B LoRA) on TREC-6 (K=6)

Execution is strictly sequential with full VRAM cleanup between models.
Outputs saved natively to outputs/ablations/1_cross_architecture/cross_architecture_summary.json.
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
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
CACHE_DIR = ROOT / "outputs" / "main_experiments" / "cache"
OUT_DIR = ROOT / "outputs" / "ablations" / "1_cross_architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN3_4B_PATH = (
    "/home/vishrut/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
QWEN25_05B_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ROBERTA_NAME = "roberta-base"

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
        torch.cuda.ipc_collect()


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def load_splits(task_name: str):
    task_dir = DATA_BASE / task_name
    tr_txt, tr_y, te_txt, te_y = [], [], [], []
    with open(task_dir / "train_1200.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tr_txt.append(r["text"])
            tr_y.append(int(r["label"]))
    with open(task_dir / "test_400.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            te_txt.append(r["text"])
            te_y.append(int(r["label"]))
    return tr_txt, tr_y, te_txt, te_y


class CausalKDDataset(Dataset):
    def __init__(self, prompts, probs):
        self.prompts = prompts
        self.probs = probs

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.probs[idx]


class EncoderKDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]


def train_causal_student(
    model_path: str,
    task_desc: str,
    verbalizers: list[str],
    train_texts: list[str],
    train_probs: np.ndarray,
    test_texts: list[str],
    test_golds: list[int],
    batch_size: int,
    grad_accum: int,
    label: str,
) -> dict:
    """Fine-tunes a causal LLM student (Qwen3-4B or Qwen2.5-0.5B) with LoRA using KL-divergence."""
    set_seed(SEED)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    v_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]

    def fmt(t: str) -> str:
        return (
            f"{task_desc}\n\n"
            f"Text: {t.strip()[:600]}\n"
            "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
        )

    train_prompts = [fmt(t) for t in train_texts]
    test_prompts = [fmt(t) for t in test_texts]

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    def collate_fn(batch):
        ps = [b[0] for b in batch]
        pr = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(ps, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = pr
        return enc

    loader = DataLoader(CausalKDDataset(train_prompts, train_probs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 2
    total_steps = (len(loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=max(1, total_steps))
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        for step, b in enumerate(loader):
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            logits = out.logits[:, -1, v_ids].float()
            loss = kl_loss(torch.log_softmax(logits, dim=-1), b["probs"].to(device)) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        print(f"    [{label}] Epoch {ep + 1}/{epochs} complete", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_prompts), 8):
            enc = tokenizer(test_prompts[i : i + 8], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**enc)
            logits = out.logits[:, -1, v_ids].float()
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            del enc, out

    del model, optimizer, scheduler, loader, tokenizer
    clear_vram()

    return {
        "macro_f1": float(f1_score(test_golds, preds, average="macro") * 100.0),
        "acc": float(accuracy_score(test_golds, preds) * 100.0),
    }


def train_encoder_student(
    train_texts: list[str],
    train_probs: np.ndarray,
    test_texts: list[str],
    test_golds: list[int],
    num_classes: int,
    label: str,
) -> dict:
    """Fine-tunes RoBERTa-base student using KL-divergence."""
    set_seed(SEED)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME, local_files_only=True)

    def collate_fn(batch):
        texts = [b[0] for b in batch]
        probs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc["probs"] = probs
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_NAME, num_labels=num_classes, local_files_only=True
    ).to(device)
    loader = DataLoader(EncoderKDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate_fn)
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
        print(f"    [{label}] Epoch {ep + 1}/{epochs} complete", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            enc = tokenizer(test_texts[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            del enc, out

    del model, optimizer, scheduler, loader, tokenizer
    clear_vram()

    return {
        "macro_f1": float(f1_score(test_golds, preds, average="macro") * 100.0),
        "acc": float(accuracy_score(test_golds, preds) * 100.0),
    }


def run_config1_same_size():
    """Config 1: Gemma-3-4B -> Qwen3-4B LoRA on BBC News (K=5)."""
    print("\n" + "=" * 75)
    print("ABLATION 1A: SAME-SIZE (Gemma-3-4B -> Qwen3-4B LoRA) on BBC NEWS")
    print("=" * 75)
    tr_txt, _, te_txt, te_y = load_splits("bbc_news")
    cached = np.load(CACHE_DIR / "bbc_news_gemma4b_targets.npz")
    p_std = softmax(cached["train_logits"], axis=-1)
    p_ckd = softmax(cached["train_logits"] - cached["null_logits"], axis=-1)

    task_desc = (
        "Classify the news article into one of five categories:\n"
        "A: Tech\nB: Business\nC: Sport\nD: Entertainment\nE: Politics"
    )
    verbs = ["A", "B", "C", "D", "E"]
    r_std = train_causal_student(QWEN3_4B_PATH, task_desc, verbs, tr_txt, p_std, te_txt, te_y, 2, 16, "Qwen3-4B Std KD")
    r_ckd = train_causal_student(QWEN3_4B_PATH, task_desc, verbs, tr_txt, p_ckd, te_txt, te_y, 2, 16, "Qwen3-4B CKD")
    print(f"  [BBC News Same-Size] Std: {r_std['macro_f1']:.2f}% ({r_std['acc']:.2f}%) | CKD: {r_ckd['macro_f1']:.2f}% ({r_ckd['acc']:.2f}%)")
    return {"dataset": "bbc_news", "teacher": "Gemma-3-4B-IT", "student": "Qwen3-4B (LoRA)", "standard_kd": r_std, "calibrated_kd": r_ckd, "delta_f1": r_ckd["macro_f1"] - r_std["macro_f1"]}


def run_config2_cross_family():
    """Config 2: Qwen3-4B -> RoBERTa-base on TweetEval Emotion (K=4)."""
    print("\n" + "=" * 75)
    print("ABLATION 1B: CROSS-FAMILY (Qwen3-4B -> RoBERTa-base) on TWEETEVAL EMOTION")
    print("=" * 75)
    clear_vram()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tr_txt, _, te_txt, te_y = load_splits("tweeteval_emotion")
    task_desc = "Classify the tweet into one of four emotions:\nA: Anger\nB: Joy\nC: Optimism\nD: Sadness"
    verbs = ["A", "B", "C", "D"]

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_4B_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    v_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbs]

    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_4B_PATH, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True
    )
    model.eval()

    def fmt(t: str) -> str:
        return f"{task_desc}\n\nText: {t.strip()[:600]}\nRespond with ONLY the single letter corresponding to the correct category.\nAnswer:"

    with torch.no_grad():
        inp_null = tokenizer(fmt("N/A"), return_tensors="pt").to(device)
        null_logits = model(**inp_null).logits[0, -1, v_ids].float().cpu().numpy()
        del inp_null

    train_logits = []
    with torch.no_grad():
        for i in range(0, len(tr_txt), 8):
            enc = tokenizer([fmt(t) for t in tr_txt[i : i + 8]], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model(**enc)
            train_logits.append(out.logits[:, -1, v_ids].float().cpu().numpy())
            del enc, out
    train_logits = np.concatenate(train_logits, axis=0)

    del model, tokenizer
    clear_vram()

    p_std = softmax(train_logits, axis=-1)
    p_ckd = softmax(train_logits - null_logits, axis=-1)

    r_std = train_encoder_student(tr_txt, p_std, te_txt, te_y, 4, "RoBERTa Std KD")
    r_ckd = train_encoder_student(tr_txt, p_ckd, te_txt, te_y, 4, "RoBERTa CKD")
    print(f"  [TweetEval Cross-Family] Std: {r_std['macro_f1']:.2f}% ({r_std['acc']:.2f}%) | CKD: {r_ckd['macro_f1']:.2f}% ({r_ckd['acc']:.2f}%)")
    return {"dataset": "tweeteval_emotion", "teacher": "Qwen3-4B", "student": "RoBERTa-base", "standard_kd": r_std, "calibrated_kd": r_ckd, "delta_f1": r_ckd["macro_f1"] - r_std["macro_f1"]}


def run_config3_small_slm():
    """Config 3: Gemma-3-4B -> Qwen2.5-0.5B LoRA on TREC-6 (K=6)."""
    print("\n" + "=" * 75)
    print("ABLATION 1C: SMALL SLM (Gemma-3-4B -> Qwen2.5-0.5B LoRA) on TREC-6")
    print("=" * 75)
    tr_txt, _, te_txt, te_y = load_splits("trec6")
    cached = np.load(CACHE_DIR / "trec6_gemma4b_targets.npz")
    p_std = softmax(cached["train_logits"], axis=-1)
    p_ckd = softmax(cached["train_logits"] - cached["null_logits"], axis=-1)

    task_desc = (
        "Classify the question into one of six question types:\n"
        "A: Abbreviation\nB: Entity\nC: Description\nD: Human\nE: Location\nF: Numeric"
    )
    verbs = ["A", "B", "C", "D", "E", "F"]
    r_std = train_causal_student(QWEN25_05B_PATH, task_desc, verbs, tr_txt, p_std, te_txt, te_y, 4, 8, "Qwen2.5-0.5B Std KD")
    r_ckd = train_causal_student(QWEN25_05B_PATH, task_desc, verbs, tr_txt, p_ckd, te_txt, te_y, 4, 8, "Qwen2.5-0.5B CKD")
    print(f"  [TREC-6 Small SLM] Std: {r_std['macro_f1']:.2f}% ({r_std['acc']:.2f}%) | CKD: {r_ckd['macro_f1']:.2f}% ({r_ckd['acc']:.2f}%)")
    return {"dataset": "trec6", "teacher": "Gemma-3-4B-IT", "student": "Qwen2.5-0.5B (LoRA)", "standard_kd": r_std, "calibrated_kd": r_ckd, "delta_f1": r_ckd["macro_f1"] - r_std["macro_f1"]}


def main() -> None:
    res = {
        "config1_same_size": run_config1_same_size(),
        "config2_cross_family": run_config2_cross_family(),
        "config3_small_slm": run_config3_small_slm(),
    }
    out_file = OUT_DIR / "cross_architecture_summary.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2)
    print(f"\n[✓] Saved Ablation 1 summary to: {out_file}")


if __name__ == "__main__":
    main()
