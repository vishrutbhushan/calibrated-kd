"""
run_full_student_questions_benchmark.py
---------------------------------------
Executes a complete, unified distillation benchmark on Student Questions:
  - Teacher: Gemma-3-4B-IT (BF16, frozen)
  - Student: RoBERTa-base (125M, full fine-tune)
  - Train: 1,200 | Test: 400 (Seed: 42)

Evaluates:
  1. Teacher: Raw Zero-Shot
  2. Teacher: Standard Null-Calibrated (b_mc)
  3. Teacher: Dual-Calibrated (b_dummy + b_umbrella)
  4. Student: Standard KD (No CKD)
  5. Student: Post-Hoc Output Calibration
  6. Student: Calibrated KD (CKD, Ours with b_mc)
  7. Student: CKD + Post-Hoc Combined
  8. Student: Dual-Calibrated KD (Dual-CKD with b_dummy + b_umbrella)
  9. Student: Dual-CKD + Post-Hoc Combined
"""

import os
import gc
import csv
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
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "datasets" / "natural_benchmarks" / "student_questions"

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_MODEL_NAME = "roberta-base"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def compute_metrics(golds, preds, num_classes=4):
    acc = float(accuracy_score(golds, preds) * 100)
    macro_f1 = float(f1_score(golds, preds, average="macro") * 100)
    per_class = (f1_score(golds, preds, average=None) * 100).tolist()
    counts = np.bincount(preds, minlength=num_classes).tolist()
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "counts": counts
    }

class KDDataset(Dataset):
    def __init__(self, texts, probs):
        self.texts = texts
        self.probs = probs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.probs[idx]

def load_data():
    train_file = DATA_DIR / "train_1200.csv"
    test_file = DATA_DIR / "test_400.csv"

    train_texts, train_golds = [], []
    with open(train_file, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            train_texts.append(r["text"])
            train_golds.append(int(r["label"]))

    test_texts, test_golds = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            test_texts.append(r["text"])
            test_golds.append(int(r["label"]))

    return train_texts, train_golds, test_texts, test_golds

def prompt_mc(text):
    return (
        "Classify the academic question into the correct subject:\n"
        "A: Biology\nB: Physics\nC: Maths\nD: Chemistry\n\n"
        f"Text: {text.strip()}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )

def extract_teacher_all(train_texts, test_texts):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mc_v_ids = [tok.encode(v, add_special_tokens=False)[-1] for v in ["A", "B", "C", "D"]]
    words = [" Biology", " Physics", " Math", " Chemistry"]
    word_v_ids = [tok.encode(w, add_special_tokens=False)[-1] for w in words]

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True
    )
    model.eval()

    # 1. Multiple-Choice Null Vector (b_mc)
    p_mc_null = prompt_mc("N/A")
    inp = tok(p_mc_null, return_tensors="pt").to(device)
    with torch.no_grad():
        b_mc = model(**inp).logits[0, -1, mc_v_ids].float().cpu().numpy()

    # 2. Dummy Option Null Vector (b_dummy)
    p_dummy_null = (
        "Classify the academic question into the correct subject:\n"
        "A: Option 1\nB: Option 2\nC: Option 3\nD: Option 4\n\n"
        "Text: N/A\nRespond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )
    inp = tok(p_dummy_null, return_tensors="pt").to(device)
    with torch.no_grad():
        b_dummy = model(**inp).logits[0, -1, mc_v_ids].float().cpu().numpy()

    # 3. Open Umbrella Null Vector (b_umbrella)
    p_umb_null = "Question: N/A\nWhich academic subject is this question about? Answer:"
    inp = tok(p_umb_null, return_tensors="pt").to(device)
    with torch.no_grad():
        b_umbrella = model(**inp).logits[0, -1, word_v_ids].float().cpu().numpy()

    b_dual = b_dummy + b_umbrella

    print("\n[Teacher Null Extraction Complete]")
    print(f"  b_mc:       {b_mc.round(2).tolist()}")
    print(f"  b_dummy:    {b_dummy.round(2).tolist()}")
    print(f"  b_umbrella: {b_umbrella.round(2).tolist()}")
    print(f"  b_dual:     {b_dual.round(2).tolist()}")

    def run_batches(texts_list, desc=""):
        prompts = [prompt_mc(t) for t in texts_list]
        res = []
        bs = 4
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                chunk = prompts[i : i + bs]
                inp = tok(chunk, return_tensors="pt", padding=True, truncation=False).to(device)
                out = model(**inp)
                res.append(out.logits[:, -1, mc_v_ids].float().cpu().numpy())
                del inp, out
                if (i // bs) % 60 == 0:
                    print(f"  {desc}: {min(i + bs, len(prompts))}/{len(prompts)}...", flush=True)
            torch.cuda.empty_cache()
        return np.concatenate(res, axis=0)

    print("\nExtracting Teacher logits on Test Set (400 items)...", flush=True)
    test_logits = run_batches(test_texts, "Test")

    print("\nExtracting Teacher logits on Train Set (1200 items)...", flush=True)
    train_logits = run_batches(train_texts, "Train")

    # Complete unload
    print("\nUnloading Teacher model from GPU memory...", flush=True)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print("[✓] Teacher successfully unloaded. VRAM completely freed.")

    return b_mc, b_dual, test_logits, train_logits

def train_student_model(train_texts, train_probs, test_texts, test_golds, num_classes=4, seed=42):
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, local_files_only=True)
    max_len = 512

    def collate_fn(batch):
        txts = [b[0] for b in batch]
        prbs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(txts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["probs"] = prbs
        return enc

    np.random.seed(seed)
    cal_indices = np.random.choice(len(train_texts), size=min(300, len(train_texts)), replace=False)
    cal_texts = [train_texts[i] for i in cal_indices]

    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=num_classes, local_files_only=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loader = DataLoader(KDDataset(train_texts, train_probs), batch_size=16, shuffle=True, collate_fn=collate_fn)
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

    # Inference on test set
    model.eval()
    test_logits = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            chunk = test_texts[i : i + 32]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            test_logits.append(out.logits.cpu().numpy())
    test_logits = np.concatenate(test_logits, axis=0)

    # Post-hoc calibration vector extraction
    cal_logits = []
    with torch.no_grad():
        for i in range(0, len(cal_texts), 32):
            chunk = cal_texts[i : i + 32]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cal_logits.append(out.logits.cpu().numpy())
    cal_logits = np.concatenate(cal_logits, axis=0)
    mean_bias = np.mean(cal_logits, axis=0)

    del model, optimizer, scheduler, loader, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    base_preds = test_logits.argmax(axis=-1)
    base_metrics = compute_metrics(test_golds, base_preds, num_classes)

    post_logits = test_logits - mean_bias
    post_preds = post_logits.argmax(axis=-1)
    post_metrics = compute_metrics(test_golds, post_preds, num_classes)

    return base_metrics, post_metrics

def main():
    print("=" * 80)
    print("STARTING COMPLETE BENCHMARK ON: STUDENT QUESTIONS (4 CLASSES)")
    print("=" * 80)

    train_texts, train_golds, test_texts, test_golds = load_data()
    print(f"Loaded {len(train_texts)} train items, {len(test_texts)} test items.")

    # 1. Extract Teacher Targets
    b_mc, b_dual, test_logits_raw, train_logits_raw = extract_teacher_all(train_texts, test_texts)

    # 2. Evaluate Teacher Baselines
    m_teacher_raw = compute_metrics(test_golds, test_logits_raw.argmax(axis=-1))
    m_teacher_mc = compute_metrics(test_golds, (test_logits_raw - b_mc).argmax(axis=-1))
    m_teacher_dual = compute_metrics(test_golds, (test_logits_raw - b_dual).argmax(axis=-1))

    print("\n--- TEACHER ZERO-SHOT BASELINES ---")
    print(f"  Teacher Raw:            Macro-F1={m_teacher_raw['macro_f1']:.2f}% | Acc={m_teacher_raw['acc']:.2f}% | Counts={m_teacher_raw['counts']}")
    print(f"  Teacher Null-Cal (b_mc): Macro-F1={m_teacher_mc['macro_f1']:.2f}% | Acc={m_teacher_mc['acc']:.2f}% | Counts={m_teacher_mc['counts']}")
    print(f"  Teacher Dual-Cal:       Macro-F1={m_teacher_dual['macro_f1']:.2f}% | Acc={m_teacher_dual['acc']:.2f}% | Counts={m_teacher_dual['counts']}")

    # 3. Probability Target Formations for Students
    p_raw = softmax(train_logits_raw)
    p_ckd = softmax(train_logits_raw - b_mc)
    p_dual = softmax(train_logits_raw - b_dual)

    # 4. Student Distillation Runs
    print("\n[Run 1] Student: Standard KD (No CKD)...", flush=True)
    m_std, m_std_post = train_student_model(train_texts, p_raw, test_texts, test_golds)

    print("\n[Run 2] Student: Calibrated KD (CKD, b_mc)...", flush=True)
    m_ckd, m_ckd_post = train_student_model(train_texts, p_ckd, test_texts, test_golds)

    print("\n[Run 3] Student: Dual-Calibrated KD (Dual-CKD, b_dummy + b_umbrella)...", flush=True)
    m_dual, m_dual_post = train_student_model(train_texts, p_dual, test_texts, test_golds)

    print("\n" + "=" * 80)
    print("FINAL COMPLETE BENCHMARK TABLE: STUDENT QUESTIONS")
    print("=" * 80)
    print(f"{'Method / Paradigm':<38} | {'Macro-F1':<12} | {'Accuracy':<12} | {'Prediction Counts'}")
    print("-" * 80)
    print(f"{'Teacher: Raw Zero-Shot':<38} | {m_teacher_raw['macro_f1']:>6.2f}%    | {m_teacher_raw['acc']:>6.2f}%    | {m_teacher_raw['counts']}")
    print(f"{'Teacher: Null-Calibrated (b_mc)':<38} | {m_teacher_mc['macro_f1']:>6.2f}%    | {m_teacher_mc['acc']:>6.2f}%    | {m_teacher_mc['counts']}")
    print(f"{'Teacher: Dual-Calibrated (Dual-CKD)':<38} | {m_teacher_dual['macro_f1']:>6.2f}%    | {m_teacher_dual['acc']:>6.2f}%    | {m_teacher_dual['counts']}")
    print("-" * 80)
    print(f"{'Student: Standard KD (No CKD)':<38} | {m_std['macro_f1']:>6.2f}%    | {m_std['acc']:>6.2f}%    | {m_std['counts']}")
    print(f"{'Student: Post-Hoc Output Cal':<38} | {m_std_post['macro_f1']:>6.2f}%    | {m_std_post['acc']:>6.2f}%    | {m_std_post['counts']}")
    print(f"{'Student: Calibrated KD (CKD, b_mc)':<38} | {m_ckd['macro_f1']:>6.2f}%    | {m_ckd['acc']:>6.2f}%    | {m_ckd['counts']}")
    print(f"{'Student: CKD + Post-Hoc':<38} | {m_ckd_post['macro_f1']:>6.2f}%    | {m_ckd_post['acc']:>6.2f}%    | {m_ckd_post['counts']}")
    print(f"{'Student: Dual-CKD (Ours - Novel)':<38} | {m_dual['macro_f1']:>6.2f}%    | {m_dual['acc']:>6.2f}%    | {m_dual['counts']}")
    print(f"{'Student: Dual-CKD + Post-Hoc':<38} | {m_dual_post['macro_f1']:>6.2f}%    | {m_dual_post['acc']:>6.2f}%    | {m_dual_post['counts']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
