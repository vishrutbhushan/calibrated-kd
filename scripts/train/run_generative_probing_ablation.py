"""
run_generative_probing_ablation.py
----------------------------------
Ablation study on internal representation quality via Generative Probing:
Probing mid-sized distilled LLM students (Qwen2.5-1.5B-Instruct) on open-ended
natural language category generation vs. closed-choice multiple-choice.

Hypothesis:
1. In multiple-choice (A/B/C/D), post-hoc logit adjustment (z_hat = z - z_bar) can be applied.
2. In open-ended natural language generation, the model generates multi-token words autoregressively
   (e.g., 'Optimism' is 3 tokens, 'Sadness' is 2 tokens in Qwen2.5). Post-hoc logit subtraction is
   mathematically impossible because there are no static K logits over closed options.
3. If CKD induces genuinely superior internal semantic representations during training, the CKD student
   will succeed in open generation and rescue minority classes (e.g. 'Optimism' in TweetEval Emotion),
   whereas Standard KD will suffer from minority collapse.

Execution Protocol:
- Strictly ONE model loaded at a time.
- GPU VRAM released completely between phases.
- Results saved to outputs/results/generative_probing/qwen1_5b_generative_probing_summary.json.
"""

import os
import sys
import csv
import json
import random
import re
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "datasets" / "natural_benchmarks" / "tweeteval_emotion"
OUTPUT_DIR = ROOT / "outputs" / "results" / "generative_probing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

CONFIG = {
    "num_classes": 4,
    "class_names": ["Anger", "Joy", "Optimism", "Sadness"],
    "verbalizers": ["A", "B", "C", "D"],
    "task_desc": (
        "Classify the tweet into one of four emotions:\n"
        "A: Anger\n"
        "B: Joy\n"
        "C: Optimism\n"
        "D: Sadness"
    )
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def load_data():
    train_file = DATA_PATH / "train_1200.csv"
    test_file = DATA_PATH / "test_400.csv"

    train_texts, train_golds = [], []
    with open(train_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_texts.append(row["text"])
            train_golds.append(int(row["label"]))

    test_texts, test_golds = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_texts.append(row["text"])
            test_golds.append(int(row["label"]))

    return train_texts, train_golds, test_texts, test_golds

def format_teacher_prompt(text):
    return (
        f"{CONFIG['task_desc']}\n\n"
        f"Text: {text.strip()[:600]}\n"
        "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
    )

def extract_teacher_probabilities(train_texts, test_texts):
    cache_path = DATA_PATH / "gemma4b_teacher_targets.npz"
    if cache_path.exists():
        print(f"[+] Loading cached teacher probabilities from: {cache_path}")
        data = np.load(cache_path)
        return data["raw_train_probs"], data["cal_train_probs"], data["test_logits"], data["null_logits"]

    print("\n" + "="*70)
    print("STAGE 1: EXTRACTING TEACHER PROBABILITIES (Gemma-3-4B-IT, BF16)")
    print("="*70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in CONFIG["verbalizers"]]

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True
    )
    model.eval()

    null_prompt = format_teacher_prompt("N/A")
    inp_null = tokenizer(null_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        null_logits = model(**inp_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()
    print("Teacher Null Probe Logits:", null_logits)

    def extract_logits(texts_list, desc=""):
        prompts = [format_teacher_prompt(t) for t in texts_list]
        res = []
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                b = prompts[i : i + batch_size]
                inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                out = model(**inp)
                l = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
                res.append(l)
                del inp, out
                if (i // batch_size) % 25 == 0:
                    print(f"  {desc}: {min(i + batch_size, len(prompts))}/{len(prompts)}...", flush=True)
        return np.concatenate(res, axis=0)

    print("Extracting test logits...", flush=True)
    test_logits = extract_logits(test_texts, "Test")

    print("Extracting train logits...", flush=True)
    train_logits = extract_logits(train_texts, "Train")

    raw_train_probs = softmax(train_logits, axis=-1)
    cal_train_probs = softmax(train_logits - null_logits, axis=-1)

    np.savez(
        cache_path,
        raw_train_probs=raw_train_probs,
        cal_train_probs=cal_train_probs,
        test_logits=test_logits,
        null_logits=null_logits
    )
    print(f"[✓] Teacher targets cached to: {cache_path}")

    print("\nUnloading Teacher Model from GPU...", flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("[✓] Teacher successfully unloaded. VRAM freed.")

    return raw_train_probs, cal_train_probs, test_logits, null_logits

def format_student_mc_prompt(text):
    return (
        f"<|im_start|>system\nYou are a precise text classification assistant.<|im_end|>\n"
        f"<|im_start|>user\n{CONFIG['task_desc']}\n\nText: {text.strip()[:600]}\n"
        f"Respond with ONLY the single letter corresponding to the correct category.<|im_end|>\n"
        f"<|im_start|>assistant\nAnswer: "
    )

def format_student_gen_prompt(text):
    return (
        f"<|im_start|>system\nYou are a concise text classification assistant. Respond with the emotion word only (Anger, Joy, Optimism, or Sadness).<|im_end|>\n"
        f"<|im_start|>user\nClassify the tweet into one of four emotions: Anger, Joy, Optimism, Sadness.\n\n"
        f"Text: {text.strip()[:600]}\n\n"
        f"Emotion:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

class StudentDistillDataset(Dataset):
    def __init__(self, texts, target_probs, prompt_formatter):
        self.samples = [(prompt_formatter(t), p) for t, p in zip(texts, target_probs)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def evaluate_student(model, tokenizer, test_texts, test_golds, device):
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in CONFIG["verbalizers"]]
    # verbalizer_ids are [32, 33, 34, 35] for 'A', 'B', 'C', 'D'

    # 1. Closed-Choice Multiple-Choice Evaluation
    mc_raw_logits = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            b = test_texts[i : i + batch_size]
            prompts = [format_student_mc_prompt(t) for t in b]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model(**enc)
            l = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            mc_raw_logits.append(l)
    mc_raw_logits = np.concatenate(mc_raw_logits, axis=0)
    mc_preds = np.argmax(mc_raw_logits, axis=-1)

    mc_acc = float(accuracy_score(test_golds, mc_preds) * 100)
    mc_f1 = float(f1_score(test_golds, mc_preds, average="macro") * 100)
    mc_per_class = (f1_score(test_golds, mc_preds, average=None) * 100).tolist()
    mc_counts = np.bincount(mc_preds, minlength=CONFIG["num_classes"]).tolist()

    # 1b. Post-Hoc Calibration on Multiple-Choice
    # Using null prompt on student
    null_prompt = format_student_mc_prompt("N/A")
    enc_null = tokenizer(null_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        student_null_logits = model(**enc_null).logits[0, -1, verbalizer_ids].float().cpu().numpy()
    posthoc_logits = mc_raw_logits - student_null_logits
    posthoc_preds = np.argmax(posthoc_logits, axis=-1)
    posthoc_acc = float(accuracy_score(test_golds, posthoc_preds) * 100)
    posthoc_f1 = float(f1_score(test_golds, posthoc_preds, average="macro") * 100)
    posthoc_per_class = (f1_score(test_golds, posthoc_preds, average=None) * 100).tolist()
    posthoc_counts = np.bincount(posthoc_preds, minlength=CONFIG["num_classes"]).tolist()

    # 2. Generative Probing (Free-Form Category Generation, Post-Hoc Inapplicable)
    gen_preds = []
    raw_generations = []
    print("  Running autoregressive generation on 400 test samples...", flush=True)
    for i in range(0, len(test_texts), batch_size):
        b = test_texts[i : i + batch_size]
        prompts = [format_student_gen_prompt(t) for t in b]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        input_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen_out[:, input_len:], skip_special_tokens=True)

        for text_out in decoded:
            raw_generations.append(text_out.strip())
            lowered = text_out.lower()
            matched_idx = -1
            for idx, cname in enumerate(CONFIG["class_names"]):
                if re.search(r'\b' + re.escape(cname.lower()) + r'\b', lowered):
                    matched_idx = idx
                    break
            if matched_idx == -1:
                for idx, cname in enumerate(CONFIG["class_names"]):
                    if cname.lower() in lowered:
                        matched_idx = idx
                        break
            gen_preds.append(matched_idx)

    # If any token is unmatched, mark as 0 (for standard metric computation) but track unmatched count
    unmatched = sum(1 for p in gen_preds if p == -1)
    eval_preds = [p if p != -1 else 0 for p in gen_preds]
    gen_acc = float(accuracy_score(test_golds, eval_preds) * 100)
    gen_f1 = float(f1_score(test_golds, eval_preds, average="macro") * 100)
    gen_per_class = (f1_score(test_golds, eval_preds, average=None) * 100).tolist()
    valid_counts = [p for p in gen_preds if p != -1]
    gen_counts = np.bincount(valid_counts, minlength=CONFIG["num_classes"]).tolist() if valid_counts else [0]*CONFIG["num_classes"]

    return {
        "multiple_choice_raw": {
            "accuracy": mc_acc,
            "macro_f1": mc_f1,
            "per_class_f1": [round(x, 2) for x in mc_per_class],
            "counts": mc_counts
        },
        "multiple_choice_posthoc": {
            "accuracy": posthoc_acc,
            "macro_f1": posthoc_f1,
            "per_class_f1": [round(x, 2) for x in posthoc_per_class],
            "counts": posthoc_counts
        },
        "generative_probing": {
            "accuracy": gen_acc,
            "macro_f1": gen_f1,
            "per_class_f1": [round(x, 2) for x in gen_per_class],
            "counts": gen_counts,
            "unmatched_count": unmatched,
            "sample_generations": raw_generations[:8]
        }
    }

def train_and_eval_student(method_name, train_texts, target_probs, test_texts, test_golds, seed=42):
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*70)
    print(f"TRAINING STUDENT: {method_name} (Qwen2.5-1.5B-Instruct + LoRA)")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in CONFIG["verbalizers"]]

    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0}
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    def collate_fn(batch):
        prompts = [b[0] for b in batch]
        probs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc["target_probs"] = probs
        return enc

    train_dataset = StudentDistillDataset(train_texts, target_probs, format_student_mc_prompt)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    model.train()
    print(f"Starting LoRA training ({epochs} epochs, {total_steps} steps)...", flush=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_p = batch["target_probs"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            step_logits = outputs.logits[:, -1, verbalizer_ids]
            log_probs = torch.log_softmax(step_logits, dim=-1)

            loss = kl_loss_fn(log_probs, target_p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} | Avg KL Loss: {avg_loss:.4f}", flush=True)

    print("\nTraining complete. Evaluating Student...", flush=True)
    model.eval()
    results = evaluate_student(model, tokenizer, test_texts, test_golds, device)

    print(f"\n[{method_name}] Evaluation Results:")
    print(f"  MC Raw Macro-F1:        {results['multiple_choice_raw']['macro_f1']:.2f}% | Counts: {results['multiple_choice_raw']['counts']}")
    print(f"  MC Post-Hoc Macro-F1:   {results['multiple_choice_posthoc']['macro_f1']:.2f}% | Counts: {results['multiple_choice_posthoc']['counts']}")
    print(f"  Generative Macro-F1:    {results['generative_probing']['macro_f1']:.2f}% | Counts: {results['generative_probing']['counts']}")
    print(f"  Generative Per-Class:   {results['generative_probing']['per_class_f1']}")
    print(f"  Sample Generations:     {results['generative_probing']['sample_generations']}")

    print(f"\nUnloading {method_name} model from GPU...", flush=True)
    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[✓] {method_name} model unloaded. VRAM freed.")

    return results

def eval_base_student(test_texts, test_golds):
    print("\n" + "="*70)
    print("EVALUATING ZERO-SHOT BASE STUDENT (Qwen2.5-1.5B-Instruct, no distillation)")
    print("="*70)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0}
    )
    model.eval()
    results = evaluate_student(model, tokenizer, test_texts, test_golds, device)

    print(f"\n[Base_Student_ZeroShot] Results:")
    print(f"  MC Raw Macro-F1:     {results['multiple_choice_raw']['macro_f1']:.2f}% | Counts: {results['multiple_choice_raw']['counts']}")
    print(f"  Generative Macro-F1: {results['generative_probing']['macro_f1']:.2f}% | Counts: {results['generative_probing']['counts']}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("[✓] Base student unloaded. VRAM freed.")
    return results

def main():
    print("="*70)
    print("STARTING GENERATIVE PROBING ABLATION STUDY")
    print(f"Teacher: Gemma-3-4B-IT (BF16) | Student: {STUDENT_MODEL_ID} (BF16 + LoRA)")
    print("="*70)

    train_texts, train_golds, test_texts, test_golds = load_data()
    print(f"Dataset: TweetEval Emotion (4 classes: Anger, Joy, Optimism, Sadness)")
    print(f"Splits: Train N={len(train_texts)}, Test N={len(test_texts)}")

    # Stage 1: Extract Teacher Probabilities (cached if already done)
    raw_probs, cal_probs, test_logits, null_logits = extract_teacher_probabilities(train_texts, test_texts)

    # Stage 2: Evaluate Base Student Zero-Shot (baseline)
    base_results = eval_base_student(test_texts, test_golds)

    # Stage 3: Train and Evaluate Standard KD Student
    std_kd_results = train_and_eval_student(
        "Standard_KD",
        train_texts,
        raw_probs,
        test_texts,
        test_golds
    )

    # Stage 4: Train and Evaluate CKD Student
    ckd_results = train_and_eval_student(
        "CKD",
        train_texts,
        cal_probs,
        test_texts,
        test_golds
    )

    summary = {
        "dataset": "tweeteval_emotion",
        "student_model": STUDENT_MODEL_ID,
        "base_student_zeroshot": base_results,
        "standard_kd": std_kd_results,
        "ckd": ckd_results,
        "comparisons": {
            "generative_macro_f1": {
                "base_zeroshot": base_results["generative_probing"]["macro_f1"],
                "standard_kd": std_kd_results["generative_probing"]["macro_f1"],
                "ckd": ckd_results["generative_probing"]["macro_f1"],
                "delta_ckd_vs_std_kd": ckd_results["generative_probing"]["macro_f1"] - std_kd_results["generative_probing"]["macro_f1"]
            },
            "mc_raw_macro_f1": {
                "standard_kd": std_kd_results["multiple_choice_raw"]["macro_f1"],
                "ckd": ckd_results["multiple_choice_raw"]["macro_f1"],
                "delta": ckd_results["multiple_choice_raw"]["macro_f1"] - std_kd_results["multiple_choice_raw"]["macro_f1"]
            },
            "mc_posthoc_macro_f1": {
                "standard_kd": std_kd_results["multiple_choice_posthoc"]["macro_f1"],
                "ckd": ckd_results["multiple_choice_posthoc"]["macro_f1"],
                "delta": ckd_results["multiple_choice_posthoc"]["macro_f1"] - std_kd_results["multiple_choice_posthoc"]["macro_f1"]
            }
        }
    }

    out_file = OUTPUT_DIR / "qwen1_5b_generative_probing_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("STUDY COMPLETE! Summary:")
    print(f"  Base Student Zero-Shot Generative Macro-F1: {base_results['generative_probing']['macro_f1']:.2f}%")
    print(f"  Standard KD Student Generative Macro-F1:    {std_kd_results['generative_probing']['macro_f1']:.2f}%")
    print(f"  CKD Student Generative Macro-F1:            {ckd_results['generative_probing']['macro_f1']:.2f}%")
    print(f"  --> Generative Representation Advantage:   {summary['comparisons']['generative_macro_f1']['delta_ckd_vs_std_kd']:+.2f} pp")
    print(f"  Output saved to: {out_file}")
    print("="*70)

if __name__ == "__main__":
    main()
