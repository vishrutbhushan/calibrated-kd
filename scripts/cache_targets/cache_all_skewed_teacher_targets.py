"""
cache_all_skewed_teacher_targets.py
-----------------------------------
Extracts Teacher (google/gemma-3-4b-it BF16) logits on all 4 natural/skewed benchmarks:
  1) Emotion Natural (1,200 train items, 500 test items)
  2) TREC-6 Natural (1,200 train items, 500 test items)
  3) AG News Skewed (1,000 train items, 1,000 test items)
  4) SST-2 Skewed (1,000 train items, 200 test items)

Computes for each dataset:
  - raw_logits, raw_probs
  - null_logits (W_null)
  - ckd_logits, ckd_probs (z - W_null)
  - batch_cal_logits, batch_cal_probs (z - log(p_batch))
  - prior_aware_logits, prior_aware_probs (z - W_null + log P_prior(y))
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
SKEWED_DIR = BENCHMARK_DIR / "skewed"
OUT_TARGET_DIR = ROOT / "data" / "teacher_targets" / "skewed"
OUT_TARGET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

CONFIGS = {
    "emotion": {
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "class_names": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "prompt_fn": lambda t: (
            "Classify the emotional sentiment of the text into one of six categories:\n"
            "A: Sadness\nB: Joy\nC: Love\nD: Anger\nE: Fear\nF: Surprise\n\n"
            f"Text: {t.strip()[:600]}\n"
            "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
        ),
        "null_prompt": (
            "Classify the emotional sentiment of the text into one of six categories:\n"
            "A: Sadness\nB: Joy\nC: Love\nD: Anger\nE: Fear\nF: Surprise\n\n"
            "Text: N/A\n"
            "Respond with ONLY the single letter corresponding to the correct category.\nAnswer:"
        ),
        "train_csv": SKEWED_DIR / "emotion_natural_train_1200.csv",
        "test_csv": SKEWED_DIR / "emotion_natural_test_500.csv",
        "save_name": "teacher_targets_emotion_natural_1200_gemma3_4b.npz"
    },
    "trec": {
        "verbalizers": ["A", "B", "C", "D", "E", "F"],
        "class_names": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "prompt_fn": lambda t: (
            f"Question: {t.strip()[:600]}\n"
            "What type of answer is the question asking for?\nOptions:\n"
            "A. Abbreviation\nB. Entity\nC. Description\nD. Human\nE. Location\nF. Numeric\nAnswer:"
        ),
        "null_prompt": (
            "Question: N/A\n"
            "What type of answer is the question asking for?\nOptions:\n"
            "A. Abbreviation\nB. Entity\nC. Description\nD. Human\nE. Location\nF. Numeric\nAnswer:"
        ),
        "train_csv": SKEWED_DIR / "trec_natural_train_1200.csv",
        "test_csv": BENCHMARK_DIR / "trec_test_500.csv",
        "save_name": "teacher_targets_trec_natural_1200_gemma3_4b.npz"
    },
    "agnews": {
        "verbalizers": ["A", "B", "C", "D"],
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        "prompt_fn": lambda t: (
            "Classify the news article into one of four topics:\n"
            "A: World\nB: Sports\nC: Business\nD: Science and Technology\n\n"
            "Respond with ONLY the single letter corresponding to the correct category.\n\n"
            f"Article: {t.strip()[:1000]}\nAnswer:"
        ),
        "null_prompt": (
            "Classify the news article into one of four topics:\n"
            "A: World\nB: Sports\nC: Business\nD: Science and Technology\n\n"
            "Respond with ONLY the single letter corresponding to the correct category.\n\n"
            "Article: N/A\nAnswer:"
        ),
        "train_csv": SKEWED_DIR / "agnews_skewed_train_1000.csv",
        "test_csv": BENCHMARK_DIR / "ag_news_test_1000.csv",
        "save_name": "teacher_targets_agnews_skewed_1000_gemma3_4b.npz"
    },
    "sst2": {
        "verbalizers": ["A", "B"],
        "class_names": ["Negative", "Positive"],
        "prompt_fn": lambda t: (
            "Classify the sentiment of the following sentence into one of two categories:\n"
            "A: Negative\nB: Positive\n\n"
            "Respond with ONLY the single letter corresponding to the correct category.\n\n"
            f"Sentence: {t.strip()[:1000]}\nAnswer:"
        ),
        "null_prompt": (
            "Classify the sentiment of the following sentence into one of two categories:\n"
            "A: Negative\nB: Positive\n\n"
            "Respond with ONLY the single letter corresponding to the correct category.\n\n"
            "Sentence: N/A\nAnswer:"
        ),
        "train_csv": SKEWED_DIR / "sst2_skewed_train_1000.csv",
        "test_csv": BENCHMARK_DIR / "sst2_test_200.csv",
        "save_name": "teacher_targets_sst2_skewed_1000_gemma3_4b.npz"
    }
}

def extract_logits_batched(model, tokenizer, prompts, verbalizer_ids, batch_size=8):
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            b = prompts[i : i + batch_size]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            out = model(**inp)
            logits = out.logits[:, -1, verbalizer_ids].float().cpu().numpy()
            all_logits.append(logits)
            del inp, out
    return np.concatenate(all_logits, axis=0)

def main():
    print(f"Loading Teacher model from {MODEL_PATH} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True
    )
    model.eval()

    for key, cfg in CONFIGS.items():
        out_path = OUT_TARGET_DIR / cfg["save_name"]
        if out_path.exists():
            print(f"\n[✓] Skipping {key.upper()}: already cached at {out_path}", flush=True)
            continue
        print(f"\n{'='*70}\nProcessing benchmark: {key.upper()} (Natural / Skewed)\n{'='*70}", flush=True)
        verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in cfg["verbalizers"]]

        # 1. Null prompt
        null_inp = tokenizer([cfg["null_prompt"]], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            null_out = model(**null_inp)
            null_logits = null_out.logits[0, -1, verbalizer_ids].float().cpu().numpy()
        del null_inp, null_out
        print(f"Null Logits: {null_logits}", flush=True)

        # 2. Train set
        with open(cfg["train_csv"]) as f:
            tr_rows = list(csv.DictReader(f))
        tr_prompts = [cfg["prompt_fn"](r["text"]) for r in tr_rows]
        tr_golds = [int(r["label"]) for r in tr_rows]
        tr_sids = [r["sample_id"] for r in tr_rows]
        print(f"Extracting train logits ({len(tr_prompts)} items)...", flush=True)
        train_logits_raw = extract_logits_batched(model, tokenizer, tr_prompts, verbalizer_ids, batch_size=8)

        # 3. Test set
        with open(cfg["test_csv"]) as f:
            te_rows = list(csv.DictReader(f))
        te_prompts = [cfg["prompt_fn"](r["text"]) for r in te_rows]
        te_golds = [int(r["label"]) for r in te_rows]
        te_sids = [r.get("sample_id", f"test_{i}") for i, r in enumerate(te_rows)]
        print(f"Extracting test logits ({len(te_prompts)} items)...", flush=True)
        test_logits_raw = extract_logits_batched(model, tokenizer, te_prompts, verbalizer_ids, batch_size=8)

        # 4. Compute Targets
        # Raw probs
        train_probs_raw = np.exp(train_logits_raw - np.max(train_logits_raw, axis=-1, keepdims=True))
        train_probs_raw /= train_probs_raw.sum(axis=-1, keepdims=True)
        test_probs_raw = np.exp(test_logits_raw - np.max(test_logits_raw, axis=-1, keepdims=True))
        test_probs_raw /= test_probs_raw.sum(axis=-1, keepdims=True)

        # CKD (z - W_null)
        train_logits_ckd = train_logits_raw - null_logits
        train_probs_ckd = np.exp(train_logits_ckd - np.max(train_logits_ckd, axis=-1, keepdims=True))
        train_probs_ckd /= train_probs_ckd.sum(axis=-1, keepdims=True)
        test_logits_ckd = test_logits_raw - null_logits
        test_probs_ckd = np.exp(test_logits_ckd - np.max(test_logits_ckd, axis=-1, keepdims=True))
        test_probs_ckd /= test_probs_ckd.sum(axis=-1, keepdims=True)

        # Batch Cal (z - log(p_batch))
        mean_p_batch = train_probs_raw.mean(axis=0)
        log_mean_p = np.log(np.maximum(mean_p_batch, 1e-12))
        train_logits_batch = train_logits_raw - log_mean_p
        train_probs_batch = np.exp(train_logits_batch - np.max(train_logits_batch, axis=-1, keepdims=True))
        train_probs_batch /= train_probs_batch.sum(axis=-1, keepdims=True)
        test_logits_batch = test_logits_raw - log_mean_p
        test_probs_batch = np.exp(test_logits_batch - np.max(test_logits_batch, axis=-1, keepdims=True))
        test_probs_batch /= test_probs_batch.sum(axis=-1, keepdims=True)

        # Prior-Aware CKD (z - W_null + log P_prior(y))
        # Compute empirical training label prior
        class_counts = np.bincount(tr_golds, minlength=len(cfg["verbalizers"]))
        prior_p = class_counts / len(tr_golds)
        log_prior = np.log(np.maximum(prior_p, 1e-12))
        train_logits_prior = train_logits_raw - null_logits + log_prior
        train_probs_prior = np.exp(train_logits_prior - np.max(train_logits_prior, axis=-1, keepdims=True))
        train_probs_prior /= train_probs_prior.sum(axis=-1, keepdims=True)
        test_logits_prior = test_logits_raw - null_logits + log_prior
        test_probs_prior = np.exp(test_logits_prior - np.max(test_logits_prior, axis=-1, keepdims=True))
        test_probs_prior /= test_probs_prior.sum(axis=-1, keepdims=True)

        # Teacher Direct Inference Evaluation
        raw_preds = test_logits_raw.argmax(axis=-1)
        ckd_preds = test_logits_ckd.argmax(axis=-1)
        batch_preds = test_logits_batch.argmax(axis=-1)
        prior_preds = test_logits_prior.argmax(axis=-1)

        print(f"Teacher Raw:         Acc={accuracy_score(te_golds, raw_preds)*100:.2f}%, Macro-F1={f1_score(te_golds, raw_preds, average='macro')*100:.2f}%")
        print(f"Teacher CKD:         Acc={accuracy_score(te_golds, ckd_preds)*100:.2f}%, Macro-F1={f1_score(te_golds, ckd_preds, average='macro')*100:.2f}%")
        print(f"Teacher Batch Cal:   Acc={accuracy_score(te_golds, batch_preds)*100:.2f}%, Macro-F1={f1_score(te_golds, batch_preds, average='macro')*100:.2f}%")
        print(f"Teacher Prior-Aware: Acc={accuracy_score(te_golds, prior_preds)*100:.2f}%, Macro-F1={f1_score(te_golds, prior_preds, average='macro')*100:.2f}%")

        out_path = OUT_TARGET_DIR / cfg["save_name"]
        np.savez_compressed(
            out_path,
            train_sids=np.array(tr_sids),
            train_golds=np.array(tr_golds),
            train_logits_raw=train_logits_raw,
            train_probs_raw=train_probs_raw,
            train_logits_ckd=train_logits_ckd,
            train_probs_ckd=train_probs_ckd,
            train_logits_batch=train_logits_batch,
            train_probs_batch=train_probs_batch,
            train_logits_prior=train_logits_prior,
            train_probs_prior=train_probs_prior,
            test_sids=np.array(te_sids),
            test_golds=np.array(te_golds),
            test_logits_raw=test_logits_raw,
            test_probs_raw=test_probs_raw,
            test_logits_ckd=test_logits_ckd,
            test_probs_ckd=test_probs_ckd,
            test_logits_batch=test_logits_batch,
            test_probs_batch=test_probs_batch,
            test_logits_prior=test_logits_prior,
            test_probs_prior=test_probs_prior,
            null_logits=null_logits,
            mean_p_batch=mean_p_batch,
            prior_p=prior_p,
            verbalizers=cfg["verbalizers"],
            class_names=cfg["class_names"]
        )
        print(f"[✓] Saved targets to {out_path}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()
    print("\n[✓] All skewed teacher targets successfully cached!", flush=True)

if __name__ == "__main__":
    main()

