"""
run_all_ablations.py
--------------------
Unified, strictly sequential runner for all Ablation Studies (Section 11 of
paper_structure.md):
  - Ablation 1: Cross-Architecture Triad (Same-size, Cross-family, Small SLM)
  - Ablation 2: Multi-Task Distillation (Combos 1-4 & 5-Task Unified)
  - Ablation 3: Hard Pseudo-Labels vs. Soft Label Distillation
  - Ablation 4: Permutation-Averaged Teacher Distillation vs. CKD
  - Ablation 5: Multi-Seed Statistical Validation (Seeds 42, 43, 44)
  - Ablation 6: Teacher Model Scaling (Gemma-3-12B-IT -> RoBERTa-base)

Each experiment runs in an isolated subprocess so GPU VRAM is 100% freed
between runs. All stdout/stderr is streamed to logs/ablations_<timestamp>.log.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"

ABLATION_STEPS = [
    (
        "Ablation 1a: Same-Size Causal (Gemma-4B -> Qwen3-4B on AG News Skewed)",
        ["scripts/ablations/train_skewed_causal.py", "--model", "qwen3", "--dataset", "agnews"],
    ),
    (
        "Ablation 1b: Cross-Family (Qwen3-4B -> RoBERTa-base on SST-2 Skewed)",
        ["scripts/ablations/train_cross_family_qwen_roberta_sst2_skewed.py"],
    ),
    (
        "Ablation 1c: Small SLM (Gemma-4B -> Qwen2.5-0.5B on TREC-6 Skewed)",
        ["scripts/ablations/train_skewed_causal.py", "--model", "qwen05", "--dataset", "trec"],
    ),
    (
        "Ablation 2a: Multi-Task Distillation (Combos 1-4)",
        ["scripts/ablations/run_all_multitask_experiments.py"],
    ),
    (
        "Ablation 2b: Multi-Task Distillation (5-Task Unified Suite)",
        [
            "scripts/ablations/run_multitask_distillation.py",
            "--tasks",
            "tweeteval_emotion",
            "bbc_news",
            "praxis",
            "twitter_emotion",
            "trec6",
            "--combo_id",
            "5",
        ],
    ),
    (
        "Ablation 3: Hard Pseudo-Labels vs. Soft Label Distillation",
        ["scripts/ablations/train_hard_pseudo_label_experiment.py"],
    ),
    (
        "Ablation 4: Permutation-Averaged Teacher Distillation vs. CKD (AG News Skewed)",
        ["scripts/ablations/train_permutation_kd_experiment.py"],
    ),
    (
        "Ablation 5: Multi-Seed Statistical Significance & Beta Sweep",
        ["scripts/ablations/train_roberta_audit_experiments.py"],
    ),
    (
        "Ablation 6: Teacher Model Scaling (Gemma-3-12B -> RoBERTa-base)",
        ["scripts/ablations/train_students_gemma12b_sequential.py"],
    ),
]


def main() -> None:
    """Runs all ablation steps strictly sequentially and logs to a timestamped file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"ablations_{timestamp}.log"

    total_start = time.time()
    print(f"Starting sequential ablation suite. Log: {log_path}", flush=True)

    with open(log_path, "w", encoding="utf-8") as log_fp:
        for idx, (label, cmd_args) in enumerate(ABLATION_STEPS, 1):
            banner = f"\n{'=' * 75}\n[{idx}/{len(ABLATION_STEPS)}] {label}\n{'=' * 75}\n"
            print(banner, end="", flush=True)
            log_fp.write(banner)
            log_fp.flush()

            step_start = time.time()
            proc = subprocess.Popen(
                [sys.executable] + cmd_args,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fp.write(line)
                log_fp.flush()

            ret = proc.wait()
            elapsed = time.time() - step_start
            if ret != 0:
                msg = f"\n[ERROR] Step {idx} failed with exit code {ret} after {elapsed:.1f}s!\n"
                print(msg, flush=True)
                log_fp.write(msg)
                sys.exit(ret)

            done_msg = f"[✓] Completed Step {idx}/{len(ABLATION_STEPS)} in {elapsed:.1f}s\n"
            print(done_msg, flush=True)
            log_fp.write(done_msg)
            log_fp.flush()

    total_min = (time.time() - total_start) / 60.0
    print(f"\n[✓] ALL {len(ABLATION_STEPS)} ABLATION STEPS COMPLETED IN {total_min:.2f} MINUTES!")


if __name__ == "__main__":
    main()
