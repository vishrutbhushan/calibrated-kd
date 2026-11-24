# Calibrated Knowledge Distillation (CKD)

**Preventing Prompt-Bias Transfer and Minority Class Collapse in LLM-to-SLM Distillation via Offline Null Probing.**

Autoregressive LLM teachers carry strong static verbalizer and position biases from pre-training. When distilling into compact student encoders (such as `roberta-base`) using multiple-choice prompt verbalizers (`A, B, C, ...`), this static bias dominates soft probability targets, starving minority classes of gradient signal and causing deterministic class collapse.

**Calibrated Knowledge Distillation (CKD)** probes the frozen teacher *once* offline with an uninformative null input (`"N/A"`) to measure its static logit bias $b_{\text{null}} = z_T(\text{"N/A"})$, and subtracts it directly before softmax inside the distillation objective:

$$p_T^{\text{cal}}(x) = \text{softmax}\left(\frac{z_T(x) - b_{\text{null}}}{\tau}\right), \quad \mathcal{L}_{\text{CKD}} = \tau^2 \, \text{KL}\big(p_T^{\text{cal}}(x) \parallel p_S(x)\big)$$

CKD requires **zero ground-truth labels**, **one extra teacher forward pass total ($O(1)$)**, and imposes **zero inference-time overhead** on the deployed student model.

---

## Repository Structure

All scripts correspond directly to the experiments and sections specified in [`paper_structure.md`](paper_structure.md):

```text
calibrated-kd/
├── paper_structure.md                  # Full paper specification, methodology, and results tables
├── README.md                           # Replication guide & documentation
├── latex/                              # LaTeX paper source & build scripts
│   ├── main.tex
│   ├── references.bib
│   ├── acl.sty
│   ├── acl_natbib.bst
│   ├── build.sh
│   └── figures/
├── data/                               # Prepared dataset splits & cached teacher targets
└── scripts/
    ├── data/                           # Step 1: Dataset download & stratified splitting
    │   ├── prepare_natural_benchmarks.py
    │   ├── prepare_3_new_benchmarks.py
    │   ├── prepare_3_additional_benchmarks.py
    │   └── prepare_sst2_natural.py
    ├── main_experiments/               # Step 2: Section 6 Main 6 Benchmarks (Gemma-3-4B -> RoBERTa-base)
    │   ├── run_main_experiments.py     # Unified sequential runner for all 6 main benchmarks
    │   ├── run_single_benchmark.py
    │   └── run_full_student_questions_benchmark.py
    ├── ablations/                      # Step 3: Section 11 Ablations & Teacher Scaling
    │   ├── exp_combine_ckd_posthoc.py              # Post-Hoc & Combined CKD+Post-Hoc Ablation
    │   ├── train_skewed_causal.py                  # Ablation 1: Same-size (Gemma-4B -> Qwen3-4B)
    │   ├── train_cross_family_qwen_roberta_sst2.py # Ablation 1: Cross-family (Qwen3-4B -> RoBERTa, Balanced)
    │   ├── train_cross_family_qwen_roberta_sst2_skewed.py # Ablation 1: Cross-family (Skewed)
    │   ├── train_qwen05_benchmarks.py              # Ablation 1: Small SLM compression (Gemma-4B -> Qwen2.5-0.5B)
    │   ├── run_multitask_distillation.py           # Ablation 2: Multi-task unified student backbone
    │   ├── run_multitask_distillation_ablation.py  # Ablation 2: Multi-task core runner
    │   ├── run_all_multitask_experiments.py        # Ablation 2: Multi-task suite wrapper
    │   ├── train_hard_pseudo_label_experiment.py   # Ablation 3: Hard argmax vs. Soft KL distillation
    │   ├── train_permutation_kd_experiment.py      # Ablation 4: Permutation-averaged KD (AG News)
    │   ├── train_permutation_kd_sst2.py            # Ablation 4: Permutation-averaged KD (SST-2)
    │   ├── train_roberta_audit_experiments.py      # Ablation 5: Multi-seed statistical validation
    │   ├── extract_gemma12b_pytorch_targets.py     # Teacher Scaling: Gemma-3-12B target extraction
    │   └── train_students_gemma12b_sequential.py   # Teacher Scaling: Sequential RoBERTa KD on 12B targets
    └── plotting/                       # Step 4: Publication figures and table graphics
        ├── generate_benchmark_barchart.py
        ├── generate_blog_table_image.py
        └── plot_3_key_class_distributions.py
```

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate peft bitsandbytes scikit-learn numpy matplotlib
```

**Models Used:**
* **Primary Teacher:** `google/gemma-3-4b-it` (BF16)
* **Primary Student:** `roberta-base` (125M)
* **Ablation & Scaling Models:** `Qwen/Qwen3-4B`, `Qwen/Qwen2.5-0.5B-Instruct`, `google/gemma-3-12b-it` (4-bit NF4)

---

## Benchmark Datasets & Links

All main benchmarks use stratified random splits ($N_{\text{train}} = 1,200$, $N_{\text{test}} = 400$, `seed=42`) preserving authentic, naturally skewed class distributions:

| Benchmark | $K$ | Domain | HuggingFace Dataset | Link |
| :--- | :---: | :--- | :--- | :--- |
| **Student Questions** | 4 | Academic STEM Subjects | `SetFit/student-question-categories` | https://huggingface.co/datasets/SetFit/student-question-categories |
| **TREC-6** | 6 | Question Taxonomy | `CogComp/trec` | https://huggingface.co/datasets/CogComp/trec |
| **App Reviews** | 5 | Customer Star Ratings | `sealuzh/app_reviews` | https://huggingface.co/datasets/sealuzh/app_reviews |
| **Twitter Emotion** | 6 | Fine-Grained Social Affect | `dair-ai/emotion` | https://huggingface.co/datasets/dair-ai/emotion |
| **TweetEval Emotion** | 4 | Social Media Affect | `cardiffnlp/tweet_eval` (`emotion`) | https://huggingface.co/datasets/cardiffnlp/tweet_eval |
| **BBC News** | 5 | Multi-Topic Journalism | `SetFit/bbc-news` | https://huggingface.co/datasets/SetFit/bbc-news |

**Ablation & Diagnostic Datasets:**
* **AG News ($K=4$, Ablations 1, 3, 4, 5):** https://huggingface.co/datasets/fancyzhx/ag_news
* **SST-2 ($K=2$, Ablations 1 & 4):** https://huggingface.co/datasets/SetFit/sst2
* **PRAXIS ($K=5$, Multi-Task Ablation):** https://huggingface.co/datasets/au-hf/praxis

---

## Step-by-Step Replication Guide

### Step 1: Download & Prepare Datasets

Run the data preparation scripts to create the stratified 1,200 Train / 400 Test natural splits under `data/datasets/natural_benchmarks/`:

```bash
python scripts/data/prepare_natural_benchmarks.py
python scripts/data/prepare_3_new_benchmarks.py
python scripts/data/prepare_3_additional_benchmarks.py
python scripts/data/prepare_sst2_natural.py
```

### Step 2: Run Main Benchmarks (Section 6)

Run all 6 main benchmarks strictly sequentially in a single command. The script automatically verifies (and downloads if missing) all 6 datasets from HuggingFace, runs Teacher Zero-Shot (Raw & Null-Calibrated) alongside Standard KD, Batch-Calibrated KD, and Null Calibrated KD (Ours), logs everything to a single timestamped file in `logs/`, and saves all results to `outputs/main_experiments/`.

```bash
python scripts/main_experiments/run_main_experiments.py
```

Results are saved to `outputs/main_experiments/<dataset>_results.json` and `outputs/main_experiments/all_6_main_benchmarks_summary.json`.

### Step 3: Run Ablation Studies (Section 11)

**Ablation 1: Cross-Architecture Triad**
```bash
# Config 1: Same-Size (Gemma-3-4B -> Qwen3-4B LoRA on AG News)
python scripts/ablations/train_skewed_causal.py

# Config 2: Cross-Family (Qwen3-4B -> RoBERTa-base on SST-2)
python scripts/ablations/train_cross_family_qwen_roberta_sst2.py
python scripts/ablations/train_cross_family_qwen_roberta_sst2_skewed.py

# Config 3: Small SLM Compression (Gemma-3-4B -> Qwen2.5-0.5B LoRA on TREC-6)
python scripts/ablations/train_qwen05_benchmarks.py
```

**Ablation 2: Multi-Task Distillation (Unified Student Backbone)**
```bash
python scripts/ablations/run_all_multitask_experiments.py
```

**Ablation 3: Hard Pseudo-Labels vs. Soft Label Distillation**
```bash
python scripts/ablations/train_hard_pseudo_label_experiment.py
```

**Ablation 4: Permutation-Averaged Teacher Distillation vs. CKD**
```bash
python scripts/ablations/train_permutation_kd_experiment.py
python scripts/ablations/train_permutation_kd_sst2.py
```

**Ablation 5: Multi-Seed Statistical Significance (Seeds 42, 43, 44)**
```bash
python scripts/ablations/train_roberta_audit_experiments.py
```

**Teacher Model Scaling (Gemma-3-12B-IT in 4-bit NF4)**
```bash
# Stage 1: Extract and cache 12B teacher targets sequentially
python scripts/ablations/extract_gemma12b_pytorch_targets.py

# Stage 2: Train RoBERTa-base students sequentially from cached targets
python scripts/ablations/train_students_gemma12b_sequential.py
```

**Post-Hoc Output Calibration & Combined (CKD + Post-Hoc) Ablation**
```bash
python scripts/ablations/exp_combine_ckd_posthoc.py
```

### Step 4: Generate Publication Figures & Tables

All 3 figures are generated automatically at the end of `run_main_experiments.py`, or can be run standalone:

```bash
python scripts/plotting/generate_blog_table_image.py
python scripts/plotting/generate_benchmark_barchart.py
python scripts/plotting/plot_3_key_class_distributions.py
```

---

## Main Benchmark Results Summary (Section 6)

Teacher: **Gemma-3-4B-IT** $\longrightarrow$ Student: **RoBERTa-base (125M)**. Reported as **Macro-F1 (%)** with **Accuracy (%)** in parentheses:

| Method | Student Questions ($K=4$) | TREC-6 ($K=6$) | App Reviews ($K=5$) | Twitter Emotion ($K=6$) | TweetEval Emotion ($K=4$) | BBC News ($K=5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher Raw Zero-Shot** | 28.13% (33.8%) | 50.15% (52.3%) | 29.69% (48.3%) | 39.80% (53.0%) | 45.78% (57.3%) | 54.81% (56.0%) |
| **Teacher Null-Calibrated** | 29.08% (33.8%) | 52.90% (54.3%) | 32.46% (49.3%) | 40.35% (52.0%) | 56.05% (65.0%) | 64.04% (63.3%) |
| **Student: Standard KD** | 22.86% (32.3%) | 62.28% (59.8%) | 30.25% (51.8%) | 29.78% (50.5%) | 34.16% (54.3%) | 55.56% (58.5%) |
| **Student: Batch-Cal KD** | 23.19% (32.0%) | 63.29% (61.0%) | 32.04% (52.3%) | 31.13% (50.0%) | 38.85% (56.5%) | 57.13% (60.3%) |
| **Student: Null Calibrated KD (Ours)** | **37.10% (39.3%)** | **64.71% (61.8%)** | **33.24% (53.3%)** | **34.46% (50.8%)** | **55.48% (66.5%)** | **71.40% (70.5%)** |
