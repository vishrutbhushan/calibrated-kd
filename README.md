# Calibrated Knowledge Distillation (CKD)
<img width="7075" height="4407" alt="lol" src="https://github.com/user-attachments/assets/98d72fd4-d058-419c-acaf-b79ffa128b9f" />


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
    │   └── prepare_3_additional_benchmarks.py
    ├── main_experiments/               # Step 2: Section 6 Main 6 Benchmarks (Gemma-3-4B -> RoBERTa-base)
    │   ├── run_main_experiments.py     # Unified sequential runner for all 6 main benchmarks
    │   ├── run_single_benchmark.py
    │   └── run_full_student_questions_benchmark.py
    ├── ablations/                      # Step 3: Section 11 Ablations (Main 6 Benchmarks Only)
    │   ├── run_all_ablations.py                    # Unified sequential runner for all ablations
    │   ├── ablation1_cross_architecture.py         # Ablation 1: Same-size, Cross-family & Small SLM
    │   ├── run_multitask_distillation.py           # Ablation 2: Multi-task unified student backbone
    │   ├── run_all_multitask_experiments.py        # Ablation 2: Multi-task suite wrapper
    │   ├── ablation3_hard_vs_soft.py               # Ablation 3: Hard argmax vs. Soft KL distillation
    │   ├── ablation4_permutation_kd.py             # Ablation 4: Permutation-averaged KD vs. CKD
    │   └── ablation5_multi_seed.py                 # Ablation 5: Multi-seed statistical validation
    ├── plotting/                       # Step 4: Publication figures and table graphics
    │   ├── generate_benchmark_barchart.py
    │   ├── generate_blog_table_image.py
    │   └── plot_3_key_class_distributions.py
    └── blog_plotting/                  # Blog figures with 1-line summary captions
        ├── blog_table_image.py
        ├── blog_benchmark_barchart.py
        └── blog_class_distributions.py
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
* **Ablation Models:** `Qwen/Qwen3-4B`, `Qwen/Qwen2.5-0.5B-Instruct`

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

---

## Step-by-Step Replication Guide

### Step 1: Download & Prepare Datasets

Run the data preparation scripts to create the stratified 1,200 Train / 400 Test natural splits under `data/datasets/natural_benchmarks/`:

```bash
python scripts/data/prepare_natural_benchmarks.py
python scripts/data/prepare_3_new_benchmarks.py
python scripts/data/prepare_3_additional_benchmarks.py
```

### Step 2: Run Main Benchmarks (Section 6)

Run all 6 main benchmarks strictly sequentially in a single command. The script automatically verifies (and downloads if missing) all 6 datasets from HuggingFace, runs Teacher Zero-Shot (Raw & Null-Calibrated) alongside Standard KD, Batch-Calibrated KD, and Null Calibrated KD (Ours), logs everything to a single timestamped file in `logs/`, and saves all results to `outputs/main_experiments/`.

```bash
python scripts/main_experiments/run_main_experiments.py
```

Results are saved to `outputs/main_experiments/<dataset>_results.json` and `outputs/main_experiments/all_6_main_benchmarks_summary.json`.

### Step 3: Run Ablation Studies (Section 11)

Run all ablation experiments (Ablations 1–5, strictly on the Main 6 benchmarks) sequentially in a single command (logging to `logs/ablations_<timestamp>.log`):

```bash
python scripts/ablations/run_all_ablations.py
```

Or run individual ablations standalone:

**Ablation 1: Cross-Architecture Triad**
```bash
python scripts/ablations/ablation1_cross_architecture.py
```

**Ablation 2: Multi-Task Distillation (Unified Student Backbone)**
```bash
python scripts/ablations/run_all_multitask_experiments.py
python scripts/ablations/run_multitask_distillation.py --tasks tweeteval_emotion bbc_news app_reviews twitter_emotion trec6 --combo_id 5
```

**Ablation 3: Hard Pseudo-Labels vs. Soft Label Distillation**
```bash
python scripts/ablations/ablation3_hard_vs_soft.py
```

**Ablation 4: Permutation-Averaged Teacher Distillation vs. CKD**
```bash
python scripts/ablations/ablation4_permutation_kd.py
```

**Ablation 5: Multi-Seed Statistical Significance (Seeds 42, 43, 44)**
```bash
python scripts/ablations/ablation5_multi_seed.py
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
