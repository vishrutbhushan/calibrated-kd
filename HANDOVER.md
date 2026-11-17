# 📋 PROJECT HANDOVER & CONTEXT SPECIFICATION

> **Target Audience:** Any incoming AI agent or researcher working in `/home/vishrut/calibrated-kd`.  
> **Last Updated:** November 10, 2026  
> **Core Theme:** Null-Conditioned Verbalizer Calibration for Label-Free Knowledge Distillation (CKD)  
> **Status:** Fully Audited, Multi-Seed Validated, Camera-Ready 8-Page Conference Paper Completed  

---

## 1. Executive Summary & Core Methodology

This repository contains the complete empirical and mathematical investigation into **Null-Conditioned Verbalizer Calibration in Label-Free Knowledge Distillation (CKD)**.

### Core Problem
Autoregressive LLM teachers exhibit severe **null-conditioned verbalizer prior biases** ($W_{\text{null}} = z_T(x_{\text{null}})$) across candidate classification tokens even on content-free prompts (e.g., `"Text: N/A\nAnswer:"`). Under forward Kullback–Leibler (KL) distillation:
$$\mathcal{L}_{\text{KD}} = \mathbb{E}_{x \sim \mathcal{D}}\left[ \mathcal{D}_{\text{KL}}\!\left(P_T(y \mid x) \,\|\, P_S(y \mid x)\right) \right]$$
This unigram prior transfers directly into the supervisory soft targets, causing:
1. **Severe class starvation & minority collapse** (e.g., RoBERTa predicting the 10% minority Sci/Tech class 0 times out of 1,000).
2. **Artificial degradation of Macro-F1 and Accuracy** across both encoder and causal student architectures.

### The Solution: Calibrated Knowledge Distillation (CKD)
We extract the teacher's response across candidate verbalizers on a single content-free prompt as $W_{\text{null}}$, and calibrate the teacher logits offline before computing supervisory soft targets:
$$\tilde{z}(x) = z_T(x) - W_{\text{null}}$$
$$P_{\text{cal}}(y \mid x) = \text{Softmax}(\tilde{z}(x))$$
$$\mathcal{L}_{\text{CKD}} = \mathcal{D}_{\text{KL}}\!\left(P_{\text{cal}}(y \mid x) \,\|\, P_S(y \mid x)\right)$$

* Requires **zero human labels** and exactly **one null teacher evaluation** per task ($O(1)$ task overhead).
* Deployed student operates with standard inference, incurring **zero test-time latency overhead**.

---

## 2. Directory Layout & Organization

The repository has been rigorously partitioned into inputs, primary outputs, documentation, and archival history:

```
/home/vishrut/calibrated-kd/
├── README.md                          # Repository guide & comprehensive benchmark summary
├── HANDOVER.md                        # THIS FILE (Read upon starting new chat)
├── data/
│   ├── benchmarks/                    # Clean dataset splits (Emotion, AG News, TREC-6, SST-2)
│   │   └── skewed/                    # Non-balanced splits (AG News 10% min, Emotion natural, etc.)
│   └── teacher_targets/               # Cached teacher logits & null vectors (.npz)
│       └── skewed/                    # Cached teacher logits for non-balanced splits
├── docs/
│   ├── README.md                      # Directory guide separating active docs from archives
│   ├── master_research_report.pdf     # Primary Camera-Ready Paper (8-page conference format)
│   ├── master_research_report.md      # Up-to-date Markdown companion report (10 verified tables)
│   ├── master_research_report.docx    # Formatted Word Document
│   ├── master_research_report.html    # Standalone HTML report with tables & styling
│   └── archive/                       # Archived historical drafts & exploratory plans
├── latex/
│   ├── main.tex                       # Master LaTeX source for publication paper
│   ├── main.pdf                       # Compiled camera-ready PDF
│   ├── build.sh                       # Clean PDF build script with asset mirroring
│   └── figures/                       # Publication vector figures (fig1 to fig6)
├── figures/
│   ├── publication/                   # Publication-grade figures (PDF + 300 DPI PNG)
│   └── archive/                       # Historical & per-dataset exploratory plots
├── results/
│   ├── 1_balanced_benchmarks/         # Master results on balanced splits (RoBERTa, Qwen05, Qwen3)
│   ├── 2_skewed_benchmarks/           # 5-method comparison on non-balanced splits
│   ├── 3_cross_architecture_scaling/  # Capacity scaling across 125M, 500M, and 4B models
│   ├── 4_oracle_validation/           # Empirical oracle (b*) validation & alignment metrics
│   ├── 5_calibration_strength_beta_sweep/ # Sensitivity sweep across beta in {0.0, 0.5, 1.0, 1.5}
│   ├── 6_multi_seed_validation/       # Multi-seed validation (seeds 42, 43, 44 Mean +- Std)
│   └── raw_runs/                      # Organized raw per-task output files & metrics
└── scripts/
    ├── prepare_data/                  # Split extraction & diagnostic null-bias checkers
    ├── cache_targets/                 # Teacher inference & null-vector calibration caching
    ├── train/                         # Standalone student distillation runners
    ├── analysis/                      # Empirical oracle fitting & statistical evaluation
    ├── plotting/                      # Merged publication figure generators (figs 1–6)
    └── export/                        # Result organization, auditing & doc converters
```

---

## 3. Validated Experimental Findings (Audited & Verified)

All 24+ numerical metrics in the paper and documentation are 100% audited and verified by `scripts/export/organize_and_validate_all_results.py`.

### 1. Balanced Benchmarks (Macro-F1 %)
* **Qwen3-4B (4B Peer Causal):** Emotion **37.59 $\to$ 39.03 (+1.44 pp)**, AG News **69.91 $\to$ 74.40 (+4.49 pp)**, TREC-6 **75.08 $\to$ 77.12 (+2.04 pp)**, SST-2 **80.92 $\to$ 80.92 (0.00 pp)**.
* **Qwen2.5-0.5B (500M Causal):** Emotion **30.80 $\to$ 32.51 (+1.71 pp)**, AG News **60.53 $\to$ 62.93 (+2.40 pp)**, TREC-6 **59.70 $\to$ 62.98 (+3.28 pp)**, SST-2 **82.66 $\to$ 82.66 (0.00 pp)**.
* **RoBERTa-base (125M Encoder):** Emotion **36.38 $\to$ 39.63 (+3.25 pp)**, AG News **68.78 $\to$ 70.23 (+1.45 pp)**, TREC-6 **68.36 $\to$ 69.43 (+1.07 pp)**, SST-2 **87.90 $\to$ 87.41 (-0.49 pp)**.

### 2. Non-Balanced Benchmarks on RoBERTa-base
* **AG News Skewed (10% Sci/Tech minority):** Standard KD = 57.22% (0 predictions), Batch Cal = 57.15% (0 predictions), Post-Hoc = 65.53% (165 predictions), **CKD = 64.22% (29 predictions, +7.00 pp)**.
* **Emotion Natural (Saravia):** Standard KD = 35.02%, Batch Cal = 41.29%, **CKD = 44.58% (+9.56 pp)**, Post-Hoc = 44.60%.
* **TREC-6 Natural (1.6% Abbr):** Standard KD = 64.62%, Batch Cal = 70.15%, **CKD = 75.09% (+10.47 pp)**, Post-Hoc = 68.91%.
* **SST-2 Skewed (80/20 Imbalance):** Standard KD = 65.03%, Post-Hoc = 62.97%, **CKD = 78.69% (+13.66 pp)**, Oracle Prior = 84.36%.

### 3. Empirical Oracle Validation ($b^*$)
Additive correction vector $b^* \in \mathbb{R}^K$ fitted via L-BFGS cross-entropy minimization on 100 held-out labeled samples:
* AG News: Centered Cosine Sim = **0.3367**, Pearson $r$ = **0.3367** (World suppressed: $-45.75 \leftrightarrow -2.95$)
* Emotion: Centered Cosine Sim = **0.3953**, Pearson $r$ = **0.3953** (Sadness suppressed: $-16.38 \leftrightarrow -3.11$)
* TREC-6: Centered Cosine Sim = **0.4592**, Pearson $r$ = **0.4592** (Description suppressed: $-15.62 \leftrightarrow -3.52$)

### 4. Calibration Strength ($\beta$) Sensitivity Sweep
$\tilde{z}(x) = z_T(x) - \beta W_{\text{null}}$ on RoBERTa-base AG News Skewed:
* $\beta = 0.0$ (Std KD): 57.22% F1, 0 Sci/Tech (Collapsed)
* $\beta = 0.5$ (Under-calibrated): 57.50% F1, 0 Sci/Tech (Collapsed)
* $\beta = 1.0$ (CKD): **64.22% F1, 29 Sci/Tech (Optimal)**
* $\beta = 1.5$ (Over-calibrated): 48.93% F1, 307 Sci/Tech (Boundary distortion, Acc drops to 55.20%)

### 5. Multi-Seed Statistical Significance (Seeds 42, 43, 44)
* Standard KD: Macro-F1 = $56.62 \pm 0.46\%$, Accuracy = $65.03 \pm 0.58\%$, Sci/Tech = $0.0 \pm 0.0$
* Calibrated KD: Macro-F1 = **$64.82 \pm 0.54\%$**, Accuracy = **$69.83 \pm 0.09\%$**, Sci/Tech = **$36.7 \pm 6.5$**
* **Net Statistically Significant Gain:** **+8.19 pp Macro-F1**, **+4.80 pp Accuracy**, $+36.7$ minority predictions ($p < 0.001$, $>15\sigma$ above seed variance).

---

## 4. Key Operational Instructions

* **Python Virtualenv:** Always use `.venv/bin/python` (or `source .venv/bin/activate` in `/home/vishrut/calibrated-kd`).
* **LaTeX Compilation:** Always compile via `bash latex/build.sh`, which automatically updates `latex/main.pdf`, copies it to `docs/master_research_report.pdf`, and mirrors to the artifact directory.
* **Audit & Validation:** Run `python scripts/export/organize_and_validate_all_results.py` to confirm that all 24+ numerical assertions pass with zero discrepancy.
* **Docx & HTML Synchronizer:** Run `python scripts/export/convert_report_to_docs.py` to re-sync `docs/master_research_report.docx` and `docs/master_research_report.html` whenever `docs/master_research_report.md` changes.
