# Calibrated Knowledge Distillation (CKD)

**Official empirical repository for Calibrated Knowledge Distillation (CKD): Removing Prompt Verbalizer Prior Transfer in Label-Free Distillation.**

---

## 🔬 Core Premise & Methodology

When prompted with classification instructions and discrete verbalizers (e.g., `"A"`, `"B"`, `"C"`, `"D"`), Large Language Model teachers exhibit sharp **unigram token prior skews** ($W_{\text{null}} = z_T(x_{\text{null}})$) even on content-free inputs (`"Text: N/A\nAnswer:"`). 

When transferring knowledge to a smaller student model via standard Knowledge Distillation (Forward KL), this unigram bias gets transferred directly into the supervisory soft targets, causing:
1. **Severe class starvation & collapse** for lower-probability minority classes.
2. **Artificial degradation** of student accuracy and Macro-F1 across both encoder and causal student architectures.

### The Formulation
For teacher logits $z_T(x)$ and task-specific null vector $W_{\text{null}} = z_T(x_{\text{null}})$:
$$\tilde{z}(x) = z_T(x) - W_{\text{null}}$$
$$P_{\text{cal}}(y \mid x) = \text{Softmax}(\tilde{z}(x))$$
$$\mathcal{L}_{\text{CKD}} = D_{\text{KL}}\!\left(P_{\text{cal}}(y \mid x) \,\|\, P_{\text{student}}(y \mid x)\right)$$

* Requires **zero human labels** and exactly **one null teacher evaluation** per task ($O(1)$ task overhead).
* Deployed student operates with standard inference, incurring **zero test-time latency overhead**.

---

## 📁 Repository Structure

```
calibrated-kd/
├── README.md                          # Repository guide & comprehensive benchmark summary
├── HANDOVER.md                        # Context specification & operational protocol
├── data/
│   ├── benchmarks/                    # Clean dataset splits (Emotion, AG News, TREC-6, SST-2)
│   │   └── skewed/                    # Non-balanced & natural splits (AG News 10% min, etc.)
│   └── teacher_targets/               # Extracted teacher logits & null vectors (.npz)
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

## 📊 Summary of Master Empirical Results

### 1. Balanced Benchmarks (Macro-F1 %)

| Model / Architecture | Emotion (6 cl) | AG News (4 cl) | TREC-6 (6 cl) | SST-2 (2 cl) |
| :--- | :---: | :---: | :---: | :---: |
| Teacher (`gemma-3-4b-it`) | 34.92 | 70.24 | 69.93 | 84.21 |
| Teacher + CKD | 35.62 | 67.45 | 69.21 | 84.74 |
| **Qwen3-4B (Std KD)** | 37.59 | 69.91 | 75.08 | 80.92 |
| **Qwen3-4B (CKD)** | **39.03 (+1.44)** | **74.40 (+4.49)** | **77.12 (+2.04)** | 80.92 (0.00) |
| **Qwen2.5-0.5B (Std KD)** | 30.80 | 60.53 | 59.70 | 82.66 |
| **Qwen2.5-0.5B (CKD)** | **32.51 (+1.71)** | **62.93 (+2.40)** | **62.98 (+3.28)** | 82.66 (0.00) |
| **RoBERTa-base (Std KD)** | 36.38 | 68.78 | 68.36 | **87.90** |
| **RoBERTa-base (CKD)** | **39.63 (+3.25)** | **70.23 (+1.45)** | **69.43 (+1.07)** | 87.41 (-0.49) |

### 2. Non-Balanced & Skewed Benchmarks on RoBERTa-base (Macro-F1 %)

| Benchmark Split | Standard KD | Post-Hoc Cal | Batch Cal KD | Calibrated KD | Prior-Aware CKD | Net CKD Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AG News Skewed (10% Sci/Tech)** | 57.22 | **65.53** | 57.15 | 64.22 | 58.83 | **+7.00 pp** |
| **Emotion Natural (Saravia)** | 35.02 | **44.60** | 41.29 | 44.58 | 42.74 | **+9.56 pp** |
| **TREC-6 Natural (1.6% Abbr)** | 64.62 | 68.91 | 70.15 | **75.09** | 74.02 | **+10.47 pp** |
| **SST-2 Skewed (80/20 Imbalance)** | 65.03 | 62.97 | 67.03 | 78.69 | **84.36** | **+13.66 pp** |

* **Minority Class Rescue:** On AG News Skewed (10% Sci/Tech), Standard KD predicts Sci/Tech **0 times out of 1,000**. CKD recovers predictions to **29 / 1,000**, lifting Macro-F1 from 57.22% to **64.22%**.
* **Failure of Batch Calibration on Skew:** Batch Calibration averages soft probabilities over the training set ($\bar{p}_{\text{batch}}$), conflating low corpus frequency with prompt bias and resulting in **0 Sci/Tech predictions** (57.15% F1). Null calibration ($W_{\text{null}}$) isolates unigram token bias independently of corpus composition.

### 3. Empirical Oracle ($b^*$) Validation
Comparing unsupervised CKD correction ($-W_{\text{null}}$) against an empirical supervised oracle additive vector ($b^*$) fitted on 100 held-out labeled examples:
* **AG News:** Centered Cosine Sim = **0.3367**, Pearson $r$ = **0.3367** (World suppressed: $-45.75 \leftrightarrow -2.95$)
* **Emotion:** Centered Cosine Sim = **0.3953**, Pearson $r$ = **0.3953** (Sadness suppressed: $-16.38 \leftrightarrow -3.11$)
* **TREC-6:** Centered Cosine Sim = **0.4592**, Pearson $r$ = **0.4592** (Description suppressed: $-15.62 \leftrightarrow -3.52$)

### 4. Calibration Strength ($\beta$) Sensitivity Sweep
Evaluating $\tilde{z}(x) = z_T(x) - \beta W_{\text{null}}$ on RoBERTa-base AG News Skewed:
* $\beta = 0.0$ (Standard KD): 57.22% F1 | 0 Sci/Tech predictions (Collapsed)
* $\beta = 0.5$ (Under-calibrated): 57.50% F1 | 0 Sci/Tech predictions (Collapsed)
* $\beta = 1.0$ (Calibrated KD): **64.22% F1** | **29 Sci/Tech predictions** (**Optimal**)
* $\beta = 1.5$ (Over-calibrated): 48.93% F1 | 307 Sci/Tech predictions (Boundary distortion, Acc drops to 55.20%)

### 5. Multi-Seed Statistical Significance
Evaluating Seeds 42, 43, 44 on RoBERTa-base AG News Skewed:
* **Standard KD:** Macro-F1 = $56.62 \pm 0.46\%$, Accuracy = $65.03 \pm 0.58\%$, Sci/Tech = $0.0 \pm 0.0$
* **Calibrated KD:** Macro-F1 = **$64.82 \pm 0.54\%$**, Accuracy = **$69.83 \pm 0.09\%$**, Sci/Tech = **$36.7 \pm 6.5$**
* **Net Statistically Significant Gain:** **+8.19 pp Macro-F1**, **+4.80 pp Accuracy**, $+36.7$ minority predictions ($p < 0.001$, $>15\sigma$ above seed variance).

---

## 🚀 Reproduction & Verification

### 1. Environment Activation
```bash
# In /home/vishrut/calibrated-kd
source .venv/bin/activate
```

### 2. Verify All Numerical Results
```bash
python scripts/export/organize_and_validate_all_results.py
```

### 3. Regenerate Figures & Build Camera-Ready PDF
```bash
# Generate high-resolution publication figures (PDF + PNG)
python scripts/plotting/generate_merged_publication_figures.py

# Compile camera-ready 8-page paper and mirror to docs/
bash latex/build.sh

# Re-generate DOCX and HTML documents
python scripts/export/convert_report_to_docs.py
```
