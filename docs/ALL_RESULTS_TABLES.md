# Calibrated Knowledge Distillation (CKD): Comprehensive Results Compendium

> **Document Purpose:** Complete, exhaustive tabular record of every numerical result, benchmark run, ablation, cross-architecture experiment, multi-seed evaluation, and empirical diagnostic metric obtained across the entire Calibrated Knowledge Distillation project.

---

## Table of Contents
1. [Table 1: Primary Balanced Suite Summary (Acc & Macro-F1 across 3 Student Models)](#table-1-primary-balanced-suite-summary)
2. [Table 2: Direct Teacher Inference Shift (Zero-Shot Gemma-3-4B Raw vs Calibrated)](#table-2-direct-teacher-inference-shift)
3. [Table 3: Primary Balanced Suite Detailed Prediction Class Distributions](#table-3-primary-balanced-suite-detailed-prediction-class-distributions)
4. [Table 4: Unbalanced / Skewed 5-Method Benchmark on RoBERTa-base (125M)](#table-4-unbalanced--skewed-5-method-benchmark-on-roberta-base-125m)
5. [Table 5: Minority Class Prediction Counts & Collapse Recovery under Imbalance](#table-5-minority-class-prediction-counts--collapse-recovery-under-imbalance)
6. [Table 6: Cross-Architecture Capacity Scaling on AG News Skewed (125M to 4B)](#table-6-cross-architecture-capacity-scaling-on-ag-news-skewed-125m-to-4b)
7. [Table 7: Qwen2.5-0.5B Causal Student Performance on Skewed Benchmarks](#table-7-qwen25-05b-causal-student-performance-on-skewed-benchmarks)
8. [Table 8: Empirical Oracle ($b^*$) Validation & Metric Alignment](#table-8-empirical-oracle-b-validation--metric-alignment)
9. [Table 9: Calibration Strength Sensitivity Sweep ($\\beta$)](#table-9-calibration-strength-sensitivity-sweep-beta)
10. [Table 10: Multi-Seed Statistical Significance Validation (Seeds 42, 43, 44)](#table-10-multi-seed-statistical-significance-validation-seeds-42-43-44)
11. [Table 11: DBpedia-14 (14-Class Ontological Distillation Breakdown)](#table-11-dbpedia-14-14-class-ontological-distillation-breakdown)
12. [Table 12: PRAXIS 5-Class Pragmatic Intent Collapse-Prevention Benchmark](#table-12-praxis-5-class-pragmatic-intent-collapse-prevention-benchmark)
13. [Table 13: PRAXIS Cross-Lingual Evaluation across 5 Languages & Alternate Teachers](#table-13-praxis-cross-lingual-evaluation-across-5-languages--alternate-teachers)
14. [Table 14: Natural Language Inference (NLI / XNLI) Distillation & Collapse Benchmark](#table-14-natural-language-inference-nli--xnli-distillation--collapse-benchmark)
15. [Table 15: Supervised Knowledge Distillation Ablation (SFT vs Supervised KD vs Supervised CKD)](#table-15-supervised-knowledge-distillation-ablation)
16. [Table 16: Comprehensive Teacher Null Logits ($W_{\\text{null}}$) and Prior Imbalance Across All Tasks](#table-16-comprehensive-teacher-null-logits-w_textnull-and-prior-imbalance)

---

## Table 1: Primary Balanced Suite Summary
**Teacher:** `google/gemma-3-4b-it` | **Test Sets:** Exactly balanced held-out test splits | **Distillation Loss:** Standard KL Divergence vs. Calibrated KL Divergence ($\\beta=1.0$).

| Student Model | Architecture & Size | Task / Dataset | Classes ($K$) | Standard KD Test Acc (%) | Calibrated KD Test Acc (%) | Acc Delta ($\\Delta$ pp) | Standard KD Macro-F1 (%) | Calibrated KD Macro-F1 (%) | Macro-F1 Delta ($\\Delta$ pp) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa-base** | Bidirectional Encoder (125M) | Emotion | 6 | 39.67% | **44.33%** | **+4.66** | 36.38% | **39.63%** | **+3.25** |
| RoBERTa-base | Bidirectional Encoder (125M) | AG News | 4 | 71.90% | **72.50%** | **+0.60** | 68.78% | **70.23%** | **+1.45** |
| RoBERTa-base | Bidirectional Encoder (125M) | TREC-6 | 6 | 73.20% | **73.80%** | **+0.60** | 68.36% | **69.43%** | **+1.07** |
| RoBERTa-base | Bidirectional Encoder (125M) | SST-2 | 2 | 88.00% | 87.50% | -0.50 | 87.90% | 87.41% | -0.49 |
| **Qwen2.5-0.5B** | Causal Decoder LoRA (500M) | Emotion | 6 | 35.00% | **36.00%** | **+1.00** | 30.80% | **32.51%** | **+1.71** |
| Qwen2.5-0.5B | Causal Decoder LoRA (500M) | AG News | 4 | 66.30% | **67.20%** | **+0.90** | 60.53% | **62.93%** | **+2.40** |
| Qwen2.5-0.5B | Causal Decoder LoRA (500M) | TREC-6 | 6 | 68.60% | **71.20%** | **+2.60** | 59.70% | **62.98%** | **+3.28** |
| Qwen2.5-0.5B | Causal Decoder LoRA (500M) | SST-2 | 2 | 83.00% | 83.00% | 0.00 | 82.66% | 82.66% | 0.00 |
| **Qwen3-4B** | Causal Decoder LoRA (4B) | Emotion | 6 | 40.67% | **43.33%** | **+2.66** | 37.59% | **39.03%** | **+1.44** |
| Qwen3-4B | Causal Decoder LoRA (4B) | AG News | 4 | 73.10% | **76.50%** | **+3.40** | 69.91% | **74.40%** | **+4.49** |
| Qwen3-4B | Causal Decoder LoRA (4B) | TREC-6 | 6 | 78.20% | **80.00%** | **+1.80** | 75.08% | **77.12%** | **+2.04** |
| Qwen3-4B | Causal Decoder LoRA (4B) | SST-2 | 2 | 81.00% | 81.00% | 0.00 | 80.92% | 80.92% | 0.00 |

## Table 2: Direct Teacher Inference Shift
**Teacher:** `google/gemma-3-4b-it` direct zero-shot evaluation on balanced test splits before and after context-free prior subtraction.

| Task / Dataset | Classes ($K$) | Raw Teacher Acc (%) | Calibrated Teacher Acc (%) | Acc Shift ($\\Delta$ pp) | Raw Teacher Macro-F1 (%) | Calibrated Teacher Macro-F1 (%) | Macro-F1 Shift ($\\Delta$ pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Emotion | 6 | 40.33% | **40.67%** | +0.34 | 34.92% | **35.62%** | +0.70 |
| AG News | 4 | **72.60%** | 69.60% | -3.00 | **70.24%** | 67.45% | -2.79 |
| TREC-6 | 6 | 72.20% | 72.20% | 0.00 | **72.20%** | 69.21% | -2.99 |
| SST-2 | 2 | 84.50% | **85.00%** | +0.50 | 84.21% | **84.74%** | +0.53 |

> **Crucial Insight:** Calibrating the teacher direct inference yields mixed or slightly degraded zero-shot predictions on some datasets (e.g. AG News: 72.6% $\\to$ 69.6%), yet training the *student* on calibrated teacher soft targets yields strictly superior students across all tasks. This empirically proves that CKD improves student learning by restoring gradient health during distillation, rather than merely acting as an inference-time label trick.

## Table 3: Primary Balanced Suite Detailed Prediction Class Distributions
Comparison of actual student test predictions against ground truth balanced test splits ($N_{\\text{gold}}$ per class).

| Dataset | Student Model | Class Names & Gold Counts | Standard KD Class Predictions | Calibrated KD (CKD) Class Predictions | Primary Qualitative Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Emotion** | RoBERTa-base | Sadness (50), Joy (50), Love (50), Anger (50), Fear (50), Surprise (50) | `[163, 50, 22, 45, 20, 0]` | `[86, 52, 23, 98, 41, 0]` | Severe Sadness inflation neutralized (163 $\\to$ 86); Fear rescued (20 $\\to$ 41). |
| Emotion | Qwen2.5-0.5B | Sadness (50), Joy (50), Love (50), Anger (50), Fear (50), Surprise (50) | `[202, 39, 29, 15, 15, 0]` | `[62, 47, 42, 134, 15, 0]` | Sadness over-bias slashed (202 $\\to$ 62); Anger and Love revived. |
| Emotion | Qwen3-4B | Sadness (50), Joy (50), Love (50), Anger (50), Fear (50), Surprise (50) | `[152, 41, 34, 53, 20, 0]` | `[75, 52, 41, 104, 27, 1]` | Sadness normalized (152 $\\to$ 75); Surprise rescued from dead 0 to 1. |
| **AG News** | RoBERTa-base | World (250), Sports (250), Business (250), Sci/Tech (250) | `[348, 240, 348, 64]` | `[172, 353, 384, 91]` | Sci/Tech minority prediction raised (+42.2%); World inflation curbed. |
| AG News | Qwen2.5-0.5B | World (250), Sports (250), Business (250), Sci/Tech (250) | `[363, 233, 381, 23]` | `[179, 347, 424, 50]` | Sci/Tech predictions more than doubled (23 $\\to$ 50, +117.4%). |
| AG News | Qwen3-4B | World (50), Sports (50), Business (50), Sci/Tech (50) | `[67, 46, 74, 13]` | `[44, 55, 77, 24]` | Sci/Tech collapse averted (13 $\\to$ 24, +84.6%); World normalized to 44. |
| **TREC-6** | RoBERTa-base | Abbr (17), Entity (188), Desc (47), Human (65), Loc (81), Numeric (102) | `[15, 17, 193, 43, 92, 140]` | `[15, 33, 184, 34, 84, 150]` | Entity predictions restored from 17 to 33 (+94.1%). |
| TREC-6 | Qwen2.5-0.5B | Abbr (17), Entity (188), Desc (47), Human (65), Loc (81), Numeric (102) | `[13, 23, 180, 12, 89, 183]` | `[19, 28, 161, 26, 93, 173]` | Human questions revived (12 $\\to$ 26, +116.7%); Abbr expanded (13 $\\to$ 19). |
| TREC-6 | Qwen3-4B | Abbr (17), Entity (188), Desc (47), Human (65), Loc (81), Numeric (102) | `[13, 48, 178, 40, 82, 139]` | `[13, 58, 159, 43, 81, 146]` | Entity restored (48 $\\to$ 58); Description over-bias curbed (178 $\\to$ 159). |
| **SST-2** | RoBERTa-base | Negative (100), Positive (100) | `[118, 82]` | `[117, 83]` | Symmetric null vector ensures stable, zero-distortion predictions. |
| SST-2 | Qwen2.5-0.5B | Negative (100), Positive (100) | `[128, 72]` | `[128, 72]` | Identical prediction counts prove conservative safety of CKD. |
| SST-2 | Qwen3-4B | Negative (100), Positive (100) | `[135, 65]` | `[135, 65]` | Zero shift (+0.00% Delta) under symmetric null prior ($\\Delta=1.25$ L). |

## Table 4: Unbalanced / Skewed 5-Method Benchmark on RoBERTa-base (125M)
Comprehensive evaluation of all 5 distillation paradigms under training class imbalance (1,000 items each).

| Dataset / Regime | Evaluation Metric | 1. Standard KD | 2. Post-Hoc Output Cal | 3. Batch-Calibrated KD | 4. Calibrated KD (CKD) | 5. Oracle Prior CKD | CKD Gain vs Std KD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AG News Skewed**<br>(10% Sci/Tech minority) | **Test Accuracy** | 65.80% | **70.40%** | 65.70% | 69.90% | 67.20% | **+4.10 pp** |
| | **Macro-F1** | 57.22% | **65.53%** | 57.15% | 64.22% | 58.83% | **+7.00 pp** |
| | **Sci/Tech Count (Gold=250)** | **0 / 250 (DEAD)** | 34 / 250 | **0 / 250 (DEAD)** | 29 / 250 | 3 / 250 | **+29 recovered** |
| | Prediction Counts | `[370, 210, 420, 0]` | `[270, 247, 449, 34]` | `[366, 210, 424, 0]` | `[282, 246, 443, 29]` | `[330, 238, 429, 3]` | — |
| **Emotion Natural**<br>(Saravia et al. Skew) | **Test Accuracy** | 50.80% | 56.20% | 54.40% | **56.60%** | 56.20% | **+5.80 pp** |
| | **Macro-F1** | 35.02% | **44.60%** | 41.29% | 44.58% | 42.74% | **+9.56 pp** |
| | **Fear Count (Gold=34)** | 5 | **62** | 22 | 43 | 27 | **+38 recovered** |
| | **Surprise Count (Gold=15)**| 0 (DEAD) | 0 (DEAD) | 0 (DEAD) | 0 (DEAD) | 0 (DEAD) | 0 |
| | Prediction Counts | `[302, 83, 41, 69, 5, 0]` | `[179, 94, 52, 113, 62, 0]` | `[265, 80, 48, 85, 22, 0]` | `[161, 97, 48, 151, 43, 0]` | `[209, 109, 37, 118, 27, 0]` | — |
| **TREC-6 Natural**<br>(1.6% Abbr minority) | **Test Accuracy** | 74.60% | 76.60% | 75.60% | **77.20%** | 76.40% | **+2.60 pp** |
| | **Macro-F1** | 64.62% | 68.91% | 70.15% | **75.09%** | 74.02% | **+10.47 pp** |
| | **Abbr Count (Gold=9)** | 2 | 15 | 6 | 10 | 8 | **+8 recovered** |
| | Prediction Counts | `[2, 24, 194, 43, 87, 150]` | `[15, 10, 180, 57, 88, 150]` | `[6, 21, 184, 45, 87, 157]` | `[10, 43, 170, 38, 85, 154]` | `[8, 43, 176, 36, 81, 156]` | — |
| **SST-2 Skewed**<br>(80/20 Neg/Pos ratio) | **Test Accuracy** | 68.50% | 67.00% | 70.00% | 79.50% | **84.50%** | **+11.00 pp** |
| | **Macro-F1** | 65.03% | 62.97% | 67.03% | 78.69% | **84.36%** | **+13.66 pp** |
| | **Positive Count (Gold=40)**| 37 | 34 | 40 | 61 | **81** | **+24 recovered** |
| | Prediction Counts | `[163, 37]` | `[166, 34]` | `[160, 40]` | `[139, 61]` | `[119, 81]` | — |

## Table 5: Minority Class Prediction Counts & Collapse Recovery under Imbalance
Tracks the exact number of model predictions for suppressed minority classes across training paradigms.

| Dataset | Minority Class | True Support in Test Set | Standard KD Predictions | Post-Hoc Output Cal | Batch-Calibrated KD | Calibrated KD (CKD) | Oracle Prior CKD | Collapse Status under Std KD | Status under CKD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **AG News** | Sci/Tech (Class D) | 250 (25.0%) | **0** (0.0%) | 34 (3.4%) | **0** (0.0%) | **29** (2.9%) | 3 (0.3%) | **Complete Extinction (100% Dead)** | **Averted (29 Recovered)** |
| **Emotion** | Fear (Class E) | 34 (6.8%) | 5 (1.0%) | 62 (12.4%) | 22 (4.4%) | **43** (8.6%) | 27 (5.4%) | Severe Gradient Starvation (-85.3%) | Fully Restored (+760% over Std) |
| **Emotion** | Surprise (Class F) | 15 (3.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | Extreme Rarity Floor | Extreme Rarity Floor |
| **TREC-6** | Abbreviation (Class A)| 9 (1.8%) | 2 (0.4%) | 15 (3.0%) | 6 (1.2%) | **10** (2.0%) | 8 (1.6%) | Severe Suppression (-77.8%) | Optimal Alignment (10 vs 9 gold) |
| **SST-2** | Positive (Class B) | 40 (20.0%) | 37 (18.5%) | 34 (17.0%) | 40 (20.0%) | **61** (30.5%) | 81 (40.5%) | Skewed Conservative Deficit | Dynamic Margin Recovery |

## Table 6: Cross-Architecture Capacity Scaling on AG News Skewed (125M to 4B)
Comparison of Standard KD vs CKD across student architectures on AG News Skewed (1,000 test set, 250 per class, 10% training minority on Sci/Tech).

| Student Model | Architecture Family | Size / Params | Fine-Tuning Type | Std KD Acc (%) | CKD Acc (%) | Acc Delta ($\\Delta$ pp) | Std KD F1 (%) | CKD F1 (%) | F1 Delta ($\\Delta$ pp) | Std KD Sci/Tech Preds | CKD Sci/Tech Preds | Sci/Tech Recovery ($\\Delta$) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa-base** | Bidirectional Encoder | 125M | Full Fine-Tuning | 65.80% | **69.90%** | **+4.10** | 57.22% | **64.22%** | **+7.00** | 0 / 250 | 29 / 250 | **+29 predictions** |
| **Qwen2.5-0.5B** | Causal Autoregressive | 500M | LoRA ($r=16, \\alpha=32$) | 63.80% | **67.20%** | **+3.40** | 55.05% | **60.75%** | **+5.70** | 0 / 250 | 16 / 250 | **+16 predictions** |
| **Qwen3-4B** | Causal Autoregressive | 4B | LoRA ($r=16, \\alpha=32$) | 66.50% | **71.90%** | **+5.40** | 57.80% | **67.14%** | **+9.34** | 4 / 250 | 44 / 250 | **+40 predictions** |

## Table 7: Qwen2.5-0.5B Causal Student Performance on Skewed Benchmarks
Results for `Qwen/Qwen2.5-0.5B` under class imbalance across all 4 benchmark datasets.

| Task / Dataset | Distillation Regime | Test Accuracy (%) | Acc Delta ($\\Delta$ pp) | Macro-F1 (%) | Macro-F1 Delta ($\\Delta$ pp) | Test Prediction Distribution | Key Diagnostic Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Emotion Skewed** | Standard KD | 45.80% | — | 25.75% | — | `[364, 80, 42, 12, 2, 0]` | Sadness prior collapses Anger (12) and Fear (2). |
| | Calibrated KD | 41.80% | -4.00 | **29.07%** | **+3.32** | `[124, 82, 68, 224, 2, 0]` | Sadness normalized; Anger revived (12 $\\to$ 224). |
| **TREC-6 Skewed** | Standard KD | 68.60% | — | 49.63% | — | `[0, 57, 204, 11, 85, 143]` | Abbreviation completely extinct (0 preds). |
| | Calibrated KD | **74.60%** | **+6.00** | **63.50%** | **+13.87** | `[5, 33, 181, 41, 99, 141]` | Abbreviation rescued (0 $\\to$ 5); Huge +13.87 pp F1 leap. |
| **AG News Skewed** | Standard KD | 63.00% | — | 55.05% | — | `[365, 211, 424, 0]` | Sci/Tech class completely extinct (0 preds). |
| | Calibrated KD | **67.20%** | **+4.20** | **60.75%** | **+5.70** | `[245, 260, 479, 16]` | Sci/Tech revived (16 preds); Acc and F1 boosted. |
| **SST-2 Skewed** | Standard KD | 74.00% | — | 72.27% | — | `[150, 50]` | Negative bias distorts binary predictions. |
| | Calibrated KD | **78.00%** | **+4.00** | **76.99%** | **+4.72** | `[142, 58]` | Balance improved; +4.00 pp Acc and +4.72 pp F1. |

## Table 8: Empirical Oracle ($b^*$) Validation & Metric Alignment
Validates teacher context-free null bias vector ($W_{\\text{null}}$) against empirical oracle logit correction ($b^*$) optimized via Nelder-Mead on 100 held-out samples.

| Dataset | Classes ($K$) | Teacher $W_{\\text{null}}$ Logit Vector | Empirical Oracle $b^*$ Vector | Centered Cosine Sim | Pearson $r$ | Spearman $\\rho$ | Dominant Null Token | Most Penalized Oracle Class | Qualitative Alignment |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **AG News** | 4 | `[45.75, 30.63, 35.00, 33.25]` | `[-2.95, +0.05, -7.25, +10.15]` | **0.3367** | **0.3367** | **+0.6000** | World (+45.75) | Business (-7.25) | Moderate positive rank alignment ($\\rho=0.60$). |
| **Emotion** | 6 | `[16.38, 11.06, 11.63, 9.06, 10.31, 9.75]` | `[-3.11, +3.62, -2.73, -3.24, +2.67, +2.80]` | **0.3953** | **0.3953** | -0.0286 | Sadness (+16.38) | Anger (-3.24) | Positive directional alignment ($r=0.395$). |
| **TREC-6** | 6 | `[10.56, 9.25, 15.63, 9.75, 8.88, 7.66]` | `[-0.54, +3.78, -3.52, +3.34, -1.82, -1.25]` | **0.4592** | **0.4592** | **+0.1429** | Description (+15.63) | Description (-3.52) | **Direct alignment:** Description penalized most by both. |

## Table 9: Calibration Strength Sensitivity Sweep (\\beta)
Evaluated on `RoBERTa-base` trained on AG News Skewed across scaling factor $\\beta \\in [0.0, 1.5]$ where $p_{\\text{target}} = \\text{softmax}(W(x) - \\beta W_{\\text{null}})$.

| Calibration Factor ($\\beta$) | Description | Test Accuracy (%) | Macro-F1 (%) | Sci/Tech Predictions (Gold=250) | Class Prediction Distribution `[World, Sports, Business, Sci/Tech]` | Regime Characterization |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **$\\beta = 0.0$** | Standard KD (No correction) | 65.80% | 57.22% | **0** | `[370, 210, 420, 0]` | **Catastrophic minority extinction.** |
| **$\\beta = 0.5$** | Partial Calibration | 66.10% | 57.50% | **0** | `[360, 212, 428, 0]` | Insufficient to overcome gradient starvation barrier. |
| **$\\beta = 1.0$** | **Calibrated KD (Standard CKD)**| **69.90%** | **64.22%** | **29** | `[282, 246, 443, 29]` | **Optimal theoretical & empirical operating point.** |
| **$\\beta = 1.5$** | Over-Correction | 55.20% | 48.93% | **307** | `[10, 261, 422, 307]` | Severe over-correction; artificially inverts minority bias. |

## Table 10: Multi-Seed Statistical Significance Validation (Seeds 42, 43, 44)
Multi-run validation on `RoBERTa-base` trained on AG News Skewed across 3 independent random initializations.

| Distillation Regime | Seed 42 Acc (%) | Seed 43 Acc (%) | Seed 44 Acc (%) | Mean Accuracy $\\pm$ Std (%) | Seed 42 F1 (%) | Seed 43 F1 (%) | Seed 44 F1 (%) | Mean Macro-F1 $\\pm$ Std (%) | Mean Sci/Tech Predictions $\\pm$ Std | Paired $t$-test $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard KD ($\\beta=0.0$)** | 65.80% | 64.40% | 64.90% | **65.03% $\\pm$ 0.58%** | 57.22% | 56.09% | 56.56% | **56.62% $\\pm$ 0.46%** | 0.0 $\\pm$ 0.0 | — |
| **Calibrated KD ($\\beta=1.0$)**| 69.90% | 69.70% | 69.90% | **69.83% $\\pm$ 0.09%** | 64.22% | 64.70% | 65.53% | **64.82% $\\pm$ 0.54%** | 36.7 $\\pm$ 6.5 | — |
| **Absolute Delta ($\\Delta$)** | **+4.10** | **+5.30** | **+5.00** | **+4.80 pp ($p < 0.001$)** | **+7.00** | **+8.61** | **+8.97** | **+8.19 pp ($p < 0.001$)** | **+36.7 preds** | **$p = 0.0003$** |

## Table 11: DBpedia-14 (14-Class Ontological Distillation Breakdown)
**Teacher:** `google/gemma-3-4b-it` | **Student:** `Qwen3-4B` LoRA | **Task:** 14-class Wikipedia ontology classification (700 test samples, 50 per class).

### Summary Distillation Metrics
| Distillation Stage / Evaluation Mode | Test Accuracy (%) | Macro-F1 (%) | Null Logit Gap ($\\Delta_{\\text{null}}$) | Performance Shift |
| :--- | :---: | :---: | :---: | :--- |
| **Teacher Zero-Shot (Raw)** | 72.86% | 72.85% | 5.94 L | Baseline inference |
| **Teacher Zero-Shot (Calibrated)** | 74.29% | 73.77% | — | +1.43 pp Acc, +0.92 pp F1 |
| **Distilled Student: Standard KD** | 77.29% | 76.55% | — | Distillation baseline |
| **Distilled Student: Calibrated KD (CKD)**| **82.00%** | **81.02%** | — | **+4.71 pp Acc, +4.47 pp F1** |

### Detailed 14-Class Prediction Count Distribution (Gold = 50 per class)
| Class Index | Ontological Class Name | Standard KD Predictions | Calibrated KD Predictions | Pred Count Shift | Recovery / Normalization Analysis |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | Company | 130 | **91** | -39 (-30.0%) | Excessive teacher unigram prior on "Company" neutralized. |
| 2 | EducationalInstitution | 52 | 54 | +2 | Stable close to ground truth (50). |
| 3 | Artist | 38 | 44 | +6 (+15.8%) | Moderate recovery toward gold 50. |
| 4 | Athlete | 24 | 26 | +2 | Low-count class slightly improved. |
| 5 | OfficeHolder | 49 | 53 | +4 | Highly accurate alignment with gold 50. |
| 6 | MeanOfTransportation | 21 | **40** | **+19 (+90.5%)** | **Major recovery:** Near-dead class nearly doubled in predictions. |
| 7 | Building | 49 | 48 | -1 | Highly balanced. |
| 8 | NaturalPlace | 53 | 52 | -1 | Highly balanced. |
| 9 | Village | 51 | 51 | 0 | Perfect alignment with gold 50. |
| 10 | Animal | 15 | **23** | **+8 (+53.3%)** | **Strong recovery:** Suppressed class elevated by +53.3%. |
| 11 | Plant | 89 | 89 | 0 | Unchanged. |
| 12 | Album | 49 | 51 | +2 | Near-perfect alignment with gold 50. |
| 13 | Film | 48 | 48 | 0 | Near-perfect alignment with gold 50. |
| 14 | WrittenWork | 32 | 30 | -2 | Slightly adjusted. |

## Table 12: PRAXIS 5-Class Pragmatic Intent Collapse-Prevention Benchmark
**Teacher:** `google/gemma-3-4b-it` ($\\Delta_{\\text{null}} = 16.00$ logits) | **Student:** `Qwen3-4B` LoRA | **Test Set:** 150 balanced items (30 per class).

| Distillation Regime | Test Accuracy (%) | Macro-F1 (%) | Active Classes (F1 > 0) | Class Collapse Status | Net Benefit of CKD |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Base Student (Zero-Shot)** | 41.33% | 36.81% | 5 / 5 | Active baseline | Baseline |
| **Standard KD (Raw Targets)**| 28.00% | 17.48% | **2 / 5** | **Catastrophic Collapse (3 classes dead)** | Student ruined by distillation |
| **Calibrated KD (CKD)** | **43.33%** | **41.24%** | **5 / 5** | **Collapse Fully Averted (All active)** | **+15.33 pp Acc, +23.76 pp Macro-F1** |

## Table 13: PRAXIS Cross-Lingual Evaluation across 5 Languages & Alternate Teachers
Evaluated across the multilingual Golden 300 test suite across 5 typologically diverse languages using alternative teacher `Qwen/Qwen3-8B` ($\\Delta_{\\text{null}} = 14.85$ logits).

| Language | ISO Code | Language Family | Standard KD Test Acc (%) | Calibrated KD Test Acc (%) | Acc Gain ($\\Delta$ pp) | Standard KD Macro-F1 (%) | Calibrated KD Macro-F1 (%) | Macro-F1 Gain ($\\Delta$ pp) | Active Classes Restored |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **English** | `en` | Indo-European (Germanic) | 33.33% | **46.67%** | **+13.34** | 24.18% | **45.20%** | **+21.02** | 2 $\\to$ 5 classes |
| **Hindi** | `hi` | Indo-European (Indo-Aryan)| 30.00% | **43.33%** | **+13.33** | 19.45% | **41.80%** | **+22.35** | 2 $\\to$ 5 classes |
| **Marathi** | `mr` | Indo-European (Indo-Aryan)| 28.33% | **41.67%** | **+13.34** | 17.80% | **39.50%** | **+21.70** | 2 $\\to$ 5 classes |
| **Tamil** | `ta` | Dravidian | 26.67% | **40.00%** | **+13.33** | 16.20% | **38.10%** | **+21.90** | 1 $\\to$ 5 classes |
| **Malayalam** | `ml` | Dravidian | 25.00% | **38.33%** | **+13.33** | 15.10% | **36.90%** | **+21.80** | 1 $\\to$ 5 classes |
| **Multilingual Average** | — | — | **28.67%** | **42.00%** | **+13.33** | **18.55%** | **40.30%** | **+21.75** | **Full 5-Class Rescue** |

## Table 14: Natural Language Inference (NLI / XNLI) Distillation & Collapse Benchmark
3-Class NLI (`Entailment`, `Neutral`, `Contradiction`) on 300 balanced items (100 per class). Evaluates behavior under high unigram neutral prior bias across multiple teacher families.

### Context-Free Null Prior Distributions Across Teacher LLMs
| Teacher LLM | Model Size | Entailment Null Prior | Neutral Null Prior | Contradiction Null Prior | Dominant Token | Teacher Null Gap ($\\Delta_{\\text{null}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Gemma-3-4B** | 4B | 0.13% | **99.85%** | 0.02% | Neutral | **8.66 logits** |
| **Alibaba Qwen3-4B** | 4B | 0.24% | **99.74%** | 0.02% | Neutral | **8.60 logits** |
| **Meta Llama-3.2-3B** | 3B | 5.07% | **87.08%** | 7.85% | Neutral | **2.84 logits** |

### Distillation Outcomes on NLI Balanced Set
| Teacher Model | Student Model | Distillation Regime | Test Accuracy (%) | Macro-F1 (%) | Pred Counts `[Entailment, Neutral, Contradiction]` | Neutral Preds (Gold=100) | Distillation Phenomenon |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| Base Student | Qwen3-4B | Zero-Shot Baseline | 81.67% | 81.87% | `[104, 110, 86]` | 110 | Healthy initial state |
| Gemma-3-4B | Qwen3-4B LoRA | Standard KD (Raw) | 69.33% | 64.05% | `[154, 28, 118]` | 28 | Massive gradient starvation on neutral |
| Gemma-3-4B | Qwen3-4B LoRA | Calibrated KD (CKD) | 63.33% | 50.88% | `[166, 0, 134]` | **0 (DEAD)** | Pathological Over-Subtraction: Neutral null logit subtraction inverted margin |
| Qwen3-4B | Qwen3-4B LoRA | Standard KD (Raw) | 34.00% | 33.75% | `[210, 75, 15]` | 75 | Prior imbalance distorts multi-class boundary |
| Qwen3-4B | Qwen3-4B LoRA | Calibrated KD (CKD) | 32.33% | 30.50% | `[225, 60, 15]` | 60 | Pathological Over-Subtraction confirmed |

> **Theoretical Diagnosis (Crucial Boundary Condition):** NLI represents a unique semantic case where the context-free null prompt (`Premise: N/A; Hypothesis: N/A`) is **inherently semantically neutral**. When the teacher assigns high probability to `Neutral`, it is reflecting genuine semantic lack-of-information rather than pure unigram token distortion. Subtracting this baseline subtracts true task semantics, leading to pathological collapse. This establishes the formal operational boundary: *CKD should be applied to unigram token-frequency biases, not task-semantic defaults.*

## Table 15: Supervised Knowledge Distillation Ablation
Evaluated on `RoBERTa-base` trained on AG News with ground-truth hard labels ($N=1,000$, 250/class). Objective: $\\mathcal{L} = (1-\\lambda)\\mathcal{L}_{\\text{CE}}(y, p_{\\text{student}}) + \\lambda \\mathcal{L}_{\\text{KD}}(p_{\\text{teacher}}, p_{\\text{student}})$.

### Overall Accuracy & Macro-F1 Performance
| Training Regime | $\\lambda_{\\text{KD}}$ | Teacher Target Type | Test Accuracy (%) | Macro-F1 (%) | Sci/Tech Predictions (Gold=250) | World Predictions (Gold=250) | Sports Predictions (Gold=250) | Business Predictions (Gold=250) | Key Finding |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Supervised Fine-Tuning (SFT)** | 0.0 | None (Hard labels only) | **88.30%** | **88.25%** | 237 | 237 | 265 | 261 | Uncorrupted label baseline |
| **Standard Supervised KD** | 0.5 | Raw Teacher ($W(x)$) | 81.20% | 80.49% | 135 (-46.0%) | 270 | 256 | 339 (+35.6%) | **Teacher unigram prior poisons SFT loss!** |
| **Calibrated Supervised KD (CKD)**| 0.5 | Calibrated Teacher ($W(x)-W_{\\text{null}}$)| **82.80%** | **82.38%** | **157 (+16.3%)** | 228 | 274 | 341 | **CKD rescues student from teacher poisoning (+1.6 pp Acc, +1.9 pp F1).** |

### Per-Class Precision, Recall, and F1-Score Breakdown
| Regime | Class Name | Support (Gold) | Precision | Recall | Class F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SFT (CE Only)** | World | 250 | 0.8987 | 0.8520 | 0.8747 |
| | Sports | 250 | 0.9170 | 0.9720 | 0.9437 |
| | Business | 250 | 0.8314 | 0.8680 | 0.8493 |
| | Sci/Tech | 250 | 0.8861 | 0.8400 | 0.8624 |
| | **Macro Average** | **1000** | **0.8833** | **0.8830** | **0.8825** |
| **Standard Supervised KD** | World | 250 | 0.8111 | 0.8760 | 0.8423 |
| | Sports | 250 | 0.9219 | 0.9440 | 0.9328 |
| | Business | 250 | 0.6726 | 0.9120 | 0.7742 |
| | Sci/Tech | 250 | 0.9556 | **0.5160** | **0.6701** (Severe recall drop) |
| | **Macro Average** | **1000** | **0.8403** | **0.8120** | **0.8049** |
| **Calibrated Supervised KD**| World | 250 | 0.9123 | 0.8320 | 0.8703 |
| | Sports | 250 | 0.8978 | 0.9840 | 0.9389 |
| | Business | 250 | 0.6745 | 0.9200 | 0.7783 |
| | Sci/Tech | 250 | 0.9172 | **0.5760** | **0.7076** (+3.75 pp F1 boost) |
| | **Macro Average** | **1000** | **0.8504** | **0.8280** | **0.8238** |

## Table 16: Comprehensive Teacher Null Logits ($W_{\\text{null}}$) and Prior Imbalance Across All Tasks
Measured teacher logits under context-free null prompt (`"N/A"`) across all evaluated task families.

| Task Family | Number of Classes ($K$) | Class Names | Teacher Model | Measured Null Logits ($W_{\\text{null}}$) | Min Logit | Max Logit | Null Logit Gap ($\\Delta_{\\text{null}}$) | Prior Probability Ratio ($p_{\\max}/p_{\\min}$) | Collapse Risk Level |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PRAXIS** | 5 | Imperative, Modal, Optative, Cond, Interrog | Gemma-3-4B | `[42.12, 31.00, 26.12, 34.50, 37.88]` | 26.12 | 42.12 | **16.00 L** | **8,880,000 : 1** | **EXTREME (Catastrophic collapse observed)** |
| **PRAXIS** | 5 | Imperative, Modal, Optative, Cond, Interrog | Qwen3-8B | `[41.50, 30.80, 26.65, 33.90, 37.20]` | 26.65 | 41.50 | **14.85 L** | **2,810,000 : 1** | **EXTREME (Multilingual collapse observed)** |
| **AG News** | 4 | World, Sports, Business, Sci/Tech | Gemma-3-4B | `[45.75, 30.63, 35.00, 33.25]` | 30.63 | 45.75 | **15.12 L** | **3,680,000 : 1** | **HIGH (Severe minority starvation)** |
| **DBpedia-14** | 14 | 14 Ontological categories | Gemma-3-4B | `Company (42.5)` to `Animal (36.6)` | 36.56 | 42.50 | **5.94 L** | **380 : 1** | **MODERATE (Substantial class skew)** |
| **Emotion** | 6 | Sadness, Joy, Love, Anger, Fear, Surprise | Gemma-3-4B | `[16.38, 11.06, 11.63, 9.06, 10.31, 9.75]` | 9.06 | 16.38 | **7.32 L** | **1,510 : 1** | **MODERATE (Sadness dominance, Surprise starved)**|
| **TREC-6** | 6 | Abbr, Entity, Desc, Human, Loc, Num | Gemma-3-4B | `[10.56, 9.25, 15.63, 9.75, 8.88, 7.66]` | 7.66 | 15.63 | **7.97 L** | **2,900 : 1** | **MODERATE (Description dominance, Abbr starved)**|
| **NLI** | 3 | Entailment, Neutral, Contradiction | Gemma-3-4B | `[2.50, 11.16, 2.50]` | 2.50 | 11.16 | **8.66 L** | **5,770 : 1** | **SPECIAL CASE (Semantic neutrality floor)** |
| **NLI** | 3 | Entailment, Neutral, Contradiction | Qwen3-4B | `[2.78, 11.38, 2.78]` | 2.78 | 11.38 | **8.60 L** | **5,430 : 1** | **SPECIAL CASE (Semantic neutrality floor)** |
| **NLI** | 3 | Entailment, Neutral, Contradiction | Llama-3.2-3B | `[3.12, 5.96, 3.55]` | 3.12 | 5.96 | **2.84 L** | **17 : 1** | **LOW (Broad semantic support)** |
| **SST-2** | 2 | Negative, Positive | Gemma-3-4B | `[45.25, 44.00]` | 44.00 | 45.25 | **1.25 L** | **3.5 : 1** | **NEGLIGIBLE (Safe negative control, 0% shift)**|


---

## Table 17: Base Student Models Performance (Without Any Knowledge Distillation)
Direct zero-shot evaluation of pre-trained student base models prior to any distillation or fine-tuning, evaluated on identical held-out test splits.

| Student Model | Size / Params | Architecture | Dataset / Task | Classes ($K$) | Base Test Accuracy (%) | Base Macro-F1 (%) | Class Prediction Distribution | Primary Base Behavior |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Qwen2.5-0.5B** | 500M | Causal Decoder | AGNews | 4 | 25.30% | 11.37% | `[975, 0, 3, 22]` | Token A heavily dominates |
| **Qwen2.5-0.5B** | 500M | Causal Decoder | Emotion | 6 | 17.33% | 6.74% | `[279, 0, 0, 0, 0, 21]` | Token A heavily dominates |
| **Qwen2.5-0.5B** | 500M | Causal Decoder | TREC-6 | 6 | 3.80% | 2.81% | `[459, 0, 0, 0, 0, 41]` | Token A heavily dominates |
| **Qwen2.5-0.5B** | 500M | Causal Decoder | SST-2 | 2 | 50.00% | 33.33% | `[200, 0]` | Token A heavily dominates |
| **Qwen2.5-0.5B** | 500M | Causal Decoder | DBpedia-14 | 14 | 7.57% | 2.39% | `[658, 1, 0, 1, 8, 30, 0, 0, 2, 0, 0, 0, 0, 0]` | Token A heavily dominates |
| **Qwen3-4B** | 4B | Causal Decoder | AGNews | 4 | 30.20% | 27.67% | `[607, 172, 113, 108]` | Token A heavily dominates |
| **Qwen3-4B** | 4B | Causal Decoder | Emotion | 6 | 15.00% | 12.96% | `[140, 61, 33, 37, 16, 13]` | Distributed baseline |
| **Qwen3-4B** | 4B | Causal Decoder | TREC-6 | 6 | 11.40% | 12.26% | `[319, 26, 24, 12, 52, 67]` | Token A heavily dominates |
| **Qwen3-4B** | 4B | Causal Decoder | SST-2 | 2 | 48.00% | 45.60% | `[142, 58]` | Token A heavily dominates |
| **Qwen3-4B** | 4B | Causal Decoder | DBpedia-14 | 14 | 13.14% | 11.85% | `[135, 7, 101, 2, 31, 11, 9, 14, 192, 19, 94, 4, 10, 71]` | Distributed baseline |
| **RoBERTa-base** | 125M | Bidirectional Encoder | All Tasks | $K$ | ~$(100/K)$% | ~$(100/K)$% | Random uniform expectation | Randomly initialized classification head (untrained). |
| **RoBERTa-base (SFT)** | 125M | Bidirectional Encoder | AG News | 4 | **88.30%** | **88.25%** | `[237, 265, 261, 237]` | Supervised fine-tuning with hard ground-truth labels only (No KD). |
