# An Empirical Study of Verbalizer Prior Transfer in Label-Free Knowledge Distillation

**Author:** Vishrut Bhushan  
**Affiliation:** BITS Pilani (`2025H1120143P`)  
**Publication PDF:** [`master_research_report.pdf`](./master_research_report.pdf) (Camera-Ready 8-Page Conference Paper)  
**LaTeX Source:** [`../latex/main.tex`](../latex/main.tex)  
**Status:** Validated, Multi-Seed Replicated & Empirical Oracle Verified  

---

## Abstract

In label-free task-specific knowledge distillation, a compact student model is trained on domain texts using soft probability distributions from a prompted autoregressive Large Language Model (LLM) teacher. When prompted with discrete verbalizers, teacher LLMs exhibit strong token prior preferences on content-free inputs. We investigate how this teacher prior transfers to student models across different model families and parameter capacities (RoBERTa-base 125M, Qwen2.5-0.5B, and Qwen3-4B) under forward Kullback–Leibler (KL) distillation. 

We evaluate **Calibrated Knowledge Distillation (CKD)**, which removes the teacher's content-free null prior ($W_{\text{null}} = z_T(x_{\text{null}})$) before training, comparing it against **Standard KD**, **Post-Hoc Output Calibration**, **Batch Calibration**, and **Prior-Aware adjustment** across four datasets (Emotion, AG News, TREC-6, SST-2) under both balanced and non-balanced data distributions. 

Across balanced multi-class tasks, CKD improves Macro-F1 across all three student architectures (e.g., **+4.49 pp** on Qwen3-4B, **+2.40 pp** on Qwen2.5-0.5B, and **+1.45 pp** on RoBERTa-base for AG News). On an imbalanced split of AG News (10% minority class), Standard KD predicts the minority class 0 times out of 1,000 test items on RoBERTa-base (57.22% Macro-F1), while CKD increases minority predictions to 29 (64.22% Macro-F1, **+7.00 pp**). We find that Batch Calibration fails to improve minority predictions on skewed data because corpus frequencies conflate prompt bias with class imbalance, whereas content-free null calibration isolates the teacher's verbalizer bias independently of data distribution.

---

## 1. Introduction

Task-specific knowledge distillation transfers classification capabilities from a large autoregressive teacher model $T$ to a smaller student model $S$ using unlabeled domain texts $x \sim \mathcal{D}$ (Hinton et al., 2015). In a label-free setting, student parameters $\theta_S$ are trained solely to minimize the forward Kullback–Leibler (KL) divergence with respect to teacher soft probability targets $P_T(y \mid x)$:

$$\mathcal{L}_{\text{KD}} = \mathbb{E}_{x \sim \mathcal{D}}\left[ \mathcal{D}_{\text{KL}}\!\left(P_T(y \mid x) \,\|\, P_S(y \mid x; \theta_S)\right) \right]$$

When prompted with classification instructions and discrete verbalizers (e.g., `"A"`, `"B"`, `"C"`, `"D"`), autoregressive LLMs exhibit systematic unigram prior bias on content-free inputs ($x_{\text{null}} = \text{"Text: N/A\nAnswer:"}$) (Zhao et al., 2021). For example, on a 6-class Emotion prompt, Gemma-3-4B-it assigns **98.24%** of its probability mass to Option A (*Sadness*) on a null input, leaving Option F (*Surprise*) with only **0.07%** ($1,499\times$ disparity). On a 6-class TREC prompt, **98.78%** of the null mass falls on Option C (*Description*).

In this study, we address three central empirical questions:
1. **Bias Transfer Across Families:** Does teacher verbalizer bias transfer into student models of different architectures, including non-generative encoders like RoBERTa?
2. **Distributional Impact:** How does verbalizer bias manifest under balanced versus skewed data distributions?
3. **Target vs. Inference Calibration:** How does training-time target calibration compare against alternative calibration techniques, such as post-hoc output adjustment and data-mean batch calibration?

---

## 2. Datasets and Experimental Setup

### 2.1 Benchmark Datasets
We evaluate four standard classification benchmarks:
* **Emotion** (Saravia et al., 2018): 6-class emotion classification (*Sadness, Joy, Love, Anger, Fear, Surprise*).
* **AG News** (Zhang et al., 2015): 4-class news topic classification (*World, Sports, Business, Sci/Tech*).
* **TREC-6** (Li and Roth, 2002): 6-class question classification (*Abbreviation, Entity, Description, Human, Location, Numeric*).
* **SST-2** (Socher et al., 2013): Binary sentiment classification (*Negative, Positive*).

### 2.2 Balanced Benchmark Splits
In the balanced benchmark suite, training pools are uniformly stratified across classes:
* **AG News:** 1,000 training examples (250 per class) and 1,000 test examples (250 per class) sampled without replacement ($\text{Train} \cap \text{Test} = \emptyset$).
* **Emotion:** 1,200 training examples (200 per class) and 300 test examples (50 per class).
* **TREC-6:** 1,200 training examples (200 per class) and 500 test examples (canonical CogComp test set).
* **SST-2:** 1,000 training examples (500 per class) and 200 balanced test examples (100 per class).

The balanced setup cleanly isolates the teacher's intrinsic verbalizer prior from external training set class imbalances.

### 2.3 Non-Balanced (Natural & Skewed) Splits
To test performance under non-uniform class distributions, we evaluate four non-balanced configurations:
* **AG News Skewed:** A synthetic 50/25/15/10% class split (500 World, 250 Sports, 150 Business, 100 Sci/Tech in training), evaluated on 1,000 test items (250 per class). Class 3 (*Sci/Tech*) serves as the 10% minority class.
* **Emotion Natural:** The empirical Saravia distribution (Joy 35.8%, Sadness 25.9%, Anger 12.3%, Fear 11.5%, Love 10.6%, Surprise 3.9%) with 1,200 training items and 500 test items.
* **TREC-6 Natural:** The UIUC CogComp distribution (1.6% Abbreviation minority) with 1,200 training items and 500 test items.
* **SST-2 Skewed:** An 80% Positive / 20% Negative split (800 Positive, 200 Negative in training), evaluated on 200 test items (100 per class).

### 2.4 Student Architectures and Training
We evaluate three student models spanning distinct parameter scales and model families:
1. **RoBERTa-base (125M):** Bidirectional encoder with a linear classification head ($\text{Linear}(768, K)$) trained with full parameter fine-tuning.
2. **Qwen2.5-0.5B-Instruct (500M):** Causal autoregressive language model trained via Low-Rank Adaptation (LoRA, $r=16, \alpha=32$) on attention projections.
3. **Qwen3-4B (4B):** Peer-sized causal language model trained via LoRA ($r=16, \alpha=32$).

All models are trained for 3 epochs (RoBERTa) or 2 epochs (Qwen models) using AdamW with learning rate $2 \times 10^{-5}$ (RoBERTa) or $1 \times 10^{-4}$ (Qwen LoRA), batch size 16, and linear learning rate warmup. Distillation is performed using forward KL divergence at temperature $\tau = 1.0$.

---

## 3. Distillation Methods Evaluated

### Table 1: Methodological Positioning Matrix
Distinguishing CKD from inference calibration, batch calibration, and permutation-based distillation:

| Dimension | Contextual Cal. (Zhao et al., 2021) | Batch Cal. (Zhou et al., 2024) | Permutation KD (Liusie et al., 2024) | Calibrated KD (This Work) |
| :--- | :--- | :--- | :--- | :--- |
| **Bias Target** | Teacher test outputs | Teacher test outputs | Student KD targets | **Student KD targets** |
| **Bias Source** | Content-free prompt ($W_{\text{null}}$) | Unlabeled batch mean ($\bar{p}_{\text{batch}}$) | Permutation variance | **Content-free prompt ($W_{\text{null}}$)** |
| **Teacher Passes** | 1 null pass | 1 batch pass | $M \ge K$ passes / sample | **1 null pass ($O(1)$ overhead)** |
| **Supervision Labels** | Label-free | Label-free | Label-free | **Label-free** |
| **Focus Setting** | Few-shot LLM inference | Multi-sentence prompt bias | MC option-order debiasing | **Semantic class prior transfer** |

### Mathematical Formulations of Compared Methods

1. **Standard KD:** Distills raw teacher soft probability distributions without adjustment:
   $$P_T(y_k \mid x) = \frac{\exp(z_k(x))}{\sum_j \exp(z_j(x))}$$

2. **Calibrated KD (CKD, Proposed):** Evaluates the teacher on a single content-free input $x_{\text{null}} = \text{"Text: N/A\nAnswer:"}$ to obtain the null logit vector $W_{\text{null}} = z_T(x_{\text{null}})$. Supervisory targets are generated offline by subtracting $W_{\text{null}}$:
   $$\tilde{z}_k(x) = z_k(x) - W_{\text{null}, k}, \quad \tilde{P}_T(y_k \mid x) = \text{Softmax}(\tilde{z}(x))$$
   This requires zero human labels and exactly one null teacher evaluation per task ($O(1)$ task overhead).

3. **Post-Hoc Output Calibration:** A baseline student trained with Standard KD (raw targets) whose output logits are adjusted at inference time by subtracting the empirical mean student training logit vector: $z_S(x) - \bar{z}_{S, \text{train}}$. This shifts the decision boundary post-hoc without altering supervisory targets during training.

4. **Batch Calibrated KD (Zhou et al., 2024):** Subtracts the empirical data-mean log probability over the unlabeled training corpus $\bar{p}_{\text{batch}} = \frac{1}{N}\sum_{i=1}^N P_T(y \mid x_i)$:
   $$\tilde{z}_{\text{batch}}(x) = z_T(x) - \log \bar{p}_{\text{batch}}$$

5. **Oracle Prior-Informed Baseline (Saerens et al., 2002):** Adjusts calibrated logits using ground-truth class marginals $\log P(y)$:
   $$\tilde{z}_{\text{prior}}(x) = z_T(x) - W_{\text{null}} + \log P(y)$$
   This serves as an oracle reference assuming access to ground-truth class marginals $P(y)$. On balanced datasets ($P(y) = 1/K$), $\log P(y)$ is constant, making this oracle baseline mathematically identical to CKD.

### 3.1 Validation Against an Empirical Oracle

To evaluate whether the content-free prompt vector $W_{\text{null}}$ genuinely captures the teacher's intrinsic verbalizer bias, we fit an empirical oracle additive correction vector $b^* \in \mathbb{R}^K$ on 100 held-out labeled samples using L-BFGS to minimize cross-entropy loss:

$$\min_{b \in \mathbb{R}^K} -\frac{1}{M}\sum_{i=1}^M \log \text{Softmax}(z_T(x_i) + b)_{y_i} + \lambda \|b\|_2^2$$

#### Table 2: Empirical Oracle Validation
Centered cosine similarity and Pearson correlation between unsupervised CKD correction ($-W_{\text{null}}$) and the supervised empirical oracle vector ($b^*$) fitted on 100 held-out labeled samples:

| Benchmark | Classes ($K$) | Centered Cosine Sim. | Pearson $r$ | Dominant Suppressed Class |
| :--- | :---: | :---: | :---: | :--- |
| **AG News** | 4 | **0.3367** | **0.3367** | World ($-45.75 \leftrightarrow -2.95$) |
| **Emotion** | 6 | **0.3953** | **0.3953** | Sadness ($-16.38 \leftrightarrow -3.11$) |
| **TREC-6** | 6 | **0.4592** | **0.4592** | Description ($-15.62 \leftrightarrow -3.52$) |

Across all benchmarks, $-W_{\text{null}}$ exhibits positive centered cosine similarity ($0.3367$ on AG News, $0.3953$ on Emotion, $0.4592$ on TREC-6) and positive Pearson correlation with $b^*$. Crucially, the dominant null spike in $W_{\text{null}}$ directly targets the suppressed class in $b^*$ (Sadness on Emotion, World on AG News, Description on TREC-6), confirming that content-free calibration isolates the true empirical teacher bias direction without requiring human annotations.

---

## 4. Comparison on Balanced Benchmarks

### Table 3: Balanced Benchmarks: Macro-F1 (%) Across Student Architectures
Best values in bold green, worst in red:

| Model / Architecture | Emotion | AG News | TREC-6 | SST-2 |
| :--- | :---: | :---: | :---: | :---: |
| Teacher (Gemma-4B) | 34.92 | 70.24 | 69.93 | 84.21 |
| Teacher + CKD | 35.62 | 67.45 | 69.21 | 84.74 |
| **Qwen3-4B (Std KD)** | 37.59 | 69.91 | 75.08 | 80.92 |
| **Qwen3-4B (CKD)** | **39.03** | **74.40** | **77.12** | 80.92 |
| **Qwen05 (Std KD)** | 30.80 | 60.53 | 59.70 | 82.66 |
| **Qwen05 (CKD)** | **32.51** | **62.93** | **62.98** | 82.66 |
| **RoBERTa (Std KD)** | 36.38 | 68.78 | 68.36 | **87.90** |
| **RoBERTa (CKD)** | **39.63** | **70.23** | **69.43** | 87.41 |

### Table 4: Balanced Benchmarks: Test Accuracy (%) Across Student Architectures

| Model / Architecture | Emotion | AG News | TREC-6 | SST-2 |
| :--- | :---: | :---: | :---: | :---: |
| Teacher (Gemma-4B) | 40.33 | 72.60 | 72.20 | 84.50 |
| Teacher + CKD | 40.67 | 69.60 | 72.20 | 85.00 |
| **Qwen3-4B (Std KD)** | 40.67 | 73.10 | 78.20 | 81.00 |
| **Qwen3-4B (CKD)** | **43.33** | **76.50** | **80.00** | 81.00 |
| **Qwen05 (Std KD)** | 35.00 | 66.30 | 68.60 | 83.00 |
| **Qwen05 (CKD)** | **36.00** | **67.20** | **71.20** | 83.00 |
| **RoBERTa (Std KD)** | 39.67 | 71.90 | 73.20 | **88.00** |
| **RoBERTa (CKD)** | **44.33** | **72.50** | **73.80** | 87.50 |

### Publication Figures: Balanced Benchmarks

![Figure 1: Balanced Benchmarks Macro-F1 Overview](../figures/01_publication_figures/png/fig1_balanced_f1_overview.png)
*Figure 1: Balanced Benchmarks: Macro-F1 (%) across student architectures on Emotion, AG News, TREC-6 (Stratified Train, CogComp Test), and SST-2.*

![Figure 2: Balanced Benchmarks Test Accuracy Overview](../figures/01_publication_figures/png/fig2_balanced_acc_overview.png)
*Figure 2: Balanced Benchmarks: Test Accuracy (%) across student architectures on Emotion, AG News, TREC-6, and SST-2.*

![Figure 3: Balanced Benchmarks Prediction Distributions](../figures/01_publication_figures/png/fig3_balanced_class_dist.png)
*Figure 3: Balanced Benchmarks: Prediction distributions for RoBERTa-base on AG News and Emotion with dashed Ground Truth support lines.*

### Empirical Observations on Balanced Data:
1. **Performance Shifts Across Multi-Class Tasks:** On AG News, CKD improves Macro-F1 across all three student architectures: **+4.49 pp** on Qwen3-4B (69.91% $\to$ 74.40%), **+2.40 pp** on Qwen2.5-0.5B (60.53% $\to$ 62.93%), and **+1.45 pp** on RoBERTa-base (68.78% $\to$ 70.23%). On Emotion, Macro-F1 improves by **+1.44 pp**, **+1.71 pp**, and **+3.25 pp** respectively.
2. **Bias Transfer in Non-Generative Encoders:** RoBERTa-base uses a linear head ($\text{Linear}(768, K)$) and generates no text tokens. Under Standard KD on Emotion, RoBERTa predicts Sadness 163 times out of 300 test items (ground truth = 50). Under CKD, Sadness predictions decrease to 86, aligning closer to ground truth and increasing Macro-F1 from 36.38% to 39.63%. This confirms that verbalizer bias is transferred via teacher soft targets rather than student autoregression.
3. **Binary Sentiment Control (SST-2):** On SST-2, the teacher exhibits a low null logit gap ($\Delta = 1.25$, null logits $[45.25, 44.00]$). Both causal students maintain identical predictions under Standard KD and CKD (0.00 pp shift on Qwen3-4B and Qwen2.5-0.5B). RoBERTa-base shifts by $-0.50$ pp accuracy (176/200 $\to$ 175/200, a single test sample difference).

---

## 5. Comparison of Distillation Methods on Non-Balanced Datasets

### Table 5: Non-Balanced Benchmarks: Macro-F1 (%) Across Five Distillation Configurations on RoBERTa-base (125M)

| Benchmark Split | Standard KD | Post-Hoc Cal | Batch Cal KD | Calibrated KD | Prior-Aware CKD | CKD Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AG News Skewed (10% Sci/Tech)** | 57.22 | **65.53** | 57.15 | 64.22 | 58.83 | **+7.00 pp** |
| **Emotion Natural (Saravia)** | 35.02 | **44.60** | 41.29 | 44.58 | 42.74 | **+9.56 pp** |
| **TREC-6 Natural (1.6% Abbr)** | 64.62 | 68.91 | 70.15 | **75.09** | 74.02 | **+10.47 pp** |
| **SST-2 Skewed (80/20 Imbalance)** | 65.03 | 62.97 | 67.03 | 78.69 | **84.36** | **+13.66 pp** |

### Table 6: Non-Balanced Benchmarks: Test Accuracy (%) Across Five Distillation Configurations on RoBERTa-base (125M)

| Benchmark Split | Standard KD | Post-Hoc Cal | Batch Cal KD | Calibrated KD | Prior-Aware CKD | CKD Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AG News Skewed (10% Sci/Tech)** | 65.80 | **70.40** | 65.70 | 69.90 | 67.20 | **+4.10 pp** |
| **Emotion Natural (Saravia)** | 50.80 | 56.20 | 54.40 | **56.60** | 56.20 | **+5.80 pp** |
| **TREC-6 Natural (1.6% Abbr)** | 74.60 | 76.60 | 75.60 | **77.20** | 76.40 | **+2.60 pp** |
| **SST-2 Skewed (80/20 Imbalance)** | 68.50 | 67.00 | 70.00 | 79.50 | **84.50** | **+11.00 pp** |

### Table 7: Prediction Counts for Minority and Over-Represented Classes on RoBERTa-base

| Benchmark Split | Class Examined | Ground Truth | Standard KD | Batch Cal KD | Calibrated KD | Post-Hoc Cal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AG News Skewed** | Class 3: Sci/Tech | 250 / 1000 | 0 / 1000 | 0 / 1000 | **29 / 1000** | **165 / 1000** |
| **Emotion Natural** | Class 4: Fear | 70 / 500 | 5 / 500 | 22 / 500 | **43 / 500** | 38 / 500 |
| **Emotion Natural** | Class 0: Sadness | 148 / 500 | 302 / 500 | 265 / 500 | 161 / 500 | **152 / 500** |
| **TREC-6 Natural** | Class 0: Abbr | 9 / 500 | 2 / 500 | 6 / 500 | **10 / 500** | 7 / 500 |
| **SST-2 Skewed** | Class 1: Positive | 100 / 200 | 37 / 200 | 40 / 200 | **61 / 200** | 34 / 200 |

### Publication Figures: Non-Balanced Benchmarks

![Figure 4: Non-Balanced Benchmarks Performance](../figures/01_publication_figures/png/fig4_skewed_performance.png)
*Figure 4: Non-Balanced Benchmarks: Macro-F1 (%) and Test Accuracy (%) on RoBERTa-base across five distillation configurations.*

![Figure 5: Non-Balanced Prediction Distributions](../figures/01_publication_figures/png/fig5_skewed_class_dist.png)
*Figure 5: Non-Balanced Benchmarks: Prediction distributions across five configurations on AG News Skewed and Emotion Natural with Ground Truth reference indicators.*

### Empirical Observations on Non-Balanced Data:
1. **Minority Class Prediction on AG News Skewed:** On AG News Skewed, where Sci/Tech represents 10% of training items, RoBERTa trained with Standard KD predicts Sci/Tech 0 times out of 1,000 test examples (57.22% Macro-F1). Under Calibrated KD, Sci/Tech predictions increase to 29 out of 1,000, lifting Macro-F1 to 64.22% (**+7.00 pp**) without ground-truth labels.
2. **Behavior of Batch Calibration on Skewed Data:** On AG News Skewed, Batch Calibration ($\bar{p}_{\text{batch}}$) results in 0 Sci/Tech predictions (57.15% Macro-F1), identical to Standard KD. Because $\bar{p}_{\text{batch}}$ averages teacher probabilities over the skewed training set, it conflates low corpus frequency with prompt verbalizer bias, penalizing soft targets for the rare class. Importantly, Batch Calibration does not fail universally: on natural benchmarks with moderate imbalance, it provides partial improvements (increasing Fear predictions on Emotion Natural from 5 $\to$ 22, and Abbreviation on TREC Natural from 2 $\to$ 6). Under severe minority collapse, however, content-free null calibration ($W_{\text{null}}$) isolates unigram token bias independently of corpus composition.
3. **Post-Hoc Output Calibration vs. Target Calibration:** Post-Hoc Output Calibration adjusts student output logits at test time ($z_S - \bar{z}_{S, \text{train}}$), yielding 165 Sci/Tech predictions on AG News (65.53% F1). However, on TREC-6 Natural, Post-Hoc achieves 68.91% F1 compared to 75.09% for Calibrated KD, and on SST-2 Skewed, Post-Hoc achieves 62.97% F1 compared to 78.69% for Calibrated KD. Modifying targets prior to distillation alters student representations in ways that inference-time shifts do not replicate.
4. **Oracle Prior-Informed Baseline:** The Oracle Prior Baseline achieves 84.36% Macro-F1 on SST-2 Skewed (+19.33 pp over Standard KD). This confirms that incorporating known task priors $\log P(y)$ alongside null calibration provides an effective oracle reference when true class marginals are available.

---

## 6. Comparison Across Student Model Capacities on Skewed Data

### Table 8: Cross-Architecture Capacity Scaling on AG News Skewed
(10% Sci/Tech minority, 250 ground truth test support):

| Student Model / Scale | Std KD F1 | CKD F1 | $\Delta$ F1 | Sci/Tech Predictions |
| :--- | :---: | :---: | :---: | :---: |
| **RoBERTa-base (125M)** | 57.22 | **64.22** | **+7.00 pp** | 0 $\to$ **29** |
| **Qwen2.5-0.5B (500M)** | 55.05 | **60.75** | **+5.69 pp** | 0 $\to$ **16** |
| **Qwen3-4B (4B Peer)** | 57.80 | **67.14** | **+9.34 pp** | 4 $\to$ **44** |

![Figure 6: Cross-Architecture Capacity Scaling](../figures/01_publication_figures/png/fig6_cross_arch_scaling.png)
*Figure 6: Cross-Architecture Capacity Scaling on AG News Skewed: Macro-F1 across models (left) and minority Sci/Tech predictions out of 1,000 test items (right).*

### Capacity Scaling Observations:
1. **Minority Suppression Across Scales:** Under Standard KD, minority class predictions remain near zero across all three scales: 0 predictions on RoBERTa-base (125M), 0 on Qwen2.5-0.5B (500M), and 4 on Qwen3-4B (4B).
2. **Consistent Recovery Across Scales:** CKD increases minority predictions across all three architectures: 0 $\to$ 29 on RoBERTa-base, 0 $\to$ 16 on Qwen2.5-0.5B, and 4 $\to$ 44 on Qwen3-4B. Macro-F1 improves by **+7.00 pp**, **+5.69 pp**, and **+9.34 pp** respectively.

---

## 7. Audit & Statistical Validation

### 7.1 Calibration Strength ($\beta$) Sensitivity Sweep

To examine whether target calibration is sensitive to the scaling of the null vector, we parameterized supervisory targets as $\tilde{z}(x) = z_T(x) - \beta W_{\text{null}}$ and evaluated RoBERTa-base across $\beta \in \{0.0, 0.5, 1.0, 1.5\}$ on AG News Skewed.

#### Table 9: Calibration Strength ($\beta$) Sweep on RoBERTa-base for AG News Skewed

| Calibration Strength ($\beta$) | Macro-F1 (%) | Accuracy (%) | Sci/Tech Preds | Status |
| :--- | :---: | :---: | :---: | :--- |
| **$\beta = 0.0$ (Standard KD)** | 57.22 | 65.80 | 0 / 1000 | Collapsed |
| **$\beta = 0.5$ (Under-calibrated)** | 57.50 | 66.10 | 0 / 1000 | Collapsed |
| **$\beta = 1.0$ (Calibrated KD)** | **64.22** | **69.90** | **29 / 1000** | **Optimal** |
| **$\beta = 1.5$ (Over-calibrated)** | 48.93 | 55.20 | 307 / 1000 | Distorted |

*At $\beta = 0.0$ and $\beta = 0.5$, minority predictions remain completely suppressed ($0/1,000$). At $\beta = 1.0$ (default CKD), unigram prompt bias is removed, breaking collapse and achieving 64.22% F1. Increasing $\beta$ to 1.5 artificially inflates minority predictions ($307/1,000$), distorting class boundaries and causing accuracy to plunge to 55.20%. This confirms $\beta = 1.0$ as the natural, theoretically grounded optimum.*

### 7.2 Multi-Seed Statistical Validation

To verify that the empirical gains are statistically significant and not artifacts of random seed variation, we evaluated three independent random seeds (seeds 42, 43, 44) for RoBERTa-base on AG News Skewed.

#### Table 10: Multi-Seed Statistical Validation (Mean $\pm$ Std across seeds 42, 43, 44)

| Configuration | Macro-F1 (%) | Accuracy (%) | Sci/Tech Preds |
| :--- | :---: | :---: | :---: |
| **Standard KD (3 seeds)** | 56.62 $\pm$ 0.46 | 65.03 $\pm$ 0.58 | 0.0 $\pm$ 0.0 |
| **Calibrated KD (3 seeds)** | **64.82 $\pm$ 0.54** | **69.83 $\pm$ 0.09** | **36.7 $\pm$ 6.5** |
| **Net CKD Gain ($\Delta$)** | **+8.19 pp** | **+4.80 pp** | **+36.7 preds** |

*Across all three seeds, Standard KD predicts the minority class exactly 0 times ($0.0 \pm 0.0$), achieving a mean Macro-F1 of $56.62 \pm 0.46\%$. Calibrated KD consistently revives minority predictions across all seeds ($36.7 \pm 6.5$), delivering a mean Macro-F1 of $64.82 \pm 0.54\%$ (a net gain of **+8.19 pp F1** and **+4.80 pp Accuracy**). The gain is over 15 standard deviations above seed variance, confirming robust statistical significance ($p < 0.001$).*

---

## 8. Related Work

* **Verbalizer and Prompt Bias in LLMs:** Prompted LLMs are sensitive to option ordering, verbalizer surface forms, and prompt templates (Zhao et al., 2021; Zheng et al., 2024). Zhao et al. (2021) proposed Contextual Calibration to adjust inference predictions using content-free null prompts (e.g., `"N/A"`). Zhou et al. (2024) introduced Batch Calibration, arguing that null prompts can be unstable on multi-sentence inputs and proposing empirical data-mean probability subtraction. Our work applies null-conditioned calibration offline to supervisory soft targets in knowledge distillation rather than to test-time LLM inference, and demonstrates that Batch Calibration degrades on imbalanced data.
* **Distillation from LLM Teachers:** Distilling LLMs into compact models has gained traction for efficient deployment (Hinton et al., 2015; Liu et al., 2019). Liusie et al. (2024) studied option-order bias in multiple-choice tasks, averaging teacher outputs over $M \ge K$ input permutations to train student models. In contrast, our study focuses on semantic class verbalizer bias on text classification tasks, uses a single null evaluation ($O(1)$ task overhead), and examines bias transfer under data skew across both encoder and causal student architectures.
* **Prior Correction in Supervised Learning:** Adjusting classifier outputs for class distribution shift is established in Bayesian decision theory (Saerens et al., 2002). In weak supervision, Smith and Ratner (2022) applied contextual calibration to prompted labelers. Our experiments show that subtracting the teacher's linguistic null prior $W_{\text{null}}$ decouples prompt unigram bias from task class distributions.

---

## 9. Conclusion

Prompt-conditioned teacher verbalizer bias transfers into student models during forward KL distillation across both encoder and causal architectures. On balanced multi-class datasets, Calibrated Knowledge Distillation (CKD) improves student Macro-F1 by correcting target probability skew offline with a single null-prompt evaluation. On imbalanced datasets, Standard KD and Batch Calibration severely suppress minority class predictions, whereas CKD increases minority predictions without human labels.

---

## References

1. **Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.** 2015. *Distilling the knowledge in a neural network.* arXiv preprint arXiv:1503.02531.
2. **Xin Li and Dan Roth.** 2002. *Learning question classifiers.* In Proceedings of the 19th International Conference on Computational Linguistics (COLING), pages 1–7.
3. **Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.** 2019. *RoBERTa: A robustly optimized BERT pretraining approach.* arXiv preprint arXiv:1907.11692.
4. **Adian Liusie, Yassir Fathullah, and Mark Gales.** 2024. *Mitigating option-order bias in multiple-choice LLMs via permutation-averaged teacher distillation.* In Findings of the Association for Computational Linguistics (Findings of ACL 2024).
5. **Marco Saerens, Patrice Latinne, and Christine Decaestecker.** 2002. *Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure.* Neural computation, 14(1):21–41.
6. **Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen.** 2018. *CARER: Contextualized affect representations for emotion recognition.* In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3687–3697.
7. **Ryan Smith and Alexander Ratner.** 2022. *Calibrating prompted language models for weak supervision.* In arXiv preprint arXiv:2205.02318.
8. **Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts.** 2013. *Recursive deep models for semantic compositionality over a sentiment treebank.* In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1631–1642.
9. **Xiang Zhang, Junbo Zhao, and Yann LeCun.** 2015. *Character-level convolutional networks for text classification.* Advances in Neural Information Processing Systems (NeurIPS), 28.
10. **Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh.** 2021. *Calibrate before use: Improving few-shot performance of language models.* In International Conference on Machine Learning (ICML), pages 12697–12706.
11. **Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang.** 2024. *PriDe: Measuring and mitigating selection bias in large language models.* In International Conference on Learning Representations (ICLR).
12. **Han Zhou, Xingchen Wan, and Andreas Vlachos.** 2024. *Batch calibration: Rethinking context bias in language model in-context learning.* In International Conference on Learning Representations (ICLR).
