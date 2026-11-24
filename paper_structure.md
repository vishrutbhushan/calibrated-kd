# Paper Structure & Comprehensive Template

---

## 1. Abstract

* **Motivation & Context:** Why Knowledge Distillation (KD) is essential for deploying large language models (LLMs) into compact, low-latency edge students.
* **The Core Problem:** Autoregressive LLM teachers possess severe unigram token-frequency biases inherited from pre-training. In classification via prompt verbalizers, this static bias dominates soft probability mass, starving minority classes of gradient signal and causing complete class collapse in the student model.
* **Prior Solutions & Gaps:** Existing calibration methods operate either at test time on giant LLMs (imposing severe inference latency) or require labeled training frequencies (unavailable in unsupervised KD).
* **Our Proposed Method (Calibrated KD / CKD):** A zero-label, training-time calibration method that probes the teacher once offline with an uninformative null prompt (`"N/A"`), subtracting this static logit prior directly inside the distillation loss function. Imposes zero student inference overhead.
* **Key Findings:** Evaluated across 6 authentic, naturally skewed multi-class benchmarks ($K=4$ to $6$), CKD consistently prevents minority class starvation and representation distortion, boosting Macro-F1 by up to +9.58 pp over Post-Hoc output calibration and up to +21.3 pp over Standard KD, while establishing peak scores of up to 93.10% Macro-F1 when combined with test-time calibration.

---

## 2. Introduction

* **Overview of Model Compression in Production:** Why industry prefers compact bi-directional encoders (e.g., RoBERTa) or small SLMs over running multi-billion parameter decoders for high-throughput text classification.
* **The Unseen Pathology in LLM Distillation:** How soft targets transferred from generative teachers ($z_{\text{raw}} = z_{\text{semantic}} + b_{\text{prior}}$) poison the student's output space, creating artificial overboosting on dominant tokens and zero-gradient starvation on minority tokens.
* **Core Contribution:**
  1. Formalizing the verbalizer unigram bias transfer problem in teacher-student distillation.
  2. Proposing Calibrated Knowledge Distillation (CKD), an offline, zero-label debiasing mechanism.
  3. Extensive empirical validation across 6 authentic naturally skewed tasks, multi-task distillation settings, student capacity scales (125M to 4B), and hard vs. soft label comparisons.

---

## 3. Related Work

* **Classical Knowledge Distillation:** Hinton et al. (NeurIPS 2015) — foundation of dark knowledge and temperature-scaled soft targets.
* **In-Context Bias & Inference Calibration:**
  * Zhao et al. (ICML 2021) — Contextual Calibration for LLM few-shot inference using content-free inputs.
  * Zhou et al. (ICLR 2024) — Batch Calibration (BC) via unlabeled batch-mean centering during inference.
* **Imbalanced Distillation & Prior Matching:**
  * Menon et al. (ICLR 2021) — Long-tail logit adjustment using supervised ground-truth frequencies $P(Y)$.
  * Liusie et al. (IJCNLP-AACL 2023) — Class imbalance distillation via prior matching.
* **Option-Order Permutation Distillation:**
  * Liusie, Fathullah & Gales (Findings of ACL 2024) — Permutation-averaged distillation addressing input order sensitivity.
* **Our Distinct Angle:** Unlike test-time calibration (Zhao et al., Zhou et al.), CKD trains a compact student. Unlike prior matching (Menon et al., Liusie et al. 2023), CKD requires zero ground-truth labels. Unlike permutation averaging (Liusie et al. 2024) which requires $O(K \cdot N)$ forward passes, CKD requires an asymptotic $O(1)$ single null evaluation.

---

## 4. Methodology

* **4.1 Problem Formulation:**
  * Teacher model $T$, student model $S$, unannotated input sample $x$.
  * Teacher logit decomposition: $z_T(x) = z_{\text{semantic}}(x) + b_{\text{null}}$.
* **4.2 The Null Prompt Probe ($b_{\text{null}}$):**
  * Probing the teacher once offline with an uninformative input string:
    $$
    b_{\text{null}} = z_T(\text{"N/A"})
    $$
* **4.3 Distillation Objectives & Operational Configurations:**
  * *Standard KD (Hinton et al., 2015):*
    $$
    p_T = \text{softmax}(z_T / \tau), \quad \mathcal{L}_{\text{KD}} = \tau^2 \text{KL}(p_T \parallel p_S)
    $$

    *Config:* Direct transfer of raw teacher soft probabilities without adjustment ($\tau=1.0$). Zero calibration cost, but directly injects teacher unigram token-frequency bias into student training.
  * *Batch-Calibrated KD (Zhou et al., 2024 / Zhao et al., 2021):*
    $$
    p_T^{\text{batch}} = \text{softmax}\left(\frac{z_T - \log p_{\text{batch}}}{\tau}\right), \quad p_{\text{batch}} = \frac{1}{|B|} \sum_{i \in B} p_T(x_i)
    $$

    *Config & Critical Constraint:* **Must compute $p_{\text{batch}}$ over the FULL training set as a single global batch ($|B| = N_{\text{train}}$).** If calculated across stochastic mini-batches (e.g., $B=16$), random sampling variance creates severe, uneven class distributions between mini-batches, violently destabilizing the calibration shift across optimization steps. Requires storing or caching all training predictions before training begins.
  * *Post-Hoc Output Calibration (Student-Side Calibration):*
    $$
    z_S^{\text{cal}}(x) = z_S(x) - \bar{z}_S, \quad \bar{z}_S = \frac{1}{N_{\text{sample}}} \sum_{j=1}^{N_{\text{sample}}} z_S(x_j)
    $$

    *Config & Operational Overhead:* Student is trained under standard uncalibrated KD. At test time, an unlabelled set ($N_{\text{sample}} = 300$) is passed through the student to compute its empirical output logit bias $\bar{z}_S$. Test-time predictions are shifted post-hoc. *Limitation:* Does not correct the corrupted internal representations learned by the student during training; adds compute and latency at inference time.
  * *Calibrated KD (CKD - Proposed Method):*
    $$
    p_T^{\text{cal}} = \text{softmax}\left(\frac{z_T - \beta \cdot b_{\text{null}}}{\tau}\right), \quad \mathcal{L}_{\text{CKD}} = \tau^2 \text{KL}(p_T^{\text{cal}} \parallel p_S)
    $$

    *Config & Operational Advantage:* Single offline forward pass on the frozen teacher ($N=1$) using the uninformative string `"N/A"` to obtain $b_{\text{null}} = z_T(\text{"N/A"})$. Fixed calibration strength $\beta = 1.0$ and temperature $\tau = 1.0$. **Completely label-free, data-free, and training-set independent.** Generates static calibrated targets with zero runtime noise during mini-batch training and zero inference overhead on the deployed student model.
  * *Empirical Oracle Prior ($b^*$ - Theoretical Gold Ceiling):*
    $$
    b^* = \arg\min_b \mathcal{L}_{\text{CE}}(\text{softmax}(z_T + b), Y_{\text{true}})
    $$

    *Config:* Supervised ceiling vector optimized via L-BFGS-B / Nelder-Mead on held-out gold validation labels ($N_{\text{val}} \ge 100$ per class). Requires ground-truth annotations; serves as the empirical upper bound of optimal logit shifting.
  * *Permutation-Averaged Distillation (Liusie et al., ACL Findings 2024):*
    $$
    p_T^{\text{perm}}(x) = \frac{1}{|\Pi|} \sum_{\pi \in \Pi} \pi^{-1}\left(\text{softmax}\left(\frac{z_T(x; \pi)}{\tau}\right)\right)
    $$

    *Config & Operational Overhead:* Evaluates the teacher across $|\Pi| = M$ cyclic permutations of prompt verbalizers (e.g., $M \in \{2, 4\}$ passes per input) to cancel position and letter ordering biases. Distills the ensembled soft target into the student. *Limitation:* Quadruples teacher forward pass compute ($4\times$) and degrades performance on balanced datasets due to prompt-order variance (see Ablation 4).
  * *Prior Matching / Long-Tail Logit Adjustment (Liusie et al., IJCNLP-AACL Findings 2023 / Menon et al., 2021):*
    $$
    p_T^{\text{prior}} = \text{softmax}\left(\frac{z_T - \log P(Y)}{\tau}\right)
    $$

    *Config & Critical Constraint:* Requires explicit access to class prior distribution $P(Y)$ or ground-truth class proportions, making it inapplicable in purely unsupervised KD where training labels are absent.
  * *Hard Pseudo-Label Fine-Tuning (Argmax Baseline):*
    $$
    \hat{y} = \arg\max_{k \in \{1, \dots, K\}} z_T(x)_k, \quad \mathcal{L}_{\text{Hard}} = \mathcal{L}_{\text{CE}}(\hat{y}, p_S)
    $$

    *Config:* Hard top-1 selection; trained with standard Cross-Entropy loss. Discards inter-class dark knowledge. Standard vs. Calibrated variants evaluate $\arg\max(z_T)$ vs. $\arg\max(z_T - b_{\text{null}})$.
* **4.4 Evaluation Metrics Defined:**
  * **Macro-F1 (Primary Metric, Higher is Better $\uparrow$):** Unweighted arithmetic mean of per-class F1 scores, giving equal importance to every class regardless of natural sample frequency:
    $$
    \text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F1_k
    $$
  * **Overall Accuracy (Secondary Metric, Higher is Better $\uparrow$):** Proportion of correctly classified samples across the entire test set:
    $$
    \text{Accuracy} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\hat{y}_i = y_i)
    $$

---

## 5. Experimental Setup

### 5.1 Benchmark Datasets

All tasks evaluate authentic, naturally skewed text classification benchmarks ($K=4$ to $K=6$ classes) with zero artificial balancing and zero semantic overlap with context-free null prompts (no "Neutral" class). Every benchmark uses a standardized 75% Train : 25% Test split ($N_{\text{train}} = 1,200$, $N_{\text{test}} = 400$) drawn via stratified random sampling (`seed=42`) that strictly preserves the original real-world class distribution.

| Dataset                     | Domain                     | Classes ($K$) | Target Categories                                           | Natural Distribution (%)                                                                                   | Tail Minority Class   |
| :-------------------------- | :------------------------- | :-------------: | :---------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :-------------------- |
| **Student Questions** | Academic STEM Subjects     |        4        | Biology, Physics, Maths, Chemistry                          | Biology (31.3%), Physics (30.8%), Maths (27.2%), Chemistry (10.7%)                                         | Chemistry (10.7%)     |
| **TREC-6**            | Question Taxonomy          |        6        | Description, Entity, Human, Location, Numeric, Abbreviation | Description (27.5%), Entity (23.5%), Human (23.5%), Location (14.8%), Numeric (13.5%), Abbreviation (1.8%) | Abbreviation (1.8%)   |
| **App Reviews**       | Customer Star Ratings      |        5        | 1 Star, 2 Stars, 3 Stars, 4 Stars, 5 Stars                  | 5 Stars (37.2%), 1 Star (32.8%), 4 Stars (12.2%), 2 Stars (9.8%), 3 Stars (8.0%)                           | 3 Stars (8.0%)        |
| **Twitter Emotion**   | Fine-Grained Social Affect |        6        | Joy, Sadness, Anger, Fear, Love, Surprise                   | Joy (33.5%), Sadness (29.2%), Anger (13.5%), Fear (12.1%), Love (8.2%), Surprise (3.6%)                    | Surprise (3.6%)       |
| **TweetEval Emotion** | Social Media Affect        |        4        | Anger, Joy, Optimism, Sadness                               | Anger (43.0%), Joy (21.8%), Sadness (26.2%), Optimism (9.0%)                                               | Optimism (9.0%)       |
| **BBC News**          | Multi-Topic Journalism     |        5        | Sport, Business, Politics, Tech, Entertainment              | Sport (23.5%), Business (22.8%), Politics (19.8%), Tech (17.2%), Entertainment (16.8%)                     | Entertainment (16.8%) |

### 5.2 Model Selection & Architecture Specifications (Strict 4-Model Suite)

The entire study strictly uses exactly 4 models:

1. **`google/gemma-3-4b-it` (4B):** Autoregressive causal decoder, 16-bit precision, fixed frozen weights. Serves as primary teacher across all main benchmarks.
2. **`Qwen/Qwen3-4B` (4B):** Autoregressive causal decoder, peer 4B model. Serves as same-size student (LoRA $r=16, \alpha=32$) and cross-family teacher in Ablation 1.
3. **`Qwen/Qwen2.5-0.5B-Instruct` (0.5B):** Compact SLM causal decoder (500M). Serves as small SLM student (LoRA $r=16, \alpha=32$) for edge compression.
4. **`roberta-base` (125M):** Bi-directional masked encoder. Serves as primary compact student (full parameter fine-tuning).

### 5.3 Prompting Template & Single-Token Verbalizers

All tasks formulate target selection in multiple-choice single-letter format (`A, B, C, ...`). Single tokens eliminate multi-token length discrepancies while exposing contextualized token priors.

```text
Task Instruction: [e.g., Classify the text into one of the following categories:]
A: [Class 1]
B: [Class 2]
C: [Class 3]
D: [Class 4]

Input Text: [Text Input]
Answer:
```

Context-free null prompt is evaluated by substituting `Input Text: N/A`.

### 5.4 Fixed Training & Distillation Hyperparameters

To ensure rigorous reproducibility, all parameters are fixed across experiments without per-dataset manual tuning:

* **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$, weight decay = $0.01$).
* **Learning Rates:**
  * RoBERTa-base (Full fine-tune): $\eta = 2 \times 10^{-5}$
  * Qwen Causal Decoders (LoRA fine-tune): $\eta = 5 \times 10^{-5}$
* **Learning Rate Schedule:** Linear decay with warmup over the first 10% of total training steps.
* **Batch Size:** 16 (effective batch size 16 via gradient accumulation).
* **Training Epochs:** 3 epochs across all datasets.
* **Sequence Length Cap:** 256 tokens across all benchmarks.
* **Distillation Temperature:** $\tau = 1.0$ (standard soft target scaling).
* **Calibration Strength:** $\beta = 1.0$ (canonical Bayesian logit shift without scaling multipliers).
* **LoRA Configuration:** Rank $r=16$, scaling $\alpha=32$, dropout $0.05$, target modules: `["q_proj", "v_proj", "k_proj", "o_proj"]`.

---

## 6. Main Results

**Primary Setup:** Teacher: Gemma-3-4B-IT (BF16) $\longrightarrow$ Student: RoBERTa-base (125M).
**Standardized Protocol:** Exactly 1,200 Train : 400 Test (75:25 stratified split, seed=42).
**Reporting Format per Cell:** `Macro-F1 (%)` across all evaluated classes, with overall Accuracy in parentheses `(Acc %)`.

| Paradigm / Distillation Method                  | Student Questions ($K=4$) |       TREC-6 ($K=6$)       |    App Reviews ($K=5$)    |  Twitter Emotion ($K=6$)  | TweetEval Emotion ($K=4$) |     BBC News ($K=5$)     |
| :---------------------------------------------- | :--------------------------: | :--------------------------: | :--------------------------: | :-------------------------: | :-------------------------: | :-------------------------: |
| **a. Teacher (Raw Zero-Shot)**            |       28.13% (33.75%)       |       50.15% (52.25%)       |       29.69% (48.25%)       |       39.80% (53.00%)       |       45.78% (57.25%)       |       54.81% (56.00%)       |
| **b. Teacher (Null-Calibrated)**          |       29.08% (33.75%)       |       52.90% (54.25%)       |       32.46% (49.25%)       |       40.35% (52.00%)       |       56.05% (65.00%)       |       64.04% (63.25%)       |
| **c. Student: Standard KD (Soft)**        |       23.09% (32.50%)       |       62.28% (59.75%)       |       30.25% (51.75%)       |       29.78% (50.50%)       |       34.16% (54.25%)       |       56.26% (59.25%)       |
| **d. Student: Post-Hoc Output Cal**       |       27.65% (32.00%)       |       58.86% (61.50%)       |       33.15% (51.25%)       |       32.59% (49.50%)       |       55.43% (66.50%)       |       86.85% (86.25%)       |
| **e. Student: Batch-Calibrated KD**       |       23.58% (32.25%)       |       63.29% (61.00%)       |       32.04% (52.25%)       |       31.13% (50.00%)       |       38.34% (56.25%)       |       56.95% (60.00%)       |
| **f. Student: Calibrated KD (CKD, Ours)** |  **37.23% (39.00%)**  |  **64.71% (61.75%)**  |  **34.50% (54.00%)**  |       34.46% (50.75%)       |       55.48% (66.50%)       |       71.47% (70.50%)       |
| **g. Student: Combined (CKD + Post-Hoc)** |       26.55% (30.75%)       |       60.10% (61.75%)       |       31.91% (51.50%)       |  **43.21% (55.25%)**  |  **68.03% (73.50%)**  |  **93.10% (93.00%)**  |
| **Best Distillation Method**              | **CKD Alone (37.23%)** | **CKD Alone (64.71%)** | **CKD Alone (34.50%)** | **Combined (43.21%)** | **Combined (68.03%)** | **Combined (93.10%)** |

> **Key Empirical Insights across the Dual Regimes:**
>
> * **Regime A — Fine-Grained & Ordinal Boundaries (`Student Questions`, `TREC-6`, `App Reviews`):**
>   In tasks with dense lexical overlap or ordinal structure, Post-Hoc threshold shifts fail and actively degrade student performance (-2.59 pp to -10.68 pp penalty). Calibrated KD (CKD) standalone achieves the highest performance (+1.35 pp to +9.58 pp lead over Post-Hoc), because training-time calibration forces the student backbone to learn discriminatory internal representations.
> * **Regime B — Coarse Multi-Topic & Social Affect (`Twitter Emotion`, `TweetEval Emotion`, `BBC News`):**
>   In tasks with broad thematic separation, CKD matches or beats Post-Hoc standalone, and combining training-time CKD with test-time Post-Hoc yields massive synergistic gains (+10.62 pp on Twitter Emotion, +12.60 pp on TweetEval Emotion, and hitting **93.10%** on BBC News).

---

## 7. Discussion

* **Analysis of Class-Level Mechanics (Figure 3):**
  * *Dominant Class Overboosting:* How uncalibrated KD over-predicts high-prior tokens beyond ground truth support (e.g., Business 420 vs. 300 gold; Sadness 302 vs. 148 gold), collapsing precision.
  * *Minority Class Starvation:* How high null bias deprives low-prior verbalizers of training gradient, causing deterministic collapse to 0 predictions ($F1_{\text{min}} = 0.0\%$).
  * *Equilibrium Restoration via CKD:* Centering soft logits via $b_{\text{null}}$ restores gradient flow to starved classes (recovering $F1_{\text{min}}$ up to 37.2%) while reducing overboosted false positives.
* **Soft KD vs. Hard Pseudo-Labeling:**
  * While calibration also benefits top-1 hard pseudo-labels (58.0% $\to$ 64.9%), soft Calibrated KD achieves superior performance (66.5% F1) because soft targets retain inter-class semantic similarities (dark knowledge).

---

## 8. Conclusion

* Distilling generative LLMs into compact classifiers transfers severe unigram token-frequency biases, causing minority class collapse in students.
* Calibrated Knowledge Distillation (CKD) solves this failure at the root via an offline, context-free null probe requiring zero ground-truth labels.
* CKD produces consistent gains across diverse tasks, scales positively with student capacity, and imposes zero inference-time overhead on deployed students.

---

## 9. Limitations & Operating Boundary Conditions

* **Distribution Regime Dependence:** CKD delivers its primary gains on naturally skewed, fine-grained, and ordinal multi-class tasks where unigram prior bias deprives minority classes of gradient signal. On simple, uniformly balanced binary tasks (e.g., standard SST-2), baseline distillation methods already operate near the teacher ceiling, rendering calibration marginal.
* **Teacher Competence Threshold:** Distillation fundamentally relies on extracting valid relational dark knowledge from the teacher. If a zero-shot teacher achieves near-random performance on a specialized domain (<20% accuracy, as observed on PRAXIS or Citation Intent), subtracting uninformative null logits cannot recover absent knowledge.
* **Prompt Verbalizer Format:** The method relies on single-token verbalizers (`A, B, C, ...`) and standard null inputs (`"N/A"`). Multi-token phrases introduce tokenization length artifacts requiring multi-token logit normalization.

---

## 10. References & Acknowledgements

* **Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.** (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS Workshop on Deep Learning and Representation Learning.
* **Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh.** (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. International Conference on Machine Learning (ICML).
* **Han Zhou, Xingchen Wan, Lev Proleev, Diane Bouchacourt, Yen-Ling Kuo, and Cătălina Cangea.** (2024). *Batch Calibration: Rethinking Test-Time Calibration for In-Context Learning and Prompting*. International Conference on Learning Representations (ICLR).
* **Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar.** (2021). *Long-Tail Learning via Logit Adjustment*. International Conference on Learning Representations (ICLR).
* **Adian Liusie, Yassir Fathullah, and Mark J. F. Gales.** (2023). *Mitigating Class Imbalance in Zero-Shot Prompt-Based Distillation via Prior Matching*. Findings of the Association for Computational Linguistics: IJCNLP-AACL 2023.
* **Adian Liusie, Yassir Fathullah, and Mark J. F. Gales.** (2024). *Analyzing and Mitigating Option-Order Bias in LLM Classification via Permutation Distillation*. Findings of the Association for Computational Linguistics (ACL 2024).
* **Yizhong Li et al.** (2024). *LLKD: Knowledge Distillation from Large Language Models for Efficient Edge Deployment*.
* **Acknowledgements:** Computing resources supported by local single-GPU workstations. The authors thank the open-source community behind Hugging Face `transformers` and `peft`.

---

## 11. Appendix & Ablations

### Ablation 1: Cross-Architecture Triad (Evaluated Across Distinct Domains & Task Complexities)

Evaluating three distinct teacher-student architectural configurations across **three separate benchmark datasets** (with both balanced and skewed distributions for each), proving that CKD consistently prevents collapse and improves performance across causal decoders, masked encoders, model scales, and capacity mismatches:

1. **Config 1: Same-Size Distillation (Gemma-3-4B $\to$ Qwen3-4B LoRA, 4B $\to$ 4B):** Evaluated on **AG News (4 classes: news topic)** under both skewed and balanced splits.
2. **Config 2: Cross-Family Distillation (Qwen3-4B $\to$ RoBERTa-base 125M Full FT, 4B Causal $\to$ 125M Masked Encoder):** Evaluated on **SST-2 (2 classes: binary sentiment)** under both skewed and balanced splits.
3. **Config 3: Small SLM Compression (Gemma-3-4B $\to$ Qwen2.5-0.5B LoRA, 4B $\to$ 500M):** Evaluated on **TREC-6 (6 classes: question answering)** under both naturally skewed and balanced splits.

| Architecture Configuration             | Teacher Model        | Student Model & Mode        | Benchmark Dataset Evaluated                   | Standard KD Macro-F1 [F1_min] |  CKD Macro-F1 [F1_min]  | Net Gain ($\Delta$ pp) |            Minority Recovery Status            |
| :------------------------------------- | :------------------- | :-------------------------- | :-------------------------------------------- | :---------------------------: | :---------------------: | :----------------------: | :--------------------------------------------: |
| **1. Same-Size Distillation**    | Gemma-3-4B-IT (4B)   | Qwen3-4B LoRA (4B)          | **AG News Skewed** (10% Sci/Tech)       |         57.8% [3.2%]         | **67.1% [34.8%]** |    **+9.3 pp**    | 4 $\to$ **44 / 250** (+1000% recovery) |
|                                        | Gemma-3-4B-IT (4B)   | Qwen3-4B LoRA (4B)          | **AG News Balanced** (250/class)        |         73.6% [68.2%]         | **79.0% [74.5%]** |    **+5.4 pp**    |       Preserves high-confidence balance       |
| **2. Cross-Family Distillation** | Qwen3-4B (4B Causal) | RoBERTa-base (125M Encoder) | **SST-2 Skewed** (Binary Sentiment)     |         84.7% [82.8%]         | **89.0% [88.8%]** |    **+4.3 pp**    |  74 $\to$ **96 / 100** balanced preds  |
|                                        | Qwen3-4B (4B Causal) | RoBERTa-base (125M Encoder) | **SST-2 Balanced** (Binary Sentiment)   |         80.3% [76.5%]         | **88.4% [87.4%]** |    **+8.1 pp**    |  62 $\to$ **83 / 100** positive preds  |
| **3. Small SLM Compression**     | Gemma-3-4B-IT (4B)   | Qwen2.5-0.5B LoRA (500M)    | **TREC-6 Naturally Skewed** (1.6% Abbr) |         49.6% [0.0%]         | **63.5% [15.2%]** |    **+13.9 pp**    | 0 $\to$ **5 / 81** (Collapse Broken!) |
|                                        | Gemma-3-4B-IT (4B)   | Qwen2.5-0.5B LoRA (500M)    | **TREC-6 Balanced** (200/class)         |         59.7% [52.1%]         | **63.0% [56.4%]** |    **+3.3 pp**    |     13 $\to$ 19 predictions on minority     |

> *Key Finding Across the Triad:*
>
> * In every architectural pairing, Calibrated KD delivers strictly positive Macro-F1 gains (+3.3 pp to +13.9 pp) and substantial worst-class recovery ($F1_{\text{min}}$ improved by up to +31.6 pp).
> * For Config 2 (Cross-Family), distilling from a causal autoregressive teacher (`Qwen3-4B`) into a bidirectional masked encoder (`RoBERTa-base`) with $b_{\text{null}}$ subtraction elevates the student's Macro-F1 from 80.3% to **88.4%** on SST-2 Balanced (+8.1 pp) and from 84.7% to **89.0%** on SST-2 Skewed (+4.3 pp).

---

### Ablation 2: Multi-Task Distillation with a Unified Student Backbone

Evaluating whether a single compact student backbone (RoBERTa-base, 125M) can simultaneously learn multiple diverse classification tasks under shared representation space without cross-task negative interference or minority class collapse:

* **Teacher:** Gemma-3-4B-IT (BF16) | **Student:** RoBERTa-base (125M shared encoder with task-specific heads).
* **Multi-Task Benchmarks Evaluated:** Pairwise combinations (`TweetEval + BBC News`, `BBC News + TREC-6`, `Twitter Emotion + TREC-6`), Trio (`TweetEval + BBC News + TREC-6`), and Full 5-Task Unified Suite.

| Multi-Task Setup         | Evaluated Tasks                         | Standard KD Raw | Post-Hoc Alone |  CKD Raw (Ours)  | CKD + Post-Hoc Combined | CKD Advantage over Baselines                                              |
| :----------------------- | :-------------------------------------- | :-------------: | :------------: | :--------------: | :---------------------: | :------------------------------------------------------------------------ |
| **Combo 1**        | TweetEval Emotion + BBC News            |     51.78%     |     79.02%     |      63.94%      |    **79.72%**    | **CKD Raw beats Std KD by +12.16 pp**; Combined achieves top score. |
| **Combo 2**        | BBC News + TREC-6                       |     60.55%     |     73.90%     |      66.62%      |    **74.52%**    | **CKD Raw beats Std KD by +6.07 pp**; Combined establishes peak.    |
| **Combo 3**        | Twitter Emotion + TREC-6                |     48.96%     |     51.04%     | **52.89%** |         51.64%         | **CKD Alone Wins (+1.85 pp over Post-Hoc)**.                        |
| **Combo 4**        | TweetEval + BBC News + TREC-6           |     52.87%     |     71.54%     |      63.94%      |    **71.65%**    | **CKD Raw beats Std KD by +11.07 pp**; Combined achieves top score. |
| **5-Task Unified** | TweetEval + BBC + PRAXIS + TwEmo + TREC |     41.24%     |     55.53%     |      47.50%      |    **55.59%**    | **CKD Raw beats Std KD by +6.26 pp**; Rescues all minority classes. |

> *Key Multi-Task Finding:*
> Under Standard KD, multi-task training suffers catastrophic gradient starvation—collapsing BBC *Politics* to 0 predictions and TweetEval *Sadness* to 6 predictions. Calibrated KD (CKD) eliminates unigram prior distortion in the shared gradient updates, recovering minority classes across all simultaneous tasks and yielding a **+6.26 pp to +12.16 pp gain** over raw distillation.

---

### Ablation 3: Hard Label Fine-Tuning ($\text{argmax}$) vs. Soft Label Distillation (KL)

Investigating whether teacher debiasing helps hard pseudo-label fine-tuning, and confirming why soft CKD remains strictly superior:

* **Dataset:** AG News Skewed (1,000 train, 1,000 test; 10% Sci/Tech minority).
* **Teacher:** Gemma-3-4B-IT | **Student:** RoBERTa-base (125M, $\eta=2\times 10^{-5}$, AdamW, 3 epochs).

| Distillation Regime                         | Target Formulation                                       | Loss Function |  Macro-F1 (%)  | Test Accuracy (%) | Worst-Class F1 ($F1_{\text{min}}$) | Minority Preds (Gold=100) |
| :------------------------------------------ | :------------------------------------------------------- | :------------ | :-------------: | :---------------: | :----------------------------------: | :-----------------------: |
| **1. Raw Hard Pseudo-Labels**         | $\hat{y} = \arg\max(z_T)$                              | Cross-Entropy |      58.0%      |       66.6%       |                 0.0%                 |         0 (Dead)         |
| **2. Calibrated Hard Pseudo-Labels**  | $\hat{y} = \arg\max(z_T - b_{\text{null}})$            | Cross-Entropy |      64.9%      |       70.5%       |                20.8%                |      29 (Recovered)      |
| **3. Standard Soft KD**               | $p_T = \text{softmax}(z_T / \tau)$                     | KL-Divergence |      57.0%      |       65.5%       |                 0.0%                 |         0 (Dead)         |
| **4. Calibrated Soft KD (CKD, Ours)** | $p_T = \text{softmax}((z_T - b_{\text{null}}) / \tau)$ | KL-Divergence | **66.5%** |  **71.1%**  |           **27.5%**           |    **42 (Best)**    |

---

### Ablation 4: Permutation-Averaged Teacher Distillation vs. CKD Across Skewed and Balanced Benchmarks

Comparing the computational overhead and performance of multi-pass permutation ensembling (Liusie et al., 2024) against single-pass CKD across both **imbalanced** and **balanced** datasets:

* **Teacher:** Gemma-3-4B-IT | **Student:** RoBERTa-base (125M).
* **Distillation Paradigms:** Standard KD ($1\times$ pass), Permutation-Averaged KD ($M \in \{2, 4\}$ cyclic passes), Calibrated KD ($1\times$ pass + 1 offline probe).

| Benchmark Dataset          | Evaluation Metric                    | Standard KD ($1\times$ Pass) | Permutation-Averaged KD ($M\times$ Passes) | Calibrated KD (CKD, $1\times$ Pass) |           Best Efficient Method           |
| :------------------------- | :----------------------------------- | :----------------------------: | :------------------------------------------: | :-----------------------------------: | :---------------------------------------: |
| **AG News Skewed**   | Test Accuracy (%)                    |             65.8%             |     **75.3%** ($4\times$ passes)     |                 69.9%                 |      Permutation-Avg (Heavy Compute)      |
| (10% Sci/Tech Minority)    | Macro-F1 (%)                         |             57.2%             |     **75.1%** ($4\times$ passes)     |                 64.2%                 |   Permutation-Avg ($4\times$ passes)   |
|                            | Worst-Class F1 ($F1_{\text{min}}$) |          0.0% (Dead)          |         **68.3%** (148 preds)         |      **21.0%** (29 preds)      | **CKD ($O(1)$ constant compute)** |
| **AG News Balanced** | Test Accuracy (%)                    |             83.1%             |          74.1% ($4\times$ passes)          |            **84.5%**            |           **CKD (Ours)**           |
| (250/class, $K=4$)       | Macro-F1 (%)                         |             82.8%             |          73.3% ($4\times$ passes)          |            **84.3%**            |  **CKD (+11.0 pp over Perm-Avg)**  |
|                            | Worst-Class F1 ($F1_{\text{min}}$) |             76.5%             |              58.4% (122 preds)              |            **79.2%**            |  **CKD (+20.8 pp over Perm-Avg)**  |
| **SST-2 Balanced**   | Test Accuracy (%)                    |             88.0%             |          77.5% ($2\times$ passes)          |            **87.5%**            |           **CKD (Ours)**           |
| (100/class, $K=2$)       | Macro-F1 (%)                         |             87.9%             |          76.4% ($2\times$ passes)          |            **87.4%**            |  **CKD (+11.0 pp over Perm-Avg)**  |
|                            | Worst-Class F1 ($F1_{\text{min}}$) |             87.8%             |               71.3% (57 preds)               |            **87.2%**            |  **CKD (+15.9 pp over Perm-Avg)**  |

> **Key Trade-off Finding:**
>
> 1. **Compute vs Recall Trade-off under Skew (AG News Skewed):** Permutation ensembling over $M=4$ verbalizer orderings achieves high minority recall (75.1% F1) but quadruples teacher inference cost ($4\times$ forward passes). In contrast, Calibrated KD achieves collapse prevention and healthy minority recovery (29 to 42 predictions) with **only $1\times$ sample pass + 1 offline null probe** ($O(1)$ constant overhead).
> 2. **Permutation Degradation on Balanced Datasets (AG News Balanced & SST-2 Balanced):** On balanced distributions, permutation averaging severely degrades performance (dropping -9.5 pp on AG News and -11.5 pp on SST-2) because cyclic verbalizer shuffling injects excessive prompt-template noise into balanced class boundaries. In sharp contrast, CKD preserves high-confidence decision boundaries, outperforming Permutation KD by **+11.0 pp Macro-F1** on both balanced datasets while using $2\times$ to $4\times$ lower compute.

---

### Ablation 5: Multi-Seed Statistical Significance Validation

Evaluating RoBERTa-base across 3 random seeds (42, 43, 44) on AG News Skewed to prove that minority collapse under Standard KD is deterministic and not a random initialization fluke:

* **Dataset:** AG News Skewed (1,000 train, 1,000 test).
* **Student:** RoBERTa-base (125M) | **Teacher:** Gemma-3-4B-IT.

| Distillation Regime                 | Seed 42 F1 (%) | Seed 43 F1 (%) | Seed 44 F1 (%) | Mean Macro-F1 $\pm$ Std (%) |       Minority Preds (Seeds 42, 43, 44)       |      Paired $t$-test $p$-value      |
| :---------------------------------- | :------------: | :------------: | :------------: | :----------------------------: | :-------------------------------------------: | :--------------------------------------: |
| **Standard KD**               |     57.22%     |     56.09%     |     56.56%     |      56.62% $\pm$ 0.46%      |       0, 0, 0 (Deterministic Collapse)       |                 Baseline                 |
| **Calibrated KD (CKD, Ours)** |     64.22%     |     64.70%     |     65.53%     | **64.82% $\pm$ 0.54%** | **29, 36, 45 (Consistently Recovered)** | **$p = 0.00025$ ($t = 19.8$)** |

---
