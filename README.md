# Calibrated Knowledge Distillation (CKD)

**Official empirical repository for Calibrated Knowledge Distillation (CKD) and Prior-Invariant Knowledge Distillation (PI-KD).**

---

## 🔬 Core Premise & Methodology

Large Language Model teachers exhibit sharp **unigram token prior skews** ($W_{\text{null}}$) even under prompt-guided in-context learning. When transferring knowledge to a smaller student via standard Knowledge Distillation (Forward KL), this unigram bias gets transferred into the training soft targets, causing:
1. **Severe class starvation & collapse** for lower-probability minority classes.
2. **Artificial degradation** of student accuracy and Macro-F1.

### The Formulation
For teacher logits $z(x)$ and null vector $W_{\text{null}} = z(\text{null\_prompt})$:
$$\tilde{z}(x) = z(x) - W_{\text{null}}$$
$$P_{\text{cal}}(y \mid x) = \text{Softmax}(\tilde{z}(x))$$
$$\mathcal{L}_{\text{CKD}} = D_{\text{KL}}\!\left(P_{\text{cal}}(y \mid x) \,\|\, P_{\text{student}}(y \mid x)\right)$$

---

## 📁 Repository Structure

```
calibrated-kd/
├── README.md                          # Repository documentation & reproduction
├── .gitignore                         # Standard git ignore rules
├── data/
│   ├── benchmarks/                    # Clean gold-standard splits (CSV / Parquet)
│   └── teacher_targets/               # Extracted teacher logits & null vectors (.npz)
├── scripts/
│   ├── prepare_data/                  # Split extraction & diagnostic null-bias checkers
│   ├── cache_targets/                 # Teacher inference & null-vector calibration caching
│   ├── train/                         # Standalone student distillation runners (85% VRAM cap)
│   └── plotting/                      # Dedicated 300 DPI publication figure generators
├── results/
│   ├── agnews/                        # Evaluation metrics for AG News 4-way ablation
│   ├── emotion/                       # Evaluation metrics for Emotion 4-way ablation
│   └── sst2/                          # Evaluation metrics for SST-2 zero-harm control
├── figures/
│   ├── agnews/                        # Dedicated standalone figures for AG News
│   └── emotion/                       # Dedicated standalone figures for Emotion
├── logs/                              # Full execution logs of all training runs
└── docs/                              # Master execution plan & formal research report
```

---

## 📊 Summary of Empirical Results

**Fixed Setup:** Teacher = `google/gemma-3-4b-it` (BF16) $\to$ Student = `Qwen/Qwen3-4B` (LoRA, BF16).

### 1. AG News (4-Class Topic Classification, $N = 1,000$)
* **Prior Skew:** $\Delta_{\text{null}} = 2.84\text{ logits}$
* **Standard KD (Old):** 73.10% Acc | 69.91% Macro-F1 (`Sci/Tech` starved to 62 / 250)
* **Calibrated KD (New):** **76.50% Acc (+3.40%)** | **74.40% Macro-F1 (+4.49%)** (`Sci/Tech` rescued to 89 / 250, **+43.5%**)

### 2. Emotion (6-Class Affect Classification, $N = 300$)
* **Prior Skew:** $\Delta_{\text{null}} = 7.31\text{ logits}$ (Sadness prior = 98.24%, 1,498.9× skew)
* **Standard KD (Old):** 40.67% Acc | 37.59% Macro-F1 (**`Surprise` completely wiped out: 0 / 50 DEAD**)
* **Calibrated KD (New):** **43.33% Acc (+2.66%)** | **39.03% Macro-F1 (+1.44%)** (Sadness cut in half: 152 $\to$ 75; **Surprise revived**)

---

## 🚀 Reproduction Quickstart

```bash
# 1. Environment
source /home/vishrut/annotator-bias-fingerprints/.venv/bin/activate

# 2. Run AG News Distillation
python scripts/train/train_agnews.py --variant static_ckd

# 3. Run Emotion Distillation
python scripts/train/train_emotion.py --variant static_ckd

# 4. Generate Dedicated Figures
python scripts/plotting/plot_agnews.py
python scripts/plotting/plot_emotion.py
```
