# 📋 PROJECT HANDOVER & CONTEXT SPECIFICATION

> **Target Audience:** Any incoming AI agent or researcher working in `/home/vishrut/calibrated-kd`.
> **Last Updated:** October 31, 2026
> **Core Theme:** Null-Conditioned Verbalizer Calibration for Autoregressive Knowledge Distillation

---

## 1. Executive Summary & Research Goal

This repository investigates **Null-Conditioned Verbalizer Calibration in Autoregressive Knowledge Distillation**.

### Core Problem
Autoregressive LLM teachers exhibit severe **null-conditioned verbalizer biases** ($W_0$) across candidate output tokens, even when prompted in-context. Under standard distillation, this teacher prompt preference transfers into the training soft targets, causing:
1. **Suppression / starvation** on classes whose verbalizers have lower null-prompt likelihood.
2. **Artificial degradation** of distilled student accuracy and Macro-F1.

### The Method: Static Verbalizer-Calibrated KD
We extract the teacher's response across candidate verbalizers to content-free null prompts as $W_0$, and calibrate the teacher logits before computing soft target probabilities:
$$\tilde{z}(x) = z(x) - \beta W_0$$
$$P_{\text{cal}}(y \mid x) = \text{Softmax}(\tilde{z}(x))$$
$$\mathcal{L} = (1 - \lambda)\mathcal{L}_{\text{CE}} + \lambda D_{\text{KL}}\!\left(P_{\text{cal}}(y \mid x) \,\|\, P_{\text{student}}(y \mid x)\right)$$

### ⚠️ Methodological Pruning & Scope
* **Pruned Methods:** Failed entropy-adaptive gating (Ada-CKD) and null-invariance regularizer (PI-KD) have been dropped. The core focus is strictly on robust **Static Verbalizer Calibration**.
* **Dropped Datasets:** PRAXIS has been permanently dropped due to noisy subjective translations.
* **Focus Benchmarks:** High-agreement gold-standard NLP benchmarks (AG News, Emotion, TREC-6, SST-2).

---

## 2. Experimental Setup (Fixed Across All Runs)

To ensure scientific validity, the model pair and hyperparameter regime are fixed:

| Role | Model | Precision | Location / Source | Notes |
| :--- | :--- | :---: | :--- | :--- |
| **Teacher** | `google/gemma-3-4b-it` | BF16 | HF cache snapshot (`models--google--gemma-3-4b-it`) | Direct inference & target extractor |
| **Student** | `Qwen/Qwen3-4B` | BF16 | HF cache snapshot (`models--Qwen--Qwen3-4B`) | LoRA ($r=16, \alpha=32$, dropout=0.05) |

### Hardware & Environment Constraints
* **Platform:** Linux / WSL2 on Lenovo Legion Laptop (RTX 5070 Ti, 12GB VRAM, up to 140W TGP).
* **Python Environment:** `/home/vishrut/annotator-bias-fingerprints/.venv/bin/python`.
* **VRAM Headroom Guard (MANDATORY):**
  Always include `torch.cuda.set_per_process_memory_fraction(0.85, device=0)` in training scripts. WSL2 requires 2+ GB free for Windows Desktop Window Manager (`dwm.exe`). Exceeding this causes DirectX kernel paging freezes (`dxgkio_make_resident: -12`).
* **Process Isolation:**
  Never run multiple distillation passes inside a single continuous Python loop. Always launch isolated scripts (`python scripts/train/train_<dataset>.py --variant <variant>`) so memory is 100% released on exit.
* **Evaluation Batch Size:**
  Keep `batch_size = 4` during student test evaluation. `Qwen3-4B` has a 151,646 vocabulary size; large batch sizes trigger OOM on the final classification projection tensor.

---

## 3. Strict User Directives & Formatting Rules

1. **NEVER MIX BENCHMARKS:**
   Treat each dataset completely in isolation. Do not combine metrics or predictions from different datasets into one diagram or table unless explicitly asked.
2. **DIAGRAM RULES (Old vs. New):**
   * Every figure must explicitly contrast **Old (Standard KD)** vs. **New (Calibrated KD)** side-by-side.
   * Color palette: Coral Red (`#d9534f`) for Old, Teal (`#2a9d8f`) for New, Slate Gray (`#4a5568`) for Ground Truth.
   * Annotate exact values on bars and highlight recovery deltas with callout boxes.
   * Generate multiple focused, separate figures (300 DPI) rather than cramming everything into one cluttered plot.
3. **STEPWISE EXECUTION:**
   Execute one step at a time, report the results, and **pause for instructions** before moving to the next.

---

## 4. Current Benchmark Status

### Benchmark 1: AG News (4-Class Topic Classification) — ✅ COMPLETE
* **Splits:** $N_{\text{train}} = 1,000$ (250/class) | $N_{\text{test}} = 1,000$ (250/class)
* **Classes:** `[World, Sports, Business, Sci/Tech]`
* **Null Skew:** $\Delta_{\text{null}} = 2.84\text{ logits}$
* **Results:**
  * Standard KD (Old): 73.10% Acc | 69.91% Macro-F1 (`Sci/Tech` starved to 62 / 250)
  * Static CKD (New): **76.50% Acc (+3.40%)** | **74.40% Macro-F1 (+4.49%)** (`Sci/Tech` rescued to 89 / 250, **+43.5% gain**)
* **Dedicated Figures:** `figures/agnews/` (`fig1` to `fig4`).

### Benchmark 2: Emotion (6-Class Affect Classification) — ✅ COMPLETE
* **Splits:** $N_{\text{train}} = 1,200$ (200/class) | $N_{\text{test}} = 300$ (50/class)
* **Classes:** `[Sadness, Joy, Love, Anger, Fear, Surprise]`
* **Null Skew:** $\Delta_{\text{null}} = 7.31\text{ logits}$ (Sadness = 98.24% prior; 1,498.9× skew)
* **Results:**
  * Standard KD (Old): 40.67% Acc | 37.59% Macro-F1 (**`Surprise` completely collapsed: 0 / 50 DEAD**)
  * Static CKD (New): **43.33% Acc (+2.66%)** | **39.03% Macro-F1 (+1.44%)** (Sadness cut in half: 152 $\to$ 75; **Surprise revived**)
* **Dedicated Figures:** `figures/emotion/` (`fig1` to `fig4`).

### Benchmark 3: SST-2 (2-Class Sentiment Classification) — ✅ COMPLETE
* **Splits:** $N_{\text{train}} = 1,000$ | $N_{\text{test}} = 200$ balanced
* **Classes:** `[Negative, Positive]`
* **Null Skew:** $\Delta_{\text{null}} = 1.25\text{ logits}$ (Minimal skew)
* **Purpose:** **Zero-Harm Safety Control** (proves calibration does not harm tasks with low prior bias).
* **Results:**
  * Standard KD: 81.50% Acc | 80.92% Macro-F1 (`[135, 65]`)
  * Calibrated KD: 81.50% Acc | 80.92% Macro-F1 (`[135, 65]`)
  * Exactly **0.00% delta** (zero distortion / perfect stability).
* **Dedicated Figures:** `figures/sst2/` (`fig1` to `fig4`).

### Benchmark 4: TREC-6 (6-Class Question Classification) — ✅ COMPLETE
* **Splits:** $N_{\text{train}} = 1,200$ (200/class) | $N_{\text{test}} = 500$ (Canonical test set)
* **Classes:** `[Abbreviation, Entity, Description, Human, Location, Numeric]`
* **Null Skew:** $\Delta_{\text{null}} = 7.97\text{ logits}$ (`Description` monopolizes 98.78% of null prior; 2,889.2× ratio vs `Numeric`)
* **Results:**
  * Standard KD (Old): 78.20% Acc | 75.08% Macro-F1 (`Description` bloated to 178 vs 138 truth; `Entity` starved to 48 vs 94 truth)
  * Static CKD (New): **80.00% Acc (+1.80%)** | **77.12% Macro-F1 (+2.04%)** (`Entity` rescued to 58, **+20.8% gain**; `Description` suppressed to 159, **-10.7% closer to truth**)
* **Dedicated Figures:** `figures/trec/` (`fig1` to `fig4`).

---

## 5. Repository Map

```
/home/vishrut/calibrated-kd/
├── HANDOVER.md                    <-- THIS FILE (Read upon starting new chat)
├── README.md                      <-- Project reproduction & methodology
├── data/
│   ├── benchmarks/                # CSV and Parquet files for AG News, Emotion, SST-2, TREC-6, NLI
│   └── teacher_targets/           # Cached teacher logits & null vectors (.npz)
├── scripts/
│   ├── prepare_data/              # Split generation & null bias diagnostic scripts
│   ├── cache_targets/             # Scripts caching teacher logits & null vectors
│   ├── train/                     # Standalone distillation training scripts
│   └── plotting/                  # Standalone 300 DPI plotting scripts
├── results/                       # JSON evaluation logs partitioned by dataset
│   ├── agnews/
│   ├── emotion/
│   ├── sst2/
│   ├── trec/
│   └── nli/
├── figures/                       # Dedicated Old vs New figures partitioned by dataset
│   ├── agnews/
│   ├── emotion/
│   ├── sst2/
│   └── trec/
├── logs/                          # Background task output logs
└── docs/                          # Detailed execution plan and research reports
```

---

## 6. What To Do When Starting the Next Chat

1. Confirm you have read `HANDOVER.md`.
2. Check the active state of the GPU (`nvidia-smi`).
3. Ask the user or wait for instructions on the next step:
   - Finalizing Benchmark 3 (SST-2 standalone package).
   - Running the cross-architecture check (`Llama-3.2-3B` student).
   - Compiling the ACL/EMNLP Short Paper draft.
4. Remember: **Execute one step at a time, keep datasets isolated, and pause when done.**
