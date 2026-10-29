# 📋 PROJECT HANDOVER & CONTEXT SPECIFICATION

> **Target Audience:** Any new AI agent or researcher starting a conversation in this repository (`/home/vishrut/calibrated-kd`).
> **Last Updated:** October 30, 2026

---

## 1. Executive Summary & Research Goal

This repository investigates **Calibrated Knowledge Distillation (CKD)** (also referred to as **Prior-Invariant Knowledge Distillation / PI-KD**).

### Core Problem
Autoregressive LLM teachers exhibit heavy **unigram prior skews** ($W_{\text{null}}$) across classification verbalizers, even when prompted in-context. Under standard forward-KL distillation, this teacher prior leaks into the training soft targets, causing:
1. **Severe class collapse / starvation** on less frequent or lower-prior minority classes.
2. **Artificial degradation** of distilled student accuracy and Macro-F1.

### The Solution: Static Calibrated KD (CKD)
We extract the teacher's response to an empty / null input (`"Text: N/A\nAnswer:"`) as $W_{\text{null}}$, and purge it before soft target computation:
$$\tilde{z}(x) = z(x) - W_{\text{null}}$$
$$P_{\text{cal}}(y \mid x) = \text{Softmax}(\tilde{z}(x))$$
$$\mathcal{L}_{\text{KD}} = D_{\text{KL}}\!\left(P_{\text{cal}}(y \mid x) \,\|\, P_{\text{student}}(y \mid x)\right)$$

### ⚠️ Crucial Project Constraint: DROP PRAXIS
* The initial project examined human annotator bias on PRAXIS (an Hindi-English translation dataset).
* **PRAXIS has been completely dropped** due to subjective, noisy human annotations.
* The focus is strictly on **canonical, high-agreement gold-standard NLP benchmarks** (AG News, Emotion, SST-2).

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

### Benchmark 3: SST-2 (2-Class Sentiment Classification) — ⏳ READY FOR PACKAGING
* **Splits:** $N_{\text{train}} = 1,000$ | $N_{\text{test}} = 200$ balanced
* **Classes:** `[Negative, Positive]`
* **Null Skew:** $\Delta_{\text{null}} = 1.25\text{ logits}$ (Minimal skew)
* **Purpose:** **Zero-Harm Safety Control** (proves calibration does not harm tasks with low prior bias).
* **Current Status:**
  * Standard KD: 81.50% Acc | 80.92% Macro-F1 (`[135, 65]`)
  * Calibrated KD: 81.50% Acc | 80.92% Macro-F1 (`[135, 65]`)
  * Exactly **0.00% delta** (zero distortion).
* **Next Action:** Generate dedicated standalone SST-2 figures (`figures/sst2/`) strictly when instructed.

---

## 5. Repository Map

```
/home/vishrut/calibrated-kd/
├── HANDOVER.md                    <-- THIS FILE (Read upon starting new chat)
├── README.md                      <-- Project reproduction & methodology
├── data/
│   ├── benchmarks/                # CSV and Parquet files for AG News, Emotion, SST-2, NLI
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
│   └── nli/
├── figures/                       # Dedicated Old vs New figures partitioned by dataset
│   ├── agnews/
│   └── emotion/
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
