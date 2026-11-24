"""
generate_benchmark_barchart.py
------------------------------
Generates a publication-grade bar chart comparing:
  1. Teacher Base (Gemma-3-4B Raw Zero-Shot)
  2. Standard KD (RoBERTa-base)
  3. Batch-Calibrated KD (RoBERTa-base)
  4. Null Calibrated KD (Ours) (RoBERTa-base)

Note:
  - Removed Post-Hoc Output Cal.
  - Added Teacher Base (Raw Zero-Shot).
  - Tilted x-axis labels (-25 degrees, ha='right') with proper bottom margin to guarantee ZERO overlap.
  - Clean titles without em dashes.
  - Refined ColorBrewer qualitative palette.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "results" / "main_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10.5,
    "axes.titlesize": 12.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10.0,
})

datasets = [
    "Student Questions (K=4)",
    "TREC-6 (K=6)",
    "App Reviews (K=5)",
    "Twitter Emotion (K=6)",
    "TweetEval Emotion (K=4)",
    "BBC News (K=5)"
]

# Benchmark Data
f1_teacher = [28.13, 50.15, 29.69, 39.80, 45.78, 54.81]
f1_std     = [23.09, 62.28, 30.25, 29.78, 34.16, 56.26]
f1_batch   = [23.58, 63.29, 32.04, 31.13, 38.34, 56.95]
f1_ckd     = [37.23, 64.71, 34.50, 34.46, 55.48, 71.47]

acc_teacher = [33.75, 52.25, 48.25, 53.00, 57.25, 56.00]
acc_std     = [32.50, 59.75, 51.75, 50.50, 54.25, 59.25]
acc_batch   = [32.25, 61.00, 52.25, 50.00, 56.25, 60.00]
acc_ckd     = [39.00, 61.75, 54.00, 50.75, 66.50, 70.50]

# Refined Qualitative Palette:
C_TEACHER = "#94A3B8"  # Slate Neutral
E_TEACHER = "#475569"

C_STD     = "#E15759"  # Soft Crimson
E_STD     = "#B91C1C"

C_BATCH   = "#8DA0CB"  # Periwinkle
E_BATCH   = "#4F46E5"

C_CKD     = "#66C2A5"  # Sage Teal
E_CKD     = "#0D9488"

def render_plot(save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")

    x = np.arange(len(datasets))
    bar_width = 0.185
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    # -------------------------------------------------------------
    # SUBPLOT 1: Macro-F1 (Primary Metric)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#FAFAFA")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax1.set_axisbelow(True)

    r1_tea = ax1.bar(x + offsets[0], f1_teacher, bar_width, label="Teacher Base (Gemma-3-4B)", 
                     color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.9, zorder=3)
    r1_std = ax1.bar(x + offsets[1], f1_std, bar_width, label="Standard KD", 
                     color=C_STD, edgecolor=E_STD, linewidth=0.9, zorder=3)
    r1_batch = ax1.bar(x + offsets[2], f1_batch, bar_width, label="Batch-Calibrated KD", 
                       color=C_BATCH, edgecolor=E_BATCH, linewidth=0.9, zorder=3)
    r1_ckd = ax1.bar(x + offsets[3], f1_ckd, bar_width, label="Null Calibrated KD (Ours)", 
                     color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3)

    ax1.set_title("(a) Macro-F1 (%) : Primary Metric", pad=12, color="#0F172A")
    ax1.set_ylabel("Macro-F1 (%)", color="#0F172A")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=25, ha="right", color="#1E293B")
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 10))

    for spine in ax1.spines.values():
        spine.set_color("#CBD5E1")

    # -------------------------------------------------------------
    # SUBPLOT 2: Overall Accuracy (Secondary Metric)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax2.set_axisbelow(True)

    r2_tea = ax2.bar(x + offsets[0], acc_teacher, bar_width, 
                     color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.9, zorder=3)
    r2_std = ax2.bar(x + offsets[1], acc_std, bar_width, 
                     color=C_STD, edgecolor=E_STD, linewidth=0.9, zorder=3)
    r2_batch = ax2.bar(x + offsets[2], acc_batch, bar_width, 
                       color=C_BATCH, edgecolor=E_BATCH, linewidth=0.9, zorder=3)
    r2_ckd = ax2.bar(x + offsets[3], acc_ckd, bar_width, 
                     color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3)

    ax2.set_title("(b) Overall Accuracy (%) : Secondary Metric", pad=12, color="#0F172A")
    ax2.set_ylabel("Accuracy (%)", color="#0F172A")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=25, ha="right", color="#1E293B")
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.arange(0, 101, 10))

    for spine in ax2.spines.values():
        spine.set_color("#CBD5E1")

    # -------------------------------------------------------------
    # Unified Top Legend
    # -------------------------------------------------------------
    handles = [r1_tea, r1_std, r1_batch, r1_ckd]
    labels = ["Teacher Base (Gemma-3-4B)", "Standard KD", "Batch-Calibrated KD", "Null Calibrated KD (Ours)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), 
               ncol=4, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", 
               fontsize=10.0, borderpad=0.5, handlelength=1.4)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"Chart with tilted labels saved to: {save_path}")

if __name__ == "__main__":
    out1 = OUT_DIR / "chart_main_benchmark_f1_accuracy.png"
    out2 = ARTIFACT_DIR / "chart_main_benchmark_f1_accuracy.png"
    render_plot(out1)
    if ARTIFACT_DIR.exists():
        shutil.copy(out1, out2)
        print(f"Mirrored to artifact: {out2}")
