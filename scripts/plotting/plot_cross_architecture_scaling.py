"""
plot_cross_architecture_scaling.py
----------------------------------
Generates publication-quality charts (300 DPI) summarizing the Cross-Architecture
and Capacity Scaling Distillation Suite:
  1) Macro-F1 Delta across 3 architectures (Qwen3-4B, Qwen2.5-0.5B, RoBERTa-base)
  2) Accuracy Comparison (Standard KD vs Calibrated KD across architectures)
  3) Minority Class Underprediction Mitigation across architectures
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures" / "cross_architecture"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Palette
COLOR_4B = "#2a9d8f"    # Teal (Qwen3-4B)
COLOR_05B = "#e76f51"   # Coral Orange (Qwen2.5-0.5B)
COLOR_ROB = "#457b9d"   # Steel Blue (RoBERTa-base)
STD_COLOR = "#d9534f"   # Coral Red
CAL_COLOR = "#2a9d8f"   # Teal

def make_figure():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # -------------------------------------------------------------
    # PANEL 1: Macro-F1 Delta Across Architectures & Datasets
    # -------------------------------------------------------------
    ax1 = axes[0]
    datasets = ["Emotion (6)", "AG News (4)", "TREC-6 (6)", "SST-2 (2) Control"]
    x = np.arange(len(datasets))
    w = 0.25

    # F1 deltas
    f1_4b = [1.44, 4.49, 2.04, 0.00]
    f1_05b = [1.71, 2.40, 3.28, 0.00]
    f1_rob = [3.25, 1.45, 1.07, -0.49]

    r1 = ax1.bar(x - w, f1_4b, w, label="Qwen3-4B (4B Causal, 1× Peer)", color=COLOR_4B, edgecolor="black", linewidth=1.1)
    r2 = ax1.bar(x, f1_05b, w, label="Qwen2.5-0.5B (500M Causal, 8× Comp)", color=COLOR_05B, edgecolor="black", linewidth=1.1)
    r3 = ax1.bar(x + w, f1_rob, w, label="RoBERTa-base (125M Encoder, 32× Comp)", color=COLOR_ROB, edgecolor="black", linewidth=1.1)

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title(r"(A) Macro-F1 Improvement ($\Delta_{\text{F1}}$) Across 3 Student Scales", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Macro-F1 Delta (% Points)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=10, fontweight="bold")
    ax1.set_ylim(-1.0, 5.5)
    ax1.legend(loc="upper right", fontsize=9, frameon=True)

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            va = "bottom" if h >= 0 else "top"
            ax1.annotate(f"{h:+.2f}%",
                         xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3 if h >= 0 else -8),
                         textcoords="offset points",
                         ha="center", va=va, fontsize=8, fontweight="bold")

    # -------------------------------------------------------------
    # PANEL 2: Target Suppression vs Prediction Behavior (Emotion Sadness Bloat)
    # -------------------------------------------------------------
    ax2 = axes[1]
    models = ["Qwen3-4B\n(4B Causal)", "Qwen2.5-0.5B\n(500M Causal)", "RoBERTa-base\n(125M Encoder)"]
    x2 = np.arange(len(models))
    w2 = 0.30

    # Emotion Sadness predictions (Ground truth = 50)
    std_sad = [152, 202, 163]
    cal_sad = [75, 62, 86]

    b1 = ax2.bar(x2 - w2/2, std_sad, w2, label="Standard KD (Raw Targets)", color=STD_COLOR, edgecolor="black", linewidth=1.1)
    b2 = ax2.bar(x2 + w2/2, cal_sad, w2, label="Calibrated KD (CKD)", color=CAL_COLOR, edgecolor="black", linewidth=1.1)

    ax2.axhline(50, color="#4a5568", linestyle=":", linewidth=2.0, label="Ground Truth (50/class)")
    ax2.set_title("(B) Emotion 'Sadness' Over-Prediction Halved Across All Students", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Test Prediction Count (N=300, Truth=50)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 240)
    ax2.legend(loc="upper right", fontsize=9, frameon=True)

    for b in b1:
        ax2.annotate(f"{int(b.get_height())}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=STD_COLOR)
    for b in b2:
        ax2.annotate(f"{int(b.get_height())}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=CAL_COLOR)

    plt.tight_layout()
    out1 = FIGURES_DIR / "fig5_cross_architecture_capacity_scaling.png"
    out2 = ARTIFACT_DIR / "fig5_cross_architecture_capacity_scaling.png"
    plt.savefig(out1, dpi=300)
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[✓] Saved {out1} and {out2}")

if __name__ == "__main__":
    make_figure()

