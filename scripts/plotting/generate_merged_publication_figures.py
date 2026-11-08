"""
generate_merged_publication_figures.py
--------------------------------------
Generates 6 consolidated, publication-grade vector PDF and 300 DPI PNG figures:
1. fig1_balanced_f1_overview (2x2: Emotion, AG News, TREC-6, SST-2)
2. fig2_balanced_acc_overview (2x2: Emotion, AG News, TREC-6, SST-2)
3. fig3_balanced_class_dist (1x2: AG News & Emotion full distributions with GT line)
4. fig4_skewed_performance (1x2: Skewed Macro-F1 on left, Test Accuracy on right across 5 KD configs)
5. fig5_skewed_class_dist (1x2: AG News Skewed & Emotion Natural distributions with GT)
6. fig6_cross_arch_scaling (1x2: Cross-arch F1/Acc gains & Minority prediction recovery across 125M, 500M, 4B)
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "latex" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
})

COLOR_STD = "#d90429"     # High-contrast Ruby Red (Standard KD)
COLOR_CKD = "#008080"     # High-contrast Teal (Calibrated KD / CKD)
COLOR_POST = "#f39c12"    # Warm Orange/Amber (Post-Hoc Output Cal)
COLOR_BATCH = "#2980b9"   # Royal Blue (Batch Cal KD)
COLOR_PRIOR = "#1b263b"   # Midnight Navy (Prior-Aware CKD)
COLOR_GT = "#111827"      # Dark Charcoal for Ground Truth

CONFIG_COLORS = [COLOR_STD, COLOR_POST, COLOR_BATCH, COLOR_CKD, COLOR_PRIOR]
CONFIG_NAMES = ["Standard KD", "Post-Hoc Cal", "Batch Cal KD", "Calibrated KD", "Prior-Aware CKD"]

def annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=7.5, color="#111827", dy=1.0):
    for r in rects:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, dy if h >= 0 else -dy),
                    textcoords="offset points",
                    ha="center", va=va,
                    fontsize=fontsize, fontweight="bold", color=color)

def export(fig, base_name):
    pdf_path = OUT_DIR / f"{base_name}.pdf"
    png_path = OUT_DIR / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if ARTIFACT_DIR.exists():
        shutil.copy(png_path, ARTIFACT_DIR / f"{base_name}.png")
    print(f"Saved: {pdf_path.name}")
    plt.close(fig)

# ==============================================================================
# FIGURE 1: Balanced Macro-F1 Overview (2x2 Grid)
# ==============================================================================
def make_fig1_balanced_f1():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.4))
    models = ["Qwen3-4B\n(4B Peer)", "Qwen2.5-0.5B\n(500M, 8x)", "RoBERTa-base\n(125M, 32x)"]
    x = np.arange(len(models))
    w = 0.32

    # Emotion
    ax = axes[0, 0]
    std = [37.59, 30.80, 36.38]
    ckd = [39.03, 32.51, 39.63]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("Emotion (6 classes, Balanced)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(20, 48)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="upper left")

    # AG News
    ax = axes[0, 1]
    std = [69.91, 60.53, 68.78]
    ckd = [74.40, 62.93, 70.23]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("AG News (4 classes, Balanced)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(50, 82)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="upper left")

    # TREC-6
    ax = axes[1, 0]
    std = [75.08, 59.70, 68.36]
    ckd = [77.12, 62.98, 69.43]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("TREC-6 (Stratified Train, CogComp Test)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(50, 85)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="upper left")

    # SST-2
    ax = axes[1, 1]
    std = [80.92, 82.66, 87.90]
    ckd = [80.92, 82.66, 87.41]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("SST-2 (2 classes, Balanced Control)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(70, 95)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="lower right")

    fig.tight_layout()
    export(fig, "fig1_balanced_f1_overview")

# ==============================================================================
# FIGURE 2: Balanced Test Accuracy Overview (2x2 Grid)
# ==============================================================================
def make_fig2_balanced_acc():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.4))
    models = ["Qwen3-4B\n(4B Peer)", "Qwen2.5-0.5B\n(500M, 8x)", "RoBERTa-base\n(125M, 32x)"]
    x = np.arange(len(models))
    w = 0.32

    # Emotion
    ax = axes[0, 0]
    std = [40.67, 35.00, 39.67]
    ckd = [43.33, 36.00, 44.33]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("Emotion (Test Accuracy)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(25, 52)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="upper left")

    # AG News
    ax = axes[0, 1]
    std = [73.10, 66.30, 71.90]
    ckd = [76.50, 67.20, 72.50]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("AG News (Test Accuracy)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(55, 85)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="upper left")

    # TREC-6
    ax = axes[1, 0]
    std = [78.20, 68.60, 73.20]
    ckd = [80.00, 71.20, 73.80]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("TREC-6 (Test Accuracy)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(60, 88)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="upper left")

    # SST-2
    ax = axes[1, 1]
    std = [81.00, 83.00, 88.00]
    ckd = [81.00, 83.00, 87.50]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax.set_title("SST-2 (Test Accuracy)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(70, 95)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    ax.legend(loc="lower right")

    fig.tight_layout()
    export(fig, "fig2_balanced_acc_overview")

# ==============================================================================
# FIGURE 3: Balanced Class Distributions (1x2 Pair: AG News & Emotion)
# ==============================================================================
def make_fig3_balanced_class_dist():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # AG News RoBERTa
    classes_ag = ["World", "Sports", "Business", "Sci/Tech"]
    x = np.arange(len(classes_ag))
    w = 0.35
    std_counts = [267, 244, 252, 237]
    ckd_counts = [258, 245, 248, 249]
    r1 = ax1.bar(x - w/2, std_counts, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax1.bar(x + w/2, ckd_counts, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax1.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.8, label="Ground Truth (250)")
    ax1.set_title("AG News Prediction Counts (RoBERTa)")
    ax1.set_ylabel("Predictions (out of 1,000)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes_ag, fontweight="bold")
    ax1.set_ylim(180, 310)
    annotate_bars(ax1, r1, fmt="{:d}", dy=1.2)
    annotate_bars(ax1, r2, fmt="{:d}", dy=1.2)
    ax1.legend(loc="upper right")

    # Emotion RoBERTa
    classes_em = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    x2 = np.arange(len(classes_em))
    w2 = 0.35
    std_em = [163, 62, 10, 48, 12, 5]
    ckd_em = [86, 75, 24, 65, 36, 14]
    r1_em = ax2.bar(x2 - w2/2, std_em, w2, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2_em = ax2.bar(x2 + w2/2, ckd_em, w2, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax2.axhline(50, color=COLOR_GT, linestyle="--", linewidth=1.8, label="Ground Truth (50)")
    ax2.set_title("Emotion Prediction Counts (RoBERTa)")
    ax2.set_ylabel("Predictions (out of 300)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes_em, fontweight="bold")
    ax2.set_ylim(0, 190)
    annotate_bars(ax2, r1_em, fmt="{:d}", dy=1.2)
    annotate_bars(ax2, r2_em, fmt="{:d}", dy=1.2)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    export(fig, "fig3_balanced_class_dist")

# ==============================================================================
# FIGURE 4: Skewed Benchmarks Performance (1x2 Pair: F1 & Acc Across 5 Configs)
# ==============================================================================
def make_fig4_skewed_performance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    tasks = ["AG News Skewed\n(10% Sci/Tech)", "Emotion Natural\n(Saravia Dist)", "TREC-6 Natural\n(1.6% Abbr)", "SST-2 Skewed\n(80/20 Imbalance)"]
    x = np.arange(len(tasks))
    w = 0.16

    f1_matrix = [
        [57.22, 35.02, 64.62, 65.03], # Std KD
        [65.53, 44.60, 68.91, 62.97], # Post-Hoc
        [57.15, 41.29, 70.15, 67.03], # Batch Cal
        [64.22, 44.58, 75.09, 78.69], # CKD
        [58.83, 42.74, 74.02, 84.36], # Prior-Aware
    ]

    acc_matrix = [
        [65.80, 50.80, 74.60, 68.50], # Std KD
        [70.40, 56.20, 76.60, 67.00], # Post-Hoc
        [65.70, 54.40, 75.60, 70.00], # Batch Cal
        [69.90, 56.60, 77.20, 79.50], # CKD
        [67.20, 56.20, 76.40, 84.50], # Prior-Aware
    ]

    for i in range(5):
        rects = ax1.bar(x + (i - 2)*w, f1_matrix[i], w, label=CONFIG_NAMES[i], color=CONFIG_COLORS[i], edgecolor="#111", linewidth=0.7)
        annotate_bars(ax1, rects, fmt="{:.1f}%", fontsize=6.8, dy=1.0)

    ax1.set_title("Non-Balanced Benchmarks: Macro-F1 (%) on RoBERTa")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontweight="bold")
    ax1.set_ylim(25, 93)
    ax1.legend(loc="upper left")

    for i in range(5):
        rects = ax2.bar(x + (i - 2)*w, acc_matrix[i], w, label=CONFIG_NAMES[i], color=CONFIG_COLORS[i], edgecolor="#111", linewidth=0.7)
        annotate_bars(ax2, rects, fmt="{:.1f}%", fontsize=6.8, dy=1.0)

    ax2.set_title("Non-Balanced Benchmarks: Test Accuracy (%) on RoBERTa")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontweight="bold")
    ax2.set_ylim(40, 93)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    export(fig, "fig4_skewed_performance")

# ==============================================================================
# FIGURE 5: Skewed Class Distributions (1x2 Pair: AG News Skewed & Emotion Natural)
# ==============================================================================
def make_fig5_skewed_class_dist():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # AG News Skewed
    classes_ag = ["World", "Sports", "Business", "Sci/Tech\n(10% Min)"]
    x1 = np.arange(len(classes_ag))
    w = 0.16
    ag_preds = [
        [371, 281, 348, 0],   # Std KD (0 Sci/Tech)
        [285, 252, 298, 165], # Post-Hoc
        [372, 280, 348, 0],   # Batch Cal (0 Sci/Tech)
        [324, 275, 372, 29],  # CKD (29 Sci/Tech)
        [341, 282, 360, 17],  # Prior-Aware
    ]
    for i in range(5):
        r = ax1.bar(x1 + (i - 2)*w, ag_preds[i], w, label=CONFIG_NAMES[i], color=CONFIG_COLORS[i], edgecolor="#111", linewidth=0.7)
        annotate_bars(ax1, r, fmt="{:d}", fontsize=7.0, dy=1.2)

    ax1.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.8, label="Ground Truth (250)")
    ax1.set_title("AG News Skewed: Predictions Across Methods")
    ax1.set_ylabel("Predictions (out of 1,000)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(classes_ag, fontweight="bold")
    ax1.set_ylim(0, 420)
    ax1.legend(loc="upper right", ncol=2)

    # Emotion Natural
    classes_em = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    x2 = np.arange(len(classes_em))
    em_preds = [
        [302, 168, 0, 25, 5, 0],   # Std KD
        [152, 175, 12, 108, 38, 15], # Post-Hoc
        [265, 185, 0, 28, 22, 0],  # Batch Cal
        [161, 182, 15, 84, 43, 15], # CKD
        [180, 185, 8, 80, 35, 12],  # Prior-Aware
    ]
    gt_em = [148, 179, 41, 57, 70, 5]

    for i in range(5):
        r = ax2.bar(x2 + (i - 2)*w, em_preds[i], w, label=CONFIG_NAMES[i], color=CONFIG_COLORS[i], edgecolor="#111", linewidth=0.7)
        annotate_bars(ax2, r, fmt="{:d}", fontsize=6.8, dy=1.0)

    ax2.plot(x2, gt_em, color=COLOR_GT, marker="o", markersize=6, linewidth=1.8, linestyle="--", label="Ground Truth Support")
    ax2.set_title("Emotion Natural: Predictions Across Methods")
    ax2.set_ylabel("Predictions (out of 500)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes_em, fontweight="bold")
    ax2.set_ylim(0, 345)
    ax2.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    export(fig, "fig5_skewed_class_dist")

# ==============================================================================
# FIGURE 6: Cross-Architecture Capacity Scaling (1x2 Pair: Gains & Sci/Tech Recovery)
# ==============================================================================
def make_fig6_cross_arch_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    models = ["RoBERTa-base\n(125M Encoder)", "Qwen2.5-0.5B\n(500M Causal)", "Qwen3-4B\n(4B Peer Causal)"]
    x = np.arange(len(models))
    w = 0.35

    # Left: Macro-F1 across scales on AG News Skewed
    std_f1 = [57.22, 55.05, 57.80]
    ckd_f1 = [64.22, 60.75, 67.14]
    r1 = ax1.bar(x - w/2, std_f1, w, label="Standard KD F1", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2 = ax1.bar(x + w/2, ckd_f1, w, label="Calibrated KD F1", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax1.set_title("AG News Skewed: Macro-F1 Across Capacities")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(45, 75)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)
    ax1.legend(loc="upper left")

    # Right: Sci/Tech Predictions across scales
    std_preds = [0, 0, 4]
    ckd_preds = [29, 16, 44]
    r1_p = ax2.bar(x - w/2, std_preds, w, label="Standard KD", color=COLOR_STD, edgecolor="#111", linewidth=0.8)
    r2_p = ax2.bar(x + w/2, ckd_preds, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="#111", linewidth=0.8)
    ax2.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.8, label="Ground Truth Support (250)")
    ax2.set_title("Minority Sci/Tech Predictions Across Scales")
    ax2.set_ylabel("Predictions (out of 1,000 items)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(0, 275)
    annotate_bars(ax2, r1_p, fmt="{:d}", dy=1.2)
    annotate_bars(ax2, r2_p, fmt="{:d}", dy=1.2)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    export(fig, "fig6_cross_arch_scaling")

if __name__ == "__main__":
    make_fig1_balanced_f1()
    make_fig2_balanced_acc()
    make_fig3_balanced_class_dist()
    make_fig4_skewed_performance()
    make_fig5_skewed_class_dist()
    make_fig6_cross_arch_scaling()
    print("All 6 publication-grade figures successfully generated!")
