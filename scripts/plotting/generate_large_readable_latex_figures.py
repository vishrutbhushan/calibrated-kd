"""
generate_large_readable_latex_figures.py
----------------------------------------
Generates high-contrast, large-font, publication-grade vector PDF and 300 DPI PNG figures
specifically calibrated for LaTeX documents.
- Maximum 2 diagrams per row.
- Large, crystal-clear typography (10-12pt bold).
- Thick lines, high-contrast palette, exact data values on bars.
- Dashed Ground Truth reference lines/markers on all class distribution plots.
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
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
})

COLOR_STD = "#d90429"     # High-contrast Ruby Red (Standard KD)
COLOR_CKD = "#008080"     # High-contrast Teal (Calibrated KD / CKD)
COLOR_POST = "#f39c12"    # Warm Orange/Amber (Post-Hoc Output Cal)
COLOR_BATCH = "#2980b9"   # Royal Blue (Batch Cal KD)
COLOR_PRIOR = "#1b263b"   # Midnight Navy (Prior-Aware CKD)
COLOR_GT = "#111827"      # Dark Charcoal for Ground Truth

CONFIG_COLORS = [COLOR_STD, COLOR_POST, COLOR_BATCH, COLOR_CKD, COLOR_PRIOR]
CONFIG_NAMES = ["Standard KD", "Post-Hoc Cal", "Batch Cal KD", "Calibrated KD", "Prior-Aware CKD"]

def annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=9, color="#111827", dy=1.2):
    for r in rects:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, dy if h >= 0 else -dy-8),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=fontsize, fontweight="bold", color=color)

def export_fig(fig, base_name):
    plt.tight_layout()
    pdf_path = OUT_DIR / f"{base_name}.pdf"
    png_path = OUT_DIR / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(png_path, ARTIFACT_DIR / png_path.name)
    print(f"[✓] Exported {base_name}.pdf and .png")

# ==============================================================================
# 1. BALANCED BENCHMARK FIGURES (2 IN A ROW MAX)
# ==============================================================================

def make_balanced_f1_figures():
    models = ["Teacher\n(4B)", "Qwen3\n(4B)", "Qwen05\n(500M)", "RoBERTa\n(125M)"]
    x = np.arange(len(models))
    w = 0.35

    # Pair 1: Emotion & AG News
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    # Emotion
    r1 = ax1.bar(x - w/2, [34.92, 37.59, 30.80, 36.38], w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x + w/2, [35.62, 39.03, 32.51, 39.63], w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.set_title("Emotion (Balanced 6-Class) — Macro-F1 (%)")
    ax1.set_ylabel("Macro-F1 Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(25, 45)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # AG News
    r3 = ax2.bar(x - w/2, [70.24, 69.91, 60.53, 68.78], w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x + w/2, [67.45, 74.40, 62.93, 70.23], w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.set_title("AG News (Balanced 4-Class) — Macro-F1 (%)")
    ax2.set_ylabel("Macro-F1 Score (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(55, 80)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "fig_balanced_f1_pair1")

    # Pair 2: TREC-6 & SST-2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    # TREC-6
    r1 = ax1.bar(x - w/2, [69.93, 75.08, 59.70, 68.36], w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x + w/2, [69.21, 77.12, 62.98, 69.43], w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.set_title("TREC-6 (Balanced 6-Class) — Macro-F1 (%)")
    ax1.set_ylabel("Macro-F1 Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(55, 83)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # SST-2 Control
    r3 = ax2.bar(x - w/2, [84.21, 80.92, 82.66, 87.90], w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x + w/2, [84.74, 80.92, 82.66, 87.41], w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.set_title("SST-2 (Balanced Control) — Macro-F1 (%)")
    ax2.set_ylabel("Macro-F1 Score (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(75, 93)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "fig_balanced_f1_pair2")

def make_balanced_accuracy_figures():
    models = ["Teacher\n(4B)", "Qwen3\n(4B)", "Qwen05\n(500M)", "RoBERTa\n(125M)"]
    x = np.arange(len(models))
    w = 0.35

    # Pair 1: Emotion & AG News
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    r1 = ax1.bar(x - w/2, [40.33, 40.67, 35.00, 39.67], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x + w/2, [40.67, 43.33, 36.00, 44.33], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.set_title("Emotion (Balanced 6-Class) — Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(30, 49)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    r3 = ax2.bar(x - w/2, [72.60, 73.10, 66.30, 71.90], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x + w/2, [69.60, 76.50, 67.20, 72.50], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.set_title("AG News (Balanced 4-Class) — Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(60, 82)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "fig_balanced_acc_pair1")

    # Pair 2: TREC-6 & SST-2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    r1 = ax1.bar(x - w/2, [72.20, 78.20, 68.60, 73.20], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x + w/2, [72.20, 80.00, 71.20, 73.80], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.set_title("TREC-6 (Balanced 6-Class) — Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(62, 85)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    r3 = ax2.bar(x - w/2, [84.50, 81.00, 83.00, 88.00], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x + w/2, [85.00, 81.00, 83.00, 87.50], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.set_title("SST-2 (Balanced Control) — Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(75, 93)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "fig_balanced_acc_pair2")

def make_balanced_class_distributions():
    # Pair 1: AG News Balanced & Emotion Balanced
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=300)

    # AG News Balanced (Qwen3-4B)
    classes_ag = ["World", "Sports", "Business", "Sci/Tech"]
    x1 = np.arange(len(classes_ag))
    w = 0.35
    r1 = ax1.bar(x1 - w/2, [320, 241, 377, 62], w, label="Standard KD (Qwen3)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x1 + w/2, [226, 293, 392, 89], w, label="Calibrated KD (Qwen3)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.6, label="Ground Truth Expected (250)")
    ax1.set_title("AG News Balanced — Full Class Distribution (N=1,000)")
    ax1.set_ylabel("Test Prediction Count")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(classes_ag, fontweight="bold")
    ax1.set_ylim(0, 450)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    annotate_bars(ax1, r1, fmt="{:d}", fontsize=8.5)
    annotate_bars(ax1, r2, fmt="{:d}", fontsize=8.5)

    # Emotion Balanced (RoBERTa)
    classes_em = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    x2 = np.arange(len(classes_em))
    r3 = ax2.bar(x2 - w/2, [163, 50, 22, 45, 20, 0], w, label="Standard KD (RoBERTa)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x2 + w/2, [86, 52, 23, 98, 41, 0], w, label="Calibrated KD (RoBERTa)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.axhline(50, color=COLOR_GT, linestyle="--", linewidth=1.6, label="Ground Truth Expected (50)")
    ax2.set_title("Emotion Balanced — Full Class Distribution (N=300)")
    ax2.set_ylabel("Test Prediction Count")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes_em, fontweight="bold", fontsize=9)
    ax2.set_ylim(0, 195)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    annotate_bars(ax2, r3, fmt="{:d}", fontsize=8.5)
    annotate_bars(ax2, r4, fmt="{:d}", fontsize=8.5)
    export_fig(fig, "fig_balanced_class_dist_pair1")

    # Pair 2: TREC-6 Balanced & SST-2 Balanced
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=300)

    # TREC-6 Balanced (Qwen3-4B)
    classes_tr = ["Abbr", "Entity", "Desc", "Human", "Loc", "Num"]
    x1 = np.arange(len(classes_tr))
    gt_trec = [9, 94, 138, 65, 81, 113]
    r1 = ax1.bar(x1 - w/2, [18, 48, 178, 20, 86, 150], w, label="Standard KD (Qwen3)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x1 + w/2, [18, 58, 159, 28, 86, 151], w, label="Calibrated KD (Qwen3)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.plot(x1, gt_trec, color=COLOR_GT, marker="o", markersize=6, linestyle="--", linewidth=1.6, label="Ground Truth Counts")
    ax1.set_title("TREC-6 Balanced — Full Class Distribution (N=500)")
    ax1.set_ylabel("Test Prediction Count")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(classes_tr, fontweight="bold")
    ax1.set_ylim(0, 215)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    annotate_bars(ax1, r1, fmt="{:d}", fontsize=8.5)
    annotate_bars(ax1, r2, fmt="{:d}", fontsize=8.5)

    # SST-2 Balanced (RoBERTa)
    classes_sst = ["Negative", "Positive"]
    x2 = np.arange(len(classes_sst))
    r3 = ax2.bar(x2 - w/2, [118, 82], w, label="Standard KD (RoBERTa)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x2 + w/2, [117, 83], w, label="Calibrated KD (RoBERTa)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.axhline(100, color=COLOR_GT, linestyle="--", linewidth=1.6, label="Ground Truth Expected (100)")
    ax2.set_title("SST-2 Balanced Control — Class Distribution (N=200)")
    ax2.set_ylabel("Test Prediction Count")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes_sst, fontweight="bold")
    ax2.set_ylim(40, 145)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    annotate_bars(ax2, r3, fmt="{:d}", fontsize=8.5)
    annotate_bars(ax2, r4, fmt="{:d}", fontsize=8.5)
    export_fig(fig, "fig_balanced_class_dist_pair2")


# ==============================================================================
# 2. SKEWED BENCHMARK FIGURES (ROBERTA 5 CONFIGS)
# ==============================================================================

def make_skewed_performance_pair():
    """F1 and Accuracy across 5 configurations side-by-side (2 in a row)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    tasks = ["Emotion\nNatural", "TREC-6\nNatural", "AG News\nSkewed", "SST-2\nSkewed"]
    x = np.arange(len(tasks))
    w = 0.15

    # Panel 1: Macro-F1 ONLY
    f1_matrix = np.array([
        [35.02, 44.60, 41.29, 44.58, 42.74],
        [64.62, 68.91, 70.15, 75.09, 74.02],
        [57.22, 65.53, 57.15, 64.22, 58.83],
        [65.03, 62.97, 67.03, 78.69, 84.36],
    ])
    for i in range(5):
        rects = ax1.bar(x + (i - 2) * w, f1_matrix[:, i], w, label=CONFIG_NAMES[i],
                        color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax1, rects, fmt="{:.1f}%", fontsize=7.8)
    ax1.set_title("Macro-F1 (%) Across 5 Distillation Configs (RoBERTa)")
    ax1.set_ylabel("Macro-F1 Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontweight="bold")
    ax1.set_ylim(25, 93)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5, ncol=2)

    # Panel 2: Accuracy ONLY
    acc_matrix = np.array([
        [50.80, 56.20, 54.40, 56.60, 56.20],
        [74.60, 76.60, 75.60, 77.20, 76.40],
        [65.80, 70.40, 65.70, 69.90, 67.20],
        [68.50, 67.00, 70.00, 79.50, 84.50],
    ])
    for i in range(5):
        rects = ax2.bar(x + (i - 2) * w, acc_matrix[:, i], w, label=CONFIG_NAMES[i],
                        color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax2, rects, fmt="{:.1f}%", fontsize=7.8)
    ax2.set_title("Test Accuracy (%) Across 5 Distillation Configs (RoBERTa)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontweight="bold")
    ax2.set_ylim(40, 93)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5, ncol=2)

    export_fig(fig, "fig_skewed_performance_pair")

def make_skewed_class_distributions():
    # Pair 1: AG News Skewed & Emotion Natural
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    # AG News Skewed (4 classes across 5 configs)
    classes_ag = ["World\n(50%)", "Sports\n(25%)", "Business\n(15%)", "Sci/Tech\n(10%)"]
    x1 = np.arange(len(classes_ag))
    w = 0.15
    counts_ag = np.array([
        [370, 210, 420, 0],
        [270, 247, 449, 34],
        [366, 210, 424, 0],
        [282, 246, 443, 29],
        [330, 238, 429, 3]
    ])
    for i in range(5):
        rects = ax1.bar(x1 + (i - 2) * w, counts_ag[i], w, label=CONFIG_NAMES[i],
                        color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax1, rects, fmt="{:d}", fontsize=7.5)
    ax1.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.6, label="Ground Truth Expected (250 / cl)")
    ax1.set_title("AG News Skewed — Full Class Distribution (N=1,000)")
    ax1.set_ylabel("Test Predictions")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(classes_ag, fontweight="bold")
    ax1.set_ylim(0, 520)
    ax1.legend(loc="upper right", frameon=True, fontsize=8)

    # Emotion Natural (6 classes across 5 configs)
    classes_em = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    x2 = np.arange(len(classes_em))
    counts_em = np.array([
        [302, 83, 41, 69, 5, 0],
        [179, 94, 52, 113, 62, 0],
        [265, 80, 48, 85, 22, 0],
        [161, 97, 48, 151, 43, 0],
        [209, 109, 37, 118, 27, 0]
    ])
    gt_emotion = [148, 151, 39, 79, 70, 13]
    for i in range(5):
        rects = ax2.bar(x2 + (i - 2) * w, counts_em[i], w, label=CONFIG_NAMES[i],
                        color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax2, rects, fmt="{:d}", fontsize=7.2)
    ax2.plot(x2, gt_emotion, color=COLOR_GT, marker="o", markersize=6, linestyle="--", linewidth=1.6, label="Ground Truth Distribution")
    ax2.set_title("Emotion Natural — Full Class Distribution (N=500)")
    ax2.set_ylabel("Test Predictions")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes_em, fontweight="bold", fontsize=9)
    ax2.set_ylim(0, 360)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)
    export_fig(fig, "fig_skewed_class_dist_pair1")

    # Pair 2: TREC-6 Natural & SST-2 Skewed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    # TREC-6 Natural (6 classes across 5 configs)
    classes_tr = ["Abbr\n(1.6%)", "Entity", "Desc", "Human", "Loc", "Num"]
    x1 = np.arange(len(classes_tr))
    counts_tr = np.array([
        [2, 24, 194, 43, 87, 150],
        [15, 10, 180, 57, 88, 150],
        [6, 21, 184, 45, 87, 157],
        [10, 43, 170, 38, 85, 154],
        [8, 43, 176, 36, 81, 156]
    ])
    gt_trec = [9, 94, 138, 65, 81, 113]
    for i in range(5):
        rects = ax1.bar(x1 + (i - 2) * w, counts_tr[i], w, label=CONFIG_NAMES[i],
                        color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax1, rects, fmt="{:d}", fontsize=7.2)
    ax1.plot(x1, gt_trec, color=COLOR_GT, marker="o", markersize=6, linestyle="--", linewidth=1.6, label="Ground Truth Distribution")
    ax1.set_title("TREC-6 Natural — Full Class Distribution (N=500)")
    ax1.set_ylabel("Test Predictions")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(classes_tr, fontweight="bold")
    ax1.set_ylim(0, 230)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # SST-2 Skewed (2 classes across 5 configs)
    classes_sst = ["Negative\n(20% Train)", "Positive\n(80% Train)"]
    x2 = np.arange(len(classes_sst))
    counts_sst = np.array([
        [163, 37],
        [166, 34],
        [160, 40],
        [139, 61],
        [119, 81]
    ])
    for i in range(5):
        rects = ax2.bar(x2 + (i - 2) * w, counts_sst[i], w, label=CONFIG_NAMES[i],
                        color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax2, rects, fmt="{:d}", fontsize=8)
    ax2.axhline(100, color=COLOR_GT, linestyle="--", linewidth=1.6, label="Ground Truth Expected (100 / cl)")
    ax2.set_title("SST-2 Skewed — Full Class Distribution (N=200)")
    ax2.set_ylabel("Test Predictions")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes_sst, fontweight="bold")
    ax2.set_ylim(0, 195)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)
    export_fig(fig, "fig_skewed_class_dist_pair2")


# ==============================================================================
# 3. CROSS-ARCHITECTURE FIGURES (AG NEWS SKEWED)
# ==============================================================================

def make_cross_architecture_figures():
    models = ["RoBERTa\n(125M)", "Qwen05\n(500M)", "Qwen3\n(4B)"]
    x = np.arange(len(models))
    w = 0.35

    # Pair 1: F1 and Accuracy Across Scales (2 in a row)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    # F1
    r1 = ax1.bar(x - w/2, [57.22, 55.05, 57.80], w, label="Standard KD (Collapsed)", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r2 = ax1.bar(x + w/2, [64.22, 60.75, 67.14], w, label="Calibrated KD (Revived)", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax1.set_title("AG News Skewed — Macro-F1 (%) Across Capacities")
    ax1.set_ylabel("Macro-F1 Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(45, 75)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Accuracy
    r3 = ax2.bar(x - w/2, [65.80, 63.00, 65.70], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.9)
    r4 = ax2.bar(x + w/2, [69.90, 67.20, 71.30], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
    ax2.set_title("AG News Skewed — Accuracy (%) Across Capacities")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(55, 78)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "fig_cross_arch_performance_pair")

    # Multi-class distribution across the 3 models (3-column wide figure)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300, sharey=True)
    classes = ["World\n(50%)", "Sports\n(25%)", "Business\n(15%)", "Sci/Tech\n(10%)"]
    xc = np.arange(len(classes))
    wc = 0.35

    arch_counts = [
        ("RoBERTa-base (125M)", [370, 210, 420, 0], [282, 246, 443, 29]),
        ("Qwen2.5-0.5B (500M)", [365, 211, 424, 0], [245, 260, 479, 16]),
        ("Qwen3-4B (4B Peer)", [348, 214, 434, 4], [237, 256, 463, 44])
    ]
    for idx, (m_name, std_c, ckd_c) in enumerate(arch_counts):
        ax = axes[idx]
        r1 = ax.bar(xc - wc/2, std_c, wc, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.9)
        r2 = ax.bar(xc + wc/2, ckd_c, wc, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.9)
        ax.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.6, label="Ground Truth Expected (250)")
        ax.set_title(m_name)
        ax.set_ylabel("Predictions (out of 1,000)" if idx == 0 else "")
        ax.set_xticks(xc)
        ax.set_xticklabels(classes, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 520)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        annotate_bars(ax, r1, fmt="{:d}", fontsize=8)
        annotate_bars(ax, r2, fmt="{:d}", fontsize=8)
    export_fig(fig, "fig_cross_arch_class_dist_wide")

def main():
    print("Generating large, ultra-readable figures for LaTeX...")
    make_balanced_f1_figures()
    make_balanced_accuracy_figures()
    make_balanced_class_distributions()
    make_skewed_performance_pair()
    make_skewed_class_distributions()
    make_cross_architecture_figures()
    print("All large figures generated successfully!")

if __name__ == "__main__":
    main()

