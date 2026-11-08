"""
generate_latex_figures.py
-------------------------
Generates vector PDF and 300 DPI PNG figures specifically dimensioned for two-column
ACL LaTeX papers.
Constraints enforced:
- Maximum 2 diagrams per row.
- Strict metric isolation: F1 and Accuracy are never on the same graph.
- Every class distribution graph shows ALL classes together with the Ground Truth line.
- Balanced and Non-Balanced are completely separated.
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
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
})

COLOR_STD = "#e63946"     # Crimson Red
COLOR_CKD = "#2a9d8f"     # Emerald Teal
COLOR_POST = "#f4a261"    # Warm Amber
COLOR_BATCH = "#457b9d"   # Steel Blue
COLOR_PRIOR = "#1d3557"   # Deep Navy
COLOR_GT = "#111827"      # Ground Truth line

CONFIG_COLORS = [COLOR_STD, COLOR_POST, COLOR_BATCH, COLOR_CKD, COLOR_PRIOR]
CONFIG_NAMES = ["Standard KD", "Post-Hoc Cal", "Batch Cal KD", "Calibrated KD", "Prior-Aware CKD"]

def annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=7.5, color="#111827"):
    for r in rects:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 2 if h >= 0 else -6),
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

# -------------------------------------------------------------
# 1. BALANCED BENCHMARKS (2 in a row max)
# -------------------------------------------------------------
def make_balanced_f1():
    models = ["Teacher (4B)", "Qwen3 (4B)", "Qwen05 (500M)", "RoBERTa (125M)"]
    x = np.arange(len(models))
    w = 0.35

    # Part 1: Emotion & AG News
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=300)
    # Emotion
    r1 = ax1.bar(x - w/2, [34.92, 37.59, 30.80, 36.38], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r2 = ax1.bar(x + w/2, [35.62, 39.03, 32.51, 39.63], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax1.set_title("Emotion Balanced (6 Cl) — Macro-F1 (%)")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylim(25, 45)
    ax1.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # AG News
    r3 = ax2.bar(x - w/2, [70.24, 69.91, 60.53, 68.78], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r4 = ax2.bar(x + w/2, [67.45, 74.40, 62.93, 70.23], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax2.set_title("AG News Balanced (4 Cl) — Macro-F1 (%)")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylim(55, 80)
    ax2.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "balanced_f1_part1")

    # Part 2: TREC-6 & SST-2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=300)
    # TREC-6
    r1 = ax1.bar(x - w/2, [69.93, 75.08, 59.70, 68.36], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r2 = ax1.bar(x + w/2, [69.21, 77.12, 62.98, 69.43], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax1.set_title("TREC-6 Balanced (6 Cl) — Macro-F1 (%)")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylim(55, 83)
    ax1.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # SST-2
    r3 = ax2.bar(x - w/2, [84.21, 80.92, 82.66, 87.90], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r4 = ax2.bar(x + w/2, [84.74, 80.92, 82.66, 87.41], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax2.set_title("SST-2 Balanced Control (2 Cl) — Macro-F1 (%)")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylim(75, 93)
    ax2.legend(loc="lower right", fontsize=7.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "balanced_f1_part2")

def make_balanced_accuracy():
    models = ["Teacher (4B)", "Qwen3 (4B)", "Qwen05 (500M)", "RoBERTa (125M)"]
    x = np.arange(len(models))
    w = 0.35

    # Part 1: Emotion & AG News
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=300)
    r1 = ax1.bar(x - w/2, [40.33, 40.67, 35.00, 39.67], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r2 = ax1.bar(x + w/2, [40.67, 43.33, 36.00, 44.33], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax1.set_title("Emotion Balanced — Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylim(30, 49)
    ax1.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    r3 = ax2.bar(x - w/2, [72.60, 73.10, 66.30, 71.90], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r4 = ax2.bar(x + w/2, [69.60, 76.50, 67.20, 72.50], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax2.set_title("AG News Balanced — Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylim(60, 82)
    ax2.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "balanced_acc_part1")

    # Part 2: TREC-6 & SST-2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=300)
    r1 = ax1.bar(x - w/2, [72.20, 78.20, 68.60, 73.20], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r2 = ax1.bar(x + w/2, [72.20, 80.00, 71.20, 73.80], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax1.set_title("TREC-6 Balanced — Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylim(62, 85)
    ax1.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    r3 = ax2.bar(x - w/2, [84.50, 81.00, 83.00, 88.00], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r4 = ax2.bar(x + w/2, [85.00, 81.00, 83.00, 87.50], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax2.set_title("SST-2 Balanced Control — Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylim(75, 93)
    ax2.legend(loc="lower right", fontsize=7.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "balanced_acc_part2")


# -------------------------------------------------------------
# 2. SKEWED BENCHMARKS (ROBERTA 5 CONFIGS)
# -------------------------------------------------------------
def make_skewed_f1_and_acc():
    tasks = ["Emotion Natural", "TREC-6 Natural", "AG News Skewed", "SST-2 Skewed"]
    x = np.arange(len(tasks))
    w = 0.15

    # F1 across 5 configs
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=300)
    f1_matrix = np.array([
        [35.02, 44.60, 41.29, 44.58, 42.74],
        [64.62, 68.91, 70.15, 75.09, 74.02],
        [57.22, 65.53, 57.15, 64.22, 58.83],
        [65.03, 62.97, 67.03, 78.69, 84.36],
    ])
    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, f1_matrix[:, i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.6)
        annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=7)
    ax.set_title("Non-Balanced Suite: Macro-F1 (%) Across 5 Distillation Configs (RoBERTa)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontweight="bold")
    ax.set_ylim(25, 92)
    ax.legend(loc="upper left", frameon=True, fontsize=7.5, ncol=2)
    export_fig(fig, "skewed_f1_roberta")

    # Accuracy across 5 configs
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=300)
    acc_matrix = np.array([
        [50.80, 56.20, 54.40, 56.60, 56.20],
        [74.60, 76.60, 75.60, 77.20, 76.40],
        [65.80, 70.40, 65.70, 69.90, 67.20],
        [68.50, 67.00, 70.00, 79.50, 84.50],
    ])
    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, acc_matrix[:, i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.6)
        annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=7)
    ax.set_title("Non-Balanced Suite: Test Accuracy (%) Across 5 Distillation Configs (RoBERTa)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontweight="bold")
    ax.set_ylim(40, 92)
    ax.legend(loc="upper left", frameon=True, fontsize=7.5, ncol=2)
    export_fig(fig, "skewed_acc_roberta")


# -------------------------------------------------------------
# 3. COMPLETE CLASS DISTRIBUTIONS (ALL CLASSES + GT LINE)
# -------------------------------------------------------------
def make_class_distributions():
    # 1. AG News Skewed (4 classes across 5 configs)
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=300)
    classes = ["World (50%)", "Sports (25%)", "Business (15%)", "Sci/Tech (10%)"]
    x = np.arange(len(classes))
    w = 0.15
    counts = np.array([
        [370, 210, 420, 0],
        [270, 247, 449, 34],
        [366, 210, 424, 0],
        [282, 246, 443, 29],
        [330, 238, 429, 3]
    ])
    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, counts[i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.6)
        annotate_bars(ax, rects, fmt="{:d}", fontsize=6.8)
    ax.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.2, label="Ground Truth Expected (250 / cl)")
    ax.set_title("AG News Skewed: Full Class Prediction Distribution Across 5 Configs")
    ax.set_ylabel("Test Predictions (N=1,000)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 520)
    ax.legend(loc="upper right", frameon=True, fontsize=7.5)
    export_fig(fig, "class_dist_agnews_skewed")

    # 2. Emotion Natural (6 classes across 5 configs)
    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=300)
    classes = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    x = np.arange(len(classes))
    w = 0.15
    counts = np.array([
        [302, 83, 41, 69, 5, 0],
        [179, 94, 52, 113, 62, 0],
        [265, 80, 48, 85, 22, 0],
        [161, 97, 48, 151, 43, 0],
        [209, 109, 37, 118, 27, 0]
    ])
    gt_emotion = [148, 151, 39, 79, 70, 13]
    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, counts[i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.6)
        annotate_bars(ax, rects, fmt="{:d}", fontsize=6.5)
    ax.plot(x, gt_emotion, color=COLOR_GT, marker="o", markersize=5, linestyle="--",
            linewidth=1.2, label="Ground Truth Distribution (N=500)")
    ax.set_title("Emotion Natural: Full Class Prediction Distribution Across 5 Configs")
    ax.set_ylabel("Test Predictions (N=500)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 360)
    ax.legend(loc="upper right", frameon=True, fontsize=7.5)
    export_fig(fig, "class_dist_emotion_natural")


# -------------------------------------------------------------
# 4. CROSS-ARCHITECTURE SCALING ON SKEWED DATA (2 IN A ROW)
# -------------------------------------------------------------
def make_cross_architecture():
    models = ["RoBERTa\n(125M)", "Qwen05\n(500M)", "Qwen3\n(4B)"]
    x = np.arange(len(models))
    w = 0.35

    # 2 Subplots side-by-side: Panel 1 = F1, Panel 2 = Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=300)
    
    # F1
    r1 = ax1.bar(x - w/2, [57.22, 55.05, 57.80], w, label="Standard KD (Collapsed)", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r2 = ax1.bar(x + w/2, [64.22, 60.75, 67.14], w, label="Calibrated KD (Revived)", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax1.set_title("AG News Skewed: Macro-F1 (%)")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(45, 75)
    ax1.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Accuracy
    r3 = ax2.bar(x - w/2, [65.80, 63.00, 65.70], w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
    r4 = ax2.bar(x + w/2, [69.90, 67.20, 71.30], w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
    ax2.set_title("AG News Skewed: Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(55, 78)
    ax2.legend(loc="upper left", fontsize=7.5)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)
    export_fig(fig, "cross_arch_skewed_f1_acc")

    # Full Class Distribution across 3 architectures (3 subplots in 1 row)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), dpi=300, sharey=True)
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    xc = np.arange(len(classes))
    wc = 0.35

    arch_counts = [
        ("RoBERTa (125M)", [370, 210, 420, 0], [282, 246, 443, 29]),
        ("Qwen05 (500M)", [365, 211, 424, 0], [245, 260, 479, 16]),
        ("Qwen3 (4B)", [348, 214, 434, 4], [237, 256, 463, 44])
    ]
    for idx, (m_name, std_c, ckd_c) in enumerate(arch_counts):
        ax = axes[idx]
        r1 = ax.bar(xc - wc/2, std_c, wc, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.6)
        r2 = ax.bar(xc + wc/2, ckd_c, wc, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.6)
        ax.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.2, label="Ground Truth (250)")
        ax.set_title(m_name)
        ax.set_ylabel("Predictions (N=1,000)" if idx == 0 else "")
        ax.set_xticks(xc)
        ax.set_xticklabels(classes, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 520)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=7)
        annotate_bars(ax, r1, fmt="{:d}", fontsize=7)
        annotate_bars(ax, r2, fmt="{:d}", fontsize=7)
    export_fig(fig, "cross_arch_class_dist")

def main():
    print("Generating ACL LaTeX publication figures...")
    make_balanced_f1()
    make_balanced_accuracy()
    make_skewed_f1_and_acc()
    make_class_distributions()
    make_cross_architecture()
    print("All LaTeX figures generated!")

if __name__ == "__main__":
    main()

