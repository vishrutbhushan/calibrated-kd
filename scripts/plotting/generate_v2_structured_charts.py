"""
generate_v2_structured_charts.py
--------------------------------
Generates unified, structured publication-grade charts (300 DPI) adhering strictly to:
1. Strict Metric Separation: Accuracy and Macro-F1 are never mixed on the same plot.
2. Complete Class Visibility: Class distribution charts always show ALL classes together
   with the dashed Ground Truth reference line/points (no single-class isolated charts).
3. Benchmark Isolation: Balanced and Non-Balanced (Skewed/Natural) are kept strictly separate.
4. Architectural Isolation: Cross-architecture comparisons are cleanly partitioned.
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = ROOT / "figures" / "v2"
BALANCED_DIR = V2_DIR / "balanced"
SKEWED_DIR = V2_DIR / "skewed"
CROSS_DIR = V2_DIR / "cross_arch"

for d in [BALANCED_DIR, SKEWED_DIR, CROSS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
})

# Standard Palette
COLOR_STD = "#e63946"     # Crimson Red (Standard KD)
COLOR_CKD = "#2a9d8f"     # Emerald Teal (Calibrated KD / CKD)
COLOR_POST = "#f4a261"    # Warm Amber (Post-Hoc Output Cal)
COLOR_BATCH = "#457b9d"   # Steel Blue (Batch Cal KD)
COLOR_PRIOR = "#1d3557"   # Deep Navy (Prior-Aware CKD)
COLOR_GT = "#111827"      # Dark Slate for Ground Truth

CONFIG_COLORS = [COLOR_STD, COLOR_POST, COLOR_BATCH, COLOR_CKD, COLOR_PRIOR]
CONFIG_NAMES = ["Standard KD", "Post-Hoc Cal", "Batch Cal KD", "Calibrated KD", "Prior-Aware CKD"]

def annotate_bars(ax, rects, fmt="{:.1f}%", dy=1.0, fontsize=8, fontweight="bold", color="#111827"):
    for r in rects:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -8),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=fontsize, fontweight=fontweight, color=color)

def save_and_mirror(fig, out_path):
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out_path, ARTIFACT_DIR / out_path.name)
    print(f"[✓] Generated & Mirrored: {out_path.name}")


# ==============================================================================
# SECTION A: BALANCED BENCHMARKS SUITE
# ==============================================================================

def plot_balanced_f1_by_architecture():
    """Macro-F1 ONLY across 4 Balanced Datasets by Student Architecture"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300, sharey=False)
    fig.suptitle("Balanced Benchmarks Suite: Macro-F1 Score (%) Across Student Architectures", y=1.03)

    tasks = ["Emotion (6 Cl)", "AG News (4 Cl)", "TREC-6 (6 Cl)", "SST-2 (2 Cl Control)"]
    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer)", "Qwen2.5-0.5B\n(8× Comp)", "RoBERTa-base\n(32× Comp)"]
    x = np.arange(len(models))
    w = 0.35

    f1_data = {
        "Emotion": ([34.92, 37.59, 30.80, 36.38], [35.62, 39.03, 32.51, 39.63], (25, 45)),
        "AG News": ([70.24, 69.91, 60.53, 68.78], [67.45, 74.40, 62.93, 70.23], (55, 80)),
        "TREC-6": ([69.93, 75.08, 59.70, 68.36], [69.21, 77.12, 62.98, 69.43], (55, 83)),
        "SST-2": ([84.21, 80.92, 82.66, 87.90], [84.74, 80.92, 82.66, 87.41], (75, 93))
    }

    for idx, (task_key, (std, ckd, y_lim)) in enumerate(f1_data.items()):
        ax = axes[idx]
        r1 = ax.bar(x - w/2, std, w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
        r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
        ax.set_title(tasks[idx])
        ax.set_ylabel("Macro-F1 (%)" if idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(y_lim)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=8.5)
        annotate_bars(ax, r1)
        annotate_bars(ax, r2)

    save_and_mirror(fig, BALANCED_DIR / "f1_balanced_by_architecture.png")

def plot_balanced_accuracy_by_architecture():
    """Test Accuracy ONLY across 4 Balanced Datasets by Student Architecture"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300, sharey=False)
    fig.suptitle("Balanced Benchmarks Suite: Test Accuracy (%) Across Student Architectures", y=1.03)

    tasks = ["Emotion (6 Cl)", "AG News (4 Cl)", "TREC-6 (6 Cl)", "SST-2 (2 Cl Control)"]
    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer)", "Qwen2.5-0.5B\n(8× Comp)", "RoBERTa-base\n(32× Comp)"]
    x = np.arange(len(models))
    w = 0.35

    acc_data = {
        "Emotion": ([40.33, 40.67, 35.00, 39.67], [40.67, 43.33, 36.00, 44.33], (30, 49)),
        "AG News": ([72.60, 73.10, 66.30, 71.90], [69.60, 76.50, 67.20, 72.50], (60, 82)),
        "TREC-6": ([72.20, 78.20, 68.60, 73.20], [72.20, 80.00, 71.20, 73.80], (62, 85)),
        "SST-2": ([84.50, 81.00, 83.00, 88.00], [85.00, 81.00, 83.00, 87.50], (75, 93))
    }

    for idx, (task_key, (std, ckd, y_lim)) in enumerate(acc_data.items()):
        ax = axes[idx]
        r1 = ax.bar(x - w/2, std, w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
        r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
        ax.set_title(tasks[idx])
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(y_lim)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=8.5)
        annotate_bars(ax, r1)
        annotate_bars(ax, r2)

    save_and_mirror(fig, BALANCED_DIR / "accuracy_balanced_by_architecture.png")

def plot_balanced_class_distributions():
    """All Classes Combined with Ground Truth Line for Balanced Datasets"""
    # 1. AG News Balanced (4 classes)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    x = np.arange(len(classes))
    w = 0.25

    # Qwen3-4B predictions on balanced test set (N=1000)
    std_counts = [320, 241, 377, 62]
    ckd_counts = [226, 293, 392, 89]

    r1 = ax.bar(x - w/2, std_counts, w, label="Standard KD (Qwen3-4B)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd_counts, w, label="Calibrated KD (Qwen3-4B)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.5, label="Ground Truth Expected (250 / class)")

    ax.set_title("AG News (Balanced) — Full Class Prediction Distribution (N=1,000)")
    ax.set_ylabel("Number of Test Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 440)
    ax.legend(loc="upper right", frameon=True)
    annotate_bars(ax, r1, fmt="{:d}")
    annotate_bars(ax, r2, fmt="{:d}")
    save_and_mirror(fig, BALANCED_DIR / "class_dist_agnews_balanced.png")

    # 2. Emotion Balanced (6 classes)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    classes = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    x = np.arange(len(classes))

    # RoBERTa-base predictions on balanced test set (N=300)
    std_counts = [163, 50, 22, 45, 20, 0]
    ckd_counts = [86, 52, 23, 98, 41, 0]

    r1 = ax.bar(x - w/2, std_counts, w, label="Standard KD (RoBERTa-base)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd_counts, w, label="Calibrated KD (RoBERTa-base)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.axhline(50, color=COLOR_GT, linestyle="--", linewidth=1.5, label="Ground Truth Expected (50 / class)")

    ax.set_title("Emotion (Balanced) — Full Class Prediction Distribution (N=300)")
    ax.set_ylabel("Number of Test Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 190)
    ax.legend(loc="upper right", frameon=True)
    annotate_bars(ax, r1, fmt="{:d}")
    annotate_bars(ax, r2, fmt="{:d}")
    save_and_mirror(fig, BALANCED_DIR / "class_dist_emotion_balanced.png")

    # 3. TREC-6 Balanced (6 classes)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    classes = ["Abbr", "Entity", "Desc", "Human", "Loc", "Num"]
    x = np.arange(len(classes))
    gt_counts = [9, 94, 138, 65, 81, 113]
    std_counts = [18, 48, 178, 20, 86, 150]
    ckd_counts = [18, 58, 159, 28, 86, 151]

    r1 = ax.bar(x - w/2, std_counts, w, label="Standard KD (Qwen3-4B)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd_counts, w, label="Calibrated KD (Qwen3-4B)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.plot(x, gt_counts, color=COLOR_GT, marker="o", linestyle="--", linewidth=1.5, label="Ground Truth Test Counts")

    ax.set_title("TREC-6 (Balanced) — Full Class Prediction Distribution (N=500)")
    ax.set_ylabel("Number of Test Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 210)
    ax.legend(loc="upper left", frameon=True)
    annotate_bars(ax, r1, fmt="{:d}")
    annotate_bars(ax, r2, fmt="{:d}")
    save_and_mirror(fig, BALANCED_DIR / "class_dist_trec_balanced.png")

    # 4. SST-2 Balanced Control (2 classes)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    classes = ["Negative", "Positive"]
    x = np.arange(len(classes))
    std_counts = [118, 82]
    ckd_counts = [117, 83]

    r1 = ax.bar(x - w/2, std_counts, w, label="Standard KD (RoBERTa-base)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd_counts, w, label="Calibrated KD (RoBERTa-base)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.axhline(100, color=COLOR_GT, linestyle="--", linewidth=1.5, label="Ground Truth Expected (100 / class)")

    ax.set_title("SST-2 (Balanced Control) — Class Distribution (N=200)")
    ax.set_ylabel("Number of Test Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(50, 140)
    ax.legend(loc="upper right", frameon=True)
    annotate_bars(ax, r1, fmt="{:d}")
    annotate_bars(ax, r2, fmt="{:d}")
    save_and_mirror(fig, BALANCED_DIR / "class_dist_sst2_balanced.png")


# ==============================================================================
# SECTION B: SKEWED & NATURAL BENCHMARKS SUITE (5 CONFIGS ON ROBERTA-BASE)
# ==============================================================================

def plot_skewed_f1_by_configuration():
    """Macro-F1 ONLY across 4 Skewed/Natural Datasets for all 5 Configurations"""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    tasks = ["Emotion Natural\n(Saravia Split)", "TREC-6 Natural\n(CogComp Split)", "AG News Skewed\n(10% Minority)", "SST-2 Skewed\n(80/20 Imbalance)"]
    x = np.arange(len(tasks))
    w = 0.15

    # 5 configs: Std, Post-Hoc, Batch Cal, CKD, Prior-Aware
    f1_matrix = np.array([
        [35.02, 44.60, 41.29, 44.58, 42.74],  # Emotion
        [64.62, 68.91, 70.15, 75.09, 74.02],  # TREC-6
        [57.22, 65.53, 57.15, 64.22, 58.83],  # AG News
        [65.03, 62.97, 67.03, 78.69, 84.36],  # SST-2
    ])

    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, f1_matrix[:, i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=7.5)

    ax.set_title("Non-Balanced (Skewed & Natural) Suite: Macro-F1 (%) Across 5 Configurations (RoBERTa-base)")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontweight="bold")
    ax.set_ylim(25, 92)
    ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=2)
    save_and_mirror(fig, SKEWED_DIR / "f1_skewed_by_configuration.png")

def plot_skewed_accuracy_by_configuration():
    """Test Accuracy ONLY across 4 Skewed/Natural Datasets for all 5 Configurations"""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    tasks = ["Emotion Natural\n(Saravia Split)", "TREC-6 Natural\n(CogComp Split)", "AG News Skewed\n(10% Minority)", "SST-2 Skewed\n(80/20 Imbalance)"]
    x = np.arange(len(tasks))
    w = 0.15

    acc_matrix = np.array([
        [50.80, 56.20, 54.40, 56.60, 56.20],  # Emotion
        [74.60, 76.60, 75.60, 77.20, 76.40],  # TREC-6
        [65.80, 70.40, 65.70, 69.90, 67.20],  # AG News
        [68.50, 67.00, 70.00, 79.50, 84.50],  # SST-2
    ])

    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, acc_matrix[:, i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=7.5)

    ax.set_title("Non-Balanced (Skewed & Natural) Suite: Test Accuracy (%) Across 5 Configurations (RoBERTa-base)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontweight="bold")
    ax.set_ylim(40, 92)
    ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=2)
    save_and_mirror(fig, SKEWED_DIR / "accuracy_skewed_by_configuration.png")

def plot_skewed_class_distributions_all_classes():
    """Combined Class Distribution with Ground Truth line for ALL classes across the 4 Skewed Tasks"""
    # 1. AG News Skewed (4 classes across 5 configs)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    classes = ["World (50% Train)", "Sports (25% Train)", "Business (15% Train)", "Sci/Tech (10% Train)"]
    x = np.arange(len(classes))
    w = 0.15

    # Predictions matrix for each config: [World, Sports, Business, Sci/Tech]
    # Test set is 1,000 items (250/class)
    counts = np.array([
        [370, 210, 420, 0],   # Standard KD (0 on Sci/Tech!)
        [270, 247, 449, 34],  # Post-Hoc Cal
        [366, 210, 424, 0],   # Batch Cal (0 on Sci/Tech!)
        [282, 246, 443, 29],  # CKD (29 on Sci/Tech!)
        [330, 238, 429, 3]    # Prior-Aware CKD
    ])

    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, counts[i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax, rects, fmt="{:d}", fontsize=7)

    ax.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.5, label="Ground Truth Expected (250 / class in Test Set)")
    ax.set_title("AG News Skewed — Full Class Prediction Distribution Across 5 Configs (RoBERTa-base)")
    ax.set_ylabel("Test Prediction Count (out of 1,000)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 520)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    save_and_mirror(fig, SKEWED_DIR / "class_dist_agnews_skewed_configs.png")

    # 2. Emotion Natural (6 classes across 5 configs)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
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
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax, rects, fmt="{:d}", fontsize=7)

    ax.plot(x, gt_emotion, color=COLOR_GT, marker="o", markersize=6, linestyle="--",
            linewidth=1.5, label="Ground Truth Natural Test Counts (N=500)")
    ax.set_title("Emotion Natural — Full Class Prediction Distribution Across 5 Configs (RoBERTa-base)")
    ax.set_ylabel("Test Prediction Count (out of 500)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 360)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    save_and_mirror(fig, SKEWED_DIR / "class_dist_emotion_natural_configs.png")

    # 3. TREC-6 Natural (6 classes across 5 configs)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    classes = ["Abbr (1.6%)", "Entity", "Desc", "Human", "Loc", "Num"]
    x = np.arange(len(classes))
    w = 0.15

    counts = np.array([
        [2, 24, 194, 43, 87, 150],
        [15, 10, 180, 57, 88, 150],
        [6, 21, 184, 45, 87, 157],
        [10, 43, 170, 38, 85, 154],
        [8, 43, 176, 36, 81, 156]
    ])
    gt_trec = [9, 94, 138, 65, 81, 113]

    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, counts[i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax, rects, fmt="{:d}", fontsize=7)

    ax.plot(x, gt_trec, color=COLOR_GT, marker="o", markersize=6, linestyle="--",
            linewidth=1.5, label="Ground Truth Natural Test Counts (N=500)")
    ax.set_title("TREC-6 Natural — Full Class Prediction Distribution Across 5 Configs (RoBERTa-base)")
    ax.set_ylabel("Test Prediction Count (out of 500)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 230)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    save_and_mirror(fig, SKEWED_DIR / "class_dist_trec_natural_configs.png")

    # 4. SST-2 Skewed (2 classes across 5 configs)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    classes = ["Negative (20% Train)", "Positive (80% Train)"]
    x = np.arange(len(classes))
    w = 0.15

    counts = np.array([
        [163, 37],
        [166, 34],
        [160, 40],
        [139, 61],
        [119, 81]
    ])

    for i in range(5):
        rects = ax.bar(x + (i - 2) * w, counts[i], w, label=CONFIG_NAMES[i],
                       color=CONFIG_COLORS[i], edgecolor="black", linewidth=0.8)
        annotate_bars(ax, rects, fmt="{:d}", fontsize=7.5)

    ax.axhline(100, color=COLOR_GT, linestyle="--", linewidth=1.5, label="Ground Truth Expected (100 / class in Test Set)")
    ax.set_title("SST-2 Skewed — Full Class Prediction Distribution Across 5 Configs (RoBERTa-base)")
    ax.set_ylabel("Test Prediction Count (out of 200)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 195)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    save_and_mirror(fig, SKEWED_DIR / "class_dist_sst2_skewed_configs.png")


# ==============================================================================
# SECTION C: CROSS-ARCHITECTURE SCALING ON SKEWED BENCHMARKS
# ==============================================================================

def plot_cross_architecture_skewed():
    """Cross-Architecture Analysis on AG News Skewed: F1, Accuracy, and ALL Classes Combined"""
    models = ["RoBERTa-base\n(125M Encoder)", "Qwen2.5-0.5B\n(500M Causal)", "Qwen3-4B\n(4B Peer Causal)"]
    x = np.arange(len(models))
    w = 0.35

    # 1. Macro-F1 ONLY
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    std_f1 = [57.22, 55.05, 57.80]
    ckd_f1 = [64.22, 60.75, 67.14]
    r1 = ax.bar(x - w/2, std_f1, w, label="Standard KD (Collapsed)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd_f1, w, label="Calibrated KD (Revived)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.set_title("AG News Skewed: Macro-F1 (%) Across Student Capacities")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(45, 75)
    ax.legend(loc="upper left", frameon=True)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    save_and_mirror(fig, CROSS_DIR / "f1_cross_arch_agnews_skewed.png")

    # 2. Accuracy ONLY
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    std_acc = [65.80, 63.00, 65.70]
    ckd_acc = [69.90, 67.20, 71.30]
    r1 = ax.bar(x - w/2, std_acc, w, label="Standard KD (Collapsed)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd_acc, w, label="Calibrated KD (Revived)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.set_title("AG News Skewed: Test Accuracy (%) Across Student Capacities")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(55, 78)
    ax.legend(loc="upper left", frameon=True)
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)
    save_and_mirror(fig, CROSS_DIR / "accuracy_cross_arch_agnews_skewed.png")

    # 3. ALL Classes Combined for All 3 Architectures with Ground Truth Line
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300, sharey=True)
    fig.suptitle("AG News Skewed: Full Class Prediction Distributions Across Student Capacities (N=1,000)", y=1.03)

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
        r1 = ax.bar(xc - wc/2, std_c, wc, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
        r2 = ax.bar(xc + wc/2, ckd_c, wc, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
        ax.axhline(250, color=COLOR_GT, linestyle="--", linewidth=1.5, label="Ground Truth (250 / cl)")
        ax.set_title(m_name)
        ax.set_ylabel("Prediction Count (out of 1,000)" if idx == 0 else "")
        ax.set_xticks(xc)
        ax.set_xticklabels(classes, fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, 520)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        annotate_bars(ax, r1, fmt="{:d}", fontsize=7.5)
        annotate_bars(ax, r2, fmt="{:d}", fontsize=7.5)

    save_and_mirror(fig, CROSS_DIR / "class_dist_cross_arch_agnews_skewed.png")


def main():
    print("Generating v2 clean structured figures...")
    # Section A: Balanced
    plot_balanced_f1_by_architecture()
    plot_balanced_accuracy_by_architecture()
    plot_balanced_class_distributions()

    # Section B: Skewed
    plot_skewed_f1_by_configuration()
    plot_skewed_accuracy_by_configuration()
    plot_skewed_class_distributions_all_classes()

    # Section C: Cross-Architecture Skewed
    plot_cross_architecture_skewed()
    print("All v2 structured figures successfully generated!")

if __name__ == "__main__":
    main()

