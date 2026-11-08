"""
generate_simplified_evaluation_charts.py
----------------------------------------
Generates simplified, ultra-clear, publication-ready charts (300 DPI)
showing the exact performance of each configuration without mixing datasets
and strictly separating Balanced from Non-Balanced (Skewed/Natural) datasets.

Generated Charts:
1. Balanced Suite (Individual):
   - agnews_balanced.png
   - emotion_balanced.png
   - trec_balanced.png
   - sst2_balanced.png
2. Skewed / Natural Suite (Individual):
   - agnews_skewed_configs.png
   - emotion_natural_configs.png
   - trec_natural_configs.png
   - sst2_skewed_configs.png
   - cross_architecture_agnews_skewed.png
3. Combined Comparative Summaries:
   - balanced_master_comparison.png (Balanced datasets only)
   - skewed_master_comparison.png (Non-balanced datasets only)

All figures are copied directly to the artifact directory for instant viewing.
"""

import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "simplified"
BALANCED_DIR = FIG_DIR / "balanced"
SKEWED_DIR = FIG_DIR / "skewed"
COMBINED_DIR = FIG_DIR / "combined"

for d in [BALANCED_DIR, SKEWED_DIR, COMBINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

# Standardized Color Palette
COLOR_STD = "#e63946"       # Crimson Red (Standard KD)
COLOR_POST = "#f4a261"      # Warm Amber (Post-Hoc Output Cal)
COLOR_BATCH = "#457b9d"     # Muted Steel Blue (Batch Cal KD)
COLOR_CKD = "#2a9d8f"       # Emerald Teal (Calibrated KD)
COLOR_PRIOR = "#1d3557"     # Deep Navy (Prior-Aware CKD)

COLOR_125M = "#457b9d"      # Steel Blue (RoBERTa 125M)
COLOR_500M = "#f4a261"      # Warm Amber (Qwen05 500M)
COLOR_4B = "#2a9d8f"        # Emerald Teal (Qwen3 4B)

def annotate_bars(ax, rects, fmt="{:.1f}%", dy=1.2, fontsize=8.5, fontweight="bold", color="#111827"):
    for r in rects:
        h = r.get_height()
        va = "bottom" if h >= 0 else "top"
        y = h + (dy if h >= 0 else -dy)
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -8),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=fontsize, fontweight=fontweight, color=color)

# ==============================================================================
# 1. BALANCED BENCHMARK CHARTS (INDIVIDUAL PER DATASET)
# ==============================================================================

def plot_agnews_balanced():
    """AG News (Balanced 4-Class Topic Classification) across Architectures"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    fig.suptitle("AG News (Balanced 4-Class) — Performance Across Architectures", fontsize=13, y=1.02)
    
    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer Causal)", "Qwen2.5-0.5B\n(8× Causal)", "RoBERTa-base\n(32× Encoder)"]
    x = np.arange(len(models))
    w = 0.35

    # Accuracy
    std_acc = [72.60, 73.10, 62.40, 70.00]
    ckd_acc = [69.60, 76.50, 63.30, 70.60]
    r1 = ax1.bar(x - w/2, std_acc, w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, ckd_acc, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax1.set_title("Test Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(55, 82)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Macro-F1
    std_f1 = [70.24, 69.91, 60.53, 68.78]
    ckd_f1 = [67.45, 74.40, 62.93, 70.23]
    r3 = ax2.bar(x - w/2, std_f1, w, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + w/2, ckd_f1, w, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax2.set_title("Macro-F1 Score (%)")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(55, 80)
    ax2.legend(loc="upper left", frameon=True)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)

    plt.tight_layout()
    out = BALANCED_DIR / "agnews_balanced.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "agnews_balanced.png")
    print(f"[✓] Saved {out.name}")

def plot_emotion_balanced():
    """Emotion (Balanced 6-Class Affect Classification) — Configs on RoBERTa & Architecture comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    fig.suptitle("Emotion (Balanced 6-Class) — Configuration & Architecture Comparison", fontsize=13, y=1.02)

    # Panel 1: 5 Configurations on RoBERTa-base
    configs = ["Standard KD", "Post-Hoc Null", "Post-Hoc Mean", "Batch Cal KD", "Calibrated KD"]
    f1_configs = [39.25, 29.43, 42.34, 37.68, 43.29]
    acc_configs = [42.00, 35.33, 46.33, 41.00, 47.67]
    colors_c = [COLOR_STD, COLOR_POST, COLOR_BATCH, "#8ecae6", COLOR_CKD]
    
    x1 = np.arange(len(configs))
    w1 = 0.38
    r_acc = ax1.bar(x1 - w1/2, acc_configs, w1, label="Accuracy", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r_f1 = ax1.bar(x1 + w1/2, f1_configs, w1, label="Macro-F1", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax1.set_title("(A) RoBERTa-base (125M) Across Distillation Configs")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(configs, rotation=15, ha="right")
    ax1.set_ylim(20, 55)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r_acc)
    annotate_bars(ax1, r_f1)

    # Panel 2: Standard KD vs CKD Across Architectures
    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer)", "Qwen2.5-0.5B\n(8× Comp)", "RoBERTa-base\n(32× Comp)"]
    x2 = np.arange(len(models))
    w2 = 0.35
    std_f1_arch = [34.92, 37.59, 30.80, 36.38]
    ckd_f1_arch = [35.62, 39.03, 32.51, 39.63]
    r_std = ax2.bar(x2 - w2/2, std_f1_arch, w2, label="Standard KD (Raw)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r_ckd = ax2.bar(x2 + w2/2, ckd_f1_arch, w2, label="Calibrated KD (CKD)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax2.set_title("(B) Macro-F1 Across Student Architectures")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models)
    ax2.set_ylim(25, 45)
    ax2.legend(loc="upper left", frameon=True)
    annotate_bars(ax2, r_std)
    annotate_bars(ax2, r_ckd)

    plt.tight_layout()
    out = BALANCED_DIR / "emotion_balanced.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "emotion_balanced.png")
    print(f"[✓] Saved {out.name}")

def plot_trec_balanced():
    """TREC-6 (Balanced 6-Class Question Classification)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    fig.suptitle("TREC-6 (Balanced 6-Class) — Performance Across Architectures", fontsize=13, y=1.02)

    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer Causal)", "Qwen2.5-0.5B\n(8× Causal)", "RoBERTa-base\n(32× Encoder)"]
    x = np.arange(len(models))
    w = 0.35

    std_acc = [72.20, 78.20, 67.40, 71.80]
    ckd_acc = [72.20, 80.00, 68.60, 72.20]
    r1 = ax1.bar(x - w/2, std_acc, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, ckd_acc, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax1.set_title("Test Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(60, 85)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    std_f1 = [69.93, 75.08, 59.70, 68.36]
    ckd_f1 = [69.21, 77.12, 62.98, 69.43]
    r3 = ax2.bar(x - w/2, std_f1, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + w/2, ckd_f1, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax2.set_title("Macro-F1 Score (%)")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(55, 82)
    ax2.legend(loc="upper left", frameon=True)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)

    plt.tight_layout()
    out = BALANCED_DIR / "trec_balanced.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "trec_balanced.png")
    print(f"[✓] Saved {out.name}")

def plot_sst2_balanced():
    """SST-2 (Balanced 2-Class Sentiment Classification) — Safety Control"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    fig.suptitle("SST-2 (Balanced Binary Sentiment) — Calibration Safety Control", fontsize=13, y=1.02)

    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer Causal)", "Qwen2.5-0.5B\n(8× Causal)", "RoBERTa-base\n(32× Encoder)"]
    x = np.arange(len(models))
    w = 0.35

    std_acc = [84.50, 81.00, 83.00, 88.00]
    ckd_acc = [85.00, 81.00, 83.00, 87.50]
    r1 = ax1.bar(x - w/2, std_acc, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, ckd_acc, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax1.set_title("Test Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(70, 93)
    ax1.legend(loc="lower right", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    std_f1 = [84.21, 80.92, 82.66, 87.90]
    ckd_f1 = [84.74, 80.92, 82.66, 87.41]
    r3 = ax2.bar(x - w/2, std_f1, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + w/2, ckd_f1, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax2.set_title("Macro-F1 Score (%)")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(70, 93)
    ax2.legend(loc="lower right", frameon=True)
    annotate_bars(ax2, r3)
    annotate_bars(ax2, r4)

    plt.tight_layout()
    out = BALANCED_DIR / "sst2_balanced.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "sst2_balanced.png")
    print(f"[✓] Saved {out.name}")


# ==============================================================================
# 2. SKEWED & NATURAL BENCHMARK CHARTS (INDIVIDUAL PER DATASET)
# ==============================================================================

def plot_agnews_skewed_configs():
    """AG News Skewed (10% Sci/Tech Minority) — 5 Configurations on RoBERTa-base"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("AG News Skewed (50/25/15/10% Split) — 5 Distillation Configurations on RoBERTa-base", fontsize=13, y=1.02)

    configs = [
        "Standard KD\n(Raw Targets)",
        "Post-Hoc Output Cal\n(Inference Shift)",
        "Batch Cal KD\n(Data-Mean Prior)",
        "Calibrated KD\n(CKD Null Prompt)",
        "Prior-Aware CKD\n(CKD + Log-Prior)"
    ]
    x = np.arange(len(configs))
    w = 0.35

    # Panel 1: Accuracy & F1
    acc = [65.80, 70.40, 65.70, 69.90, 67.20]
    f1 = [57.22, 65.53, 57.15, 64.22, 58.83]

    r1 = ax1.bar(x - w/2, acc, w, label="Accuracy (%)", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, f1, w, label="Macro-F1 (%)", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax1.set_title("(A) Accuracy and Macro-F1 Across Configurations")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylim(50, 76)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Panel 2: Minority Class Sci/Tech Predictions (Out of 1000 items)
    preds_scitech = [0, 34, 0, 29, 3]
    bar_colors = [COLOR_STD, COLOR_POST, COLOR_STD, COLOR_CKD, COLOR_PRIOR]
    r3 = ax2.bar(x, preds_scitech, 0.5, color=bar_colors, edgecolor="black", linewidth=1.0)
    ax2.set_title("(B) Minority Class (Sci/Tech) Test Predictions (Expected ~100 / 1000)")
    ax2.set_ylabel("Number of Predictions (out of 1,000)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylim(0, 45)

    # Special callout annotations
    ax2.annotate("100% Total\nCollapse (0)", xy=(0, 1), xytext=(0, 12),
                 ha="center", fontsize=8.5, fontweight="bold", color=COLOR_STD,
                 arrowprops=dict(arrowstyle="->", color=COLOR_STD, lw=1.2))
    ax2.annotate("100% Total\nCollapse (0)", xy=(2, 1), xytext=(2, 12),
                 ha="center", fontsize=8.5, fontweight="bold", color=COLOR_STD,
                 arrowprops=dict(arrowstyle="->", color=COLOR_STD, lw=1.2))
    ax2.annotate("Revived to 29\n(+7.0 pp F1)", xy=(3, 29), xytext=(3, 36),
                 ha="center", fontsize=8.5, fontweight="bold", color=COLOR_CKD,
                 arrowprops=dict(arrowstyle="->", color=COLOR_CKD, lw=1.2))

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = SKEWED_DIR / "agnews_skewed_configs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "agnews_skewed_configs.png")
    print(f"[✓] Saved {out.name}")

def plot_emotion_natural_configs():
    """Emotion Natural (Saravia Natural Distribution) — 5 Configurations on RoBERTa-base"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("Emotion Natural (Saravia Distribution) — 5 Distillation Configurations on RoBERTa-base", fontsize=13, y=1.02)

    configs = [
        "Standard KD\n(Raw Targets)",
        "Post-Hoc Output Cal\n(Inference Shift)",
        "Batch Cal KD\n(Data-Mean Prior)",
        "Calibrated KD\n(CKD Null Prompt)",
        "Prior-Aware CKD\n(CKD + Log-Prior)"
    ]
    x = np.arange(len(configs))
    w = 0.35

    acc = [50.80, 56.20, 54.40, 56.60, 56.20]
    f1 = [35.02, 44.60, 41.29, 44.58, 42.74]

    r1 = ax1.bar(x - w/2, acc, w, label="Accuracy (%)", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, f1, w, label="Macro-F1 (%)", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax1.set_title("(A) Accuracy and Macro-F1 Across Configurations")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylim(28, 62)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Panel 2: Sadness Overprediction vs Fear Minority Recovery (500 Test Items)
    sadness_counts = [302, 179, 265, 161, 209]
    fear_counts = [5, 62, 22, 43, 27]

    r3 = ax2.bar(x - w/2, sadness_counts, w, label="Sadness (True: ~130 / 500)", color="#e76f51", edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + w/2, fear_counts, w, label="Fear (True: ~58 / 500)", color="#2a9d8f", edgecolor="black", linewidth=0.8)
    ax2.set_title("(B) Prediction Bias Mitigation: Sadness Suppression & Fear Recovery")
    ax2.set_ylabel("Test Prediction Count (out of 500)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylim(0, 350)
    ax2.legend(loc="upper right", frameon=True)

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in r4:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out = SKEWED_DIR / "emotion_natural_configs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "emotion_natural_configs.png")
    print(f"[✓] Saved {out.name}")

def plot_trec_natural_configs():
    """TREC-6 Natural (1.6% Abbreviation Minority) — 5 Configurations on RoBERTa-base"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("TREC-6 Natural (CogComp Question Distribution) — 5 Configurations on RoBERTa-base", fontsize=13, y=1.02)

    configs = [
        "Standard KD\n(Raw Targets)",
        "Post-Hoc Output Cal\n(Inference Shift)",
        "Batch Cal KD\n(Data-Mean Prior)",
        "Calibrated KD\n(CKD Null Prompt)",
        "Prior-Aware CKD\n(CKD + Log-Prior)"
    ]
    x = np.arange(len(configs))
    w = 0.35

    acc = [74.60, 76.60, 75.60, 77.20, 76.40]
    f1 = [64.62, 68.91, 70.15, 75.09, 74.02]

    r1 = ax1.bar(x - w/2, acc, w, label="Accuracy (%)", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, f1, w, label="Macro-F1 (%)", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax1.set_title("(A) Accuracy and Macro-F1 Across Configurations")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylim(55, 83)
    ax1.legend(loc="lower right", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Panel 2: Abbreviation Minority Predictions (True: 9 / 500)
    abbr_preds = [2, 15, 6, 10, 8]
    bar_colors = [COLOR_STD, COLOR_POST, COLOR_BATCH, COLOR_CKD, COLOR_PRIOR]
    r3 = ax2.bar(x, abbr_preds, 0.45, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax2.axhline(9, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="Ground Truth Expected (9 / 500)")
    ax2.set_title("(B) Rare Minority (Abbreviation, 1.6%) Predictions")
    ax2.set_ylabel("Abbr Prediction Count (out of 500)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylim(0, 20)
    ax2.legend(loc="upper left", frameon=True)

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = SKEWED_DIR / "trec_natural_configs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "trec_natural_configs.png")
    print(f"[✓] Saved {out.name}")

def plot_sst2_skewed_configs():
    """SST-2 Skewed (80% Positive / 20% Negative) — 5 Configurations on RoBERTa-base"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("SST-2 Skewed (80% Positive / 20% Negative) — 5 Configurations on RoBERTa-base", fontsize=13, y=1.02)

    configs = [
        "Standard KD\n(Raw Targets)",
        "Post-Hoc Output Cal\n(Inference Shift)",
        "Batch Cal KD\n(Data-Mean Prior)",
        "Calibrated KD\n(CKD Null Prompt)",
        "Prior-Aware CKD\n(CKD + Log-Prior)"
    ]
    x = np.arange(len(configs))
    w = 0.35

    acc = [68.50, 67.00, 70.00, 79.50, 84.50]
    f1 = [65.03, 62.97, 67.03, 78.69, 84.36]

    r1 = ax1.bar(x - w/2, acc, w, label="Accuracy (%)", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, f1, w, label="Macro-F1 (%)", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax1.set_title("(A) Accuracy and Macro-F1 Across Configurations")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylim(55, 90)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Panel 2: Predicted Sentiment Balance (Positive vs Negative out of 200 items, True is 100/100 test items)
    neg_preds = [163, 166, 160, 139, 119]
    pos_preds = [37, 34, 40, 61, 81]

    r3 = ax2.bar(x - w/2, neg_preds, w, label="Negative Predictions (Test: 100)", color="#e76f51", edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + w/2, pos_preds, w, label="Positive Predictions (Test: 100)", color="#2a9d8f", edgecolor="black", linewidth=0.8)
    ax2.set_title("(B) Predicted Class Balance (True Test Split: 100 Neg / 100 Pos)")
    ax2.set_ylabel("Prediction Count (out of 200)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylim(0, 190)
    ax2.legend(loc="upper right", frameon=True)

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in r4:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out = SKEWED_DIR / "sst2_skewed_configs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "sst2_skewed_configs.png")
    print(f"[✓] Saved {out.name}")

def plot_cross_architecture_agnews_skewed():
    """Cross-Architecture Validation on AG News Skewed (RoBERTa 125M vs Qwen05 500M vs Qwen3 4B)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle("AG News Skewed (10% Minority) — Cross-Architecture Capacity Scaling", fontsize=13, y=1.02)

    models = [
        "RoBERTa-base\n(125M Encoder)",
        "Qwen2.5-0.5B\n(500M Causal LM)",
        "Qwen3-4B\n(4B Peer Causal)"
    ]
    x = np.arange(len(models))
    w = 0.35

    # Panel 1: Macro-F1 Improvement
    std_f1 = [57.22, 55.05, 57.80]
    ckd_f1 = [64.22, 60.75, 67.14]

    r1 = ax1.bar(x - w/2, std_f1, w, label="Standard KD (Collapsed)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + w/2, ckd_f1, w, label="Calibrated KD (Revived)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax1.set_title("(A) Macro-F1 Under 10% Minority Skew")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(45, 75)
    ax1.legend(loc="upper left", frameon=True)
    annotate_bars(ax1, r1)
    annotate_bars(ax1, r2)

    # Panel 2: Minority Class (Sci/Tech) Prediction Counts (Expected: ~100 / 1000)
    std_scitech = [0, 0, 4]
    ckd_scitech = [29, 16, 44]

    r3 = ax2.bar(x - w/2, std_scitech, w, label="Standard KD (0, 0, 4 Preds)", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + w/2, ckd_scitech, w, label="Calibrated KD (29, 16, 44 Preds)", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax2.set_title("(B) Sci/Tech Minority Recovery (1,000 Test Items)")
    ax2.set_ylabel("Sci/Tech Predictions (out of 1,000)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 55)
    ax2.legend(loc="upper left", frameon=True)

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLOR_STD)
    for r in r4:
        h = r.get_height()
        ax2.annotate(f"{h}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLOR_CKD)

    plt.tight_layout()
    out = SKEWED_DIR / "cross_architecture_agnews_skewed.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "cross_architecture_agnews_skewed.png")
    print(f"[✓] Saved {out.name}")


# ==============================================================================
# 3. MASTER COMBINED SUMMARY CHARTS (SEPARATING BALANCED VS SKEWED)
# ==============================================================================

def plot_balanced_master_comparison():
    """Master Comparison across ALL 4 BALANCED Datasets (Standard KD vs CKD)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle("Master Overview: Balanced Benchmarks Suite (Standard KD vs. Calibrated KD)", fontsize=15, y=0.98)

    models = ["Direct Teacher\n(Gemma 4B)", "Qwen3-4B\n(1× Peer)", "Qwen2.5-0.5B\n(8× Comp)", "RoBERTa-base\n(32× Comp)"]
    x = np.arange(len(models))
    w = 0.35

    # Subplot 1: Emotion Balanced
    ax = axes[0, 0]
    std = [34.92, 37.59, 30.80, 36.38]
    ckd = [35.62, 39.03, 32.51, 39.63]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.set_title("Emotion (Balanced 6-Class) — Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(25, 45)
    ax.legend(loc="upper left")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    # Subplot 2: AG News Balanced
    ax = axes[0, 1]
    std = [70.24, 69.91, 60.53, 68.78]
    ckd = [67.45, 74.40, 62.93, 70.23]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.set_title("AG News (Balanced 4-Class) — Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(55, 80)
    ax.legend(loc="upper left")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    # Subplot 3: TREC-6 Balanced
    ax = axes[1, 0]
    std = [69.93, 75.08, 59.70, 68.36]
    ckd = [69.21, 77.12, 62.98, 69.43]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.set_title("TREC-6 (Balanced 6-Class) — Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(55, 82)
    ax.legend(loc="upper left")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    # Subplot 4: SST-2 Balanced Control
    ax = axes[1, 1]
    std = [84.21, 80.92, 82.66, 87.90]
    ckd = [84.74, 80.92, 82.66, 87.41]
    r1 = ax.bar(x - w/2, std, w, label="Standard KD", color=COLOR_STD, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, ckd, w, label="Calibrated KD", color=COLOR_CKD, edgecolor="black", linewidth=0.8)
    ax.set_title("SST-2 (Balanced Binary Sentiment) — Macro-F1 (%) [Safety Control]")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(75, 92)
    ax.legend(loc="lower right")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    plt.tight_layout()
    out = COMBINED_DIR / "balanced_master_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "balanced_master_comparison.png")
    print(f"[✓] Saved {out.name}")

def plot_skewed_master_comparison():
    """Master Comparison across ALL 4 SKEWED/NATURAL Datasets (5 Configs on RoBERTa-base)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.suptitle("Master Overview: Skewed & Natural Benchmarks Suite (5 Configurations on RoBERTa-base)", fontsize=15, y=0.98)

    configs = ["Standard KD", "Post-Hoc Output", "Batch Cal KD", "Calibrated KD", "Prior-Aware CKD"]
    x = np.arange(len(configs))
    w = 0.35

    # Subplot 1: Emotion Natural
    ax = axes[0, 0]
    acc = [50.80, 56.20, 54.40, 56.60, 56.20]
    f1 = [35.02, 44.60, 41.29, 44.58, 42.74]
    r1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, f1, w, label="Macro-F1", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax.set_title("Emotion Natural (Saravia Distribution)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(28, 62)
    ax.legend(loc="upper left")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    # Subplot 2: TREC-6 Natural
    ax = axes[0, 1]
    acc = [74.60, 76.60, 75.60, 77.20, 76.40]
    f1 = [64.62, 68.91, 70.15, 75.09, 74.02]
    r1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, f1, w, label="Macro-F1", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax.set_title("TREC-6 Natural (1.6% Abbreviation Minority)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(55, 83)
    ax.legend(loc="lower right")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    # Subplot 3: AG News Skewed
    ax = axes[1, 0]
    acc = [65.80, 70.40, 65.70, 69.90, 67.20]
    f1 = [57.22, 65.53, 57.15, 64.22, 58.83]
    r1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, f1, w, label="Macro-F1", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax.set_title("AG News Skewed (10% Sci/Tech Minority)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(50, 76)
    ax.legend(loc="upper left")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    # Subplot 4: SST-2 Skewed
    ax = axes[1, 1]
    acc = [68.50, 67.00, 70.00, 79.50, 84.50]
    f1 = [65.03, 62.97, 67.03, 78.69, 84.36]
    r1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#3a86ff", edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + w/2, f1, w, label="Macro-F1", color="#ff006e", edgecolor="black", linewidth=0.8)
    ax.set_title("SST-2 Skewed (80% Positive / 20% Negative)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(55, 90)
    ax.legend(loc="upper left")
    annotate_bars(ax, r1)
    annotate_bars(ax, r2)

    plt.tight_layout()
    out = COMBINED_DIR / "skewed_master_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    shutil.copy(out, ARTIFACT_DIR / "skewed_master_comparison.png")
    print(f"[✓] Saved {out.name}")

def main():
    print("Generating simplified diagrams...")
    # Balanced (Separate)
    plot_agnews_balanced()
    plot_emotion_balanced()
    plot_trec_balanced()
    plot_sst2_balanced()
    
    # Skewed / Natural (Separate)
    plot_agnews_skewed_configs()
    plot_emotion_natural_configs()
    plot_trec_natural_configs()
    plot_sst2_skewed_configs()
    plot_cross_architecture_agnews_skewed()

    # Master Combined Overviews (Distinct Balanced vs. Skewed)
    plot_balanced_master_comparison()
    plot_skewed_master_comparison()
    print("All charts generated and copied to artifact directory.")

if __name__ == "__main__":
    main()

