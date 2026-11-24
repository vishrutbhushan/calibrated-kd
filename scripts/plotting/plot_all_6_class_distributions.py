"""
plot_all_6_class_distributions.py
---------------------------------
Generates a publication-grade 6-panel grid (2 rows x 3 columns) showing the
predicted class distributions across all 6 benchmark datasets.

Key Design Features:
  - Dotted black horizontal/step line showing the exact Ground Truth distribution.
  - Bars showing:
      1. Teacher Base (Gemma-3-4B Raw)
      2. Standard KD (RoBERTa-base)
      3. Batch-Calibrated KD (RoBERTa-base)
      4. Null Calibrated KD (Ours) (RoBERTa-base)
  - ColorBrewer Set2 / editorial palette.
  - Clear tilted class names, no overlapping text.
  - Clean publication formatting (DejaVu Serif, no em dashes).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "results" / "main_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10.0,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.5,
})

# Complete benchmark data extracted from json results and ground truth csvs (N=400 test set per benchmark)
benchmark_data = [
    {
        "id": "student_questions",
        "title": "(a) Student Questions (K=4)",
        "classes": ["Biology", "Physics", "Maths", "Chemistry"],
        "gold": [126, 123, 108, 43],
        "teacher": [100, 106, 183, 11],
        "std_kd": [91, 120, 189, 0],
        "batch_kd": [97, 120, 176, 7],
        "ckd": [0, 140, 209, 51]
    },
    {
        "id": "trec6",
        "title": "(b) TREC-6 Question Types (K=6)",
        "classes": ["Abbrev", "Entity", "Desc", "Human", "Location", "Numeric"],
        "gold": [7, 91, 94, 76, 62, 70],
        "teacher": [19, 37, 152, 90, 57, 45],
        "std_kd": [6, 35, 152, 99, 52, 56],
        "batch_kd": [9, 42, 138, 101, 51, 59],
        "ckd": [5, 37, 132, 117, 61, 48]
    },
    {
        "id": "app_reviews",
        "title": "(c) App Reviews (K=5)",
        "classes": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        "gold": [68, 24, 36, 55, 217],
        "teacher": [164, 48, 37, 22, 129],
        "std_kd": [157, 33, 60, 1, 149],
        "batch_kd": [145, 41, 50, 15, 149],
        "ckd": [105, 67, 24, 38, 166]
    },
    {
        "id": "twitter_emotion",
        "title": "(d) Twitter Emotion (K=6)",
        "classes": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "gold": [117, 135, 32, 54, 48, 14],
        "teacher": [209, 72, 33, 67, 19, 0],
        "std_kd": [249, 94, 18, 39, 0, 0],
        "batch_kd": [223, 90, 24, 63, 0, 0],
        "ckd": [187, 92, 24, 92, 5, 0]
    },
    {
        "id": "tweeteval_emotion",
        "title": "(e) TweetEval Emotion (K=4)",
        "classes": ["Anger", "Joy", "Optimism", "Sadness"],
        "gold": [167, 91, 36, 106],
        "teacher": [331, 46, 10, 13],
        "std_kd": [343, 52, 0, 5],
        "batch_kd": [335, 55, 1, 9],
        "ckd": [277, 67, 6, 50]
    },
    {
        "id": "bbc_news",
        "title": "(f) BBC News Topics (K=5)",
        "classes": ["Tech", "Business", "Sport", "Entertain", "Politics"],
        "gold": [72, 92, 92, 69, 75],
        "teacher": [54, 247, 58, 27, 14],
        "std_kd": [41, 253, 64, 42, 0],
        "batch_kd": [43, 250, 65, 42, 0],
        "ckd": [40, 209, 71, 54, 26]
    }
]

# Color palette
C_TEACHER = "#94A3B8"  # Slate Neutral
E_TEACHER = "#475569"

C_STD     = "#E15759"  # Crimson Red
E_STD     = "#B91C1C"

C_BATCH   = "#8DA0CB"  # Periwinkle
E_BATCH   = "#4F46E5"

C_CKD     = "#66C2A5"  # Mint Teal
E_CKD     = "#0D9488"

C_GOLD    = "#111827"  # Deep Charcoal / Black for Ground Truth Dotted Line

def render_plots():
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.8), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for i, data in enumerate(benchmark_data):
        ax = axes[i]
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
        ax.set_axisbelow(True)

        classes = data["classes"]
        K = len(classes)
        x = np.arange(K)
        w = 0.19
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

        # 1. Bars for methods
        b_tea = ax.bar(x + offsets[0], data["teacher"], w, color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.8, zorder=3)
        b_std = ax.bar(x + offsets[1], data["std_kd"], w, color=C_STD, edgecolor=E_STD, linewidth=0.8, zorder=3)
        b_batch = ax.bar(x + offsets[2], data["batch_kd"], w, color=C_BATCH, edgecolor=E_BATCH, linewidth=0.8, zorder=3)
        b_ckd = ax.bar(x + offsets[3], data["ckd"], w, color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3)

        # 2. Ground Truth Dotted Marker / Line
        # Plot continuous dotted line connecting the gold points, plus individual dashed horizontal markers over each cluster
        line_gold, = ax.plot(x, data["gold"], color=C_GOLD, linestyle=":", linewidth=2.4, 
                             marker="o", markersize=6.0, markerfacecolor="#F3F4F6", markeredgecolor=C_GOLD, 
                             markeredgewidth=1.8, zorder=5, label="Ground Truth (Actual Data)")

        # Collect legend items from first plot
        if i == 0:
            legend_handles = [line_gold, b_tea, b_std, b_batch, b_ckd]
            legend_labels = [
                "Ground Truth (Actual Data)",
                "Teacher Base (Gemma-3-4B)",
                "Standard KD",
                "Batch-Calibrated KD",
                "Null Calibrated KD (Ours)"
            ]

        ax.set_title(data["title"], pad=10, color="#0F172A")
        ax.set_ylabel("Predicted Sample Count (N=400)", color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", color="#1E293B")
        
        # Set dynamic headroom for y limit
        max_y = max(max(data["teacher"]), max(data["std_kd"]), max(data["batch_kd"]), max(data["ckd"]), max(data["gold"]))
        ax.set_ylim(0, max(max_y + 40, 200))

        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    # Unified Legend at the Top
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1",
               fontsize=10.5, borderpad=0.5, handlelength=1.6)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    save_path = OUT_DIR / "chart_6_datasets_ground_truth_vs_predictions.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"6-dataset distribution chart saved to: {save_path}")

    # Mirror to artifact directory
    artifact_path = ARTIFACT_DIR / "chart_6_datasets_ground_truth_vs_predictions.png"
    shutil.copy(save_path, artifact_path)
    print(f"Mirrored to artifact: {artifact_path}")

if __name__ == "__main__":
    render_plots()
