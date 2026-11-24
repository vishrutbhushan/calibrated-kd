"""
plot_3_key_class_distributions.py
---------------------------------
Generates a publication-grade 3-panel horizontal figure for:
  1. BBC News Topics (K=5)
  2. App Reviews (K=5)
  3. TweetEval Emotion (K=4)

Shows:
  - Dotted black line with markers for Ground Truth (Actual Data).
  - Bars for:
      1. Teacher Base (Gemma-3-4B Raw)
      2. Standard KD (RoBERTa-base)
      3. Batch-Calibrated KD (RoBERTa-base)
      4. Null Calibrated KD (Ours) (RoBERTa-base)
  - ColorBrewer Set2 / editorial palette.
  - Tilted labels with zero overlap.
  - No em dashes anywhere.
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
    "axes.labelsize": 11.0,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.5,
})

benchmark_data = [
    {
        "id": "bbc_news",
        "title": "(a) BBC News Topics (K=5)",
        "classes": ["Tech", "Business", "Sport", "Entertain", "Politics"],
        "gold": [72, 92, 92, 69, 75],
        "teacher": [54, 247, 58, 27, 14],
        "std_kd": [41, 253, 64, 42, 0],
        "batch_kd": [43, 250, 65, 42, 0],
        "ckd": [40, 209, 71, 54, 26]
    },
    {
        "id": "app_reviews",
        "title": "(b) App Reviews (K=5)",
        "classes": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        "gold": [68, 24, 36, 55, 217],
        "teacher": [164, 48, 37, 22, 129],
        "std_kd": [157, 33, 60, 1, 149],
        "batch_kd": [145, 41, 50, 15, 149],
        "ckd": [105, 67, 24, 38, 166]
    },
    {
        "id": "tweeteval_emotion",
        "title": "(c) TweetEval Emotion (K=4)",
        "classes": ["Anger", "Joy", "Optimism", "Sadness"],
        "gold": [167, 91, 36, 106],
        "teacher": [331, 46, 10, 13],
        "std_kd": [343, 52, 0, 5],
        "batch_kd": [335, 55, 1, 9],
        "ckd": [277, 67, 6, 50]
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

C_GOLD    = "#111827"  # Deep Charcoal for Ground Truth Line

def render_plots():
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")

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
        w = 0.185
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

        # 1. Bars for methods
        b_tea = ax.bar(x + offsets[0], data["teacher"], w, color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.8, zorder=3)
        b_std = ax.bar(x + offsets[1], data["std_kd"], w, color=C_STD, edgecolor=E_STD, linewidth=0.8, zorder=3)
        b_batch = ax.bar(x + offsets[2], data["batch_kd"], w, color=C_BATCH, edgecolor=E_BATCH, linewidth=0.8, zorder=3)
        b_ckd = ax.bar(x + offsets[3], data["ckd"], w, color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3)

        # 2. Ground Truth Dotted Line with circular nodes
        line_gold, = ax.plot(x, data["gold"], color=C_GOLD, linestyle=":", linewidth=2.4, 
                             marker="o", markersize=6.5, markerfacecolor="#F3F4F6", markeredgecolor=C_GOLD, 
                             markeredgewidth=1.8, zorder=5, label="Ground Truth (Actual Data)")

        if i == 0:
            legend_handles = [line_gold, b_tea, b_std, b_batch, b_ckd]
            legend_labels = [
                "Ground Truth (Actual Data)",
                "Teacher Base (Gemma-3-4B)",
                "Standard KD",
                "Batch-Calibrated KD",
                "Null Calibrated KD (Ours)"
            ]

        ax.set_title(data["title"], pad=12, color="#0F172A")
        ax.set_ylabel("Predicted Count (N=400)", color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", color="#1E293B")
        
        # Roomy y-axis limit
        max_y = max(max(data["teacher"]), max(data["std_kd"]), max(data["batch_kd"]), max(data["ckd"]), max(data["gold"]))
        ax.set_ylim(0, max_y + 40)

        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    # Unified Centered Legend on Top
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1",
               fontsize=10.5, borderpad=0.5, handlelength=1.6)

    plt.tight_layout(rect=[0, 0.04, 1, 0.90])

    save_path = OUT_DIR / "chart_3_key_datasets_ground_truth_vs_predictions.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"3-dataset distribution chart saved to: {save_path}")

    # Mirror to artifact directory
    artifact_path = ARTIFACT_DIR / "chart_3_key_datasets_ground_truth_vs_predictions.png"
    shutil.copy(save_path, artifact_path)
    print(f"Mirrored to artifact: {artifact_path}")

if __name__ == "__main__":
    render_plots()
