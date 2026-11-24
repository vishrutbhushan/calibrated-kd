"""
plot_3_key_class_distributions.py
---------------------------------
Generates a publication-grade 3-panel horizontal class distribution figure
dynamically from outputs/main_experiments/all_6_main_benchmarks_summary.json
and test_400.csv ground-truth splits for:
  (a) BBC News Topics (K=5)
  (b) App Reviews (K=5)
  (c) TweetEval Emotion (K=4)
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "data" / "datasets" / "natural_benchmarks"
SUMMARY_FILE = ROOT / "outputs" / "main_experiments" / "all_6_main_benchmarks_summary.json"
OUT_DIR = ROOT / "outputs" / "main_experiments" / "figures"
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

PANEL_SPECS = [
    {
        "id": "bbc_news",
        "title": "(a) BBC News Topics (K=5)",
        "classes": ["Tech", "Business", "Sport", "Entertain", "Politics"],
    },
    {
        "id": "app_reviews",
        "title": "(b) App Reviews (K=5)",
        "classes": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
    },
    {
        "id": "tweeteval_emotion",
        "title": "(c) TweetEval Emotion (K=4)",
        "classes": ["Anger", "Joy", "Optimism", "Sadness"],
    },
]

C_TEACHER, E_TEACHER = "#94A3B8", "#475569"
C_STD, E_STD = "#E15759", "#B91C1C"
C_BATCH, E_BATCH = "#8DA0CB", "#4F46E5"
C_CKD, E_CKD = "#66C2A5", "#0D9488"
C_GOLD = "#111827"


def load_gold_counts(task_name: str, num_classes: int) -> list[int]:
    """Reads test_400.csv and computes exact ground-truth class counts."""
    test_file = DATA_BASE / task_name / "test_400.csv"
    golds = []
    with open(test_file, "r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            golds.append(int(row["label"]))
    return np.bincount(golds, minlength=num_classes).tolist()


def load_distribution_data(summary_path: Path) -> list[dict]:
    """Loads ground-truth and predicted class counts dynamically."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary results not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    panels = []
    for spec in PANEL_SPECS:
        ds_id = spec["id"]
        d = summary[ds_id]
        k = len(spec["classes"])
        panels.append({
            "id": ds_id,
            "title": spec["title"],
            "classes": spec["classes"],
            "gold": load_gold_counts(ds_id, k),
            "teacher": d["teacher_raw"]["counts"],
            "std_kd": d["standard_kd"]["counts"],
            "batch_kd": d["batch_calibrated_kd"]["counts"],
            "ckd": d["calibrated_kd"]["counts"],
        })
    return panels


def render_plots(panels: list[dict], save_path: Path) -> None:
    """Renders the 3-panel Ground Truth vs. Predicted Counts figure."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")

    legend_handles = []
    legend_labels = []

    for i, data in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
        ax.set_axisbelow(True)

        classes = data["classes"]
        x = np.arange(len(classes))
        w = 0.185
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

        b_tea = ax.bar(x + offsets[0], data["teacher"], w, color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.8, zorder=3)
        b_std = ax.bar(x + offsets[1], data["std_kd"], w, color=C_STD, edgecolor=E_STD, linewidth=0.8, zorder=3)
        b_batch = ax.bar(x + offsets[2], data["batch_kd"], w, color=C_BATCH, edgecolor=E_BATCH, linewidth=0.8, zorder=3)
        b_ckd = ax.bar(x + offsets[3], data["ckd"], w, color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3)

        line_gold, = ax.plot(
            x,
            data["gold"],
            color=C_GOLD,
            linestyle=":",
            linewidth=2.4,
            marker="o",
            markersize=6.5,
            markerfacecolor="#F3F4F6",
            markeredgecolor=C_GOLD,
            markeredgewidth=1.8,
            zorder=5,
        )

        if i == 0:
            legend_handles = [line_gold, b_tea, b_std, b_batch, b_ckd]
            legend_labels = [
                "Ground Truth (Actual Data)",
                "Teacher Base (Gemma-3-4B)",
                "Standard KD",
                "Batch-Calibrated KD",
                "Null Calibrated KD (Ours)",
            ]

        ax.set_title(data["title"], pad=12, color="#0F172A")
        ax.set_ylabel("Predicted Count (N=400)", color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", color="#1E293B")

        max_y = max(
            max(data["teacher"]),
            max(data["std_kd"]),
            max(data["batch_kd"]),
            max(data["ckd"]),
            max(data["gold"]),
        )
        ax.set_ylim(0, max_y + 40)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=10.5,
        borderpad=0.5,
        handlelength=1.6,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.90])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"[✓] 3-dataset class distribution chart saved to: {save_path}")


def main() -> None:
    """Entry point for generating the dynamic 3-dataset class distribution figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = load_distribution_data(SUMMARY_FILE)
    out_path = OUT_DIR / "chart_3_key_datasets_ground_truth_vs_predictions.png"
    render_plots(panels, out_path)
    if ARTIFACT_DIR.exists():
        shutil.copy(out_path, ARTIFACT_DIR / "chart_3_key_datasets_ground_truth_vs_predictions.png")


if __name__ == "__main__":
    main()
