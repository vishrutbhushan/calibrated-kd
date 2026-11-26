"""
blog_benchmark_barchart.py
--------------------------
Generates the 2-panel Macro-F1 and Accuracy bar chart for the blog from
outputs/main_experiments/all_6_main_benchmarks_summary.json, with a 1-line
result caption at the bottom.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FILE = ROOT / "outputs" / "main_experiments" / "all_6_main_benchmarks_summary.json"
OUT_DIR = ROOT / "outputs" / "blog_figures"

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

DATASET_SPEC = [
    ("student_questions", "Student Questions (K=4)"),
    ("trec6", "TREC-6 (K=6)"),
    ("app_reviews", "App Reviews (K=5)"),
    ("twitter_emotion", "Twitter Emotion (K=6)"),
    ("tweeteval_emotion", "TweetEval Emotion (K=4)"),
    ("bbc_news", "BBC News (K=5)"),
]

C_TEACHER, E_TEACHER = "#94A3B8", "#475569"
C_STD, E_STD = "#E15759", "#B91C1C"
C_BATCH, E_BATCH = "#8DA0CB", "#4F46E5"
C_CKD, E_CKD = "#66C2A5", "#0D9488"


def load_metrics(summary_path: Path) -> tuple[dict, str]:
    """Loads Macro-F1 and Accuracy arrays dynamically and builds a 1-line caption."""
    with open(summary_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    metrics = {
        "labels": [label for _, label in DATASET_SPEC],
        "f1": {"teacher": [], "std": [], "batch": [], "ckd": []},
        "acc": {"teacher": [], "std": [], "batch": [], "ckd": []},
    }
    for key, _ in DATASET_SPEC:
        d = data[key]
        for target, method in [("teacher", "teacher_raw"), ("std", "standard_kd"), ("batch", "batch_calibrated_kd"), ("ckd", "calibrated_kd")]:
            metrics["f1"][target].append(d[method]["macro_f1"])
            metrics["acc"][target].append(d[method]["acc"])

    avg_f1_std = float(np.mean(np.array(metrics["f1"]["ckd"]) - np.array(metrics["f1"]["std"])))
    avg_acc_std = float(np.mean(np.array(metrics["acc"]["ckd"]) - np.array(metrics["acc"]["std"])))
    avg_f1_batch = float(np.mean(np.array(metrics["f1"]["ckd"]) - np.array(metrics["f1"]["batch"])))
    caption = (
        f"Average gains over Standard KD: +{avg_f1_std:.2f} pp Macro-F1 and +{avg_acc_std:.2f} pp Accuracy "
        f"across 6 benchmarks (+{avg_f1_batch:.2f} pp average Macro-F1 over Batch-Calibrated KD)."
    )
    return metrics, caption


def render_plot(metrics: dict, caption: str, save_path: Path) -> None:
    """Renders the 2-panel Macro-F1 and Accuracy bar chart with a 1-line caption."""
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.4), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")

    labels = metrics["labels"]
    x = np.arange(len(labels))
    bar_width = 0.185
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    panels = [
        (axes[0], metrics["f1"], "(a) Macro-F1 (%) : Primary Metric", "Macro-F1 (%)"),
        (axes[1], metrics["acc"], "(b) Overall Accuracy (%) : Secondary Metric", "Accuracy (%)"),
    ]

    legend_handles = []
    for idx, (ax, series, title, ylabel) in enumerate(panels):
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
        ax.set_axisbelow(True)

        b_tea = ax.bar(x + offsets[0], series["teacher"], bar_width, color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.9, zorder=3)
        b_std = ax.bar(x + offsets[1], series["std"], bar_width, color=C_STD, edgecolor=E_STD, linewidth=0.9, zorder=3)
        b_batch = ax.bar(x + offsets[2], series["batch"], bar_width, color=C_BATCH, edgecolor=E_BATCH, linewidth=0.9, zorder=3)
        b_ckd = ax.bar(x + offsets[3], series["ckd"], bar_width, color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3)

        if idx == 0:
            legend_handles = [b_tea, b_std, b_batch, b_ckd]

        ax.set_title(title, pad=12, color="#0F172A")
        ax.set_ylabel(ylabel, color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", color="#1E293B")
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    fig.legend(
        legend_handles,
        ["Teacher Base (Gemma-3-4B)", "Standard KD", "Batch-Calibrated KD", "Null Calibrated KD (Ours)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=True,
        facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=10.0, borderpad=0.5, handlelength=1.4,
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=10.2, fontstyle="italic", color="#334155")

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"[✓] Blog bar chart saved to: {save_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, caption = load_metrics(SUMMARY_FILE)
    render_plot(metrics, caption, OUT_DIR / "blog_chart_main_benchmark_f1_accuracy.png")


if __name__ == "__main__":
    main()
