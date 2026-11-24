"""
generate_benchmark_barchart.py
------------------------------
Generates a publication-grade 2-panel bar chart (Macro-F1 and Overall Accuracy)
dynamically from outputs/main_experiments/all_6_main_benchmarks_summary.json.

Compares:
  1. Teacher Base (Gemma-3-4B Raw Zero-Shot)
  2. Standard KD (RoBERTa-base)
  3. Batch-Calibrated KD (RoBERTa-base)
  4. Null Calibrated KD (Ours) (RoBERTa-base)
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FILE = ROOT / "outputs" / "main_experiments" / "all_6_main_benchmarks_summary.json"
OUT_DIR = ROOT / "outputs" / "main_experiments" / "figures"
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

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


def load_metrics(summary_path: Path) -> dict:
    """Loads Macro-F1 and Accuracy arrays dynamically from the summary JSON."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary results not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    metrics = {
        "labels": [label for _, label in DATASET_SPEC],
        "f1": {"teacher": [], "std": [], "batch": [], "ckd": []},
        "acc": {"teacher": [], "std": [], "batch": [], "ckd": []},
    }

    for key, _ in DATASET_SPEC:
        d = data[key]
        metrics["f1"]["teacher"].append(d["teacher_raw"]["macro_f1"])
        metrics["f1"]["std"].append(d["standard_kd"]["macro_f1"])
        metrics["f1"]["batch"].append(d["batch_calibrated_kd"]["macro_f1"])
        metrics["f1"]["ckd"].append(d["calibrated_kd"]["macro_f1"])

        metrics["acc"]["teacher"].append(d["teacher_raw"]["acc"])
        metrics["acc"]["std"].append(d["standard_kd"]["acc"])
        metrics["acc"]["batch"].append(d["batch_calibrated_kd"]["acc"])
        metrics["acc"]["ckd"].append(d["calibrated_kd"]["acc"])

    return metrics


def render_plot(metrics: dict, save_path: Path) -> None:
    """Renders the 2-panel Macro-F1 and Accuracy bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), dpi=350)
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

        b_tea = ax.bar(
            x + offsets[0], series["teacher"], bar_width, color=C_TEACHER, edgecolor=E_TEACHER, linewidth=0.9, zorder=3
        )
        b_std = ax.bar(
            x + offsets[1], series["std"], bar_width, color=C_STD, edgecolor=E_STD, linewidth=0.9, zorder=3
        )
        b_batch = ax.bar(
            x + offsets[2], series["batch"], bar_width, color=C_BATCH, edgecolor=E_BATCH, linewidth=0.9, zorder=3
        )
        b_ckd = ax.bar(
            x + offsets[3], series["ckd"], bar_width, color=C_CKD, edgecolor=E_CKD, linewidth=1.2, zorder=3
        )

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

    legend_labels = [
        "Teacher Base (Gemma-3-4B)",
        "Standard KD",
        "Batch-Calibrated KD",
        "Null Calibrated KD (Ours)",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        fontsize=10.0,
        borderpad=0.5,
        handlelength=1.4,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"[✓] Benchmark bar chart saved to: {save_path}")


def main() -> None:
    """Entry point for generating the dynamic 2-panel Macro-F1 & Accuracy bar chart."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(SUMMARY_FILE)
    out_path = OUT_DIR / "chart_main_benchmark_f1_accuracy.png"
    render_plot(metrics, out_path)
    if ARTIFACT_DIR.exists():
        shutil.copy(out_path, ARTIFACT_DIR / "chart_main_benchmark_f1_accuracy.png")


if __name__ == "__main__":
    main()
