"""
generate_blog_table_image.py
----------------------------
Generates a publication-grade, editorial-styled graphic table of the main
benchmark results directly from outputs/main_experiments/all_6_main_benchmarks_summary.json.

Rows:
  - Teacher (Gemma-3-4B): Base (Raw Zero-Shot), Null-Calibrated
  - Student (RoBERTa-base): Standard KD, Batch-Calibrated KD, Null Calibrated KD (Ours)
"""

import json
import shutil
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FILE = ROOT / "outputs" / "main_experiments" / "all_6_main_benchmarks_summary.json"
OUT_DIR = ROOT / "outputs" / "main_experiments" / "figures"
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

DATASET_KEYS = [
    "student_questions",
    "trec6",
    "app_reviews",
    "twitter_emotion",
    "tweeteval_emotion",
    "bbc_news",
]

HEADERS = [
    "Student Questions\n(K=4)",
    "TREC-6\n(K=6)",
    "App Reviews\n(K=5)",
    "Twitter Emotion\n(K=6)",
    "TweetEval Emotion\n(K=4)",
    "BBC News\n(K=5)",
]


def load_table_rows(summary_path: Path) -> list:
    """Loads latest Macro-F1 (%) and Accuracy (%) from the summary JSON."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary results not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    def fmt(method_key: str) -> list[str]:
        cells = []
        for ds in DATASET_KEYS:
            m = data[ds][method_key]
            cells.append(f"{m['macro_f1']:.2f}% ({m['acc']:.1f}%)")
        return cells

    return [
        ("Teacher\n(Gemma-3-4B)", "Base (Raw Zero-Shot)", fmt("teacher_raw"), False),
        ("Teacher\n(Gemma-3-4B)", "Null-Calibrated", fmt("teacher_calibrated"), False),
        ("Student\n(RoBERTa-base)", "Standard KD", fmt("standard_kd"), False),
        ("Student\n(RoBERTa-base)", "Batch-Calibrated KD", fmt("batch_calibrated_kd"), False),
        ("Student\n(RoBERTa-base)", "Null Calibrated KD (Ours)", fmt("calibrated_kd"), True),
    ]


def render_table_image(rows_data: list, save_path: Path) -> None:
    """Renders the formatted table graphic and saves it to disk."""
    fig, ax = plt.subplots(figsize=(15.8, 5.2), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")

    plt.rcParams["font.family"] = "DejaVu Serif"

    x_group, w_group = 0.02, 0.12
    x_method = 0.15
    col_starts = [0.35, 0.46, 0.57, 0.68, 0.79, 0.90]
    col_width = 0.10

    y_top = 0.88
    row_height = 0.10

    ax.text(
        0.02,
        0.965,
        "Benchmark Evaluation: Macro-F1 (Primary) and Accuracy (Secondary)",
        fontsize=15.0,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    ax.text(
        0.02,
        0.915,
        "Values reported as Macro-F1% with overall Accuracy% in parentheses. "
        "Null Calibrated KD operates at zero inference cost.",
        fontsize=10.5,
        fontstyle="italic",
        color="#4B5563",
        va="top",
    )

    y_header = y_top - 0.04
    ax.plot([0.02, 0.99], [y_header, y_header], color="#111827", lw=1.8)

    ax.text(
        x_group + w_group / 2,
        y_header - 0.045,
        "Architecture",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        ha="center",
        va="center",
    )
    ax.text(
        x_method + 0.01,
        y_header - 0.045,
        "Method",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
    )

    for i, header_text in enumerate(HEADERS):
        ax.text(
            col_starts[i] + col_width / 2,
            y_header - 0.045,
            header_text,
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
            ha="center",
            va="center",
            multialignment="center",
        )

    y_mid = y_header - 0.09
    ax.plot([0.02, 0.99], [y_mid, y_mid], color="#111827", lw=1.2)

    # Teacher Group Box
    y_teacher_top = y_mid
    y_teacher_bottom = y_mid - 2 * row_height
    rect_teacher = patches.FancyBboxPatch(
        (x_group, y_teacher_bottom + 0.008),
        w_group,
        (y_teacher_top - y_teacher_bottom - 0.016),
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#F3F4F6",
        edgecolor="#E5E7EB",
        lw=1,
    )
    ax.add_patch(rect_teacher)
    ax.text(
        x_group + w_group / 2,
        (y_teacher_top + y_teacher_bottom) / 2,
        "Teacher\n(Gemma-3-4B)",
        fontsize=10.5,
        fontweight="bold",
        color="#374151",
        ha="center",
        va="center",
        multialignment="center",
    )

    for r_idx in range(2):
        y_row = y_mid - (r_idx + 0.5) * row_height
        _, method_name, vals, _ = rows_data[r_idx]
        ax.text(x_method + 0.01, y_row, method_name, fontsize=10.5, color="#1F2937", va="center")
        for c_idx, val in enumerate(vals):
            ax.text(
                col_starts[c_idx] + col_width / 2,
                y_row,
                val,
                fontsize=10,
                color="#4B5563",
                ha="center",
                va="center",
            )

    y_sep = y_teacher_bottom
    ax.plot([0.02, 0.99], [y_sep, y_sep], color="#D1D5DB", lw=0.9, ls="--")

    # Student Group Box
    y_student_top = y_sep
    y_student_bottom = y_sep - 3 * row_height
    rect_student = patches.FancyBboxPatch(
        (x_group, y_student_bottom + 0.008),
        w_group,
        (y_student_top - y_student_bottom - 0.016),
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#EEF2FF",
        edgecolor="#E0E7FF",
        lw=1,
    )
    ax.add_patch(rect_student)
    ax.text(
        x_group + w_group / 2,
        (y_student_top + y_student_bottom) / 2,
        "Student\n(RoBERTa-base)",
        fontsize=10.5,
        fontweight="bold",
        color="#3730A3",
        ha="center",
        va="center",
        multialignment="center",
    )

    for r_idx in range(2, 5):
        curr_row_idx = r_idx - 2
        y_row = y_sep - (curr_row_idx + 0.5) * row_height
        _, method_name, vals, is_our_method = rows_data[r_idx]

        if is_our_method:
            rect_row = patches.Rectangle(
                (x_method - 0.005, y_row - row_height / 2 + 0.006),
                0.99 - (x_method - 0.005),
                row_height - 0.012,
                facecolor="#F0FDF4",
                edgecolor="#BBF7D0",
                lw=1.2,
                zorder=0,
            )
            ax.add_patch(rect_row)

        label_color = "#047857" if is_our_method else "#1F2937"
        ax.text(
            x_method + 0.01, y_row, method_name, fontsize=10.5, color=label_color, va="center", zorder=2
        )

        for c_idx, val in enumerate(vals):
            if is_our_method:
                pill = patches.FancyBboxPatch(
                    (col_starts[c_idx] + 0.002, y_row - 0.026),
                    col_width - 0.004,
                    0.052,
                    boxstyle="round,pad=0.004,rounding_size=0.012",
                    facecolor="#DCFCE7",
                    edgecolor="#86EFAC",
                    lw=1,
                    zorder=1,
                )
                ax.add_patch(pill)
                ax.text(
                    col_starts[c_idx] + col_width / 2,
                    y_row,
                    val,
                    fontsize=10.2,
                    fontweight="bold",
                    color="#14532D",
                    ha="center",
                    va="center",
                    zorder=2,
                )
            else:
                ax.text(
                    col_starts[c_idx] + col_width / 2,
                    y_row,
                    val,
                    fontsize=10,
                    color="#374151",
                    ha="center",
                    va="center",
                    zorder=2,
                )

    ax.plot([0.02, 0.99], [y_student_bottom, y_student_bottom], color="#111827", lw=1.8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"[✓] Table image saved to: {save_path}")


def main() -> None:
    """Entry point for generating the dynamic benchmark table graphic."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_data = load_table_rows(SUMMARY_FILE)
    out_path = OUT_DIR / "table_main_benchmark_results.png"
    render_table_image(rows_data, out_path)
    if ARTIFACT_DIR.exists():
        shutil.copy(out_path, ARTIFACT_DIR / "table_main_benchmark_results.png")


if __name__ == "__main__":
    main()
