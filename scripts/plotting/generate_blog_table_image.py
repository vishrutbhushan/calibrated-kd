"""
generate_blog_table_image.py
-----------------------------
Generates a publication-grade, blog-styled graphic table of the main benchmark results.
Specifications:
  - Font: Clean serif matching Substack editorial aesthetic (DejaVu Serif).
  - Columns:
      [Architecture on left] | Method | Student Questions | TREC-6 | App Reviews | Twitter Emotion | TweetEval Emotion | BBC News
  - Group on left: "Teacher (Gemma-3-4B)" spanning 2 rows (Base and Null-Calibrated), "Student (RoBERTa-base)" spanning 3 rows (Standard KD, Batch-Calibrated KD, Null Calibrated KD).
  - Removed Post-Hoc Output Cal row completely.
  - Method names: NON-BOLD ("Base (Raw Zero-Shot)", "Null-Calibrated", "Standard KD", "Batch-Calibrated KD", "Null Calibrated KD (Ours)").
  - Cells: "MacroF1% (Acc%)", e.g., "37.23% (39.0%)".
  - Null Calibrated KD row highlighted with soft green accent.
  - No em dashes anywhere.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "results" / "main_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")

# Headers
headers = [
    "Student Questions\n(K=4)", 
    "TREC-6\n(K=6)", 
    "App Reviews\n(K=5)", 
    "Twitter Emotion\n(K=6)", 
    "TweetEval Emotion\n(K=4)", 
    "BBC News\n(K=5)"
]

# Rows data: (Group, Method_name, values, is_our_method)
rows_data = [
    # Teacher rows
    ("Teacher\n(Gemma-3-4B)", "Base (Raw Zero-Shot)", 
     ["28.13% (33.8%)", "50.15% (52.3%)", "29.69% (48.3%)", "39.80% (53.0%)", "45.78% (57.3%)", "54.81% (56.0%)"], 
     False),
    ("Teacher\n(Gemma-3-4B)", "Null-Calibrated", 
     ["29.08% (33.8%)", "52.90% (54.3%)", "32.46% (49.3%)", "40.35% (52.0%)", "56.05% (65.0%)", "64.04% (63.3%)"], 
     False),
    
    # Student rows (Post-Hoc REMOVED)
    ("Student\n(RoBERTa-base)", "Standard KD", 
     ["23.09% (32.5%)", "62.28% (59.8%)", "30.25% (51.8%)", "29.78% (50.5%)", "34.16% (54.3%)", "56.26% (59.3%)"], 
     False),
    ("Student\n(RoBERTa-base)", "Batch-Calibrated KD", 
     ["23.58% (32.3%)", "63.29% (61.0%)", "32.04% (52.3%)", "31.13% (50.0%)", "38.34% (56.3%)", "56.95% (60.0%)"], 
     False),
    ("Student\n(RoBERTa-base)", "Null Calibrated KD (Ours)", 
     ["37.23% (39.0%)", "64.71% (61.8%)", "34.50% (54.0%)", "34.46% (50.8%)", "55.48% (66.5%)", "71.47% (70.5%)"], 
     True),
]

def render_table_image(save_path):
    fig, ax = plt.subplots(figsize=(15.8, 5.2), dpi=350)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")

    font_family = "DejaVu Serif"
    plt.rcParams["font.family"] = font_family

    x_group = 0.02
    w_group = 0.12
    x_method = 0.15
    w_method = 0.19
    col_starts = [0.35, 0.46, 0.57, 0.68, 0.79, 0.90]
    col_width = 0.10

    y_top = 0.88
    row_height = 0.10
    
    # Clean header text without em dashes
    ax.text(0.02, 0.965, "Benchmark Evaluation: Macro-F1 (Primary) and Accuracy (Secondary)", 
            fontsize=15.0, fontweight="bold", color="#111827", va="top")
    ax.text(0.02, 0.915, "Values reported as Macro-F1% with overall Accuracy% in parentheses. Null Calibrated KD operates at zero inference cost.", 
            fontsize=10.5, fontstyle="italic", color="#4B5563", va="top")

    y_header = y_top - 0.04

    # Top Rule
    ax.plot([0.02, 0.99], [y_header, y_header], color="#111827", lw=1.8)

    # Column Headers
    ax.text(x_group + w_group/2, y_header - 0.045, "Architecture", fontsize=11, fontweight="bold", color="#111827", ha="center", va="center")
    ax.text(x_method + 0.01, y_header - 0.045, "Method", fontsize=11, fontweight="bold", color="#111827", ha="left", va="center")
    
    for i, h in enumerate(headers):
        ax.text(col_starts[i] + col_width/2, y_header - 0.045, h, fontsize=10.5, fontweight="bold", color="#111827", ha="center", va="center", multialignment="center")

    # Midrule under headers
    y_mid = y_header - 0.09
    ax.plot([0.02, 0.99], [y_mid, y_mid], color="#111827", lw=1.2)

    # 1. Teacher Group Box on left (2 rows)
    y_teacher_top = y_mid
    y_teacher_bottom = y_mid - 2 * row_height
    rect_teacher = patches.FancyBboxPatch((x_group, y_teacher_bottom + 0.008), w_group, (y_teacher_top - y_teacher_bottom - 0.016),
                                         boxstyle="round,pad=0.006,rounding_size=0.012",
                                         facecolor="#F3F4F6", edgecolor="#E5E7EB", lw=1)
    ax.add_patch(rect_teacher)
    ax.text(x_group + w_group/2, (y_teacher_top + y_teacher_bottom)/2, "Teacher\n(Gemma-3-4B)", 
            fontsize=10.5, fontweight="bold", color="#374151", ha="center", va="center", multialignment="center")

    # Teacher rows
    for r_idx in range(2):
        y_row = y_mid - (r_idx + 0.5) * row_height
        _, method_name, vals, _ = rows_data[r_idx]
        
        # Method name (Non-bold)
        ax.text(x_method + 0.01, y_row, method_name, fontsize=10.5, color="#1F2937", va="center")

        # Values
        for c_idx, val in enumerate(vals):
            ax.text(col_starts[c_idx] + col_width/2, y_row, val, fontsize=10, color="#4B5563", ha="center", va="center")

    # Separator line between Teacher and Student
    y_sep = y_teacher_bottom
    ax.plot([0.02, 0.99], [y_sep, y_sep], color="#D1D5DB", lw=0.9, ls="--")

    # 2. Student Group Box on left (3 rows)
    y_student_top = y_sep
    y_student_bottom = y_sep - 3 * row_height
    rect_student = patches.FancyBboxPatch((x_group, y_student_bottom + 0.008), w_group, (y_student_top - y_student_bottom - 0.016),
                                         boxstyle="round,pad=0.006,rounding_size=0.012",
                                         facecolor="#EEF2FF", edgecolor="#E0E7FF", lw=1)
    ax.add_patch(rect_student)
    ax.text(x_group + w_group/2, (y_student_top + y_student_bottom)/2, "Student\n(RoBERTa-base)", 
            fontsize=10.5, fontweight="bold", color="#3730A3", ha="center", va="center", multialignment="center")

    # Student rows (Standard KD, Batch-Calibrated KD, Null Calibrated KD)
    for r_idx in range(2, 5):
        curr_row_idx = r_idx - 2
        y_row = y_sep - (curr_row_idx + 0.5) * row_height
        _, method_name, vals, is_our_method = rows_data[r_idx]

        # Highlight our method row
        if is_our_method:
            rect_row = patches.Rectangle((x_method - 0.005, y_row - row_height/2 + 0.006), 0.99 - (x_method - 0.005), row_height - 0.012,
                                         facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=1.2, zorder=0)
            ax.add_patch(rect_row)

        # Method name (Non-bold as requested)
        label_color = "#047857" if is_our_method else "#1F2937"
        ax.text(x_method + 0.01, y_row, method_name, fontsize=10.5, color=label_color, va="center", zorder=2)

        # Values
        for c_idx, val in enumerate(vals):
            if is_our_method:
                pill = patches.FancyBboxPatch((col_starts[c_idx] + 0.002, y_row - 0.026), col_width - 0.004, 0.052,
                                              boxstyle="round,pad=0.004,rounding_size=0.012",
                                              facecolor="#DCFCE7", edgecolor="#86EFAC", lw=1, zorder=1)
                ax.add_patch(pill)
                ax.text(col_starts[c_idx] + col_width/2, y_row, val, fontsize=10.2, fontweight="bold", color="#14532D", ha="center", va="center", zorder=2)
            else:
                ax.text(col_starts[c_idx] + col_width/2, y_row, val, fontsize=10, color="#374151", ha="center", va="center", zorder=2)

    # Bottom Rule
    y_bottom = y_student_bottom
    ax.plot([0.02, 0.99], [y_bottom, y_bottom], color="#111827", lw=1.8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"Table image saved to: {save_path}")

if __name__ == "__main__":
    out1 = OUT_DIR / "table_main_benchmark_results.png"
    out2 = ARTIFACT_DIR / "table_main_benchmark_results.png"
    render_table_image(out1)
    if ARTIFACT_DIR.exists():
        shutil.copy(out1, out2)
        print(f"Mirrored table to artifact: {out2}")
