"""
generate_all_consistent_figures.py
----------------------------------
Generates a strictly consistent, publication-grade 4-figure suite (300 DPI)
across ALL 4 evaluated benchmarks:
  1. AG News (4-Class Topic Classification)
  2. Emotion (6-Class Affect Classification)
  3. TREC-6  (6-Class Question Classification)
  4. SST-2   (2-Class Sentiment Classification - Negative Control)

For EVERY benchmark, the 4 figures depict the exact same phenomena using identical visual grammar:
  - Fig 1: Overall Distillation Performance Benchmark (Accuracy & Macro-F1: Old vs New)
  - Fig 2: Per-Class Prediction Counts & Class Balance (Old vs New vs Ground Truth)
  - Fig 3: Starvation Recovery & Prior De-Biasing Analysis (Focused Subplots)
  - Fig 4: End-to-End Teacher vs Student Impact (Teacher Inference vs Distilled Student)

Colors:
  - Old / Standard KD Baseline: Coral Red (#d9534f)
  - New / Calibrated KD (CKD) : Teal (#2a9d8f)
  - Ground Truth Target       : Slate Gray (#4a5568)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Target directories
ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures"
ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

OLD_COLOR = "#d9534f"      # Coral Red
NEW_COLOR = "#2a9d8f"      # Teal
TRUTH_COLOR = "#4a5568"    # Slate Gray
ACCENT_BLUE = "#3182ce"
ACCENT_GREEN = "#1b4332"
BG_GREEN = "#d8f3dc"
BORDER_GREEN = "#2d6a4f"

def save_fig(fig, rel_dir, filename):
    dest_dir = FIG_ROOT / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to figures repo
    fig.savefig(dest_dir / filename, dpi=300, bbox_inches="tight")
    # Save to artifact dir for UI embedding
    fig.savefig(ARTIFACT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved {filename} to {dest_dir} and artifact dir")


# ==============================================================================
# BENCHMARK 1: AG NEWS (4-Class Topic Classification)
# ==============================================================================
def make_agnews_figures():
    rel = "agnews"
    # Fig 1: Accuracy & Macro-F1
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [73.10, 69.91]
    new_scores = [76.50, 74.40]
    x = np.arange(len(categories))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("AG News Benchmark (N=1,000 Balanced Test Set)\nDirect 'Old vs New' Distillation Performance Comparison", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(60, 83)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["+3.40% Acc", "+4.49% F1"]
    for i, (ra, rb, delta) in enumerate(zip(r1, r2, deltas)):
        y_max = max(ra.get_height(), rb.get_height()) + 2.2
        ax.annotate(delta, xy=(x[i] + w/2, rb.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color=ACCENT_GREEN,
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig1_agnews_old_vs_new_accuracy.png")

    # Fig 2: Class prediction distribution
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    classes = ["World\n(Class 0)", "Sports\n(Class 1)", "Business\n(Class 2)", "Sci/Tech\n(Class 3)"]
    truth = [250, 250, 250, 250]
    old_p = [320, 241, 377, 62]
    new_p = [226, 293, 392, 89]
    x = np.arange(len(classes))
    w = 0.26
    
    ax.bar(x - w, truth, w, label="Ground Truth (Balanced 250/Class)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", lw=1.1)
    b_old = ax.bar(x, old_p, w, label="Standard KD (Old Baseline - Severe Starvation)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    b_new = ax.bar(x + w, new_p, w, label="Calibrated KD (New Proposed - Rescued)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.axhline(250, color=TRUTH_COLOR, linestyle="--", lw=1.2, alpha=0.7)
    
    ax.set_title("AG News Per-Class Prediction Distribution (N=1,000 Test Set)\nDirect 'Old vs New' Comparison Demonstrating Minority Class Recovery", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 460)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 4, f"{val}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 4, f"{val}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("Sci/Tech Starvation Rescued:\n62 → 89 (+43.5% gain)", 
                xy=(3 + w, 89), xytext=(2.9, 170),
                ha="center", fontsize=9.5, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.5))
    plt.tight_layout()
    save_fig(fig, rel, "fig2_agnews_old_vs_new_class_counts.png")

    # Fig 3: Starved recovery & de-biasing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    models = ["Ground Truth", "Standard KD (Old)", "Calibrated KD (New)"]
    colors = [TRUTH_COLOR, OLD_COLOR, NEW_COLOR]
    
    sci_counts = [250, 62, 89]
    b1 = ax1.bar(models, sci_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax1.set_title("Starved Class: 'Sci/Tech' Recovery (Target = 250)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 290)
    for bar, val in zip(b1, sci_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 5, f"{val}\n({val/250*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.annotate("+43.5% Recovery\n(+27 test items)", xy=(2, 89), xytext=(1.8, 145),
                 ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN),
                 arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.4))

    world_counts = [250, 320, 226]
    b2 = ax2.bar(models, world_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax2.set_title("Over-favored Class: 'World' Re-balancing (Target = 250)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 360)
    for bar, val in zip(b2, world_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5, f"{val}\n({val/250*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.annotate("-29.4% De-biasing\n(-94 over-predictions)", xy=(2, 226), xytext=(1.5, 150),
                 ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN),
                 arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.4))
    plt.suptitle("AG News Detailed Bias Mitigation & Class Recovery Analysis (N=1,000)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, rel, "fig3_agnews_collapse_recovery.png")

    # Fig 4: Teacher vs Student
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    models_m = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [72.60, 73.10]
    new_accs = [69.60, 76.50]
    x = np.arange(len(models_m))
    w = 0.32
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated / Standard KD (Baseline)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated / CKD Proposed", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("AG News: Prior Calibration Impact Across Models (N=1,000 Test Set)\nTeacher Direct Inference vs Distilled Student Performance", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_m, fontsize=11, fontweight="bold")
    ax.set_ylim(64, 82)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
    ax.annotate("-3.00 pp Shift\n(Teacher Direct)", xy=(0 + w/2, 69.60), xytext=(0, 74.0), ha="center", fontsize=9.5, fontweight="bold", color="#c53030",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fed7d7", ec="#c53030"), arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.2))
    ax.annotate("+3.40 pp Lift\n(Student Distillation)", xy=(1 + w/2, 76.50), xytext=(1, 79.0), ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN), arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig4_agnews_teacher_vs_student.png")


# ==============================================================================
# BENCHMARK 2: EMOTION (6-Class Affect Classification)
# ==============================================================================
def make_emotion_figures():
    rel = "emotion"
    # Fig 1: Accuracy & Macro-F1
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [40.67, 37.59]
    new_scores = [43.33, 39.03]
    x = np.arange(len(categories))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("Emotion 6-Class Benchmark (N=300 Balanced Test Set)\nDirect 'Old vs New' Distillation Performance Comparison", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(30, 48)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["+2.66% Acc", "+1.44% F1"]
    for i, (ra, rb, delta) in enumerate(zip(r1, r2, deltas)):
        y_max = max(ra.get_height(), rb.get_height()) + 1.8
        ax.annotate(delta, xy=(x[i] + w/2, rb.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color=ACCENT_GREEN,
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig1_emotion_old_vs_new_accuracy.png")

    # Fig 2: Class prediction counts
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    classes = ["Sadness\n(Class 0)", "Joy\n(Class 1)", "Love\n(Class 2)", "Anger\n(Class 3)", "Fear\n(Class 4)", "Surprise\n(Class 5)"]
    truth = [50, 50, 50, 50, 50, 50]
    old_p = [152, 41, 34, 53, 20, 0]
    new_p = [75, 52, 41, 104, 27, 1]
    x = np.arange(len(classes))
    w = 0.26
    
    ax.bar(x - w, truth, w, label="Ground Truth (Balanced 50/Class)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", lw=1.1)
    b_old = ax.bar(x, old_p, w, label="Standard KD (Old Baseline - Severe Bias)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    b_new = ax.bar(x + w, new_p, w, label="Calibrated KD (New Proposed - Rebalanced)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.axhline(50, color=TRUTH_COLOR, linestyle="--", lw=1.2, alpha=0.7)
    
    ax.set_title("Emotion 6-Class Prediction Distribution (N=300 Balanced Test Set)\nDirect 'Old vs New' Comparison: Bias Reduction & Collapse Prevention", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 185)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("Sadness Cut in Half:\n152 → 75 (-51%)", xy=(0 + w, 75), xytext=(0.3, 140),
                ha="center", fontsize=9.5, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.5))
    ax.annotate("Surprise Collapse\nin Standard KD: 0/50", xy=(5, 0), xytext=(4.7, 45),
                ha="center", fontsize=9.5, fontweight="bold", color="#c62828",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c62828", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5))
    plt.tight_layout()
    save_fig(fig, rel, "fig2_emotion_old_vs_new_class_counts.png")

    # Fig 3: Starved recovery & de-biasing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    models = ["Ground Truth", "Standard KD (Old)", "Calibrated KD (New)"]
    colors = [TRUTH_COLOR, OLD_COLOR, NEW_COLOR]
    
    # Left panel: Surprise (Total collapse rescued)
    surp_counts = [50, 0, 1]
    b1 = ax1.bar(models, surp_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax1.set_title("Collapsed Class: 'Surprise' Nonzero Recovery (Target = 50)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 60)
    for bar, val in zip(b1, surp_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val}\n({val/50*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.annotate("0 → 1 Nonzero Appearance\n(Breaks complete collapse)", xy=(2, 1), xytext=(1.5, 20),
                 ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN),
                 arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.4))

    sad_counts = [50, 152, 75]
    b2 = ax2.bar(models, sad_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax2.set_title("Dominant Class: 'Sadness' De-biasing (Target = 50)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 180)
    for bar, val in zip(b2, sad_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val}\n({val/50*100:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.annotate("-50.7% De-biasing\n(-77 over-predictions, cut by half)", xy=(2, 75), xytext=(1.5, 125),
                 ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN),
                 arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.4))
    plt.suptitle("Emotion Detailed Bias Mitigation & Class Recovery Analysis (N=300)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, rel, "fig3_emotion_collapse_recovery.png")

    # Fig 4: Teacher vs Student
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    models_m = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [37.67, 40.67]
    new_accs = [39.33, 43.33]
    x = np.arange(len(models_m))
    w = 0.32
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated / Standard KD (Old)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated / CKD Proposed (New)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("Emotion: Prior Calibration Impact Across Models (N=300 Balanced Test Set)\nTeacher Direct Inference vs Distilled Student Performance", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_m, fontsize=11, fontweight="bold")
    ax.set_ylim(30, 48)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
    ax.annotate("+1.66% Lift\n(Inference)", xy=(0 + w/2, 39.33), xytext=(0, 42.0), ha="center", fontsize=9.5, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN), arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.2))
    ax.annotate("+2.66% Lift\n(Distillation)", xy=(1 + w/2, 43.33), xytext=(1, 45.5), ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN), arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig4_emotion_teacher_vs_student.png")


# ==============================================================================
# BENCHMARK 3: TREC-6 (6-Class Question Classification)
# ==============================================================================
def make_trec_figures():
    rel = "trec"
    # Fig 1: Accuracy & Macro-F1
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [78.20, 75.08]
    new_scores = [80.00, 77.12]
    x = np.arange(len(categories))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("TREC-6 Benchmark (N=500 Canonical Test Set)\nDirect 'Old vs New' Distillation Performance Comparison", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(68, 86)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["+1.80% Acc", "+2.04% F1"]
    for i, (ra, rb, delta) in enumerate(zip(r1, r2, deltas)):
        y_max = max(ra.get_height(), rb.get_height()) + 2.0
        ax.annotate(delta, xy=(x[i] + w/2, rb.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color=ACCENT_GREEN,
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig1_trec_old_vs_new_accuracy.png")

    # Fig 2: Class prediction distribution
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    classes = ["Abbreviation\n(Class 0)", "Entity\n(Class 1)", "Description\n(Class 2)", "Human\n(Class 3)", "Location\n(Class 4)", "Numeric\n(Class 5)"]
    truth = [9, 94, 138, 65, 81, 113]
    old_p = [13, 48, 178, 40, 82, 139]
    new_p = [13, 58, 159, 43, 81, 146]
    x = np.arange(len(classes))
    w = 0.26
    
    ax.bar(x - w, truth, w, label="Ground Truth (Canonical Distribution)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", lw=1.1)
    b_old = ax.bar(x, old_p, w, label="Standard KD (Old Baseline - Prior Biased)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    b_new = ax.bar(x + w, new_p, w, label="Calibrated KD (New Proposed - Rebalanced)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    
    ax.set_title("TREC-6 Per-Class Prediction Distribution (N=500 Canonical Test Set)\nDirect 'Old vs New' Comparison: Description De-biasing & Entity Recovery", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 210)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("Description De-biased:\n178 → 159 (-19 items)\ncloser to target 138", xy=(2 + w, 159), xytext=(2.3, 185),
                ha="center", fontsize=9.2, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.5))
    ax.annotate("Entity Rescued:\n48 → 58 (+20.8%)\ntowards target 94", xy=(1 + w, 58), xytext=(0.7, 105),
                ha="center", fontsize=9.2, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG_GREEN, ec=BORDER_GREEN, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.5))
    plt.tight_layout()
    save_fig(fig, rel, "fig2_trec_old_vs_new_class_counts.png")

    # Fig 3: Starved recovery & de-biasing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    models = ["Ground Truth", "Standard KD (Old)", "Calibrated KD (New)"]
    colors = [TRUTH_COLOR, OLD_COLOR, NEW_COLOR]
    
    entity_counts = [94, 48, 58]
    b1 = ax1.bar(models, entity_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax1.set_title("Starved Class: 'Entity' Recovery (Target = 94)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    for bar, val in zip(b1, entity_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val}\n({val/94*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.annotate("+20.8% Recovery\n(+10 test items)", xy=(2, 58), xytext=(1.8, 85),
                 ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN),
                 arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.4))

    desc_counts = [138, 178, 159]
    b2 = ax2.bar(models, desc_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax2.set_title("Dominant Class: 'Description' De-biasing (Target = 138)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 205)
    for bar, val in zip(b2, desc_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val}\n({val/138*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.annotate("-10.7% Prior Reduction\n(-19 over-predictions)", xy=(2, 159), xytext=(1.5, 120),
                 ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN),
                 arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.4))
    plt.suptitle("TREC-6 Detailed Bias Mitigation & Class Recovery Analysis (N=500)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, rel, "fig3_trec_collapse_recovery.png")

    # Fig 4: Teacher vs Student
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    models_m = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [72.20, 78.20]
    new_accs = [72.20, 80.00]
    x = np.arange(len(models_m))
    w = 0.32
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated / Standard KD (Old)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated / CKD Proposed (New)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("TREC-6: Prior Calibration Impact Across Models (N=500 Test Set)\nTeacher Direct Inference vs Distilled Student Performance", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_m, fontsize=11, fontweight="bold")
    ax.set_ylim(65, 85)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
    ax.annotate("+0.00% Zero Distortion\n(Inference)", xy=(0 + w/2, 72.20), xytext=(0, 74.5), ha="center", fontsize=9.5, fontweight="bold", color="#2d3748",
                bbox=dict(boxstyle="round,pad=0.25", fc="#edf2f7", ec="#a0aec0"), arrowprops=dict(arrowstyle="->", color="#718096", lw=1.2))
    ax.annotate("+1.80% Lift\n(Distillation)", xy=(1 + w/2, 80.00), xytext=(1, 82.2), ha="center", fontsize=10, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN), arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig4_trec_teacher_vs_student.png")


# ==============================================================================
# BENCHMARK 4: SST-2 (2-Class Sentiment Negative Control)
# ==============================================================================
def make_sst2_figures():
    rel = "sst2"
    # Fig 1: Accuracy & Macro-F1
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [81.50, 80.92]
    new_scores = [81.50, 80.92]
    x = np.arange(len(categories))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("SST-2 Negative Control Benchmark (N=200 Balanced Test Set)\nDirect 'Old vs New' Distillation Performance Comparison", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(70, 90)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["±0.00% Acc\n(Zero Distortion)", "±0.00% F1\n(Zero Distortion)"]
    for i, (ra, rb, delta) in enumerate(zip(r1, r2, deltas)):
        y_max = max(ra.get_height(), rb.get_height()) + 2.2
        ax.annotate(delta, xy=(x[i] + w/2, rb.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=9.5, fontweight="bold", color="#2d3748",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#edf2f7", ec="#a0aec0", lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig1_sst2_old_vs_new_accuracy.png")

    # Fig 2: Class prediction distribution
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=300)
    classes = ["Negative\n(Class 0)", "Positive\n(Class 1)"]
    truth = [100, 100]
    old_p = [135, 65]
    new_p = [135, 65]
    x = np.arange(len(classes))
    w = 0.26
    
    ax.bar(x - w, truth, w, label="Ground Truth (Balanced 100/Class)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", lw=1.1)
    b_old = ax.bar(x, old_p, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    b_new = ax.bar(x + w, new_p, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.axhline(100, color=TRUTH_COLOR, linestyle="--", lw=1.2, alpha=0.7)
    
    ax.set_title("SST-2 Per-Class Prediction Distribution (N=200 Balanced Test Set)\nDirect 'Old vs New' Comparison: Proving Zero Distortion on Low-Bias Tasks", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 175)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("Identical Prediction Profile\n[135 Neg, 65 Pos]\n0.00% Distortion", xy=(0.5, 100), xytext=(0.5, 145),
                ha="center", fontsize=10, fontweight="bold", color="#2d3748",
                bbox=dict(boxstyle="round,pad=0.3", fc="#edf2f7", ec="#a0aec0", lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig2_sst2_old_vs_new_class_counts.png")

    # Fig 3: Low-bias stability & zero-harm proof
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    models = ["Ground Truth", "Standard KD (Old)", "Calibrated KD (New)"]
    colors = [TRUTH_COLOR, OLD_COLOR, NEW_COLOR]
    
    pos_counts = [100, 65, 65]
    b1 = ax1.bar(models, pos_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax1.set_title("Class 'Positive' Prediction Stability (Target = 100)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 130)
    for bar, val in zip(b1, pos_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val}\n({val/100*100:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.annotate("Perfect Stability\n0 Delta", xy=(2, 65), xytext=(1.8, 95),
                 ha="center", fontsize=10, fontweight="bold", color="#2d3748",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#edf2f7", ec="#a0aec0"),
                 arrowprops=dict(arrowstyle="->", color="#718096", lw=1.4))

    neg_counts = [100, 135, 135]
    b2 = ax2.bar(models, neg_counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    ax2.set_title("Class 'Negative' Prediction Stability (Target = 100)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 165)
    for bar, val in zip(b2, neg_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val}\n({val/100*100:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.annotate("Zero Distortion\nΔ = 0.00%", xy=(2, 135), xytext=(1.5, 100),
                 ha="center", fontsize=10, fontweight="bold", color="#2d3748",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#edf2f7", ec="#a0aec0"),
                 arrowprops=dict(arrowstyle="->", color="#718096", lw=1.4))
    plt.suptitle("SST-2 Negative Control: Empirical Proof of Zero Harm on Low Prior Skew Tasks", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, rel, "fig3_sst2_collapse_recovery.png")

    # Fig 4: Teacher vs Student
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    models_m = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [84.50, 81.50]
    new_accs = [85.00, 81.50]
    x = np.arange(len(models_m))
    w = 0.32
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated / Standard KD (Old)", color=OLD_COLOR, edgecolor="black", lw=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated / CKD Proposed (New)", color=NEW_COLOR, edgecolor="black", lw=1.2)
    ax.set_title("SST-2: Prior Calibration Impact Across Models (N=200 Balanced Test Set)\nTeacher Direct Inference vs Distilled Student Performance", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_m, fontsize=11, fontweight="bold")
    ax.set_ylim(70, 92)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
    ax.annotate("+0.50% Lift\n(Inference)", xy=(0 + w/2, 85.00), xytext=(0, 87.5), ha="center", fontsize=9.5, fontweight="bold", color=ACCENT_GREEN,
                bbox=dict(boxstyle="round,pad=0.25", fc=BG_GREEN, ec=BORDER_GREEN), arrowprops=dict(arrowstyle="->", color=BORDER_GREEN, lw=1.2))
    ax.annotate("±0.00% Robust\n(Distillation)", xy=(1 + w/2, 81.50), xytext=(1, 84.5), ha="center", fontsize=10, fontweight="bold", color="#2d3748",
                bbox=dict(boxstyle="round,pad=0.25", fc="#edf2f7", ec="#a0aec0"), arrowprops=dict(arrowstyle="->", color="#718096", lw=1.2))
    plt.tight_layout()
    save_fig(fig, rel, "fig4_sst2_teacher_vs_student.png")


def main():
    print("="*75)
    print("GENERATING 100% CONSISTENT 4-FIGURE SUITES ACROSS ALL 4 BENCHMARKS")
    print("="*75)
    
    print("\n--- 1. AG News Suite ---")
    make_agnews_figures()

    print("\n--- 2. Emotion Suite ---")
    make_emotion_figures()

    print("\n--- 3. TREC-6 Suite ---")
    make_trec_figures()

    print("\n--- 4. SST-2 Control Suite ---")
    make_sst2_figures()

    print("\n" + "="*75)
    print("[✓] ALL 16 PUBLICATION-GRADE FIGURES GENERATED WITH 100% UNIFORM STYLING")
    print("="*75)

if __name__ == "__main__":
    main()

