"""
generate_dedicated_publication_figures.py
-----------------------------------------
Generates multiple focused, publication-grade figures (300 DPI) for AG News:
  Figure 1: Old vs New Performance Benchmark (Accuracy & Macro-F1)
  Figure 2: Old vs New Per-Class Prediction Counts & Collapse Recovery
  Figure 3: Old vs New Per-Class F1 & Minority Class Gain
  Figure 4: Full 4-Way Method Ablation Progression
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/7c46b483-1886-465b-a169-56ae303c4794")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Palette definition
OLD_COLOR = "#d9534f"      # Coral Red for Old / Baseline
NEW_COLOR = "#2a9d8f"      # Teal for New / Calibrated
TRUTH_COLOR = "#4a5568"    # Slate Gray for Ground Truth
ACCENT_BLUE = "#3182ce"

# ==============================================================================
# FIGURE 1: Overall Performance Comparison (Old vs New on 1,000 Test Set)
# ==============================================================================
def make_figure1():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [73.10, 69.91]   # Standard KD
    new_scores = [76.50, 74.40]   # Static CKD
    
    x = np.arange(len(categories))
    w = 0.32
    
    rects1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old / Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    rects2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New / Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("AG News Distillation Benchmark (N=1,000 Balanced Test Set)\nDirect 'Before vs After' Performance Comparison", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(60, 82)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    # Value annotations on bars
    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.5, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.5, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    # Delta callout brackets
    deltas = ["+3.40% Acc", "+4.49% F1"]
    for i, (r1, r2, delta) in enumerate(zip(rects1, rects2, deltas)):
        y_max = max(r1.get_height(), r2.get_height()) + 2.8
        ax.annotate(delta, xy=(x[i] + w/2, r2.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color="#1b4332",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig1_agnews_old_vs_new_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Figure 1 to {out_path}")

# ==============================================================================
# FIGURE 2: Per-Class Prediction Counts & Collapse Recovery
# ==============================================================================
def make_figure2():
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    
    classes = ["World\n(Class 0)", "Sports\n(Class 1)", "Business\n(Class 2)", "Sci/Tech\n(Class 3)"]
    ground_truth = [250, 250, 250, 250]
    old_preds = [320, 241, 377, 62]    # Standard KD (Sci/Tech severely starved to 62)
    new_preds = [226, 293, 392, 89]    # Calibrated KD (World rebalanced, Sci/Tech rescued)
    
    x = np.arange(len(classes))
    w = 0.26
    
    b_truth = ax.bar(x - w, ground_truth, w, label="Ground Truth (Balanced 250/Class)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", linewidth=1.1)
    b_old = ax.bar(x, old_preds, w, label="Standard KD (Old Baseline - Severe Starvation)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    b_new = ax.bar(x + w, new_preds, w, label="Calibrated KD (New Proposed - Rescued)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.axhline(250, color=TRUTH_COLOR, linestyle="--", linewidth=1.2, alpha=0.7)
    
    ax.set_title("AG News Per-Class Prediction Distribution (N=1,000 Test Set)\nDirect 'Old vs New' Comparison Demonstrating Minority Class Recovery", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 450)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 4, f"{val}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 4, f"{val}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NEW_COLOR)
        
    # Highlighting the Sci/Tech recovery
    ax.annotate("Sci/Tech Rescued:\n62 → 89 (+43.5%)", 
                xy=(3 + w, 89), xytext=(3, 160),
                ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))
    
    # Highlighting World de-skewing
    ax.annotate("World Rebalanced:\n320 → 226", 
                xy=(0 + w, 226), xytext=(0, 360),
                ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig2_agnews_old_vs_new_class_counts.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Figure 2 to {out_path}")

# ==============================================================================
# FIGURE 3: Full 4-Way Methodological Ablation Progression
# ==============================================================================
def make_figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    
    variants = ["Standard KD\n(Baseline)", "Ada-CKD\n(Adaptive)", "PI-KD\n(Loss Reg)", "Static CKD\n(Prior Purged)"]
    acc_scores = [73.10, 73.80, 73.60, 76.50]
    f1_scores = [69.91, 70.80, 70.64, 74.40]
    scitech_counts = [62, 66, 66, 89]
    
    colors = [OLD_COLOR, "#e9c46a", "#f4a261", NEW_COLOR]
    
    # Left Subplot: Accuracy & F1 Progression
    x = np.arange(len(variants))
    w = 0.35
    b1 = ax1.bar(x - w/2, acc_scores, w, label="Test Accuracy (%)", color="#3182ce", edgecolor="black", linewidth=1.1)
    b2 = ax1.bar(x + w/2, f1_scores, w, label="Macro-F1 (%)", color="#38a169", edgecolor="black", linewidth=1.1)
    
    ax1.set_title("Distillation Performance Across 4 Variants", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(65, 80)
    ax1.legend(loc="upper left", fontsize=9.5, frameon=True)
    
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        
    ax1.annotate("Best Overall:\n+3.40% Acc\n+4.49% F1", 
                 xy=(3, 76.5), xytext=(2.4, 77.8),
                 fontsize=9, fontweight="bold", color="#1b4332",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                 arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    # Right Subplot: Minority Class 'Sci/Tech' Prediction Progression
    b_sci = ax2.bar(variants, scitech_counts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.axhline(250, color=TRUTH_COLOR, linestyle="--", linewidth=1.3, label="Ground Truth (250)")
    
    ax2.set_title("Minority Class 'Sci/Tech' Predictions (Target = 250)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Count", fontsize=11, fontweight="bold")
    ax2.set_xticklabels(variants, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 270)
    ax2.legend(loc="upper left", fontsize=9.5, frameon=True)
    
    for bar, val in zip(b_sci, scitech_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5, f"{val} / 250\n({val/250*100:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax2.annotate("Highest Recovery:\n+43.5% vs Baseline", 
                 xy=(3, 89), xytext=(2.2, 140),
                 fontsize=9, fontweight="bold", color="#1b4332",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                 arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    plt.suptitle("AG News Comprehensive 4-Way Methodological Ablation (N=1,000 Test Set)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig3_agnews_ablation_progression.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Figure 3 to {out_path}")

# ==============================================================================
# FIGURE 4: Teacher vs Student Complete Direct Comparison
# ==============================================================================
def make_figure4():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    models = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [72.60, 73.10]    # Raw Teacher / Standard KD Student
    new_accs = [73.30, 76.50]    # Ada Teacher / Static CKD Student
    
    x = np.arange(len(models))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated (Old / Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated (New / Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("End-to-End Impact of Prior Calibration (N=1,000 Balanced Test Set)\nTeacher Direct Inference vs Distilled Student Performance", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(65, 80)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("+0.70% Lift\n(Inference)", xy=(0 + w/2, 73.30), xytext=(0, 75.5),
                ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
                
    ax.annotate("+3.40% Lift\n(Distillation)", xy=(1 + w/2, 76.50), xytext=(1, 78.5),
                ha="center", fontsize=10, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig4_teacher_vs_student_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Figure 4 to {out_path}")

def main():
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()

if __name__ == "__main__":
    main()

