"""
plot_trec.py
------------
Generates the complete, dedicated 4-figure suite (300 DPI) exclusively for
Benchmark: TREC-6 Question Classification (N=500 canonical test set).
Completely unmixed, strictly mirroring the AG News and Emotion reporting suites.

Figure 1: TREC-6 Old vs New Performance Benchmark (Accuracy & Macro-F1)
Figure 2: TREC-6 Old vs New Per-Class Prediction Counts & Collapse Recovery (6 Classes)
Figure 3: TREC-6 Starved Entity Recovery & Description De-biasing Analysis
Figure 4: TREC-6 Teacher Direct Inference vs Distilled Student Performance (Old vs New)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
REPO_FIG_DIR = Path("/home/vishrut/calibrated-kd/figures/trec")
REPO_FIG_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Color palette (fixed across project)
OLD_COLOR = "#d9534f"      # Coral Red for Old / Baseline
NEW_COLOR = "#2a9d8f"      # Teal for New / Calibrated
TRUTH_COLOR = "#4a5568"    # Slate Gray for Ground Truth
ACCENT_BLUE = "#3182ce"

# ==============================================================================
# FIGURE 1: Overall Performance Comparison (Old vs New on 500 Test Set)
# ==============================================================================
def make_figure1():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [78.20, 75.08]   # Standard KD
    new_scores = [80.00, 77.12]   # Static CKD
    
    x = np.arange(len(categories))
    w = 0.32
    
    rects1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    rects2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("TREC-6 Benchmark (N=500 Canonical Test Set)\nDirect 'Old vs New' Distillation Performance Comparison", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(68, 86)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["+1.80% Acc", "+2.04% F1"]
    for i, (r1, r2, delta) in enumerate(zip(rects1, rects2, deltas)):
        y_max = max(r1.get_height(), r2.get_height()) + 2.0
        ax.annotate(delta, xy=(x[i] + w/2, r2.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color="#1b4332",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    for d in [REPO_FIG_DIR, ARTIFACT_DIR]:
        plt.savefig(d / "fig1_trec_old_vs_new_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved TREC Figure 1")

# ==============================================================================
# FIGURE 2: Per-Class Prediction Counts & Collapse Recovery
# ==============================================================================
def make_figure2():
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    
    classes = [
        "Abbreviation\n(Class 0)", 
        "Entity\n(Class 1)", 
        "Description\n(Class 2)", 
        "Human\n(Class 3)", 
        "Location\n(Class 4)", 
        "Numeric\n(Class 5)"
    ]
    ground_truth = [9, 94, 138, 65, 81, 113]
    old_preds = [13, 48, 178, 40, 82, 139]    # Standard KD: Description inflated (+40), Entity starved (-49%)
    new_preds = [13, 58, 159, 43, 81, 146]    # Calibrated KD: Description reduced by 19, Entity rescued +20.8%
    
    x = np.arange(len(classes))
    w = 0.26
    
    ax.bar(x - w, ground_truth, w, label="Ground Truth (Canonical Distribution)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", linewidth=1.1)
    b_old = ax.bar(x, old_preds, w, label="Standard KD (Old Baseline - Prior Biased)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    b_new = ax.bar(x + w, new_preds, w, label="Calibrated KD (New Proposed - Rebalanced)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("TREC-6 Per-Class Prediction Distribution (N=500 Canonical Test Set)\nDirect 'Old vs New' Comparison: Description De-biasing & Entity Recovery", 
                 fontsize=12, fontweight="bold", pad=15)
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
        
    ax.annotate("Description De-biased:\n178 → 159 (-19 items)\ncloser to target 138", 
                xy=(2 + w, 159), xytext=(2.3, 185),
                ha="center", fontsize=9.2, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))
    
    ax.annotate("Entity Rescued:\n48 → 58 (+20.8%)\ntowards target 94", 
                xy=(1 + w, 58), xytext=(0.7, 105),
                ha="center", fontsize=9.2, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))

    plt.tight_layout()
    for d in [REPO_FIG_DIR, ARTIFACT_DIR]:
        plt.savefig(d / "fig2_trec_old_vs_new_class_counts.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved TREC Figure 2")

# ==============================================================================
# FIGURE 3: Minority Class Starvation & Recovery Analysis
# ==============================================================================
def make_figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    models = ["Ground Truth", "Standard KD (Old)", "Calibrated KD (New)"]
    colors = [TRUTH_COLOR, OLD_COLOR, NEW_COLOR]
    
    # Left Subplot: Starved Class 'Entity' Predictions
    entity_counts = [94, 48, 58]
    b_ent = ax1.bar(models, entity_counts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_title("Starved Class: 'Entity' Recovery (Target = 94)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    
    for bar, val in zip(b_ent, entity_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val}\n({val/94*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax1.annotate("+20.8% Recovery\n(+10 test items)", 
                 xy=(2, 58), xytext=(1.8, 85),
                 ha="center", fontsize=10, fontweight="bold", color="#1b4332",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                 arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.4))

    # Right Subplot: Over-favored Class 'Description' De-biasing
    desc_counts = [138, 178, 159]
    b_desc = ax2.bar(models, desc_counts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_title("Dominant Class: 'Description' De-biasing (Target = 138)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 205)
    
    for bar, val in zip(b_desc, desc_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val}\n({val/138*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax2.annotate("-10.7% Prior Reduction\n(-19 over-predictions)", 
                 xy=(2, 159), xytext=(1.5, 120),
                 ha="center", fontsize=10, fontweight="bold", color="#1b4332",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                 arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.4))

    plt.suptitle("TREC-6 Detailed Bias Mitigation & Class Recovery Analysis (N=500)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    for d in [REPO_FIG_DIR, ARTIFACT_DIR]:
        plt.savefig(d / "fig3_trec_collapse_recovery.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved TREC Figure 3")

# ==============================================================================
# FIGURE 4: Teacher vs Student Direct Comparison
# ==============================================================================
def make_figure4():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    models = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [72.20, 78.20]    # Raw Teacher / Standard KD Student
    new_accs = [72.20, 80.00]    # Calibrated Teacher / Static CKD Student
    
    x = np.arange(len(models))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated / Standard KD (Old)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated / CKD Proposed (New)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("TREC-6: Impact of Prior Calibration Across Models (N=500 Test Set)\nTeacher Direct Inference vs Distilled Student Performance", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(65, 85)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("+0.00% Zero Distortion\n(Inference)", xy=(0 + w/2, 72.20), xytext=(0, 74.5),
                ha="center", fontsize=9.5, fontweight="bold", color="#2d3748",
                bbox=dict(boxstyle="round,pad=0.25", fc="#edf2f7", ec="#a0aec0"),
                arrowprops=dict(arrowstyle="->", color="#718096", lw=1.2))
                
    ax.annotate("+1.80% Lift\n(Distillation)", xy=(1 + w/2, 80.00), xytext=(1, 82.2),
                ha="center", fontsize=10, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    for d in [REPO_FIG_DIR, ARTIFACT_DIR]:
        plt.savefig(d / "fig4_trec_teacher_vs_student.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved TREC Figure 4")

def main():
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()

if __name__ == "__main__":
    main()
