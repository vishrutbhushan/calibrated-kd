"""
generate_dedicated_dbpedia_figures.py
-------------------------------------
Generates focused, publication-grade 300 DPI figures for DBpedia-14:
  Figure 1: Old vs New Performance Benchmark (Accuracy & Macro-F1)
  Figure 2: Old vs New Per-Class Prediction Distribution (14 Classes)
  Figure 3: Minority Class Starvation & Recovery Breakdown
  Figure 4: Teacher vs Student Complete Comparison
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/7c46b483-1886-465b-a169-56ae303c4794")
CAL_KD_FIG_DIR = Path("/home/vishrut/calibrated-kd/figures/dbpedia")
CAL_KD_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

OLD_COLOR = "#d9534f"      # Coral Red for Old / Baseline
NEW_COLOR = "#2a9d8f"      # Teal for New / Calibrated
TRUTH_COLOR = "#4a5568"    # Slate Gray for Ground Truth

# ==============================================================================
# FIGURE 1: DBpedia Old vs New Accuracy & Macro-F1
# ==============================================================================
def make_figure1():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [77.29, 76.55]   # Standard KD
    new_scores = [82.00, 81.02]   # Calibrated KD
    
    x = np.arange(len(categories))
    w = 0.32
    
    rects1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    rects2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("DBpedia-14 Distillation Benchmark (N=700 Balanced Test Set)\nDirect 'Before vs After' Performance Comparison", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(65, 90)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.5, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.5, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["+4.71% Acc", "+4.47% F1"]
    for i, (r1, r2, delta) in enumerate(zip(rects1, rects2, deltas)):
        y_max = max(r1.get_height(), r2.get_height()) + 2.4
        ax.annotate(delta, xy=(x[i] + w/2, r2.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color="#1b4332",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    for d in [ARTIFACT_DIR, CAL_KD_FIG_DIR]:
        plt.savefig(d / "fig1_dbpedia_old_vs_new_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved DBpedia Figure 1")

# ==============================================================================
# FIGURE 2: DBpedia 14-Class Prediction Distribution
# ==============================================================================
def make_figure2():
    fig, ax = plt.subplots(figsize=(16, 6.2), dpi=300)
    
    classes = [
        "Company", "EduInst", "Artist", "Athlete",
        "OfficeHolder", "Transport", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
    ]
    ground_truth = [50] * 14
    old_preds = [130, 52, 38, 24, 49, 21, 49, 53, 51, 15, 89, 49, 48, 32]
    new_preds = [91, 54, 44, 26, 53, 40, 48, 52, 51, 23, 89, 51, 48, 30]
    
    x = np.arange(len(classes))
    w = 0.27
    
    ax.bar(x - w, ground_truth, w, label="Ground Truth (Balanced 50/Class)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", linewidth=1.1)
    b_old = ax.bar(x, old_preds, w, label="Standard KD (Old Baseline - Severe Skew)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    b_new = ax.bar(x + w, new_preds, w, label="Calibrated KD (New Proposed - Rescued)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.axhline(50, color=TRUTH_COLOR, linestyle="--", linewidth=1.2, alpha=0.7)
    
    ax.set_title("DBpedia 14-Class Prediction Distribution (N=700 Balanced Test Set)\nDirect 'Old vs New' Comparison Demonstrating Minority Class Recovery", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9.5, fontweight="bold", rotation=25, ha="right")
    ax.set_ylim(0, 150)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=NEW_COLOR)
        
    # Highlighting Company deflating
    ax.annotate("Company Deflated:\n130 → 91 (-30%)", 
                xy=(0 + w, 91), xytext=(0.8, 125),
                ha="center", fontsize=9, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))
    
    # Highlighting Transport rescue
    ax.annotate("Transport Rescued:\n21 → 40 (+90.5%)", 
                xy=(5 + w, 40), xytext=(5.2, 85),
                ha="center", fontsize=9, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))

    # Highlighting Animal rescue
    ax.annotate("Animal Rescued:\n15 → 23 (+53%)", 
                xy=(9 + w, 23), xytext=(9.2, 65),
                ha="center", fontsize=8.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))

    plt.tight_layout()
    for d in [ARTIFACT_DIR, CAL_KD_FIG_DIR]:
        plt.savefig(d / "fig2_dbpedia_old_vs_new_class_counts.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved DBpedia Figure 2")

# ==============================================================================
# FIGURE 3: Teacher vs Student Direct Comparison
# ==============================================================================
def make_figure3():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    models = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [71.29, 77.29]    # Raw Teacher / Standard KD Student
    new_accs = [73.29, 82.00]    # Calibrated Teacher / Static CKD Student
    
    x = np.arange(len(models))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated (Old / Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated (New / Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("DBpedia-14: End-to-End Impact of Prior Calibration (N=700 Balanced Test Set)\nTeacher Direct Inference vs Distilled Student Performance", 
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(65, 87)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("+2.00% Lift\n(Inference)", xy=(0 + w/2, 73.29), xytext=(0, 76.5),
                ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
                
    ax.annotate("+4.71% Lift\n(Distillation)", xy=(1 + w/2, 82.00), xytext=(1, 84.5),
                ha="center", fontsize=10, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    for d in [ARTIFACT_DIR, CAL_KD_FIG_DIR]:
        plt.savefig(d / "fig3_dbpedia_teacher_vs_student.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Saved DBpedia Figure 3")

def main():
    make_figure1()
    make_figure2()
    make_figure3()

if __name__ == "__main__":
    main()
