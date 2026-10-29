"""
generate_dedicated_emotion_figures.py
-------------------------------------
Generates the complete, dedicated 4-figure suite (300 DPI) exclusively for
Benchmark 2: Emotion (6-Class Affect Classification, N=300 balanced test set).
Completely unmixed, strictly mirroring the AG News reporting suite.

Figure 1: Emotion Old vs New Performance Benchmark (Accuracy & Macro-F1)
Figure 2: Emotion Old vs New Per-Class Prediction Counts & Collapse Recovery (6 Emotions)
Figure 3: Emotion Full 4-Way Methodological Ablation Progression
Figure 4: Emotion Teacher Direct Inference vs Distilled Student Performance (Old vs New)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ARTIFACT_DIR = Path("/home/vishrut/.gemini/antigravity/brain/7c46b483-1886-465b-a169-56ae303c4794")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

OLD_COLOR = "#d9534f"      # Coral Red for Old / Baseline
NEW_COLOR = "#2a9d8f"      # Teal for New / Calibrated
TRUTH_COLOR = "#4a5568"    # Slate Gray for Ground Truth
ACCENT_BLUE = "#3182ce"

# ==============================================================================
# FIGURE 1: Overall Performance Comparison (Old vs New on 300 Test Set)
# ==============================================================================
def make_figure1():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    categories = ["Test Accuracy", "Macro-F1 Score"]
    old_scores = [40.67, 37.59]   # Standard KD
    new_scores = [43.33, 39.03]   # Static CKD
    
    x = np.arange(len(categories))
    w = 0.32
    
    rects1 = ax.bar(x - w/2, old_scores, w, label="Standard KD (Old Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    rects2 = ax.bar(x + w/2, new_scores, w, label="Calibrated KD (New Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("Emotion 6-Class Benchmark (N=300 Balanced Test Set)\nDirect 'Old vs New' Performance Comparison", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(30, 48)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=OLD_COLOR)
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f"{r.get_height():.2f}%", 
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NEW_COLOR)
        
    deltas = ["+2.66% Acc", "+1.44% F1"]
    for i, (r1, r2, delta) in enumerate(zip(rects1, rects2, deltas)):
        y_max = max(r1.get_height(), r2.get_height()) + 1.8
        ax.annotate(delta, xy=(x[i] + w/2, r2.get_height()), xytext=(x[i], y_max),
                    ha="center", fontsize=10.5, fontweight="bold", color="#1b4332",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig1_emotion_old_vs_new_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Emotion Figure 1 to {out_path}")

# ==============================================================================
# FIGURE 2: Per-Class Prediction Counts & Collapse Recovery (6 Emotions)
# ==============================================================================
def make_figure2():
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    
    classes = ["Sadness\n(Class 0)", "Joy\n(Class 1)", "Love\n(Class 2)", "Anger\n(Class 3)", "Fear\n(Class 4)", "Surprise\n(Class 5)"]
    ground_truth = [50, 50, 50, 50, 50, 50]
    old_preds = [152, 41, 34, 53, 20, 0]    # Surprise completely collapsed to 0 in Standard KD
    new_preds = [75, 52, 41, 104, 27, 1]    # Sadness cut in half, Surprise revived in Calibrated KD
    
    x = np.arange(len(classes))
    w = 0.26
    
    ax.bar(x - w, ground_truth, w, label="Ground Truth (Balanced 50/Class)", color=TRUTH_COLOR, alpha=0.55, edgecolor="black", linewidth=1.1)
    b_old = ax.bar(x, old_preds, w, label="Standard KD (Old Baseline - Severe Prior Bias)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    b_new = ax.bar(x + w, new_preds, w, label="Calibrated KD (New Proposed - Rebalanced)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.axhline(50, color=TRUTH_COLOR, linestyle="--", linewidth=1.2, alpha=0.7)
    
    ax.set_title("Emotion 6-Class Prediction Distribution (N=300 Balanced Test Set)\nDirect 'Old vs New' Comparison: Bias Reduction & Collapse Prevention", 
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Predicted Test Items (Count)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 180)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    
    for b in b_old:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=OLD_COLOR)
    for b in b_new:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, val + 2, f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("Sadness Cut in Half:\n152 → 75 (-51%)", 
                xy=(0 + w, 75), xytext=(0.3, 140),
                ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="#2d6a4f", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))
    
    ax.annotate("Surprise Collapse\nin Standard KD: 0/50", 
                xy=(5, 0), xytext=(4.7, 40),
                ha="center", fontsize=9.5, fontweight="bold", color="#c62828",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c62828", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5))

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig2_emotion_old_vs_new_class_counts.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Emotion Figure 2 to {out_path}")

# ==============================================================================
# FIGURE 3: Full 4-Way Methodological Ablation Progression
# ==============================================================================
def make_figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    variants = [
        "Variant 1\nStandard KD\n(Baseline)", 
        "Variant 2\nStatic CKD\n(Prior Purged)", 
        "Variant 3\nAda-CKD\n(Adaptive)", 
        "Variant 4\nPI-KD\n(Regularized)"
    ]
    accs = [40.67, 43.33, 41.67, 40.67]
    f1s = [37.59, 39.03, 38.37, 37.29]
    sadness_counts = [152, 75, 142, 146] # Target is 50
    
    x = np.arange(len(variants))
    w = 0.32
    
    # Left Subplot: Metric Comparison
    b1 = ax1.bar(x - w/2, accs, w, label="Student Accuracy", color=OLD_COLOR, edgecolor="black", linewidth=1.1)
    b2 = ax1.bar(x + w/2, f1s, w, label="Student Macro-F1", color=NEW_COLOR, edgecolor="black", linewidth=1.1)
    
    ax1.set_title("Methodological Ablation Performance (N=300)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(30, 48)
    ax1.legend(loc="lower right", fontsize=9.5, frameon=True)
    
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
        
    ax1.annotate("Best Overall:\n+2.66% Acc\n+1.44% F1", 
                 xy=(1, 43.33), xytext=(1, 45.4),
                 ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                 arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    # Right Subplot: Sadness De-biasing Progression (Ground Truth = 50)
    colors = [OLD_COLOR, NEW_COLOR, "#e9c46a", "#f4a261"]
    b_sad = ax2.bar(variants, sadness_counts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.axhline(50, color=TRUTH_COLOR, linestyle="--", linewidth=1.3, label="Ground Truth Target (50)")
    
    ax2.set_title("Dominant Class 'Sadness' Predictions (Target = 50)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Test Items", fontsize=11, fontweight="bold")
    ax2.set_xticklabels(variants, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 180)
    ax2.legend(loc="upper right", fontsize=9.5, frameon=True)
    
    for bar, val in zip(b_sad, sadness_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val} / 50\n({val/50*100:.0f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax2.annotate("Deepest De-biasing:\n-51% Over-prediction", 
                 xy=(1, 75), xytext=(1.2, 120),
                 ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                 arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    plt.suptitle("Emotion 6-Class 4-Way Distillation Ablation (N=300 Test Set)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig3_emotion_ablation_progression.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Emotion Figure 3 to {out_path}")

# ==============================================================================
# FIGURE 4: Teacher vs Student Direct Comparison
# ==============================================================================
def make_figure4():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    models = ["Teacher Direct Inference\n(Gemma-3-4B-it)", "Distilled Student\n(Qwen3-4B LoRA)"]
    old_accs = [37.67, 40.67]    # Raw Teacher / Standard KD Student
    new_accs = [39.33, 43.33]    # Static Cal Teacher / Static CKD Student
    
    x = np.arange(len(models))
    w = 0.32
    
    r1 = ax.bar(x - w/2, old_accs, w, label="Uncalibrated (Old / Baseline)", color=OLD_COLOR, edgecolor="black", linewidth=1.2)
    r2 = ax.bar(x + w/2, new_accs, w, label="Calibrated (New / Proposed)", color=NEW_COLOR, edgecolor="black", linewidth=1.2)
    
    ax.set_title("Emotion: Impact of Prior Calibration Across Models (N=300 Balanced Test Set)\nTeacher Direct Inference vs Distilled Student Performance", 
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(30, 48)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    
    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=OLD_COLOR)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.2f}%", ha="center", fontsize=10, fontweight="bold", color=NEW_COLOR)
        
    ax.annotate("+1.66% Lift\n(Inference)", xy=(0 + w/2, 39.33), xytext=(0, 42.0),
                ha="center", fontsize=9.5, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
                
    ax.annotate("+2.66% Lift\n(Distillation)", xy=(1 + w/2, 43.33), xytext=(1, 45.5),
                ha="center", fontsize=10, fontweight="bold", color="#1b4332",
                bbox=dict(boxstyle="round,pad=0.25", fc="#d8f3dc", ec="#2d6a4f"),
                arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.2))
    
    plt.tight_layout()
    out_path = ARTIFACT_DIR / "fig4_emotion_teacher_vs_student.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Emotion Figure 4 to {out_path}")

def main():
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()

if __name__ == "__main__":
    main()
