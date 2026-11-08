"""
validate_null_bias_oracle.py
----------------------------
Validates the content-free null-prompt bias vector (W_null) against an empirical oracle
fitted on 100 held-out labeled examples across AG News, Emotion, and TREC-6.
- Fits regularized per-class additive correction vector b* using L-BFGS.
- Compares b* with -W_null using centered cosine similarity, Pearson correlation, and rank alignment.
"""

import json
from pathlib import Path
import numpy as np
import scipy.optimize as opt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TARGETS_DIR = ROOT / "data" / "teacher_targets"
RESULTS_DIR = ROOT / "results" / "oracle_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def fit_oracle(logits, golds, num_classes, l2_reg=1e-3):
    M = len(golds)
    def obj(b):
        adj = logits + b
        p = softmax(adj)
        log_p = np.log(np.maximum(p[np.arange(M), golds], 1e-12))
        loss = -np.mean(log_p) + l2_reg * np.sum(b**2)
        grad = np.zeros(num_classes)
        for c in range(num_classes):
            grad[c] = np.mean(p[:, c] - (golds == c).astype(float)) + 2 * l2_reg * b[c]
        return loss, grad
    res = opt.minimize(obj, np.zeros(num_classes), method="L-BFGS-B", jac=True)
    return res.x

def centered_cosine(v1, v2):
    c1 = v1 - np.mean(v1)
    c2 = v2 - np.mean(v2)
    return float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2)))

def main():
    benchmarks = [
        ("AG News", "teacher_targets_agnews_1000_gemma3_4b.npz", ["World", "Sports", "Business", "Sci/Tech"]),
        ("Emotion", "teacher_targets_emotion_1200_gemma3_4b.npz", ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]),
        ("TREC-6", "teacher_targets_trec_1200_gemma3_4b.npz", ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]),
    ]

    results = {}
    print(f"{'='*75}\nEMPIRICAL ORACLE VALIDATION OF NULL BIAS VECTOR (W_null)\n{'='*75}")

    for name, fname, classes in benchmarks:
        path = TARGETS_DIR / fname
        d = np.load(path)
        logits = d["train_logits_raw"][:100]
        golds = d["train_golds"][:100]
        W_null = d["null_logits"]
        K = len(W_null)

        b_oracle = fit_oracle(logits, golds, K)
        ckd_vec = -W_null

        cos = centered_cosine(b_oracle, ckd_vec)
        r = float(np.corrcoef(b_oracle, ckd_vec)[0, 1])
        rho = float(spearmanr(b_oracle, ckd_vec).correlation)

        dominant_null_class = classes[int(np.argmax(W_null))]
        most_penalized_oracle_class = classes[int(np.argmin(b_oracle))]

        results[name] = {
            "num_classes": K,
            "classes": classes,
            "W_null": W_null.tolist(),
            "minus_W_null": ckd_vec.tolist(),
            "b_oracle": b_oracle.tolist(),
            "centered_cosine_sim": cos,
            "pearson_r": r,
            "spearman_rho": rho,
            "dominant_null_class": dominant_null_class,
            "most_penalized_oracle_class": most_penalized_oracle_class,
            "alignment_match": dominant_null_class == most_penalized_oracle_class
        }

        print(f"\nBenchmark: {name} (K = {K})")
        print(f"  Classes: {classes}")
        print(f"  Null Vector W_null:      {np.round(W_null, 2)}")
        print(f"  CKD Vector (-W_null):    {np.round(ckd_vec, 2)}")
        print(f"  Empirical Oracle b*:     {np.round(b_oracle, 2)}")
        print(f"  Centered Cosine Sim:     {cos:.4f}")
        print(f"  Pearson Correlation (r): {r:.4f}")
        print(f"  Dominant Null Spike:     {dominant_null_class} (W_null = {np.max(W_null):.2f})")
        print(f"  Oracle Max Penalty:      {most_penalized_oracle_class} (b* = {np.min(b_oracle):.2f})")
        print(f"  Directional Alignment:   {'PERFECT MATCH (Both target ' + dominant_null_class + ')' if dominant_null_class == most_penalized_oracle_class else 'Different'}")

    out_json = RESULTS_DIR / "oracle_validation_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Oracle validation metrics saved to {out_json}")

if __name__ == "__main__":
    main()
