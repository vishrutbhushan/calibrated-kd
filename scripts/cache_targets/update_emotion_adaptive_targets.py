"""
update_emotion_adaptive_targets.py
----------------------------------
Computes Entropy-Adaptive targets (z - alpha(x) * W_null) for the Emotion dataset
and updates data/teacher_targets/teacher_targets_emotion_1200_gemma3_4b.npz.
Evaluates Teacher Direct Inference with Ada-CKD on the 300 test samples.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

NPZ_PATH = Path("data/teacher_targets/teacher_targets_emotion_1200_gemma3_4b.npz")
data = dict(np.load(NPZ_PATH))

train_logits = data["train_logits_raw"]
test_logits = data["test_logits_raw"]
null_logits = data["null_logits"]
test_golds = data["test_golds"]

def compute_adaptive(logits, null_logits):
    raw_probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    raw_probs = raw_probs / np.sum(raw_probs, axis=-1, keepdims=True)
    
    eps = 1e-12
    safe_probs = np.clip(raw_probs, eps, 1.0)
    entropy = -np.sum(safe_probs * np.log(safe_probs), axis=-1)
    max_entropy = np.log(6.0) # 6 classes
    
    alpha = np.clip(entropy / max_entropy, 0.0, 1.0)[:, np.newaxis]
    ada_logits = logits - alpha * null_logits
    ada_probs = np.exp(ada_logits - np.max(ada_logits, axis=-1, keepdims=True))
    ada_probs = ada_probs / np.sum(ada_probs, axis=-1, keepdims=True)
    return ada_logits, ada_probs, alpha.flatten()

train_ada_logits, train_probs_ada, train_alphas = compute_adaptive(train_logits, null_logits)
test_ada_logits, test_probs_ada, test_alphas = compute_adaptive(test_logits, null_logits)

# Teacher Ada Metrics on Test
ada_preds = np.argmax(test_ada_logits, axis=-1)
raw_preds = np.argmax(test_logits, axis=-1)
static_preds = np.argmax(data["test_logits_cal"], axis=-1)

raw_acc = accuracy_score(test_golds, raw_preds) * 100
static_acc = accuracy_score(test_golds, static_preds) * 100
ada_acc = accuracy_score(test_golds, ada_preds) * 100

raw_f1 = f1_score(test_golds, raw_preds, average="macro") * 100
static_f1 = f1_score(test_golds, static_preds, average="macro") * 100
ada_f1 = f1_score(test_golds, ada_preds, average="macro") * 100

print("="*75)
print("EMOTION TEACHER DIRECT INFERENCE COMPARISON (N=300)")
print("="*75)
print(f"Raw Teacher:        Acc = {raw_acc:.2f}% | Macro-F1 = {raw_f1:.2f}% | Counts = {np.bincount(raw_preds, minlength=6).tolist()}")
print(f"Static Calibrated:  Acc = {static_acc:.2f}% | Macro-F1 = {static_f1:.2f}% | Counts = {np.bincount(static_preds, minlength=6).tolist()}")
print(f"Entropy-Adaptive:   Acc = {ada_acc:.2f}% | Macro-F1 = {ada_f1:.2f}% | Counts = {np.bincount(ada_preds, minlength=6).tolist()}")
print(f"Average Alpha:      {np.mean(test_alphas):.3f}")
print("="*75)

data["train_logits_ada"] = train_ada_logits
data["train_probs_ada"] = train_probs_ada
data["train_alphas"] = train_alphas
data["test_logits_ada"] = test_ada_logits
data["test_probs_ada"] = test_probs_ada
data["test_alphas"] = test_alphas

np.savez_compressed(NPZ_PATH, **data)
print(f"[✓] Updated {NPZ_PATH} with Entropy-Adaptive targets.")

