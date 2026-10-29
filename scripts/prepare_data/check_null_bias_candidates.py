"""
Check Teacher Null Bias (Unigram Prior Skew) across 4 Candidate Datasets
- Candidate 1: NLI (MNLI / XNLI - 3 Classes: Entailment, Contradiction, Neutral)
- Candidate 2: Emotion (6 Classes: Sadness, Joy, Love, Anger, Fear, Surprise)
- Candidate 3: TREC-6 (6 Classes: Abbreviation, Entity, Description, Human, Location, Numeric)
- Candidate 4: Stance Detection (3 Classes: Favor, Against, None)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

TEACHER_MODEL_PATH = "/home/vishrut/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

CANDIDATES = [
    {
        "name": "1. NLI (MNLI / XNLI - 3 Classes)",
        "classes": ["Entailment", "Contradiction", "Neutral"],
        "prompt": (
            "Premise: N/A\n"
            "Hypothesis: N/A\n"
            "Does the premise entail, contradict, or remain neutral to the hypothesis?\n"
            "Options:\n"
            "A. Entailment\n"
            "B. Contradiction\n"
            "C. Neutral\n"
            "Answer:"
        ),
        "verbalizers": ["A", "B", "C"]
    },
    {
        "name": "2. Emotion (6 Classes: Sadness, Joy, Love, Anger, Fear, Surprise)",
        "classes": ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        "prompt": (
            "Text: N/A\n"
            "What emotion is expressed in the text?\n"
            "Options:\n"
            "A. Sadness\n"
            "B. Joy\n"
            "C. Love\n"
            "D. Anger\n"
            "E. Fear\n"
            "F. Surprise\n"
            "Answer:"
        ),
        "verbalizers": ["A", "B", "C", "D", "E", "F"]
    },
    {
        "name": "3. TREC-6 (6 Classes: Question Type Classification)",
        "classes": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"],
        "prompt": (
            "Question: N/A\n"
            "What type of answer is the question asking for?\n"
            "Options:\n"
            "A. Abbreviation\n"
            "B. Entity\n"
            "C. Description\n"
            "D. Human\n"
            "E. Location\n"
            "F. Numeric\n"
            "Answer:"
        ),
        "verbalizers": ["A", "B", "C", "D", "E", "F"]
    },
    {
        "name": "4. Stance Detection (SemEval-2016 - 3 Classes)",
        "classes": ["Favor", "Against", "None"],
        "prompt": (
            "Target: N/A\n"
            "Tweet: N/A\n"
            "What is the stance towards the target?\n"
            "Options:\n"
            "A. Favor\n"
            "B. Against\n"
            "C. None\n"
            "Answer:"
        ),
        "verbalizers": ["A", "B", "C"]
    }
]

def main():
    print("Loading Gemma-3-4B-it teacher model...", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True
    )
    model.eval()

    print("\n" + "="*80, flush=True)
    print("EVALUATING TEACHER NULL BIAS (PRIOR SKEW) ACROSS 4 CANDIDATES", flush=True)
    print("="*80, flush=True)

    results_summary = []

    for cand in CANDIDATES:
        name = cand["name"]
        classes = cand["classes"]
        prompt = cand["prompt"]
        verbalizers = cand["verbalizers"]

        # Tokenize prompt and verbalizers
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        verb_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in verbalizers]

        with torch.no_grad():
            out = model(**inputs)
            last_logits = out.logits[0, -1, verb_ids].float().cpu().numpy()

        # Compute softmax probabilities of null vector
        probs = np.exp(last_logits - np.max(last_logits))
        probs = probs / np.sum(probs)

        min_logit = float(np.min(last_logits))
        max_logit = float(np.max(last_logits))
        gap = max_logit - min_logit
        max_class_idx = int(np.argmax(last_logits))
        min_class_idx = int(np.argmin(last_logits))
        ratio = probs[max_class_idx] / max(1e-12, probs[min_class_idx])

        print(f"\n--- {name} ---", flush=True)
        print(f"Prompt template:\n{prompt}", flush=True)
        print("Logits and Null Probabilities per Class:", flush=True)
        for i, (c, v, l, p) in enumerate(zip(classes, verbalizers, last_logits, probs)):
            print(f"  Option {v} [{c:14s}]: Logit = {l:6.2f}  |  Null Prior Prob = {p*100:6.2f}%", flush=True)
        print(f"-> Null Skew Gap (Max - Min Logit): {gap:.2f} logits", flush=True)
        print(f"-> Favored Class: Option {verbalizers[max_class_idx]} ({classes[max_class_idx]}) with {probs[max_class_idx]*100:.2f}% prior", flush=True)
        print(f"-> Starved Class: Option {verbalizers[min_class_idx]} ({classes[min_class_idx]}) with {probs[min_class_idx]*100:.4f}% prior", flush=True)
        print(f"-> Ratio (Favored / Starved): {ratio:.1f}x", flush=True)

        results_summary.append({
            "name": name,
            "classes": len(classes),
            "gap": gap,
            "favored": f"{verbalizers[max_class_idx]} ({classes[max_class_idx]})",
            "starved": f"{verbalizers[min_class_idx]} ({classes[min_class_idx]})",
            "ratio": ratio
        })

    print("\n" + "="*80, flush=True)
    print("CANDIDATE RANKING FOR CALIBRATED KD VALIDATION", flush=True)
    print("="*80, flush=True)
    print(f"{'Candidate Task':<45} | {'Classes':<7} | {'Null Gap':<10} | {'Favored / Starved Ratio':<20}")
    print("-" * 88)
    for res in sorted(results_summary, key=lambda x: x["gap"], reverse=True):
        print(f"{res['name']:<45} | {res['classes']:<7} | {res['gap']:<8.2f} L | {res['ratio']:<18.1f}x")
    print("="*80, flush=True)

if __name__ == "__main__":
    main()

