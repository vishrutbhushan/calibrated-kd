import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForImageTextToText

MODEL_ID = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    local_files_only=True
)
model.eval()

VERBALIZERS = ["A", "B", "C", "D", "E"]
verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in VERBALIZERS]
print(f"Verbalizers: {list(zip(VERBALIZERS, verbalizer_ids))}")

SYSTEM_INSTRUCTION = (
    "You are an expert linguistic annotator. Classify the pragmatic suggestion form of the provided customer review "
    "into exactly one of the following five categories:\n"
    "A: direct_imperative (Direct command/order to the business)\n"
    "B: modal_deontic (Should/ought to/must statements)\n"
    "C: optative (Wishes/hopes/desires)\n"
    "D: conditional (If/then hypothetical recommendations)\n"
    "E: interrogative (Suggestion phrased as a rhetorical question)\n\n"
    "Respond with ONLY the single letter corresponding to the correct category."
)

null_contents = {
    "en": "Review: N/A\nAnswer:",
    "hi": "समीक्षा: लागू नहीं\nउत्तर:",
    "mr": "पुनरावलोकन: लागू नाही\nउत्तर:",
    "ta": "மதிப்பாய்வு: பொருந்தாது\nபதில்:",
    "ml": "അവലോകനം: ബാധകമല്ല\nഉത്തരം:"
}

null_vectors = {}

with torch.no_grad():
    for lang, user_content in null_contents.items():
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model(**inputs)
        last_logits = outputs.logits[0, -1, verbalizer_ids].float().cpu().numpy()
        probs = torch.softmax(torch.tensor(last_logits), dim=-1).tolist()
        null_vectors[lang] = {
            "logits": [float(x) for x in last_logits],
            "probs": [float(p) for p in probs]
        }
        print(f"\nLanguage: {lang}")
        print(f"  Logits [A, B, C, D, E]: {np.round(last_logits, 2).tolist()}")
        print(f"  Probs  [A, B, C, D, E]: {[f'{p*100:.2f}%' for p in probs]}")

os.makedirs("data/teacher_targets", exist_ok=True)
with open("data/teacher_targets/null_vectors_gemma3_4b.json", "w", encoding="utf-8") as f:
    json.dump(null_vectors, f, indent=2)

print("\nSaved null vectors to data/teacher_targets/null_vectors_gemma3_4b.json")

