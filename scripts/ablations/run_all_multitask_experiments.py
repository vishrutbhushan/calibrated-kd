"""
run_all_multitask_experiments.py
--------------------------------
Runs all 4 multi-task distillation combinations strictly sequentially,
evaluating both Raw and Task-Specific Post-Hoc for both Standard KD and CKD:
  Combo 1 (2 tasks): TweetEval Emotion + BBC News
  Combo 2 (2 tasks): BBC News + TREC-6
  Combo 3 (2 tasks): Twitter Emotion + TREC-6
  Combo 4 (3 tasks): TweetEval Emotion + BBC News + TREC-6
"""

import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "ablations" / "2_multitask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_EXEC = sys.executable

COMBOS = [
    {
        "combo_id": 1,
        "name": "tweeteval_bbc",
        "tasks": ["tweeteval_emotion", "bbc_news"]
    },
    {
        "combo_id": 2,
        "name": "bbc_news_trec6",
        "tasks": ["bbc_news", "trec6"]
    },
    {
        "combo_id": 3,
        "name": "twitter_emotion_trec6",
        "tasks": ["twitter_emotion", "trec6"]
    },
    {
        "combo_id": 4,
        "name": "tweeteval_bbc_trec6",
        "tasks": ["tweeteval_emotion", "bbc_news", "trec6"]
    }
]

def main():
    print("="*70)
    print("STARTING COMPLETE MULTI-TASK EXPERIMENTAL SUITE (WITH POST-HOC)")
    print("Strictly sequential GPU execution: 1 model loaded at a time.")
    print("="*70)

    for item in COMBOS:
        cid = item["combo_id"]
        tasks = item["tasks"]
        print(f"\n>>> EXECUTING COMBO {cid}: {tasks} <<<")
        cmd = [
            PYTHON_EXEC,
            "scripts/ablations/run_multitask_distillation.py",
            "--tasks"
        ] + tasks + ["--combo_id", str(cid)]

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[ERROR] Combo {cid} failed with code {ret.returncode}!")
            sys.exit(ret.returncode)
        print(f"[✓] Combo {cid} successfully completed.")

    print("\n" + "="*70)
    print("ALL MULTI-TASK EXPERIMENTS COMPLETED!")
    print("="*70)

    # Master summary consolidation
    all_files = list(OUTPUT_DIR.glob("combo*.json"))
    master_records = []
    for f in sorted(all_files):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
            master_records.append(d)

    master_path = OUTPUT_DIR / "all_multitask_summary.json"
    with open(master_path, "w", encoding="utf-8") as fp:
        json.dump(master_records, fp, indent=2)

    print(f"Master summary saved to: {master_path}")

if __name__ == "__main__":
    main()
