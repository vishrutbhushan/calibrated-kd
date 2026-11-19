"""
run_3_additional_benchmarks.py
------------------------------
Runs the complete distillation evaluation suite across the 3 NEW authentic benchmarks:
  1. go_emotions_ekman (6 classes)
  2. sst4_finegrained (4 classes)
  3. student_questions (4 classes)

Strict constraints:
- Strictly sequential GPU execution (1 model loaded at a time).
- Outputs saved to outputs/results/main_results/<task>_all_methods.json.
"""

import subprocess
import sys
from pathlib import Path

PYTHON_EXEC = sys.executable
TASKS = ["go_emotions_ekman", "sst4_finegrained", "student_questions"]

def main():
    print("="*70)
    print("STARTING 3 ADDITIONAL BENCHMARKS EVALUATION SUITE")
    print("Strictly sequential: 1 model loaded at a time.")
    print("="*70)

    for task in TASKS:
        print(f"\n" + "#"*70)
        print(f">>> RUNNING BENCHMARK: {task} <<<")
        print("#"*70)
        cmd = [
            PYTHON_EXEC,
            "scripts/train/run_single_benchmark.py",
            "--dataset", task
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[ERROR] Benchmark {task} failed with code {ret.returncode}!")
            sys.exit(ret.returncode)
        print(f"[✓] Benchmark {task} completed successfully.\n")

    print("="*70)
    print("ALL 3 ADDITIONAL BENCHMARKS COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
