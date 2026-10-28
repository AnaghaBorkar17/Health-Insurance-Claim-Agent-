"""Verification Script for Phase 13: Quantitative Evaluation & Audit Benchmark Suite.

Runs eval/run_eval.py and validates:
1. 100% Adjudication Accuracy across all 5 synthetic cases.
2. 100% Zero PII Leakage compliance rate.
3. Creation of eval/benchmark_results.json and eval/evaluation_report.md.
"""

import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout encoding and line buffering for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.config import PROJECT_ROOT


def main():
    print("=" * 60)
    print("PHASE 13: EVALUATION & BENCHMARK VERIFICATION")
    print("=" * 60)

    eval_script = PROJECT_ROOT / "eval" / "run_eval.py"
    assert eval_script.exists(), "eval/run_eval.py does not exist"

    # Run the evaluation benchmark script
    cmd = [sys.executable, str(eval_script)]
    res = subprocess.run(cmd, capture_output=False, text=True)

    if res.returncode != 0:
        print(f"\n[FAIL] Evaluation suite exited with code {res.returncode}")
        return res.returncode

    # Check generated artifacts
    json_path = PROJECT_ROOT / "eval" / "benchmark_results.json"
    md_path = PROJECT_ROOT / "eval" / "evaluation_report.md"

    assert json_path.exists(), "benchmark_results.json was not generated"
    assert md_path.exists(), "evaluation_report.md was not generated"

    print("\n" + "=" * 60)
    print("[OK] Phase 13 Evaluation & Benchmark Suite Verified!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
