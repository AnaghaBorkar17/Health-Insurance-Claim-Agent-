"""Verification Script for Phase 12: Streamlit Interactive Web Application Dashboard.

Tests:
1. Streamlit installation and dependency verification.
2. Synthetic case data loading & document preparation helper functions.
3. Syntax validation of ui/streamlit_app.py.
4. Streamlit CLI execution readiness check.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout encoding and line buffering for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.config import PROJECT_ROOT


def verify_streamlit_installation() -> bool:
    print("[1/4] Testing Streamlit Package Installation...")
    try:
        import streamlit
        print(f"  Streamlit version {streamlit.__version__} is installed and importable.")
        print("  [PASS] Streamlit package verified.")
        return True
    except ImportError as e:
        print(f"  [FAIL] Streamlit not installed: {e}")
        return False


def verify_synthetic_fixtures_and_loader() -> bool:
    print("\n[2/4] Testing Synthetic Case Loading for UI...")
    fixture_files = [
        "case_001_approval_cataract.json",
        "case_002_query_missing_docs.json",
        "case_003_reject_cosmetic_exclusion.json",
        "case_004_fraud_duplicate_claim.json",
        "case_005_fraud_inflated_bill.json",
    ]

    for fname in fixture_files:
        fpath = PROJECT_ROOT / "data" / "synthetic_cases" / fname
        assert fpath.exists(), f"Fixture {fname} does not exist"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "claim_id" in data, f"Fixture {fname} missing claim_id"
            assert "estimated_cost" in data, f"Fixture {fname} missing estimated_cost"
            print(f"  Loaded fixture: {fname} -> {data['claim_id']} ({data.get('expected_outcome')})")

    print("  [PASS] All 5 synthetic cases verified for UI ingestion.")
    return True


def verify_app_syntax() -> bool:
    print("\n[3/4] Testing Python AST Syntax of ui/streamlit_app.py...")
    app_path = PROJECT_ROOT / "ui" / "streamlit_app.py"
    assert app_path.exists(), "ui/streamlit_app.py does not exist"

    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
        ast.parse(code)  # Raises SyntaxError if code is invalid

    print(f"  Syntax tree parsed successfully ({len(code.splitlines())} lines of code).")
    print("  [PASS] ui/streamlit_app.py syntax verified.")
    return True


def verify_streamlit_cli_ready() -> bool:
    print("\n[4/4] Testing Streamlit CLI Execution Readiness...")
    venv_python = Path(sys.executable)
    venv_streamlit = venv_python.parent / "streamlit.exe"

    if not venv_streamlit.exists():
        # Fallback to module execution: python -m streamlit
        cmd = [str(venv_python), "-m", "streamlit", "--version"]
    else:
        cmd = [str(venv_streamlit), "--version"]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    ver_str = res.stdout.strip()
    print(f"  Streamlit CLI response: '{ver_str}'")

    print("  [PASS] Streamlit CLI is ready to launch the dashboard.")
    return True


def main():
    print("=" * 60)
    print("PHASE 12: STREAMLIT WEB APPLICATION DASHBOARD VERIFICATION")
    print("=" * 60)

    try:
        verify_streamlit_installation()
        verify_synthetic_fixtures_and_loader()
        verify_app_syntax()
        verify_streamlit_cli_ready()

        print("\n" + "=" * 60)
        print("[OK] Phase 12 Streamlit Web Application Dashboard Verified!")
        print("=" * 60)
        return 0
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion failed: {ae}")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
