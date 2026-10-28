"""Phase 2 Verification Script.

Tests:
1. Core dependency imports (langgraph, langchain, chromadb, pdfplumber, fitz, pandas)
2. Typed State import and instantiation (ClaimState, ExtractedClaim, PolicyClause)
3. Fictional policy documents availability and section coverage
4. Synthetic test cases validation (5 scenarios)
"""

import sys
import json
from pathlib import Path

# Color styling for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"  {status} {name}")
    if details:
        indent = " " * 9
        print(f"{indent}{details}")


def check_dependencies() -> bool:
    print(f"\n{BLUE}[1/4] Verifying Phase 2 Core Dependencies...{RESET}")
    all_ok = True
    packages = [
        ("langgraph", "langgraph"),
        ("langchain", "langchain"),
        ("chromadb", "chromadb"),
        ("pdfplumber", "pdfplumber"),
        ("pymupdf (fitz)", "fitz"),
        ("pandas", "pandas"),
    ]
    for display_name, module_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print_check(f"Package: {display_name}", True, f"Version: {ver}")
        except ImportError as e:
            print_check(f"Package: {display_name}", False, f"ImportError: {e}")
            all_ok = False
    return all_ok


def check_typed_state() -> bool:
    print(f"\n{BLUE}[2/4] Testing Typed State (app/state.py)...{RESET}")
    try:
        from app.state import (
            ClaimState,
            ExtractedClaim,
            PolicyClause,
            AdjudicationOutcome,
            AdjudicationRecommendation,
            HumanReviewAction,
            DocumentType,
        )

        # Test Pydantic model validation
        sample_claim = ExtractedClaim(
            claim_id="CLM-TEST-001",
            patient_id="PAT-001",
            patient_name_masked="[REDACTED]",
            patient_age=45,
            diagnosis_codes=["H25.0"],
            procedure_codes=["66984"],
            estimated_cost=35000.0,
        )
        assert sample_claim.claim_id == "CLM-TEST-001"
        assert sample_claim.estimated_cost == 35000.0
        print_check("Pydantic ExtractedClaim Schema", True, "Instantiated and validated fields")

        # Test ClaimState TypedDict
        sample_state: ClaimState = {
            "claim_id": "CLM-TEST-001",
            "created_at": "2026-10-26T12:00:00",
            "extracted_claim": sample_claim,
            "missing_mandatory_documents": [],
            "pii_redacted": True,
            "retrieved_clauses": [],
            "audit_trail": ["Claim initialized for testing"],
            "errors": [],
            "iteration_count": 0,
            "current_node": "init",
        }
        assert sample_state["claim_id"] == "CLM-TEST-001"
        print_check("LangGraph ClaimState TypedDict", True, "Initialized typed state dictionary")
        return True
    except Exception as e:
        print_check("Typed State Validation", False, f"Error: {e}")
        return False


def check_fictional_policies() -> bool:
    print(f"\n{BLUE}[3/4] Checking Fictional Policy Documents...{RESET}")
    policy_path = Path(__file__).parent / "data" / "policies" / "careplus_comprehensive_policy.txt"
    if not policy_path.exists():
        print_check("CarePlus Policy Document", False, f"Missing: {policy_path}")
        return False

    content = policy_path.read_text(encoding="utf-8")
    has_s1 = "SECTION 1" in content
    has_s2 = "SECTION 2" in content
    has_s3 = "SECTION 3" in content
    has_s4 = "SECTION 4" in content
    has_s5 = "SECTION 5" in content

    all_sections = has_s1 and has_s2 and has_s3 and has_s4 and has_s5
    print_check(
        "CarePlus Policy Document Structure",
        all_sections,
        f"Verified 5 sections ({len(content)} characters): Inpatient, Sub-limits, Waiting Periods, Exclusions, Docs",
    )
    return all_sections


def check_synthetic_cases() -> bool:
    print(f"\n{BLUE}[4/4] Checking Synthetic Cases & Sample Documents...{RESET}")
    cases_dir = Path(__file__).parent / "data" / "synthetic_cases"
    case_files = list(cases_dir.glob("case_*.json"))
    passed = len(case_files) >= 5

    print_check(
        f"Synthetic Scenarios ({len(case_files)} found)",
        passed,
        f"Scenarios: {[f.name for f in case_files]}",
    )

    docs_dir = cases_dir / "sample_documents"
    raw_docs = list(docs_dir.glob("*.txt")) if docs_dir.exists() else []
    docs_ok = len(raw_docs) >= 3
    print_check(
        f"Sample Raw Claim Text Files ({len(raw_docs)} found)",
        docs_ok,
        f"Files: {[f.name for f in raw_docs]}",
    )
    return passed and docs_ok


def main():
    print("=" * 60)
    print(" CLAIM AGENTS: PHASE 2 STATE & SYNTHETIC DATA VERIFICATION")
    print("=" * 60)

    d_ok = check_dependencies()
    s_ok = check_typed_state()
    p_ok = check_fictional_policies()
    c_ok = check_synthetic_cases()

    print("\n" + "=" * 60)
    if d_ok and s_ok and p_ok and c_ok:
        print(f"{GREEN}[OK] Phase 2 State, Dependencies & Synthetic Data Ready!{RESET}")
    else:
        print(f"{RED}[FAIL] Phase 2 Verification encountered issues.{RESET}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
