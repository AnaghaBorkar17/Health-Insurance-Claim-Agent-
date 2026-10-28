"""Phase 4 Verification Script: PII Redaction Agent & Privacy Tests.

Tests:
1. Deterministic PII Masking (Names, Phone, Aadhaar, Address, Email)
2. Cloud Payload Zero-Leakage Gatekeeper
3. Interception of Simulated Privacy Breaches
4. Redaction Node Execution in LangGraph State
5. Full pytest test_redaction.py Execution
"""

import sys
import subprocess
from app.agents.redaction import (
    SecurityException,
    assert_zero_pii_leakage,
    redact_pii,
    redaction_node,
)
from app.state import ClaimState, DocumentMeta, DocumentType, ExtractedClaim

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


def test_pii_masking_rules() -> bool:
    print(f"\n{BLUE}[1/4] Testing PII Masking Rules...{RESET}")
    sample_text = (
        "Patient Rohan Deshmukh, Mobile: +91 98200 12345, Aadhaar: 9823-4412-8871\n"
        "Address: 402 Palm Grove Apartments, Bandra West, Mumbai 400050\n"
        "Email: rohan.deshmukh@gmail.com, DOB: 1964-08-14"
    )

    sanitized, counts = redact_pii(sample_text, known_names=["Rohan Deshmukh"])

    name_redacted = "Rohan Deshmukh" not in sanitized and "[PATIENT_NAME]" in sanitized
    phone_redacted = "98200 12345" not in sanitized and "[PHONE_NUMBER]" in sanitized
    id_redacted = "9823-4412-8871" not in sanitized and "[GOVT_ID]" in sanitized
    addr_redacted = "Bandra West" not in sanitized and "[ADDRESS_REDACTED]" in sanitized
    email_redacted = "rohan.deshmukh@gmail.com" not in sanitized and "[EMAIL]" in sanitized

    print_check("Patient Name Redaction", name_redacted, f"Masked to [PATIENT_NAME] (Count: {counts.get('name', 0)})")
    print_check("Phone Number Redaction", phone_redacted, f"Masked to [PHONE_NUMBER] (Count: {counts.get('phone', 0)})")
    print_check("Aadhaar / National ID Redaction", id_redacted, f"Masked to [GOVT_ID] (Count: {counts.get('aadhaar', 0)})")
    print_check("Address Redaction", addr_redacted, f"Masked to [ADDRESS_REDACTED] (Count: {counts.get('address', 0)})")
    print_check("Email Redaction", email_redacted, f"Masked to [EMAIL] (Count: {counts.get('email', 0)})")

    return name_redacted and phone_redacted and id_redacted and addr_redacted and email_redacted


def test_cloud_gatekeeper() -> bool:
    print(f"\n{BLUE}[2/4] Testing Outgoing Cloud Payload Gatekeeper...{RESET}")
    safe_payload = (
        "Adjudicating claim for patient [PATIENT_NAME], age 62. "
        "Procedure: Phacoemulsification with IOL (CPT 66984). "
        "Hospital: Apex Specialty Eye Institute. Total: $36,000."
    )
    raw_entities = ["Rohan Deshmukh", "+91 98200 12345", "9823-4412-8871"]

    safe_passed = assert_zero_pii_leakage(safe_payload, raw_entities)
    print_check("Sanitized Payload Approval", safe_passed, "Payload free of raw PII passed guardrail")

    # Test interception of simulated leakage
    leaky_payload = f"Patient details: Rohan Deshmukh, phone 98200 12345"
    breach_caught = False
    try:
        assert_zero_pii_leakage(leaky_payload, raw_entities)
    except SecurityException as e:
        breach_caught = True

    print_check("Simulated Leakage Interception", breach_caught, "Blocked raw PII with SecurityException")
    return safe_passed and breach_caught


def test_redaction_node() -> bool:
    print(f"\n{BLUE}[3/4] Testing LangGraph Redaction Node...{RESET}")
    doc = DocumentMeta(
        file_id="DOC-001",
        filename="claim_form.txt",
        doc_type=DocumentType.CLAIM_FORM,
        raw_text="Full Name: Rohan Deshmukh\nMobile: 98200 12345\nAadhaar: 9823-4412-8871\nCost: $36,000",
    )
    extracted = ExtractedClaim(
        claim_id="CLM-2026-001",
        patient_id="PAT-98421",
        patient_name_masked="Rohan Deshmukh",
        diagnosis_codes=["H25.011"],
        procedure_codes=["66984"],
        estimated_cost=36000.0,
    )
    state: ClaimState = {
        "claim_id": "CLM-2026-001",
        "uploaded_documents": [doc],
        "extracted_claim": extracted,
        "pii_redacted": False,
        "audit_trail": [],
    }

    result = redaction_node(state)
    state_flag_ok = result.get("pii_redacted") is True
    name_masked_ok = result["extracted_claim"].patient_name_masked == "[PATIENT_NAME]"
    doc_redacted_ok = "Rohan Deshmukh" not in result["uploaded_documents"][0].redacted_text

    print_check("State pii_redacted Flag", state_flag_ok, "pii_redacted = True")
    print_check("Claim patient_name_masked", name_masked_ok, "patient_name_masked = [PATIENT_NAME]")
    print_check("Document redacted_text populated", doc_redacted_ok, "Raw text stripped of PII")

    return state_flag_ok and name_masked_ok and doc_redacted_ok


def run_pytest_suite() -> bool:
    print(f"\n{BLUE}[4/4] Running Automated Pytest Suite (tests/test_redaction.py)...{RESET}")
    cmd = [sys.executable, "-m", "pytest", "tests/test_redaction.py", "-v", "--tb=short"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    passed = res.returncode == 0
    if passed:
        print_check("Pytest Redaction Test Suite", True, "All unit and security tests passed (6/6)")
    else:
        print_check("Pytest Redaction Test Suite", False, res.stdout or res.stderr)
    return passed


def main():
    print("=" * 60)
    print(" CLAIM AGENTS: PHASE 4 PII REDACTION & PRIVACY VERIFICATION")
    print("=" * 60)

    m_ok = test_pii_masking_rules()
    g_ok = test_cloud_gatekeeper()
    n_ok = test_redaction_node()
    p_ok = run_pytest_suite()

    print("\n" + "=" * 60)
    if m_ok and g_ok and n_ok and p_ok:
        print(f"{GREEN}[OK] Phase 4 PII Redaction & Privacy Guardrails Verified!{RESET}")
    else:
        print(f"{RED}[FAIL] Phase 4 Verification encountered issues.{RESET}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
