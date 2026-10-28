"""Verification Script for Phase 6: Code Validation Engine & Reference Tables.

Tests:
1. Database initialization and CSV loading into SQLite tables.
2. Code lookup for ICD-10 and CPT codes.
3. Clinical compatibility checks (valid vs. mismatched pairs).
4. Demographic rule enforcement (gender restrictions).
5. Code Validation Agent node execution within LangGraph ClaimState.
"""

import sys
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.agents.code_validation import code_validation_node, validate_claim_codes
from app.state import ClaimState, ExtractedClaim
from app.tools.database import (
    check_clinical_compatibility,
    get_cpt_info,
    get_db_connection,
    get_icd10_info,
    load_reference_tables,
)


def verify_database_and_references() -> bool:
    print("[1/5] Testing SQLite Database Initialization & Reference Tables...")
    load_reference_tables(force_reload=True)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM icd10_reference")
    icd_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cpt_reference")
    cpt_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM claims_history")
    history_count = cursor.fetchone()[0]

    conn.close()

    print(f"  Loaded {icd_count} ICD-10 codes, {cpt_count} CPT codes, {history_count} claim history rows.")
    assert icd_count >= 10, f"Expected at least 10 ICD-10 codes, got {icd_count}"
    assert cpt_count >= 5, f"Expected at least 5 CPT codes, got {cpt_count}"
    assert history_count >= 1, "Expected at least 1 seeded claim history record"

    print("  [PASS] SQLite Reference Tables loaded successfully.")
    return True


def verify_code_lookups() -> bool:
    print("\n[2/5] Testing Exact & Prefix Reference Lookups...")

    # Look up Cataract ICD-10
    h25 = get_icd10_info("H25.011")
    assert h25 is not None, "Failed to look up H25.011"
    assert "cataract" in h25["description"].lower(), f"Unexpected description: {h25['description']}"
    print(f"  Found ICD-10: {h25['code']} -> {h25['description']} (Category: {h25['category']})")

    # Look up Cataract CPT procedure
    cpt_66984 = get_cpt_info("66984")
    assert cpt_66984 is not None, "Failed to look up CPT 66984"
    assert cpt_66984["benchmark_median_cost"] > 0, "Benchmark median cost missing"
    print(
        f"  Found CPT: {cpt_66984['code']} -> {cpt_66984['description']} "
        f"[Median Fee: ${cpt_66984['benchmark_median_cost']:,.2f}, Stay: {cpt_66984['expected_stay_days']}d]"
    )

    print("  [PASS] Reference Lookups verified.")
    return True


def verify_clinical_compatibility() -> bool:
    print("\n[3/5] Testing Clinical Compatibility & Indication Rules...")

    # Compatible pair: Cataract diagnosis + Cataract surgery
    is_compat, note = check_clinical_compatibility("H25.011", "66984")
    assert is_compat, f"Expected compatible, got: {note}"
    print(f"  Compatible Pair [H25.011 + 66984]: {note}")

    # Incompatible pair: Cataract diagnosis + Appendectomy surgery
    is_mismatch, mismatch_note = check_clinical_compatibility("H25.011", "44970")
    assert not is_mismatch, "Expected mismatch for cataract + appendectomy!"
    print(f"  Mismatch Caught [H25.011 + 44970]: {mismatch_note}")

    print("  [PASS] Clinical compatibility rules verified.")
    return True


def verify_demographic_and_invalid_codes() -> bool:
    print("\n[4/5] Testing Demographic Restrictions & Unrecognized Codes...")

    # Test Gender Mismatch: Female-only condition on Male patient
    claim_gender_mismatch = ExtractedClaim(
        claim_id="CLM-TEST-GENDER",
        patient_id="PAT-TEST-1",
        patient_gender="Male",
        patient_age=32,
        diagnosis_codes=["N80.0"],  # Endometriosis (Female only)
        diagnosis_descriptions=["Endometriosis of uterus"],
        procedure_codes=["99213"],  # General visit
        procedure_descriptions=["Office visit"],
        estimated_cost=2500.0,
    )
    report_gender = validate_claim_codes(claim_gender_mismatch)
    assert not report_gender.all_codes_valid, "Expected gender mismatch to fail validation"
    gender_flag_found = any("Gender mismatch" in f for item in report_gender.items for f in item.flags)
    assert gender_flag_found, "Gender mismatch flag not raised"
    print(f"  Demographic Mismatch Caught: {report_gender.items[0].flags[0]}")

    # Test Unrecognized Code
    claim_bad_code = ExtractedClaim(
        claim_id="CLM-TEST-BAD",
        patient_id="PAT-TEST-2",
        diagnosis_codes=["ZZ99.999"],
        procedure_codes=["99999"],
        estimated_cost=5000.0,
    )
    report_bad = validate_claim_codes(claim_bad_code)
    assert not report_bad.all_codes_valid, "Expected invalid codes to fail validation"
    assert not report_bad.items[0].is_valid, "Unrecognized ICD-10 should have is_valid=False"
    assert not report_bad.items[1].is_valid, "Unrecognized CPT should have is_valid=False"
    print(f"  Invalid Codes Caught: {report_bad.items[0].flags[0]} | {report_bad.items[1].flags[0]}")

    print("  [PASS] Demographic and invalid code guardrails verified.")
    return True


def verify_node_execution() -> bool:
    print("\n[5/5] Testing LangGraph Code Validation Node Execution...")

    # Case 001: Cataract Claim (fully valid)
    extracted = ExtractedClaim(
        claim_id="CLM-2026-001",
        patient_id="PAT-98421",
        patient_age=62,
        patient_gender="Male",
        diagnosis_codes=["H25.011"],
        diagnosis_descriptions=["Senile nuclear cataract, right eye"],
        procedure_codes=["66984"],
        procedure_descriptions=["Extracapsular cataract removal with insertion of IOL"],
        estimated_cost=36000.0,
    )

    state: ClaimState = {
        "claim_id": "CLM-2026-001",
        "created_at": "2026-11-05T10:00:00Z",
        "extracted_claim": extracted,
        "audit_trail": ["[ExtractionNode] Claim extracted."],
        "errors": [],
    }

    result = code_validation_node(state)

    assert "code_validation" in result, "code_validation missing from output dictionary"
    report = result["code_validation"]
    assert report.all_codes_valid, f"Expected all codes valid for Case 001, got: {report.clinical_consistency_note}"
    assert len(report.items) == 2, f"Expected 2 code items, got {len(report.items)}"

    assert "audit_trail" in result, "audit_trail missing from output"
    audit_entry = result["audit_trail"][0]
    print(f"  Audit entry: {audit_entry}")
    assert "[CodeValidationAgent]" in audit_entry, "Audit message format invalid"

    print("  [PASS] LangGraph Node integration verified.")
    return True


def main():
    print("=" * 60)
    print("PHASE 6: CODE VALIDATION ENGINE & REFERENCE TABLES VERIFICATION")
    print("=" * 60)

    try:
        verify_database_and_references()
        verify_code_lookups()
        verify_clinical_compatibility()
        verify_demographic_and_invalid_codes()
        verify_node_execution()

        print("\n" + "=" * 60)
        print("[OK] Phase 6 Code Validation Engine & SQLite Tables Verified!")
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
