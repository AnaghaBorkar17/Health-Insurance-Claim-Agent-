"""Verification Script for Phase 8: Decision Synthesis Agent.

Tests:
1. Case 001: Cataract pre-authorization -> APPROVE (Within Clause 2.1 sub-limit).
2. Case 002: Missing MRI report -> QUERY (Clause 5.1 mandatory records).
3. Case 003: Cosmetic Rhinoplasty -> REJECT (Clause 4.1 permanent exclusion).
4. Case 004: Duplicate pre-authorization -> QUERY (Duplicate submission hold).
5. Case 005: Inflated bill & prolonged stay -> QUERY (Audit & justification hold).
6. LangGraph Decision Synthesis Node integration with ClaimState.
"""

import sys
from pathlib import Path

# Force UTF-8 stdout encoding and line buffering for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.agents.decision import decision_synthesis_node, synthesize_decision
from app.state import (
    AdjudicationOutcome,
    AdjudicationRecommendation,
    ClaimState,
    CodeValidationItem,
    CodeValidationReport,
    ExtractedClaim,
    FraudRiskReport,
    PolicyClause,
)


def verify_case_001_approval() -> bool:
    print("[1/6] Testing Case 001: Cataract Surgery Pre-Authorization (Expected: APPROVE)...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-001",
        patient_id="PAT-98421",
        patient_age=62,
        patient_gender="Male",
        diagnosis_codes=["H25.011"],
        diagnosis_descriptions=["Senile nuclear cataract, right eye"],
        procedure_codes=["66984"],
        procedure_descriptions=["Extracapsular cataract removal with IOL"],
        estimated_cost=36000.0,
        length_of_stay=1,
    )

    clauses = [
        PolicyClause(
            clause_id="CLAUSE 2.1",
            title="Cataract Treatment Sub-Limit",
            text="Coverage for Cataract surgery is subject to a maximum sub-limit of $40,000 per eye.",
            document_source="careplus_comprehensive_policy.txt",
            page_reference=2,
            retrieval_score=0.85,
        )
    ]

    code_val = CodeValidationReport(all_codes_valid=True, items=[])
    fraud = FraudRiskReport(risk_score=0)

    rec = synthesize_decision(
        extracted=extracted,
        missing_docs=[],
        retrieved_clauses=clauses,
        code_validation=code_val,
        fraud_report=fraud,
    )

    assert rec.outcome == AdjudicationOutcome.APPROVE, f"Expected APPROVE, got {rec.outcome}"
    assert rec.confidence >= 0.90, f"Expected high confidence, got {rec.confidence}"
    assert len(rec.policy_citations) > 0, "No policy citations provided"
    assert any("2.1" in c for c in rec.policy_citations), "Clause 2.1 not cited"
    assert len(rec.unresolved_issues) == 0, f"Expected 0 unresolved issues, got {rec.unresolved_issues}"

    print(f"  Decision: {rec.outcome.value} (Confidence: {rec.confidence:.2f})")
    print(f"  Citations: {rec.policy_citations}")
    print(f"  Summary: {rec.summary}")
    print("  [PASS] Case 001 Approval verified.")
    return True


def verify_case_002_query_missing_docs() -> bool:
    print("\n[2/6] Testing Case 002: Missing MRI Diagnostic Report (Expected: QUERY)...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-002",
        patient_id="PAT-33109",
        patient_age=45,
        patient_gender="Female",
        diagnosis_codes=["M23.22"],
        procedure_codes=["29881"],
        estimated_cost=75000.0,
        length_of_stay=2,
    )

    rec = synthesize_decision(
        extracted=extracted,
        missing_docs=["diagnostic_report"],
        retrieved_clauses=[],
        code_validation=CodeValidationReport(all_codes_valid=True),
        fraud_report=FraudRiskReport(risk_score=0),
    )

    assert rec.outcome == AdjudicationOutcome.QUERY, f"Expected QUERY, got {rec.outcome}"
    assert any("diagnostic_report" in u.lower() for u in rec.unresolved_issues), "diagnostic_report not in unresolved"
    assert any("5.1" in c for c in rec.policy_citations), "Clause 5.1 not cited for missing documents"

    print(f"  Decision: {rec.outcome.value}")
    print(f"  Unresolved Items: {rec.unresolved_issues}")
    print(f"  Summary: {rec.summary}")
    print("  [PASS] Case 002 Query verified.")
    return True


def verify_case_003_reject_cosmetic() -> bool:
    print("\n[3/6] Testing Case 003: Cosmetic Rhinoplasty Exclusion (Expected: REJECT)...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-003",
        patient_id="PAT-77291",
        patient_age=28,
        patient_gender="Female",
        diagnosis_codes=["Z41.1"],
        diagnosis_descriptions=["Encounter for cosmetic surgery for aesthetic purposes"],
        procedure_codes=["30410"],
        procedure_descriptions=["Rhinoplasty, primary complete"],
        estimated_cost=120000.0,
        length_of_stay=1,
    )

    clauses = [
        PolicyClause(
            clause_id="CLAUSE 4.1",
            title="Cosmetic and Aesthetic Procedures Exclusion",
            text="Expenses incurred for cosmetic or aesthetic treatments or surgeries are permanently excluded.",
            document_source="careplus_comprehensive_policy.txt",
            page_reference=3,
            retrieval_score=0.92,
        )
    ]

    rec = synthesize_decision(
        extracted=extracted,
        missing_docs=[],
        retrieved_clauses=clauses,
        code_validation=CodeValidationReport(all_codes_valid=True),
        fraud_report=FraudRiskReport(risk_score=0),
    )

    assert rec.outcome == AdjudicationOutcome.REJECT, f"Expected REJECT, got {rec.outcome}"
    assert rec.confidence >= 0.95, f"Expected high rejection confidence, got {rec.confidence}"
    assert any("4.1" in c for c in rec.policy_citations), "Clause 4.1 not cited"

    print(f"  Decision: {rec.outcome.value} (Confidence: {rec.confidence:.2f})")
    print(f"  Cited Exclusion: {rec.policy_citations}")
    print(f"  Summary: {rec.summary}")
    print("  [PASS] Case 003 Rejection verified.")
    return True


def verify_case_004_query_duplicate() -> bool:
    print("\n[4/6] Testing Case 004: Duplicate Claim Submission (Expected: QUERY)...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-004",
        patient_id="PAT-98421",
        procedure_codes=["66984"],
        diagnosis_codes=["H25.011"],
        estimated_cost=36000.0,
    )

    fraud = FraudRiskReport(
        risk_score=50,
        is_duplicate=True,
        reasons=["Duplicate claim submission detected matching prior claim CLM-2026-001 within 48 hours."],
    )

    rec = synthesize_decision(
        extracted=extracted,
        missing_docs=[],
        retrieved_clauses=[],
        code_validation=CodeValidationReport(all_codes_valid=True),
        fraud_report=fraud,
    )

    assert rec.outcome == AdjudicationOutcome.QUERY, f"Expected QUERY, got {rec.outcome}"
    assert any("duplicate" in u.lower() for u in rec.unresolved_issues), "Duplicate warning not in unresolved issues"

    print(f"  Decision: {rec.outcome.value}")
    print(f"  Unresolved Items: {rec.unresolved_issues}")
    print(f"  Summary: {rec.summary}")
    print("  [PASS] Case 004 Duplicate Query verified.")
    return True


def verify_case_005_query_inflated_bill() -> bool:
    print("\n[5/6] Testing Case 005: Inflated Bill & Prolonged Stay (Expected: QUERY)...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-005",
        patient_id="PAT-61520",
        procedure_codes=["44970"],
        procedure_descriptions=["Laparoscopy, surgical, appendectomy"],
        diagnosis_codes=["K35.80"],
        estimated_cost=275000.0,
        length_of_stay=14,
    )

    fraud = FraudRiskReport(
        risk_score=60,
        billing_spike_detected=True,
        stay_anomaly_detected=True,
        reasons=[
            "Significant billing spike: Claimed $275,000 exceeds regional benchmark median ($45,000) by 511.1%.",
            "Length of stay anomaly: Claimed stay of 14 days is abnormal for laparoscopic appendectomy (expected 2 days).",
        ],
    )

    rec = synthesize_decision(
        extracted=extracted,
        missing_docs=[],
        retrieved_clauses=[],
        code_validation=CodeValidationReport(all_codes_valid=True),
        fraud_report=fraud,
    )

    assert rec.outcome == AdjudicationOutcome.QUERY, f"Expected QUERY, got {rec.outcome}"
    assert len(rec.unresolved_issues) >= 2, "Expected at least 2 audit queries"

    print(f"  Decision: {rec.outcome.value}")
    print(f"  Audit Queries: {rec.unresolved_issues}")
    print(f"  Summary: {rec.summary}")
    print("  [PASS] Case 005 Inflated Bill Query verified.")
    return True


def verify_langgraph_node_execution() -> bool:
    print("\n[6/6] Testing LangGraph Decision Synthesis Node Execution...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-001",
        patient_id="PAT-98421",
        patient_age=62,
        patient_gender="Male",
        diagnosis_codes=["H25.011"],
        procedure_codes=["66984"],
        estimated_cost=36000.0,
        length_of_stay=1,
    )

    clauses = [
        PolicyClause(
            clause_id="CLAUSE 2.1",
            title="Cataract Treatment Sub-Limit",
            text="Coverage for Cataract surgery is subject to a maximum sub-limit of $40,000 per eye.",
            document_source="careplus_comprehensive_policy.txt",
            page_reference=2,
            retrieval_score=0.85,
        )
    ]

    state: ClaimState = {
        "claim_id": "CLM-2026-001",
        "created_at": "2026-11-05T10:00:00Z",
        "extracted_claim": extracted,
        "missing_mandatory_documents": [],
        "retrieved_clauses": clauses,
        "code_validation": CodeValidationReport(all_codes_valid=True),
        "fraud_report": FraudRiskReport(risk_score=0),
        "audit_trail": ["[ExtractionNode] Claim extracted."],
        "errors": [],
    }

    result = decision_synthesis_node(state)
    assert "recommendation" in result, "recommendation missing from output"
    assert "audit_trail" in result, "audit_trail missing from output"

    rec: AdjudicationRecommendation = result["recommendation"]
    assert rec.outcome == AdjudicationOutcome.APPROVE, f"Expected APPROVE, got {rec.outcome}"

    audit_entry = result["audit_trail"][0]
    print(f"  Audit entry: {audit_entry}")
    assert "[DecisionSynthesisAgent]" in audit_entry, "Audit message format invalid"

    print("  [PASS] LangGraph Node integration verified.")
    return True


def main():
    print("=" * 60)
    print("PHASE 8: DECISION SYNTHESIS AGENT VERIFICATION")
    print("=" * 60)

    try:
        verify_case_001_approval()
        verify_case_002_query_missing_docs()
        verify_case_003_reject_cosmetic()
        verify_case_004_query_duplicate()
        verify_case_005_query_inflated_bill()
        verify_langgraph_node_execution()

        print("\n" + "=" * 60)
        print("[OK] Phase 8 Decision Synthesis Agent Verified across all cases!")
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
