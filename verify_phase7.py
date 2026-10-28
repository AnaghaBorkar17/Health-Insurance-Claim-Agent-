"""Verification Script for Phase 7: Fraud & Anomaly Detection Subgraph.

Tests:
1. Duplicate pre-authorization detection via SQLite claims history (Case 004).
2. Billing spike calculation vs regional CPT cost benchmarks (Case 005).
3. Length-of-stay anomaly detection vs clinical baselines (Case 005).
4. Clean claim baseline verification (Case 001: 0 risk score).
5. Compiled LangGraph Subgraph execution end-to-end.
6. LangGraph node integration within shared ClaimState.
"""

import sys
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.agents.fraud import (
    FraudSubgraphState,
    create_fraud_subgraph,
    fraud_detection_node,
    fraud_subgraph,
)
from app.agents.fraud.nodes import (
    billing_spike_node,
    duplicate_check_node,
    length_of_stay_node,
    risk_aggregator_node,
)
from app.state import ClaimState, ExtractedClaim, FraudRiskReport
from app.tools.database import load_reference_tables


def verify_duplicate_detection() -> bool:
    print("[1/6] Testing Duplicate Pre-Authorization Detection...")
    load_reference_tables()

    # Case 004 matches Case 001 (PAT-98421, procedure 66984)
    duplicate_state: FraudSubgraphState = {
        "claim_id": "CLM-2026-004",
        "patient_id": "PAT-98421",
        "admission_date": "2026-11-05",
        "procedure_codes": ["66984"],
        "diagnosis_codes": ["H25.011"],
        "estimated_cost": 36000.0,
        "length_of_stay": 1,
        "reasons": [],
        "risk_points": 0,
    }

    result = duplicate_check_node(duplicate_state)
    assert result["is_duplicate"], "Failed to detect duplicate claim for PAT-98421"
    assert result["risk_points"] >= 50, f"Expected at least 50 risk points, got {result['risk_points']}"
    assert len(result["reasons"]) > 0, "No duplicate explanation reason provided"
    print(f"  Duplicate Alert: {result['reasons'][0]}")
    print("  [PASS] Duplicate detection verified.")
    return True


def verify_billing_spike() -> bool:
    print("\n[2/6] Testing Billing Spike Detection Against Benchmarks...")
    # Case 005: Appendectomy 44970 ($275,000 claimed vs $45,000 median benchmark)
    inflated_state: FraudSubgraphState = {
        "claim_id": "CLM-2026-005",
        "patient_id": "PAT-61520",
        "procedure_codes": ["44970"],
        "estimated_cost": 275000.0,
        "reasons": [],
        "risk_points": 0,
    }

    result = billing_spike_node(inflated_state)
    assert result["billing_spike_detected"], "Failed to detect billing spike for $275,000 appendectomy"
    assert result["risk_points"] >= 35, f"Expected at least 35 risk points, got {result['risk_points']}"
    print(f"  Spike Alert: {result['reasons'][0]}")
    print("  [PASS] Billing spike detection verified.")
    return True


def verify_length_of_stay_anomaly() -> bool:
    print("\n[3/6] Testing Length-of-Stay Anomaly Detection...")
    # Case 005: 14 days claimed vs 2 days expected for 44970
    stay_state: FraudSubgraphState = {
        "claim_id": "CLM-2026-005",
        "procedure_codes": ["44970"],
        "length_of_stay": 14,
        "reasons": [],
        "risk_points": 0,
    }

    result = length_of_stay_node(stay_state)
    assert result["stay_anomaly_detected"], "Failed to detect stay anomaly (14 days for appendectomy)"
    assert result["risk_points"] >= 25, f"Expected at least 25 risk points, got {result['risk_points']}"
    print(f"  Stay Anomaly Alert: {result['reasons'][0]}")
    print("  [PASS] Length of stay anomaly detection verified.")
    return True


def verify_clean_claim() -> bool:
    print("\n[4/6] Testing Clean Baseline Claim (Case 001)...")
    clean_state: FraudSubgraphState = {
        "claim_id": "CLM-2026-001-NEW",
        "patient_id": "PAT-NEW-PATIENT",
        "admission_date": "2026-11-20",
        "procedure_codes": ["66984"],
        "estimated_cost": 35000.0,  # Matches median benchmark
        "length_of_stay": 1,
        "reasons": [],
        "risk_points": 0,
        "is_duplicate": False,
        "billing_spike_detected": False,
        "stay_anomaly_detected": False,
    }

    # Run through full subgraph
    result = fraud_subgraph.invoke(clean_state)
    report: FraudRiskReport = result["fraud_report"]

    assert report.risk_score == 0, f"Expected 0 risk score for clean claim, got {report.risk_score}"
    assert not report.is_duplicate, "Expected is_duplicate=False"
    assert not report.billing_spike_detected, "Expected billing_spike_detected=False"
    assert not report.stay_anomaly_detected, "Expected stay_anomaly_detected=False"
    print(f"  Clean Claim Result: Risk Score {report.risk_score}/100. Status: {report.reasons[0]}")
    print("  [PASS] Clean baseline claim verified.")
    return True


def verify_subgraph_execution() -> bool:
    print("\n[5/6] Testing Compiled Fraud Subgraph on High-Risk Case 005...")
    case_005_state: FraudSubgraphState = {
        "claim_id": "CLM-2026-005",
        "patient_id": "PAT-61520",
        "admission_date": "2026-11-20",
        "discharge_date": "2026-12-04",
        "procedure_codes": ["44970"],
        "diagnosis_codes": ["K35.80"],
        "estimated_cost": 275000.0,
        "length_of_stay": 14,
        "is_duplicate": False,
        "billing_spike_detected": False,
        "stay_anomaly_detected": False,
        "risk_points": 0,
        "reasons": [],
    }

    result = fraud_subgraph.invoke(case_005_state)
    report: FraudRiskReport = result["fraud_report"]

    assert report.risk_score >= 60, f"Expected high risk score (>=60), got {report.risk_score}"
    assert report.billing_spike_detected, "Expected billing_spike_detected=True"
    assert report.stay_anomaly_detected, "Expected stay_anomaly_detected=True"
    print(f"  Compiled Subgraph Output: Score={report.risk_score}/100, Explanations={len(report.reasons)}")
    for r in report.reasons:
        print(f"    - {r}")
    print("  [PASS] Compiled Subgraph execution verified.")
    return True


def verify_langgraph_node_wrapper() -> bool:
    print("\n[6/6] Testing LangGraph Main Workflow Node Wrapper...")

    extracted = ExtractedClaim(
        claim_id="CLM-2026-004",
        patient_id="PAT-98421",
        patient_age=62,
        patient_gender="Male",
        admission_date="2026-11-05",
        diagnosis_codes=["H25.011"],
        procedure_codes=["66984"],
        estimated_cost=36000.0,
        length_of_stay=1,
    )

    state: ClaimState = {
        "claim_id": "CLM-2026-004",
        "created_at": "2026-11-05T10:00:00Z",
        "extracted_claim": extracted,
        "audit_trail": ["[ExtractionNode] Claim extracted."],
        "errors": [],
    }

    result = fraud_detection_node(state)
    assert "fraud_report" in result, "fraud_report missing from output"
    assert "audit_trail" in result, "audit_trail missing from output"

    report = result["fraud_report"]
    assert report.is_duplicate, "Expected is_duplicate=True for Case 004"
    assert report.risk_score >= 50, f"Expected risk_score >= 50, got {report.risk_score}"

    audit_entry = result["audit_trail"][0]
    print(f"  Audit entry: {audit_entry}")
    assert "[FraudRiskAgent]" in audit_entry, "Audit message format invalid"

    print("  [PASS] LangGraph Node Wrapper integration verified.")
    return True


def main():
    print("=" * 60)
    print("PHASE 7: FRAUD & ANOMALY DETECTION SUBGRAPH VERIFICATION")
    print("=" * 60)

    try:
        verify_duplicate_detection()
        verify_billing_spike()
        verify_length_of_stay_anomaly()
        verify_clean_claim()
        verify_subgraph_execution()
        verify_langgraph_node_wrapper()

        print("\n" + "=" * 60)
        print("[OK] Phase 7 Fraud & Anomaly Detection Subgraph Verified!")
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
