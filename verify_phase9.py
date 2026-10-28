"""Verification Script for Phase 9: LangGraph Supervisor & Workflow Graph Assembly.

Tests:
1. StateGraph topology and compilation with SQLite checkpointer.
2. End-to-end execution of Case 001 (Cataract pre-auth -> Parallel Analysis -> APPROVAL).
3. Conditional routing for Case 002 (Missing mandatory docs -> Fast-path -> QUERY).
4. End-to-end execution of Case 003 (Cosmetic Rhinoplasty -> Parallel Analysis -> REJECTION).
5. State checkpoint recovery and inspection via thread_id from SQLite checkpoints.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout encoding and line buffering for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.graph import build_claim_workflow, create_claim_graph, get_sqlite_checkpointer
from app.state import (
    AdjudicationOutcome,
    ClaimState,
    DocumentMeta,
    DocumentType,
)
from app.tools.database import load_reference_tables


def verify_graph_topology() -> bool:
    print("[1/5] Testing LangGraph Workflow Construction & Topology...")
    workflow = build_claim_workflow()

    expected_nodes = {
        "extraction",
        "redaction",
        "fan_out_barrier",
        "policy_rag",
        "code_validation",
        "fraud_detection",
        "barrier_join",
        "decision_synthesis",
    }
    actual_nodes = set(workflow.nodes.keys())
    missing_nodes = expected_nodes - actual_nodes
    assert not missing_nodes, f"Missing nodes in StateGraph: {missing_nodes}"

    print(f"  All {len(expected_nodes)} nodes verified in StateGraph topology:")
    for n in sorted(actual_nodes):
        print(f"    - Node: '{n}'")

    print("  [PASS] Workflow topology verified.")
    return True


def verify_case_001_full_pipeline() -> bool:
    print("\n[2/5] Testing Case 001 End-to-End Execution (Cataract -> APPROVE)...")
    load_reference_tables()

    # Create documents for complete submission
    doc_form = DocumentMeta(
        file_id="DOC-001-FORM",
        filename="CLM-2026-001_claim_form.txt",
        doc_type=DocumentType.CLAIM_FORM,
        raw_text="""
PRE-AUTHORIZATION REQUEST FORM - HEALTH INSURANCE
Hospital: Apex Specialty Eye Institute
Full Name: Rohan Deshmukh
Age: 62 | Gender: Male
Insurance Policy No: CP-POL-55410
Clinical Diagnosis: Senile nuclear cataract, right eye
ICD-10 Diagnosis Code: H25.011
Proposed Procedure: Extracapsular cataract removal with IOL
CPT / Procedure Code: 66984
Date of Proposed Admission: 2026-11-05
Date of Discharge: 2026-11-05
Total Estimated Cost: $36,000
        """,
    )

    doc_bill = DocumentMeta(
        file_id="DOC-001-BILL",
        filename="CLM-2026-001_hospital_bill.txt",
        doc_type=DocumentType.HOSPITAL_BILL,
        raw_text="Itemized estimate: Surgeon $24,000, Lens $8,000, Pharmacy $4,000. Total Cost: $36,000.",
    )

    doc_report = DocumentMeta(
        file_id="DOC-001-REPORT",
        filename="CLM-2026-001_diagnostic_report.txt",
        doc_type=DocumentType.DIAGNOSTIC_REPORT,
        raw_text="Ophthalmology Biometry & Slit Lamp Report: Right eye senile nuclear sclerosis Grade 3+. Vision 6/36.",
    )

    initial_state: ClaimState = {
        "claim_id": "CLM-2026-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_documents": [doc_form, doc_bill, doc_report],
        "missing_mandatory_documents": [],
        "pii_redacted": False,
        "retrieved_clauses": [],
        "audit_trail": ["[WorkflowInit] Initialized Case 001 pre-authorization."],
        "errors": [],
    }

    graph = create_claim_graph()
    thread_config = {"configurable": {"thread_id": "THREAD-CLM-2026-001"}}

    final_state = graph.invoke(initial_state, config=thread_config)

    assert final_state["pii_redacted"], "PII was not redacted"
    assert len(final_state["missing_mandatory_documents"]) == 0, "Expected 0 missing documents"
    assert len(final_state["retrieved_clauses"]) > 0, "No policy clauses retrieved"
    assert final_state["code_validation"].all_codes_valid, "Code validation flagged errors"
    assert final_state["fraud_report"].risk_score == 0, "Unexpected fraud risk"

    rec = final_state["recommendation"]
    assert rec.outcome == AdjudicationOutcome.APPROVE, f"Expected APPROVE, got {rec.outcome}"

    print(f"  Decision Outcome: {rec.outcome.value} (Confidence: {rec.confidence:.2f})")
    print(f"  Citations: {rec.policy_citations}")
    print(f"  Audit trail length: {len(final_state['audit_trail'])} entries")
    print("  [PASS] Case 001 Full Pipeline execution verified.")
    return True


def verify_case_002_conditional_fast_path() -> bool:
    print("\n[3/5] Testing Case 002 Conditional Fast-Path (Missing Docs -> QUERY)...")

    # Missing diagnostic_report
    doc_form = DocumentMeta(
        file_id="DOC-002-FORM",
        filename="CLM-2026-002_claim_form.txt",
        doc_type=DocumentType.CLAIM_FORM,
        raw_text="""
Hospital: Metro Health Super Specialty Hospital
Full Name: Meera Venkataraman
Age: 45 | Gender: Female
Insurance Policy No: CP-POL-88210
Clinical Diagnosis: Derangement of meniscus right knee
ICD-10 Diagnosis Code: M23.22
Proposed Procedure: Knee Arthroscopy Meniscectomy
CPT / Procedure Code: 29881
Total Estimated Cost: $75,000
        """,
    )

    doc_bill = DocumentMeta(
        file_id="DOC-002-BILL",
        filename="CLM-2026-002_hospital_bill.txt",
        doc_type=DocumentType.HOSPITAL_BILL,
        raw_text="Itemized estimate: OT $45,000, Implants $20,000, Room $10,000. Total Cost: $75,000.",
    )

    initial_state: ClaimState = {
        "claim_id": "CLM-2026-002",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_documents": [doc_form, doc_bill],  # Missing diagnostic report
        "missing_mandatory_documents": [],
        "pii_redacted": False,
        "retrieved_clauses": [],
        "audit_trail": ["[WorkflowInit] Initialized Case 002 pre-authorization."],
        "errors": [],
    }

    graph = create_claim_graph()
    thread_config = {"configurable": {"thread_id": "THREAD-CLM-2026-002"}}

    final_state = graph.invoke(initial_state, config=thread_config)

    assert len(final_state["missing_mandatory_documents"]) > 0, "Missing documents not detected"
    rec = final_state["recommendation"]
    assert rec.outcome == AdjudicationOutcome.QUERY, f"Expected QUERY, got {rec.outcome}"
    assert any("diagnostic" in u.lower() for u in rec.unresolved_issues), "diagnostic_report missing from queries"

    print(f"  Decision Outcome: {rec.outcome.value}")
    print(f"  Unresolved Items: {rec.unresolved_issues}")
    print("  [PASS] Case 002 Conditional Fast-Path verified.")
    return True


def verify_case_003_exclusion_rejection() -> bool:
    print("\n[4/5] Testing Case 003 Rejection (Cosmetic Rhinoplasty -> REJECT)...")

    doc_form = DocumentMeta(
        file_id="DOC-003-FORM",
        filename="CLM-2026-003_claim_form.txt",
        doc_type=DocumentType.CLAIM_FORM,
        raw_text="""
Hospital: Aura Aesthetic Plastic Surgery Center
Full Name: Kavita Sharma
Age: 28 | Gender: Female
Insurance Policy No: CP-POL-44120
Clinical Diagnosis: Encounter for cosmetic surgery for aesthetic purposes
ICD-10 Diagnosis Code: Z41.1
Proposed Procedure: Rhinoplasty primary complete
CPT / Procedure Code: 30410
Total Estimated Cost: $120,000
        """,
    )

    doc_bill = DocumentMeta(
        file_id="DOC-003-BILL",
        filename="CLM-2026-003_hospital_bill.txt",
        doc_type=DocumentType.HOSPITAL_BILL,
        raw_text="Hospital bill: $120,000 total.",
    )

    doc_report = DocumentMeta(
        file_id="DOC-003-REPORT",
        filename="CLM-2026-003_report.txt",
        doc_type=DocumentType.DIAGNOSTIC_REPORT,
        raw_text="Aesthetic evaluation report: Patient desires dorsal hump reduction for cosmetic enhancement.",
    )

    initial_state: ClaimState = {
        "claim_id": "CLM-2026-003",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_documents": [doc_form, doc_bill, doc_report],
        "missing_mandatory_documents": [],
        "pii_redacted": False,
        "retrieved_clauses": [],
        "audit_trail": ["[WorkflowInit] Initialized Case 003 pre-authorization."],
        "errors": [],
    }

    graph = create_claim_graph()
    thread_config = {"configurable": {"thread_id": "THREAD-CLM-2026-003"}}

    final_state = graph.invoke(initial_state, config=thread_config)

    rec = final_state["recommendation"]
    assert rec.outcome == AdjudicationOutcome.REJECT, f"Expected REJECT, got {rec.outcome}"
    assert any("4.1" in c for c in rec.policy_citations), "Clause 4.1 not cited for cosmetic rejection"

    print(f"  Decision Outcome: {rec.outcome.value} (Confidence: {rec.confidence:.2f})")
    print(f"  Citations: {rec.policy_citations}")
    print("  [PASS] Case 003 Exclusion Rejection verified.")
    return True


def verify_checkpoint_persistence() -> bool:
    print("\n[5/5] Testing SQLite Checkpoint Recovery & Thread Inspection...")

    graph = create_claim_graph()
    thread_config = {"configurable": {"thread_id": "THREAD-CLM-2026-001"}}

    # Retrieve saved state snapshot from SQLite
    saved_state = graph.get_state(thread_config)
    assert saved_state is not None, "Failed to retrieve state snapshot from SQLite"
    assert saved_state.values.get("claim_id") == "CLM-2026-001", "Restored claim_id mismatch"
    assert saved_state.values.get("recommendation") is not None, "Saved state missing recommendation"

    print(f"  Retrieved thread: {thread_config['configurable']['thread_id']}")
    print(f"  Stored Claim ID: {saved_state.values.get('claim_id')}")
    print(f"  Stored Outcome: {saved_state.values.get('recommendation').outcome.value}")
    print("  [PASS] SQLite Checkpoint persistence verified.")
    return True


def main():
    print("=" * 60)
    print("PHASE 9: LANGGRAPH SUPERVISOR & WORKFLOW GRAPH VERIFICATION")
    print("=" * 60)

    try:
        verify_graph_topology()
        verify_case_001_full_pipeline()
        verify_case_002_conditional_fast_path()
        verify_case_003_exclusion_rejection()
        verify_checkpoint_persistence()

        print("\n" + "=" * 60)
        print("[OK] Phase 9 LangGraph Supervisor & Workflow Graph Verified!")
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
