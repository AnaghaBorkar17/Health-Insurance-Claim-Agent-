"""Verification Script for Phase 11: Communication Agent.

Tests:
1. Drafting of formal Pre-Authorization Approval Letter (Case 001).
2. Drafting of formal Query / Missing Documents Request Letter (Case 002).
3. Drafting of formal Denial Letter citing policy exclusions and appeal rights (Case 003).
4. Drafting an Approval Letter following a Human Adjudicator Override.
5. End-to-End LangGraph workflow execution producing final CommunicationDraft.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout encoding and line buffering for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.agents.communication import (
    communication_node,
    draft_approval_letter,
    draft_denial_letter,
    draft_query_letter,
    generate_communication,
)
from app.agents.human_review import resume_human_review
from app.graph import create_claim_graph
from app.state import (
    AdjudicationOutcome,
    AdjudicationRecommendation,
    ClaimState,
    CommunicationDraft,
    DocumentMeta,
    DocumentType,
    ExtractedClaim,
    HumanReview,
    HumanReviewAction,
)
from app.tools.database import load_reference_tables


def verify_approval_letter_drafting() -> bool:
    print("[1/5] Testing Approval Letter Generation (Case 001)...")

    draft = draft_approval_letter(
        claim_id="CLM-2026-001",
        patient_name="Rohan Deshmukh",
        policy_no="CP-POL-55410",
        hospital="Apex Specialty Eye Institute",
        procedure_desc="Extracapsular cataract removal with IOL (CPT 66984)",
        amount=36000.0,
        citations=["CLAUSE 2.1: Cataract Treatment Sub-Limit ($40,000 max)"],
    )

    assert draft.letter_type == "Approval Letter", f"Expected Approval Letter, got {draft.letter_type}"
    assert "CLM-2026-001" in draft.subject, "Claim ID missing from subject"
    assert "$36,000.00" in draft.body, "Approved amount missing from body"
    assert "Apex Specialty Eye Institute" in draft.body, "Hospital missing from body"
    assert "CLAUSE 2.1" in draft.body, "Clause 2.1 missing from body"

    print(f"  Letter Type: {draft.letter_type}")
    print(f"  Subject: {draft.subject}")
    print(f"  Recipient: {draft.recipient}")
    print("  [PASS] Approval Letter generation verified.")
    return True


def verify_query_letter_drafting() -> bool:
    print("\n[2/5] Testing Query / Additional Records Letter (Case 002)...")

    draft = draft_query_letter(
        claim_id="CLM-2026-002",
        patient_name="Meera Venkataraman",
        policy_no="CP-POL-88210",
        hospital="Metro Health Super Specialty Hospital",
        missing_docs=["Diagnostic / Radiology Investigation Report (diagnostic_report)"],
        unresolved_issues=["Missing mandatory diagnostic MRI report to verify meniscal tear"],
        citations=["CLAUSE 5.1: Mandatory Pre-Authorization Documentation"],
    )

    assert draft.letter_type == "Query Letter", f"Expected Query Letter, got {draft.letter_type}"
    assert "72 hours" in draft.body, "72-hour submission deadline missing from body"
    assert len(draft.missing_items_requested) > 0, "Missing items list empty"
    assert "diagnostic_report" in draft.body.lower(), "Diagnostic report query missing from body"

    print(f"  Letter Type: {draft.letter_type}")
    print(f"  Subject: {draft.subject}")
    print(f"  Requested Items Count: {len(draft.missing_items_requested)}")
    print("  [PASS] Query Letter generation verified.")
    return True


def verify_denial_letter_drafting() -> bool:
    print("\n[3/5] Testing Denial Letter with Appeal Rights (Case 003)...")

    draft = draft_denial_letter(
        claim_id="CLM-2026-003",
        patient_name="Kavita Sharma",
        policy_no="CP-POL-44120",
        hospital="Aura Aesthetic Plastic Surgery Center",
        procedure_desc="Rhinoplasty primary complete (CPT 30410)",
        reasons=["Cosmetic and aesthetic surgeries are permanently excluded under Policy Clause 4.1."],
        citations=["CLAUSE 4.1: Cosmetic and Aesthetic Procedures Exclusion"],
    )

    assert draft.letter_type == "Denial Letter", f"Expected Denial Letter, got {draft.letter_type}"
    assert "DENIAL" in draft.subject, "DENIAL missing from subject"
    assert "Ombudsman" in draft.body, "Insurance Ombudsman appeal rights missing from body"
    assert "CLAUSE 4.1" in draft.body, "Clause 4.1 missing from body"

    print(f"  Letter Type: {draft.letter_type}")
    print(f"  Subject: {draft.subject}")
    print("  [PASS] Denial Letter generation verified.")
    return True


def verify_override_letter_generation() -> bool:
    print("\n[4/5] Testing Letter Generation Following Adjudicator Override...")

    extracted = ExtractedClaim(
        claim_id="CLM-OVERRIDE-002",
        patient_id="PAT-77291",
        patient_name_masked="Kavita Sharma",
        policy_number="CP-POL-44120",
        hospital_name="Aura Aesthetic Plastic Surgery Center",
        procedure_codes=["30410"],
        procedure_descriptions=["Rhinoplasty reconstructive"],
        estimated_cost=120000.0,
    )

    rec = AdjudicationRecommendation(
        outcome=AdjudicationOutcome.REJECT,
        confidence=0.98,
        summary="AI Recommended Denial: Cosmetic exclusion.",
        reasons=["Clause 4.1 Cosmetic exclusion"],
        policy_citations=["CLAUSE 4.1"],
    )

    # Human Adjudicator overrides to Approve
    hr = HumanReview(
        reviewer_id="CHIEF-MEDICAL-OFFICER-DR-KAPOOR",
        action=HumanReviewAction.OVERRIDE,
        final_outcome=AdjudicationOutcome.APPROVE,
        override_reason="Granted clinical exception: Functional nasal trauma reconstruction.",
        reviewer_notes="Approved following verification of traumatic fracture history.",
    )

    state: ClaimState = {
        "claim_id": "CLM-OVERRIDE-002",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extracted_claim": extracted,
        "recommendation": rec,
        "human_review": hr,
        "missing_mandatory_documents": [],
        "audit_trail": [],
    }

    draft = generate_communication(state)

    # Communication agent must respect the human adjudicator's override!
    assert draft.letter_type == "Approval Letter", f"Expected Approval Letter following override, got {draft.letter_type}"
    assert "APPROVAL" in draft.subject, "Subject did not reflect approved status"
    assert "$120,000.00" in draft.body, "Approved amount missing from body"

    print(f"  AI Recommended: REJECT -> Human Overrode: APPROVE")
    print(f"  Resulting Letter Type: {draft.letter_type}")
    print(f"  Subject: {draft.subject}")
    print("  [PASS] Override-aware letter generation verified.")
    return True


def verify_end_to_end_graph_communication() -> bool:
    print("\n[5/5] Testing End-to-End Pipeline Execution through Communication Node...")
    load_reference_tables()

    doc_form = DocumentMeta(
        file_id="DOC-COMM-001",
        filename="CLM-2026-001_claim_form.txt",
        doc_type=DocumentType.CLAIM_FORM,
        raw_text="""
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
    doc_bill = DocumentMeta(file_id="DOC-B", filename="b.txt", doc_type=DocumentType.HOSPITAL_BILL, raw_text="Cost: $36,000")
    doc_report = DocumentMeta(file_id="DOC-R", filename="r.txt", doc_type=DocumentType.DIAGNOSTIC_REPORT, raw_text="Biometry cataract confirmed")

    initial_state: ClaimState = {
        "claim_id": "CLM-COMM-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_documents": [doc_form, doc_bill, doc_report],
        "missing_mandatory_documents": [],
        "pii_redacted": False,
        "retrieved_clauses": [],
        "audit_trail": ["[WorkflowInit] Initialized end-to-end communication test claim."],
        "errors": [],
    }

    graph = create_claim_graph(enable_human_review=True)
    thread_id = "THREAD-COMM-001"
    thread_config = {"configurable": {"thread_id": thread_id}}

    # 1. Run until human_review pause
    graph.invoke(initial_state, config=thread_config)
    snapshot = graph.get_state(thread_config)
    assert "human_review" in snapshot.next, f"Expected pause at human_review, got {snapshot.next}"

    # 2. Adjudicator authorizes approval
    resumed_state = resume_human_review(
        graph=graph,
        thread_id=thread_id,
        reviewer_id="DR-SENIOR-ADJUDICATOR",
        action="Approve",
        reviewer_notes="Cashless pre-auth approved within sub-limit.",
    )

    # 3. Verify communication_draft is generated
    draft: CommunicationDraft = resumed_state.get("communication_draft")
    assert draft is not None, "communication_draft missing from final graph state"
    assert draft.letter_type == "Approval Letter", f"Expected Approval Letter, got {draft.letter_type}"

    # 4. Verify audit entry
    audit_trail = resumed_state.get("audit_trail", [])
    comm_entries = [entry for entry in audit_trail if "[CommunicationAgent]" in entry]
    assert len(comm_entries) > 0, "No [CommunicationAgent] audit entry found"

    print(f"  Final State Letter Type: {draft.letter_type}")
    print(f"  Recipient: {draft.recipient}")
    print(f"  Audit entry: {comm_entries[-1]}")
    print("  [PASS] Full pipeline communication generation verified.")
    return True


def main():
    print("=" * 60)
    print("PHASE 11: COMMUNICATION AGENT VERIFICATION")
    print("=" * 60)

    try:
        verify_approval_letter_drafting()
        verify_query_letter_drafting()
        verify_denial_letter_drafting()
        verify_override_letter_generation()
        verify_end_to_end_graph_communication()

        print("\n" + "=" * 60)
        print("[OK] Phase 11 Communication Agent Verified!")
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
