"""Verification Script for Phase 10: Human Review Node & Checkpointing (interrupt).

Tests:
1. Execution pauses at `human_review` node via LangGraph `interrupt()`.
2. State snapshot persistence in SQLite while waiting for human input.
3. Human Adjudicator confirmation of AI recommendation (Approve).
4. Human Adjudicator Override with mandatory clinical rationale (Override).
5. Audit trail records adjudicator identity, timestamp, and notes.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout encoding and line buffering for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.agents.human_review import resume_human_review
from app.graph import create_claim_graph
from app.state import (
    AdjudicationOutcome,
    ClaimState,
    DocumentMeta,
    DocumentType,
    HumanReviewAction,
)
from app.tools.database import load_reference_tables


def verify_pause_on_interrupt() -> bool:
    print("[1/4] Testing Graph Pause at Human Review Node...")
    load_reference_tables()

    doc_form = DocumentMeta(
        file_id="DOC-HITL-001",
        filename="CLM-HITL-001_claim_form.txt",
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
Total Estimated Cost: $36,000
        """,
    )
    doc_bill = DocumentMeta(
        file_id="DOC-HITL-BILL",
        filename="bill.txt",
        doc_type=DocumentType.HOSPITAL_BILL,
        raw_text="Cost: $36,000",
    )
    doc_report = DocumentMeta(
        file_id="DOC-HITL-REPORT",
        filename="report.txt",
        doc_type=DocumentType.DIAGNOSTIC_REPORT,
        raw_text="Ophthalmology Biometry Report: Cataract confirmed.",
    )

    initial_state: ClaimState = {
        "claim_id": "CLM-HITL-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_documents": [doc_form, doc_bill, doc_report],
        "missing_mandatory_documents": [],
        "pii_redacted": False,
        "retrieved_clauses": [],
        "audit_trail": ["[WorkflowInit] Initialized HITL test claim."],
        "errors": [],
    }

    graph = create_claim_graph(enable_human_review=True)
    thread_config = {"configurable": {"thread_id": "THREAD-HITL-001"}}

    # Invoke graph - should pause at human_review
    graph.invoke(initial_state, config=thread_config)

    state_snapshot = graph.get_state(thread_config)
    assert len(state_snapshot.next) > 0, "Graph did not pause: next nodes list is empty"
    assert "human_review" in state_snapshot.next, f"Expected next node to be 'human_review', got {state_snapshot.next}"

    # Verify interrupt payload
    tasks = state_snapshot.tasks
    assert len(tasks) > 0, "No interrupted task found in checkpoint"
    interrupts = tasks[0].interrupts
    assert len(interrupts) > 0, "No interrupt payload captured"
    payload = interrupts[0].value
    print(f"  Captured Interrupt Prompt: {payload.get('prompt')}")
    print(f"  AI Pre-Review Outcome: {payload.get('recommendation', {}).get('outcome')}")

    print("  [PASS] Execution successfully paused at human_review node.")
    return True


def verify_adjudicator_approval() -> bool:
    print("\n[2/4] Testing Adjudicator Confirmation & Resumption...")

    graph = create_claim_graph(enable_human_review=True)
    thread_id = "THREAD-HITL-001"

    resumed_state = resume_human_review(
        graph=graph,
        thread_id=thread_id,
        reviewer_id="DR-MEHTA-SR-ADJUDICATOR",
        action="Approve",
        reviewer_notes="Medical necessity verified against biometry report. Approved under Clause 2.1 sub-limit.",
    )

    hr = resumed_state.get("human_review")
    assert hr is not None, "human_review missing from resumed state"
    assert hr.action == HumanReviewAction.APPROVE, f"Expected action APPROVE, got {hr.action}"
    assert hr.final_outcome == AdjudicationOutcome.APPROVE, f"Expected final outcome APPROVE, got {hr.final_outcome}"
    assert hr.reviewer_id == "DR-MEHTA-SR-ADJUDICATOR"

    # Verify graph has now completed
    final_snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    assert len(final_snapshot.next) == 0, f"Graph did not terminate; remaining next nodes: {final_snapshot.next}"

    print(f"  Adjudicator: {hr.reviewer_id}")
    print(f"  Action: {hr.action.value} -> Final Outcome: {hr.final_outcome.value}")
    print(f"  Reviewer Notes: {hr.reviewer_notes}")
    print("  [PASS] Adjudicator approval resumption verified.")
    return True


def verify_adjudicator_override() -> bool:
    print("\n[3/4] Testing Adjudicator Override (Reject -> Approve Exception)...")

    # Claim for cosmetic surgery (normally rejected under Clause 4.1)
    doc_form = DocumentMeta(
        file_id="DOC-OVERRIDE-FORM",
        filename="claim_form.txt",
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
    doc_bill = DocumentMeta(file_id="DOC-B", filename="b.txt", doc_type=DocumentType.HOSPITAL_BILL, raw_text="Cost: $120,000")
    doc_report = DocumentMeta(file_id="DOC-R", filename="r.txt", doc_type=DocumentType.DIAGNOSTIC_REPORT, raw_text="Surgery report")

    initial_state: ClaimState = {
        "claim_id": "CLM-OVERRIDE-002",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_documents": [doc_form, doc_bill, doc_report],
        "missing_mandatory_documents": [],
        "pii_redacted": False,
        "retrieved_clauses": [],
        "audit_trail": ["[WorkflowInit] Initialized Override test claim."],
        "errors": [],
    }

    graph = create_claim_graph(enable_human_review=True)
    thread_id = "THREAD-HITL-OVERRIDE-002"
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Run to pause
    graph.invoke(initial_state, config=thread_config)
    snapshot = graph.get_state(thread_config)
    ai_outcome = snapshot.values.get("recommendation").outcome
    print(f"  AI Recommended Outcome: {ai_outcome.value}")
    assert ai_outcome == AdjudicationOutcome.REJECT, f"Expected AI to reject Case 003, got {ai_outcome}"

    # Human Adjudicator Overrides AI Reject -> Approve with clinical exception
    override_reason = (
        "Adjudicator Override Granted: Medical review confirms procedure is reconstructive "
        "following traumatic nasal bone fracture sustained in road accident, not cosmetic."
    )

    resumed_state = resume_human_review(
        graph=graph,
        thread_id=thread_id,
        reviewer_id="CHIEF-MEDICAL-OFFICER-DR-KAPOOR",
        action="Override",
        final_outcome="Approve",
        override_reason=override_reason,
        reviewer_notes="Approved under exceptional reconstructive trauma coverage clause.",
    )

    hr = resumed_state.get("human_review")
    assert hr.action == HumanReviewAction.OVERRIDE, f"Expected OVERRIDE, got {hr.action}"
    assert hr.final_outcome == AdjudicationOutcome.APPROVE, f"Expected final APPROVE, got {hr.final_outcome}"
    assert hr.override_reason == override_reason

    print(f"  Override Adjudicator: {hr.reviewer_id}")
    print(f"  Action: {hr.action.value} -> Final Outcome: {hr.final_outcome.value}")
    print(f"  Override Rationale: {hr.override_reason[:80]}...")
    print("  [PASS] Adjudicator override mechanism verified.")
    return True


def verify_audit_trail_logging() -> bool:
    print("\n[4/4] Testing Audit Trail Verification for Human Review...")

    graph = create_claim_graph(enable_human_review=True)
    thread_id = "THREAD-HITL-OVERRIDE-002"
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})

    audit_trail = snapshot.values.get("audit_trail", [])
    human_entries = [entry for entry in audit_trail if "[HumanReviewer]" in entry]

    assert len(human_entries) > 0, "No [HumanReviewer] audit trail entry found"
    print(f"  Captured Human Audit Entry: {human_entries[-1]}")
    assert "CHIEF-MEDICAL-OFFICER-DR-KAPOOR" in human_entries[-1], "Reviewer ID missing from audit log"

    print("  [PASS] Human Review audit trail verified.")
    return True


def main():
    print("=" * 60)
    print("PHASE 10: HUMAN REVIEW NODE & CHECKPOINTING VERIFICATION")
    print("=" * 60)

    try:
        verify_pause_on_interrupt()
        verify_adjudicator_approval()
        verify_adjudicator_override()
        verify_audit_trail_logging()

        print("\n" + "=" * 60)
        print("[OK] Phase 10 Human Review Node & Checkpointing Verified!")
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
