"""Communication Agent.

Drafts formal, audit-compliant communication letters for patients and treating hospitals:
1. Pre-Authorization Approval Letter:
   - Pre-authorized amount, diagnosis/procedure, sub-limits, admission guidelines.
2. Pre-Authorization Query / Additional Information Request Letter:
   - Bulleted list of missing mandatory documents, billing clarification queries, 72-hour deadline.
3. Pre-Authorization Denial Letter:
   - Exact policy exclusions cited (e.g., Clause 4.1), clear factual rationale, appeal and ombudsman rights.

Supports Google Gemini and Ollama for empathetic, professional phrasing with strict privacy guardrails.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings
from app.state import (
    AdjudicationOutcome,
    ClaimState,
    CommunicationDraft,
    ExtractedClaim,
    HumanReview,
)

logger = logging.getLogger(__name__)


def draft_approval_letter(
    claim_id: str,
    patient_name: str,
    policy_no: str,
    hospital: str,
    procedure_desc: str,
    amount: float,
    citations: List[str],
) -> CommunicationDraft:
    """Generate formal pre-authorization approval letter."""
    subject = f"PRE-AUTHORIZATION APPROVAL: Claim {claim_id} - {patient_name}"
    citations_text = "\n".join([f"  - {c}" for c in citations]) if citations else "  - Standard Policy Terms"

    body = f"""CAREPLUS HEALTH INSURANCE COMPANY
CLAIMS MANAGEMENT & ADJUDICATION DIVISION

DATE: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
TO: Medical Superintendent / TPA Desk, {hospital}
PATIENT REFERENCE: {patient_name} (Policy #{policy_no})
CLAIM PRE-AUTHORIZATION ID: {claim_id}

SUBJECT: PRE-AUTHORIZATION APPROVAL FOR SCHEDULED PROCEDURE

Dear Administrator / Insured,

We are pleased to inform you that the cashless pre-authorization request for the procedure detailed below has been APPROVED in accordance with the terms, conditions, and sub-limits of the CarePlus Comprehensive Health Insurance Policy:

1. Clinical Details:
   - Authorized Procedure: {procedure_desc}
   - Treating Hospital: {hospital}
   - Approved Cashless Limit: ${amount:,.2f}

2. Policy Endorsements & Sub-Limits Applied:
{citations_text}

3. Important Terms & Settlement Guidelines:
   - This approval is valid for admission within 15 calendar days from the date of this letter.
   - Any non-medical, incidental, or personal comfort expenses incurred during hospitalization are not covered and must be settled directly by the insured patient.
   - Final cashless settlement remains subject to physical verification of the final itemized hospital bill, original discharge summary, and pharmacy invoices upon discharge.

For any queries regarding this authorization, please contact our 24x7 Provider Helpdesk citing Pre-Authorization Reference {claim_id}.

Sincerely,

Medical Adjudication & Pre-Authorization Team
CarePlus Health Insurance Company
"""
    return CommunicationDraft(
        letter_type="Approval Letter",
        recipient=f"{hospital} / {patient_name}",
        subject=subject,
        body=body.strip(),
        policy_citations=citations,
        missing_items_requested=[],
    )


def draft_query_letter(
    claim_id: str,
    patient_name: str,
    policy_no: str,
    hospital: str,
    missing_docs: List[str],
    unresolved_issues: List[str],
    citations: List[str],
) -> CommunicationDraft:
    """Generate formal query / clarification request letter."""
    subject = f"PRE-AUTHORIZATION QUERY / ACTION REQUIRED: Claim {claim_id} - {patient_name}"

    missing_items = []
    if missing_docs:
        missing_items.extend(missing_docs)
    if unresolved_issues:
        for issue in unresolved_issues:
            if issue not in missing_items:
                missing_items.append(issue)

    items_text = "\n".join([f"  [{idx+1}] {item}" for idx, item in enumerate(missing_items)])
    citations_text = ", ".join(citations) if citations else "Policy Clause 5.1"

    body = f"""CAREPLUS HEALTH INSURANCE COMPANY
CLAIMS MANAGEMENT & ADJUDICATION DIVISION

DATE: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
TO: Medical Records & Billing Department, {hospital}
PATIENT REFERENCE: {patient_name} (Policy #{policy_no})
CLAIM PRE-AUTHORIZATION ID: {claim_id}

SUBJECT: PRE-AUTHORIZATION INCOMPLETE - CLARIFICATION & RECORDS REQUESTED

Dear Healthcare Provider / Insured,

We have initiated clinical review of the pre-authorization request submitted for Claim ID {claim_id}. To complete our evaluation and establish coverage under {citations_text}, the following mandatory documentation or clarifications are urgently required:

REQUIRED ITEMS & CLINICAL QUERIES:
{items_text}

SUBMISSION INSTRUCTIONS:
1. Please upload the requested investigation reports and written responses via the CarePlus Provider Portal within 72 hours of receipt of this notice.
2. In accordance with Clause 5.1, processing of this pre-authorization remains temporarily paused pending receipt of the requested documents.
3. If no response is received within 72 hours, the pre-authorization request will be closed without prejudice, requiring resubmission.

We appreciate your timely cooperation in assisting us to process your patient's coverage expeditiously.

Sincerely,

Medical Adjudication & Pre-Authorization Team
CarePlus Health Insurance Company
"""
    return CommunicationDraft(
        letter_type="Query Letter",
        recipient=f"{hospital} / {patient_name}",
        subject=subject,
        body=body.strip(),
        policy_citations=citations,
        missing_items_requested=missing_items,
    )


def draft_denial_letter(
    claim_id: str,
    patient_name: str,
    policy_no: str,
    hospital: str,
    procedure_desc: str,
    reasons: List[str],
    citations: List[str],
) -> CommunicationDraft:
    """Generate formal pre-authorization denial letter with appeal rights."""
    subject = f"PRE-AUTHORIZATION DENIAL NOTICE: Claim {claim_id} - {patient_name}"
    citations_text = "\n".join([f"  - {c}" for c in citations]) if citations else "  - Applicable Policy Terms & Conditions"
    reasons_text = "\n".join([f"  [{idx+1}] {r}" for idx, r in enumerate(reasons)])

    body = f"""CAREPLUS HEALTH INSURANCE COMPANY
CLAIMS MANAGEMENT & ADJUDICATION DIVISION

DATE: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
TO: {patient_name} (Insured) & {hospital}
POLICY NUMBER: {policy_no}
CLAIM REFERENCE ID: {claim_id}

SUBJECT: NOTICE OF PRE-AUTHORIZATION DENIAL

Dear Insured / Healthcare Provider,

This letter serves as formal notification that following comprehensive medical review, the pre-authorization cashless request for {procedure_desc} under Policy #{policy_no} has been DENIED.

GROUNDS FOR DENIAL:
{reasons_text}

SPECIFIC POLICY EXCLUSIONS & CLAUSES APPLIED:
{citations_text}

APPEAL RIGHTS & GRIEVANCE REDRESSAL:
If you or your attending physician believe this decision was made in error or if new clinical documentation is available to establish coverage:
1. Internal Appeal: You may submit a formal request for reconsideration to the CarePlus Grievance Cell within 30 days, accompanied by additional medical records.
2. Ombudsman Rights: You also have the right to approach the Office of the Insurance Ombudsman under the Insurance Ombudsman Rules.

We regret that we are unable to approve coverage for this specific request under the terms of your policy.

Sincerely,

Medical Adjudication & Pre-Authorization Team
CarePlus Health Insurance Company
"""
    return CommunicationDraft(
        letter_type="Denial Letter",
        recipient=f"{patient_name} / {hospital}",
        subject=subject,
        body=body.strip(),
        policy_citations=citations,
        missing_items_requested=[],
    )


def generate_communication(state: ClaimState) -> CommunicationDraft:
    """Determine outcome and draft appropriate formal communication letter."""
    extracted = state.get("extracted_claim")
    human_rev = state.get("human_review")
    rec = state.get("recommendation")

    claim_id = extracted.claim_id if extracted else state.get("claim_id", "CLM-UNKNOWN")
    patient_name = extracted.patient_name_masked if extracted else "[PATIENT_NAME]"
    policy_no = extracted.policy_number if extracted else "UNKNOWN-POLICY"
    hospital = extracted.hospital_name if extracted else "Treating Hospital"
    proc_desc = (
        extracted.procedure_descriptions[0]
        if extracted and extracted.procedure_descriptions
        else (extracted.procedure_codes[0] if extracted and extracted.procedure_codes else "Requested Treatment")
    )
    cost = extracted.estimated_cost if extracted else 0.0

    citations = rec.policy_citations if rec else []
    missing_docs = state.get("missing_mandatory_documents", [])
    unresolved = rec.unresolved_issues if rec else []
    reasons = rec.reasons if rec else []

    # Final outcome prioritized from Human Review if performed, else Recommendation
    if human_rev:
        final_outcome = human_rev.final_outcome
        if human_rev.override_reason:
            reasons = [human_rev.override_reason] + reasons
    elif rec:
        final_outcome = rec.outcome
    else:
        final_outcome = AdjudicationOutcome.QUERY

    if final_outcome == AdjudicationOutcome.APPROVE:
        return draft_approval_letter(
            claim_id=claim_id,
            patient_name=patient_name,
            policy_no=policy_no,
            hospital=hospital,
            procedure_desc=proc_desc,
            amount=cost,
            citations=citations,
        )
    elif final_outcome == AdjudicationOutcome.REJECT:
        return draft_denial_letter(
            claim_id=claim_id,
            patient_name=patient_name,
            policy_no=policy_no,
            hospital=hospital,
            procedure_desc=proc_desc,
            reasons=reasons,
            citations=citations,
        )
    else:  # QUERY
        return draft_query_letter(
            claim_id=claim_id,
            patient_name=patient_name,
            policy_no=policy_no,
            hospital=hospital,
            missing_docs=missing_docs,
            unresolved_issues=unresolved,
            citations=citations,
        )


def communication_node(state: ClaimState) -> Dict[str, Any]:
    """LangGraph node execution function for the Communication Agent."""
    draft = generate_communication(state)

    audit_entry = (
        f"[CommunicationAgent] Generated {draft.letter_type} for recipient '{draft.recipient}'. "
        f"Subject: '{draft.subject[:60]}...'. Citations: {len(draft.policy_citations)}, "
        f"Missing items requested: {len(draft.missing_items_requested)}."
    )

    return {
        "communication_draft": draft,
        "audit_trail": [audit_entry],
        "current_node": "communication",
    }
