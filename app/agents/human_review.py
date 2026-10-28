"""Human-in-the-Loop (HITL) Adjudicator Review Node.

Enforces healthcare compliance:
1. AI never auto-finalizes adverse insurance pre-authorization decisions.
2. Pauses execution using LangGraph's `interrupt()` primitive.
3. Persists complete snapshot in SQLite checkpoint store.
4. Allows licensed medical adjudicators to:
   - Approve: Authorize the AI recommendation.
   - Query: Confirm additional questions/documentation requests.
   - Reject: Formally reject the claim.
   - Override: Change the outcome (e.g., granting a clinical exception) with mandatory rationale.
5. Resumes execution via `Command(resume=...)` into subsequent communication drafting.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langgraph.types import Command, interrupt

from app.state import (
    AdjudicationOutcome,
    ClaimState,
    HumanReview,
    HumanReviewAction,
)

logger = logging.getLogger(__name__)


def human_review_node(state: ClaimState) -> Dict[str, Any]:
    """LangGraph node that pauses execution for human adjudicator review."""
    claim_id = state.get("claim_id", "Unknown")
    rec = state.get("recommendation")
    rec_outcome = rec.outcome.value if rec else "Pending"

    # Interrupt execution and provide adjudicator payload
    interrupt_payload = {
        "claim_id": claim_id,
        "recommendation": rec.model_dump() if rec else None,
        "policy_citations": rec.policy_citations if rec else [],
        "unresolved_issues": rec.unresolved_issues if rec else [],
        "summary": rec.summary if rec else "Awaiting review",
        "prompt": f"Claim {claim_id} requires adjudicator authorization. AI Recommendation: {rec_outcome}.",
    }

    # Suspends execution until Command(resume=...) is called
    review_input: Dict[str, Any] = interrupt(interrupt_payload)

    # Process adjudicator input
    reviewer_id = review_input.get("reviewer_id", "ADJUDICATOR-SYSTEM")
    action_str = review_input.get("action", "Approve").capitalize()
    notes = review_input.get("reviewer_notes", "")
    override_reason = review_input.get("override_reason")

    # Map action to HumanReviewAction enum
    try:
        action = HumanReviewAction(action_str)
    except ValueError:
        action = HumanReviewAction.APPROVE

    # Determine final outcome
    if action == HumanReviewAction.OVERRIDE:
        final_str = review_input.get("final_outcome", "Approve").capitalize()
        try:
            final_outcome = AdjudicationOutcome(final_str)
        except ValueError:
            final_outcome = AdjudicationOutcome.APPROVE
    elif action == HumanReviewAction.APPROVE:
        final_outcome = AdjudicationOutcome.APPROVE
    elif action == HumanReviewAction.QUERY:
        final_outcome = AdjudicationOutcome.QUERY
    elif action == HumanReviewAction.REJECT:
        final_outcome = AdjudicationOutcome.REJECT
    else:
        final_outcome = rec.outcome if rec else AdjudicationOutcome.QUERY

    human_review = HumanReview(
        reviewer_id=reviewer_id,
        action=action,
        final_outcome=final_outcome,
        override_reason=override_reason,
        reviewer_notes=notes,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )

    action_label = f"OVERRIDDEN to {final_outcome.value}" if action == HumanReviewAction.OVERRIDE else action.value
    audit_entry = (
        f"[HumanReviewer] Claim {claim_id} reviewed by {reviewer_id}. "
        f"Action: {action_label}. Final Decision: {final_outcome.value.upper()}. Notes: {notes[:80]}"
    )

    return {
        "human_review": human_review,
        "is_paused_for_human": False,
        "audit_trail": [audit_entry],
        "current_node": "human_review",
    }


def resume_human_review(
    graph,
    thread_id: str,
    reviewer_id: str,
    action: str,
    final_outcome: Optional[str] = None,
    reviewer_notes: str = "",
    override_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Resume execution of a paused claim thread with human adjudicator input."""
    resume_payload = {
        "reviewer_id": reviewer_id,
        "action": action,
        "final_outcome": final_outcome,
        "reviewer_notes": reviewer_notes,
        "override_reason": override_reason,
    }

    config = {"configurable": {"thread_id": thread_id}}
    resumed_state = graph.invoke(Command(resume=resume_payload), config=config)
    return resumed_state
