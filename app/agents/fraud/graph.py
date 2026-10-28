"""Fraud & Anomaly Detection LangGraph Subgraph Assembly.

Constructs an isolated, multi-node StateGraph that sequentially evaluates:
1. Duplicate claim history
2. Regional billing benchmarks & spikes
3. Length-of-stay hospital variances
4. Composite risk aggregation

Also provides `fraud_detection_node` for seamless integration into the main
pre-authorization graph.
"""

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from app.agents.fraud.nodes import (
    FraudSubgraphState,
    billing_spike_node,
    duplicate_check_node,
    length_of_stay_node,
    risk_aggregator_node,
)
from app.state import ClaimState, FraudRiskReport

logger = logging.getLogger(__name__)


def create_fraud_subgraph():
    """Build and compile the isolated LangGraph Fraud Detection Subgraph."""
    workflow = StateGraph(FraudSubgraphState)

    # Add individual analysis nodes
    workflow.add_node("duplicate_check", duplicate_check_node)
    workflow.add_node("billing_spike", billing_spike_node)
    workflow.add_node("length_of_stay", length_of_stay_node)
    workflow.add_node("risk_aggregator", risk_aggregator_node)

    # Define linear execution flow
    workflow.add_edge(START, "duplicate_check")
    workflow.add_edge("duplicate_check", "billing_spike")
    workflow.add_edge("billing_spike", "length_of_stay")
    workflow.add_edge("length_of_stay", "risk_aggregator")
    workflow.add_edge("risk_aggregator", END)

    return workflow.compile()


# Pre-compiled singleton subgraph
fraud_subgraph = create_fraud_subgraph()


def fraud_detection_node(state: ClaimState) -> Dict[str, Any]:
    """LangGraph node execution function wrapping the Fraud Subgraph for the main workflow."""
    extracted = state.get("extracted_claim")

    if not extracted:
        default_report = FraudRiskReport(
            risk_score=0,
            reasons=["No extracted claim data available for fraud risk evaluation."],
        )
        return {
            "fraud_report": default_report,
            "audit_trail": ["[FraudRiskAgent] Skipped: No extracted claim data present."],
            "current_node": "fraud_detection",
        }

    # Map parent state into isolated subgraph state
    initial_subgraph_state: FraudSubgraphState = {
        "claim_id": extracted.claim_id,
        "patient_id": extracted.patient_id,
        "admission_date": extracted.admission_date or "",
        "discharge_date": extracted.discharge_date or "",
        "procedure_codes": extracted.procedure_codes,
        "diagnosis_codes": extracted.diagnosis_codes,
        "estimated_cost": extracted.estimated_cost,
        "length_of_stay": extracted.length_of_stay or 1,
        "is_duplicate": False,
        "billing_spike_detected": False,
        "stay_anomaly_detected": False,
        "risk_points": 0,
        "reasons": [],
    }

    # Execute compiled subgraph
    result = fraud_subgraph.invoke(initial_subgraph_state)
    report: FraudRiskReport = result.get("fraud_report") or FraudRiskReport()

    audit_msg = (
        f"[FraudRiskAgent] Risk assessment complete: Score {report.risk_score}/100. "
        f"Duplicate: {report.is_duplicate}, Billing Spike: {report.billing_spike_detected}, "
        f"Stay Anomaly: {report.stay_anomaly_detected}. Reasons count: {len(report.reasons)}."
    )

    return {
        "fraud_report": report,
        "audit_trail": [audit_msg],
        "current_node": "fraud_detection",
    }
