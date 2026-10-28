"""Central LangGraph Multi-Agent Claim Pre-Authorization Graph.

Orchestrates the entire claim evaluation workflow:
1. Document Extraction Node (extracts text, metadata, classifies docs)
2. PII Redaction Node (sanitizes patient data with strict privacy guardrails)
3. Conditional Branching (fast-path to Decision if mandatory documents are missing)
4. Parallel Fan-out Barrier:
   - Policy RAG Agent (retrieves grounded clauses & sub-limits from ChromaDB)
   - Code Validation Agent (verifies ICD-10, CPT, demographic & clinical rules)
   - Fraud & Anomaly Subgraph (evaluates duplicates, billing spikes, stay durations)
5. Synchronization Barrier Join (merges parallel analysis streams)
6. Decision Synthesis Agent (synthesizes evidence into typed recommendation)
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agents.code_validation import code_validation_node
from app.agents.communication import communication_node
from app.agents.decision import decision_synthesis_node
from app.agents.extraction import extraction_node
from app.agents.fraud import fraud_detection_node
from app.agents.human_review import human_review_node
from app.agents.policy_rag import policy_rag_node
from app.agents.redaction import redaction_node
from app.config import settings
from app.state import ClaimState

logger = logging.getLogger(__name__)


# ==============================================================================
# Routing & Barrier Nodes
# ==============================================================================

def fan_out_node(state: ClaimState) -> Dict[str, Any]:
    """Fan-out dispatcher that initiates concurrent evaluation branches."""
    return {
        "audit_trail": [
            "[WorkflowSupervisor] Mandatory documents verified. Dispatching parallel evaluation streams: "
            "[Policy RAG, Code Validation, Fraud Risk Subgraph]."
        ],
        "current_node": "fan_out_barrier",
    }


def barrier_join_node(state: ClaimState) -> Dict[str, Any]:
    """Synchronization barrier that confirms completion of all parallel branches."""
    clauses = state.get("retrieved_clauses") or []
    code_val = state.get("code_validation")
    fraud = state.get("fraud_report")

    clauses_count = len(clauses)
    codes_ok = code_val.all_codes_valid if code_val else False
    risk_score = fraud.risk_score if fraud else 0

    audit_entry = (
        f"[WorkflowSupervisor] Parallel analysis streams synchronized. "
        f"Policy Clauses: {clauses_count}, Code Validity: {codes_ok}, Fraud Risk Score: {risk_score}/100. "
        f"Transitioning to Decision Synthesis."
    )

    return {
        "audit_trail": [audit_entry],
        "current_node": "barrier_join",
    }


def route_after_redaction(state: ClaimState) -> str:
    """Conditional router: Fast-path to Decision if mandatory documents are missing."""
    missing = state.get("missing_mandatory_documents") or []
    if missing:
        logger.info(
            "Claim '%s' missing mandatory documents %s. Routing to fast-path decision.",
            state.get("claim_id"),
            missing,
        )
        return "fast_path_decision"
    return "parallel_analysis"


# ==============================================================================
# Workflow Graph Construction
# ==============================================================================

def build_claim_workflow(enable_human_review: bool = True) -> StateGraph:
    """Assemble and configure nodes and edges for the claim pre-authorization StateGraph."""
    workflow = StateGraph(ClaimState)

    # 1. Sequential Ingestion Nodes
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("redaction", redaction_node)

    # 2. Parallel Dispatcher & Barrier
    workflow.add_node("fan_out_barrier", fan_out_node)

    # 3. Parallel Specialized Agents
    workflow.add_node("policy_rag", policy_rag_node)
    workflow.add_node("code_validation", code_validation_node)
    workflow.add_node("fraud_detection", fraud_detection_node)

    # 4. Synchronization Barrier
    workflow.add_node("barrier_join", barrier_join_node)

    # 5. Evidence Synthesis
    workflow.add_node("decision_synthesis", decision_synthesis_node)

    # 6. Human Review Node (Human-in-the-Loop)
    if enable_human_review:
        workflow.add_node("human_review", human_review_node)

    # 7. Final Patient & Provider Communication Letter
    workflow.add_node("communication", communication_node)

    # --------------------------------------------------------------------------
    # Edges & Flow Control
    # --------------------------------------------------------------------------
    workflow.add_edge(START, "extraction")
    workflow.add_edge("extraction", "redaction")

    # Conditional Branching after PII Redaction
    workflow.add_conditional_edges(
        "redaction",
        route_after_redaction,
        {
            "fast_path_decision": "decision_synthesis",
            "parallel_analysis": "fan_out_barrier",
        },
    )

    # Fan-out to 3 concurrent branches
    workflow.add_edge("fan_out_barrier", "policy_rag")
    workflow.add_edge("fan_out_barrier", "code_validation")
    workflow.add_edge("fan_out_barrier", "fraud_detection")

    # Fan-in from all 3 concurrent branches to synchronization barrier
    workflow.add_edge("policy_rag", "barrier_join")
    workflow.add_edge("code_validation", "barrier_join")
    workflow.add_edge("fraud_detection", "barrier_join")

    # Barrier transitions to Decision Synthesis
    workflow.add_edge("barrier_join", "decision_synthesis")

    if enable_human_review:
        workflow.add_edge("decision_synthesis", "human_review")
        workflow.add_edge("human_review", "communication")
        workflow.add_edge("communication", END)
    else:
        workflow.add_edge("decision_synthesis", "communication")
        workflow.add_edge("communication", END)

    return workflow


def get_sqlite_checkpointer(db_path: Optional[str] = None) -> SqliteSaver:
    """Create a persistent SQLite checkpointer for thread state recovery."""
    resolved_path = Path(db_path or (Path(settings.database_path).parent / "checkpoints.db"))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def create_claim_graph(checkpointer: Optional[Any] = None, enable_human_review: bool = True):
    """Compile the claim workflow with persistent checkpointing."""
    workflow = build_claim_workflow(enable_human_review=enable_human_review)
    if checkpointer is None:
        checkpointer = get_sqlite_checkpointer()
    elif checkpointer is False:
        return workflow.compile()
    return workflow.compile(checkpointer=checkpointer)
