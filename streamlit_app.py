"""Streamlit Interactive Web Application Dashboard.

Multi-Agent Health Insurance Claim Pre-Authorization & Denial Prevention System
Powered by LangGraph, Ollama, ChromaDB, and Google Gemini.
"""

# Modern SQLite support for ChromaDB on Streamlit Cloud Linux
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_FILE_DIR = Path(__file__).resolve().parent
if (_FILE_DIR / "app").exists():
    _PROJECT_ROOT = _FILE_DIR
else:
    _PROJECT_ROOT = _FILE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project components
from app.agents.human_review import resume_human_review
from app.config import PROJECT_ROOT, is_ollama_available, settings
from app.graph import create_claim_graph
from app.state import (
    AdjudicationOutcome,
    ClaimState,
    CommunicationDraft,
    DocumentMeta,
    DocumentType,
    ExtractedClaim,
    HumanReview,
    HumanReviewAction,
)
from app.tools.database import load_reference_tables

# Set page configuration
st.set_page_config(
    page_title="CarePlus AI Claims Pre-Authorization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern enterprise aesthetics
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .badge-approve { background-color: #dcfce7; color: #166534; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; border: 1px solid #bbf7d0; display: inline-block; }
    .badge-query { background-color: #fef9c3; color: #854d0e; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; border: 1px solid #fef08a; display: inline-block; }
    .badge-reject { background-color: #fee2e2; color: #991b1b; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; border: 1px solid #fecaca; display: inline-block; }
    .card { background: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .letter-box { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 25px; font-family: 'Courier New', Courier, monospace; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; color: #1e293b; }
    .audit-step { border-left: 3px solid #3b82f6; padding-left: 12px; margin-bottom: 8px; font-family: monospace; font-size: 0.85rem; color: #334155; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper Functions: Data Loading & Graph Execution
# ==============================================================================

@st.cache_resource
def get_compiled_graph():
    """Cache and return compiled LangGraph instance."""
    load_reference_tables()
    return create_claim_graph(enable_human_review=True)


def load_synthetic_case(case_filename: str) -> Dict[str, Any]:
    """Load synthetic case JSON fixture."""
    path = PROJECT_ROOT / "data" / "synthetic_cases" / case_filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_documents_for_case(case_data: Dict[str, Any]) -> List[DocumentMeta]:
    """Construct DocumentMeta objects for the synthetic claim."""
    claim_id = case_data.get("claim_id", "CLM-UNKNOWN")
    docs = []

    # Claim Form
    form_text = f"""
PRE-AUTHORIZATION REQUEST FORM - HEALTH INSURANCE
Hospital: {case_data.get('hospital_name', 'General Hospital')}
Patient Full Name: {case_data.get('patient_name', 'Patient')}
Patient ID: {case_data.get('patient_id', 'PAT-00000')}
Patient Age: {case_data.get('patient_age', 50)} | Gender: {case_data.get('patient_gender', 'Male')}
Insurance Policy No: {case_data.get('policy_number', 'CP-POL-00000')}
Clinical Diagnosis: {', '.join(case_data.get('diagnosis_descriptions', []))}
ICD-10 Diagnosis Code: {', '.join(case_data.get('diagnosis_codes', []))}
Proposed Procedure: {', '.join(case_data.get('procedure_descriptions', []))}
CPT / Procedure Code: {', '.join(case_data.get('procedure_codes', []))}
Date of Proposed Admission: {case_data.get('admission_date', '2026-11-01')}
Date of Discharge: {case_data.get('discharge_date', '2026-11-02')}
Total Estimated Cost: ${case_data.get('estimated_cost', 0.0):,.2f}
"""
    docs.append(
        DocumentMeta(
            file_id=f"DOC-{claim_id}-FORM",
            filename=f"{claim_id}_claim_form.txt",
            doc_type=DocumentType.CLAIM_FORM,
            raw_text=form_text.strip(),
        )
    )

    # Hospital Bill
    docs.append(
        DocumentMeta(
            file_id=f"DOC-{claim_id}-BILL",
            filename=f"{claim_id}_hospital_bill.txt",
            doc_type=DocumentType.HOSPITAL_BILL,
            raw_text=f"Itemized estimate for {claim_id}: Total Claimed Amount: ${case_data.get('estimated_cost', 0.0):,.2f}.",
        )
    )

    # Diagnostic Report (omitted for Case 002)
    submitted_docs = case_data.get("documents_submitted", [])
    if "diagnostic_report" in submitted_docs:
        docs.append(
            DocumentMeta(
                file_id=f"DOC-{claim_id}-REPORT",
                filename=f"{claim_id}_diagnostic_report.txt",
                doc_type=DocumentType.DIAGNOSTIC_REPORT,
                raw_text=f"Clinical Diagnostic Report for {claim_id}: Medical examination confirms clinical indication for {', '.join(case_data.get('diagnosis_descriptions', []))}.",
            )
        )

    return docs


# ==============================================================================
# Sidebar & Configuration
# ==============================================================================

st.sidebar.image("https://img.icons8.com/color/96/shield-checked--v1.png", width=64)
st.sidebar.title("CarePlus Claims AI")
st.sidebar.caption("Multi-Agent Pre-Authorization Engine")

st.sidebar.markdown("---")
st.sidebar.subheader("Select Test Case")

case_options = {
    "Case 001: Cataract Surgery (Approval)": "case_001_approval_cataract.json",
    "Case 002: Meniscus Tear - Missing MRI (Query)": "case_002_query_missing_docs.json",
    "Case 003: Cosmetic Rhinoplasty (Rejection)": "case_003_reject_cosmetic_exclusion.json",
    "Case 004: Duplicate Claim Submission (Query)": "case_004_fraud_duplicate_claim.json",
    "Case 005: Inflated Bill & 14-day Stay (Query)": "case_005_fraud_inflated_bill.json",
}

selected_case_label = st.sidebar.selectbox("Choose Fixture:", list(case_options.keys()))
selected_case_file = case_options[selected_case_label]

st.sidebar.markdown("---")
st.sidebar.subheader("System Infrastructure")

ollama_up = is_ollama_available()
if ollama_up:
    st.sidebar.success(f"Local LLM: Ollama ({settings.ollama_llm_model})")
    st.sidebar.success(f"Vector DB: ChromaDB ({settings.ollama_embed_model})")
else:
    st.sidebar.info("Local LLM: Rule Engine (Cloud Ready)")
    st.sidebar.success("Vector DB: ChromaDB (Local Semantic Engine)")

st.sidebar.success("Checkpointer: SQLite checkpoints.db")

if settings.is_gemini_configured:
    st.sidebar.info(f"Cloud LLM: Gemini ({settings.gemini_model})")
else:
    st.sidebar.info("Cloud LLM: Offline rule engine active")

with st.sidebar.expander("🔑 Cloud AI API Key (Optional)"):
    st.caption("Provide a Gemini API Key for cloud reasoning, or leave blank to run completely offline.")
    custom_gemini_key = st.text_input(
        "Gemini API Key:",
        type="password",
        value=settings.gemini_api_key or "",
        key="gemini_key_input",
    )
    if custom_gemini_key and custom_gemini_key != settings.gemini_api_key:
        settings.gemini_api_key = custom_gemini_key
        st.rerun()

run_pipeline_btn = st.sidebar.button("🚀 Process Claim Pre-Authorization", type="primary", use_container_width=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "claim_state" not in st.session_state:
    st.session_state.claim_state = None
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False


# ==============================================================================
# Pipeline Execution Handler
# ==============================================================================

graph = get_compiled_graph()

if run_pipeline_btn:
    try:
        case_data = load_synthetic_case(selected_case_file)
        claim_id = case_data.get("claim_id", "CLM-TEST")
        thread_id = f"THREAD-{claim_id}-{int(datetime.now().timestamp())}"

        docs = prepare_documents_for_case(case_data)

        initial_state: ClaimState = {
            "claim_id": claim_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_documents": docs,
            "missing_mandatory_documents": [],
            "pii_redacted": False,
            "retrieved_clauses": [],
            "audit_trail": [f"[UI] Initiated claim processing for {claim_id} ({selected_case_label})."],
            "errors": [],
        }

        config = {"configurable": {"thread_id": thread_id}}

        with st.spinner("Executing Multi-Agent Workflow (Extraction -> Redaction -> Parallel RAG/Codes/Fraud -> Decision)..."):
            # Graph runs and pauses at human_review
            graph.invoke(initial_state, config=config)
            snapshot = graph.get_state(config)

            st.session_state.current_thread_id = thread_id
            st.session_state.claim_state = snapshot.values
            st.session_state.is_paused = "human_review" in snapshot.next

        st.success(f"Claim {claim_id} processed! Awaiting Adjudicator Review.")
    except Exception as err:
        logger.exception("Error executing claim pre-authorization pipeline")
        st.error(f"Error processing claim: {err}")


# ==============================================================================
# Main Dashboard UI
# ==============================================================================

st.title("🏥 Health Insurance Claim Pre-Authorization Dashboard")
st.caption("Explainable AI Adjudication & Regulatory Compliance with LangGraph and ChromaDB")

if not st.session_state.claim_state:
    st.info("👈 Select a synthetic case fixture from the sidebar and click **'Process Claim Pre-Authorization'** to begin.")
else:
    state = st.session_state.claim_state
    extracted: Optional[ExtractedClaim] = state.get("extracted_claim")
    rec = state.get("recommendation")
    fraud = state.get("fraud_report")
    code_val = state.get("code_validation")
    human_rev: Optional[HumanReview] = state.get("human_review")
    comm_draft: Optional[CommunicationDraft] = state.get("communication_draft")

    tab1, tab2, tab3 = st.tabs([
        "📋 Adjudication & Human Review Console",
        "🔍 Multi-Agent Evidence Breakdown",
        "✉️ Formal Letters & Audit Log",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Adjudication & Human Review Console
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("1. Claim Identification & Decision Status")

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)

        outcome_val = rec.outcome.value if rec else "Pending"
        badge_class = "badge-approve" if outcome_val == "Approve" else ("badge-query" if outcome_val == "Query" else "badge-reject")

        with col1:
            st.markdown(f"**AI Recommendation:**<br><span class='{badge_class}'>{outcome_val.upper()}</span>", unsafe_allow_html=True)
        with col2:
            st.metric("Claim Reference ID", state.get("claim_id", "N/A"))
        with col3:
            st.metric("Confidence Score", f"{rec.confidence*100:.0f}%" if rec else "N/A")
        with col4:
            st.metric("Claimed Amount", f"${extracted.estimated_cost:,.2f}" if extracted else "$0.00")

        st.markdown("<br>", unsafe_allow_html=True)

        # Extracted Patient & Hospital Card
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Patient & Clinical Profile (PII Masked)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Patient Name:** `{extracted.patient_name_masked if extracted else 'N/A'}`")
                st.write(f"**Patient ID:** `{extracted.patient_id if extracted else 'N/A'}`")
                st.write(f"**Demographics:** {extracted.patient_age if extracted else 'N/A'} yrs | {extracted.patient_gender if extracted else 'N/A'}")
            with c2:
                st.write(f"**Insurance Policy No:** `{extracted.policy_number if extracted else 'N/A'}`")
                st.write(f"**Treating Facility:** {extracted.hospital_name if extracted else 'N/A'}")
                st.write(f"**Length of Stay:** {extracted.length_of_stay if extracted else 1} day(s)")
            with c3:
                st.write(f"**Diagnosis Codes:** `{', '.join(extracted.diagnosis_codes) if extracted else 'N/A'}`")
                st.write(f"**Procedure Codes:** `{', '.join(extracted.procedure_codes) if extracted else 'N/A'}`")
                st.write(f"**PII Sanitized:** {'✅ Redacted' if state.get('pii_redacted') else '❌ Raw'}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Executive Summary Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Evidence-Backed Synthesis & Rationale")
        if rec:
            st.info(rec.summary)
            st.markdown("**Adjudication Key Factors:**")
            for r in rec.reasons:
                st.write(f"- {r}")
            if rec.policy_citations:
                st.markdown("**Cited Policy Clauses:**")
                for c in rec.policy_citations:
                    st.write(f"  • {c}")
            if rec.unresolved_issues:
                st.warning(f"**Unresolved Queries / Missing Items:** {', '.join(rec.unresolved_issues)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Human-in-the-Loop Review Console
        # ----------------------------------------------------------------------
        st.subheader("2. Human-in-the-Loop Adjudication Console")
        st.caption("Regulatory compliance requires medical adjudicator authorization before issuing finalized determinations.")

        if st.session_state.is_paused:
            with st.form("human_review_form"):
                st.warning("⚠️ **Claim Paused at Human Review Node**: Review the AI recommendation above and authorize or override the determination.")

                col_act, col_user = st.columns([2, 2])
                with col_act:
                    action_choice = st.radio(
                        "Adjudicator Action:",
                        ["Approve", "Query", "Reject", "Override"],
                        index=0 if outcome_val == "Approve" else (1 if outcome_val == "Query" else 2),
                    )
                with col_user:
                    reviewer_id = st.text_input("Adjudicator Medical ID:", value="DR-ADJUDICATOR-884")

                override_outcome = None
                override_reason = None
                if action_choice == "Override":
                    st.markdown("##### Override Configuration (Mandatory Audit Protocol)")
                    cov1, cov2 = st.columns(2)
                    with cov1:
                        override_outcome = st.selectbox("New Final Outcome:", ["Approve", "Query", "Reject"])
                    with cov2:
                        override_reason = st.text_area(
                            "Mandatory Clinical Override Rationale:",
                            placeholder="Explain the clinical exception, ombudsman precedent, or discretionary coverage justification...",
                        )

                reviewer_notes = st.text_area("Adjudicator Notes / Audit Remarks:", placeholder="Add optional clinical observations...")
                submit_review = st.form_submit_button("✍️ Authorize Determination & Resume Workflow", type="primary")

                if submit_review:
                    if action_choice == "Override" and (not override_reason or len(override_reason.strip()) < 10):
                        st.error("Adjudicator Override requires a detailed clinical rationale (minimum 10 characters).")
                    else:
                        with st.spinner("Resuming workflow with human authorization..."):
                            resumed_state = resume_human_review(
                                graph=graph,
                                thread_id=st.session_state.current_thread_id,
                                reviewer_id=reviewer_id,
                                action=action_choice,
                                final_outcome=override_outcome,
                                reviewer_notes=reviewer_notes,
                                override_reason=override_reason,
                            )
                            st.session_state.claim_state = resumed_state
                            st.session_state.is_paused = False
                            st.rerun()

        elif human_rev:
            st.success(
                f"✅ **Adjudication Finalized** by **{human_rev.reviewer_id}** at {human_rev.reviewed_at[:19]}. "
                f"Action: **{human_rev.action.value.upper()}** (Final Outcome: **{human_rev.final_outcome.value}**)."
            )
            if human_rev.override_reason:
                st.info(f"**Override Justification:** {human_rev.override_reason}")

    # --------------------------------------------------------------------------
    # TAB 2: Multi-Agent Evidence Breakdown
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Parallel Multi-Agent Evaluation Streams")
        st.caption("Inspect the independent findings of each specialized agent.")

        stream_col1, stream_col2, stream_col3 = st.columns(3)

        # Stream 1: Policy RAG Agent
        with stream_col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📜 Policy RAG Agent")
            clauses = state.get("retrieved_clauses", [])
            st.metric("Retrieved Grounded Clauses", len(clauses))

            if clauses:
                for cl in clauses:
                    score = getattr(cl, "retrieval_score", 0.0)
                    st.markdown(f"**{cl.clause_id}: {cl.title}**")
                    st.caption(f"Relevance Score: {score:.4f} | Source: {cl.document_source} (p. {cl.page_reference})")
                    with st.expander("View Clause Text"):
                        st.write(cl.text)
                    st.markdown("---")
            else:
                st.write("No specific clauses retrieved (fast-path routing applied).")
            st.markdown("</div>", unsafe_allow_html=True)

        # Stream 2: Code Validation Agent
        with stream_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🩺 Code Validation Agent")
            if code_val:
                status_icon = "✅" if code_val.all_codes_valid else "⚠️"
                st.metric("Coding Validity", f"{status_icon} {'Valid' if code_val.all_codes_valid else 'Flags Raised'}")
                st.caption(f"Clinical Note: {code_val.clinical_consistency_note}")

                st.markdown("##### Verified Codes:")
                for item in code_val.items:
                    flag_text = f" ({', '.join(item.flags)})" if item.flags else ""
                    st.write(f"- `{item.code_type} {item.code}`: {item.standard_description}{flag_text}")
            else:
                st.write("Code validation skipped due to missing documents.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Stream 3: Fraud Risk Subgraph
        with stream_col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🛡️ Fraud & Anomaly Subgraph")
            if fraud:
                risk_color = "green" if fraud.risk_score < 40 else ("orange" if fraud.risk_score < 70 else "red")
                st.metric("Composite Risk Score", f"{fraud.risk_score} / 100")

                st.markdown("##### Anomaly Indicators:")
                st.write(f"- **Duplicate Claim:** {'⚠️ YES' if fraud.is_duplicate else '✅ Clean'}")
                st.write(f"- **Billing Spike:** {'⚠️ YES' if fraud.billing_spike_detected else '✅ Normal'}")
                st.write(f"- **Stay Anomaly:** {'⚠️ YES' if fraud.stay_anomaly_detected else '✅ Normal'}")

                st.markdown("##### Risk Explanations:")
                for r in fraud.reasons:
                    st.write(f"• {r}")
            else:
                st.write("Fraud checks skipped due to missing documents.")
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 3: Formal Letters & Audit Trail
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Official Communication & System Audit Trail")

        # Communication Letter Preview
        st.markdown("#### Generated Formal Letter")
        if comm_draft:
            col_l1, col_l2 = st.columns([3, 1])
            with col_l1:
                st.markdown(f"**Letter Type:** `{comm_draft.letter_type}` | **Subject:** `{comm_draft.subject}`")
            with col_l2:
                st.download_button(
                    label="📥 Download Letter (.txt)",
                    data=comm_draft.body,
                    file_name=f"{state.get('claim_id')}_{comm_draft.letter_type.replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.markdown(f"<div class='letter-box'>{comm_draft.body}</div>", unsafe_allow_html=True)
        else:
            st.info("Letter drafting will execute after the Human Adjudicator review is submitted.")

        st.markdown("---")

        # Chronological Audit Trail
        st.markdown("#### Chronological Audit Trail")
        st.caption("Immutable record of all agent execution steps, decisions, and security validations.")
        audit_trail = state.get("audit_trail", [])
        if audit_trail:
            for step in audit_trail:
                st.markdown(f"<div class='audit-step'>{step}</div>", unsafe_allow_html=True)
        else:
            st.write("No audit entries recorded.")
