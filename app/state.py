"""Shared Typed State Definition for Claim Pre-Authorization Multi-Agent Graph.

This module defines all Pydantic schemas and the central LangGraph TypedDict state.
Every agent in the workflow reads from and writes to this state.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ==============================================================================
# Enumerations for Strict Controlled Vocabularies
# ==============================================================================

class DocumentType(str, Enum):
    CLAIM_FORM = "claim_form"
    HOSPITAL_BILL = "hospital_bill"
    DISCHARGE_SUMMARY = "discharge_summary"
    PRESCRIPTION = "prescription"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    UNKNOWN = "unknown"


class AdjudicationOutcome(str, Enum):
    APPROVE = "Approve"
    QUERY = "Query"
    REJECT = "Reject"


class HumanReviewAction(str, Enum):
    APPROVE = "Approve"
    QUERY = "Query"
    REJECT = "Reject"
    OVERRIDE = "Override"


# ==============================================================================
# Pydantic Schemas for Structured Artifacts
# ==============================================================================

class DocumentMeta(BaseModel):
    """Metadata and raw content for an uploaded claim document."""
    file_id: str = Field(description="Unique identifier for uploaded file")
    filename: str = Field(description="Original filename")
    doc_type: DocumentType = Field(default=DocumentType.UNKNOWN, description="Classified document type")
    page_count: int = Field(default=1, description="Number of pages in document")
    raw_text: str = Field(default="", description="Extracted raw text before redaction")
    redacted_text: str = Field(default="", description="Sanitized text with PII masked")
    ocr_applied: bool = Field(default=False, description="Whether Tesseract OCR was used")


class ExtractedClaim(BaseModel):
    """Structured fields extracted from claim form and medical records."""
    claim_id: str = Field(description="Unique claim identifier, e.g., CLM-2026-001")
    patient_id: str = Field(description="Anonymized patient reference ID")
    patient_name_masked: str = Field(default="[PATIENT_NAME]", description="Redacted patient name")
    patient_age: Optional[int] = Field(default=None, description="Patient age in years")
    patient_gender: Optional[str] = Field(default=None, description="Patient gender (Male, Female, Other)")
    policy_number: Optional[str] = Field(default=None, description="Insurance policy number")
    hospital_name: Optional[str] = Field(default=None, description="Name of treating hospital")
    admission_date: Optional[str] = Field(default=None, description="Admission date YYYY-MM-DD")
    discharge_date: Optional[str] = Field(default=None, description="Discharge date YYYY-MM-DD")
    length_of_stay: Optional[int] = Field(default=None, description="Duration of hospital stay in days")
    diagnosis_codes: List[str] = Field(default_factory=list, description="ICD-10 diagnosis codes (e.g., H25.0)")
    diagnosis_descriptions: List[str] = Field(default_factory=list, description="Text diagnosis descriptions")
    procedure_codes: List[str] = Field(default_factory=list, description="Procedure codes (e.g., CPT 66984)")
    procedure_descriptions: List[str] = Field(default_factory=list, description="Text procedure descriptions")
    estimated_cost: float = Field(default=0.0, description="Total claimed amount in local currency")
    extracted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PolicyClause(BaseModel):
    """Retrieved policy clause from vector RAG with citation metadata."""
    clause_id: str = Field(description="Clause identifier, e.g., SEC-1.1-ROOM")
    title: str = Field(description="Clause title, e.g., Room Rent Sub-limit")
    text: str = Field(description="Exact wording from the policy booklet")
    document_source: str = Field(description="Filename of the policy document")
    page_reference: int = Field(description="Page number in policy document")
    retrieval_score: float = Field(default=0.0, description="Vector similarity/distance score")


class CodeValidationItem(BaseModel):
    """Validation report for a single ICD-10 or Procedure code."""
    code: str
    code_type: str = Field(description="'ICD-10' or 'Procedure'")
    is_valid: bool
    standard_description: Optional[str] = None
    flags: List[str] = Field(default_factory=list, description="Clinical consistency warnings")


class CodeValidationReport(BaseModel):
    """Aggregate result from Code Validation Agent."""
    all_codes_valid: bool = True
    items: List[CodeValidationItem] = Field(default_factory=list)
    clinical_consistency_note: str = Field(default="")


class FraudRiskReport(BaseModel):
    """Transparent risk report from Fraud & Anomaly Subgraph."""
    risk_score: int = Field(default=0, ge=0, le=100, description="Aggregate risk score from 0 to 100")
    is_duplicate: bool = Field(default=False, description="Duplicate submission detected")
    billing_spike_detected: bool = Field(default=False, description="Bill is unusually high vs benchmarks")
    stay_anomaly_detected: bool = Field(default=False, description="Length of stay is statistically anomalous")
    reasons: List[str] = Field(default_factory=list, description="Human-readable transparent explanations")
    benchmark_median_cost: Optional[float] = Field(default=None, description="Benchmark median for procedure")
    benchmark_max_stay: Optional[int] = Field(default=None, description="Expected max stay for diagnosis")


class AdjudicationRecommendation(BaseModel):
    """Evidence-backed recommendation from Decision Agent."""
    outcome: AdjudicationOutcome
    confidence: float = Field(ge=0.0, le=1.0, description="Heuristic confidence score (not probability)")
    summary: str = Field(description="Executive summary of recommendation")
    reasons: List[str] = Field(default_factory=list, description="Numbered factual rationale")
    policy_citations: List[str] = Field(default_factory=list, description="Exact cited policy clauses")
    unresolved_issues: List[str] = Field(default_factory=list, description="Missing items or queries")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HumanReview(BaseModel):
    """Adjudicator review confirmation or override."""
    reviewer_id: str
    action: HumanReviewAction
    final_outcome: AdjudicationOutcome
    override_reason: Optional[str] = None
    reviewer_notes: str = ""
    reviewed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CommunicationDraft(BaseModel):
    """Patient/Hospital communication letter drafted by Communication Agent."""
    letter_type: str = Field(description="'Approval Letter', 'Query Letter', or 'Denial Letter'")
    recipient: str = Field(default="Treating Hospital / Insured Patient")
    subject: str
    body: str
    policy_citations: List[str] = Field(default_factory=list)
    missing_items_requested: List[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ==============================================================================
# Central LangGraph Typed Shared State
# ==============================================================================

class ClaimState(TypedDict, total=False):
    """The central state passed through all nodes in the LangGraph StateGraph.

    Uses `operator.add` annotations for list channels that should accumulate
    results as parallel nodes execute (e.g. audit logs, retrieved clauses).
    """

    # Claim Identification
    claim_id: str
    created_at: str

    # Document Extraction & Redaction
    uploaded_documents: List[DocumentMeta]
    extracted_claim: Optional[ExtractedClaim]
    missing_mandatory_documents: List[str]
    pii_redacted: bool

    # Parallel Validation Branches
    retrieved_clauses: List[PolicyClause]
    code_validation: Optional[CodeValidationReport]
    fraud_report: Optional[FraudRiskReport]

    # Decision & Synthesis
    recommendation: Optional[AdjudicationRecommendation]

    # Human-in-the-Loop Node
    human_review: Optional[HumanReview]
    is_paused_for_human: bool

    # Final Communication & Audit
    communication_draft: Optional[CommunicationDraft]
    audit_trail: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    iteration_count: int
    current_node: Annotated[str, lambda a, b: b]
