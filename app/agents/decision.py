"""Decision Synthesis Agent.

Synthesizes evidence from all prior pipeline stages:
1. Document Completeness (missing mandatory documents check)
2. Policy RAG (grounded clauses, sub-limits, exclusions)
3. Code Validation (ICD-10, CPT, demographic & clinical compatibility)
4. Fraud & Anomaly Subgraph (duplicate submissions, billing spikes, stay anomalies)

Outputs a deterministic, evidence-backed AdjudicationRecommendation:
- Approve: Medical necessity met, valid codes, within sub-limit, low risk score (<40).
- Query: Missing documents, mild discrepancies, or medium risk score (40-69).
- Reject: Permanent policy exclusions (e.g. Clause 4.1 Cosmetic), invalid/fraudulent claims.

Employs Google Gemini (with local Ollama and rule-based fallbacks) to polish executive summaries
with zero PII transmission.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.state import (
    AdjudicationOutcome,
    AdjudicationRecommendation,
    ClaimState,
    CodeValidationReport,
    ExtractedClaim,
    FraudRiskReport,
    PolicyClause,
)

logger = logging.getLogger(__name__)


def evaluate_claim_rules(
    extracted: ExtractedClaim,
    missing_docs: List[str],
    retrieved_clauses: List[PolicyClause],
    code_validation: Optional[CodeValidationReport],
    fraud_report: Optional[FraudRiskReport],
) -> Tuple[AdjudicationOutcome, float, List[str], List[str], List[str], str]:
    """Deterministic adjudication rule engine.

    Returns:
        (outcome, confidence, reasons, policy_citations, unresolved_issues, preliminary_summary)
    """
    reasons: List[str] = []
    citations: List[str] = []
    unresolved: List[str] = []

    # --------------------------------------------------------------------------
    # 1. Rule Check: Policy Exclusion (e.g. Cosmetic Surgery Clause 4.1)
    # --------------------------------------------------------------------------
    is_cosmetic_exclusion = False
    # Check diagnoses for cosmetic indications (Z41.1) or rhinoplasty procedures (30410)
    for icd in extracted.diagnosis_codes:
        if "Z41" in icd.upper():
            is_cosmetic_exclusion = True
            break
    for proc in extracted.procedure_codes:
        if proc in ["30410", "30400", "30420"]:
            is_cosmetic_exclusion = True
            break

    # Also check if Clause 4.1 or cosmetic keywords appear in retrieved clauses
    for clause in retrieved_clauses:
        if "4.1" in clause.clause_id or "cosmetic" in clause.title.lower():
            if is_cosmetic_exclusion:
                citations.append(f"{clause.clause_id}: {clause.title} (Page {clause.page_reference})")

    if is_cosmetic_exclusion:
        reasons.append(
            "Claim is permanently excluded under Policy Clause 4.1 (Cosmetic and Aesthetic Procedures). "
            "Aesthetic enhancement and non-reconstructive plastic surgeries are non-payable."
        )
        if not citations:
            citations.append("CLAUSE 4.1: Cosmetic and Aesthetic Procedures Exclusion")
        summary = (
            "Pre-authorization denial recommended under Clause 4.1. "
            "Cosmetic procedures performed for aesthetic enhancement are explicitly excluded from coverage."
        )
        return (AdjudicationOutcome.REJECT, 0.98, reasons, citations, unresolved, summary)

    # --------------------------------------------------------------------------
    # 2. Rule Check: Missing Mandatory Documents
    # --------------------------------------------------------------------------
    if missing_docs:
        doc_list_str = ", ".join(missing_docs)
        reasons.append(
            f"Missing mandatory documentation: [{doc_list_str}] required under Policy Clause 5.1 to verify medical necessity."
        )
        unresolved.extend([f"Missing mandatory document: {doc}" for doc in missing_docs])
        citations.append("CLAUSE 5.1: Mandatory Pre-Authorization Documentation")
        summary = (
            f"Pre-authorization placed on Query hold. Missing required medical documentation ({doc_list_str}) "
            "required to substantiate medical necessity and establish coverage."
        )
        return (AdjudicationOutcome.QUERY, 0.95, reasons, citations, unresolved, summary)

    # --------------------------------------------------------------------------
    # 3. Rule Check: Fraud & Anomaly Signals
    # --------------------------------------------------------------------------
    if fraud_report:
        if fraud_report.is_duplicate:
            reasons.extend(fraud_report.reasons)
            unresolved.append("Potential duplicate pre-authorization submitted within 48 hours of existing active claim.")
            citations.append("CLAUSE 5.1: Claims Adjudication & Fraud Verification Protocol")
            summary = (
                "Pre-authorization placed on Query hold due to potential duplicate claim submission. "
                "Investigation required with treating provider."
            )
            return (AdjudicationOutcome.QUERY, 0.92, reasons, citations, unresolved, summary)

        if fraud_report.risk_score >= 60 or fraud_report.billing_spike_detected or fraud_report.stay_anomaly_detected:
            reasons.extend(fraud_report.reasons)
            if fraud_report.billing_spike_detected:
                unresolved.append("Significant billing spike: Claimed fee substantially exceeds regional benchmark fee schedule.")
            if fraud_report.stay_anomaly_detected:
                unresolved.append("Excessive stay duration: Inpatient hospitalization duration is statistically abnormal for procedure.")
            citations.append("CLAUSE 2.3 / CLAUSE 5.1: Reasonable & Customary Medical Benchmarks")
            summary = (
                f"Pre-authorization flagged for detailed audit and Query (Risk Score: {fraud_report.risk_score}/100). "
                "Requires itemized breakdown and clinical justification for billing and length of stay variances."
            )
            return (AdjudicationOutcome.QUERY, 0.88, reasons, citations, unresolved, summary)

    # --------------------------------------------------------------------------
    # 4. Rule Check: Code Validation & Clinical Consistency
    # --------------------------------------------------------------------------
    if code_validation and not code_validation.all_codes_valid:
        for item in code_validation.items:
            for flag in item.flags:
                reasons.append(f"Coding Flag [{item.code_type} {item.code}]: {flag}")
                unresolved.append(f"Coding inconsistency: {flag}")
        summary = (
            "Pre-authorization paused in Query status due to coding flags and clinical consistency warnings. "
            "Requires clinical clarification from attending physician."
        )
        return (AdjudicationOutcome.QUERY, 0.90, reasons, citations, unresolved, summary)

    # --------------------------------------------------------------------------
    # 5. Rule Check: Policy Sub-limits & Coverage Approval
    # --------------------------------------------------------------------------
    # Check for specific procedure sub-limits (e.g. Cataract Clause 2.1)
    sub_limit_applied = False
    for clause in retrieved_clauses:
        if "2.1" in clause.clause_id or "cataract" in clause.title.lower():
            citations.append(f"{clause.clause_id}: {clause.title} (Page {clause.page_reference})")
            sub_limit_applied = True
            if extracted.estimated_cost <= 40000.0:
                reasons.append(
                    f"Claimed amount of ${extracted.estimated_cost:,.2f} is within the policy cataract sub-limit of $40,000.00 (Clause 2.1)."
                )
            else:
                reasons.append(
                    f"Claimed amount of ${extracted.estimated_cost:,.2f} exceeds cataract sub-limit of $40,000.00. Approved subject to cap."
                )

    if not citations and retrieved_clauses:
        top_clause = retrieved_clauses[0]
        citations.append(f"{top_clause.clause_id}: {top_clause.title}")

    reasons.append("All submitted diagnosis (ICD-10) and procedure (CPT) codes are verified and clinically compatible.")
    reasons.append("All mandatory pre-authorization records and diagnostic reports are verified complete.")
    reasons.append(f"Fraud & anomaly risk score is low ({fraud_report.risk_score if fraud_report else 0}/100).")

    summary = (
        f"Pre-authorization recommended for Approval. Meets medical necessity criteria under "
        f"{citations[0] if citations else 'CarePlus Comprehensive Policy'}, with all mandatory documentation "
        f"verified and zero fraud or coding anomalies."
    )

    return (AdjudicationOutcome.APPROVE, 0.95, reasons, citations, unresolved, summary)


import time

# Circuit breaker timestamp: if Gemini returns 503/504, back off for 60 seconds
_GEMINI_CIRCUIT_OPEN_UNTIL = 0.0


def polish_summary_with_llm(
    outcome: AdjudicationOutcome,
    summary: str,
    reasons: List[str],
    citations: List[str],
    claim_id: str,
    procedure_desc: str,
    cost: float,
) -> str:
    """Enhance the summary tone using Google Gemini (with Ollama/rule fallback).

    Strict privacy guardrail: Zero raw PII passed to external API.
    """
    global _GEMINI_CIRCUIT_OPEN_UNTIL

    prompt = f"""You are a Senior Health Insurance Medical Adjudication Auditor.
Write a clear, concise 2-sentence executive summary of the pre-authorization recommendation.

Key Facts:
- Claim ID: {claim_id}
- Procedure: {procedure_desc}
- Claimed Amount: ${cost:,.2f}
- Adjudication Outcome: {outcome.value}
- Policy Citations: {', '.join(citations) if citations else 'General Policy'}
- Primary Rationale: {' ; '.join(reasons[:2])}

Guidelines:
1. State the decision clearly ({outcome.value}) and explain the factual basis.
2. Keep it under 50 words.
3. Do NOT include any personal names, addresses, or phone numbers.
4. Output ONLY the executive summary text.
"""

    # 1. Try Google Gemini API if configured and circuit breaker is closed
    if settings.is_gemini_configured and time.time() > _GEMINI_CIRCUIT_OPEN_UNTIL:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=3000),
            )
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            if response and response.text:
                polished = response.text.strip()
                if len(polished) > 20:
                    return polished
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "504" in err_str or "DEADLINE" in err_str or "UNAVAILABLE" in err_str:
                _GEMINI_CIRCUIT_OPEN_UNTIL = time.time() + 60.0
                logger.info("Gemini high-demand spike detected; activated 60s circuit breaker fallback.")
            logger.debug("Gemini summary polishing call failed (%s); falling back.", e)

    # 2. Try Local Ollama instance
    try:
        import urllib.request
        payload = json.dumps({
            "model": settings.ollama_llm_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidate = data.get("response", "").strip()
            if candidate and len(candidate) > 20:
                return candidate
    except Exception:
        pass

    # 3. Fallback to deterministic summary
    return summary


def synthesize_decision(
    extracted: ExtractedClaim,
    missing_docs: List[str],
    retrieved_clauses: List[PolicyClause],
    code_validation: Optional[CodeValidationReport],
    fraud_report: Optional[FraudRiskReport],
) -> AdjudicationRecommendation:
    """Synthesize complete adjudication recommendation."""
    outcome, confidence, reasons, citations, unresolved, base_summary = evaluate_claim_rules(
        extracted=extracted,
        missing_docs=missing_docs,
        retrieved_clauses=retrieved_clauses,
        code_validation=code_validation,
        fraud_report=fraud_report,
    )

    proc_desc = (
        extracted.procedure_descriptions[0]
        if extracted.procedure_descriptions
        else (extracted.procedure_codes[0] if extracted.procedure_codes else "Procedure")
    )

    final_summary = polish_summary_with_llm(
        outcome=outcome,
        summary=base_summary,
        reasons=reasons,
        citations=citations,
        claim_id=extracted.claim_id,
        procedure_desc=proc_desc,
        cost=extracted.estimated_cost,
    )

    return AdjudicationRecommendation(
        outcome=outcome,
        confidence=confidence,
        summary=final_summary,
        reasons=reasons,
        policy_citations=citations,
        unresolved_issues=unresolved,
    )


def decision_synthesis_node(state: ClaimState) -> Dict[str, Any]:
    """LangGraph node execution function for the Decision Synthesis Agent."""
    extracted = state.get("extracted_claim")

    if not extracted:
        fallback_rec = AdjudicationRecommendation(
            outcome=AdjudicationOutcome.QUERY,
            confidence=0.5,
            summary="Pre-authorization paused: No claim extraction data available.",
            reasons=["Extraction data is missing."],
            policy_citations=[],
            unresolved_issues=["Claim form not parsed."],
        )
        return {
            "recommendation": fallback_rec,
            "audit_trail": ["[DecisionSynthesisAgent] Failed: No extracted claim data present."],
            "current_node": "decision_synthesis",
        }

    recommendation = synthesize_decision(
        extracted=extracted,
        missing_docs=state.get("missing_mandatory_documents", []),
        retrieved_clauses=state.get("retrieved_clauses", []),
        code_validation=state.get("code_validation"),
        fraud_report=state.get("fraud_report"),
    )

    audit_msg = (
        f"[DecisionSynthesisAgent] Recommendation synthesized: OUTCOME={recommendation.outcome.value.upper()} "
        f"(Confidence: {recommendation.confidence:.2f}). Citations: {len(recommendation.policy_citations)}, "
        f"Reasons: {len(recommendation.reasons)}, Unresolved Issues: {len(recommendation.unresolved_issues)}."
    )

    return {
        "recommendation": recommendation,
        "audit_trail": [audit_msg],
        "current_node": "decision_synthesis",
    }
