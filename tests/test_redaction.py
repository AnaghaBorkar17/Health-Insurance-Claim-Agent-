"""Unit and Privacy Tests for PII Redaction Agent.

Validates:
1. Deterministic masking of patient names, phone numbers, Aadhaar/SSN IDs, addresses, and emails
2. Cloud Payload Guardrail: Proves raw PII is blocked from cloud payloads
3. SecurityException triggering upon simulated data leakage
4. Redaction node state transformation
"""

import pytest
from app.agents.redaction import (
    SecurityException,
    assert_zero_pii_leakage,
    redact_pii,
    redaction_node,
)
from app.state import ClaimState, DocumentMeta, DocumentType, ExtractedClaim


def test_redact_phone_numbers():
    raw_texts = [
        "Patient mobile number is +91 98200 12345 for verification.",
        "Emergency contact: 9820054321.",
        "US Provider phone: +1 (555) 234-5678.",
    ]
    for text in raw_texts:
        sanitized, counts = redact_pii(text)
        assert "[PHONE_NUMBER]" in sanitized
        assert "98200" not in sanitized
        assert "234-5678" not in sanitized
        assert counts["phone"] >= 1


def test_redact_aadhaar_and_govt_id():
    text = "Patient identity proof Aadhaar: 9823-4412-8871 submitted with claim."
    sanitized, counts = redact_pii(text)
    assert "[GOVT_ID]" in sanitized
    assert "9823-4412-8871" not in sanitized
    assert counts["aadhaar"] == 1


def test_redact_names_and_addresses():
    text = (
        "Full Name: Rohan Deshmukh | Age: 62 | Gender: Male\n"
        "Address: 402 Palm Grove Apartments, Bandra West, Mumbai 400050\n"
        "Email: rohan.deshmukh@example.com"
    )
    sanitized, counts = redact_pii(text, known_names=["Rohan Deshmukh"])
    assert "[PATIENT_NAME]" in sanitized
    assert "Rohan Deshmukh" not in sanitized
    assert "[ADDRESS_REDACTED]" in sanitized
    assert "Bandra West" not in sanitized
    assert "[EMAIL]" in sanitized
    assert "rohan.deshmukh@example.com" not in sanitized


def test_privacy_guardrail_clean_payload():
    """Proves that a sanitized payload cleanly passes the zero-leakage check."""
    safe_payload = (
        "Patient [PATIENT_NAME], age 62, underwent phacoemulsification (CPT 66984). "
        "Estimated cost: $36,000 at Apex Specialty Eye Institute."
    )
    raw_sensitive_entities = ["Rohan Deshmukh", "98200 12345", "9823-4412-8871"]

    # Must return True without raising exception
    assert assert_zero_pii_leakage(safe_payload, raw_sensitive_entities) is True


def test_privacy_guardrail_catches_breach():
    """Proves that any unredacted raw PII in an outgoing payload triggers SecurityException."""
    compromised_payload = (
        "Evaluating claim for patient Rohan Deshmukh, phone +91 98200 12345."
    )
    raw_sensitive_entities = ["Rohan Deshmukh", "+91 98200 12345", "9823-4412-8871"]

    with pytest.raises(SecurityException) as exc_info:
        assert_zero_pii_leakage(compromised_payload, raw_sensitive_entities)

    assert "Privacy Breach Guard" in str(exc_info.value)


def test_redaction_node_state_execution():
    """Tests the LangGraph redaction node updating state and masking raw fields."""
    raw_doc_text = """PRE-AUTHORIZATION REQUEST
Full Name: Kavita Sharma | Age: 28 | Gender: Female
Mobile: +91 99880 77665
Aadhaar ID: 7712-9901-3344
Address: 88 Defence Colony, New Delhi 110024
Clinical Diagnosis: Aesthetic deformity (ICD-10 Z41.1)
Total Estimated Cost: $120,000
"""
    doc = DocumentMeta(
        file_id="DOC-TEST-003",
        filename="kavita_claim.txt",
        doc_type=DocumentType.CLAIM_FORM,
        raw_text=raw_doc_text,
    )

    extracted = ExtractedClaim(
        claim_id="CLM-2026-003",
        patient_id="PAT-77291",
        patient_name_masked="Kavita Sharma",  # Raw name before redaction node
        patient_age=28,
        patient_gender="Female",
        diagnosis_codes=["Z41.1"],
        procedure_codes=["30410"],
        estimated_cost=120000.0,
    )

    state: ClaimState = {
        "claim_id": "CLM-2026-003",
        "uploaded_documents": [doc],
        "extracted_claim": extracted,
        "pii_redacted": False,
        "audit_trail": [],
    }

    result = redaction_node(state)

    # Assertions on state changes
    assert result["pii_redacted"] is True
    assert result["current_node"] == "redaction"
    assert result["extracted_claim"].patient_name_masked == "[PATIENT_NAME]"

    # Assertions on sanitized document
    sanitized_doc = result["uploaded_documents"][0]
    assert "Kavita Sharma" not in sanitized_doc.redacted_text
    assert "99880 77665" not in sanitized_doc.redacted_text
    assert "7712-9901-3344" not in sanitized_doc.redacted_text
    assert "88 Defence Colony" not in sanitized_doc.redacted_text
    assert "[PATIENT_NAME]" in sanitized_doc.redacted_text
    assert "[PHONE_NUMBER]" in sanitized_doc.redacted_text
    assert "[GOVT_ID]" in sanitized_doc.redacted_text
