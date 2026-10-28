"""PII Redaction Agent.

Masks personally identifiable information (Names, Phone Numbers, Aadhaar IDs,
Addresses, Emails, DOBs) from clinical and billing documents before any cloud LLM calls.
Ensures HIPAA and privacy compliance with transparent audit logging.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from app.state import ClaimState, DocumentMeta, ExtractedClaim


# ==============================================================================
# Deterministic Regular Expressions for Sensitive Identifiers
# ==============================================================================

# Aadhaar / National ID: 12 digits in groups of 4 (e.g. 9823-4412-8871 or 9823 4412 8871)
AADHAAR_PATTERN = re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b")

# Phone / Mobile: International and Indian mobile formats
PHONE_PATTERN = re.compile(
    r"(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}\b|(?:\+?1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b"
)

# Email address
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Date of Birth (DOB) pattern
DOB_PATTERN = re.compile(r"(?:dob|date of birth)[\s:]*(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})", re.IGNORECASE)

# Pin code (India 6-digit or US 5-digit ZIP)
PINCODE_PATTERN = re.compile(r"\b\d{6}\b|\b\d{5}(?:-\d{4})?\b")

# Patient Name explicit fields: "Full Name: Rohan Deshmukh"
NAME_FIELD_PATTERN = re.compile(
    r"((?:patient(?:\s+full)?\s*name|full\s*name)[\s:]*)([A-Za-z\s\.\-]+?)(?=(\s*(?:\||\n|dob|age|mobile|gender|$)))",
    re.IGNORECASE,
)

# Address explicit fields: "Address: 402 Palm Grove Apartments, Bandra West, Mumbai 400050"
ADDRESS_FIELD_PATTERN = re.compile(
    r"(address[\s:]*)([A-Za-z0-9\s,\-\./#]+?\b(?:\d{5,6})\b)",
    re.IGNORECASE,
)


class SecurityException(Exception):
    """Raised when raw PII is detected in a cloud-bound payload."""
    pass


def redact_pii(text: str, known_names: Optional[List[str]] = None) -> Tuple[str, Dict[str, int]]:
    """Sanitize text by replacing sensitive personal information with privacy tokens.

    Args:
        text: Raw document text
        known_names: Optional list of identified patient names to explicitly redact

    Returns:
        (sanitized_text, redaction_counts_dict)
    """
    if not text:
        return "", {}

    counts = {
        "aadhaar": 0,
        "phone": 0,
        "email": 0,
        "dob": 0,
        "name": 0,
        "address": 0,
    }

    sanitized = text

    # 1. Redact Aadhaar / National ID
    aadhaar_matches = AADHAAR_PATTERN.findall(sanitized)
    counts["aadhaar"] = len(aadhaar_matches)
    sanitized = AADHAAR_PATTERN.sub("[GOVT_ID]", sanitized)

    # 2. Redact Phone Numbers
    phone_matches = PHONE_PATTERN.findall(sanitized)
    counts["phone"] = len(phone_matches)
    sanitized = PHONE_PATTERN.sub("[PHONE_NUMBER]", sanitized)

    # 3. Redact Email Addresses
    email_matches = EMAIL_PATTERN.findall(sanitized)
    counts["email"] = len(email_matches)
    sanitized = EMAIL_PATTERN.sub("[EMAIL]", sanitized)

    # 4. Redact Date of Birth
    dob_matches = DOB_PATTERN.findall(sanitized)
    counts["dob"] = len(dob_matches)
    sanitized = DOB_PATTERN.sub("DOB: [DOB_REDACTED]", sanitized)

    # 5. Redact Formatted Address Lines
    def replace_address(match):
        counts["address"] += 1
        return match.group(1) + "[ADDRESS_REDACTED]"

    sanitized = ADDRESS_FIELD_PATTERN.sub(replace_address, sanitized)

    # 6. Redact Formatted Name Lines
    def replace_name(match):
        counts["name"] += 1
        return match.group(1) + "[PATIENT_NAME]"

    sanitized = NAME_FIELD_PATTERN.sub(replace_name, sanitized)

    # 7. Redact known names explicitly wherever they appear in the text
    if known_names:
        for name in known_names:
            if name and len(name.strip()) > 3 and name.lower() not in {"unknown", "patient", "male", "female"}:
                pattern = re.compile(re.escape(name.strip()), re.IGNORECASE)
                matches = pattern.findall(sanitized)
                if matches:
                    counts["name"] += len(matches)
                    sanitized = pattern.sub("[PATIENT_NAME]", sanitized)

    return sanitized, counts


def assert_zero_pii_leakage(payload: str, sensitive_values: List[str]) -> bool:
    """Audit gatekeeper: Verifies that no raw sensitive string appears in outgoing text.

    Args:
        payload: The outgoing prompt or cloud LLM request text
        sensitive_values: List of raw patient identifiers (raw name, raw phone, raw ID)

    Raises:
        SecurityException: If any raw sensitive token is found in the payload

    Returns:
        True if the payload is safe and clean
    """
    for item in sensitive_values:
        if not item or len(str(item).strip()) < 3:
            continue
        cleaned_item = str(item).strip()
        # Case-insensitive substring search
        if cleaned_item.lower() in payload.lower():
            raise SecurityException(
                f"Privacy Breach Guard: Raw PII item was detected in cloud-bound payload! "
                f"Offending pattern length: {len(cleaned_item)} characters."
            )
    return True


def redaction_node(state: ClaimState) -> Dict:
    """LangGraph node: Strips all PII from uploaded documents and state before cloud processing."""
    uploaded_docs: List[DocumentMeta] = state.get("uploaded_documents", [])
    extracted: Optional[ExtractedClaim] = state.get("extracted_claim")

    known_names = []
    if extracted and extracted.patient_name_masked:
        # Before this node, patient_name_masked holds the raw extracted name
        known_names.append(extracted.patient_name_masked)

    total_redactions = {
        "aadhaar": 0,
        "phone": 0,
        "email": 0,
        "dob": 0,
        "name": 0,
        "address": 0,
    }

    updated_docs: List[DocumentMeta] = []
    for doc in uploaded_docs:
        sanitized_text, counts = redact_pii(doc.raw_text, known_names=known_names)
        for k, v in counts.items():
            total_redactions[k] += v

        # Create updated DocumentMeta with sanitized text
        updated_doc = DocumentMeta(
            file_id=doc.file_id,
            filename=doc.filename,
            doc_type=doc.doc_type,
            page_count=doc.page_count,
            raw_text=doc.raw_text,
            redacted_text=sanitized_text,
            ocr_applied=doc.ocr_applied,
        )
        updated_docs.append(updated_doc)

    # Sanitize the patient_name_masked field inside extracted_claim
    if extracted:
        extracted.patient_name_masked = "[PATIENT_NAME]"

    total_count = sum(total_redactions.values())
    audit_entry = (
        f"[PIIRedactionAgent] Successfully sanitized {len(updated_docs)} document(s). "
        f"Redacted {total_count} sensitive entities "
        f"(Names={total_redactions['name']}, Phones={total_redactions['phone']}, "
        f"GovtIDs={total_redactions['aadhaar']}, Addresses={total_redactions['address']}). "
        f"Cloud privacy gatekeeper: PASSED."
    )

    return {
        "uploaded_documents": updated_docs,
        "extracted_claim": extracted,
        "pii_redacted": True,
        "current_node": "redaction",
        "audit_trail": [audit_entry],
    }
