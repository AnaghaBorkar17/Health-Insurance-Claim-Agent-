"""Document Extraction Agent.

Extracts structured clinical and financial claim fields from parsed documents
using regex heuristics, pattern matching, and Pydantic validation.
Detects missing mandatory documents according to policy submission standards.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from app.state import ClaimState, DocumentMeta, DocumentType, ExtractedClaim


# ==============================================================================
# Regex Patterns for Healthcare Claim Fields
# ==============================================================================

# ICD-10 pattern: Letter (not U), 2 digits, optional dot and 1-4 alphanumeric characters
ICD10_PATTERN = re.compile(r"\b([A-TV-Z][0-9][0-9A-B](?:\.[0-9A-TV-Z]{1,4})?)\b", re.IGNORECASE)

# CPT Procedure Code pattern: 5-digit number or 4 digits followed by letter
CPT_PATTERN = re.compile(r"\b([0-9]{4}[0-9A-Z])\b")

# Date pattern: YYYY-MM-DD or DD/MM/YYYY
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Currency / Cost pattern: $36,000 or INR 36,000 or 36000.00
COST_PATTERN = re.compile(r"(?:total estimated (?:amount|cost)|total amount|estimated cost)[\s:]*[\$₹INR\s]*([0-9,]+(?:\.[0-9]{2})?)", re.IGNORECASE)

# Policy Number pattern: CP-POL-12345 or similar
POLICY_PATTERN = re.compile(r"(?:policy\s*(?:no|number)|insurance policy\s*no)[\s:]*([A-Z0-9\-]+)", re.IGNORECASE)

# Patient Identifier pattern: PAT-12345 or similar
PATIENT_ID_PATTERN = re.compile(r"(?:patient\s*(?:id|ref|no)|member\s*id)[\s:]*([A-Za-z0-9\-]+)", re.IGNORECASE)

# Patient Name pattern
NAME_PATTERN = re.compile(r"(?:patient(?:\s+full)?\s*name|full\s*name)[\s:]*([A-Za-z\s\.\-]+?)(?:\s*(?:\||\n|dob|age))", re.IGNORECASE)

# Age and Gender pattern
AGE_PATTERN = re.compile(r"\bage[\s:]*(\d{1,3})\b", re.IGNORECASE)
GENDER_PATTERN = re.compile(r"\bgender[\s:]*(male|female|other)\b", re.IGNORECASE)

# Hospital Name pattern
HOSPITAL_PATTERN = re.compile(r"(?:hospital|facility|institute)[\s:]*([A-Za-z\s]+(?:Hospital|Institute|Clinic|Center|Centre))", re.IGNORECASE)


# Mandatory document types required for pre-authorization
MANDATORY_DOC_TYPES = {
    DocumentType.CLAIM_FORM: "Pre-Authorization Claim Form",
    DocumentType.HOSPITAL_BILL: "Itemized Hospital Bill / Cost Estimate",
    DocumentType.DIAGNOSTIC_REPORT: "Diagnostic / Radiology Investigation Report",
}


def extract_field_by_regex(pattern: re.Pattern, text: str, default: Optional[str] = None) -> Optional[str]:
    """Helper to extract a single string match."""
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return default


def extract_diagnosis_codes(text: str) -> Tuple[List[str], List[str]]:
    """Extract ICD-10 codes and associated text lines."""
    codes = []
    descriptions = []

    # Look for explicit diagnosis lines first
    for line in text.splitlines():
        if any(k in line.lower() for k in ["diagnosis", "icd-10", "clinical diagnosis"]):
            found = ICD10_PATTERN.findall(line)
            for c in found:
                upper_code = c.upper()
                if upper_code not in codes:
                    codes.append(upper_code)
                    # Use cleaned line as description
                    clean_desc = re.sub(r"(?:clinical\s*)?diagnosis[\s:]*|icd-10[\s:]*code[\s:]*", "", line, flags=re.IGNORECASE).strip()
                    descriptions.append(clean_desc or upper_code)

    # General fallback scan if not found in specific lines
    if not codes:
        all_matches = ICD10_PATTERN.findall(text)
        for m in all_matches:
            upper_code = m.upper()
            if upper_code not in codes:
                codes.append(upper_code)
                descriptions.append(f"Diagnosis code {upper_code}")

    return codes, descriptions


def extract_procedure_codes(text: str) -> Tuple[List[str], List[str]]:
    """Extract CPT procedure codes and associated text descriptions."""
    codes = []
    descriptions = []

    for line in text.splitlines():
        if any(k in line.lower() for k in ["procedure", "cpt", "surgical treatment"]):
            found = CPT_PATTERN.findall(line)
            for c in found:
                if c not in codes:
                    codes.append(c)
                    clean_desc = re.sub(r"(?:proposed\s*)?procedure[\s:]*|cpt[\s:]*code[\s:]*", "", line, flags=re.IGNORECASE).strip()
                    descriptions.append(clean_desc or f"Procedure {c}")

    return codes, descriptions


def extract_claim_cost(text: str) -> float:
    """Extract total estimated financial cost."""
    match = COST_PATTERN.search(text)
    if match:
        raw_num = match.group(1).replace(",", "")
        try:
            return float(raw_num)
        except ValueError:
            pass

    # Generic search for currency lines
    for line in text.splitlines():
        if "total" in line.lower() and ("amount" in line.lower() or "cost" in line.lower()):
            numbers = re.findall(r"[\$₹]?\s*([0-9,]+(?:\.[0-9]{2})?)", line)
            if numbers:
                try:
                    return float(numbers[-1].replace(",", ""))
                except ValueError:
                    continue
    return 0.0


def check_missing_documents(uploaded_docs: List[DocumentMeta]) -> List[str]:
    """Identify which mandatory pre-authorization document types are absent."""
    present_types = {doc.doc_type for doc in uploaded_docs}
    missing = []

    for doc_type, label in MANDATORY_DOC_TYPES.items():
        if doc_type not in present_types:
            missing.append(f"{label} ({doc_type.value})")

    return missing


def extraction_node(state: ClaimState) -> Dict:
    """LangGraph node: Ingests documents, extracts structured claim, and detects missing files."""
    uploaded_docs: List[DocumentMeta] = state.get("uploaded_documents", [])
    claim_id: str = state.get("claim_id", f"CLM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")

    # Combine text from all uploaded documents for holistic extraction
    combined_text = "\n\n".join([doc.raw_text for doc in uploaded_docs])

    # Extract demographic & administrative fields
    name = extract_field_by_regex(NAME_PATTERN, combined_text, default="Unknown Patient")
    age_str = extract_field_by_regex(AGE_PATTERN, combined_text)
    age = int(age_str) if age_str and age_str.isdigit() else None
    gender = extract_field_by_regex(GENDER_PATTERN, combined_text)
    policy_no = extract_field_by_regex(POLICY_PATTERN, combined_text, default="UNKNOWN-POLICY")
    hospital = extract_field_by_regex(HOSPITAL_PATTERN, combined_text, default="Unknown Hospital")

    # Dates & Length of stay
    dates = DATE_PATTERN.findall(combined_text)
    admission_date = dates[0] if dates else datetime.utcnow().strftime("%Y-%m-%d")
    discharge_date = dates[1] if len(dates) > 1 else admission_date

    # Calculate length of stay
    try:
        d1 = datetime.strptime(admission_date, "%Y-%m-%d")
        d2 = datetime.strptime(discharge_date, "%Y-%m-%d")
        length_of_stay = max(1, (d2 - d1).days or 1)
    except Exception:
        length_of_stay = 1

    # Extract diagnosis and procedure codes
    diag_codes, diag_descs = extract_diagnosis_codes(combined_text)
    proc_codes, proc_descs = extract_procedure_codes(combined_text)

    # Extract estimated financial amount
    estimated_cost = extract_claim_cost(combined_text)

    # Patient ID extraction with deterministic hash fallback
    import hashlib
    extracted_pid = extract_field_by_regex(PATIENT_ID_PATTERN, combined_text)
    if not extracted_pid:
        extracted_pid = f"PAT-{int(hashlib.md5((name or 'unknown').lower().strip().encode()).hexdigest(), 16) % 100000:05d}"

    # Construct validated ExtractedClaim Pydantic object
    extracted_claim = ExtractedClaim(
        claim_id=claim_id,
        patient_id=extracted_pid,
        patient_name_masked=name,  # Raw name here; Redaction Agent will mask it in next node
        patient_age=age,
        patient_gender=gender.capitalize() if gender else None,
        policy_number=policy_no,
        hospital_name=hospital,
        admission_date=admission_date,
        discharge_date=discharge_date,
        length_of_stay=length_of_stay,
        diagnosis_codes=diag_codes,
        diagnosis_descriptions=diag_descs,
        procedure_codes=proc_codes,
        procedure_descriptions=proc_descs,
        estimated_cost=estimated_cost,
    )

    # Missing mandatory documents checklist
    missing_docs = check_missing_documents(uploaded_docs)

    audit_entry = (
        f"[ExtractionAgent] Extracted claim {claim_id}: "
        f"Age={age}, Gender={gender}, Policy={policy_no}, Cost=${estimated_cost:,.2f}, "
        f"Diagnoses={diag_codes}, Procedures={proc_codes}. "
        f"Missing documents count: {len(missing_docs)}"
    )

    return {
        "extracted_claim": extracted_claim,
        "missing_mandatory_documents": missing_docs,
        "current_node": "extraction",
        "audit_trail": [audit_entry],
    }
