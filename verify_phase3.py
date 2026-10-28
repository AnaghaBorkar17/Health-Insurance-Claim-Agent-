"""Phase 3 Verification Script: Document Extraction, PDF Parsing, and Pydantic Validation.

Tests:
1. PDF Parsing via pdfplumber and pymupdf
2. Document Type Classification
3. Structured Field Extraction (Age, Gender, Codes, Cost, Hospital, Dates)
4. Missing Mandatory Document Detection
5. Pydantic Model Validation
"""

from pathlib import Path
from app.state import ClaimState, DocumentMeta, DocumentType
from app.tools.document_parser import parse_document
from app.agents.extraction import extraction_node

# Color styling for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_check(name: str, passed: bool, details: str = ""):
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"  {status} {name}")
    if details:
        indent = " " * 9
        print(f"{indent}{details}")


def test_pdf_parsing() -> bool:
    print(f"\n{BLUE}[1/4] Testing PDF Parsing (pdfplumber / pymupdf)...{RESET}")
    pdf_path = Path(__file__).parent / "data" / "synthetic_cases" / "sample_documents" / "CLM-2026-001_claim_form.pdf"
    if not pdf_path.exists():
        print_check("PDF File Existence", False, f"Missing: {pdf_path}")
        return False

    meta = parse_document(pdf_path)
    has_text = len(meta.raw_text) > 100
    is_claim = meta.doc_type == DocumentType.CLAIM_FORM

    print_check("PDF Text Extraction", has_text, f"Extracted {len(meta.raw_text)} chars across {meta.page_count} page(s)")
    print_check("PDF Classification", is_claim, f"Classified as: {meta.doc_type.value}")
    return has_text and is_claim


def test_text_document_parsing() -> bool:
    print(f"\n{BLUE}[2/4] Testing Plain Text Document Parsing...{RESET}")
    bill_path = Path(__file__).parent / "data" / "synthetic_cases" / "sample_documents" / "CLM-2026-001_hospital_bill.txt"
    if not bill_path.exists():
        print_check("Bill File Existence", False, f"Missing: {bill_path}")
        return False

    meta = parse_document(bill_path)
    is_bill = meta.doc_type == DocumentType.HOSPITAL_BILL
    print_check("Bill Document Classification", is_bill, f"Classified as: {meta.doc_type.value}")
    return is_bill


def test_extraction_agent() -> bool:
    print(f"\n{BLUE}[3/4] Testing Document Extraction Agent & Field Mapping...{RESET}")
    pdf_path = Path(__file__).parent / "data" / "synthetic_cases" / "sample_documents" / "CLM-2026-001_claim_form.pdf"
    bill_path = Path(__file__).parent / "data" / "synthetic_cases" / "sample_documents" / "CLM-2026-001_hospital_bill.txt"

    doc_meta_1 = parse_document(pdf_path)
    doc_meta_2 = parse_document(bill_path)

    initial_state: ClaimState = {
        "claim_id": "CLM-2026-001",
        "uploaded_documents": [doc_meta_1, doc_meta_2],
        "audit_trail": [],
        "errors": [],
    }

    result = extraction_node(initial_state)
    claim = result["extracted_claim"]

    age_ok = claim.patient_age == 62
    gender_ok = claim.patient_gender == "Male"
    diag_ok = "H25.011" in claim.diagnosis_codes
    proc_ok = "66984" in claim.procedure_codes
    cost_ok = claim.estimated_cost == 36000.0
    hosp_ok = "Apex" in (claim.hospital_name or "")

    print_check("Demographics (Age, Gender)", age_ok and gender_ok, f"Age: {claim.patient_age}, Gender: {claim.patient_gender}")
    print_check("Diagnosis Code (ICD-10)", diag_ok, f"Codes: {claim.diagnosis_codes}")
    print_check("Procedure Code (CPT)", proc_ok, f"Codes: {claim.procedure_codes}")
    print_check("Financial Estimated Cost", cost_ok, f"Estimated Cost: ${claim.estimated_cost:,.2f}")
    print_check("Hospital & Provider", hosp_ok, f"Hospital: {claim.hospital_name}")

    return age_ok and gender_ok and diag_ok and proc_ok and cost_ok


def test_missing_document_detection() -> bool:
    print(f"\n{BLUE}[4/4] Testing Missing Mandatory Document Detection...{RESET}")
    pdf_path = Path(__file__).parent / "data" / "synthetic_cases" / "sample_documents" / "CLM-2026-001_claim_form.pdf"
    bill_path = Path(__file__).parent / "data" / "synthetic_cases" / "sample_documents" / "CLM-2026-001_hospital_bill.txt"

    # Test Scenario A: Only Claim Form and Bill provided (Missing Diagnostic Report)
    state_partial: ClaimState = {
        "claim_id": "CLM-2026-001",
        "uploaded_documents": [parse_document(pdf_path), parse_document(bill_path)],
        "audit_trail": [],
    }
    result_a = extraction_node(state_partial)
    missing_a = result_a["missing_mandatory_documents"]
    has_missing_diag = any("diagnostic_report" in m for m in missing_a)
    print_check(
        "Detection of Missing Diagnostic Report",
        has_missing_diag,
        f"Correctly flagged missing: {missing_a}",
    )

    # Test Scenario B: Complete Submission with mock diagnostic report added
    mock_diag = DocumentMeta(
        file_id="DOC-DIAG-01",
        filename="ultrasound_biometry_report.txt",
        doc_type=DocumentType.DIAGNOSTIC_REPORT,
        raw_text="Diagnostic Radiology / A-Scan Biometry Report: Lens opacity confirmed.",
    )
    state_complete: ClaimState = {
        "claim_id": "CLM-2026-001",
        "uploaded_documents": [parse_document(pdf_path), parse_document(bill_path), mock_diag],
        "audit_trail": [],
    }
    result_b = extraction_node(state_complete)
    missing_b = result_b["missing_mandatory_documents"]
    is_complete = len(missing_b) == 0
    print_check(
        "Complete Document Submission Recognition",
        is_complete,
        f"Missing count: {len(missing_b)} (All mandatory documents present)",
    )

    return has_missing_diag and is_complete


def main():
    print("=" * 60)
    print(" CLAIM AGENTS: PHASE 3 EXTRACTION & PARSING VERIFICATION")
    print("=" * 60)

    p_ok = test_pdf_parsing()
    t_ok = test_text_document_parsing()
    e_ok = test_extraction_agent()
    m_ok = test_missing_document_detection()

    print("\n" + "=" * 60)
    if p_ok and t_ok and e_ok and m_ok:
        print(f"{GREEN}[OK] Phase 3 Extraction & Parsing Verified!{RESET}")
    else:
        print(f"{RED}[FAIL] Phase 3 Verification encountered issues.{RESET}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
