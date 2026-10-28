"""Generate sample PDF files using pymupdf for testing extraction."""

from pathlib import Path
import pymupdf

DOCS_DIR = Path(__file__).resolve().parent

SAMPLE_PDF_TEXT = """PRE-AUTHORIZATION REQUEST FORM - HEALTH INSURANCE
Hospital: Apex Specialty Eye Institute
Address: 101 Link Road, Andheri West, Mumbai
Provider Reg No: HOSP-MH-9941

PATIENT DETAILS:
Full Name: Rohan Deshmukh
Age: 62 | Gender: Male | DOB: 1964-08-14
Mobile: +91 98200 12345
Aadhaar ID: 9823-4412-8871
Address: 402 Palm Grove Apartments, Bandra West, Mumbai 400050
Insurance Policy No: CP-POL-55410
Insurer: CarePlus Health Insurance Company

CLINICAL & ADMISSION DETAILS:
Date of Proposed Admission: 2026-11-05
Expected Date of Discharge: 2026-11-05
Expected Length of Stay: 1 day (Daycare Procedure)
Chief Complaints: Progressive painless diminution of vision in Right Eye for 8 months.
Clinical Diagnosis: Senile nuclear cataract, right eye
ICD-10 Diagnosis Code: H25.011
Proposed Procedure: Phacoemulsification with Foldable Posterior Chamber IOL Implantation
CPT / Procedure Code: 66984
Attending Ophthalmologist: Dr. Arvind Natarajan, MS (Ophthalmology), Reg #77412

ESTIMATED FINANCIALS:
Surgeon & OT Charges: $24,000
Intraocular Lens (IOL) Cost: $8,000
Pharmacy & Consumables: $4,000
Total Estimated Cost: $36,000
Authorized Hospital Signatory: Dr. Arvind Natarajan
"""


def create_sample_pdf():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 size in points
    rect = pymupdf.Rect(50, 50, 545, 792)
    page.insert_textbox(rect, SAMPLE_PDF_TEXT, fontsize=11, fontname="helv")
    pdf_path = DOCS_DIR / "CLM-2026-001_claim_form.pdf"
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created sample PDF at: {pdf_path}")


if __name__ == "__main__":
    create_sample_pdf()
