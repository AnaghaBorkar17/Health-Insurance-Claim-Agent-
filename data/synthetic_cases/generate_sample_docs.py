"""Script to generate realistic synthetic document text files for testing."""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "sample_documents"
DOCS_DIR.mkdir(exist_ok=True)


# Case 1: Cataract (Approval Scenario)
CASE_1_CLAIM_FORM = """PRE-AUTHORIZATION REQUEST FORM - HEALTH INSURANCE
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
Room & Nursing Charges: $0 (Daycare)
Surgeon & OT Charges: $24,000
Intraocular Lens (IOL) Cost: $8,000
Pharmacy & Consumables: $4,000
Total Estimated Cost: $36,000
Authorized Hospital Signatory: Dr. Arvind Natarajan (Signature & Seal)
"""

CASE_1_BILL = """APEX SPECIALTY EYE INSTITUTE - PRO-FORMA ESTIMATE
Patient Name: Rohan Deshmukh | Age: 62 | Gender: Male
Estimate No: EST-2026-8812 | Date: 2026-11-01
Policy No: CP-POL-55410

ITEMIZED CHARGES:
1. Operation Theatre & Anesthesia Charges: $14,000
2. Surgeon Professional Fees (Phacoemulsification): $10,000
3. Monofocal Foldable Hydrophobic IOL: $8,000
4. Viscoelastics, Blades, Sutures & Disposables: $2,500
5. Post-operative Eye Drops & Antibiotics: $1,500
TOTAL ESTIMATED AMOUNT: $36,000
"""

# Case 3: Cosmetic Surgery (Exclusion Scenario)
CASE_3_CLAIM_FORM = """PRE-AUTHORIZATION REQUEST FORM - HEALTH INSURANCE
Hospital: Aura Aesthetic Plastic Surgery Center
Address: 88 Defence Colony, New Delhi 110024

PATIENT DETAILS:
Full Name: Kavita Sharma
Age: 28 | Gender: Female | DOB: 1998-03-22
Mobile: +91 99880 77665
Aadhaar ID: 7712-9901-3344
Address: 88 Defence Colony, New Delhi 110024
Insurance Policy No: CP-POL-44120

CLINICAL & ADMISSION DETAILS:
Date of Proposed Admission: 2026-11-15
Expected Date of Discharge: 2026-11-16
Chief Complaints: Patient desires aesthetic contouring of nasal bridge and dorsal hump reduction.
Clinical Diagnosis: Aesthetic nasal deformity
ICD-10 Diagnosis Code: Z41.1 (Encounter for cosmetic surgery)
Proposed Procedure: Open Rhinoplasty with dorsal hump reduction and tip refinement
CPT / Procedure Code: 30410
Attending Surgeon: Dr. Rajiv Mehra, MCh (Plastic Surgery)
History of Trauma: None reported. Elective aesthetic enhancement.
Total Estimated Cost: $120,000
"""


def main():
    (DOCS_DIR / "CLM-2026-001_claim_form.txt").write_text(CASE_1_CLAIM_FORM, encoding="utf-8")
    (DOCS_DIR / "CLM-2026-001_hospital_bill.txt").write_text(CASE_1_BILL, encoding="utf-8")
    (DOCS_DIR / "CLM-2026-003_claim_form.txt").write_text(CASE_3_CLAIM_FORM, encoding="utf-8")
    print(f"Generated synthetic documents in: {DOCS_DIR}")


if __name__ == "__main__":
    main()
