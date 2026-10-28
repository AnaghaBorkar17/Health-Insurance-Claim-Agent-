"""Code Validation Agent.

Verifies medical coding integrity:
1. Validates ICD-10 diagnosis codes against standard reference nomenclature.
2. Validates CPT procedure codes against standard surgical & clinical catalogs.
3. Checks patient demographic compatibility (gender & age restrictions).
4. Verifies clinical consistency between diagnosis and requested procedure.
"""

import logging
from typing import Any, Dict, List, Optional

from app.state import (
    ClaimState,
    CodeValidationItem,
    CodeValidationReport,
    ExtractedClaim,
)
from app.tools.database import (
    check_clinical_compatibility,
    get_cpt_info,
    get_icd10_info,
)

logger = logging.getLogger(__name__)


def validate_claim_codes(extracted: ExtractedClaim) -> CodeValidationReport:
    """Validate all ICD-10 and procedure codes for a claim."""
    items: List[CodeValidationItem] = []
    has_clinical_mismatch = False
    consistency_notes: List[str] = []

    # 1. Validate Diagnosis Codes (ICD-10)
    for icd in extracted.diagnosis_codes:
        flags: List[str] = []
        info = get_icd10_info(icd)

        if not info:
            items.append(
                CodeValidationItem(
                    code=icd,
                    code_type="ICD-10",
                    is_valid=False,
                    standard_description=None,
                    flags=["Unrecognized ICD-10 diagnosis code"],
                )
            )
            consistency_notes.append(f"Diagnosis code '{icd}' not found in standard reference catalog.")
            continue

        desc = info["description"]
        gender_restr = info.get("gender_restriction", "None")
        min_age = info.get("min_age", 0)
        max_age = info.get("max_age", 120)

        # Check demographic rules
        if extracted.patient_gender and gender_restr and gender_restr.lower() != "none":
            if extracted.patient_gender.strip().lower() != gender_restr.strip().lower():
                flags.append(
                    f"Gender mismatch: Diagnosis '{icd}' is restricted to {gender_restr} patients, but patient is recorded as {extracted.patient_gender}."
                )

        if extracted.patient_age is not None:
            if extracted.patient_age < min_age or extracted.patient_age > max_age:
                flags.append(
                    f"Age constraint warning: Diagnosis '{icd}' typical age range is {min_age}-{max_age} years (patient is {extracted.patient_age})."
                )

        items.append(
            CodeValidationItem(
                code=icd,
                code_type="ICD-10",
                is_valid=True,
                standard_description=desc,
                flags=flags,
            )
        )

    # 2. Validate Procedure Codes (CPT)
    for cpt in extracted.procedure_codes:
        flags: List[str] = []
        cpt_info = get_cpt_info(cpt)

        if not cpt_info:
            items.append(
                CodeValidationItem(
                    code=cpt,
                    code_type="Procedure",
                    is_valid=False,
                    standard_description=None,
                    flags=["Unrecognized CPT procedure code"],
                )
            )
            consistency_notes.append(f"Procedure code '{cpt}' not found in standard reference catalog.")
            continue

        cpt_desc = cpt_info["description"]

        # 3. Check Clinical Consistency against diagnoses
        if extracted.diagnosis_codes:
            matched_diag = False
            for icd in extracted.diagnosis_codes:
                is_compat, note = check_clinical_compatibility(icd, cpt)
                if is_compat:
                    matched_diag = True
                    break

            if not matched_diag:
                mismatch_msg = (
                    f"Clinical Inconsistency: Procedure {cpt} ({cpt_desc}) is not clinically indicated for any of the listed diagnoses: {extracted.diagnosis_codes}."
                )
                flags.append(mismatch_msg)
                consistency_notes.append(mismatch_msg)
                has_clinical_mismatch = True

        items.append(
            CodeValidationItem(
                code=cpt,
                code_type="Procedure",
                is_valid=True,
                standard_description=cpt_desc,
                flags=flags,
            )
        )

    all_valid = (
        all(item.is_valid for item in items)
        and not has_clinical_mismatch
        and not any("Gender mismatch" in f for item in items for f in item.flags)
    )

    if not consistency_notes:
        overall_note = "All diagnosis and procedure codes are verified, clinically consistent, and aligned with standard medical terminology."
    else:
        overall_note = " ; ".join(consistency_notes)

    return CodeValidationReport(
        all_codes_valid=all_valid,
        items=items,
        clinical_consistency_note=overall_note,
    )


def code_validation_node(state: ClaimState) -> Dict[str, Any]:
    """LangGraph node execution function for the Code Validation Agent."""
    extracted = state.get("extracted_claim")

    if not extracted:
        empty_report = CodeValidationReport(
            all_codes_valid=False,
            items=[],
            clinical_consistency_note="No extracted claim data available for code validation.",
        )
        return {
            "code_validation": empty_report,
            "audit_trail": ["[CodeValidationAgent] Failed: No extracted claim data present."],
            "current_node": "code_validation",
        }

    report = validate_claim_codes(extracted)

    flag_count = sum(len(item.flags) for item in report.items)
    status_str = "PASSED" if report.all_codes_valid else "FAILED / FLAGGED"
    audit_msg = (
        f"[CodeValidationAgent] Validation {status_str}: {len(report.items)} codes evaluated. "
        f"All Valid: {report.all_codes_valid}, Clinical Flags: {flag_count}. "
        f"Summary: {report.clinical_consistency_note[:120]}"
    )

    return {
        "code_validation": report,
        "audit_trail": [audit_msg],
        "current_node": "code_validation",
    }
