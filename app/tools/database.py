"""SQLite Relational Database & Reference Table Interface.

Manages SQLite storage for:
1. ICD-10 Diagnosis Reference Table
2. CPT / Procedure Reference Table with Cost Benchmarks
3. Historical Claims Repository (for duplicate & anomaly detection)
"""

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_tables"
ICD10_CSV = REFERENCE_DIR / "icd10_codes.csv"
CPT_CSV = REFERENCE_DIR / "procedure_codes.csv"


def get_db_path(db_path: Optional[str] = None) -> Path:
    """Resolve database path from argument or application settings."""
    if db_path:
        path = Path(db_path)
    else:
        path = Path(settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and configure a SQLite connection with Row factory."""
    resolved_path = get_db_path(db_path)
    conn = sqlite3.connect(str(resolved_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema if tables do not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS icd10_reference (
            code TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            category TEXT,
            gender_restriction TEXT DEFAULT 'None',
            min_age INTEGER DEFAULT 0,
            max_age INTEGER DEFAULT 120
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cpt_reference (
            code TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            category TEXT,
            benchmark_min_cost REAL,
            benchmark_median_cost REAL,
            benchmark_max_cost REAL,
            expected_stay_days INTEGER DEFAULT 1,
            compatible_icd10 TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims_history (
            claim_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            admission_date TEXT,
            discharge_date TEXT,
            diagnosis_codes TEXT,
            procedure_codes TEXT,
            estimated_cost REAL,
            length_of_stay INTEGER,
            status TEXT,
            created_at TEXT
        );
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully at %s", get_db_path(db_path))


def load_reference_tables(db_path: Optional[str] = None, force_reload: bool = False) -> None:
    """Load ICD-10 and CPT reference CSVs into SQLite tables."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Load ICD-10
    cursor.execute("SELECT COUNT(*) FROM icd10_reference")
    icd_count = cursor.fetchone()[0]
    if icd_count == 0 or force_reload:
        if force_reload:
            cursor.execute("DELETE FROM icd10_reference")
        if ICD10_CSV.exists():
            with open(ICD10_CSV, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cursor.execute("""
                        INSERT OR REPLACE INTO icd10_reference 
                        (code, description, category, gender_restriction, min_age, max_age)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row["code"].strip(),
                        row["description"].strip(),
                        row.get("category", "").strip(),
                        row.get("gender_restriction", "None").strip(),
                        int(row.get("min_age", 0) or 0),
                        int(row.get("max_age", 120) or 120),
                    ))
            logger.info("Loaded ICD-10 codes from %s", ICD10_CSV)

    # Load CPT Procedure Codes
    cursor.execute("SELECT COUNT(*) FROM cpt_reference")
    cpt_count = cursor.fetchone()[0]
    if cpt_count == 0 or force_reload:
        if force_reload:
            cursor.execute("DELETE FROM cpt_reference")
        if CPT_CSV.exists():
            with open(CPT_CSV, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cursor.execute("""
                        INSERT OR REPLACE INTO cpt_reference 
                        (code, description, category, benchmark_min_cost, benchmark_median_cost, benchmark_max_cost, expected_stay_days, compatible_icd10)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["code"].strip(),
                        row["description"].strip(),
                        row.get("category", "").strip(),
                        float(row.get("benchmark_min_cost", 0.0) or 0.0),
                        float(row.get("benchmark_median_cost", 0.0) or 0.0),
                        float(row.get("benchmark_max_cost", 0.0) or 0.0),
                        int(row.get("expected_stay_days", 1) or 1),
                        row.get("compatible_icd10", "*").strip(),
                    ))
            logger.info("Loaded CPT codes from %s", CPT_CSV)

    # Seed baseline claim history (e.g. Case 001 historical record for duplicate detection)
    cursor.execute("SELECT COUNT(*) FROM claims_history")
    history_count = cursor.fetchone()[0]
    if history_count == 0:
        cursor.execute("""
            INSERT OR IGNORE INTO claims_history
            (claim_id, patient_id, admission_date, discharge_date, diagnosis_codes, procedure_codes, estimated_cost, length_of_stay, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "CLM-2026-001",
            "PAT-98421",
            "2026-11-05",
            "2026-11-05",
            "H25.011",
            "66984",
            36000.0,
            1,
            "Approved",
            "2026-11-01T10:00:00Z"
        ))

    conn.commit()
    conn.close()


def get_icd10_info(code: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up an ICD-10 code in the reference database."""
    load_reference_tables(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cleaned_code = code.strip().upper()
    # Try exact match first
    cursor.execute("SELECT * FROM icd10_reference WHERE UPPER(code) = ?", (cleaned_code,))
    row = cursor.fetchone()

    # Try matching without dots (e.g., H25011 -> H25.011)
    if not row and "." not in cleaned_code and len(cleaned_code) > 3:
        formatted = f"{cleaned_code[:3]}.{cleaned_code[3:]}"
        cursor.execute("SELECT * FROM icd10_reference WHERE UPPER(code) = ?", (formatted,))
        row = cursor.fetchone()

    conn.close()
    if row:
        return dict(row)
    return None


def get_cpt_info(code: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up a CPT procedure code in the reference database."""
    load_reference_tables(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cleaned_code = code.strip()
    cursor.execute("SELECT * FROM cpt_reference WHERE code = ?", (cleaned_code,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def check_clinical_compatibility(
    icd_code: str, cpt_code: str, db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """Check clinical indication consistency between diagnosis and procedure codes."""
    cpt_info = get_cpt_info(cpt_code, db_path)
    if not cpt_info:
        return False, f"Procedure code '{cpt_code}' not found in standard reference catalog."

    compatible_list = cpt_info.get("compatible_icd10", "*")
    if compatible_list == "*":
        return True, "Procedure is generic/compatible with all diagnoses."

    allowed_prefixes = [item.strip().upper() for item in compatible_list.split(",")]
    icd_clean = icd_code.strip().upper()

    for prefix in allowed_prefixes:
        # Match exact code or diagnosis prefix (e.g., prefix "H25" matches "H25.011")
        if icd_clean == prefix or icd_clean.startswith(prefix) or prefix.startswith(icd_clean.split(".")[0]):
            return True, f"Procedure {cpt_code} ({cpt_info['description']}) is clinically indicated for {icd_code}."

    return (
        False,
        f"Clinical Mismatch: Procedure {cpt_code} ({cpt_info['description']}) is typically indicated for [{compatible_list}], but diagnosis is {icd_code}."
    )


def insert_claim_history(
    claim_id: str,
    patient_id: str,
    admission_date: str,
    discharge_date: str,
    diagnosis_codes: List[str],
    procedure_codes: List[str],
    estimated_cost: float,
    length_of_stay: int,
    status: str,
    db_path: Optional[str] = None,
) -> None:
    """Store adjudicated or processed claim into SQLite history."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO claims_history
        (claim_id, patient_id, admission_date, discharge_date, diagnosis_codes, procedure_codes, estimated_cost, length_of_stay, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        claim_id,
        patient_id,
        admission_date,
        discharge_date,
        ",".join(diagnosis_codes),
        ",".join(procedure_codes),
        estimated_cost,
        length_of_stay,
        status,
    ))

    conn.commit()
    conn.close()


def query_duplicate_claims(
    patient_id: str,
    procedure_codes: List[str],
    admission_date: str,
    exclude_claim_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for matching prior claims for the same patient and procedure."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    query = """
        SELECT * FROM claims_history
        WHERE patient_id = ?
    """
    params: List[Any] = [patient_id]

    if exclude_claim_id:
        query += " AND claim_id != ?"
        params.append(exclude_claim_id)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    duplicates = []
    for row in rows:
        row_dict = dict(row)
        hist_procedures = [p.strip() for p in (row_dict.get("procedure_codes") or "").split(",") if p.strip()]
        # Check if any procedure overlaps
        common_proc = set(procedure_codes).intersection(set(hist_procedures))
        if common_proc:
            duplicates.append(row_dict)

    return duplicates
