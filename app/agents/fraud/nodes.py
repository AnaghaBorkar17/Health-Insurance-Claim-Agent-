"""Individual Node Implementations for the Fraud & Anomaly Detection Subgraph.

Each node evaluates a distinct risk dimension:
1. duplicate_check_node: Detects identical/overlapping claims in SQLite claims_history.
2. billing_spike_node: Compares estimated costs against standard CPT pricing benchmarks.
3. length_of_stay_node: Checks hospital stay duration against expected recovery baselines.
4. risk_aggregator_node: Computes composite 0-100 risk score and compiles FraudRiskReport.
"""

import logging
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from app.state import FraudRiskReport
from app.tools.database import get_cpt_info, query_duplicate_claims

logger = logging.getLogger(__name__)


class FraudSubgraphState(TypedDict, total=False):
    """Isolated state passed between nodes in the Fraud Subgraph."""

    claim_id: str
    patient_id: str
    admission_date: str
    discharge_date: str
    procedure_codes: List[str]
    diagnosis_codes: List[str]
    estimated_cost: float
    length_of_stay: int

    # Risk signals accumulated across nodes
    is_duplicate: bool
    billing_spike_detected: bool
    stay_anomaly_detected: bool
    risk_points: int
    reasons: List[str]
    benchmark_median_cost: Optional[float]
    benchmark_max_stay: Optional[int]

    # Final compiled artifact
    fraud_report: Optional[FraudRiskReport]


def duplicate_check_node(state: FraudSubgraphState) -> Dict[str, Any]:
    """Check for duplicate pre-authorizations or claims in claims_history."""
    patient_id = state.get("patient_id", "")
    procedure_codes = state.get("procedure_codes", [])
    admission_date = state.get("admission_date", "")
    current_claim_id = state.get("claim_id", "")

    reasons: List[str] = list(state.get("reasons", []))
    risk_points = state.get("risk_points", 0)
    is_duplicate = False

    if patient_id and procedure_codes:
        duplicates = query_duplicate_claims(
            patient_id=patient_id,
            procedure_codes=procedure_codes,
            admission_date=admission_date,
            exclude_claim_id=current_claim_id,
        )

        if duplicates:
            is_duplicate = True
            prior_ids = [d["claim_id"] for d in duplicates]
            risk_points += 50
            reasons.append(
                f"Duplicate claim submission detected: Patient '{patient_id}' has existing historical claim(s) {prior_ids} "
                f"with overlapping procedure codes {procedure_codes}."
            )

    return {
        "is_duplicate": is_duplicate,
        "risk_points": risk_points,
        "reasons": reasons,
    }


def billing_spike_node(state: FraudSubgraphState) -> Dict[str, Any]:
    """Check whether claimed amount significantly exceeds regional benchmarks."""
    procedure_codes = state.get("procedure_codes", [])
    estimated_cost = float(state.get("estimated_cost", 0.0))

    reasons: List[str] = list(state.get("reasons", []))
    risk_points = state.get("risk_points", 0)
    billing_spike_detected = False
    median_benchmark: Optional[float] = None

    for code in procedure_codes:
        cpt_info = get_cpt_info(code)
        if not cpt_info:
            continue

        median = float(cpt_info.get("benchmark_median_cost", 0.0))
        max_cost = float(cpt_info.get("benchmark_max_cost", 0.0))
        if median > 0:
            median_benchmark = median

        if max_cost > 0 and estimated_cost > max_cost:
            ratio = estimated_cost / median if median > 0 else 1.0
            pct_over = ((estimated_cost - median) / median) * 100 if median > 0 else 0.0

            if ratio >= 2.0 or pct_over >= 100:
                billing_spike_detected = True
                risk_points += 35
                reasons.append(
                    f"Significant billing spike: Claimed cost of ${estimated_cost:,.2f} exceeds standard benchmark "
                    f"median (${median:,.2f}) by {pct_over:.1f}% (benchmark max is ${max_cost:,.2f})."
                )
            else:
                risk_points += 15
                reasons.append(
                    f"Elevated billing: Claimed cost of ${estimated_cost:,.2f} exceeds benchmark maximum (${max_cost:,.2f})."
                )

    return {
        "billing_spike_detected": billing_spike_detected,
        "risk_points": risk_points,
        "reasons": reasons,
        "benchmark_median_cost": median_benchmark,
    }


def length_of_stay_node(state: FraudSubgraphState) -> Dict[str, Any]:
    """Evaluate whether the length of stay is abnormal for the clinical procedure."""
    procedure_codes = state.get("procedure_codes", [])
    length_of_stay = int(state.get("length_of_stay", 1) or 1)

    reasons: List[str] = list(state.get("reasons", []))
    risk_points = state.get("risk_points", 0)
    stay_anomaly_detected = False
    expected_stay: Optional[int] = None

    for code in procedure_codes:
        cpt_info = get_cpt_info(code)
        if not cpt_info:
            continue

        baseline_stay = int(cpt_info.get("expected_stay_days", 1) or 1)
        expected_stay = baseline_stay

        if length_of_stay > baseline_stay:
            excess_days = length_of_stay - baseline_stay
            if length_of_stay >= (baseline_stay * 3) or excess_days >= 4:
                stay_anomaly_detected = True
                risk_points += 25
                reasons.append(
                    f"Length of stay anomaly: Claimed stay of {length_of_stay} days is statistically abnormal "
                    f"for procedure {code} (expected recovery duration is {baseline_stay} day(s))."
                )
            else:
                risk_points += 10
                reasons.append(
                    f"Stay duration warning: Claimed stay of {length_of_stay} days exceeds expected {baseline_stay} day(s)."
                )

    return {
        "stay_anomaly_detected": stay_anomaly_detected,
        "risk_points": risk_points,
        "reasons": reasons,
        "benchmark_max_stay": expected_stay,
    }


def risk_aggregator_node(state: FraudSubgraphState) -> Dict[str, Any]:
    """Compile individual risk indicators into a normalized FraudRiskReport."""
    raw_points = state.get("risk_points", 0)
    normalized_score = min(100, max(0, raw_points))

    reasons = state.get("reasons", [])
    if not reasons:
        reasons = ["No fraud flags or billing anomalies detected. All benchmarks within expected clinical bounds."]

    report = FraudRiskReport(
        risk_score=normalized_score,
        is_duplicate=state.get("is_duplicate", False),
        billing_spike_detected=state.get("billing_spike_detected", False),
        stay_anomaly_detected=state.get("stay_anomaly_detected", False),
        reasons=reasons,
        benchmark_median_cost=state.get("benchmark_median_cost"),
        benchmark_max_stay=state.get("benchmark_max_stay"),
    )

    return {
        "risk_score": normalized_score,
        "fraud_report": report,
    }
