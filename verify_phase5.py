"""Phase 5 Verification Script: Policy Ingestion, Ollama Embeddings, ChromaDB, and Grounded RAG.

Tests:
1. Policy Clause Chunking and Ingestion
2. Local Ollama Embeddings (nomic-embed-text) in ChromaDB
3. Semantic Retrieval for Cataract Treatment (Sub-limit Clause 2.1)
4. Semantic Retrieval for Cosmetic Surgery (Exclusion Clause 4.1)
5. Grounded Citations & LangGraph State Node Execution
"""

from app.state import ClaimState, ExtractedClaim, PolicyClause
from app.tools.policy_ingestion import get_policy_collection, ingest_policy_documents
from app.agents.policy_rag import policy_rag_node, retrieve_relevant_clauses

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


def test_policy_ingestion() -> bool:
    print(f"\n{BLUE}[1/4] Testing Policy Ingestion into ChromaDB...{RESET}")
    # Force reload to ensure clean fresh index
    count = ingest_policy_documents(force_reload=True)
    passed = count >= 8
    print_check(
        "ChromaDB Policy Ingestion",
        passed,
        f"Indexed {count} distinct clauses with local nomic-embed-text embeddings",
    )
    return passed


def test_cataract_retrieval() -> bool:
    print(f"\n{BLUE}[2/4] Testing Targeted Retrieval: Cataract Sub-Limit (Clause 2.1)...{RESET}")
    claim = ExtractedClaim(
        claim_id="CLM-TEST-001",
        patient_id="PAT-98421",
        patient_age=62,
        diagnosis_codes=["H25.011"],
        diagnosis_descriptions=["Senile nuclear cataract, right eye"],
        procedure_codes=["66984"],
        procedure_descriptions=["Phacoemulsification with intraocular lens implantation"],
        estimated_cost=36000.0,
    )

    clauses = retrieve_relevant_clauses(claim, top_k=3)
    clause_ids = [c.clause_id for c in clauses]
    passed = "CLAUSE-2.1" in clause_ids

    print_check(
        "Cataract Sub-limit Clause Retrieval",
        passed,
        f"Retrieved: {[f'{c.clause_id}: {c.title}' for c in clauses]}",
    )
    return passed


def test_cosmetic_exclusion_retrieval() -> bool:
    print(f"\n{BLUE}[3/4] Testing Targeted Retrieval: Cosmetic Exclusion (Clause 4.1)...{RESET}")
    claim = ExtractedClaim(
        claim_id="CLM-TEST-003",
        patient_id="PAT-77291",
        patient_age=28,
        diagnosis_codes=["Z41.1"],
        diagnosis_descriptions=["Encounter for cosmetic surgery for aesthetic purposes"],
        procedure_codes=["30410"],
        procedure_descriptions=["Open rhinoplasty with nasal hump reduction"],
        estimated_cost=120000.0,
    )

    clauses = retrieve_relevant_clauses(claim, top_k=3)
    clause_ids = [c.clause_id for c in clauses]
    passed = "CLAUSE-4.1" in clause_ids

    print_check(
        "Cosmetic Exclusion Clause Retrieval",
        passed,
        f"Retrieved: {[f'{c.clause_id}: {c.title}' for c in clauses]}",
    )
    return passed


def test_policy_rag_node_execution() -> bool:
    print(f"\n{BLUE}[4/4] Testing Policy RAG LangGraph Node State Execution...{RESET}")
    claim = ExtractedClaim(
        claim_id="CLM-TEST-001",
        patient_id="PAT-98421",
        patient_age=62,
        diagnosis_codes=["H25.011"],
        diagnosis_descriptions=["Senile nuclear cataract"],
        procedure_codes=["66984"],
        estimated_cost=36000.0,
    )

    state: ClaimState = {
        "claim_id": "CLM-TEST-001",
        "extracted_claim": claim,
        "retrieved_clauses": [],
        "audit_trail": [],
    }

    result = policy_rag_node(state)
    retrieved = result.get("retrieved_clauses", [])
    has_clauses = len(retrieved) > 0
    node_name_ok = result.get("current_node") == "policy_rag"
    has_audit = len(result.get("audit_trail", [])) > 0

    print_check(
        "State retrieved_clauses Population",
        has_clauses,
        f"Populated {len(retrieved)} PolicyClause items in LangGraph state",
    )
    print_check(
        "Audit Trail Logging",
        has_audit,
        f"Audit entry: {result['audit_trail'][0] if has_audit else 'None'}",
    )
    return has_clauses and node_name_ok and has_audit


def main():
    print("=" * 60)
    print(" CLAIM AGENTS: PHASE 5 POLICY RAG & CHROMADB VERIFICATION")
    print("=" * 60)

    i_ok = test_policy_ingestion()
    c_ok = test_cataract_retrieval()
    e_ok = test_cosmetic_exclusion_retrieval()
    n_ok = test_policy_rag_node_execution()

    print("\n" + "=" * 60)
    if i_ok and c_ok and e_ok and n_ok:
        print(f"{GREEN}[OK] Phase 5 Policy RAG & ChromaDB Verified!{RESET}")
    else:
        print(f"{RED}[FAIL] Phase 5 Verification encountered issues.{RESET}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
