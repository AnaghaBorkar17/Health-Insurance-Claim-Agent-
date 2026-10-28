"""Policy RAG Agent.

Retrieves relevant policy clauses from ChromaDB based on clinical diagnoses,
procedure descriptions, and claimed expenses. Returns evidence-grounded
citations with exact clause identifiers and text.
"""

import logging
from typing import Dict, List, Optional
from app.state import ClaimState, ExtractedClaim, PolicyClause
from app.tools.policy_ingestion import get_policy_collection, ingest_policy_documents

logger = logging.getLogger(__name__)


def retrieve_relevant_clauses(claim: ExtractedClaim, top_k: int = 3) -> List[PolicyClause]:
    """Perform targeted semantic retrieval against indexed policy booklet clauses.

    Constructs multiple clinical queries from diagnosis, procedure, and cost items,
    executes vector similarity search, and deduplicates the retrieved clauses.
    """
    collection = get_policy_collection()

    # If collection is empty, automatically trigger ingestion
    if collection.count() == 0:
        logger.info("Policy collection is empty. Triggering automated ingestion.")
        ingest_policy_documents()

    # Formulate semantic queries
    search_queries = []

    # 1. Diagnosis-based query
    if claim.diagnosis_descriptions:
        search_queries.append(" ".join(claim.diagnosis_descriptions))
    elif claim.diagnosis_codes:
        search_queries.append(f"Diagnosis condition {' '.join(claim.diagnosis_codes)}")

    # 2. Procedure-based query
    if claim.procedure_descriptions:
        search_queries.append(" ".join(claim.procedure_descriptions))
    elif claim.procedure_codes:
        search_queries.append(f"Surgical procedure treatment {' '.join(claim.procedure_codes)}")

    # 3. Specific condition checks
    combined_desc = (" ".join(claim.diagnosis_descriptions) + " " + " ".join(claim.procedure_descriptions)).lower()
    if "cataract" in combined_desc or any(c.startswith("H25") for c in claim.diagnosis_codes):
        search_queries.append("Cataract surgery treatment limit intraocular lens")
    elif "cosmetic" in combined_desc or "rhinoplasty" in combined_desc or any(c.startswith("Z41") for c in claim.diagnosis_codes):
        search_queries.append("Cosmetic aesthetic plastic surgery exclusion")
    elif "appendicitis" in combined_desc or "appendectomy" in combined_desc:
        search_queries.append("Emergency appendectomy surgery coverage")

    if not search_queries:
        search_queries.append("General hospitalization room rent and exclusions")

    retrieved_map: Dict[str, PolicyClause] = {}

    for query_text in search_queries:
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(top_k, collection.count()),
            )

            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metadatas, distances):
                clause_id = meta.get("clause_id", "CLAUSE-UNKNOWN")
                # Avoid duplicates across multiple queries; keep best relevance score
                relevance = round(max(0.0, 1.0 - float(dist)), 4) if dist is not None else 1.0

                if clause_id not in retrieved_map or relevance > retrieved_map[clause_id].retrieval_score:
                    retrieved_map[clause_id] = PolicyClause(
                        clause_id=clause_id,
                        title=meta.get("title", "Policy Clause"),
                        text=doc_text,
                        document_source=meta.get("source", "careplus_comprehensive_policy.txt"),
                        page_reference=int(meta.get("page", 1)),
                        retrieval_score=relevance,
                    )
        except Exception as e:
            logger.warning(f"Vector search failed for query '{query_text}': {e}")

    # Sort descending by relevance score
    sorted_clauses = sorted(retrieved_map.values(), key=lambda c: c.retrieval_score, reverse=True)
    return sorted_clauses[:top_k]


def policy_rag_node(state: ClaimState) -> Dict:
    """LangGraph node: Retrieves relevant policy clauses grounded in vector evidence."""
    extracted: Optional[ExtractedClaim] = state.get("extracted_claim")

    if not extracted:
        return {
            "retrieved_clauses": [],
            "current_node": "policy_rag",
            "audit_trail": ["[PolicyRAGAgent] No extracted claim available. Skipping clause retrieval."],
        }

    clauses = retrieve_relevant_clauses(extracted, top_k=3)

    clause_summaries = [f"{c.clause_id} ({c.title}) [Score: {c.retrieval_score}]" for c in clauses]
    audit_entry = (
        f"[PolicyRAGAgent] Retrieved {len(clauses)} grounded policy clauses: "
        f"{', '.join(clause_summaries) if clause_summaries else 'None found'}."
    )

    return {
        "retrieved_clauses": clauses,
        "current_node": "policy_rag",
        "audit_trail": [audit_entry],
    }
