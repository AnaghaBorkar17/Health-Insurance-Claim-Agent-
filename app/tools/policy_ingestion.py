"""Policy Ingestion Tool.

Chunks fictional insurance policies into semantic clauses, generates local
vector embeddings using Ollama (nomic-embed-text), and persists them in ChromaDB.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests

from app.config import settings

logger = logging.getLogger(__name__)

# Default policy storage directory
POLICIES_DIR = Path(settings.chroma_persist_dir).parent / "policies"
COLLECTION_NAME = "careplus_policy_clauses"


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB Embedding Function that calls local Ollama runtime."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_embed_model

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": input},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            logger.error(f"Ollama embedding failure ({self.model}): {e}")
            raise


def parse_policy_into_clauses(text: str, source_name: str) -> List[Dict]:
    """Parse policy text into structured clause chunks with metadata.

    Splits by 'Clause X.X - Title' patterns to maintain complete logical terms.
    """
    clauses = []
    current_section = "GENERAL"

    # Regex matching "Clause X.X - Title"
    clause_header_pattern = re.compile(
        r"(Clause\s+(\d+\.\d+)\s*[-–:]\s*([^\n]+))",
        re.IGNORECASE,
    )

    # Section header pattern
    section_pattern = re.compile(
        r"SECTION\s+(\d+)\s*[:\-]\s*([^\n]+)",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for section headers
        sec_match = section_pattern.search(line)
        if sec_match:
            current_section = f"SECTION {sec_match.group(1)}: {sec_match.group(2).strip()}"
            i += 1
            continue

        # Check for clause headers
        clause_match = clause_header_pattern.search(line)
        if clause_match:
            full_header = clause_match.group(1).strip()
            clause_num = clause_match.group(2).strip()
            clause_title = clause_match.group(3).strip()
            clause_id = f"CLAUSE-{clause_num}"

            # Collect body text until next clause or section
            body_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if clause_header_pattern.search(next_line) or section_pattern.search(next_line):
                    break
                if next_line:
                    body_lines.append(next_line)
                i += 1

            full_clause_text = f"{full_header}\n" + " ".join(body_lines)

            clauses.append({
                "clause_id": clause_id,
                "title": clause_title,
                "section": current_section,
                "text": full_clause_text,
                "source": source_name,
                "page": 1,  # In single document text, page 1 default
            })
            continue

        i += 1

    return clauses


def get_chroma_client() -> chromadb.PersistentClient:
    """Initialize persistent ChromaDB client using configured directory."""
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_policy_collection() -> chromadb.Collection:
    """Retrieve or create the policy clauses collection with Ollama embeddings."""
    client = get_chroma_client()
    embedding_fn = OllamaEmbeddingFunction(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_policy_documents(force_reload: bool = False) -> int:
    """Ingest all policy text files into ChromaDB.

    Args:
        force_reload: If True, clears existing collection before re-indexing.

    Returns:
        Total number of clauses indexed.
    """
    collection = get_policy_collection()

    if not force_reload and collection.count() > 0:
        logger.info(f"Policy collection already indexed with {collection.count()} clauses.")
        return collection.count()

    if force_reload:
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = get_policy_collection()

    policies_dir = POLICIES_DIR
    if not policies_dir.exists():
        policies_dir.mkdir(parents=True, exist_ok=True)

    policy_files = list(policies_dir.glob("*.txt")) + list(policies_dir.glob("*.md"))
    if not policy_files:
        logger.warning(f"No policy documents found in {policies_dir}")
        return 0

    all_clauses = []
    for p_file in policy_files:
        text = p_file.read_text(encoding="utf-8")
        parsed = parse_policy_into_clauses(text, p_file.name)
        all_clauses.extend(parsed)

    if not all_clauses:
        return 0

    documents = [c["text"] for c in all_clauses]
    ids = [f"{c['source']}_{c['clause_id']}" for c in all_clauses]
    metadatas = [
        {
            "clause_id": c["clause_id"],
            "title": c["title"],
            "section": c["section"],
            "source": c["source"],
            "page": c["page"],
        }
        for c in all_clauses
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    logger.info(f"Successfully indexed {len(all_clauses)} policy clauses into ChromaDB.")
    return len(all_clauses)
