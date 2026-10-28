"""Document Parser Tool.

Extracts text from PDF, image, and text files.
Classifies document type using content heuristics and returns DocumentMeta.
"""

import uuid
from pathlib import Path
from typing import List, Optional
import pdfplumber
import pymupdf  # PyMuPDF

from app.state import DocumentMeta, DocumentType
from app.tools.ocr import extract_text_from_image, extract_text_from_scanned_pdf, is_ocr_available


def classify_document_type(text: str, filename: str = "") -> DocumentType:
    """Classify the document type based on keywords in filename and content."""
    lower_text = text.lower()
    lower_fname = filename.lower()

    if "claim_form" in lower_fname or "pre-authorization request" in lower_text or "pre-authorization claim form" in lower_text:
        return DocumentType.CLAIM_FORM
    if "bill" in lower_fname or "estimate" in lower_fname or "pro-forma estimate" in lower_text or "itemized charges" in lower_text:
        return DocumentType.HOSPITAL_BILL
    if "discharge" in lower_fname or "discharge summary" in lower_text or "condition on discharge" in lower_text:
        return DocumentType.DISCHARGE_SUMMARY
    if "report" in lower_fname or "radiology" in lower_text or "pathology" in lower_text or "diagnostic report" in lower_text or "investigation report" in lower_text:
        return DocumentType.DIAGNOSTIC_REPORT
    if "prescription" in lower_fname or "rx" in lower_text or "prescribed medicines" in lower_text:
        return DocumentType.PRESCRIPTION

    # Secondary heuristic from body text
    if "patient details" in lower_text and "hospital" in lower_text:
        return DocumentType.CLAIM_FORM

    return DocumentType.UNKNOWN


def extract_from_pdf(pdf_path: Path) -> tuple[str, int, bool]:
    """Extract text from a PDF file using pdfplumber, falling back to pymupdf and OCR.

    Returns:
        (extracted_text, page_count, ocr_applied)
    """
    text_content = []
    page_count = 0
    ocr_applied = False

    # Attempt 1: pdfplumber (best for layout and tables)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num + 1} ---\n" + page_text.strip())
    except Exception:
        text_content = []

    combined_text = "\n\n".join(text_content).strip()

    # Attempt 2: pymupdf fallback if pdfplumber extracted nothing
    if len(combined_text) < 30:
        try:
            doc = pymupdf.open(str(pdf_path))
            page_count = len(doc)
            alt_pages = []
            for page_num in range(len(doc)):
                p_text = doc[page_num].get_text()
                if p_text.strip():
                    alt_pages.append(f"--- Page {page_num + 1} ---\n" + p_text.strip())
            doc.close()
            if alt_pages:
                combined_text = "\n\n".join(alt_pages).strip()
        except Exception:
            pass

    # Attempt 3: Scanned PDF OCR fallback
    if len(combined_text) < 30 and is_ocr_available():
        ocr_text = extract_text_from_scanned_pdf(pdf_path)
        if ocr_text:
            combined_text = ocr_text
            ocr_applied = True

    return combined_text, page_count, ocr_applied


def parse_document(file_path: str | Path) -> DocumentMeta:
    """Parse any supported document (PDF, TXT, PNG, JPG) into DocumentMeta.

    Args:
        file_path: Absolute or relative path to file.

    Returns:
        DocumentMeta containing extracted text, page count, and classification.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file does not exist: {path}")

    file_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    suffix = path.suffix.lower()
    raw_text = ""
    page_count = 1
    ocr_applied = False

    if suffix in [".txt", ".md", ".csv"]:
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
    elif suffix == ".pdf":
        raw_text, page_count, ocr_applied = extract_from_pdf(path)
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        raw_text = extract_text_from_image(path)
        ocr_applied = True
    else:
        # Fallback raw text attempt
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()

    doc_type = classify_document_type(raw_text, path.name)

    return DocumentMeta(
        file_id=file_id,
        filename=path.name,
        doc_type=doc_type,
        page_count=max(1, page_count),
        raw_text=raw_text,
        redacted_text="",  # Will be populated by PII Redaction Agent
        ocr_applied=ocr_applied,
    )


def parse_document_batch(file_paths: List[str | Path]) -> List[DocumentMeta]:
    """Parse multiple documents and return a list of DocumentMeta objects."""
    return [parse_document(p) for p in file_paths]
