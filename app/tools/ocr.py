"""Optical Character Recognition (OCR) Tool.

Extracts text from scanned images or non-selectable PDFs using Tesseract OCR.
Includes graceful fallback if Tesseract binary is not installed on the host system.
"""

import logging
from pathlib import Path
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Check if pytesseract is available
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


def is_ocr_available() -> bool:
    """Check if Tesseract binary is available on system PATH."""
    if not PYTESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_from_image(image_path: str | Path) -> str:
    """Extract text from an image file using OCR.

    Args:
        image_path: Path to the image (.png, .jpg, .jpeg, .tiff, etc.)

    Returns:
        Extracted text string, or empty string if OCR is unavailable/fails.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image path not found: {path}")
        return ""

    if not is_ocr_available():
        logger.info(
            f"Tesseract OCR is not installed or not in PATH. Skipping OCR for: {path.name}"
        )
        return ""

    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed for {path.name}: {e}")
        return ""


def extract_text_from_scanned_pdf(pdf_path: str | Path) -> str:
    """Render PDF pages as images and apply OCR when text layer is absent."""
    if not is_ocr_available():
        return ""

    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        extracted_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                extracted_pages.append(f"--- Page {page_num + 1} ---\n" + page_text.strip())

        doc.close()
        return "\n\n".join(extracted_pages)
    except Exception as e:
        logger.warning(f"Scanned PDF OCR failed for {pdf_path}: {e}")
        return ""
