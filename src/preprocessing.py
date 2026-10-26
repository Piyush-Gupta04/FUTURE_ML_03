"""
src/preprocessing.py
-------------------
Text extraction, cleaning, normalization, and contact-information redaction for resumes.

DISCLAIMER ON PII REDACTION:
Heuristic redaction of contact details (emails, phone numbers, URLs) is performed
using regular expressions to reduce bias and protect candidate privacy.
This redaction is heuristic-based and is NOT guaranteed to be 100% comprehensive
or mathematically privacy-preserving (e.g. differential privacy). Certain indirect
identifiers or non-standard formatting may remain in the extracted text.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

# Robust PyMuPDF import with backward-compatible fallback
try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None


@dataclass
class ExtractionResult:
    """Container for the output of document parsing and text preprocessing."""
    text: str = ""
    raw_text: str = ""
    file_type: str = "unknown"
    success: bool = True
    is_empty: bool = True
    warnings: List[str] = field(default_factory=list)


# --- Heuristic Regex Patterns for Contact Information Redaction ---
# Matches common email addresses
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Matches URLs and web profiles (e.g., http://, https://, www., linkedin, github)
URL_REGEX = re.compile(
    r"(?:https?://|www\.)[^\s/$.?#].[^\s]*|"
    r"\b(?:linkedin\.com|github\.com|gitlab\.com)/[^\s]+\b",
    re.IGNORECASE,
)

# Matches varied telephone formats (international prefixes, brackets, dots, hyphens)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,5}\b"
)


def redact_contact_info(text: str) -> str:
    """
    Redact emails, phone numbers, and URLs from raw text using documented heuristics.

    Note: This is an advisory, heuristic redaction step and does not guarantee
    perfect removal of all personally identifiable information (PII).
    """
    if not text:
        return ""

    # Replace emails first
    redacted = EMAIL_REGEX.sub("[EMAIL]", text)
    # Replace URLs
    redacted = URL_REGEX.sub("[URL]", redacted)
    # Replace phone numbers
    redacted = PHONE_REGEX.sub("[PHONE]", redacted)

    return redacted


def clean_text(text: str, redact: bool = True) -> str:
    """
    Normalize text encoding, whitespace, and optionally redact contact info.

    - Normalizes Unicode (NFKC)
    - Removes non-printable characters
    - Normalizes consecutive whitespace and line breaks
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""

    # Normalize unicode characters (e.g., curly quotes, accents)
    normalized = unicodedata.normalize("NFKC", text)

    # Optionally apply heuristic PII redaction
    if redact:
        normalized = redact_contact_info(normalized)

    # Remove non-printable control characters (retaining common whitespace)
    normalized = "".join(
        ch for ch in normalized if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C")
    )

    # Normalize multiple whitespace characters into a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    # Join consecutive non-empty lines with single newline, drop excessive blank lines
    cleaned = "\n".join(line for line in lines if line)

    return cleaned.strip()


def extract_text_from_txt_bytes(txt_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Decode raw bytes of a plain text file into a string.
    Tries UTF-8, then falls back to Latin-1.
    """
    warnings: List[str] = []
    if not txt_bytes:
        warnings.append("Text file is empty.")
        return "", warnings

    try:
        text = txt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = txt_bytes.decode("latin-1")
            warnings.append("Decoded using latin-1 fallback due to non-UTF-8 characters.")
        except Exception as exc:
            warnings.append(f"Failed to decode text file: {str(exc)}")
            return "", warnings

    return text, warnings


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Extract text content from in-memory PDF bytes using PyMuPDF.

    The PDF bytes are parsed entirely in memory without writing to disk,
    safeguarding candidate privacy.
    """
    warnings: List[str] = []

    if pymupdf is None:
        warnings.append("PyMuPDF library is not installed.")
        return "", warnings

    if not pdf_bytes:
        warnings.append("PDF byte buffer is empty (0 bytes).")
        return "", warnings

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        warnings.append(f"Corrupt or unreadable PDF document: {str(exc)}")
        return "", warnings

    try:
        extracted_pages: List[str] = []
        for page_idx, page in enumerate(doc):
            try:
                page_text = page.get_text()
                if page_text:
                    extracted_pages.append(page_text)
            except Exception as page_exc:
                warnings.append(f"Could not extract text from page {page_idx + 1}: {str(page_exc)}")

        doc.close()

        full_text = "\n".join(extracted_pages).strip()
        if not full_text:
            warnings.append(
                "PDF contains no extractable text. It may be a scanned image or protected document."
            )

        return full_text, warnings
    except Exception as exc:
        try:
            doc.close()
        except Exception:
            pass
        warnings.append(f"Error processing PDF pages: {str(exc)}")
        return "", warnings


def extract_text_from_file(
    file_source: Union[bytes, io.BytesIO, str, Path],
    filename: str = "",
    redact_pii: bool = True,
) -> ExtractionResult:
    """
    Unified entry point for extracting and cleaning text from TXT or PDF sources.

    - Operates purely in-memory if bytes or BytesIO is provided.
    - Gracefully handles corrupt, empty, or unsupported inputs.
    """
    warnings: List[str] = []
    file_bytes: bytes = b""
    detected_name = filename

    # Resolve input source into in-memory bytes
    if isinstance(file_source, bytes):
        file_bytes = file_source
    elif isinstance(file_source, io.BytesIO):
        file_bytes = file_source.getvalue()
    elif isinstance(file_source, (str, Path)):
        path = Path(file_source)
        detected_name = detected_name or path.name
        if not path.exists():
            return ExtractionResult(
                success=False,
                is_empty=True,
                warnings=[f"File not found: {str(path)}"],
            )
        try:
            file_bytes = path.read_bytes()
        except Exception as exc:
            return ExtractionResult(
                success=False,
                is_empty=True,
                warnings=[f"Unable to read file: {str(exc)}"],
            )
    else:
        return ExtractionResult(
            success=False,
            is_empty=True,
            warnings=[f"Unsupported input type: {type(file_source).__name__}"],
        )

    # Determine file extension
    ext = Path(detected_name).suffix.lower() if detected_name else ""
    if not ext:
        # Simple magic bytes heuristic
        if file_bytes.startswith(b"%PDF"):
            ext = ".pdf"
        else:
            ext = ".txt"

    raw_text = ""
    file_type = ext.lstrip(".").upper() if ext else "UNKNOWN"

    if ext == ".pdf":
        raw_text, pdf_warnings = extract_text_from_pdf_bytes(file_bytes)
        warnings.extend(pdf_warnings)
    elif ext in [".txt", ".text"]:
        raw_text, txt_warnings = extract_text_from_txt_bytes(file_bytes)
        warnings.extend(txt_warnings)
    else:
        return ExtractionResult(
            file_type=file_type,
            success=False,
            is_empty=True,
            warnings=[f"Unsupported file format '{ext}'. Only .pdf and .txt are supported."],
        )

    cleaned = clean_text(raw_text, redact=redact_pii)
    is_empty = len(cleaned.strip()) == 0

    return ExtractionResult(
        text=cleaned,
        raw_text=raw_text,
        file_type=file_type,
        success=len(warnings) == 0 or len(cleaned) > 0,
        is_empty=is_empty,
        warnings=warnings,
    )
