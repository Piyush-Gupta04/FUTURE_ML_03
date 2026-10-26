"""
tests/test_preprocessing.py
---------------------------
Unit tests for text extraction, text cleaning, in-memory PDF parsing,
and contact-information redaction.
"""

import io
import pytest

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

from src.preprocessing import (
    clean_text,
    extract_text_from_file,
    extract_text_from_pdf_bytes,
    extract_text_from_txt_bytes,
    redact_contact_info,
)


def _generate_synthetic_pdf_bytes(text_content: str) -> bytes:
    """Helper to generate an in-memory PDF without writing to disk."""
    assert pymupdf is not None, "PyMuPDF is required to create synthetic PDF bytes"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text_content)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestTextCleaningAndRedaction:
    """Tests for text normalization and contact-information redaction."""

    def test_clean_text_normalizes_whitespace_and_unicode(self):
        raw = "   Senior \t  Data   Scientist   \n\n\n\r   with   Python   "
        cleaned = clean_text(raw, redact=False)
        assert cleaned == "Senior Data Scientist\nwith Python"

    def test_redact_contact_info_emails(self):
        sample = "Contact candidate at john.doe@example.com or hr_dept@company.co.in for details."
        redacted = redact_contact_info(sample)
        assert "[EMAIL]" in redacted
        assert "john.doe@example.com" not in redacted
        assert "hr_dept@company.co.in" not in redacted

    def test_redact_contact_info_phone_numbers(self):
        sample = "Reach me at +1 (555) 123-4567 or 987-654-3210 anytime."
        redacted = redact_contact_info(sample)
        assert "[PHONE]" in redacted
        assert "123-4567" not in redacted
        assert "987-654-3210" not in redacted

    def test_redact_contact_info_urls_and_profiles(self):
        sample = "Portfolio: https://candidate.io/profile and github.com/candidate-dev"
        redacted = redact_contact_info(sample)
        assert "[URL]" in redacted
        assert "https://candidate.io/profile" not in redacted
        assert "github.com/candidate-dev" not in redacted

    def test_clean_text_empty_and_whitespace(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t  ") == ""
        assert clean_text(None) == ""


class TestTxtExtraction:
    """Tests for plain-text file extraction."""

    def test_valid_txt_bytes_extraction(self):
        raw_bytes = "Experienced Machine Learning Engineer\nProficient in PyTorch and SQL".encode("utf-8")
        text, warnings = extract_text_from_txt_bytes(raw_bytes)
        assert "Machine Learning Engineer" in text
        assert len(warnings) == 0

    def test_empty_txt_bytes(self):
        text, warnings = extract_text_from_txt_bytes(b"")
        assert text == ""
        assert any("empty" in w.lower() for w in warnings)


class TestPdfExtraction:
    """Tests for PDF extraction from in-memory bytes."""

    def test_valid_pdf_in_memory_extraction(self):
        content = "Synthetic Resume: Alice Smith\nSpecialized in NLP, PyTorch, and Docker."
        pdf_bytes = _generate_synthetic_pdf_bytes(content)

        text, warnings = extract_text_from_pdf_bytes(pdf_bytes)
        assert "Alice Smith" in text
        assert "PyTorch" in text
        assert len(warnings) == 0

    def test_empty_pdf_bytes(self):
        text, warnings = extract_text_from_pdf_bytes(b"")
        assert text == ""
        assert any("empty" in w.lower() for w in warnings)

    def test_corrupt_pdf_bytes(self):
        corrupt_bytes = b"NOT_A_VALID_PDF_HEADER_DATA_12345678"
        text, warnings = extract_text_from_pdf_bytes(corrupt_bytes)
        assert text == ""
        assert any("corrupt" in w.lower() or "error" in w.lower() for w in warnings)

    def test_pdf_without_extractable_text(self):
        """Simulate scanned/blank PDF."""
        doc = pymupdf.open()
        _ = doc.new_page()  # Blank page with no text
        blank_pdf_bytes = doc.tobytes()
        doc.close()

        text, warnings = extract_text_from_pdf_bytes(blank_pdf_bytes)
        assert text == ""
        assert any("no extractable text" in w.lower() for w in warnings)


class TestUnifiedExtractTextFromFile:
    """Tests for the high-level extract_text_from_file function."""

    def test_bytesio_pdf_input(self):
        pdf_bytes = _generate_synthetic_pdf_bytes("Backend Developer: FastAPI, PostgreSQL, Redis")
        buffer = io.BytesIO(pdf_bytes)

        result = extract_text_from_file(buffer, filename="candidate.pdf")
        assert result.success is True
        assert result.is_empty is False
        assert result.file_type == "PDF"
        assert "FastAPI" in result.text
        assert "PostgreSQL" in result.text

    def test_unsupported_file_extension(self):
        raw_bytes = b"dummy image binary"
        result = extract_text_from_file(raw_bytes, filename="resume.png")
        assert result.success is False
        assert result.is_empty is True
        assert any("unsupported" in w.lower() for w in result.warnings)

    def test_nonexistent_file_path(self):
        result = extract_text_from_file("non_existent_resume_file.pdf")
        assert result.success is False
        assert any("not found" in w.lower() for w in result.warnings)
