"""
tests/test_app_helpers.py
-------------------------
Unit tests for Streamlit application helper functions (Kaggle dataset parsing,
in-memory file upload preparation, and privacy-safe CSV export formatting).
"""

import io
import pandas as pd
import pytest

from app import (
    format_export_dataframe,
    prepare_candidates_from_kaggle,
    prepare_candidates_from_uploads,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_prepare_candidates_from_kaggle():
    df = pd.DataFrame({
        "ID": [101, 102],
        "Category": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
        "Resume_str": ["Python developer with Docker", "Mechanical engineer with CAD"],
        "Resume_html": ["<p>html</p>", "<p>html</p>"],
    })

    candidates = prepare_candidates_from_kaggle(df)
    assert len(candidates) == 2
    assert "Candidate_101 (INFORMATION-TECHNOLOGY)" in candidates
    assert candidates["Candidate_101 (INFORMATION-TECHNOLOGY)"] == "Python developer with Docker"


def test_prepare_candidates_from_kaggle_empty():
    assert prepare_candidates_from_kaggle(pd.DataFrame()) == {}


def test_prepare_candidates_from_uploads_valid_txt():
    txt_bytes = "Data Scientist proficient in Python and SQL".encode("utf-8")
    mock_file = MockUploadedFile("resume.txt", txt_bytes)

    candidates, warnings = prepare_candidates_from_uploads([mock_file])
    assert len(candidates) == 1
    assert "Uploaded_Candidate_001" in candidates
    assert "resume.txt" not in candidates
    assert "Python and SQL" in candidates["Uploaded_Candidate_001"]
    assert len(warnings) == 0


def test_prepare_candidates_from_uploads_empty_file():
    mock_file = MockUploadedFile("empty.txt", b"")
    candidates, warnings = prepare_candidates_from_uploads([mock_file])
    assert "Uploaded_Candidate_001" in candidates
    assert "empty.txt" not in candidates
    assert candidates["Uploaded_Candidate_001"] == ""
    assert any("empty.txt" in w and "no extractable text" in w.lower() for w in warnings)


def test_prepare_candidates_from_uploads_multiple_sequential_ids():
    files = [
        MockUploadedFile("first.txt", b"First candidate text with Python"),
        MockUploadedFile("second.txt", b"Second candidate text with SQL"),
        MockUploadedFile("third.txt", b"Third candidate text with Docker"),
    ]
    candidates, warnings = prepare_candidates_from_uploads(files)
    assert len(candidates) == 3
    assert list(candidates.keys()) == [
        "Uploaded_Candidate_001",
        "Uploaded_Candidate_002",
        "Uploaded_Candidate_003",
    ]
    assert len(warnings) == 0


def test_format_export_dataframe_strips_pii_and_raw_text():
    raw_df = pd.DataFrame({
        "Rank": [1],
        "Candidate ID": ["Candidate_101"],
        "Overall Match Score (%)": [85.5],
        "Skill Score (0-1)": [0.9],
        "Text Similarity (0-1)": [0.8],
        "Skill Coverage (%)": [80.0],
        "Matched Skills Count": [4],
        "Missing Skills Count": [1],
        "Matched Skills": ["Python, SQL"],
        "Missing Skills": ["Docker"],
        "Explanation": ["Rank 1: 85.5% match"],
        # Potentially sensitive or bulky columns that MUST be excluded from export:
        "Resume_str": ["Full resume text containing john.doe@email.com and 555-1234"],
        "Resume_html": ["<html>...</html>"],
        "raw_text": ["raw private career history"],
    })

    exported = format_export_dataframe(raw_df)
    assert "Resume_str" not in exported.columns
    assert "Resume_html" not in exported.columns
    assert "raw_text" not in exported.columns
    assert "Rank" in exported.columns
    assert "Candidate ID" in exported.columns
    assert "Overall Match Score (%)" in exported.columns
    assert "Explanation" in exported.columns


def test_csv_export_content_has_no_contact_info_or_full_paragraphs():
    """Verify that exporting to CSV never leaks contact info or full resume text."""
    from src.preprocessing import EMAIL_REGEX, PHONE_REGEX
    from src.ranking import rank_candidates, candidates_to_dataframe

    # Simulate realistic candidates with contact details in raw resume text
    candidates = {
        "Candidate_101 (ENGINEERING)": (
            "Software engineer. Reach me at candidate1@workmail.com or 555-432-1098. "
            "Skilled in Python and Docker."
        ),
        "Candidate_102 (HR)": (
            "HR manager. Email: hr.recruiter@company.org, Phone: (555) 987-6543. "
            "Skilled in Communication and Team Collaboration."
        ),
    }

    jd = "Software Engineer: Python, Docker, and Team Collaboration."
    ranked = rank_candidates(candidates, job_description=jd)
    ranked_df = candidates_to_dataframe(ranked)
    exported_df = format_export_dataframe(ranked_df)

    csv_text = exported_df.to_csv(index=False)

    # 1. Verify no emails leak into CSV
    assert not EMAIL_REGEX.search(csv_text), "CSV export leaked email addresses!"

    # 2. Verify no phone numbers leak into CSV
    assert not PHONE_REGEX.search(csv_text), "CSV export leaked phone numbers!"

    # 3. Verify no raw paragraph fields leak into CSV
    assert "Resume_str" not in exported_df.columns
    assert "raw_text" not in exported_df.columns

    # 4. Allowed columns only
    expected_allowed = {
        "Rank", "Candidate ID", "Overall Match Score (%)", "Skill Score (0-1)",
        "Text Similarity (0-1)", "Skill Coverage (%)", "Matched Skills Count",
        "Missing Skills Count", "Matched Skills", "Missing Skills", "Explanation"
    }
    assert set(exported_df.columns).issubset(expected_allowed)


def test_uploaded_filenames_do_not_leak_into_csv_candidate_ids():
    """
    Verify that personal filenames from uploaded files (PDF/TXT) are replaced by
    neutral IDs (Uploaded_Candidate_001, etc.) and CANNOT leak into CSV candidate IDs.
    """
    from src.ranking import rank_candidates, candidates_to_dataframe

    # Uploaded files with personal identifiers in the filename
    file1 = MockUploadedFile(
        "John_Doe_Confidential_Senior_Resume_2026.txt",
        b"Senior Machine Learning Engineer with Python, PyTorch, Docker, and SQL experience."
    )
    file2 = MockUploadedFile(
        "Jane_Smith_Private_CV.txt",
        b"Data Scientist skilled in Python, Scikit-Learn, Pandas, and AWS."
    )

    candidates, warnings = prepare_candidates_from_uploads([file1, file2])

    # Verify dictionary keys are strictly neutral IDs
    assert "Uploaded_Candidate_001" in candidates
    assert "Uploaded_Candidate_002" in candidates
    assert "John_Doe_Confidential_Senior_Resume_2026.txt" not in candidates
    assert "Jane_Smith_Private_CV.txt" not in candidates

    # Run ranking and export pipeline
    jd = "Machine Learning Engineer: Python, PyTorch, Docker, SQL, and AWS."
    ranked = rank_candidates(candidates, job_description=jd)
    ranked_df = candidates_to_dataframe(ranked)
    exported_df = format_export_dataframe(ranked_df)

    csv_text = exported_df.to_csv(index=False)

    # 1. Candidate ID column values must be neutral IDs
    exported_candidate_ids = exported_df["Candidate ID"].tolist()
    assert set(exported_candidate_ids) == {"Uploaded_Candidate_001", "Uploaded_Candidate_002"}

    # 2. Uploaded filenames and personal names must NOT appear in the CSV
    assert "John_Doe" not in csv_text
    assert "Jane_Smith" not in csv_text
    assert "Confidential" not in csv_text
    assert "Private_CV" not in csv_text
    assert "John_Doe_Confidential_Senior_Resume_2026.txt" not in csv_text
    assert "Jane_Smith_Private_CV.txt" not in csv_text
