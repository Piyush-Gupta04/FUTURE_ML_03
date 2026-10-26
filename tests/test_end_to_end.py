"""
tests/test_end_to_end.py
------------------------
End-to-end integration tests verifying text preprocessing, skill extraction,
TF-IDF cosine matching, and candidate ranking using inline text strings.
"""

import pytest

from src.matching import calculate_match
from src.preprocessing import clean_text
from src.ranking import candidates_to_dataframe, rank_candidates
from src.skill_extraction import extract_skills


def test_end_to_end_pipeline_integration():
    """Verify that the screening pipeline produces sensible, explainable ranking from inline strings."""
    # 1. User-provided Job Description
    job_description = (
        "We are hiring a Machine Learning Engineer. "
        "Required skills include Python, PyTorch, Docker, Kubernetes, and SQL."
    )

    # 2. Candidate Resumes (represented as inline strings with distinct skill profiles)
    candidates = {
        "candidate_ml": (
            "Senior Machine Learning Engineer with 4 years experience. "
            "Skilled in Python, PyTorch, Docker, Kubernetes, and SQL."
        ),
        "candidate_partial": (
            "Software engineer with 2 years of experience in Python and SQL database queries."
        ),
        "candidate_unrelated": (
            "Experienced accountant managing corporate tax audits and general ledger accounts."
        ),
    }

    # 3. Clean and verify skills extraction
    cleaned_jd = clean_text(job_description, redact=False)
    jd_skills = extract_skills(cleaned_jd)
    assert "Python" in jd_skills
    assert "PyTorch" in jd_skills
    assert "Docker" in jd_skills

    # 4. Rank candidates against the user job description
    ranked = rank_candidates(candidates, job_description=job_description)
    assert len(ranked) == 3

    # Rank 1 must be candidate_ml (full skill match)
    top_candidate = ranked[0]
    assert top_candidate.candidate_id == "candidate_ml"
    assert top_candidate.rank == 1
    assert top_candidate.overall_score > 50.0
    assert "PyTorch" in top_candidate.matched_skills
    assert "Docker" in top_candidate.matched_skills

    # Rank 2 must be candidate_partial (partial skill match: Python, SQL)
    mid_candidate = ranked[1]
    assert mid_candidate.candidate_id == "candidate_partial"
    assert mid_candidate.rank == 2
    assert "Python" in mid_candidate.matched_skills
    assert "SQL" in mid_candidate.matched_skills
    assert top_candidate.overall_score > mid_candidate.overall_score

    # Rank 3 must be candidate_unrelated (no matching skills)
    lowest_candidate = ranked[2]
    assert lowest_candidate.candidate_id == "candidate_unrelated"
    assert lowest_candidate.rank == 3
    assert mid_candidate.overall_score > lowest_candidate.overall_score

    # 5. Verify tabular DataFrame export
    df = candidates_to_dataframe(ranked)
    assert len(df) == 3
    assert df.iloc[0]["Candidate ID"] == "candidate_ml"
    assert "Explanation" in df.columns
    assert len(df.iloc[0]["Explanation"]) > 10


def test_end_to_end_cross_role_sensitivity():
    """Verify that changing the job description dynamically alters candidate ranking."""
    candidates = {
        "candidate_python_dev": "Backend developer with Python, FastAPI, PostgreSQL, and Docker.",
        "candidate_react_dev": "Frontend developer with JavaScript, React, Next.js, and CSS.",
    }

    # Screening for Web Frontend Role
    frontend_jd = "Frontend Developer: React, JavaScript, HTML, CSS."
    ranked_frontend = rank_candidates(candidates, job_description=frontend_jd)
    assert ranked_frontend[0].candidate_id == "candidate_react_dev"

    # Screening for Backend Python Role
    backend_jd = "Backend Engineer: Python, FastAPI, PostgreSQL, Docker."
    ranked_backend = rank_candidates(candidates, job_description=backend_jd)
    assert ranked_backend[0].candidate_id == "candidate_python_dev"
