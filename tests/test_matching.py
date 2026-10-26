"""
tests/test_matching.py
----------------------
Unit tests for skill metrics calculation, TF-IDF cosine similarity,
composite scoring, weight validation, and edge case fallbacks.
"""

import pytest

from src.matching import (
    DEFAULT_SKILL_WEIGHT,
    DEFAULT_TEXT_WEIGHT,
    MatchResult,
    calculate_match,
    compute_skill_metrics,
    compute_text_similarity,
    validate_weights,
)


class TestWeightValidation:
    """Tests for scoring weight checks and boundary enforcement."""

    def test_valid_weights(self):
        w_skill, w_text = validate_weights(0.60, 0.40)
        assert w_skill == 0.60
        assert w_text == 0.40

        w_skill, w_text = validate_weights(1.0, 0.0)
        assert w_skill == 1.0
        assert w_text == 0.0

    def test_negative_weights_raise_value_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_weights(-0.1, 1.1)

    def test_weights_not_summing_to_one_raise_value_error(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_weights(0.5, 0.3)


class TestSkillMetricsCalculation:
    """Deterministic tests for skill coverage, Jaccard similarity, and gap analysis."""

    def test_full_skill_overlap(self):
        jd_skills = ["Docker", "Python", "PyTorch", "SQL"]
        cand_skills = ["Docker", "Python", "PyTorch", "SQL"]

        skill_score, coverage, jaccard, matched, missing, additional, notes = compute_skill_metrics(
            cand_skills, jd_skills
        )

        assert coverage == 1.0
        assert jaccard == 1.0
        assert skill_score == 1.0
        assert matched == sorted(jd_skills)
        assert missing == []
        assert additional == []

    def test_partial_skill_overlap_with_missing_and_additional(self):
        jd_skills = ["Docker", "Python", "PyTorch", "SQL"]
        # Candidate has Python, SQL (matched) and React, Vue (additional)
        cand_skills = ["Python", "React", "SQL", "Vue"]

        skill_score, coverage, jaccard, matched, missing, additional, notes = compute_skill_metrics(
            cand_skills, jd_skills
        )

        assert coverage == 0.5  # 2 matched out of 4 required
        assert matched == ["Python", "SQL"]
        assert missing == ["Docker", "PyTorch"]
        assert additional == ["React", "Vue"]
        # Union size: 6 ('Docker', 'PyTorch', 'Python', 'React', 'SQL', 'Vue')
        # Jaccard = 2/6 = 1/3
        expected_jaccard = 2.0 / 6.0
        assert abs(jaccard - expected_jaccard) < 1e-5
        expected_skill_score = (0.8 * 0.5) + (0.2 * expected_jaccard)
        assert abs(skill_score - expected_skill_score) < 1e-5

    def test_higher_skill_overlap_yields_higher_score(self):
        jd_skills = ["AWS", "Docker", "Kubernetes", "Linux", "Python"]

        cand_high = ["AWS", "Docker", "Kubernetes", "Linux", "Python"]
        cand_med = ["Docker", "Linux", "Python"]
        cand_low = ["Python"]

        score_high, _, _, _, _, _, _ = compute_skill_metrics(cand_high, jd_skills)
        score_med, _, _, _, _, _, _ = compute_skill_metrics(cand_med, jd_skills)
        score_low, _, _, _, _, _, _ = compute_skill_metrics(cand_low, jd_skills)

        assert score_high > score_med > score_low


class TestTextSimilarity:
    """Verify TF-IDF cosine similarity bounds and semantic properties."""

    def test_identical_text_similarity_close_to_one(self):
        text = "Senior Machine Learning Engineer with extensive experience in neural networks."
        sim, notes = compute_text_similarity(text, text)
        assert abs(sim - 1.0) < 1e-3
        assert len(notes) == 0

    def test_similarity_is_bounded_between_zero_and_one(self):
        doc1 = "Python developer building microservices with FastAPI and Docker."
        doc2 = "Accountant managing general ledger, tax filing, and auditing."
        sim, _ = compute_text_similarity(doc1, doc2)
        assert 0.0 <= sim <= 1.0

    def test_empty_documents_return_zero_similarity_with_note(self):
        sim, notes = compute_text_similarity("", "Valid document text")
        assert sim == 0.0
        assert any("no extractable text" in n.lower() for n in notes)


class TestCompositeCalculateMatch:
    """End-to-end matching integration and result container verification."""

    def test_match_result_structure_and_formula(self):
        jd = (
            "We are seeking a Machine Learning Engineer proficient in Python, "
            "PyTorch, and Docker. Experience with SQL and Kubernetes is a plus."
        )
        resume = (
            "Machine Learning Engineer with 4 years experience. "
            "Expert in Python, PyTorch, Docker, and SQL. Built pipelines with FastAPI."
        )

        result: MatchResult = calculate_match(
            resume_text=resume,
            job_description=jd,
            candidate_id="cand_001",
            skill_weight=0.60,
            text_weight=0.40,
        )

        assert result.candidate_id == "cand_001"
        assert 0.0 <= result.overall_score <= 100.0
        assert 0.0 <= result.skill_score <= 1.0
        assert 0.0 <= result.text_similarity <= 1.0
        assert result.skill_weight == 0.60
        assert result.text_weight == 0.40
        assert "Python" in result.matched_skills
        assert "PyTorch" in result.matched_skills
        assert "Docker" in result.matched_skills
        assert "Kubernetes" in result.missing_skills
        assert "FastAPI" in result.additional_skills

        # Verify math consistency
        expected_composite = (0.60 * result.skill_score + 0.40 * result.text_similarity) * 100.0
        assert abs(result.overall_score - round(expected_composite, 2)) < 0.05

    def test_empty_resume_handling(self):
        jd = "Requirements: Python, SQL, Docker."
        result = calculate_match(resume_text="", job_description=jd, candidate_id="empty_cand")
        assert result.overall_score == 0.0
        assert result.skill_score == 0.0
        assert result.text_similarity == 0.0
        assert len(result.missing_skills) > 0
        assert any("no text content" in n.lower() for n in result.notes)

    def test_empty_job_description_handling(self):
        resume = "Experienced in Python and Docker."
        result = calculate_match(resume_text=resume, job_description="", candidate_id="cand_x")
        assert result.overall_score == 0.0
        assert result.total_jd_skills == 0
        assert any("no text content" in n.lower() for n in result.notes)

    def test_jd_without_recognized_skills(self):
        jd = "Seeking a motivated hard-working individual for general office operations."
        resume = "Skilled in Python, SQL, and Docker."
        result = calculate_match(resume_text=resume, job_description=jd)
        assert result.total_jd_skills == 0
        assert result.skill_score == 0.0
        assert any("no recognized technical skills" in n.lower() for n in result.notes)
