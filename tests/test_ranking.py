"""
tests/test_ranking.py
---------------------
Unit tests for candidate ranking order, deterministic tie-breaking,
explanation text generation, and DataFrame export.
"""

import pandas as pd
import pytest

from src.matching import MatchResult
from src.ranking import (
    RankedCandidate,
    candidates_to_dataframe,
    generate_candidate_explanation,
    rank_candidates,
)


class TestCandidateRankingAndTieBreaking:
    """Tests for descending sort order and deterministic tie resolution."""

    def test_descending_ranking_order(self):
        jd = "Machine Learning Engineer: Python, PyTorch, SQL, Docker, AWS."

        resumes = {
            "cand_low": "Junior developer with basic Python knowledge.",
            "cand_high": (
                "Senior ML Engineer with Python, PyTorch, SQL, Docker, and AWS. "
                "Deployed models at scale."
            ),
            "cand_mid": "Data scientist with Python, SQL, and Docker experience.",
        }

        ranked = rank_candidates(resumes, job_description=jd)

        assert len(ranked) == 3
        # First rank should be cand_high
        assert ranked[0].candidate_id == "cand_high"
        assert ranked[0].rank == 1
        # Second rank should be cand_mid
        assert ranked[1].candidate_id == "cand_mid"
        assert ranked[1].rank == 2
        # Third rank should be cand_low
        assert ranked[2].candidate_id == "cand_low"
        assert ranked[2].rank == 3

        # Verify scores are strictly descending
        assert ranked[0].overall_score >= ranked[1].overall_score >= ranked[2].overall_score

    def test_deterministic_tie_breaking(self):
        """When overall scores tie, breaks by skill score, text similarity, then candidate_id."""
        result_a = MatchResult(
            candidate_id="cand_b_alpha",
            overall_score=75.0,
            skill_score=0.8,
            required_skill_coverage=0.8,
            jaccard_similarity=0.7,
            text_similarity=0.6,
            matched_skills=["Python", "SQL"],
            missing_skills=["Docker"],
            additional_skills=[],
            total_jd_skills=3,
            total_candidate_skills=2,
        )

        result_b = MatchResult(
            candidate_id="cand_a_alpha",
            overall_score=75.0,
            skill_score=0.8,
            required_skill_coverage=0.8,
            jaccard_similarity=0.7,
            text_similarity=0.6,
            matched_skills=["Python", "SQL"],
            missing_skills=["Docker"],
            additional_skills=[],
            total_jd_skills=3,
            total_candidate_skills=2,
        )

        # Both have identical overall_score (75.0), skill_score (0.8), text_similarity (0.6)
        # Tie-breaker falls back to alphabetical candidate_id ('cand_a_alpha' comes before 'cand_b_alpha')
        ranked = rank_candidates([result_a, result_b])

        assert ranked[0].candidate_id == "cand_a_alpha"
        assert ranked[0].rank == 1
        assert ranked[1].candidate_id == "cand_b_alpha"
        assert ranked[1].rank == 2


class TestCandidateExplanation:
    """Tests for natural language explanation generation."""

    def test_explanation_contains_score_and_skills(self):
        result = MatchResult(
            candidate_id="cand_test",
            overall_score=82.5,
            skill_score=0.85,
            required_skill_coverage=0.80,
            jaccard_similarity=0.70,
            text_similarity=0.78,
            matched_skills=["Docker", "Python", "SQL"],
            missing_skills=["Kubernetes"],
            additional_skills=["FastAPI"],
            total_jd_skills=4,
            total_candidate_skills=4,
        )

        explanation = generate_candidate_explanation(result, rank=1)
        assert "Rank 1:" in explanation
        assert "82.5% match" in explanation
        assert "Python" in explanation
        assert "Kubernetes" in explanation
        assert "TF-IDF text alignment" in explanation

    def test_explanation_for_empty_resume(self):
        result = MatchResult(
            candidate_id="cand_empty",
            overall_score=0.0,
            skill_score=0.0,
            required_skill_coverage=0.0,
            jaccard_similarity=0.0,
            text_similarity=0.0,
            matched_skills=[],
            missing_skills=["Python"],
            additional_skills=[],
            total_jd_skills=1,
            total_candidate_skills=0,
            notes=["Resume contains no text content."],
        )

        explanation = generate_candidate_explanation(result, rank=2)
        assert "Rank 2:" in explanation
        assert "no extractable text" in explanation


class TestDataFrameExport:
    """Tests for conversion to pandas DataFrame."""

    def test_candidates_to_dataframe_schema(self):
        jd = "Job: Python, Docker, AWS."
        resumes = {
            "cand_1": "Python and Docker developer.",
            "cand_2": "AWS and Python engineer.",
        }

        ranked = rank_candidates(resumes, job_description=jd)
        df = candidates_to_dataframe(ranked)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        expected_columns = [
            "Rank",
            "Candidate ID",
            "Overall Match Score (%)",
            "Skill Score (0-1)",
            "Text Similarity (0-1)",
            "Skill Coverage (%)",
            "Jaccard Similarity",
            "Matched Skills Count",
            "Missing Skills Count",
            "Matched Skills",
            "Missing Skills",
            "Additional Skills",
            "Explanation",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"

        assert df.iloc[0]["Rank"] == 1
        assert df.iloc[1]["Rank"] == 2
