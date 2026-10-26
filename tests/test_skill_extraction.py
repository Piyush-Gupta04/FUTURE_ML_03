"""
tests/test_skill_extraction.py
------------------------------
Unit tests for ontology-based skill extraction, alias resolution,
case-insensitivity, and boundary false-positive prevention.
"""

import pytest
from src.skill_extraction import (
    extract_categorized_skills,
    extract_skills,
    extract_skills_summary,
    get_skill_category,
    normalize_skill,
)


class TestSingleAndMultiWordSkillDetection:
    """Verify single-word and complex multi-word phrase extraction."""

    def test_single_word_skills(self):
        text = "Strong experience with Python, Docker, PostgreSQL, and Linux."
        skills = extract_skills(text)
        assert "Python" in skills
        assert "Docker" in skills
        assert "PostgreSQL" in skills
        assert "Linux" in skills

    def test_multi_word_skills(self):
        text = (
            "Specialized in Natural Language Processing, Machine Learning, "
            "Large Language Models, and Apache Spark."
        )
        skills = extract_skills(text)
        assert "Natural Language Processing" in skills
        assert "Machine Learning" in skills
        assert "Large Language Models" in skills
        assert "Apache Spark" in skills


class TestAliasNormalizationAndCaseInsensitivity:
    """Verify canonical naming and case-insensitive recognition."""

    def test_case_insensitivity(self):
        text = "proficient in pyTHon, PANDAS, and dOCKEr"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "Pandas" in skills
        assert "Docker" in skills

    def test_alias_normalization_ml_ai(self):
        text = "Background in sklearn, tf, k8s, postgres, and aws."
        skills = extract_skills(text)
        assert "Scikit-Learn" in skills
        assert "TensorFlow" in skills
        assert "Kubernetes" in skills
        assert "PostgreSQL" in skills
        assert "AWS" in skills

    def test_normalize_skill_function(self):
        assert normalize_skill("sklearn") == "Scikit-Learn"
        assert normalize_skill("k8s") == "Kubernetes"
        assert normalize_skill("postgres") == "PostgreSQL"
        assert normalize_skill("llm") == "Large Language Models"
        assert normalize_skill("non_existent_skill_xyz") is None
        assert normalize_skill("") is None


class TestFalsePositivePrevention:
    """Verify that single letters and special symbols do not trigger false positives."""

    def test_r_matches_in_valid_context(self):
        text = "Languages: Python, R, and SQL."
        skills = extract_skills(text)
        assert "R" in skills

    def test_r_does_not_match_inside_unrelated_words(self):
        # The letter 'r' appears inside 'car', 'regular', 'trend', 'programming', 'research'
        text = "Our car has regular trends and rigorous research with linear programming."
        skills = extract_skills(text)
        assert "R" not in skills

    def test_r_does_not_match_in_rd(self):
        text = "Led corporate R&D initiatives."
        skills = extract_skills(text)
        assert "R" not in skills

    def test_special_symbols_cpp_csharp_dotnet(self):
        text = "Full-stack engineer using C++, C#, .NET, and CI/CD pipelines."
        skills = extract_skills(text)
        assert "C++" in skills
        assert "C#" in skills
        assert ".NET" in skills
        assert "CI/CD" in skills

    def test_c_does_not_match_inside_words(self):
        text = "Participated in an academic contest with excellent results."
        skills = extract_skills(text)
        assert "C" not in skills


class TestCategorizationAndEmptyInputs:
    """Verify grouping by taxonomy and predictable empty input handling."""

    def test_empty_and_whitespace_inputs(self):
        assert extract_skills("") == []
        assert extract_skills("   \n\t  ") == []
        assert extract_skills(None) == []

        summary = extract_skills_summary("")
        assert summary.skills == []
        assert summary.categorized_skills == {}
        assert summary.total_count == 0

    def test_extract_categorized_skills(self):
        text = "Proficient in Python, PyTorch, PostgreSQL, AWS, and Agile."
        categorized = extract_categorized_skills(text)

        assert "programming_languages" in categorized
        assert "Python" in categorized["programming_languages"]

        assert "machine_learning" in categorized
        assert "PyTorch" in categorized["machine_learning"]

        assert "databases" in categorized
        assert "PostgreSQL" in categorized["databases"]

        assert "cloud_devops" in categorized
        assert "AWS" in categorized["cloud_devops"]

        assert "soft_skills" in categorized
        assert "Agile" in categorized["soft_skills"]

    def test_get_skill_category(self):
        assert get_skill_category("PyTorch") == "machine_learning"
        assert get_skill_category("PostgreSQL") == "databases"
        assert get_skill_category("AWS") == "cloud_devops"
        assert get_skill_category("UnknownTool") == "other"
