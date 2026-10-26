"""
src/matching.py
---------------
Deterministic, explainable candidate-to-job-description matching engine.

Formulation:
- Required-Skill Coverage (Recall): |S_R ∩ S_J| / |S_J|
- Jaccard Skill Similarity: |S_R ∩ S_J| / |S_R ∪ S_J|
- Skill Score: 0.8 * Coverage + 0.2 * Jaccard
- Text Similarity: Cosine similarity of TF-IDF vectors (unigrams + bigrams)
- Overall Score: (skill_weight * Skill Score + text_weight * Text Similarity) * 100
  (Default weights: 0.60 skill_weight, 0.40 text_weight)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.skill_extraction import extract_skills


DEFAULT_SKILL_WEIGHT = 0.60
DEFAULT_TEXT_WEIGHT = 0.40


@dataclass
class MatchResult:
    """Detailed breakdown of matching metrics between a resume and job description."""
    candidate_id: str
    overall_score: float                     # Scaled 0.0 to 100.0
    skill_score: float                       # Scaled 0.0 to 1.0
    required_skill_coverage: float           # Scaled 0.0 to 1.0 (Recall against JD)
    jaccard_similarity: float                # Scaled 0.0 to 1.0
    text_similarity: float                   # Scaled 0.0 to 1.0 (TF-IDF cosine similarity)
    matched_skills: List[str]                # Skills present in both JD and resume
    missing_skills: List[str]                # Skills required by JD but missing in resume
    additional_skills: List[str]             # Skills in resume not required by JD
    total_jd_skills: int
    total_candidate_skills: int
    skill_weight: float = DEFAULT_SKILL_WEIGHT
    text_weight: float = DEFAULT_TEXT_WEIGHT
    notes: List[str] = field(default_factory=list)


def validate_weights(skill_weight: float, text_weight: float) -> Tuple[float, float]:
    """
    Validate that scoring weights are non-negative and sum to 1.0.

    Raises:
        ValueError: If weights are negative or do not sum to 1.0 within numerical tolerance.
    """
    if skill_weight < 0.0 or text_weight < 0.0:
        raise ValueError(
            f"Weights must be non-negative. Got skill_weight={skill_weight}, text_weight={text_weight}."
        )

    weight_sum = skill_weight + text_weight
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Weights must sum to 1.0 (within tolerance 1e-4). Got sum={weight_sum:.5f} "
            f"(skill_weight={skill_weight}, text_weight={text_weight})."
        )

    return float(skill_weight), float(text_weight)


def compute_skill_metrics(
    candidate_skills: List[str],
    jd_skills: List[str],
) -> Tuple[float, float, float, List[str], List[str], List[str], List[str]]:
    """
    Compute explainable skill set overlap metrics.

    Returns:
        (skill_score, coverage, jaccard, matched, missing, additional, notes)
    """
    notes: List[str] = []
    set_c = set(candidate_skills)
    set_j = set(jd_skills)

    matched = sorted(list(set_c.intersection(set_j)))
    missing = sorted(list(set_j.difference(set_c)))
    additional = sorted(list(set_c.difference(set_j)))

    # Edge Case: Job description has no recognized technical skills
    if len(set_j) == 0:
        notes.append("Job description has no recognized technical skills; skill score defaults to 0.0.")
        return 0.0, 0.0, 0.0, matched, missing, additional, notes

    # Required skill coverage (Recall)
    coverage = len(matched) / len(set_j)

    # Jaccard similarity across the union of declared skills
    union_set = set_c.union(set_j)
    jaccard = len(matched) / len(union_set) if len(union_set) > 0 else 0.0

    # Approved formula: 0.8 * Coverage + 0.2 * Jaccard
    skill_score = (0.8 * coverage) + (0.2 * jaccard)
    skill_score = float(np.clip(skill_score, 0.0, 1.0))

    return skill_score, coverage, jaccard, matched, missing, additional, notes


def compute_text_similarity(
    doc_a: str,
    doc_b: str,
) -> Tuple[float, List[str]]:
    """
    Compute cosine similarity between two documents using TF-IDF with unigrams and bigrams.

    Returns:
        (similarity_score, notes)
    """
    notes: List[str] = []
    clean_a = clean_text(doc_a, redact=False).strip()
    clean_b = clean_text(doc_b, redact=False).strip()

    if not clean_a or not clean_b:
        notes.append("One or both documents contain no extractable text; text similarity defaults to 0.0.")
        return 0.0, notes

    # Check if text consists exclusively of punctuation or digits
    words_a = re.findall(r"\b[a-zA-Z]{2,}\b", clean_a)
    words_b = re.findall(r"\b[a-zA-Z]{2,}\b", clean_b)
    if not words_a or not words_b:
        notes.append("Insufficient alphabetic vocabulary for TF-IDF; text similarity defaults to 0.0.")
        return 0.0, notes

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            lowercase=True,
        )
        tfidf_matrix = vectorizer.fit_transform([clean_a, clean_b])
        # Compute cosine similarity between doc_a and doc_b
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Bounded mathematically between 0.0 and 1.0
        bounded_score = float(np.clip(cos_sim, 0.0, 1.0))
        return bounded_score, notes
    except ValueError as val_err:
        # e.g., empty vocabulary if all terms are stopwords
        notes.append(f"TF-IDF vocabulary exception: {str(val_err)}; text similarity defaults to 0.0.")
        return 0.0, notes


def calculate_match(
    resume_text: str,
    job_description: str,
    candidate_id: str = "Candidate",
    skill_weight: float = DEFAULT_SKILL_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
) -> MatchResult:
    """
    Calculate an explainable candidate match score comparing resume text to a job description.

    - Gracefully handles empty or unreadable text.
    - Validates scoring weights.
    - Decomposes match score into visible components.
    """
    skill_w, text_w = validate_weights(skill_weight, text_weight)
    all_notes: List[str] = []

    clean_resume = clean_text(resume_text, redact=True)
    clean_jd = clean_text(job_description, redact=False)

    if not clean_resume:
        all_notes.append("Resume contains no text content.")
    if not clean_jd:
        all_notes.append("Job description contains no text content.")

    # 1. Skill Extraction
    candidate_skills = extract_skills(clean_resume)
    jd_skills = extract_skills(clean_jd)

    # 2. Skill Overlap Metrics
    skill_score, coverage, jaccard, matched, missing, additional, skill_notes = compute_skill_metrics(
        candidate_skills, jd_skills
    )
    all_notes.extend(skill_notes)

    # 3. Text Semantic Similarity
    text_sim, text_notes = compute_text_similarity(clean_jd, clean_resume)
    all_notes.extend(text_notes)

    # 4. Overall Composite Match Score
    # Formula: (skill_weight * skill_score + text_weight * text_similarity) * 100.0
    overall_score = (skill_w * skill_score + text_w * text_sim) * 100.0
    overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 2)

    return MatchResult(
        candidate_id=candidate_id,
        overall_score=overall_score,
        skill_score=round(skill_score, 4),
        required_skill_coverage=round(coverage, 4),
        jaccard_similarity=round(jaccard, 4),
        text_similarity=round(text_sim, 4),
        matched_skills=matched,
        missing_skills=missing,
        additional_skills=additional,
        total_jd_skills=len(jd_skills),
        total_candidate_skills=len(candidate_skills),
        skill_weight=skill_w,
        text_weight=text_w,
        notes=all_notes,
    )
