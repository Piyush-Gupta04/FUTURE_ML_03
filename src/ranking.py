"""
src/ranking.py
--------------
Candidate ranking engine, deterministic tie-breaking, natural language
explanation generation, and tabular export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pandas as pd

from src.matching import MatchResult, calculate_match


@dataclass
class RankedCandidate:
    """Represents a scored candidate with assigned rank and generated explanation."""
    rank: int
    candidate_id: str
    overall_score: float
    skill_score: float
    text_similarity: float
    required_skill_coverage: float
    jaccard_similarity: float
    matched_skills: List[str]
    missing_skills: List[str]
    additional_skills: List[str]
    matched_skill_count: int
    missing_skill_count: int
    explanation: str
    match_result: MatchResult


def generate_candidate_explanation(result: MatchResult, rank: Optional[int] = None) -> str:
    """
    Generate an explainable, concise summary of why a candidate received their score.

    Clearly separates matched skills, skill gaps, and textual semantic alignment.
    """
    prefix = f"Rank {rank}: " if rank is not None else ""

    # Edge Case: Zero overall score or empty inputs
    if result.overall_score == 0.0:
        if "Resume contains no text content." in result.notes:
            return f"{prefix}0.0% match. The resume contains no extractable text content."
        if "Job description contains no text content." in result.notes:
            return f"{prefix}0.0% match. The job description contains no criteria."
        if result.total_jd_skills == 0:
            return (
                f"{prefix}{result.overall_score:.1f}% match. Job description has no recognized skills; "
                f"text similarity is {result.text_similarity * 100.0:.1f}%."
            )
        return (
            f"{prefix}0.0% match. No matching skills found among the {result.total_jd_skills} "
            f"required skills; text similarity is 0.0%."
        )

    # Summary of skill overlap
    matched_cnt = len(result.matched_skills)
    total_jd = result.total_jd_skills
    matched_str = ", ".join(result.matched_skills[:6]) if result.matched_skills else "None"
    if matched_cnt > 6:
        matched_str += f" (+{matched_cnt - 6} more)"

    missing_cnt = len(result.missing_skills)
    missing_str = ", ".join(result.missing_skills[:5]) if result.missing_skills else "None"
    if missing_cnt > 5:
        missing_str += f" (+{missing_cnt - 5} more)"

    coverage_pct = result.required_skill_coverage * 100.0
    text_pct = result.text_similarity * 100.0

    if missing_cnt == 0 and total_jd > 0:
        skill_summary = f"Full coverage (100%) of all {total_jd} required skills: [{matched_str}]."
    elif matched_cnt > 0:
        skill_summary = (
            f"Matched {matched_cnt}/{total_jd} required skills ({coverage_pct:.1f}% coverage): "
            f"[{matched_str}]. Missing required skills: [{missing_str}]."
        )
    else:
        skill_summary = (
            f"Zero matching skills out of {total_jd} required. Missing: [{missing_str}]."
        )

    text_summary = f"TF-IDF text alignment is {text_pct:.1f}%."

    return (
        f"{prefix}{result.overall_score:.1f}% match score. "
        f"{skill_summary} {text_summary}"
    )


def rank_candidates(
    candidates_data: Union[Dict[str, str], List[MatchResult]],
    job_description: Optional[str] = None,
    skill_weight: float = 0.60,
    text_weight: float = 0.40,
) -> List[RankedCandidate]:
    """
    Score and rank candidates in descending order with deterministic tie-breaking.

    Arguments:
        candidates_data: Either a mapping of {candidate_id: resume_text} OR
                         a pre-computed list of MatchResult objects.
        job_description: Required if candidates_data is a dictionary of texts.
        skill_weight: Weight allocated to skill score (default: 0.60).
        text_weight: Weight allocated to text similarity (default: 0.40).

    Tie-breaking hierarchy:
        1. Higher overall_score (descending)
        2. Higher skill_score (descending)
        3. Higher text_similarity (descending)
        4. Candidate ID string comparison (ascending, ensures strict determinism)
    """
    match_results: List[MatchResult] = []

    if isinstance(candidates_data, dict):
        if job_description is None:
            raise ValueError("job_description must be provided when passing candidate text dictionary.")
        for cand_id, resume_text in candidates_data.items():
            match_res = calculate_match(
                resume_text=resume_text,
                job_description=job_description,
                candidate_id=cand_id,
                skill_weight=skill_weight,
                text_weight=text_weight,
            )
            match_results.append(match_res)
    elif isinstance(candidates_data, list):
        match_results = candidates_data
    else:
        raise TypeError(f"Unsupported candidates_data type: {type(candidates_data).__name__}")

    # Deterministic sorting
    sorted_results = sorted(
        match_results,
        key=lambda res: (
            -res.overall_score,
            -res.skill_score,
            -res.text_similarity,
            res.candidate_id,
        ),
    )

    ranked_list: List[RankedCandidate] = []
    for idx, res in enumerate(sorted_results, start=1):
        explanation = generate_candidate_explanation(res, rank=idx)
        ranked_list.append(
            RankedCandidate(
                rank=idx,
                candidate_id=res.candidate_id,
                overall_score=res.overall_score,
                skill_score=res.skill_score,
                text_similarity=res.text_similarity,
                required_skill_coverage=res.required_skill_coverage,
                jaccard_similarity=res.jaccard_similarity,
                matched_skills=res.matched_skills,
                missing_skills=res.missing_skills,
                additional_skills=res.additional_skills,
                matched_skill_count=len(res.matched_skills),
                missing_skill_count=len(res.missing_skills),
                explanation=explanation,
                match_result=res,
            )
        )

    return ranked_list


def candidates_to_dataframe(ranked_candidates: List[RankedCandidate]) -> pd.DataFrame:
    """
    Convert a list of RankedCandidate objects into a structured Pandas DataFrame
    suitable for UI tables and CSV export.
    """
    rows = []
    for c in ranked_candidates:
        rows.append(
            {
                "Rank": c.rank,
                "Candidate ID": c.candidate_id,
                "Overall Match Score (%)": c.overall_score,
                "Skill Score (0-1)": c.skill_score,
                "Text Similarity (0-1)": c.text_similarity,
                "Skill Coverage (%)": round(c.required_skill_coverage * 100.0, 1),
                "Jaccard Similarity": c.jaccard_similarity,
                "Matched Skills Count": c.matched_skill_count,
                "Missing Skills Count": c.missing_skill_count,
                "Matched Skills": ", ".join(c.matched_skills) if c.matched_skills else "None",
                "Missing Skills": ", ".join(c.missing_skills) if c.missing_skills else "None",
                "Additional Skills": ", ".join(c.additional_skills) if c.additional_skills else "None",
                "Explanation": c.explanation,
            }
        )

    df = pd.DataFrame(rows)
    return df
