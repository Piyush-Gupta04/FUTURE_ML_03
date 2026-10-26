"""
app.py
------
Streamlit Interactive Dashboard for Task 3: Resume / Candidate Screening System.

Features:
- Screen candidates against a user-entered job description.
- Load resumes from the local Kaggle Resume Dataset (with category filtering and sample limit).
- Support in-memory PDF/TXT resume uploads (zero disk persistence).
- Compute explainable match scores decomposed into skill coverage, missing skills, and TF-IDF similarity.
- Export privacy-safe ranking results to CSV (zero raw resume text or PII).
- Responsible AI disclaimer.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.dataset import get_kaggle_csv_path, load_kaggle_resumes, resolve_kaggle_dataset_dir
from src.matching import DEFAULT_SKILL_WEIGHT, DEFAULT_TEXT_WEIGHT, validate_weights
from src.preprocessing import clean_text, extract_text_from_file
from src.ranking import candidates_to_dataframe, rank_candidates
from src.skill_extraction import extract_categorized_skills, extract_skills


# --- Helper Functions (Modular & Testable) ---

def prepare_candidates_from_kaggle(df: pd.DataFrame) -> Dict[str, str]:
    """
    Transform Kaggle dataset rows into a candidate_id -> resume_text mapping.
    Uses 'Resume_str' for text. 'Category' is strictly metadata for the ID label.
    """
    candidates: Dict[str, str] = {}
    if df.empty or "ID" not in df.columns or "Resume_str" not in df.columns:
        return candidates

    for _, row in df.iterrows():
        cand_id = f"Candidate_{row['ID']} ({row.get('Category', 'N/A')})"
        candidates[cand_id] = str(row["Resume_str"])
    return candidates


def prepare_candidates_from_uploads(
    uploaded_files: list,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse uploaded PDF/TXT files in memory without saving to disk.
    Assigns neutral candidate IDs (e.g. Uploaded_Candidate_001) to protect
    candidate privacy and ensure personal filenames never leak into exported
    screening results.
    Returns (candidates_dict, warnings_list).
    """
    candidates: Dict[str, str] = {}
    warnings: List[str] = []

    for idx, file_obj in enumerate(uploaded_files, start=1):
        cand_id = f"Uploaded_Candidate_{idx:03d}"
        filename = getattr(file_obj, "name", f"file_{idx}")
        try:
            content_bytes = file_obj.getvalue()
            extraction = extract_text_from_file(content_bytes, filename=filename)

            if extraction.is_empty:
                warnings.append(f"'{filename}' contained no extractable text.")
                candidates[cand_id] = ""
            else:
                candidates[cand_id] = extraction.text
                if extraction.warnings:
                    warnings.extend([f"'{filename}': {w}" for w in extraction.warnings])
        except Exception as exc:
            warnings.append(f"Error parsing '{filename}': {str(exc)}")

    return candidates, warnings


def format_export_dataframe(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the ranking DataFrame for export, ensuring NO raw resume text,
    emails, phone numbers, or personal details are included.
    """
    safe_columns = [
        "Rank",
        "Candidate ID",
        "Overall Match Score (%)",
        "Skill Score (0-1)",
        "Text Similarity (0-1)",
        "Skill Coverage (%)",
        "Matched Skills Count",
        "Missing Skills Count",
        "Matched Skills",
        "Missing Skills",
        "Explanation",
    ]
    present_cols = [c for c in safe_columns if c in ranked_df.columns]
    return ranked_df[present_cols].copy()


# --- Streamlit Application UI ---

def main():
    st.set_page_config(
        page_title="Resume Screening System | Task 3",
        page_icon="📄",
        layout="wide",
    )

    # Header & Purpose
    st.title("📄 Resume / Candidate Screening System")
    st.caption("Future Interns Machine Learning Internship — Task 3 | Explainable Candidate Alignment")

    # Responsible AI & Privacy Disclaimer Banner
    st.warning(
        "⚖️ **Responsible AI & Advisory Disclaimer**: "
        "Match scores indicate textual and skill keyword alignment against the provided job description. "
        "They do **not** measure candidate hiring suitability, competence, work ethic, or character. "
        "Candidate evaluation relies purely on declared skills and text content; no personal demographic data is used."
    )

    # Sidebar: Input Source & Configuration
    with st.sidebar:
        st.header("⚙️ Screening Configuration")

        input_mode = st.radio(
            "Select Resume Source:",
            ["Kaggle Resume Dataset", "Upload Resumes (PDF / TXT)"],
            index=0,
            help="Choose whether to evaluate resumes from the local Kaggle dataset or upload files.",
        )

        st.markdown("---")
        st.subheader("Scoring Weights")
        skill_weight = st.slider(
            "Skill Alignment Weight",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_SKILL_WEIGHT,
            step=0.05,
            help="Weight allocated to required-skill recall and Jaccard overlap.",
        )
        text_weight = round(1.0 - skill_weight, 2)
        st.caption(f"Semantic Text Similarity Weight: **{text_weight:.2f}** (Auto-balanced to sum to 1.0)")

        candidates_dict: Dict[str, str] = {}
        dataset_warning_msg = ""

        if input_mode == "Kaggle Resume Dataset":
            st.markdown("---")
            st.subheader("Kaggle Dataset Filter")
            kaggle_dir = resolve_kaggle_dataset_dir()

            if not kaggle_dir:
                st.error("❌ Kaggle Resume Dataset not found.")
                st.info(
                    "Please extract the dataset to:\n"
                    "`%USERPROFILE%\\Downloads\\resume dataset\\Resume\\Resume.csv`"
                )
                dataset_warning_msg = "Kaggle Resume Dataset is not available at the expected path."
            else:
                st.success(f"✓ Dataset detected at:\n`{kaggle_dir.name}`")

                # Sample category options from Kaggle dataset
                available_categories = [
                    "INFORMATION-TECHNOLOGY",
                    "ENGINEERING",
                    "FINANCE",
                    "ACCOUNTANT",
                    "SALES",
                    "HEALTHCARE",
                    "HR",
                    "BUSINESS-DEVELOPMENT",
                    "CONSULTANT",
                ]

                selected_categories = st.multiselect(
                    "Filter by Category:",
                    available_categories,
                    default=["INFORMATION-TECHNOLOGY", "ENGINEERING"],
                    help="Filter resumes from specific occupational sectors.",
                )

                sample_limit = st.slider(
                    "Resumes to Screen:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Limit the number of resumes loaded for responsive interactive screening.",
                )

                if st.button("Load Dataset Resumes", width="stretch"):
                    with st.spinner("Loading resumes from Resume.csv..."):
                        df, msg = load_kaggle_resumes(limit=sample_limit, categories=selected_categories)
                        if not df.empty:
                            st.session_state["kaggle_candidates"] = prepare_candidates_from_kaggle(df)
                            st.success(f"Loaded {len(st.session_state['kaggle_candidates'])} resumes.")
                        else:
                            st.warning(msg)

                if "kaggle_candidates" in st.session_state:
                    candidates_dict = st.session_state["kaggle_candidates"]

        else:
            st.markdown("---")
            st.subheader("Upload Candidate Resumes")
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Uploaded files are processed entirely in-memory and never saved to disk.",
            )

            if uploaded_files:
                candidates_dict, upload_warnings = prepare_candidates_from_uploads(uploaded_files)
                for w in upload_warnings:
                    st.warning(w)
                st.info(f"Loaded {len(candidates_dict)} uploaded resumes in-memory.")

    # Main Area: Job Description Input
    col_jd, col_info = st.columns([2, 1])

    with col_jd:
        st.subheader("1. Enter Job Description")
        job_title = st.text_input("Target Job Title (Optional):", value="Machine Learning & Python Engineer")
        job_description_input = st.text_area(
            "Paste Job Description / Requirements:",
            height=200,
            value=(
                "We are hiring a Machine Learning Engineer.\n\n"
                "Key Requirements:\n"
                "- Strong proficiency in Python and SQL.\n"
                "- Hands-on experience with Machine Learning and Deep Learning frameworks (PyTorch or TensorFlow, Scikit-Learn).\n"
                "- Experience containerizing applications with Docker and orchestration with Kubernetes.\n"
                "- Cloud deployment experience using AWS.\n"
                "- Solid knowledge of CI/CD, Git, and Linux.\n"
                "- Strong problem solving and communication skills."
            ),
            help="Enter the technical requirements and responsibilities for the target role.",
        )

    with col_info:
        st.subheader("2. Detected Requirements")
        if job_description_input.strip():
            cleaned_jd = clean_text(job_description_input, redact=False)
            jd_skills = extract_skills(cleaned_jd)
            st.metric("Required Skills Identified", len(jd_skills))
            if jd_skills:
                categorized = extract_categorized_skills(cleaned_jd)
                for cat, skills in categorized.items():
                    st.markdown(f"**{cat.replace('_', ' ').title()}**: " + ", ".join(skills))
            else:
                st.info("No specific technical skills from our ontology were detected. Ranking will rely on semantic text similarity.")
        else:
            st.info("Enter a job description to preview extracted skills.")

    st.markdown("---")

    # Screening Action
    st.subheader("3. Screen & Rank Candidates")

    if dataset_warning_msg and input_mode == "Kaggle Resume Dataset" and not candidates_dict:
        st.error(dataset_warning_msg)
        return

    if not candidates_dict:
        if input_mode == "Kaggle Resume Dataset":
            st.info("👈 Click **'Load Dataset Resumes'** in the sidebar to load candidates from Resume.csv.")
        else:
            st.info("👈 Upload PDF or TXT resumes using the sidebar uploader.")
        return

    st.write(f"Ready to screen **{len(candidates_dict)}** candidate resumes against the job description.")

    if st.button("🚀 Run Screening Analysis", type="primary", width="stretch"):
        if not job_description_input.strip():
            st.error("Please enter a job description before running screening.")
            return

        with st.spinner("Analyzing resumes, extracting skills, and calculating match scores..."):
            # Execute explainable ranking
            ranked_candidates = rank_candidates(
                candidates_data=candidates_dict,
                job_description=job_description_input,
                skill_weight=skill_weight,
                text_weight=text_weight,
            )
            ranking_df = candidates_to_dataframe(ranked_candidates)

        st.success(f"Successfully screened and ranked {len(ranked_candidates)} candidates!")

        # KPI Summary Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        top_score = ranked_candidates[0].overall_score if ranked_candidates else 0.0
        avg_score = round(sum(c.overall_score for c in ranked_candidates) / len(ranked_candidates), 1) if ranked_candidates else 0.0
        top_cand_id = ranked_candidates[0].candidate_id if ranked_candidates else "None"

        kpi1.metric("Screened Resumes", len(ranked_candidates))
        kpi2.metric("Top Match Score", f"{top_score:.1f}%")
        kpi3.metric("Average Score", f"{avg_score:.1f}%")
        kpi4.metric("Top Candidate", top_cand_id.split()[0])

        # Leaderboard Table
        st.subheader("🏆 Candidate Leaderboard")
        display_columns = [
            "Rank",
            "Candidate ID",
            "Overall Match Score (%)",
            "Skill Score (0-1)",
            "Text Similarity (0-1)",
            "Skill Coverage (%)",
            "Matched Skills Count",
            "Missing Skills Count",
        ]
        st.dataframe(ranking_df[display_columns], width="stretch")

        # CSV Export (Privacy-Safe: Zero PII or resume text)
        export_df = format_export_dataframe(ranking_df)
        csv_buffer = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Screening Results (CSV)",
            data=csv_buffer,
            file_name="screening_results.csv",
            mime="text/csv",
            help="Download explainable ranking metrics. Contains no raw resume text or personal contact info.",
        )

        st.markdown("---")

        # Detailed Candidate Drilldown Cards
        st.subheader("🔍 Detailed Candidate Breakdown & Explanations")
        for cand in ranked_candidates[:10]:
            with st.expander(f"Rank {cand.rank}: {cand.candidate_id} — {cand.overall_score:.1f}% Match", expanded=(cand.rank == 1)):
                col_score, col_exp = st.columns([1, 2])

                with col_score:
                    st.metric("Composite Score", f"{cand.overall_score:.1f}%")
                    st.write(f"- **Skill Score**: {cand.skill_score:.3f} (Weight: {skill_weight:.2f})")
                    st.write(f"- **Text Similarity**: {cand.text_similarity:.3f} (Weight: {text_weight:.2f})")
                    st.write(f"- **Skill Coverage**: {cand.required_skill_coverage * 100.0:.1f}%")

                with col_exp:
                    st.markdown(f"**Explanation**: {cand.explanation}")

                    col_m, col_g = st.columns(2)
                    with col_m:
                        st.markdown("**Matched Skills (Present in Resume & JD):**")
                        if cand.matched_skills:
                            st.success(", ".join(cand.matched_skills))
                        else:
                            st.write("None")

                    with col_g:
                        st.markdown("**Missing Required Skills (In JD, Missing in Resume):**")
                        if cand.missing_skills:
                            st.error(", ".join(cand.missing_skills))
                        else:
                            st.write("None (Full Coverage)")


if __name__ == "__main__":
    main()
