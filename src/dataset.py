"""
src/dataset.py
--------------
Dataset discovery, dynamic path resolution, and safe loading utilities
for local benchmark datasets (e.g., the Kaggle Resume Dataset).

Design:
- Configurable via environment variable or default user download locations.
- Zero hardcoding of user directories or usernames.
- Graceful fallback when external datasets are not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Environment variable name for custom dataset directory override
ENV_DATASET_DIR = "KAGGLE_RESUME_DATASET_DIR"


def resolve_kaggle_dataset_dir() -> Optional[Path]:
    """
    Dynamically discover the location of the Kaggle Resume Dataset without
    hardcoding any specific user profiles or absolute usernames.

    Search priority:
    1. Explicit environment variable: KAGGLE_RESUME_DATASET_DIR
    2. User home download directory: Path.home() / 'Downloads' / 'resume dataset'
    3. Workspace-relative data directory: Path('data/kaggle')
    """
    # 1. Environment variable override
    env_dir = os.environ.get(ENV_DATASET_DIR)
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p

    # 2. User home downloads folder
    home_downloads = Path.home() / "Downloads" / "resume dataset"
    if home_downloads.exists() and home_downloads.is_dir():
        return home_downloads.resolve()

    # 3. Workspace-relative folder
    local_kaggle = Path("data") / "kaggle"
    if local_kaggle.exists() and local_kaggle.is_dir():
        return local_kaggle.resolve()

    return None


def get_kaggle_csv_path() -> Optional[Path]:
    """
    Return the Path to Resume.csv if discovered, or None if unavailable.
    """
    dataset_dir = resolve_kaggle_dataset_dir()
    if not dataset_dir:
        return None

    # Check common subfolder structures
    candidates = [
        dataset_dir / "Resume" / "Resume.csv",
        dataset_dir / "Resume.csv",
        dataset_dir / "resume.csv",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


def get_kaggle_pdf_dir() -> Optional[Path]:
    """
    Return the Path to the extracted PDF category folders, or None if unavailable.
    """
    dataset_dir = resolve_kaggle_dataset_dir()
    if not dataset_dir:
        return None

    candidates = [
        dataset_dir / "data" / "data",
        dataset_dir / "data",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    return None


def load_kaggle_resumes(
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Safely load resume records from Resume.csv.

    Parameters:
        limit: Maximum number of records to return (optional).
        categories: List of categories to filter by (e.g. ['INFORMATION-TECHNOLOGY', 'ENGINEERING']).

    Returns:
        (DataFrame with columns ['ID', 'Category', 'Resume_str'], status_message)
    """
    csv_path = get_kaggle_csv_path()
    if not csv_path:
        return (
            pd.DataFrame(columns=["ID", "Category", "Resume_str"]),
            "Kaggle Resume Dataset not found at configured search paths.",
        )

    try:
        # Load verified schema columns: ID, Resume_str, Resume_html, Category
        df = pd.read_csv(csv_path)

        if categories:
            upper_cats = [c.upper() for c in categories]
            df = df[df["Category"].str.upper().isin(upper_cats)]

        if limit is not None and limit > 0:
            df = df.head(limit)

        msg = f"Successfully loaded {len(df)} resumes from {csv_path.name}."
        return df, msg
    except Exception as exc:
        return (
            pd.DataFrame(columns=["ID", "Category", "Resume_str"]),
            f"Error loading {csv_path}: {str(exc)}",
        )
