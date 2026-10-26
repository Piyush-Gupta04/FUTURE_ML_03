"""
tests/test_dataset.py
---------------------
Tests for dataset discovery, path resolution, and safe loading of local Kaggle resumes.
"""

from pathlib import Path
import pytest

from src.dataset import (
    get_kaggle_csv_path,
    get_kaggle_pdf_dir,
    load_kaggle_resumes,
    resolve_kaggle_dataset_dir,
)


def test_resolve_dataset_dir_runs_without_error():
    dataset_dir = resolve_kaggle_dataset_dir()
    # It either discovers the directory or returns None gracefully
    if dataset_dir is not None:
        assert isinstance(dataset_dir, Path)
        assert dataset_dir.exists()


def test_load_kaggle_resumes_returns_expected_schema():
    df, status = load_kaggle_resumes(limit=3)
    assert "ID" in df.columns
    assert "Category" in df.columns
    assert "Resume_str" in df.columns
    assert isinstance(status, str)

    # If dataset is present on this machine, verify rows
    if resolve_kaggle_dataset_dir() is not None:
        assert len(df) == 3
        assert df["ID"].dtype in ["int64", "int32"]


def test_load_kaggle_resumes_category_filter():
    dataset_dir = resolve_kaggle_dataset_dir()
    if dataset_dir is None:
        pytest.skip("Kaggle dataset not found on host machine.")

    df, status = load_kaggle_resumes(limit=5, categories=["INFORMATION-TECHNOLOGY"])
    assert len(df) > 0
    assert all(df["Category"] == "INFORMATION-TECHNOLOGY")
