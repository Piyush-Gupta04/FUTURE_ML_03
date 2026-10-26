# Kaggle Resume Dataset & Privacy Guidelines

## 1. Overview
This project strictly utilizes the **Kaggle Resume Dataset** for candidate screening. No synthetic resumes or external secondary datasets are created or included.

The dataset is located locally on your workstation outside of the Git repository:
- **Primary CSV File**: `%USERPROFILE%\Downloads\resume dataset\Resume\Resume.csv`
- **Extracted PDF Directory**: `%USERPROFILE%\Downloads\resume dataset\data\data\`

---

## 2. Dataset Schema & Metadata

Verified schema from `Resume.csv` (2,484 rows $\times$ 4 columns, 0 null values):

| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `ID` | `int64` | `0` | Unique numerical identifier for each resume. |
| `Resume_str` | `string` | `0` | Raw text of the resume (used as the primary resume text). |
| `Resume_html` | `string` | `0` | HTML formatted resume structure. |
| `Category` | `string` | `0` | Occupational domain metadata (24 categories, e.g., `INFORMATION-TECHNOLOGY`, `ENGINEERING`, `FINANCE`). |

*Note: The `Category` field serves strictly as category metadata for filtering or inspection. It is never used as an input to the candidate match score.*

---

## 3. Local Setup Instructions

If running on a new environment, download and extract the Kaggle Resume Dataset to:
```text
%USERPROFILE%\Downloads\resume dataset\
    ├── Resume\
    │   └── Resume.csv
    └── data\
        └── data\
            ├── INFORMATION-TECHNOLOGY\
            ├── ENGINEERING\
            └── ...
```

Alternatively, specify a custom directory by setting the `KAGGLE_RESUME_DATASET_DIR` environment variable:
```powershell
$env:KAGGLE_RESUME_DATASET_DIR = "C:\path\to\your\resume dataset"
```

If the Kaggle dataset is not found at any configured location, the system raises a clear setup error with explicit instructions. It will **never** silently fall back to synthetic data.

---

## 4. Privacy & Responsible AI Guidelines

1. **Zero Git Inclusion**: The dataset contains real candidate career histories and must remain excluded from Git version control. This is enforced via `.gitignore` (`Resume.csv`, `*.csv`, `data/data/`, `data/kaggle/`).
2. **Ephemeral In-Memory PDF/TXT Parsing**: Uploaded PDF or text files are parsed in memory via `PyMuPDF` (`io.BytesIO`) without saving resumes to disk.
3. **Contact Redaction**: Heuristic regular expressions mask telephone numbers, emails, and URLs during preprocessing.
4. **Advisory Decision Support**: Scores measure text and skill keyword alignment against a user-provided job description. They do not evaluate candidate hiring suitability, character, or competence.
