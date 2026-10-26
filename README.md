# FUTURE_ML_03: Resume / Candidate Screening System

**Future Interns Machine Learning Internship — Task 3**

An original, explainable resume screening system that evaluates candidate resumes against a user-provided job description, computes explainable match scores using required-skill coverage and TF-IDF cosine similarity, identifies matched and missing skills, and ranks candidates objectively based on textual and skill relevance.

---

## 1. Project Overview & Objectives

In modern recruitment workflows, manual resume screening can be time-intensive and prone to subjective human fatigue. This project implements a deterministic, explainable screening engine designed to assist human recruiters by computing transparent lexical and skill alignment.

### Key Objectives
* **Resume Text Extraction**: In-memory parsing of PDF (via PyMuPDF) and plain text (`.txt`) resumes without writing uploaded files to disk.
* **Text Preprocessing & Anonymization**: Whitespace normalization, Unicode standardisation, and heuristic redaction of contact details (emails, phone numbers, URLs).
* **Ontology-Based Skill Extraction**: Curated, maintainable dictionary of ~150+ technical and interpersonal skills across 7 domains with alias normalization and regex word-boundary isolation (e.g., matching `R` without triggering false positives inside words like `car` or `regular`).
* **User-Provided Job Description**: Accepts arbitrary job requirements directly from the recruiter and extracts required skills on the fly.
* **Semantic & Skill Matching**:
  * Required-Skill Coverage (Recall) and Jaccard similarity.
  * TF-IDF vectorization with unigram and bigram features (`ngram_range=(1, 2)`) and cosine similarity.
* **Configurable Composite Score**: Weighted combination of skill alignment and textual semantic similarity.
* **Deterministic Ranking & Explanations**: Orders candidates in descending order with multi-tier tie-breaking and generates human-readable score explanations detailing matched, missing, and additional skills.
* **Extensible Architecture**: Modular Python package with comprehensive automated testing and a walkthrough Jupyter notebook.

---

## 2. Dataset Attribution & Local Setup

### Kaggle Resume Dataset
This project uses the publicly available **Kaggle Resume Dataset** (Snehan Kekre / Kaggle community).
* **Important**: The dataset is **not** included in this repository and must be downloaded separately to protect privacy and minimize repository size.
* **Default Expected Location (Windows)**:
  ```text
  %USERPROFILE%\Downloads\resume dataset\
      ├── Resume\
      │   └── Resume.csv
      └── data\
          └── data\
              ├── INFORMATION-TECHNOLOGY\
              ├── ENGINEERING\
              ├── FINANCE\
              └── ... (24 category folders containing individual PDFs)
  ```
* **Custom Environment Variable**:
  Alternatively, you can point to any custom dataset path:
  ```powershell
  $env:KAGGLE_RESUME_DATASET_DIR = "C:\path\to\your\resume dataset"
  ```

### Verified Dataset Schema (`Resume.csv`)
Inspected and verified using Pandas across **2,484 records**:

| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `ID` | `int64` | `0` | Unique numerical identifier for each resume (0 duplicates). |
| `Resume_str` | `string` | `0` | Full plain-text resume content (primary input for text screening). |
| `Resume_html` | `string` | `0` | HTML formatted resume structure. |
| `Category` | `string` | `0` | Occupational category metadata across 24 disciplines (e.g., `INFORMATION-TECHNOLOGY`, `ENGINEERING`). |

> **Note on Category Metadata**: The `Category` column serves strictly as descriptive metadata for filtering or inspection. It is **never** utilized as an input feature for candidate scoring or ranking.

---

## 3. Project Architecture

```text
FUTURE_ML_03/
├── app.py                     # Streamlit interactive candidate screening dashboard
├── data/
│   └── README.md              # Kaggle dataset setup instructions & privacy guide
├── notebooks/
│   └── resume_screening_walkthrough.ipynb # End-to-end interactive demonstration
├── src/
│   ├── __init__.py            # Package metadata
│   ├── dataset.py             # Dynamic Kaggle dataset resolution and safe loader
│   ├── preprocessing.py       # In-memory PDF/TXT extraction, cleaning, PII redaction
│   ├── skill_extraction.py    # Ontology, alias normalization, regex boundary matchers
│   ├── matching.py            # TF-IDF cosine similarity, skill overlap metrics, scoring
│   └── ranking.py             # Descending candidate ranking, tie-breaking, explanations
├── tests/
│   ├── __init__.py
│   ├── test_app_helpers.py    # Streamlit dashboard formatting and upload tests
│   ├── test_dataset.py        # Kaggle dataset discovery and schema validation tests
│   ├── test_end_to_end.py     # End-to-end integration tests using inline text strings
│   ├── test_matching.py       # Scoring formula, weight validation, edge case tests
│   ├── test_preprocessing.py  # In-memory PDF extraction, cleaning, contact redaction tests
│   ├── test_ranking.py        # Descending sort, tie-breaking, DataFrame export tests
│   └── test_skill_extraction.py # Token boundaries, alias mapping, false-positive tests
├── requirements.txt           # Project dependencies
├── LICENSE                    # MIT License
└── README.md                  # This documentation
```

---

## 4. Screening Methodology & Scoring Formula

For any candidate resume $R$ evaluated against a user-provided job description $J$:

### Step 1: Skill Extraction & Gap Analysis
Let $\mathcal{S}_J$ denote skills extracted from the job description, and $\mathcal{S}_R$ denote skills extracted from the resume:
* **Matched Skills**: $\mathcal{S}_{\text{matched}} = \mathcal{S}_R \cap \mathcal{S}_J$
* **Missing Required Skills**: $\mathcal{S}_{\text{missing}} = \mathcal{S}_J \setminus \mathcal{S}_R$
* **Additional Candidate Skills**: $\mathcal{S}_{\text{additional}} = \mathcal{S}_R \setminus \mathcal{S}_J$

### Step 2: Skill Alignment Score ($S_{\text{skill}}$)
Combines required-skill recall (coverage) with Jaccard similarity across declared skills:
$$\text{Coverage}(R, J) = \frac{|\mathcal{S}_R \cap \mathcal{S}_J|}{|\mathcal{S}_J|} \quad (\text{if } |\mathcal{S}_J| > 0, \text{ else } 0.0)$$
$$\text{Jaccard}(R, J) = \frac{|\mathcal{S}_R \cap \mathcal{S}_J|}{|\mathcal{S}_R \cup \mathcal{S}_J|} \quad (\text{if } |\mathcal{S}_R \cup \mathcal{S}_J| > 0, \text{ else } 0.0)$$
$$S_{\text{skill}} = 0.8 \times \text{Coverage}(R, J) + 0.2 \times \text{Jaccard}(R, J)$$

### Step 3: Semantic Text Similarity ($S_{\text{text}}$)
Resume and job description texts are transformed using Scikit-learn's `TfidfVectorizer` configured with:
* Unigram and bigram features: `ngram_range=(1, 2)`
* English stop-word filtering
* Sublinear term frequency scaling (`sublinear_tf=True`)
Cosine similarity is computed between the resulting vectors and bounded to $[0.0, 1.0]$:
$$S_{\text{text}} = \text{CosineSimilarity}(\vec{v}_R, \vec{v}_J) = \frac{\vec{v}_R \cdot \vec{v}_J}{\|\vec{v}_R\|_2 \|\vec{v}_J\|_2}$$

### Step 4: Overall Composite Match Score
$$\text{Overall Score} = \left( w_{\text{skill}} \times S_{\text{skill}} + w_{\text{text}} \times S_{\text{text}} \right) \times 100\%$$
* Default weights: $w_{\text{skill}} = 0.60$, $w_{\text{text}} = 0.40$ (weights must be non-negative and sum to $1.0$).
* Bounded mathematically between $0.0\%$ and $100.0\%$.

### Step 5: Deterministic Ranking & Tie-Breaking
Candidates are sorted using the hierarchy:
1. Higher `Overall Score` (descending)
2. Higher `Skill Score` (descending)
3. Higher `Text Similarity` (descending)
4. `Candidate ID` string comparison (ascending alphabetical, ensures strict determinism)

---

## 5. Installation & Setup Instructions (Windows)

### Prerequisites
* Windows 10/11
* Python 3.11, 3.12, 3.13, or 3.14 (Verified on Python 3.14.7 64-bit)
* Git

### Step-by-Step Setup
1. **Clone or navigate to the repository**:
   ```powershell
   cd "C:\Users\piyus\OneDrive\Desktop\code\intership project\FUTURE_ML_03"
   ```

2. **(Optional) Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

---

## 6. How to Run the Streamlit Dashboard

The web dashboard (`app.py`) provides an interactive interface to evaluate resumes against user-provided job descriptions.

### Launch Command
From the project root directory, run:
```powershell
python -m streamlit run app.py
```
Open your browser to the local URL (default: `http://localhost:8501`).

### Dashboard Features
1. **User-Entered Job Description**: Text area to enter custom role requirements. Automatically detects and displays required skills grouped by technical category.
2. **Kaggle Resume Dataset Screening**:
   * Filter resumes by professional Category (e.g., `INFORMATION-TECHNOLOGY`, `ENGINEERING`).
   * Select candidate sample size (5 to 50 resumes) for responsive screening.
3. **In-Memory File Uploads (Optional)**:
   * Upload multiple `.pdf` or `.txt` resumes.
   * Parsed in memory using PyMuPDF (`fitz`) with zero disk persistence.
4. **Scoring Weight Customization**:
   * Adjustable slider for Skill Alignment Weight ($w_{\text{skill}}$) and Semantic Text Similarity Weight ($w_{\text{text}}$), automatically normalized to sum to $1.0$.
5. **Interactive Results**:
   * KPI cards for Total Screened, Top Score, and Average Score.
   * Full candidate leaderboard table sorted descending by match score.
   * Detailed drilldown cards for each candidate showing exact matched skills, missing required skills, and natural-language explanations.
6. **Privacy-Safe CSV Export**:
   * One-click download button for `screening_results.csv`.
   * Exports only rank, candidate ID, scores, and skill lists. **Never exports raw resume text or personal contact info**.

---

## 7. How to Run the Walkthrough Notebook

The Jupyter notebook demonstrates the full screening workflow against the local Kaggle dataset.

1. **Launch Jupyter**:
   ```powershell
   python -m pip install jupyter
   python -m jupyter notebook notebooks/resume_screening_walkthrough.ipynb
   ```
2. **Execution Steps Demonstrated in Notebook**:
   * Dynamically loads resumes from `Resume.csv`.
   * Accepts a user-provided job description string.
   * Preprocesses resume text and heuristically masks contact information.
   * Extracts recognized skills and maps aliases to canonical names.
   * Computes TF-IDF cosine similarity and skill overlap metrics.
   * Ranks candidates deterministically and displays a breakdown table with natural language explanations.
   * Demonstrates in-memory extraction of a sample PDF from the Kaggle dataset without writing to disk.

---

## 8. Automated Test Suite & Results

The test suite covers unit and integration tests using small inline text strings and verified schema loaders. No external test files are required.

### Running Tests
Execute the full test suite with verbose output:
```powershell
python -m pytest -v
```

### Actual Test Results (Verified)
* **Total Tests Executed**: **56**
* **Total Passed**: **56**
* **Total Failed**: **0**
* **Execution Time**: ~8 seconds

```text
tests/test_app_helpers.py (6 tests: upload handling, dataset parsing, export filtering) PASSED
tests/test_dataset.py (3 tests: discovery, schema validation, category filtering) PASSED
tests/test_end_to_end.py (2 tests: pipeline integration, cross-role sensitivity) PASSED
tests/test_matching.py (13 tests: weight validation, skill overlap, cosine bounds, edge cases) PASSED
tests/test_preprocessing.py (14 tests: in-memory PDF/TXT parsing, cleaning, contact redaction) PASSED
tests/test_ranking.py (5 tests: descending ranking, deterministic ties, explanations, DataFrame export) PASSED
tests/test_skill_extraction.py (13 tests: single/multi-word skills, boundaries, aliases) PASSED
```

---

## 8. Privacy Safeguards, Limitations & Responsible Use

### Data Protection & Privacy Safeguards
1. **Zero Git Inclusion**: The dataset contains real candidate career histories and is strictly excluded from Git via `.gitignore` (`Resume.csv`, `*.csv`, `data/data/`, `data/kaggle/`, `uploads/`).
2. **Ephemeral In-Memory Processing**: Resumes (PDF or TXT) are read directly in memory (`io.BytesIO`) and are **never** persisted to disk.
3. **Contact Information Masking**: Preprocessing masks phone numbers, email addresses, and web profile URLs using documented regex patterns (`[EMAIL]`, `[PHONE]`, `[URL]`).
4. **No Demographic Inputs**: The screening and ranking algorithms do not accept, extract, or evaluate names, gender, age, race, religion, photos, or other sensitive personal characteristics.

### System Limitations
* **Advisory Tool Only**: Match scores reflect **lexical and skill keyword alignment** against the specific job description provided. They do **not** measure candidate intelligence, interpersonal competence, work ethic, or hiring suitability.
* **Heuristic PII Redaction Disclaimer**: Regular-expression contact redaction reduces visible contact details but does not guarantee 100% comprehensive anonymization of subtle narrative identifiers.
* **Scanned PDFs**: PDFs containing purely flattened raster images without an embedded text layer cannot be parsed without optical character recognition (OCR).
* **Ontology Boundaries**: Highly specialized, proprietary internal tools outside the ~150+ categorized skill vocabulary will not be recognized as structured skills, though their textual presence contributes to TF-IDF similarity.

---

## 9. License

This project is licensed under the terms of the [MIT License](file:///c:/Users/piyus/OneDrive/Desktop/code/intership%20project/FUTURE_ML_03/LICENSE).
