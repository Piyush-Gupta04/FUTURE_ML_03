"""
src/skill_extraction.py
-----------------------
Explainable, ontology-based skill extraction and alias normalization.

Features:
- Maintainable taxonomy across 7 technical and interpersonal domains.
- Alias-to-canonical mapping (e.g. 'sklearn' -> 'Scikit-Learn', 'k8s' -> 'Kubernetes').
- Regex boundary preservation preventing false positives (e.g., 'R' inside 'car' or 'regular',
  'C++' and 'C#' without word-boundary truncation).
- Deterministic, zero-external-download NLP design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# --- 1. Comprehensive, Categorized Skill Taxonomy ---
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "programming_languages": [
        "Python",
        "R",
        "Java",
        "C++",
        "C#",
        "C",
        "Go",
        "Rust",
        "JavaScript",
        "TypeScript",
        "SQL",
        "Scala",
        "Kotlin",
        "Swift",
        "PHP",
        "Ruby",
        "Bash",
        "Shell",
        "MATLAB",
        "Julia",
        "HTML",
        "CSS",
    ],
    "machine_learning": [
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Large Language Models",
        "Generative AI",
        "Reinforcement Learning",
        "Supervised Learning",
        "Unsupervised Learning",
        "Neural Networks",
        "Transformers",
        "PyTorch",
        "TensorFlow",
        "Keras",
        "Scikit-Learn",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Hugging Face",
        "OpenCV",
        "spaCy",
        "NLTK",
        "FastAI",
        "LangChain",
        "LlamaIndex",
    ],
    "data_science": [
        "Data Science",
        "Data Analysis",
        "Data Engineering",
        "Data Pipelines",
        "ETL",
        "Pandas",
        "NumPy",
        "SciPy",
        "Matplotlib",
        "Seaborn",
        "Plotly",
        "Apache Spark",
        "PySpark",
        "Hadoop",
        "Kafka",
        "Airflow",
        "dbt",
        "BigQuery",
        "Snowflake",
        "Databricks",
        "Tableau",
        "Power BI",
        "Excel",
        "Statistics",
        "A/B Testing",
    ],
    "databases": [
        "PostgreSQL",
        "MySQL",
        "SQLite",
        "MongoDB",
        "Redis",
        "Cassandra",
        "DynamoDB",
        "Oracle Database",
        "Microsoft SQL Server",
        "Elasticsearch",
        "Neo4j",
        "Vector Databases",
    ],
    "cloud_devops": [
        "AWS",
        "Microsoft Azure",
        "Google Cloud Platform",
        "Docker",
        "Kubernetes",
        "Terraform",
        "CI/CD",
        "GitHub Actions",
        "GitLab CI",
        "Jenkins",
        "Ansible",
        "Linux",
        "Git",
        "MLflow",
        "DVC",
        "Kubeflow",
    ],
    "web_development": [
        "FastAPI",
        "Flask",
        "Django",
        "React",
        "Angular",
        "Vue",
        "Node.js",
        "Next.js",
        "REST API",
        "GraphQL",
        "Microservices",
        "Spring Boot",
        ".NET",
    ],
    "soft_skills": [
        "Problem Solving",
        "Communication",
        "Team Collaboration",
        "Leadership",
        "Critical Thinking",
        "Project Management",
        "Agile",
        "Scrum",
        "Mentorship",
        "Adaptability",
        "Time Management",
    ],
}


# --- 2. Skill Aliases and Acronym Normalization Map ---
# All alias keys must be lowercase for case-insensitive lookup.
SKILL_ALIASES: Dict[str, str] = {
    # Machine Learning / AI
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "large language model": "Large Language Models",
    "large language models": "Large Language Models",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "sklearn": "Scikit-Learn",
    "scikit-learn": "Scikit-Learn",
    "scikit learn": "Scikit-Learn",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "keras": "Keras",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "opencv": "OpenCV",
    "spacy": "spaCy",
    "nltk": "NLTK",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "neural networks": "Neural Networks",
    "neural network": "Neural Networks",
    "transformers": "Transformers",
    "transformer": "Transformers",
    # Data Science & Analytics
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "plotly": "Plotly",
    "pyspark": "PySpark",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "hadoop": "Hadoop",
    "kafka": "Kafka",
    "airflow": "Airflow",
    "apache airflow": "Airflow",
    "dbt": "dbt",
    "bigquery": "BigQuery",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "excel": "Excel",
    "ms excel": "Excel",
    "microsoft excel": "Excel",
    "data science": "Data Science",
    "data analysis": "Data Analysis",
    "data engineering": "Data Engineering",
    "data pipelines": "Data Pipelines",
    "data pipeline": "Data Pipelines",
    "etl": "ETL",
    "statistics": "Statistics",
    "statistical analysis": "Statistics",
    "a/b testing": "A/B Testing",
    "ab testing": "A/B Testing",
    # Programming Languages
    "python": "Python",
    "python3": "Python",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "golang": "Go",
    "go": "Go",
    "rust": "Rust",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "sql": "SQL",
    "scala": "Scala",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "php": "PHP",
    "ruby": "Ruby",
    "bash": "Bash",
    "shell": "Shell",
    "matlab": "MATLAB",
    "julia": "Julia",
    "html": "HTML",
    "html5": "HTML",
    "css": "CSS",
    "css3": "CSS",
    # Databases
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "sqlite": "SQLite",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "redis": "Redis",
    "cassandra": "Cassandra",
    "dynamodb": "DynamoDB",
    "oracle": "Oracle Database",
    "mssql": "Microsoft SQL Server",
    "sql server": "Microsoft SQL Server",
    "elasticsearch": "Elasticsearch",
    "neo4j": "Neo4j",
    "vector db": "Vector Databases",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    # Cloud & DevOps
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Microsoft Azure",
    "microsoft azure": "Microsoft Azure",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    "docker": "Docker",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "jenkins": "Jenkins",
    "ansible": "Ansible",
    "linux": "Linux",
    "git": "Git",
    "mlflow": "MLflow",
    "dvc": "DVC",
    "kubeflow": "Kubeflow",
    # Web & Frameworks
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "angular": "Angular",
    "vue": "Vue",
    "vuejs": "Vue",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "node": "Node.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "rest api": "REST API",
    "restful api": "REST API",
    "graphql": "GraphQL",
    "microservices": "Microservices",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    ".net": ".NET",
    "dotnet": ".NET",
    # Soft skills
    "problem solving": "Problem Solving",
    "communication": "Communication",
    "team collaboration": "Team Collaboration",
    "collaboration": "Team Collaboration",
    "teamwork": "Team Collaboration",
    "leadership": "Leadership",
    "critical thinking": "Critical Thinking",
    "project management": "Project Management",
    "agile": "Agile",
    "scrum": "Scrum",
    "mentorship": "Mentorship",
    "adaptability": "Adaptability",
    "time management": "Time Management",
}


# --- 3. Build Reverse Category Mapping ---
def _build_category_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for category, skill_list in SKILL_TAXONOMY.items():
        for skill in skill_list:
            mapping[skill] = category
    return mapping


SKILL_TO_CATEGORY: Dict[str, str] = _build_category_map()


# --- 4. Precompiled Regex Matchers ---
# Special handling for single-character or special-symbol terms (R, C, C++, C#, .NET)
SPECIAL_CASE_PATTERNS: Dict[str, re.Pattern] = {
    # 'R' language: standalone uppercase R, avoid matching in 'car', 'regular', or 'R&D'
    "R": re.compile(r"(?<![a-zA-Z0-9#+&])R(?![a-zA-Z0-9#+&])"),
    # 'C' language: standalone uppercase C, avoid matching in 'C&C' or hex/grades
    "C": re.compile(r"(?<![a-zA-Z0-9#+&])C(?![a-zA-Z0-9#+&])"),
    # 'C++' and 'cpp'
    "C++": re.compile(r"(?<![a-zA-Z0-9#+])(?:C\+\+|cpp)(?![a-zA-Z0-9#+])", re.IGNORECASE),
    # 'C#' and 'csharp'
    "C#": re.compile(r"(?<![a-zA-Z0-9#+])(?:C\#|csharp)(?![a-zA-Z0-9#+])", re.IGNORECASE),
    # '.NET' and 'dotnet'
    ".NET": re.compile(r"(?<![a-zA-Z0-9#+])(?:\.NET|dotnet)(?![a-zA-Z0-9#+])", re.IGNORECASE),
    # 'CI/CD' and 'cicd'
    "CI/CD": re.compile(r"(?<![a-zA-Z0-9#+])(?:CI/CD|cicd)(?![a-zA-Z0-9#+])", re.IGNORECASE),
}


def _compile_standard_patterns() -> List[tuple[str, str, re.Pattern]]:
    """
    Compile regex patterns for all canonical skills and aliases.
    Sorted by alias length in descending order to prioritize specific multi-word phrases.
    """
    patterns: List[tuple[str, str, re.Pattern]] = []

    # Aggregate canonical names and aliases
    term_to_canonical: Dict[str, str] = {}
    for cat, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            term_to_canonical[skill.lower()] = skill

    for alias, canonical in SKILL_ALIASES.items():
        term_to_canonical[alias.lower()] = canonical

    # Exclude special-cased symbols from generic compilation
    excluded_keys = {"r", "c", "c++", "cpp", "c#", "csharp", ".net", "dotnet", "ci/cd", "cicd"}

    sorted_terms = sorted(
        [t for t in term_to_canonical.keys() if t not in excluded_keys],
        key=len,
        reverse=True,
    )

    for term in sorted_terms:
        canonical = term_to_canonical[term]
        escaped_term = re.escape(term)
        # Word boundary safe for alphanumeric tokens
        pattern = re.compile(
            rf"(?<![a-zA-Z0-9_]){escaped_term}(?![a-zA-Z0-9_])",
            re.IGNORECASE,
        )
        patterns.append((term, canonical, pattern))

    return patterns


_COMPILED_PATTERNS = _compile_standard_patterns()


@dataclass
class SkillExtractionResult:
    """Structured result of skill extraction."""
    skills: List[str] = field(default_factory=list)
    categorized_skills: Dict[str, List[str]] = field(default_factory=dict)
    total_count: int = 0


def normalize_skill(skill_query: str) -> Optional[str]:
    """
    Return the canonical name for a given skill or alias string.
    Returns None if not recognized.
    """
    if not skill_query:
        return None
    cleaned = skill_query.strip().lower()
    if cleaned in SKILL_ALIASES:
        return SKILL_ALIASES[cleaned]
    # Check if cleaned directly matches a lowercase canonical skill
    for canonical in SKILL_TO_CATEGORY:
        if canonical.lower() == cleaned:
            return canonical
    return None


def get_skill_category(canonical_skill: str) -> str:
    """Return the taxonomic category for a canonical skill name."""
    return SKILL_TO_CATEGORY.get(canonical_skill, "other")


def extract_skills(text: str) -> List[str]:
    """
    Extract a unique, alphabetically sorted list of canonical skill names from text.

    Returns an empty list for empty or whitespace-only inputs.
    """
    if not text or not text.strip():
        return []

    matched_skills: Set[str] = set()

    # 1. Match special boundary cases first
    for canonical_name, pattern in SPECIAL_CASE_PATTERNS.items():
        if pattern.search(text):
            matched_skills.add(canonical_name)

    # 2. Match standard terms and multi-word phrases
    for _, canonical_name, pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            matched_skills.add(canonical_name)

    return sorted(list(matched_skills))


def extract_categorized_skills(text: str) -> Dict[str, List[str]]:
    """
    Extract skills and group them into their taxonomy categories.
    Categories with zero matched skills are omitted.
    """
    skills = extract_skills(text)
    categorized: Dict[str, List[str]] = {}

    for skill in skills:
        cat = get_skill_category(skill)
        categorized.setdefault(cat, []).append(skill)

    for cat in categorized:
        categorized[cat] = sorted(categorized[cat])

    return categorized


def extract_skills_summary(text: str) -> SkillExtractionResult:
    """
    Produce a complete SkillExtractionResult container.
    """
    skills = extract_skills(text)
    categorized = extract_categorized_skills(text)
    return SkillExtractionResult(
        skills=skills,
        categorized_skills=categorized,
        total_count=len(skills),
    )
