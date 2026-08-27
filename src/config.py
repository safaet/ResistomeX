"""
Central configuration for the R-Blend AMR Predictor.

One R-Blend model is trained per dataset found in `Data/Read Data/`
(12 pathogen-antibiotic combinations). All models share the same
hyper-parameters (reproduced from the research notebook's "best configuration");
only the data and the frozen feature schema differ.

No Google Drive / Colab paths anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Primary source of training data (the user's curated CSVs), plus an optional
# secondary drop location that is checked too.
SOURCE_DATA_DIRS: tuple[Path, ...] = (
    ROOT_DIR / "Data" / "Read Data",
    ROOT_DIR / "data" / "raw",
)

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLES_DIR: Path = DATA_DIR / "samples"
MODELS_DIR: Path = ROOT_DIR / "models"
DOCS_DIR: Path = ROOT_DIR / "docs"
REGISTRY_PATH: Path = MODELS_DIR / "registry.json"

MODEL_VERSION: str = "v1"
DEFAULT_MODEL_ID: str = "meropenem-kn"

# --------------------------------------------------------------------------- #
# Dataset / model naming
# --------------------------------------------------------------------------- #
# Filenames look like  amr_ast_<antibiotic>_<PATHOGENCODE>.csv
_FILENAME_RE = re.compile(r"^amr_ast_(?P<abx>.+)_(?P<code>[A-Za-z]+)$")

PATHOGEN_NAMES: dict[str, str] = {
    "KN": "Klebsiella pneumoniae",
    "ECS": "Escherichia coli / Shigella",
    "PA": "Pseudomonas aeruginosa",
    "SE": "Salmonella enterica",
    "CJ": "Campylobacter jejuni",
}


def parse_dataset_name(path: Path | str) -> dict:
    """amr_ast_meropenem_KN.csv -> dict(model_id, antibiotic, pathogen_code, pathogen, label)."""
    stem = Path(path).stem
    m = _FILENAME_RE.match(stem)
    if not m:
        # Fallback: last token = pathogen code, second-to-last = antibiotic.
        parts = stem.split("_")
        abx, code = (parts[-2] if len(parts) >= 2 else stem), (parts[-1] if parts else "NA")
    else:
        abx, code = m.group("abx"), m.group("code")
    code_u = code.upper()
    antibiotic = abx.replace("_", " ").strip().capitalize()
    return {
        "model_id": f"{abx.lower()}-{code.lower()}",
        "antibiotic": antibiotic,
        "pathogen_code": code_u,
        "pathogen": PATHOGEN_NAMES.get(code_u, code_u),
        "label": abx.lower(),   # label column in these CSVs is the antibiotic name
    }


def model_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id / MODEL_VERSION


def artifact_path(model_id: str) -> Path:
    return model_dir(model_id) / "model.joblib"


def feature_schema_path(model_id: str) -> Path:
    return model_dir(model_id) / "feature_schema.json"


def metadata_path(model_id: str) -> Path:
    return model_dir(model_id) / "metadata.json"


def evaluation_path(model_id: str) -> Path:
    return model_dir(model_id) / "evaluation.json"


def sample_path(model_id: str, kind: str) -> Path:
    """kind in {'valid', 'invalid'}."""
    return SAMPLES_DIR / f"{model_id}_{kind}_input.csv"


# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
IDENTIFIER_COLUMN: str = "Isolate"
# The label column is the antibiotic name; these aliases cover it for the
# "don't upload the answer" warning regardless of which model is selected.
LABEL_COLUMN_ALIASES: tuple[str, ...] = (
    "phenotype", "Phenotype", "label", "Label", "resistant", "AST",
    "clindamycin", "doripenem", "ertapenem", "imipenem", "meropenem",
    "kanamycin", "streptomycin",
)
STRAY_COLUMNS: tuple[str, ...] = ("sum", "Sum", "R-Score", "R_Score", "RBI", "rbi")
ALLOWED_VALUES: tuple[int, ...] = (0, 1)

POSITIVE_LABEL: int = 1
POSITIVE_LABEL_NAME: str = "RESISTANT"
NEGATIVE_LABEL_NAME: str = "SUSCEPTIBLE"

# --------------------------------------------------------------------------- #
# Reproduced notebook "best configuration" hyper-parameters (shared by all models)
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
BASE_LEARNER_RANDOM_STATE: int = 0
TEST_SIZE: float = 0.20
DECISION_THRESHOLD: float = 0.50

RBLEND_WEIGHTS: tuple[float, float, float] = (1.0, 1.5, 1.0)  # DT, LogR, XGB

ANOVA_P_THRESHOLD: float = 0.30
XGB_IMPORTANCE_CUTOFF_PCT: int = 85
RSCORE_COLUMN: str = "R-Score"

# Datasets smaller than this are trained but flagged "low confidence".
SMALL_DATASET_ROWS: int = 60
MIN_TRAINABLE_ROWS: int = 20

# --------------------------------------------------------------------------- #
# Upload limits
# --------------------------------------------------------------------------- #
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
MAX_ROWS: int = 5_000

# --------------------------------------------------------------------------- #
# Result column order
# --------------------------------------------------------------------------- #
RESULT_COLUMNS: tuple[str, ...] = (
    "isolateId",
    "prediction",
    "resistantProbability",
    "confidence",
    "rbi",
    "resistanceGeneCount",
)
