"""Project configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path("/root/triagegeist-kaggle")
DATA_DIR = Path("/root/triagegeist_data")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ID_COLUMN = "patient_id"
TARGET_COLUMN = "triage_acuity"
TEXT_COLUMN = "chief_complaint_raw"

LEAKAGE_COLUMNS = [
    ID_COLUMN,
    TARGET_COLUMN,
    "disposition",
    "ed_los_hours",
]

SUBGROUP_COLUMNS = [
    "age_group",
    "sex",
    "language",
    "site_id",
    "arrival_mode",
]

HIGH_RISK_THRESHOLD = 2

