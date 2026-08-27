"""
Upload validation for prediction CSVs (product plan — Week 2).

`validate_upload` returns a structured report (ALL errors + warnings, not just
the first failure) and, when the file is usable, a cleaned dataframe with:
    * the identifier column preserved
    * feature columns reordered to the frozen schema order
    * unknown / label columns dropped (with a warning)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .schema import FeatureSchema


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_isolates: int = 0
    cleaned: pd.DataFrame | None = None      # identifier + ordered raw features
    identifiers: pd.Series | None = None

    @property
    def ok(self) -> bool:
        return not self.errors and self.cleaned is not None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def read_csv_bytes(raw: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """Parse bytes as CSV. Returns (df, error_message)."""
    if raw is None or len(raw) == 0:
        return None, "The uploaded file is empty (0 bytes)."
    if len(raw) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
        return None, f"File is larger than the {mb:.0f} MB upload limit."
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 — surface any parser error to the user
        return None, f"File could not be read as CSV: {exc}"
    if df.shape[1] == 0:
        return None, "The CSV has no columns."
    return df, None


def validate_dataframe(df: pd.DataFrame, schema: FeatureSchema) -> ValidationReport:
    """Apply every Week-2 rule against an already-parsed dataframe."""
    report = ValidationReport(n_rows=len(df))

    # --- not empty -------------------------------------------------------- #
    if df.empty:
        report.add_error("The CSV contains a header but no data rows.")
        return report

    # --- row count cap -------------------------------------------------- #
    if len(df) > config.MAX_ROWS:
        report.add_error(
            f"Too many rows: {len(df):,}. The limit is {config.MAX_ROWS:,} isolates per file."
        )

    # --- identifier column present ------------------------------------- #
    id_col = schema.identifier_column
    if id_col not in df.columns:
        report.add_error(
            f"Required identifier column '{id_col}' is missing. "
            "Download the template to see the exact expected header."
        )

    # --- label column must NOT be in a prediction upload -------------- #
    present_label_cols = [c for c in config.LABEL_COLUMN_ALIASES if c in df.columns]
    if present_label_cols:
        report.add_warning(
            "Prediction files should not include a phenotype/label column; "
            f"ignoring: {', '.join(present_label_cols)}."
        )

    # --- known vs unknown feature columns ---------------------------- #
    schema_features = list(schema.feature_columns)
    schema_feature_set = set(schema_features)
    known_non_feature = {id_col, *config.LABEL_COLUMN_ALIASES, *config.STRAY_COLUMNS}

    present_features = [c for c in df.columns if c in schema_feature_set]
    missing_features = [c for c in schema_features if c not in df.columns]
    unknown_cols = [
        c for c in df.columns
        if c not in schema_feature_set and c not in known_non_feature
    ]

    if missing_features:
        preview = ", ".join(missing_features[:10])
        more = "" if len(missing_features) <= 10 else f" (+{len(missing_features) - 10} more)"
        report.add_error(
            f"Missing {len(missing_features)} required feature column(s): {preview}{more}."
        )
    if unknown_cols:
        preview = ", ".join(unknown_cols[:10])
        more = "" if len(unknown_cols) <= 10 else f" (+{len(unknown_cols) - 10} more)"
        report.add_warning(
            f"{len(unknown_cols)} unrecognised column(s) will be ignored: {preview}{more}."
        )

    # --- identifier quality ----------------------------------------- #
    if id_col in df.columns:
        ids = df[id_col]
        n_blank = int(ids.isna().sum() + (ids.astype(str).str.strip() == "").sum())
        if n_blank:
            report.add_error(f"{n_blank} isolate ID(s) are empty. Every row needs an ID.")
        dupes = ids[ids.duplicated(keep=False)]
        if len(dupes):
            sample = ", ".join(map(str, pd.unique(dupes)[:5]))
            report.add_error(
                f"Isolate IDs must be unique — {dupes.nunique()} value(s) repeat "
                f"(e.g. {sample})."
            )
        report.n_isolates = int(ids.nunique(dropna=True))

    # --- feature values: 0/1 only, no missing --------------------- #
    if present_features:
        feat = df[present_features]
        n_missing = int(feat.isna().sum().sum())
        if n_missing:
            cols_with_na = feat.columns[feat.isna().any()].tolist()
            preview = ", ".join(cols_with_na[:8])
            report.add_error(
                f"{n_missing} missing value(s) in feature columns "
                f"(e.g. {preview}). Fill every cell with 0 or 1."
            )
        # non-binary detection (numeric or otherwise)
        numeric = feat.apply(pd.to_numeric, errors="coerce")
        coerced_bad = int((numeric.isna() & feat.notna()).sum().sum())
        if coerced_bad:
            report.add_error(
                f"{coerced_bad} feature cell(s) are not numbers. "
                "Feature values must be exactly 0 or 1."
            )
        out_of_range = numeric.where(numeric.notna())
        bad_vals = int(((out_of_range != 0) & (out_of_range != 1)).sum().sum())
        if bad_vals:
            report.add_error(
                f"{bad_vals} feature cell(s) hold a value other than 0 or 1."
            )

    if report.errors:
        return report

    # --- build cleaned frame (reordered to schema) ------------------ #
    ordered = df.loc[:, [id_col] + schema_features].copy()
    ordered[schema_features] = ordered[schema_features].apply(pd.to_numeric).astype(int)
    report.cleaned = ordered
    report.identifiers = ordered[id_col].astype(str).reset_index(drop=True)
    return report


def validate_upload(raw: bytes, schema: FeatureSchema) -> ValidationReport:
    """Full path: bytes -> parsed CSV -> rule checks."""
    df, err = read_csv_bytes(raw)
    if err is not None:
        rep = ValidationReport()
        rep.add_error(err)
        return rep
    return validate_dataframe(df, schema)
