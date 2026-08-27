"""
Per-model demo CSVs under data/samples/:

    <model_id>_valid_input.csv    — passes every validation rule
    <model_id>_invalid_input.csv  — several deliberate problems at once

Also exposes `csv_template_bytes()` (the app's "Download template" button) and
`example_valid_bytes()` (the app's "Use example isolates" shortcut).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config
from .schema import FeatureSchema


def _example_rows(schema: FeatureSchema, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = schema.feature_columns
    mat = (rng.random((n, len(feats))) < 0.05).astype(int)
    df = pd.DataFrame(mat, columns=feats)
    df.insert(0, schema.identifier_column, [f"DEMO-{i:03d}" for i in range(1, n + 1)])
    return df


def csv_template_bytes(schema: FeatureSchema) -> bytes:
    """Header (identifier + every feature column, schema order) + one example row."""
    df = _example_rows(schema, n=1, seed=0)
    df.loc[0, schema.identifier_column] = "EXAMPLE-001"
    return _to_csv_bytes(df)


def example_valid_bytes(schema: FeatureSchema, n: int = 8) -> bytes:
    return _to_csv_bytes(_example_rows(schema, n=n, seed=config.RANDOM_STATE))


def write_sample_files(schema: FeatureSchema) -> tuple[str, str]:
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    mid = schema.model_id

    valid = _example_rows(schema, n=8, seed=config.RANDOM_STATE)
    valid_path = config.sample_path(mid, "valid")
    valid.to_csv(valid_path, index=False)

    bad = _example_rows(schema, n=6, seed=7)
    bad = bad.rename(columns={schema.identifier_column: "SampleName"})   # wrong ID header
    bad.loc[1, bad.columns[1]] = 5                                       # value outside {0,1}
    bad.loc[2, bad.columns[2]] = np.nan                                  # missing value
    bad.loc[3, "SampleName"] = bad.loc[0, "SampleName"]                  # duplicate ID
    if len(bad.columns) > 4:
        bad = bad.drop(columns=[bad.columns[3]])                        # drop a required feature
    bad["not_a_real_gene"] = 1                                          # unknown column
    invalid_path = config.sample_path(mid, "invalid")
    bad.to_csv(invalid_path, index=False)

    return str(valid_path), str(invalid_path)


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
