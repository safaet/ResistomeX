"""
Feature-schema load / save and column-ordering helpers.

`feature_schema.json` (one per model) is the frozen data contract between
training and inference. It lists — in a fixed order — every RAW binary feature
column an uploaded CSV must provide. The engineered R-Score / RBI feature is NOT
part of the upload; it is derived at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


@dataclass(frozen=True)
class FeatureSchema:
    model_id: str
    identifier_column: str
    label_column: str
    allowed_values: list[int]
    feature_columns: list[str]          # ordered, RAW binary gene/mutation columns

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSchema":
        return cls(
            model_id=d["modelId"],
            identifier_column=d["identifierColumn"],
            label_column=d["labelColumn"],
            allowed_values=list(d["allowedValues"]),
            feature_columns=list(d["featureColumns"]),
        )

    def to_dict(self) -> dict:
        return {
            "modelId": self.model_id,
            "identifierColumn": self.identifier_column,
            "labelColumn": self.label_column,
            "allowedValues": list(self.allowed_values),
            "featureColumns": list(self.feature_columns),
        }

    @classmethod
    def load(cls, path: Path | str) -> "FeatureSchema":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load_for(cls, model_id: str) -> "FeatureSchema":
        return cls.load(config.feature_schema_path(model_id))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df's feature columns in the exact schema order (extras dropped)."""
        return df.loc[:, list(self.feature_columns)].copy()


def build_schema_from_training_frame(df: pd.DataFrame, model_id: str) -> FeatureSchema:
    """Layout: [identifier] + [raw binary features...] + [label]
    (any stray 'sum' / 'R-Score' column already removed by the caller)."""
    cols = list(df.columns)
    feature_cols = [c for c in cols[1:-1] if c not in config.STRAY_COLUMNS]
    return FeatureSchema(
        model_id=model_id,
        identifier_column=cols[0],
        label_column=cols[-1],
        allowed_values=list(config.ALLOWED_VALUES),
        feature_columns=feature_cols,
    )
