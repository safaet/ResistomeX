"""
Inference: pick a model -> validated upload -> RBI -> R-Blend -> results + summary.

The app loads the registry once, then loads individual model artifacts on demand
(cached). Every model shares the same pipeline; only the data / schema differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from . import config
from .model import RBlendArtifact
from .schema import FeatureSchema
from .validation import ValidationReport, validate_upload


class ModelNotTrainedError(FileNotFoundError):
    """Raised when a model artifact / the registry is missing — run train.py."""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    pathogen: str
    antibiotic: str
    status: str
    small_dataset: bool
    total_rows: int
    metrics: dict

    @property
    def label(self) -> str:
        return f"{self.pathogen} — {self.antibiotic}"

    @property
    def is_usable(self) -> bool:
        return self.status == "trained"


def _read_json(path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_registry() -> list[ModelInfo]:
    reg = _read_json(config.REGISTRY_PATH)
    if not reg:
        raise ModelNotTrainedError(
            f"No model registry at {config.REGISTRY_PATH}. Run `python train.py` first."
        )
    out = []
    for m in reg.get("models", []):
        out.append(ModelInfo(
            model_id=m["modelId"],
            pathogen=m.get("pathogen", m["modelId"]),
            antibiotic=m.get("antibiotic", ""),
            status=m.get("status", "unknown"),
            small_dataset=bool(m.get("smallDataset", False)),
            total_rows=int(m.get("totalRows", 0)),
            metrics=m.get("metrics", {}),
        ))
    return out


def usable_models() -> list[ModelInfo]:
    models = [m for m in load_registry() if m.is_usable]
    # default model first, then alphabetical
    models.sort(key=lambda m: (m.model_id != config.DEFAULT_MODEL_ID, m.label))
    return models


# --------------------------------------------------------------------------- #
# A single loaded model
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    model_id: str
    artifact: RBlendArtifact
    schema: FeatureSchema
    metadata: dict
    evaluation: dict

    @property
    def is_synthetic(self) -> bool:
        return self.artifact.data_mode == "synthetic" or self.metadata.get("dataMode") == "synthetic"

    @property
    def version(self) -> str:
        return f"{self.model_id}/{config.MODEL_VERSION}"

    @property
    def pathogen(self) -> str:
        return self.metadata.get("pathogen", self.model_id)

    @property
    def antibiotic(self) -> str:
        return self.metadata.get("antibiotic", "")


def load_model(model_id: str) -> LoadedModel:
    art_path = config.artifact_path(model_id)
    if not art_path.exists():
        raise ModelNotTrainedError(
            f"No model artifact at {art_path}. Run `python train.py` first."
        )
    return LoadedModel(
        model_id=model_id,
        artifact=RBlendArtifact.load(art_path),
        schema=FeatureSchema.load_for(model_id),
        metadata=_read_json(config.metadata_path(model_id)),
        evaluation=_read_json(config.evaluation_path(model_id)),
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    results: pd.DataFrame          # columns == config.RESULT_COLUMNS
    summary: dict
    model_version: str


def _summarise(results: pd.DataFrame) -> dict:
    total = len(results)
    resistant = int((results["prediction"] == config.POSITIVE_LABEL_NAME).sum())
    return {
        "totalIsolates": total,
        "resistantCount": resistant,
        "susceptibleCount": total - resistant,
        "percentResistant": round(100.0 * resistant / total, 1) if total else 0.0,
        "averageRbi": round(float(results["rbi"].mean()), 4) if total else 0.0,
        "averageConfidence": round(float(results["confidence"].mean()), 4) if total else 0.0,
    }


def predict_from_clean_frame(
    clean: pd.DataFrame,
    identifiers: pd.Series,
    model: LoadedModel,
) -> PredictionResult:
    raw_features = clean.loc[:, model.schema.feature_columns].reset_index(drop=True)
    scored = model.artifact.predict_frame(raw_features).reset_index(drop=True)

    proba = scored["resistant_probability"].astype(float)
    prediction = scored["prediction"].map(
        {config.POSITIVE_LABEL: config.POSITIVE_LABEL_NAME}
    ).fillna(config.NEGATIVE_LABEL_NAME)
    confidence = pd.concat([proba, 1.0 - proba], axis=1).max(axis=1)

    results = pd.DataFrame(
        {
            "isolateId": identifiers.reset_index(drop=True),
            "prediction": prediction,
            "resistantProbability": proba.round(4),
            "confidence": confidence.round(4),
            "rbi": scored["rbi"].astype(float).round(4),
            "resistanceGeneCount": scored["resistance_gene_count"].astype(int),
        }
    )[list(config.RESULT_COLUMNS)]

    return PredictionResult(
        results=results,
        summary=_summarise(results),
        model_version=model.version,
    )


def predict(raw_csv_bytes: bytes, model: LoadedModel) -> tuple[ValidationReport, PredictionResult | None]:
    report = validate_upload(raw_csv_bytes, model.schema)
    if not report.ok:
        return report, None
    return report, predict_from_clean_frame(report.cleaned, report.identifiers, model)
