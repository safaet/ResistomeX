"""
Inference: pick a model -> validated upload -> RBI -> R-Blend -> results + summary.

The app loads the registry once, then loads individual model artifacts on demand
(cached). If a `model.joblib` is missing or cannot be unpickled (e.g. on
Streamlit Cloud, where the binaries may be absent or were pickled by another
library version), the model is trained in-process from its source CSV in
`Data/Read Data/` and — when the filesystem is writable — persisted.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import pandas as pd

from . import config
from .model import RBlendArtifact
from .schema import FeatureSchema
from .validation import ValidationReport, validate_upload


class ModelNotTrainedError(FileNotFoundError):
    """Raised when no model artifact exists and no source CSV can be found to train one."""


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
        return f"{self.pathogen} — {self.antibiotic}" if self.antibiotic else self.pathogen

    @property
    def is_usable(self) -> bool:
        return self.status in ("trained", "available")


def _read_json(path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _registry_from_discovery() -> list[ModelInfo]:
    """No registry.json: derive the model list straight from the dataset files."""
    out: list[ModelInfo] = []
    for d in config.SOURCE_DATA_DIRS:
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            if csv.parent.name == ".ipynb_checkpoints":
                continue
            info = config.parse_dataset_name(csv)
            out.append(ModelInfo(
                model_id=info["model_id"],
                pathogen=info["pathogen"],
                antibiotic=info["antibiotic"],
                status="available",          # not trained yet, but trainable on demand
                small_dataset=False,
                total_rows=0,
                metrics={},
            ))
    # dedupe by model_id
    seen: dict[str, ModelInfo] = {}
    for mi in out:
        seen.setdefault(mi.model_id, mi)
    return list(seen.values())


def load_registry() -> list[ModelInfo]:
    reg = _read_json(config.REGISTRY_PATH)
    if not reg:
        discovered = _registry_from_discovery()
        if discovered:
            return discovered
        raise ModelNotTrainedError(
            f"No model registry at {config.REGISTRY_PATH} and no datasets found in "
            f"{', '.join(str(d) for d in config.SOURCE_DATA_DIRS)}. Run `python train.py`."
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
    trained_at_runtime: bool = False

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


def _train_on_demand(model_id: str) -> LoadedModel:
    from .training import train_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ANOVA constant-feature / lbfgs notices
        artifact, schema, metadata, evaluation = train_model(model_id, data_mode="real")

    # Best-effort persist (filesystem may be read-only on some hosts).
    try:
        schema.save(config.feature_schema_path(model_id))
        artifact.save(config.artifact_path(model_id))
        _write_json_safe(config.metadata_path(model_id), metadata)
        _write_json_safe(config.evaluation_path(model_id), evaluation)
    except OSError:
        pass

    return LoadedModel(
        model_id=model_id, artifact=artifact, schema=schema,
        metadata=metadata, evaluation=evaluation, trained_at_runtime=True,
    )


def _self_check(model: "LoadedModel") -> None:
    """Score one all-zero isolate to prove the pickled pipeline still runs."""
    row = pd.DataFrame([[0] * model.schema.n_features], columns=model.schema.feature_columns)
    model.artifact.predict_frame(row)


def _write_json_safe(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def load_model(model_id: str) -> LoadedModel:
    art_path = config.artifact_path(model_id)
    if art_path.exists():
        try:
            with warnings.catch_warnings():
                # A library-version mismatch on the pickle is a hard signal to
                # retrain; other warnings are harmless.
                try:
                    from sklearn.exceptions import InconsistentVersionWarning
                    warnings.simplefilter("error", InconsistentVersionWarning)
                except Exception:  # noqa: BLE001
                    pass
                artifact = RBlendArtifact.load(art_path)
            model = LoadedModel(
                model_id=model_id,
                artifact=artifact,
                schema=FeatureSchema.load_for(model_id),
                metadata=_read_json(config.metadata_path(model_id)),
                evaluation=_read_json(config.evaluation_path(model_id)),
            )
            # Sanity check: the loaded pipeline must actually score a row.
            _self_check(model)
            return model
        except Exception:  # noqa: BLE001 — any load/scoring failure => retrain from source
            pass
    try:
        return _train_on_demand(model_id)
    except FileNotFoundError as exc:
        raise ModelNotTrainedError(
            f"No artifact for '{model_id}' and no source CSV to train from: {exc}"
        ) from exc


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

    return PredictionResult(results=results, summary=_summarise(results), model_version=model.version)


def predict(raw_csv_bytes: bytes, model: LoadedModel) -> tuple[ValidationReport, PredictionResult | None]:
    report = validate_upload(raw_csv_bytes, model.schema)
    if not report.ok:
        return report, None
    return report, predict_from_clean_frame(report.cleaned, report.identifiers, model)
