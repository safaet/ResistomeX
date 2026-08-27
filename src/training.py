"""
Library: train ONE R-Blend model from a dataset CSV and return its artifact +
metadata + evaluation dicts (no file I/O here).

Used by:
  * train.py            — writes the results to models/<id>/v1/
  * src.predict          — trains on demand when a joblib is missing or fails to
                           load (e.g. on Streamlit Cloud, where the .joblib files
                           may not be present or were pickled by another version)
"""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import config
from .model import RBlendArtifact, train_rblend
from .schema import FeatureSchema, build_schema_from_training_frame


class DatasetTooSmall(ValueError):
    pass


def find_dataset_path(model_id: str) -> Path | None:
    """Locate the source CSV for a model_id across the configured data dirs."""
    for d in config.SOURCE_DATA_DIRS:
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            if csv.parent.name == ".ipynb_checkpoints":
                continue
            if config.parse_dataset_name(csv)["model_id"] == model_id:
                return csv
    return None


def load_frame(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in config.STRAY_COLUMNS if c in df.columns], errors="ignore")


def train_from_frame(
    df: pd.DataFrame,
    info: dict,
    *,
    data_mode: str = "real",
) -> tuple[RBlendArtifact, FeatureSchema, dict, dict]:
    """Return (artifact, schema, metadata, evaluation) for one dataset frame.

    `info` is a dict from `config.parse_dataset_name`.
    Raises DatasetTooSmall / ValueError for un-trainable data.
    """
    model_id = info["model_id"]
    identifier, label = df.columns[0], df.columns[-1]
    X = df.drop(columns=[identifier, label])
    y = df[label].astype(int)

    if y.nunique() < 2:
        raise DatasetTooSmall(f"{model_id}: only one class present")
    if len(df) < config.MIN_TRAINABLE_ROWS:
        raise DatasetTooSmall(f"{model_id}: fewer than {config.MIN_TRAINABLE_ROWS} isolates")

    schema = build_schema_from_training_frame(df, model_id)
    class_counts = {str(k): int(v) for k, v in y.value_counts().items()}
    minority = min(class_counts.values())

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        split_note = "stratified"
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        split_note = "non-stratified (too few minority samples)"

    artifact, facts = train_rblend(X_tr, y_tr, model_id=model_id, data_mode=data_mode)

    scored = artifact.predict_frame(X_te.reset_index(drop=True))
    y_pred = scored["prediction"].to_numpy()
    y_proba = scored["resistant_probability"].to_numpy()
    y_true = y_te.to_numpy()
    both = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    evaluation = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_true, y_proba)), 4) if both else None,
        "prAuc": round(float(average_precision_score(y_true, y_proba)), 4) if both else None,
        "confusionMatrix": {
            "trueNegative": int(tn), "falsePositive": int(fp),
            "falseNegative": int(fn), "truePositive": int(tp),
            "layout": "rows = actual [Susceptible, Resistant], cols = predicted [Susceptible, Resistant]",
        },
        "testRows": int(len(y_te)),
        "positiveLabel": config.POSITIVE_LABEL,
        "decisionThreshold": config.DECISION_THRESHOLD,
        "split": split_note,
        "note": (
            "Single random hold-out split, phylogeny-naive. Treat metrics as an "
            "optimistic upper bound (see docs/intended-use.md)."
        ),
    }

    metadata = {
        "modelId": model_id,
        "version": config.MODEL_VERSION,
        "modelType": "R-Blend soft-voting ensemble (DecisionTree + LogisticRegression + XGBoost)",
        "votingWeights": {"DecisionTree": config.RBLEND_WEIGHTS[0],
                          "LogisticRegression": config.RBLEND_WEIGHTS[1],
                          "XGBoost": config.RBLEND_WEIGHTS[2]},
        "pathogen": info["pathogen"],
        "pathogenCode": info["pathogen_code"],
        "antibiotic": info["antibiotic"],
        "positiveClass": config.POSITIVE_LABEL_NAME.capitalize(),
        "randomState": config.RANDOM_STATE,
        "baseLearnerRandomState": config.BASE_LEARNER_RANDOM_STATE,
        "testSize": config.TEST_SIZE,
        "decisionThreshold": config.DECISION_THRESHOLD,
        "totalRows": int(len(df)),
        "trainingRows": int(len(X_tr)),
        "testRows": int(len(X_te)),
        "classBalance": class_counts,
        "minorityClassCount": int(minority),
        "rawFeatureCount": schema.n_features,
        "selectedFeatureCount": facts["n_selected_features"],
        "featureSelection": {
            "anovaPThreshold": config.ANOVA_P_THRESHOLD,
            "xgbCumulativeImportanceCutoffPct": config.XGB_IMPORTANCE_CUTOFF_PCT,
            "combine": "union",
        },
        "balancing": facts["resample"],
        "split": split_note,
        "smallDataset": bool(len(df) < config.SMALL_DATASET_ROWS),
        "rbi": {
            "definition": "min-max normalised row-sum of binary gene/mutation features",
            "fittedOn": "training split only",
            "rMin": artifact.rbi_scaler.r_min,
            "rMax": artifact.rbi_scaler.r_max,
        },
        "dataset": info.get("dataset") or Path(info.get("path", "")).name or f"{model_id}.csv",
        "dataMode": data_mode,
        "trainedOn": date.today().isoformat(),
        "libraries": library_versions(),
        "python": platform.python_version(),
    }
    return artifact, schema, metadata, evaluation


def train_model(model_id_or_info, *, data_mode: str = "real") -> tuple[RBlendArtifact, FeatureSchema, dict, dict]:
    """Locate the dataset for a model_id (or an info dict carrying 'path'/'model_id')
    and train it. `info` is (re)derived from the actual filename."""
    if isinstance(model_id_or_info, dict):
        model_id = model_id_or_info["model_id"]
        path = model_id_or_info.get("path") or find_dataset_path(model_id)
    else:
        model_id = model_id_or_info
        path = find_dataset_path(model_id)
    if path is None:
        raise FileNotFoundError(f"No source CSV found for model '{model_id}'.")
    path = Path(path)
    info = config.parse_dataset_name(path)
    info["path"] = path
    info["dataset"] = path.name
    return train_from_frame(load_frame(path), info, data_mode=data_mode)


def registry_entry(metadata: dict, evaluation: dict) -> dict:
    return {
        "modelId": metadata["modelId"],
        "pathogen": metadata["pathogen"],
        "pathogenCode": metadata.get("pathogenCode", ""),
        "antibiotic": metadata["antibiotic"],
        "dataset": metadata["dataset"],
        "dataMode": metadata["dataMode"],
        "status": "trained",
        "version": config.MODEL_VERSION,
        "totalRows": metadata["totalRows"],
        "trainingRows": metadata["trainingRows"],
        "testRows": metadata["testRows"],
        "rawFeatureCount": metadata["rawFeatureCount"],
        "selectedFeatureCount": metadata["selectedFeatureCount"],
        "classBalance": metadata["classBalance"],
        "smallDataset": metadata["smallDataset"],
        "metrics": {
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "rocAuc": evaluation["rocAuc"],
        },
    }


def library_versions() -> dict:
    import imblearn
    import sklearn
    import xgboost

    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "imbalanced-learn": imblearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
