"""
Train the R-Blend AMR Predictor — one model per dataset in `Data/Read Data/`.

    python train.py                 # train every dataset
    python train.py --only meropenem-kn
    python train.py --list          # list datasets, train nothing

No notebook, no Colab, no Google Drive. Each model reproduces the research
notebook's executed "best configuration":

    R-Score/RBI feature  ->  ANOVA p<=0.30  UNION  XGB-importance-85% selection
    ->  adaptive SMOTETomek balancing  ->  soft-voting DT + LogR(1.5x) + XGB

Writes, per model, to models/<model_id>/v1/:
    model.joblib   feature_schema.json   metadata.json   evaluation.json
plus a top-level models/registry.json describing every model, and demo CSVs
under data/samples/.

If no dataset directory exists at all, one clearly-labelled synthetic dataset is
generated so the pipeline still runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import warnings
from datetime import date

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

warnings.filterwarnings("ignore")  # match the notebook (constant-feature / lbfgs notices)

from src import config  # noqa: E402
from src.model import train_rblend  # noqa: E402
from src.samples import write_sample_files  # noqa: E402
from src.schema import build_schema_from_training_frame  # noqa: E402
from src.synthetic import ensure_dataset  # noqa: E402


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def discover_datasets() -> list[dict]:
    """Return [{path, model_id, antibiotic, pathogen, pathogen_code, label}], deduped by model_id."""
    seen: dict[str, dict] = {}
    for d in config.SOURCE_DATA_DIRS:
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            if csv.parent.name == ".ipynb_checkpoints":
                continue
            info = config.parse_dataset_name(csv)
            info["path"] = csv
            seen.setdefault(info["model_id"], info)  # first dir wins
    return list(seen.values())


def _seed_everything() -> None:
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)


def _load_frame(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in config.STRAY_COLUMNS if c in df.columns], errors="ignore")


# --------------------------------------------------------------------------- #
# Train a single dataset
# --------------------------------------------------------------------------- #
def train_one(info: dict, *, data_mode: str = "real") -> dict:
    """Train, evaluate, persist one model. Returns its registry entry."""
    model_id = info["model_id"]
    df = _load_frame(info["path"])
    identifier, label = df.columns[0], df.columns[-1]
    X = df.drop(columns=[identifier, label])
    y = df[label].astype(int)

    entry: dict = {
        "modelId": model_id,
        "pathogen": info["pathogen"],
        "pathogenCode": info["pathogen_code"],
        "antibiotic": info["antibiotic"],
        "dataset": info["path"].name,
        "dataMode": data_mode,
        "totalRows": int(len(df)),
        "rawFeatureCount": int(X.shape[1]),
    }

    # --- guards -------------------------------------------------------- #
    if y.nunique() < 2:
        entry["status"] = "skipped"
        entry["reason"] = "only one class present"
        return entry
    if len(df) < config.MIN_TRAINABLE_ROWS:
        entry["status"] = "skipped"
        entry["reason"] = f"fewer than {config.MIN_TRAINABLE_ROWS} isolates"
        return entry

    schema = build_schema_from_training_frame(df, model_id)
    schema.save(config.feature_schema_path(model_id))

    class_counts = {str(k): int(v) for k, v in y.value_counts().items()}
    minority = min(class_counts.values())

    # Stratified hold-out; fall back to non-stratified only if it is impossible.
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
    artifact.save(config.artifact_path(model_id))

    # --- evaluate on the untouched hold-out set --------------------- #
    scored = artifact.predict_frame(X_te.reset_index(drop=True))
    y_pred = scored["prediction"].to_numpy()
    y_proba = scored["resistant_probability"].to_numpy()
    y_true = y_te.to_numpy()
    both_classes = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    evaluation = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_true, y_proba)), 4) if both_classes else None,
        "prAuc": round(float(average_precision_score(y_true, y_proba)), 4) if both_classes else None,
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
    _write_json(config.evaluation_path(model_id), evaluation)

    is_small = len(df) < config.SMALL_DATASET_ROWS
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
        "smallDataset": bool(is_small),
        "rbi": {
            "definition": "min-max normalised row-sum of binary gene/mutation features",
            "fittedOn": "training split only",
            "rMin": artifact.rbi_scaler.r_min,
            "rMax": artifact.rbi_scaler.r_max,
        },
        "dataset": info["path"].name,
        "dataMode": data_mode,
        "trainedOn": date.today().isoformat(),
        "libraries": _library_versions(),
        "python": platform.python_version(),
    }
    _write_json(config.metadata_path(model_id), metadata)

    write_sample_files(schema)

    entry.update({
        "status": "trained",
        "version": config.MODEL_VERSION,
        "trainingRows": int(len(X_tr)),
        "testRows": int(len(X_te)),
        "classBalance": class_counts,
        "selectedFeatureCount": facts["n_selected_features"],
        "smallDataset": bool(is_small),
        "metrics": {
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "rocAuc": evaluation["rocAuc"],
        },
    })
    return entry


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train R-Blend AMR models.")
    parser.add_argument("--only", metavar="MODEL_ID", help="train just this model_id")
    parser.add_argument("--list", action="store_true", help="list discovered datasets and exit")
    args = parser.parse_args()

    _seed_everything()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets()
    data_mode = "real"
    if not datasets:
        path, is_synth = ensure_dataset()
        print(f"No dataset directory found. Generated a SYNTHETIC dataset at {path}.")
        datasets = discover_datasets()
        data_mode = "synthetic"

    if args.only:
        datasets = [d for d in datasets if d["model_id"] == args.only]
        if not datasets:
            sys.exit(f"No dataset matches model_id '{args.only}'.")

    print(f"Discovered {len(datasets)} dataset(s):")
    for d in datasets:
        print(f"  - {d['model_id']:<20} {d['pathogen']} / {d['antibiotic']}   ({d['path'].name})")
    if args.list:
        return

    new_entries: dict[str, dict] = {}
    for d in datasets:
        print(f"\n=== {d['model_id']} — {d['pathogen']} / {d['antibiotic']} ===")
        try:
            entry = train_one(d, data_mode=data_mode)
        except Exception as exc:  # noqa: BLE001 — one bad dataset must not kill the run
            entry = {
                "modelId": d["model_id"], "pathogen": d["pathogen"],
                "antibiotic": d["antibiotic"], "dataset": d["path"].name,
                "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED: {entry['reason']}")
        new_entries[entry["modelId"]] = entry
        _print_entry(entry)

    # Merge into any existing registry so `--only` updates one entry in place.
    existing: dict[str, dict] = {}
    if args.only and config.REGISTRY_PATH.exists():
        with open(config.REGISTRY_PATH, encoding="utf-8") as fh:
            for m in json.load(fh).get("models", []):
                existing[m["modelId"]] = m
    existing.update(new_entries)
    entries = sorted(existing.values(), key=lambda e: e["modelId"])

    registry = {
        "generatedOn": date.today().isoformat(),
        "modelVersion": config.MODEL_VERSION,
        "defaultModelId": config.DEFAULT_MODEL_ID,
        "hyperparameters": {
            "ensemble": "soft-voting DecisionTree + LogisticRegression(1.5x) + XGBoost",
            "randomState": config.RANDOM_STATE,
            "testSize": config.TEST_SIZE,
            "anovaPThreshold": config.ANOVA_P_THRESHOLD,
            "xgbImportanceCutoffPct": config.XGB_IMPORTANCE_CUTOFF_PCT,
            "balancing": "adaptive SMOTETomek",
            "decisionThreshold": config.DECISION_THRESHOLD,
        },
        "models": entries,
    }
    _write_json(config.REGISTRY_PATH, registry)
    _print_summary(entries)


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def _library_versions() -> dict:
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


def _print_entry(entry: dict) -> None:
    if entry["status"] != "trained":
        print(f"  {entry['status'].upper()}: {entry.get('reason', '')}")
        return
    m = entry["metrics"]
    flag = "  [small dataset — low confidence]" if entry.get("smallDataset") else ""
    print(f"  trained  n={entry['totalRows']} ({entry['trainingRows']}/{entry['testRows']})  "
          f"feats={entry['selectedFeatureCount']}/{entry['rawFeatureCount']}{flag}")
    print(f"  acc={m['accuracy']}  prec={m['precision']}  recall={m['recall']}  "
          f"f1={m['f1']}  roc_auc={m['rocAuc']}")


def _print_summary(entries: list[dict]) -> None:
    trained = [e for e in entries if e["status"] == "trained"]
    other = [e for e in entries if e["status"] != "trained"]
    print("\n" + "=" * 78)
    print(f"  {'model_id':<20}{'pathogen / antibiotic':<34}{'n':>5}{'recall':>8}{'f1':>7}")
    print("-" * 78)
    for e in sorted(trained, key=lambda x: x["modelId"]):
        m = e["metrics"]
        name = f"{e['pathogen'].split(' / ')[0]} / {e['antibiotic']}"
        star = "*" if e.get("smallDataset") else " "
        print(f"{star} {e['modelId']:<20}{name:<34}{e['totalRows']:>5}{m['recall']:>8}{m['f1']:>7}")
    for e in other:
        print(f"  {e['modelId']:<20}{e['pathogen']+' / '+e['antibiotic']:<34}  {e['status']}: {e.get('reason','')}")
    print("=" * 78)
    print(f"  {len(trained)} model(s) trained, {len(other)} skipped/failed.  (* = small dataset)")
    print(f"  Registry: {config.REGISTRY_PATH}")
    print("\n  Next:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
