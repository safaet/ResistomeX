"""
Train the R-Blend AMR Predictor — one model per dataset in `Data/Read Data/`.

    python train.py                 # train every dataset
    python train.py --only meropenem-kn
    python train.py --list          # list datasets, train nothing

No notebook, no Colab, no Google Drive. Each model reproduces the research
notebook's executed "best configuration" (see docs/model-decision.md).

Writes, per model, to models/<model_id>/v1/:
    model.joblib   feature_schema.json   metadata.json   evaluation.json
plus a top-level models/registry.json and demo CSVs under data/samples/.

If no dataset directory exists at all, one clearly-labelled synthetic dataset is
generated so the pipeline still runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from datetime import date

import numpy as np

warnings.filterwarnings("ignore")  # match the notebook (constant-feature / lbfgs notices)

from src import config  # noqa: E402
from src.samples import write_sample_files  # noqa: E402
from src.synthetic import ensure_dataset  # noqa: E402
from src.training import (  # noqa: E402
    DatasetTooSmall,
    library_versions,
    load_frame,
    registry_entry,
    train_from_frame,
)


def discover_datasets() -> list[dict]:
    """[{path, model_id, antibiotic, pathogen, pathogen_code, label}], deduped by model_id."""
    seen: dict[str, dict] = {}
    for d in config.SOURCE_DATA_DIRS:
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            if csv.parent.name == ".ipynb_checkpoints":
                continue
            info = config.parse_dataset_name(csv)
            info["path"] = csv
            info["dataset"] = csv.name
            seen.setdefault(info["model_id"], info)
    return list(seen.values())


def _seed_everything() -> None:
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def train_one(info: dict, *, data_mode: str) -> dict:
    """Train + persist one model. Returns its registry entry."""
    df = load_frame(info["path"])
    try:
        artifact, schema, metadata, evaluation = train_from_frame(df, info, data_mode=data_mode)
    except DatasetTooSmall as exc:
        return {
            "modelId": info["model_id"], "pathogen": info["pathogen"],
            "antibiotic": info["antibiotic"], "dataset": info["path"].name,
            "status": "skipped", "reason": str(exc).split(": ", 1)[-1],
        }

    schema.save(config.feature_schema_path(info["model_id"]))
    artifact.save(config.artifact_path(info["model_id"]))
    _write_json(config.metadata_path(info["model_id"]), metadata)
    _write_json(config.evaluation_path(info["model_id"]), evaluation)
    write_sample_files(schema)

    entry = registry_entry(metadata, evaluation)
    _print_entry(entry, metadata)
    return entry


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
        path, _ = ensure_dataset()
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

    new: dict[str, dict] = {}
    for d in datasets:
        print(f"\n=== {d['model_id']} — {d['pathogen']} / {d['antibiotic']} ===")
        try:
            new[d["model_id"]] = train_one(d, data_mode=data_mode)
        except Exception as exc:  # noqa: BLE001 — one bad dataset must not kill the run
            new[d["model_id"]] = {
                "modelId": d["model_id"], "pathogen": d["pathogen"],
                "antibiotic": d["antibiotic"], "dataset": d["path"].name,
                "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED: {new[d['model_id']]['reason']}")

    # Merge into any existing registry so `--only` updates one entry in place.
    existing: dict[str, dict] = {}
    if args.only and config.REGISTRY_PATH.exists():
        with open(config.REGISTRY_PATH, encoding="utf-8") as fh:
            for m in json.load(fh).get("models", []):
                existing[m["modelId"]] = m
    existing.update(new)

    registry = {
        "generatedOn": date.today().isoformat(),
        "modelVersion": config.MODEL_VERSION,
        "defaultModelId": config.DEFAULT_MODEL_ID,
        "libraries": library_versions(),
        "hyperparameters": {
            "ensemble": "soft-voting DecisionTree + LogisticRegression(1.5x) + XGBoost",
            "randomState": config.RANDOM_STATE,
            "testSize": config.TEST_SIZE,
            "anovaPThreshold": config.ANOVA_P_THRESHOLD,
            "xgbImportanceCutoffPct": config.XGB_IMPORTANCE_CUTOFF_PCT,
            "balancing": "adaptive SMOTETomek",
            "decisionThreshold": config.DECISION_THRESHOLD,
        },
        "models": sorted(existing.values(), key=lambda e: e["modelId"]),
    }
    _write_json(config.REGISTRY_PATH, registry)
    _print_summary(list(existing.values()))


def _print_entry(entry: dict, metadata: dict) -> None:
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
        print(f"  {e['modelId']:<20}{e['pathogen'] + ' / ' + e['antibiotic']:<34}  {e['status']}: {e.get('reason', '')}")
    print("=" * 78)
    print(f"  {len(trained)} model(s) trained, {len(other)} skipped/failed.  (* = small dataset)")
    print(f"  Registry: {config.REGISTRY_PATH}\n\n  Next:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
