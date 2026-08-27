"""
Quick end-to-end smoke test (not a unit-test suite).

    python scripts/smoke_test.py

Assumes `python train.py` has run. Checks the registry, then for every trained
model: artifacts load, its valid sample predicts, its invalid sample is rejected,
and RBI stays in [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import load_model, load_registry, predict_from_clean_frame, usable_models  # noqa: E402
from src.validation import validate_upload  # noqa: E402

_fail = 0


def check(name: str, cond: bool) -> None:
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _fail += 1


def main() -> None:
    reg = load_registry()
    trained = usable_models()
    print(f"Registry: {len(reg)} model(s), {len(trained)} trained.\n")
    check("at least one trained model", len(trained) > 0)
    check("default model is trained", any(m.model_id == config.DEFAULT_MODEL_ID for m in trained))

    for info in trained:
        print(f"--- {info.model_id}  ({info.label}) ---")
        model = load_model(info.model_id)
        check("schema has features", model.schema.n_features > 0)
        check("evaluation has recall", "recall" in model.evaluation)
        check("RBI bounds persisted", model.artifact.rbi_scaler.r_max >= model.artifact.rbi_scaler.r_min)

        vp = config.sample_path(info.model_id, "valid")
        rep = validate_upload(vp.read_bytes(), model.schema)
        check("valid sample passes validation", rep.ok)
        if rep.ok:
            result = predict_from_clean_frame(rep.cleaned, rep.identifiers, model)
            check("row count preserved", len(result.results) == rep.n_rows)
            check("result columns correct", list(result.results.columns) == list(config.RESULT_COLUMNS))
            check("rbi within [0, 1]", result.results["rbi"].between(0, 1).all())
            check("labels valid", set(result.results["prediction"]) <=
                  {config.POSITIVE_LABEL_NAME, config.NEGATIVE_LABEL_NAME})

        ip = config.sample_path(info.model_id, "invalid")
        bad = validate_upload(ip.read_bytes(), model.schema)
        check("invalid sample rejected", (not bad.ok) and len(bad.errors) >= 1)
        print()

    if _fail:
        print(f"{_fail} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
