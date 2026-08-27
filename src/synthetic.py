"""
Synthetic dataset generator — used ONLY when `Data/Read Data/` (and `data/raw/`)
contain no CSVs at all, so the app is still runnable end-to-end for a demo.
Synthetic mode is clearly labelled in the UI.

The synthetic file mimics the real schema: an `Isolate` column, `gene=STATE`
binary feature columns, and a trailing `meropenem` 0/1 label whose probability
rises with the number of carbapenemase genes present (so the RBI feature is
genuinely informative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_GENE_STEMS = [
    "blaKPC", "blaNDM", "blaOXA-48", "blaVIM", "blaIMP", "blaCTX-M-15",
    "blaSHV", "blaTEM", "blaCMY-2", "aac(6')-Ib", "aac(3)-IIa", "aph(3')-Ia",
    "aadA1", "aadA2", "armA", "rmtB", "qnrS1", "qnrB1", "oqxA", "oqxB",
    "sul1", "sul2", "dfrA12", "dfrA14", "tet(A)", "tet(D)", "catA1", "catB3",
    "fosA", "mph(A)", "ereA", "ermB", "mcr-1", "ompK35", "ompK36", "acrR",
    "ramR", "marA", "soxS", "kpnE",
]
_STATES = ["COMPLETE", "PARTIAL_END_OF_CONTIG"]
_DRIVER_PREFIXES = ("blaKPC", "blaNDM", "blaOXA-48", "blaVIM", "blaIMP", "ompK")


def _feature_columns() -> list[str]:
    return [f"{stem}={state}" for stem in _GENE_STEMS for state in _STATES]


def generate_synthetic_dataset(n_rows: int = 160, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feat_cols = _feature_columns()
    X = (rng.random((n_rows, len(feat_cols))) < 0.06).astype(int)

    driver_idx = [i for i, c in enumerate(feat_cols) if c.startswith(_DRIVER_PREFIXES)]
    driver_load = X[:, driver_idx].sum(axis=1)
    total_load = X.sum(axis=1)

    logit = -1.2 + 1.6 * driver_load + 0.06 * total_load + rng.normal(0, 0.5, n_rows)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_rows) < prob).astype(int)
    if y.sum() < 15:
        y[rng.choice(n_rows, 15, replace=False)] = 1
    if (1 - y).sum() < 15:
        y[rng.choice(np.where(y == 1)[0], 15, replace=False)] = 0

    df = pd.DataFrame(X, columns=feat_cols)
    df.insert(0, config.IDENTIFIER_COLUMN, [f"SYN-{i:04d}" for i in range(1, n_rows + 1)])
    df["meropenem"] = y
    return df


def any_source_csv_exists() -> bool:
    for d in config.SOURCE_DATA_DIRS:
        if d.exists() and any(
            p for p in d.glob("*.csv") if p.parent.name != ".ipynb_checkpoints"
        ):
            return True
    return False


def ensure_dataset() -> tuple[str, bool]:
    """If no source CSV exists anywhere, write a synthetic one into data/raw/.
    Returns (path, is_synthetic)."""
    if any_source_csv_exists():
        return "", False
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RAW_DATA_DIR / "amr_ast_meropenem_KN.csv"
    generate_synthetic_dataset().to_csv(out, index=False)
    return str(out), True
