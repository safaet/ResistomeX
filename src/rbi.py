"""
Resistome Burden Index (RBI) — a.k.a. the notebook's "R-Score".

Paper definition (Eq. 1.1 / 1.2):

    r_i   = sum_j x_ij           # raw resistome count: row-sum of binary features
    RBI_i = (r_i - r_min) / (r_max - r_min)

The single most important correctness rule (product plan, Week 3):

    r_min and r_max are learned from the TRAINING split ONLY, persisted inside
    the model artifact, and reused unchanged at inference time. They are NEVER
    re-fitted per uploaded file.

(The research notebook fit the min/max on the full dataset before splitting —
that is data leakage; we fix it here on purpose.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def raw_resistome_count(features: pd.DataFrame) -> pd.Series:
    """r_i = row-wise sum of the binary gene/mutation indicator columns."""
    return features.sum(axis=1).astype(float)


@dataclass(frozen=True)
class RBIScaler:
    """Min-max normaliser for the raw resistome count, fitted on training data."""

    r_min: float
    r_max: float

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "RBIScaler":
        counts = raw_resistome_count(features)
        return cls(r_min=float(counts.min()), r_max=float(counts.max()))

    def transform(self, features: pd.DataFrame) -> pd.Series:
        """Apply the persisted min/max. Result is clipped to [0, 1] so that
        an uploaded isolate carrying more determinants than anything seen in
        training still yields a valid score."""
        counts = raw_resistome_count(features)
        span = self.r_max - self.r_min
        if span <= 0:  # degenerate training data — avoid divide-by-zero
            rbi = pd.Series(np.zeros(len(counts)), index=counts.index)
        else:
            rbi = (counts - self.r_min) / span
        return rbi.clip(lower=0.0, upper=1.0)

    def to_dict(self) -> dict:
        return {"r_min": self.r_min, "r_max": self.r_max}

    @classmethod
    def from_dict(cls, d: dict) -> "RBIScaler":
        return cls(r_min=float(d["r_min"]), r_max=float(d["r_max"]))
