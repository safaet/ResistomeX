"""
R-Blend ensemble: build / feature-selection / train / persist / load.

Faithful to the research notebook's executed "best configuration"
(Resistance_Gene_AMR_Prediction_v5, cells 29 / 32 / 35 / 39):

    * engineered feature  : R-Score / RBI  (row-sum of binary features, min-max)
    * feature selection    : union of  {ANOVA F-test p <= 0.30}  and
                             {XGB importance, cumulative cutoff <= 85%}
                             (R-Score is always kept)
    * balancing            : SMOTETomek on the training split, with SMOTE
                             k_neighbors adapted to the minority-class size
                             (notebook `adaptive_smotetomek`)
    * classifier           : soft-voting VotingClassifier of
                               DecisionTree(entropy)   weight 1.0
                               LogisticRegression       weight 1.5
                               XGBClassifier            weight 1.0
                             (SVM intentionally omitted — commented out upstream)

The same configuration is used for every pathogen-antibiotic model; only the
data differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .rbi import RBIScaler, raw_resistome_count


# --------------------------------------------------------------------------- #
# Ensemble definition
# --------------------------------------------------------------------------- #
def build_rblend() -> VotingClassifier:
    seed = config.BASE_LEARNER_RANDOM_STATE
    return VotingClassifier(
        estimators=[
            ("DT", DecisionTreeClassifier(criterion="entropy", random_state=seed)),
            ("LogR", LogisticRegression(random_state=seed, max_iter=1000)),
            # ("SVM", SVC(kernel="rbf", probability=True, random_state=seed)),
            ("XGB", XGBClassifier(random_state=seed, eval_metric="logloss")),
        ],
        voting="soft",
        weights=list(config.RBLEND_WEIGHTS),
    )


def _xgb_selector() -> XGBClassifier:
    return XGBClassifier(
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def _anova_keep(X: pd.DataFrame, y: pd.Series, p_thresh: float) -> list[str]:
    _f, p_vals = f_classif(X, y)
    p = pd.Series(p_vals, index=X.columns)
    keep = p[p <= p_thresh].index.tolist()
    if config.RSCORE_COLUMN in X.columns and config.RSCORE_COLUMN not in keep:
        keep.append(config.RSCORE_COLUMN)
    return keep


def _xgb_cumulative_keep(X: pd.DataFrame, y: pd.Series, cutoff_pct: int) -> list[str]:
    model = _xgb_selector()
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    frac = imp / max(imp.sum(), 1e-12)
    cum = frac.cumsum()
    keep = cum[cum <= (cutoff_pct / 100.0)].index.tolist()
    if not keep and len(imp) > 0:
        keep = [imp.index[0]]
    return keep


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    anova_p: float = config.ANOVA_P_THRESHOLD,
    xgb_cutoff_pct: int = config.XGB_IMPORTANCE_CUTOFF_PCT,
) -> list[str]:
    """Union of the ANOVA set and the XGB cumulative-importance set,
    in the original column order of `X`."""
    anova_set = set(_anova_keep(X, y, anova_p))
    xgb_set = set(_xgb_cumulative_keep(X, y, xgb_cutoff_pct))
    union = anova_set | xgb_set
    return [c for c in X.columns if c in union]


def _adaptive_smotetomek(y_train: pd.Series) -> SMOTETomek:
    """SMOTETomek whose inner SMOTE uses k_neighbors <= minority-1 (notebook)."""
    n_min = int(pd.Series(y_train).value_counts().min())
    k = max(1, min(5, n_min - 1))
    return SMOTETomek(
        random_state=config.RANDOM_STATE,
        smote=SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k),
    )


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
@dataclass
class RBlendArtifact:
    """Everything needed to score a new isolate, saved as one joblib file."""

    model: VotingClassifier
    rbi_scaler: RBIScaler
    raw_feature_columns: list[str]        # ordered RAW binary columns (== schema)
    selected_features: list[str]          # ordered model input columns (incl. R-Score)
    model_id: str = config.DEFAULT_MODEL_ID
    decision_threshold: float = config.DECISION_THRESHOLD
    positive_label: int = config.POSITIVE_LABEL
    data_mode: str = "real"               # "real" | "synthetic"
    extra: dict = field(default_factory=dict)

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RBlendArtifact":
        return joblib.load(path)

    # ---- inference helpers --------------------------------------------- #
    def build_model_matrix(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        rbi = self.rbi_scaler.transform(raw_features)
        enriched = raw_features.copy()
        enriched[config.RSCORE_COLUMN] = rbi.values
        return enriched.loc[:, self.selected_features].copy()

    def predict_frame(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        matrix = self.build_model_matrix(raw_features)
        proba = self.model.predict_proba(matrix)[:, 1]
        pred = (proba >= self.decision_threshold).astype(int)
        rbi = self.rbi_scaler.transform(raw_features)
        gene_count = raw_resistome_count(raw_features)
        return pd.DataFrame(
            {
                "resistant_probability": proba,
                "prediction": pred,
                "rbi": rbi.values,
                "resistance_gene_count": gene_count.astype(int).values,
            },
            index=raw_features.index,
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_rblend(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_id: str = config.DEFAULT_MODEL_ID,
    data_mode: str = "real",
) -> tuple[RBlendArtifact, dict]:
    """Fit the full best-configuration pipeline on the training split only."""
    raw_feature_columns = list(X_train_raw.columns)

    # 1. RBI scaler — fit on TRAIN ONLY, then append the R-Score feature.
    rbi_scaler = RBIScaler.fit(X_train_raw)
    X_train = X_train_raw.copy()
    X_train[config.RSCORE_COLUMN] = rbi_scaler.transform(X_train_raw).values

    # 2. Feature selection (ANOVA p<=0.30  UNION  XGB cumulative<=85%).
    selected_features = select_features(X_train, y_train)
    X_train_sel = X_train.loc[:, selected_features]

    # 3. Balancing — adaptive SMOTETomek on the (selected) training data only.
    minority = int(y_train.value_counts().min())
    if minority >= 2:
        try:
            X_bal, y_bal = _adaptive_smotetomek(y_train).fit_resample(X_train_sel, y_train)
            resample_note = "SMOTETomek"
        except ValueError:
            X_bal, y_bal = X_train_sel, y_train
            resample_note = "none (SMOTETomek failed)"
    else:
        X_bal, y_bal = X_train_sel, y_train
        resample_note = "none (minority < 2)"

    # 4. Fit R-Blend.
    model = build_rblend()
    model.fit(X_bal, y_bal)

    artifact = RBlendArtifact(
        model=model,
        rbi_scaler=rbi_scaler,
        raw_feature_columns=raw_feature_columns,
        selected_features=selected_features,
        model_id=model_id,
        data_mode=data_mode,
    )
    facts = {
        "resample": resample_note,
        "n_train_rows_before_balance": int(len(X_train_sel)),
        "n_train_rows_after_balance": int(len(X_bal)),
        "n_selected_features": len(selected_features),
        "selected_features": selected_features,
    }
    return artifact, facts
