# Model decision — authoritative R-Blend

**Status:** decided · **Approved by:** repo owner (cst.ics@proton.me) ·
**Approval date:** 2026-08-27

**Scope update (2026-08-27):** the same authoritative configuration below is now
trained **once per dataset in `Data/Read Data/`** — 12 pathogen–antibiotic
models, not one. Only the data and the frozen feature schema differ between
models; every hyper-parameter here is shared. `meropenem-kn` remains the default.

## The discrepancy

| Source | Ensemble described |
|---|---|
| Paper (`337_Camera Ready.pdf`, §1.3.4) | **SVM (RBF, Platt) + XGBoost**, soft-voting, **equal weights** |
| Research notebook (`Resistance_Gene_AMR_Prediction_v5`, executed cells 29 / 32 / 35 / 39) | **DecisionTree + LogisticRegression + XGBoost**, soft-voting, **weights [1, 1.5, 1]**; SVM present but **commented out** |

The notebook's "best configuration" additionally applies an engineered
**R-Score / RBI** feature, **ANOVA + XGB feature selection**, and **SMOTETomek**
balancing — none of which the paper's prose mentions.

## Decision

**Reproduce the notebook's executed "best configuration" R-Blend.** That code
produced the results in the paper's tables; the paper's F1 confidence-interval
table even reports **n = 48** test isolates for _K. pneumoniae_ Meropenem, which
is the notebook's **20 %** split of 238 isolates, not the "70/30" stated in the
paper's methods prose. The SVM+XGB equal-weight variant in the paper text is not
what was shipped.

Confirmed with the owner on 2026-08-27:

- **Test split:** `test_size = 0.20` (notebook's actual value).
- **Pipeline:** full best-config stack (feature selection + SMOTETomek), not the
  plain ensemble.

## Authoritative configuration

| Field | Value |
|---|---|
| **Authoritative model** | R-Blend — `VotingClassifier`, soft voting |
| **Estimators + weights** | `DecisionTreeClassifier(criterion="entropy", random_state=0)` → **1.0**  ·  `LogisticRegression(max_iter=1000, random_state=0)` → **1.5**  ·  `XGBClassifier(random_state=0, eval_metric="logloss")` → **1.0**  ·  SVM **excluded** (commented out in the notebook) |
| **RBI definition** | Paper Eq. 1.1–1.2. `r_i` = row-sum of the binary gene/mutation columns; `RBI_i = (r_i − r_min) / (r_max − r_min)`. **`r_min` / `r_max` are fitted on the TRAINING split only, persisted in `model.joblib`, and reused unchanged at inference (clipped to [0, 1]).** This fixes the notebook's leakage, where the min/max were fitted on the full dataset before splitting. Internally the feature is the column `R-Score`. |
| **Feature selection** | Union of: (a) ANOVA F-test features with `p ≤ 0.30` (`sklearn.feature_selection.f_classif`; `R-Score` force-kept); (b) features under an **85 %** cumulative-importance cutoff from an XGB ranker (`n_estimators=400, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42`). Selection is fitted on training data only; the resulting ordered feature list is frozen in `metadata.json → selectedFeatureCount` / the artifact's `selected_features`. |
| **Balancing** | Adaptive `SMOTETomek(random_state=42)` on the selected training features — inner SMOTE `k_neighbors = max(1, min(5, n_minority − 1))` (notebook `adaptive_smotetomek`, cell 39), so tiny datasets don't crash; skipped entirely if the minority class has < 2 samples. |
| **Split** | `train_test_split(test_size=0.20, random_state=42, stratify=y)` |
| **Seed** | `42` for split / feature-selection / SMOTETomek; base learners use `random_state=0` (as in the notebook). `random` + `numpy` seeded to 42 in `train.py`. |
| **Decision threshold** | `0.50` (argmax / `predict`) — no custom threshold in the notebook. |
| **Positive label** | `1` = **Resistant**. |
| **Datasets** | All of `Data/Read Data/amr_ast_<antibiotic>_<CODE>.csv` (12 files). Each: first column `Isolate`, then raw binary feature columns, last column = the antibiotic name (0/1 phenotype). A stray `sum` column is dropped if present. `data/raw/` is also scanned. If no CSV exists anywhere, a labelled synthetic stand-in is generated so the pipeline still runs. |
| **Reason** | This is the code path that generated the published results; the paper's SVM+XGB equal-weight description does not match the executed notebook, and the paper's own n=48 test size matches the notebook's 20 % split. |

## Deliberate deviations from the notebook

1. **RBI min/max fitted on train only** (notebook fitted on full data → leakage).
   Mandated by the product plan (Week 3) and the task brief.
2. **`use_label_encoder=` argument dropped** from `XGBClassifier` — it was
   removed in XGBoost ≥ 2.0 and raises under a 3.x `xgboost`. Behaviour is
   unchanged (labels are already 0/1).
3. No Google Drive / Colab paths anywhere.
4. **Deps are ranges, not exact pins**, and use `xgboost-cpu` — see
   `requirements.txt` for why (Python-3.13 wheels; avoiding the ~200 MB
   `nvidia-nccl` pulled by the GPU `xgboost` build). `scikit-learn` stays on the
   1.6 line for pickle compatibility with the committed artifacts; a
   version mismatch triggers an automatic in-process retrain from the bundled
   CSV (`src/predict.load_model` → `src/training.train_model`).

## As trained (2026-08-27, real data — all 12)

Environment: scikit-learn 1.6.1, xgboost 3.4.1, imbalanced-learn 0.14.2,
numpy 2.5.2, pandas 2.3.3, Python 3.12. Metrics shift by a point or two with the
xgboost minor version — that is expected, and each model's stored metrics always
match the environment that produced it (including a runtime retrain).

`meropenem-kn` (the default): 238 isolates, class balance {susceptible 132,
resistant 106}, 190 train / 48 test (stratified, seed 42 — matches the paper's
F1-CI table n = 48), 109 features kept, RBI bounds `r_min = 3` / `r_max = 29`.
Hold-out: **accuracy 0.917 · precision 0.870 · recall 0.952 · F1 0.909 ·
ROC-AUC 0.961 · PR-AUC 0.942** (TN 24 / FP 3 / FN 1 / TP 20).

All 12 (hold-out recall / F1):

| model_id | n | recall | F1 | note |
|---|---:|---:|---:|---|
| clindamycin-cj | 26 | 1.000 | 1.000 | small |
| doripenem-ecs | 49 | 0.800 | 0.889 | small |
| doripenem-kn | 316 | 0.980 | 0.970 | |
| doripenem-pa | 44 | 1.000 | 1.000 | small |
| ertapenem-ecs | 129 | 0.917 | 0.957 | |
| ertapenem-kn | 181 | 0.889 | 0.889 | |
| imipenem-ecs | 64 | 1.000 | 1.000 | |
| imipenem-kn | 200 | 0.957 | 0.917 | |
| kanamycin-se | 991 | 0.909 | 0.947 | |
| meropenem-ecs | 91 | 1.000 | 0.947 | |
| meropenem-kn | 238 | 0.952 | 0.909 | default |
| streptomycin-se | 1042 | 0.982 | 0.935 | |

Each is a single small hold-out split with a phylogeny-naive partition — treat as
an optimistic upper bound (see `docs/intended-use.md`). Live values are always in
`models/<model_id>/v1/{metadata,evaluation}.json` and `models/registry.json`.
