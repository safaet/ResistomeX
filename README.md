# R-Blend AMR Predictor — Streamlit app

A single, self-contained **Streamlit** app that proves the end-to-end
antimicrobial-resistance (AMR) prediction workflow:

**pick a pathogen–antibiotic → upload gene features (or use the example) → see
resistant/susceptible calls with probability, confidence and RBI → download.**

> **Research-use only.** Not a diagnostic. Not a substitute for antimicrobial
> susceptibility testing (AST) or clinical judgement. See
> [`docs/intended-use.md`](docs/intended-use.md).

---

## Problem statement

Most genomic AMR models treat resistance genes as independent binary
presence/absence flags and ignore the **cumulative burden** of resistance
determinants in an isolate. This project reproduces the **R-Blend** ensemble from
the accompanying book chapter and serves it for **every dataset in
`Data/Read Data/`** — 12 pathogen–antibiotic combinations across
*K. pneumoniae*, *E. coli / Shigella*, *P. aeruginosa*, *S. enterica* and
*C. jejuni*.

## What the RBI adds

The **Resistome Burden Index** is a single 0–1 feature per isolate:

1. `r = ` number of resistance genes / mutations present (row-sum of the binary
   feature columns).
2. `RBI = (r − r_min) / (r_max − r_min)`, where `r_min` / `r_max` are learned
   **from that model's training split only** and frozen inside the model artifact
   (never recomputed from an uploaded file). Values are clipped to `[0, 1]`.

## Model (same config for all 12)

R-Blend = soft-voting `VotingClassifier`:

| Estimator | Weight |
|---|---|
| `DecisionTreeClassifier(criterion="entropy")` | 1.0 |
| `LogisticRegression(max_iter=1000)` | 1.5 |
| `XGBClassifier(eval_metric="logloss")` | 1.0 |

with **ANOVA (p ≤ 0.30) ∪ XGB-cumulative-importance-85 %** feature selection and
**adaptive SMOTETomek** balancing on the training split. Stratified hold-out
split, `test_size=0.20`, `random_state=42`, positive class = resistant,
threshold 0.5.

Full rationale and the paper-vs-notebook reconciliation:
[`docs/model-decision.md`](docs/model-decision.md).

---

## Install

Python 3.10–3.12. Run everything from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Training data is read from **`Data/Read Data/*.csv`** (already in the repo).
Filenames must be `amr_ast_<antibiotic>_<PATHOGENCODE>.csv`; each file is
`Isolate` + binary `gene=STATE` columns + a trailing 0/1 phenotype column named
after the antibiotic. A stray pre-computed `sum` column is dropped automatically.

`data/raw/` is an optional extra location that is also scanned (git-ignored).
If **no** CSV is found anywhere, `train.py` writes one clearly-labelled
**synthetic** dataset so the app still runs — the UI then shows a *SYNTHETIC
MODE* banner.

## Train

```bash
python train.py                 # trains one model per dataset (~10 s total)
python train.py --only meropenem-kn
python train.py --list
```

Per model, writes `models/<model_id>/v1/`:

| File | Contents |
|---|---|
| `model.joblib` | fitted ensemble + RBI bounds + feature lists |
| `feature_schema.json` | identifier / label columns, allowed values, **ordered** raw feature columns |
| `metadata.json` | model type, seed, rows, feature counts, RBI min/max, library versions |
| `evaluation.json` | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |

Plus a top-level **`models/registry.json`** listing every model, and demo CSVs
`data/samples/<model_id>_{valid,invalid}_input.csv`.

## Run

```bash
streamlit run app.py
```

One screen: the **sidebar** picks the pathogen–antibiotic model, offers the CSV
template, and an "use built-in example isolates" toggle. The **main area** takes
your upload, shows a clear ✅/❌ validation panel, and — as soon as the file is
valid — renders summary metric cards, a per-isolate table
(`isolateId · prediction · resistantProbability · confidence · rbi ·
resistanceGeneCount`, with text labels + icons, not colour alone) and a
**Download results** button. An expandable panel shows the RBI explanation, that
model's metrics + confusion matrix, and the limitations.

## Input format

- CSV, ≤ 10 MB, ≤ 5 000 rows.
- `Isolate` column: non-empty, unique.
- Every feature column from the selected model's template present, values
  **0 or 1 only**, no blanks.
- Do **not** include a phenotype/label column (ignored with a warning).
- Unknown columns are ignored with a warning; column order is normalised to the
  schema automatically.

## Model metrics

See each `models/<model_id>/v1/evaluation.json` (rendered in the app's details
panel) and the summary in `models/registry.json`. Metrics come from a single
small hold-out split — read them as indicative.

## Limitations

- Small test sets → high-variance metrics (models trained on < 60 isolates are
  flagged `*` / "low confidence").
- Phylogeny-naive random split → **optimistic** performance estimate (closely
  related isolates can span train/test).
- No external validation; one data source, one annotation pipeline
  (NCBI AMRFinderPlus).
- Each model only knows its own training gene schema — novel determinants are
  invisible.

## Out of scope for this MVP

Auth, user accounts, databases, Docker, FastAPI, React, model retraining,
EMR/LIS integration, and any clinical treatment advice.

## Citation

> As-ad, J., Arman, S. J., Hoque, K. I., & Khaliluzzaman, M.
> *A Resistome-Driven Ensemble Framework for Antimicrobial Resistance Phenotype
> Prediction.* Book chapter (Taylor & Francis). See `337_Camera Ready.pdf`.
