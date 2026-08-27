"""
R-Blend AMR Predictor — Streamlit app (single screen: give input, see output).

    streamlit run app.py

One R-Blend model per pathogen-antibiotic dataset in `Data/Read Data/`.
No FastAPI, no React, no Docker, no database, no auth.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import config
from src.predict import (
    ModelNotTrainedError,
    load_model,
    load_registry,
    usable_models,
)
from src.samples import csv_template_bytes, example_valid_bytes
from src.validation import validate_upload

st.set_page_config(page_title="R-Blend AMR Predictor", page_icon="🧫", layout="wide")

DISCLAIMER_SHORT = (
    "Research-use only — **not** a substitute for antimicrobial susceptibility "
    "testing (AST) or clinical judgement."
)
DISCLAIMER_FULL = (
    "**Research-use result — not laboratory confirmation.** This tool predicts "
    "antimicrobial-resistance phenotypes from resistance-gene feature data for "
    "research, education, and software demonstration. It is **not** a substitute "
    "for antimicrobial susceptibility testing, clinical diagnosis, or professional "
    "medical judgement, and must not be used to guide treatment."
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _registry():
    return load_registry()


@st.cache_resource(show_spinner="Loading model (first use may take a few seconds)…")
def _model(model_id: str):
    return load_model(model_id)


def _read_doc(name: str) -> str:
    path = config.DOCS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
def not_trained_screen(exc: Exception) -> None:
    st.title("🧫 R-Blend AMR Predictor")
    st.error("No trained models found yet.")
    st.markdown(
        "Run the training pipeline once from the repository root — it trains one "
        "model per dataset in `Data/Read Data/`:\n\n```bash\npython train.py\n```\n\n"
        "Then reload this page."
    )
    with st.expander("Technical detail"):
        st.code(str(exc))


def sidebar_pick(models):
    """Header + model selector only. Returns the chosen ModelInfo."""
    with st.sidebar:
        st.header("🧫 R-Blend AMR Predictor")
        st.caption("Predict resistant / susceptible phenotype from resistance-gene data.")
        by_label = {m.label: m for m in models}
        labels = list(by_label)
        default_label = next(
            (m.label for m in models if m.model_id == config.DEFAULT_MODEL_ID), labels[0]
        )
        picked = st.selectbox(
            "1 · Pathogen – antibiotic", options=labels, index=labels.index(default_label)
        )
        return by_label[picked]


def sidebar_rest(info, model) -> bool:
    """Metric caption, template download, example toggle, disclaimer. Returns use_example."""
    with st.sidebar:
        bits = []
        if info.metrics.get("recall") is not None:
            bits.append(f"recall {info.metrics['recall']:.2f}")
        if info.metrics.get("f1") is not None:
            bits.append(f"F1 {info.metrics['f1']:.2f}")
        if bits:
            st.caption(f"Held-out {' · '.join(bits)} · n={info.total_rows}")
        if info.small_dataset:
            st.caption("⚠️ Small dataset — metrics are low-confidence.")

        st.download_button(
            "2 · Download CSV template",
            data=csv_template_bytes(model.schema),
            file_name=f"{model.model_id}_template.csv",
            mime="text/csv",
            use_container_width=True,
            help=f"{model.schema.n_features} feature columns + "
                 f"'{model.schema.identifier_column}', one example row.",
        )
        use_example = st.checkbox("…or use built-in example isolates", value=False)
        st.divider()
        st.caption(DISCLAIMER_SHORT)
        return use_example


def input_bytes(model, use_example: bool) -> tuple[bytes | None, str]:
    """Return (csv_bytes, source_label)."""
    if use_example:
        st.info("Using built-in example isolates (synthetic rows in the correct schema).")
        return example_valid_bytes(model.schema), "built-in example"

    up = st.file_uploader(
        "3 · Upload your isolate CSV",
        type=["csv"],
        accept_multiple_files=False,
        help=f"One row per isolate. {model.schema.n_features} binary (0/1) feature "
             f"columns + '{model.schema.identifier_column}'. Max 10 MB, "
             f"{config.MAX_ROWS:,} rows. Do not include a phenotype column.",
    )
    if up is None:
        return None, ""
    return up.getvalue(), up.name


def validation_panel(report) -> None:
    with st.container(border=True):
        if report.errors:
            st.markdown(f"**❌ File not usable — fix {len(report.errors)} issue(s):**")
            for e in report.errors:
                st.markdown(f"- ❌ {e}")
        else:
            st.markdown(f"**✅ File looks good — {report.n_isolates} isolate(s).**")
        for w in report.warnings:
            st.markdown(f"- ⚠️ {w}")


def results_view(result, model, source_label: str) -> None:
    res_df, s = result.results, result.summary

    st.subheader("Results")
    st.caption(f"Model **{result.model_version}** · input: {source_label}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total isolates", s["totalIsolates"])
    c2.metric("Resistant", s["resistantCount"])
    c3.metric("Susceptible", s["susceptibleCount"])
    c4, c5, c6 = st.columns(3)
    c4.metric("% Resistant", f"{s['percentResistant']}%")
    c5.metric("Average RBI", s["averageRbi"])
    c6.metric("Average confidence", s["averageConfidence"])

    show = res_df.copy()
    show["prediction"] = show["prediction"].map(
        {config.POSITIVE_LABEL_NAME: "🔴 RESISTANT", config.NEGATIVE_LABEL_NAME: "🟢 SUSCEPTIBLE"}
    )
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "isolateId": "Isolate",
            "prediction": "Prediction",
            "resistantProbability": st.column_config.NumberColumn("P(resistant)", format="%.3f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
            "rbi": st.column_config.NumberColumn("RBI", format="%.3f"),
            "resistanceGeneCount": st.column_config.NumberColumn("Resistance genes", format="%d"),
        },
    )

    out = res_df.copy()
    out.insert(0, "modelVersion", result.model_version)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download results as CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name=f"{model.model_id}_predictions.csv",
        mime="text/csv",
    )
    st.info(DISCLAIMER_FULL)


def details_expander(model) -> None:
    meta, ev = model.metadata, model.evaluation
    with st.expander("ℹ️  How this works · model performance · limitations"):
        st.markdown(
            f"""
**What the model does.** For each isolate it combines the individual
resistance-gene columns with the **Resistome Burden Index (RBI)** — a single
0–1 score of overall resistance-gene load — and predicts *resistant* or
*susceptible* to **{model.antibiotic}** for **{model.pathogen}**.

**RBI.** Count the resistance genes / mutations present, then rescale to 0–1
using the minimum and maximum counts seen in this model's **training data**
(`r_min = {meta.get('rbi', {}).get('rMin', '—')}`,
`r_max = {meta.get('rbi', {}).get('rMax', '—')}`). Those bounds are frozen in
the model — they are never recomputed from your upload. RBI ≈ 1 means the
isolate carries close to the largest gene load seen in training.

**This model**

| | |
|---|---|
| Model | `{meta.get('modelId', model.model_id)}` / `{meta.get('version', config.MODEL_VERSION)}` ({meta.get('dataMode', 'real')} data) |
| Ensemble | soft-voting DecisionTree + LogisticRegression (1.5×) + XGBoost |
| Dataset | `{meta.get('dataset', '—')}` — {meta.get('totalRows', '—')} isolates ({meta.get('trainingRows','—')} train / {meta.get('testRows','—')} test) |
| Class balance | {meta.get('classBalance', '—')} |
| Features | {meta.get('selectedFeatureCount', '—')} used of {meta.get('rawFeatureCount', '—')} (ANOVA p ≤ {config.ANOVA_P_THRESHOLD} ∪ XGB {config.XGB_IMPORTANCE_CUTOFF_PCT}% cumulative) |
| Balancing | {meta.get('balancing', '—')} · split: {meta.get('split', 'stratified')} · seed {meta.get('randomState', config.RANDOM_STATE)} |
"""
        )
        if ev:
            g1, g2, g3 = st.columns(3)
            g1.metric("Recall", ev.get("recall", "—"), help="Resistant isolates correctly detected — clinically prioritised.")
            g2.metric("Precision", ev.get("precision", "—"))
            g3.metric("F1-score", ev.get("f1", "—"))
            g4, g5, g6 = st.columns(3)
            g4.metric("Accuracy", ev.get("accuracy", "—"))
            g5.metric("ROC-AUC", ev.get("rocAuc", "—"))
            g6.metric("PR-AUC", ev.get("prAuc", "—"))
            cm = ev.get("confusionMatrix", {})
            if cm:
                st.markdown("**Confusion matrix** (hold-out test set)")
                st.table(pd.DataFrame(
                    [[cm.get("trueNegative"), cm.get("falsePositive")],
                     [cm.get("falseNegative"), cm.get("truePositive")]],
                    index=["Actual Susceptible", "Actual Resistant"],
                    columns=["Pred Susceptible", "Pred Resistant"],
                ))

        st.markdown(
            """
**Limitations**

- **Small test sets** → high-variance point estimates; read metrics as indicative.
- **Phylogeny-naive split.** Train/test are split by random sampling, not by
  lineage, so closely related isolates can span both sides — this *inflates*
  apparent performance. The underlying paper treats its numbers as an upper bound.
- **No external validation.** One data source, one annotation pipeline
  (NCBI AMRFinderPlus).
- **Bounded feature space.** Determinants absent from a model's training schema
  cannot affect its prediction.
"""
        )
        intended = _read_doc("intended-use.md")
        if intended:
            st.markdown("---")
            st.markdown(intended)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        registry = _registry()
    except ModelNotTrainedError as exc:
        not_trained_screen(exc)
        return

    models = usable_models()
    if not models:
        skipped = [m for m in registry if not m.is_usable]
        not_trained_screen(
            Exception(
                "The registry exists but no model finished training. "
                + "; ".join(f"{m.model_id}: {m.status}" for m in skipped)
            )
        )
        return

    info = sidebar_pick(models)
    try:
        model = _model(info.model_id)
    except ModelNotTrainedError as exc:
        st.title("🧫 R-Blend AMR Predictor")
        st.error(f"Could not load or build the **{info.label}** model.")
        st.caption(str(exc))
        st.info("Pick a different pathogen–antibiotic in the sidebar, or run "
                "`python train.py` locally.")
        return
    use_example = sidebar_rest(info, model)

    st.title(f"{model.antibiotic} resistance — *{model.pathogen}*")
    if model.trained_at_runtime:
        st.caption("ℹ️ This model was built on first use from the bundled dataset.")
    if model.is_synthetic:
        st.warning(
            "⚠️ **SYNTHETIC MODE** — no real dataset was found at training time, so "
            "this model was fitted on an auto-generated stand-in. Predictions are for "
            "interface demonstration only."
        )
    st.write(
        "Give the model a CSV of isolates and it returns a resistant / susceptible "
        "call, probability, confidence and RBI for each. Grab the template from the "
        "sidebar if you need the exact columns."
    )

    raw, source_label = input_bytes(model, use_example)

    if raw is None:
        st.info("⬅️ Upload a CSV in step 3, or tick **use built-in example isolates** in the sidebar.")
        details_expander(model)
        return

    report = validate_upload(raw, model.schema)
    validation_panel(report)
    if not report.ok:
        details_expander(model)
        return

    from src.predict import predict_from_clean_frame

    with st.spinner("Scoring isolates…"):
        result = predict_from_clean_frame(report.cleaned, report.identifiers, model)
    results_view(result, model, source_label)
    details_expander(model)


try:
    main()
except Exception as exc:  # noqa: BLE001 — never leave the user with a blank page
    import traceback

    st.title("🧫 R-Blend AMR Predictor")
    st.error(f"The app hit an unexpected error: {type(exc).__name__}: {exc}")
    st.caption(
        "If you just deployed, make sure dependencies installed from "
        "`requirements.txt`. Locally, run `python train.py` once, then "
        "`streamlit run app.py`."
    )
    with st.expander("Full traceback"):
        st.code("".join(traceback.format_exc()))
