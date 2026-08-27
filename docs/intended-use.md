# Intended-use statement — R-Blend AMR Predictor

## What this application does

It predicts an antimicrobial-resistance **phenotype** (resistant vs.
susceptible) from validated resistance-gene / mutation feature data supplied as a
CSV. One R-Blend model is provided per pathogen–antibiotic dataset in
`Data/Read Data/` (12 combinations across *K. pneumoniae*, *E. coli / Shigella*,
*P. aeruginosa*, *S. enterica*, *C. jejuni* — carbapenems, aminoglycosides, and
clindamycin). The user selects the relevant model in the sidebar.

For each isolate it returns:

- a predicted label (`RESISTANT` / `SUSCEPTIBLE`),
- the probability of resistance and a confidence value,
- the **Resistome Burden Index (RBI)** — a 0–1 score of overall resistance-gene
  load, normalised against bounds learned from the training data,
- the raw count of resistance determinants detected.

## What it is intended for

Research, education, and software demonstration:

- exploring how a resistome-based ensemble behaves on gene-presence data,
- teaching the RBI concept,
- portfolio / product-workflow demonstration.

## What it is **not** intended for

- It is **not** a medical device and **not** a diagnostic.
- It is **not** a substitute for antimicrobial susceptibility testing (AST),
  culture, clinical diagnosis, or professional medical judgement.
- It must **not** be used to select, start, stop, or change antibiotic therapy
  for a patient.
- Each model is specific to one pathogen–antibiotic pair and one training gene
  schema. Use the model that matches your isolates; applying a model to a
  different species or drug is out of scope and unsupported.

## Key limitations (see the paper's limitations section)

- **Small test sets.** Metrics come from a single hold-out split of a few
  hundred isolates; point estimates carry high variance.
- **Phylogeny-naive evaluation.** Random stratified splitting lets closely
  related isolates fall on both sides of the split, which produces an
  **optimistic upper-bound** estimate of performance rather than a
  clinically representative one.
- **No external validation.** All data derive from a single source and a single
  annotation pipeline (NCBI AMRFinderPlus). Behaviour on data from other
  pipelines or prospective clinical isolates is unknown.
- **Bounded feature space.** Resistance determinants not present in the training
  schema cannot influence a prediction.

## Result wording to show users

> Prediction: Resistant
> Resistant probability: 0.91
> Model version: meropenem-kn/v1
> Research-use result — not laboratory confirmation
