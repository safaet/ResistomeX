# AMR Prediction Product — 10-Week MVP Plan

## 1. Project Goal

Build and deploy **R-Blend AMR Predictor**, a research-use web application where a user can:

1. Select a pathogen–antibiotic model.
2. Download the required CSV template.
3. Upload resistance-gene feature data.
4. Validate the uploaded dataset.
5. Calculate the Resistome Burden Index (RBI).
6. Predict resistant or susceptible phenotype.
7. View probability, confidence, and RBI.
8. Download prediction results.

## 2. Recommended Technology Stack

```text
React frontend
      ↓
FastAPI prediction service
      ↓
Versioned R-Blend model artifacts
      ↓
Docker deployment
```

### Version 1 exclusions

Do not add these features to the first MVP:

- User registration
- Authentication and role management
- Database
- Spring Boot
- Hospital or laboratory integration
- Electronic medical record integration
- Automatic model retraining
- Multiple organizations
- Clinical treatment recommendations
- Automatic AMRFinderPlus execution

The first objective is to prove the complete prediction workflow.

---

# Phase 0 — Freeze the MVP Scope

## Duration

2–3 days

## Step 1: Choose one prediction model

Start with one pathogen–antibiotic combination.

Recommended first model:

```text
Pathogen: Klebsiella pneumoniae
Antibiotic: Meropenem
Model ID: meropenem-kn
```

Do not support all available datasets in version 1.

## Step 2: Write the intended-use statement

Create:

```text
docs/intended-use.md
```

Suggested content:

```text
This application predicts antimicrobial-resistance phenotypes
from validated resistance-gene feature data.

It is intended for research, education, and software demonstration.

It is not intended to replace antimicrobial susceptibility
testing, clinical diagnosis, or professional medical judgment.
```

## Step 3: Confirm the authoritative R-Blend model

Resolve the difference between:

- The README description of an SVM–XGBoost soft-voting ensemble.
- The notebook implementation that may contain Decision Tree, Logistic Regression, XGBoost, or commented-out SVM code.

Create:

```text
docs/model-decision.md
```

Record:

```text
Authoritative model:
Model parameters:
Voting weights:
Training split:
Random seed:
Decision threshold:
Positive label:
Dataset:
Reason for selection:
Approved by:
Approval date:
```

## Phase 0 completion criteria

- [ ] One dataset selected
- [ ] One model selected
- [ ] Intended use documented
- [ ] Model definition matches the paper, README, and code
- [ ] MVP exclusions documented

---

# Week 1 — Restructure the Repository

## Objective

Separate research exploration from production code.

## Target structure

```text
AMR-Prediction/
├── backend/
│   ├── app/
│   ├── models/
│   └── tests/
├── frontend/
├── training/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── notebooks/
├── docs/
├── .github/
│   └── workflows/
├── Dockerfile
├── compose.yaml
└── README.md
```

## Tasks

1. Keep the existing notebook inside `notebooks/`.
2. Move datasets to `data/raw/`.
3. Remove absolute Google Drive paths from production code.
4. Create a YAML training configuration.
5. Identify the exact:
   - Identifier column
   - Label column
   - Resistance-feature columns
   - Label values
   - Missing-value policy
6. Create small valid and invalid sample CSV files.

## Example configuration

Create:

```text
training/config/meropenem-kn.yaml
```

```yaml
model_id: meropenem-kn
model_version: 0.1.0

dataset:
  path: data/raw/amr_ast_meropenem_KN.csv
  identifier_column: Isolate
  label_column: phenotype

training:
  test_size: 0.30
  random_state: 42
  positive_class: 1
```

## Deliverables

- `training/config/meropenem-kn.yaml`
- `data/samples/valid_input.csv`
- `data/samples/invalid_input.csv`
- `docs/input-format.md`
- `docs/architecture.md`

## Completion criteria

- [ ] Training no longer depends on Google Drive
- [ ] Dataset location comes from configuration
- [ ] Input schema is documented
- [ ] Sample input files are available

---

# Week 2 — Build the Data-Validation Pipeline

## Objective

Ensure that only valid resistance-feature data reaches the model.

## Implement

```text
training/data_loader.py
training/data_validation.py
backend/app/ml/validation_service.py
```

## Validation rules

The application must check:

- [ ] File is a valid CSV
- [ ] File is not empty
- [ ] Maximum row count is enforced
- [ ] Isolate column exists
- [ ] Isolate IDs are not empty
- [ ] Isolate IDs are unique
- [ ] Required feature columns exist
- [ ] Feature values contain only `0` or `1`
- [ ] Missing feature values are rejected
- [ ] Label column is excluded from prediction uploads
- [ ] Columns are reordered according to the model schema
- [ ] Unknown columns generate a warning or are rejected according to policy

## Feature schema

Create:

```text
backend/models/meropenem-kn/v1/feature_schema.json
```

Example:

```json
{
  "modelId": "meropenem-kn",
  "identifierColumn": "Isolate",
  "labelColumn": "phenotype",
  "allowedValues": [0, 1],
  "featureColumns": [
    "feature_1",
    "feature_2",
    "feature_3"
  ]
}
```

## Required tests

Test these cases:

1. Valid input
2. Empty CSV
3. Missing isolate column
4. Duplicate isolate ID
5. Missing feature
6. Additional feature
7. Text feature value
8. Value outside `0` and `1`
9. Missing value
10. Incorrect column order

## Completion criteria

- [ ] Invalid files receive clear errors
- [ ] Feature order is deterministic
- [ ] Valid sample data passes validation
- [ ] Validation tests pass

---

# Week 3 — Extract RBI and Training Logic

## Objective

Replace notebook-dependent training with a reproducible Python pipeline.

## Implement

```text
training/feature_engineering.py
training/model_factory.py
training/train.py
backend/app/ml/rbi_transformer.py
```

## RBI training workflow

```text
Training data
    ↓
Calculate raw resistance-gene count
    ↓
Learn RBI minimum and maximum from training data
    ↓
Normalize the RBI
    ↓
Store normalization parameters inside the model pipeline
```

## RBI prediction workflow

```text
Uploaded data
    ↓
Calculate raw resistance-gene count
    ↓
Use saved training minimum and maximum
    ↓
Calculate RBI
    ↓
Run prediction
```

Do not fit RBI normalization independently for each uploaded CSV.

## Training pipeline

```text
RBI feature engineering
       ↓
SVM preprocessing
       ↓
SVM + XGBoost ensemble
       ↓
Probability output
```

## Save artifacts

```text
backend/models/meropenem-kn/v1/
├── model.joblib
├── metadata.json
├── feature_schema.json
├── evaluation.json
└── model_card.md
```

## Metadata example

```json
{
  "modelId": "meropenem-kn",
  "version": "1.0.0",
  "modelType": "SVM-XGBoost Soft Voting",
  "pathogen": "Klebsiella pneumoniae",
  "antibiotic": "Meropenem",
  "positiveClass": "Resistant",
  "randomState": 42,
  "trainingRows": 166,
  "testRows": 72,
  "featureCount": 250
}
```

## Training command

```bash
python -m training.train \
  --config training/config/meropenem-kn.yaml
```

## Completion criteria

- [ ] Model trains by running one command
- [ ] No notebook execution is required
- [ ] Training is deterministic
- [ ] RBI parameters are stored with the model
- [ ] Model artifact and metadata are generated

---

# Week 4 — Evaluate and Approve the Model

## Objective

Verify that the model is suitable for a research MVP.

## Required evaluation metrics

| Metric | Purpose |
|---|---|
| Recall | Measures how many resistant cases were detected |
| Precision | Measures reliability of resistant predictions |
| F1-score | Balances precision and recall |
| Accuracy | Measures overall correct predictions |
| ROC-AUC | Measures ranking performance |
| PR-AUC | Useful for class-imbalanced data |
| Confusion matrix | Shows false positives and false negatives |
| Calibration | Checks whether probabilities are trustworthy |

## Cross-validation

Use:

```text
5-fold stratified cross-validation
```

Keep the final test set untouched until model selection is complete.

## Golden dataset

Create:

```text
backend/tests/fixtures/golden_input.csv
backend/tests/fixtures/golden_output.json
```

Use 5–20 representative isolates. Every release must reproduce the expected predictions within an accepted tolerance.

## Model card

Create:

```text
backend/models/meropenem-kn/v1/model_card.md
```

Include:

- Intended use
- Unsupported use
- Dataset source
- Dataset size
- Feature definitions
- Label definition
- Model architecture
- Training procedure
- Evaluation metrics
- Decision threshold
- Known limitations
- Bias risks
- Model version
- Responsible owner
- Approval date

## Example acceptance gate

Set the final values after evaluating the real model.

```text
Recall ≥ 0.90
F1-score ≥ 0.88
ROC-AUC ≥ 0.90
No data leakage found
Golden dataset test passes
```

Do not adopt these thresholds without evaluating the real data.

## Completion criteria

- [ ] Cross-validation completed
- [ ] Final test evaluation completed
- [ ] Model card written
- [ ] Golden dataset created
- [ ] Acceptance gate passed

---

# Week 5 — Build the FastAPI Service

## Objective

Expose the model through a safe, documented prediction API.

## Endpoints

```text
GET  /api/v1/health
GET  /api/v1/models
GET  /api/v1/models/{modelId}
GET  /api/v1/templates/{modelId}
POST /api/v1/predictions
```

## Prediction workflow

```text
Receive CSV
    ↓
Validate file type and size
    ↓
Read CSV
    ↓
Validate feature schema
    ↓
Calculate RBI
    ↓
Run model
    ↓
Return prediction results
```

## Example response

```json
{
  "requestId": "request-123",
  "modelId": "meropenem-kn",
  "modelVersion": "1.0.0",
  "totalIsolates": 2,
  "resistantCount": 1,
  "susceptibleCount": 1,
  "results": [
    {
      "isolateId": "Sample-001",
      "prediction": "RESISTANT",
      "resistantProbability": 0.93,
      "confidence": 0.93,
      "rbi": 0.74,
      "resistanceGeneCount": 26
    }
  ]
}
```

## Standard error response

```json
{
  "code": "MISSING_FEATURES",
  "message": "The uploaded CSV is missing required features.",
  "details": {
    "missingFeatures": [
      "blaKPC=COMPLETE"
    ]
  },
  "requestId": "request-123"
}
```

## Required API tests

- [ ] Health endpoint
- [ ] Model list
- [ ] Unknown model
- [ ] Valid prediction
- [ ] Invalid extension
- [ ] Oversized upload
- [ ] Missing columns
- [ ] Invalid values
- [ ] Deterministic prediction

## Completion criteria

- [ ] Swagger documentation works
- [ ] Valid CSV produces predictions
- [ ] Invalid CSV produces useful errors
- [ ] Every prediction includes model version
- [ ] API tests pass

---

# Week 6 — Build the React Frontend

## Objective

Create a user-friendly product demonstration.

## Pages

### Home

Show:

- Product purpose
- Supported pathogen
- Supported antibiotic
- How the system works
- Research-use disclaimer

### Prediction page

Include:

- Model selector
- Download-template button
- CSV upload area
- File validation
- Predict button

### Results page

Show:

- Total isolates
- Resistant count
- Susceptible count
- Resistance percentage
- Average RBI
- Average confidence
- Prediction table
- Download CSV button

### Model information page

Show:

- Model version
- Dataset size
- Feature count
- Recall
- Precision
- F1-score
- ROC-AUC
- Known limitations

## Required user journey

```text
Open application
→ Select model
→ Download template
→ Upload CSV
→ Review validation
→ Run prediction
→ View results
→ Download output
```

## Completion criteria

- [ ] User can complete the workflow without technical help
- [ ] Errors are understandable
- [ ] Results are downloadable
- [ ] Disclaimer is visible
- [ ] UI works on desktop and mobile

---

# Week 7 — Add Quality, Security, and Accessibility

## Objective

Make the application suitable for a public research demonstration.

## Security tasks

Implement:

- [ ] 10 MB upload limit
- [ ] 5,000-row limit
- [ ] CSV-only upload
- [ ] Restricted CORS
- [ ] HTTPS in production
- [ ] No permanent upload storage
- [ ] No untrusted model uploads
- [ ] Non-root Docker user
- [ ] Dependency scanning
- [ ] Secret scanning
- [ ] Safe error handling
- [ ] Request IDs
- [ ] Rate limiting

Do not log:

- Full uploaded datasets
- Genomic feature rows
- Patient information
- Private identifiers
- API secrets

## Accessibility tasks

- [ ] Keyboard-accessible controls
- [ ] Proper form labels
- [ ] Visible focus indicator
- [ ] Accessible error messages
- [ ] Sufficient color contrast
- [ ] No color-only prediction meaning
- [ ] Responsive tables
- [ ] Screen-reader-friendly status messages

Display results clearly:

```text
Prediction: Resistant
Resistant probability: 91%
Model version: 1.0.0
Research-use result — not laboratory confirmation
```

## Completion criteria

- [ ] Security checklist completed
- [ ] No sensitive data is logged
- [ ] Upload limits are enforced
- [ ] Keyboard navigation works
- [ ] Critical dependency vulnerabilities are resolved

---

# Week 8 — Docker and CI/CD

## Objective

Make the build reproducible and automatically tested.

## Docker files

Create:

```text
Dockerfile
compose.yaml
.dockerignore
docker-entrypoint.sh
```

## Multi-stage Docker build

```text
Node stage
    ↓
Build React application
    ↓
Python stage
    ↓
Install FastAPI application
    ↓
Copy React build
    ↓
Copy model artifacts
    ↓
Start application
```

## GitHub Actions workflow

Run on every push and pull request:

```text
Checkout
→ Python setup
→ Install dependencies
→ Lint backend
→ Run unit tests
→ Run API tests
→ Node setup
→ Build frontend
→ Build Docker image
→ Start container
→ Test health endpoint
```

## Branch strategy

```text
main        Production-ready code
develop     Integration branch
feature/*   Feature development
fix/*       Bug fixes
```

Do not merge into `main` unless:

- [ ] Tests pass
- [ ] Frontend builds
- [ ] Docker image builds
- [ ] Container health test passes
- [ ] Code review is completed

## Completion criteria

- [ ] Docker image builds locally
- [ ] Container starts successfully
- [ ] Health endpoint works
- [ ] GitHub Actions passes
- [ ] Pull-request protection is enabled

---

# Week 9 — Staging Deployment and Beta Testing

## Objective

Test the complete product before public release.

## Staging configuration

```text
APP_ENV=staging
MAX_UPLOAD_MB=10
LOG_LEVEL=INFO
```

Health endpoint:

```text
/api/v1/health
```

## Beta testers

Ask 3–5 people:

- Research teammate
- Biology or microbiology student
- ML engineer
- Software engineer
- Non-technical user

## Beta-test tasks

Ask users to:

1. Download the CSV template.
2. Upload a valid file.
3. Upload an invalid file.
4. Understand the result.
5. Download prediction output.
6. Locate model limitations.

## Feedback categories

| Category | Example |
|---|---|
| Bug | File upload fails |
| Usability | User cannot find template |
| Explanation | RBI is unclear |
| Performance | Prediction is slow |
| Trust | Model version is missing |
| Accessibility | Error is color-only |

## Completion criteria

- [ ] Staging deployment works
- [ ] At least three users tested the product
- [ ] Critical bugs are fixed
- [ ] User instructions are improved
- [ ] End-to-end workflow is verified

---

# Week 10 — Production Release

## Objective

Release a portfolio-ready research MVP.

## Release checklist

- [ ] Model approved
- [ ] Model card complete
- [ ] API tests pass
- [ ] Frontend build passes
- [ ] Docker health test passes
- [ ] Security checklist complete
- [ ] Disclaimer visible
- [ ] Sample CSV available
- [ ] Architecture documented
- [ ] Deployment documented
- [ ] Rollback process documented

## Create release tag

```bash
git tag -a v1.0.0 -m "Release AMR Prediction research MVP"
git push origin v1.0.0
```

## README sections

Include:

- Problem statement
- Product screenshots
- Live demo link
- Architecture
- Supported model
- Input format
- Local setup
- Docker setup
- API documentation
- Model performance
- Known limitations
- Research-use disclaimer
- Citation

## Demonstration order

1. Explain the AMR problem.
2. Explain the RBI contribution.
3. Show the system architecture.
4. Download the CSV template.
5. Upload sample data.
6. Show validation.
7. Run prediction.
8. Explain resistant probability.
9. Explain RBI.
10. Download results.
11. Show model metrics.
12. Explain limitations.

---

# Suggested Weekly Work Schedule

Assuming you are working alongside a full-time job:

| Day | Activity |
|---|---|
| Monday | Requirements and design |
| Tuesday | Backend or ML implementation |
| Wednesday | Backend or ML implementation |
| Thursday | Frontend or integration |
| Friday | Tests and bug fixes |
| Saturday | 3–4 hour focused development session |
| Sunday | Documentation and weekly review |

Suggested effort:

```text
Weekdays: 1–1.5 hours per day
Saturday: 3–4 hours
Sunday: 1–2 hours

Total: approximately 10–13 hours per week
```

---

# First Seven Tasks

Start with these tasks in order:

1. Create `docs/intended-use.md`.
2. Confirm the authoritative R-Blend model.
3. Select `meropenem-kn` for version 1.
4. Create `training/config/meropenem-kn.yaml`.
5. Remove Google Drive paths from training logic.
6. Export and document the exact feature list.
7. Create `data/samples/valid_input.csv`.

Do not start React development or deployment until these seven tasks are complete.

---

# Milestone Map

| Milestone | Target |
|---|---|
| M1 | Scope and model frozen |
| M2 | Data contract validated |
| M3 | Reproducible model generated |
| M4 | Model evaluation approved |
| M5 | Prediction API complete |
| M6 | React user journey complete |
| M7 | Security and quality checks complete |
| M8 | Docker and CI operational |
| M9 | Staging validated |
| M10 | Version 1.0 deployed |

---

# Final Development Order

```text
Research consistency
→ Reproducible ML
→ Model validation
→ Prediction API
→ User interface
→ Testing and security
→ Docker and CI/CD
→ Staging
→ Production
```

---

# Final Definition of Done

The MVP is complete when:

- [ ] One authoritative model is selected.
- [ ] Training runs without notebook execution.
- [ ] No Google Drive paths remain in production code.
- [ ] Model definition matches the paper and README.
- [ ] RBI is fitted only from training data.
- [ ] Model artifact includes a fixed feature order.
- [ ] Dependency versions are recorded.
- [ ] Valid CSV produces predictions.
- [ ] Invalid CSV produces useful errors.
- [ ] Every result includes RBI, probability, and model version.
- [ ] Unit and API tests pass.
- [ ] React production build succeeds.
- [ ] Docker image builds successfully.
- [ ] Container health check passes.
- [ ] GitHub Actions passes.
- [ ] Staging is validated.
- [ ] Public deployment is accessible.
- [ ] Research-use disclaimer is visible.
