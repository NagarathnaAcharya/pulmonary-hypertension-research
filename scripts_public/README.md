# Public research scripts

These are reviewed, privacy-conscious copies of three analysis steps. They
require MIMIC-IV access and private intermediate tables created by the full
local research workflow. No datasets, cohort extracts, patient-level
predictions, models, or generated figures are included here.

## Environment

- Python 3.13 is the environment recorded for this project; other Python
  versions have not been validated here.
- Install the dependencies from the repository root:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  ```

- Obtain and use MIMIC-IV only under the dataset's access and data-use terms.
  Configure all input paths to your own authorized files.

## Execution order

The scripts begin after the original local pipeline has created
`landmark_table.csv` and `features_all.csv`. Those patient-level inputs are
deliberately not supplied. In the original workflow, `step_d.py` through
`step_r.py` prepare them from authorized MIMIC-IV data; those scripts remain
private because they are exploratory and create patient-level intermediate
files.

1. **`prepare_landmark_labs.py`** — Maps blood lab item IDs and extracts each
   patient's last measured value in the 24 hours after the index time. It
   combines private `landmark_table.csv` and `features_all.csv` inputs with the
   authorized MIMIC-IV `hosp/d_labitems.csv.gz` and
   `hosp/labevents.csv.gz` tables. Its output is a patient-level table and must
   be stored outside the repository.

   ```powershell
   python scripts_public\prepare_landmark_labs.py `
     --landmark-table "<PRIVATE_DATA_DIR>\landmark_table.csv" `
     --features "<PRIVATE_DATA_DIR>\features_all.csv" `
     --lab-items "<MIMIC_IV_DIR>\hosp\d_labitems.csv.gz" `
     --lab-events "<MIMIC_IV_DIR>\hosp\labevents.csv.gz" `
     --output "<PRIVATE_OUTPUT_DIR>\landmark_table_lm.csv"
   ```

2. **`evaluate_mortality.py`** — Runs the repeated stratified 10-fold
   logistic-regression and XGBoost evaluation and prints aggregate metrics.
   It does not write patient-level predictions.

   ```powershell
   python scripts_public\evaluate_mortality.py `
     --input-csv "<PRIVATE_OUTPUT_DIR>\landmark_table_lm.csv"
   ```

3. **`explain_mortality.py`** — Calculates fold-level SHAP summaries and
   bootstrap explanation stability. It writes an aggregate feature-ranking
   CSV and a summary plot only. These are research results; keep them private
   unless separately reviewed and approved for release.

   ```powershell
   python scripts_public\explain_mortality.py `
     --input-csv "<PRIVATE_OUTPUT_DIR>\landmark_table_lm.csv" `
     --output-dir "<PRIVATE_OUTPUT_DIR>\shap_results"
   ```

The scripts reject output paths inside the repository to reduce the risk of
accidentally committing patient-level or research-result files. Keep the
private inputs and outputs under your own access-controlled storage.

## Scope and limitations

This is not a one-command reproduction of the complete study. It does not
reconstruct the initial MIMIC-IV cohort, eICU cohort, or all sensitivity and
validation analyses. It expects authorized datasets and the stated private
intermediate tables. The public scripts preserve the corresponding local
analysis steps while exposing input/output paths and avoiding persistent
patient-level prediction files.
