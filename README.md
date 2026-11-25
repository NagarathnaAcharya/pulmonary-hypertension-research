# PulmoScope – Pulmonary Hypertension Research

## Project Overview
PulmoScope is a research project that explores machine learning methods for pulmonary hypertension-related risk classification and clinical outcome prediction using retrospective ICU data.

## Research Objectives
- Develop a model for elevated mean pulmonary artery pressure (mPAP) classification.
- Explore 7-day mortality prediction using a 24-hour landmark.
- Evaluate external model performance using eICU data.
- Study model interpretability and feature importance.

## Technologies Used

- Python
- Machine Learning
- Scikit-learn
- XGBoost
- SHAP
- HTML, CSS and JavaScript

## Research Pipeline

The local analysis was built as sequential research scripts rather than a single
installable application. The main MIMIC-IV path was:

1. `step_a.py` and `step_b.py`: inspect measurement identifiers and extract
   pulmonary artery catheter measurements.
2. `step_d.py`, `step_e.py`, `step_f.py` and `step_r.py`: define the study
   cohort, assemble baseline and laboratory features, and create the mortality
   landmark table.
3. `step_x3b_build_lm_labs.py` and `step_x5_tables.py`: build the updated
   landmark-window laboratory table and evaluate mortality models.
4. `step_x6_figures.py`, `step_x8_sofa.py` and `step_x9c_shap.py`: create
   aggregate figures, compare with the simplified SOFA score, and calculate
   feature explanations.

The eICU external-validation branch uses separate cohort and feature-preparation
scripts followed by the `step_x10*` validation and updating analyses. These
original scripts remain local and are not part of the public script subset.

The reviewed public examples are in [`scripts_public/README.md`](scripts_public/README.md).
They start from private, locally prepared cohort tables; they do not reproduce
the entire raw-data ingestion pipeline on their own.

## Dashboard

The project includes a static research dashboard presenting study methodology,
evaluation summaries and external validation findings. It displays a fixed
research snapshot and does not perform live inference or connect to a model
service.

## Data Privacy

MIMIC-IV and eICU datasets, credentials, patient-level records, derived cohort
tables, prediction files and model artifacts are not included in this
repository. The original `scripts/` workspace and its local artifacts are
ignored by Git. Public scripts require users to provide their own authorized
data and to keep all generated private tables and outputs outside the
repository.

Figures generated from the protected research data are also omitted from the
public repository; the dashboard uses explanatory placeholders instead of
linking to ignored local files.

## Important Disclaimer
This project is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions.

## Current Status

Research analysis and dashboard development. Results and methodology should be interpreted in the context of the study limitations.