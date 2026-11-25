"""Run aggregate SHAP explanations for the mortality model."""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute aggregate SHAP rankings and stability summaries."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for aggregate results; must be outside this repository.",
    )
    return parser.parse_args()


def make_xgb():
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        min_child_weight=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def make_lr():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000),
    )


def shap_values(model, data):
    values = shap.TreeExplainer(model).shap_values(data)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def main():
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input feature table not found: {input_csv}")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise ValueError("Aggregate outputs must be written outside the repository.")
    output_dir.mkdir(parents=True, exist_ok=True)

    landmark = pd.read_csv(input_csv)
    required = {
        "y_death7",
        "mpap_median",
        "co_median",
        "age",
        "male",
        "vaso_before_index",
        "vaso_first24h",
        "n_vaso_drugs_first24h",
        "mcs_before_index",
        "mcs_first24h",
        "epo_first24h",
    }
    missing = sorted(required - set(landmark.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    drop_columns = ["subject_id", "ph", "y_death7", "y_composite"]
    treatment = [
        "vaso_before_index",
        "vaso_first24h",
        "n_vaso_drugs_first24h",
        "mcs_before_index",
        "mcs_first24h",
        "epo_first24h",
    ]
    hemodynamics = ["mpap_median", "co_median"]
    pre_index_labs = [
        column for column in landmark.columns if column.startswith("lab_")
    ]
    labs = [
        column for column in landmark.columns if column.startswith("lm_lab_")
    ]
    if not labs:
        raise ValueError("No lm_lab_ features found in the input table.")

    baseline = [
        column
        for column in landmark.columns
        if column
        not in drop_columns + treatment + hemodynamics + pre_index_labs + labs
    ]
    features = baseline + labs + hemodynamics + treatment
    if not features or len(set(features)) != len(features):
        raise ValueError("The selected feature list is empty or contains duplicates.")
    group_of = {
        column: group
        for group, columns in [
            ("baseline", baseline),
            ("laboratory", labs),
            ("hemodynamics (mPAP, CO)", hemodynamics),
            ("early treatment", treatment),
        ]
        for column in columns
    }
    X = landmark[features]
    y = landmark["y_death7"].to_numpy()
    if pd.isna(y).any() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_death7 must contain non-missing binary labels (0/1).")
    if np.unique(y).size != 2 or np.bincount(y.astype(int)).min() < 10:
        raise ValueError("At least 10 examples of each outcome are needed for 10-fold CV.")
    sample_count, feature_count = X.shape
    top_k = 8

    shap_oof = np.zeros((sample_count, feature_count))
    compact_probabilities = np.zeros(sample_count)
    full_probabilities = np.zeros(sample_count)
    top10_counts, topk_counts = Counter(), Counter()
    splitter = StratifiedKFold(10, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        model = make_xgb().fit(X.iloc[train_index], y[train_index])
        full_probabilities[test_index] = model.predict_proba(X.iloc[test_index])[:, 1]
        shap_oof[test_index] = shap_values(model, X.iloc[test_index])
        order = np.argsort(
            -np.abs(shap_values(model, X.iloc[train_index])).mean(axis=0)
        )
        top10_counts.update(np.asarray(features)[order[:10]])
        selected = list(np.asarray(features)[order[:top_k]])
        topk_counts.update(selected)
        compact_probabilities[test_index] = (
            make_lr()
            .fit(X.iloc[train_index][selected], y[train_index])
            .predict_proba(X.iloc[test_index][selected])[:, 1]
        )
        print(f"Completed fold {fold}/10", flush=True)

    importance = np.abs(shap_oof).mean(axis=0)
    ranking = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": importance,
            "group": [group_of[column] for column in features],
        }
    )
    directions = []
    for column_index, column in enumerate(features):
        observed = X[column].notna().to_numpy()
        directions.append(
            spearmanr(
                X.loc[observed, column],
                shap_oof[observed, column_index],
            )[0]
            if observed.sum() > 50 and X.loc[observed, column].nunique() > 1
            else np.nan
        )
    ranking["direction"] = np.round(directions, 2)
    ranking["in_top10_folds"] = [top10_counts[column] for column in features]
    ranking["in_top8_folds"] = [topk_counts[column] for column in features]
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking.to_csv(output_dir / "shap_ranking_death.csv", index=False)
    print("\nTop 20 aggregate features:")
    print(ranking.head(20).round(3).to_string(index=False))

    group_share = (
        ranking.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
        / importance.sum()
    )
    print("\nAggregate share of total importance by predictor group (%):")
    print((group_share * 100).round(1).to_string())

    cardiac_output_index = features.index("co_median")
    cardiac_output_missing = X["co_median"].isna().to_numpy()
    print("\nCardiac output: measured versus missing")
    for label, mask in [
        ("measured", ~cardiac_output_missing),
        ("missing", cardiac_output_missing),
    ]:
        print(
            f"  {label:9s} n={mask.sum():5d} | death rate {y[mask].mean():.3f} | "
            f"mean SHAP {shap_oof[mask, cardiac_output_index].mean():+.3f}"
        )

    print("\nMean SHAP by fifth of selected feature values:")
    for column in ["mpap_median", "co_median", "lm_lab_creatinine", "lm_lab_lactate"]:
        if column not in features:
            continue
        values = pd.DataFrame(
            {"value": X[column].to_numpy(), "shap": shap_oof[:, features.index(column)]}
        ).dropna()
        if values["value"].nunique() < 2:
            continue
        values["bin"] = pd.qcut(values["value"], 5, duplicates="drop")
        print(f"\n{column}:")
        print(
            values.groupby("bin", observed=True)
            .agg(n=("shap", "size"), mean_shap=("shap", "mean"))
            .round(3)
            .to_string()
        )

    coefficients = np.abs(make_lr().fit(X, y)[-1].coef_[0])
    print("\nSpearman correlation, SHAP importance vs logistic coefficients:")
    print(f"{spearmanr(importance, coefficients)[0]:.2f}")
    print(
        f"\nCompact logistic AUROC {roc_auc_score(y, compact_probabilities):.3f}; "
        f"XGBoost AUROC {roc_auc_score(y, full_probabilities):.3f}"
    )

    stable = [column for column in features if topk_counts[column] >= 8]
    if len(stable) < 3:
        stable = [column for column, _ in topk_counts.most_common(top_k)]
    print("Features in the top 8 in at least 8 of 10 folds:", stable)

    train_index, test_index = train_test_split(
        np.arange(sample_count), test_size=0.2, stratify=y, random_state=42
    )
    X_train, y_train, X_test = (
        X.iloc[train_index],
        y[train_index],
        X.iloc[test_index],
    )
    original_importance = np.abs(
        shap_values(make_xgb().fit(X_train, y_train), X_test)
    ).mean(axis=0)
    rng = np.random.default_rng(1)
    correlations, top10_bootstrap = [], Counter()
    for bootstrap in range(30):
        sample = rng.integers(0, len(train_index), len(train_index))
        if np.unique(y_train[sample]).size < 2:
            continue
        importance_bootstrap = np.abs(
            shap_values(
                make_xgb().fit(X_train.iloc[sample], y_train[sample]), X_test
            )
        ).mean(axis=0)
        correlations.append(spearmanr(original_importance, importance_bootstrap)[0])
        top10_bootstrap.update(
            np.asarray(features)[np.argsort(-importance_bootstrap)[:10]]
        )
    if not correlations:
        raise ValueError("No valid bootstrap samples were available for stability analysis.")
    print("\nStability over 30 bootstrap refits:")
    print(
        f"Mean rank correlation: {np.mean(correlations):.3f} "
        f"(95% percentile range {np.percentile(correlations, 2.5):.3f}-"
        f"{np.percentile(correlations, 97.5):.3f})"
    )
    for column in np.asarray(features)[np.argsort(-original_importance)[:14]]:
        print(
            f"  {column:24s} "
            f"{top10_bootstrap[column] / len(correlations) * 100:5.1f}%"
        )

    plt.figure()
    shap.summary_plot(shap_oof, X, max_display=15, show=False)
    plt.savefig(
        output_dir / "figure_shap_death_summary.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()
    print(f"Aggregate results written to: {output_dir}")
    print("No patient-level prediction files were written.")


if __name__ == "__main__":
    main()
