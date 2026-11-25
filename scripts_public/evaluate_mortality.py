"""Evaluate mortality models with repeated and nested cross-validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute aggregate mortality-model evaluation metrics."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Private landmark feature table prepared by the research pipeline.",
    )
    return parser.parse_args()


def make_lr():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1),
    )


def make_xgb():
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=42,
    )


def main():
    input_csv = parse_args().input_csv.expanduser()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input feature table not found: {input_csv}")
    full = pd.read_csv(input_csv)

    required = {
        "y_death7",
        "co_median",
        "mpap_median",
        "age",
        "male",
        "vaso_before_index",
        "vaso_first24h",
        "n_vaso_drugs_first24h",
        "mcs_before_index",
        "mcs_first24h",
        "epo_first24h",
    }
    missing = sorted(required - set(full.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    y = full["y_death7"].to_numpy()
    if pd.isna(y).any() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_death7 must contain non-missing binary labels (0/1).")
    if np.unique(y).size != 2 or np.bincount(y.astype(int)).min() < 10:
        raise ValueError("At least 10 examples of each outcome are needed for 10-fold CV.")

    drop_cols = ["subject_id", "ph", "y_death7", "y_composite"]
    support = [
        "vaso_before_index",
        "vaso_first24h",
        "n_vaso_drugs_first24h",
        "mcs_before_index",
        "mcs_first24h",
        "epo_first24h",
    ]
    hemodynamics = ["mpap_median", "co_median"]
    pre_labs = [column for column in full.columns if column.startswith("lab_")]
    landmark_labs = [
        column for column in full.columns if column.startswith("lm_lab_")
    ]
    if not landmark_labs:
        raise ValueError("No lm_lab_ features found in the input table.")
    base = [
        column
        for column in full.columns
        if column
        not in drop_cols + support + hemodynamics + pre_labs + landmark_labs
    ]
    unit_columns = [
        column
        for column in base
        if column.startswith("first_careunit_")
        or column.startswith("admission_type_")
    ]
    cvicu_columns = [column for column in base if "CVICU" in column]
    if not cvicu_columns:
        raise ValueError("No CVICU indicator found among baseline features.")
    cvicu_column = cvicu_columns[0]

    primary = base + landmark_labs + hemodynamics + support
    if len(primary) != len(set(primary)):
        raise ValueError("The primary predictor list contains duplicate columns.")
    predictor_sets = {
        "A Primary": primary,
        "B Without treatment": base + landmark_labs + hemodynamics,
        "C Without mPAP and CO": base + landmark_labs + support,
        "D Without unit and admission type": [
            column for column in primary if column not in unit_columns
        ],
        "E Base + landmark labs": base + landmark_labs,
        "F Landmark labs only": landmark_labs,
        "G mPAP and CO only": hemodynamics,
    }
    for name, columns in predictor_sets.items():
        missing_columns = sorted(set(columns) - set(full.columns))
        if missing_columns:
            raise ValueError(f"Predictor set {name!r} lacks columns: {missing_columns}")

    model_makers = {"LogReg": make_lr, "XGBoost": make_xgb}
    repetitions = 5

    def oof_within(data, labels, columns, model_kind, repeat):
        probabilities = np.zeros(len(labels))
        splitter = StratifiedKFold(
            10, shuffle=True, random_state=100 + repeat
        )
        for train_index, test_index in splitter.split(data, labels):
            model = model_makers[model_kind]()
            model.fit(data.iloc[train_index][columns], labels[train_index])
            probabilities[test_index] = model.predict_proba(
                data.iloc[test_index][columns]
            )[:, 1]
        return probabilities

    def gain(labels, p0_repeats, p1_repeats, bootstrap_samples=500):
        gains, lower_bounds, upper_bounds = [], [], []
        for repeat in range(repetitions):
            p0, p1 = p0_repeats[repeat], p1_repeats[repeat]
            gains.append(roc_auc_score(labels, p1) - roc_auc_score(labels, p0))
            rng = np.random.default_rng(repeat)
            bootstrap_gains = []
            for _ in range(bootstrap_samples):
                sample = rng.integers(0, len(labels), len(labels))
                if np.unique(labels[sample]).size < 2:
                    continue
                bootstrap_gains.append(
                    roc_auc_score(labels[sample], p1[sample])
                    - roc_auc_score(labels[sample], p0[sample])
                )
            if not bootstrap_gains:
                raise ValueError("Bootstrap samples did not contain both outcome classes.")
            lower, upper = np.percentile(bootstrap_gains, [2.5, 97.5])
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        return (
            np.mean(gains),
            np.mean(lower_bounds),
            np.mean(upper_bounds),
            min(gains),
            max(gains),
        )

    def bootstrap_auc_ci(labels, probabilities, bootstrap_samples=500, seed=1):
        rng = np.random.default_rng(seed)
        scores = []
        for _ in range(bootstrap_samples):
            sample = rng.integers(0, len(labels), len(labels))
            if np.unique(labels[sample]).size == 2:
                scores.append(roc_auc_score(labels[sample], probabilities[sample]))
        if not scores:
            raise ValueError("Bootstrap samples did not contain both outcome classes.")
        return np.percentile(scores, [2.5, 97.5])

    def oof_nested_threshold(data, labels, columns, model_kind):
        outer = StratifiedKFold(10, shuffle=True, random_state=42)
        inner = StratifiedKFold(5, shuffle=True, random_state=1)
        probabilities = np.zeros(len(labels))
        predictions = np.zeros(len(labels), dtype=int)
        for train_index, test_index in outer.split(data, labels):
            train_x, test_x = data.iloc[train_index][columns], data.iloc[test_index][columns]
            inner_probabilities = cross_val_predict(
                model_makers[model_kind](),
                train_x,
                labels[train_index],
                cv=inner,
                method="predict_proba",
            )[:, 1]
            false_positive_rate, true_positive_rate, thresholds = roc_curve(
                labels[train_index], inner_probabilities
            )
            threshold = thresholds[
                np.argmax(true_positive_rate - false_positive_rate)
            ]
            model = model_makers[model_kind]().fit(train_x, labels[train_index])
            probabilities[test_index] = model.predict_proba(test_x)[:, 1]
            predictions[test_index] = (probabilities[test_index] >= threshold).astype(int)
        return probabilities, predictions

    def bootstrap_threshold_metrics(labels, probabilities, predictions, samples=1000):
        rng = np.random.default_rng(1)
        output = {"auc": [], "balanced_accuracy": [], "sensitivity": [], "specificity": []}
        for _ in range(samples):
            sample = rng.integers(0, len(labels), len(labels))
            if np.unique(labels[sample]).size < 2:
                continue
            output["auc"].append(
                roc_auc_score(labels[sample], probabilities[sample])
            )
            output["balanced_accuracy"].append(
                balanced_accuracy_score(labels[sample], predictions[sample])
            )
            tn, fp, fn, tp = confusion_matrix(
                labels[sample], predictions[sample], labels=[0, 1]
            ).ravel()
            output["sensitivity"].append(tp / (tp + fn))
            output["specificity"].append(tn / (tn + fp))
        if not output["auc"]:
            raise ValueError("Bootstrap samples did not contain both outcome classes.")
        return {
            name: np.percentile(values, [2.5, 97.5])
            for name, values in output.items()
        }

    def net_benefit(labels, probabilities, threshold):
        flagged = probabilities >= threshold
        return (
            np.sum(flagged & (labels == 1)) / len(labels)
            - np.sum(flagged & (labels == 0)) / len(labels)
            * threshold
            / (1 - threshold)
        )

    print(f"{len(y)} rows, {int(y.sum())} deaths, prevalence {y.mean():.4f}", flush=True)

    # Part 1: fit models among patients with measured cardiac output.
    co_measured = full[full["co_median"].notna()].reset_index(drop=True)
    y_co = co_measured["y_death7"].to_numpy()
    if np.unique(y_co).size == 2 and np.bincount(y_co.astype(int)).min() >= 10:
        co_sets = {
            "Baseline + labs": base + landmark_labs,
            "+ mPAP": base + landmark_labs + ["mpap_median"],
            "+ CO": base + landmark_labs + ["co_median"],
            "+ both": base + landmark_labs + hemodynamics,
        }
        print(f"\nCO-measured group: {len(co_measured)} rows, {int(y_co.sum())} deaths")
        for model_kind in model_makers:
            probabilities = {
                name: [
                    oof_within(co_measured, y_co, columns, model_kind, repeat)
                    for repeat in range(repetitions)
                ]
                for name, columns in co_sets.items()
            }
            print(f"  {model_kind}")
            for name, values in probabilities.items():
                score = np.mean([roc_auc_score(y_co, p) for p in values])
                print(f"    {name:16s} AUROC {score:.3f}")
            for name in ["+ mPAP", "+ CO", "+ both"]:
                mean, lower, upper, _, _ = gain(
                    y_co,
                    probabilities["Baseline + labs"],
                    probabilities[name],
                )
                print(f"    gain {name:8s}: {mean:+.3f} ({lower:+.3f} to {upper:+.3f})")
    else:
        print("\nCO-measured subgroup skipped: insufficient outcome variation for 10-fold CV.")

    # Part 2: fit a separate primary model within each cohort subgroup.
    groups = {
        "CO measured": full["co_median"].notna().to_numpy(),
        "CO not measured": full["co_median"].isna().to_numpy(),
        "Cardiac surgery ICU": full[cvicu_column].to_numpy() == 1,
        "All other units": full[cvicu_column].to_numpy() != 1,
    }
    print("\nPrimary model fitted within each group")
    for group_name, mask in groups.items():
        data = full.loc[mask].reset_index(drop=True)
        labels = data["y_death7"].to_numpy()
        if np.unique(labels).size < 2 or np.bincount(labels.astype(int)).min() < 10:
            print(f"  {group_name}: skipped; insufficient outcome variation for 10-fold CV")
            continue
        columns = [column for column in primary if data[column].notna().any()]
        for model_kind in model_makers:
            probabilities = [
                oof_within(data, labels, columns, model_kind, repeat)
                for repeat in range(repetitions)
            ]
            auc = np.mean([roc_auc_score(labels, p) for p in probabilities])
            auprc = np.mean([average_precision_score(labels, p) for p in probabilities])
            lower, upper = bootstrap_auc_ci(labels, probabilities[0])
            print(
                f"  {group_name:20s} {model_kind:8s} n={len(labels):5d} "
                f"events={int(labels.sum()):3d} | AUROC {auc:.3f} "
                f"({lower:.3f}-{upper:.3f}) | AUPRC {auprc:.3f} "
                f"(baseline {labels.mean():.3f})"
            )

    # Part 3: compare predictor blocks using five repeated 10-fold runs.
    table_sets = {
        "Age and sex only": ["age", "male"],
        "Baseline": base,
        "Early treatment only": support,
        "Baseline + labs + treatment": base + landmark_labs + support,
    }
    print("\nPredictor-block AUROCs (mean of 5 x 10-fold runs)")
    for name, columns in table_sets.items():
        values = [
            np.mean(
                [
                    roc_auc_score(
                        y,
                        oof_within(full, y, columns, model_kind, repeat),
                    )
                    for repeat in range(repetitions)
                ]
            )
            for model_kind in ("LogReg", "XGBoost")
        ]
        print(f"  {name:32s} LogReg {values[0]:.3f} | XGBoost {values[1]:.3f}")

    # Part 4: choose classification thresholds inside each training fold.
    print("\nNested threshold metrics (10-fold outer, 5-fold inner CV)")
    print(f"Accuracy of always predicting survival: {1 - y.mean():.3f}")
    nested_results = {}
    for model_kind in model_makers:
        probabilities, predictions = oof_nested_threshold(
            full, y, primary, model_kind
        )
        nested_results[model_kind] = probabilities
        intervals = bootstrap_threshold_metrics(y, probabilities, predictions)
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        calibration = LogisticRegression(C=1e6, max_iter=1000).fit(
            np.log(clipped / (1 - clipped)).reshape(-1, 1), y
        )
        print(f"  {model_kind}")
        print(
            f"    AUROC {roc_auc_score(y, probabilities):.3f} "
            f"({intervals['auc'][0]:.3f}-{intervals['auc'][1]:.3f}) | "
            f"AUPRC {average_precision_score(y, probabilities):.3f} "
            f"(baseline {y.mean():.3f}) | Brier {brier_score_loss(y, probabilities):.4f}"
        )
        print(
            f"    Balanced accuracy {balanced_accuracy_score(y, predictions):.3f} "
            f"({intervals['balanced_accuracy'][0]:.3f}-{intervals['balanced_accuracy'][1]:.3f}) | "
            f"sensitivity {tp / (tp + fn):.3f} "
            f"({intervals['sensitivity'][0]:.3f}-{intervals['sensitivity'][1]:.3f}) | "
            f"specificity {tn / (tn + fp):.3f} "
            f"({intervals['specificity'][0]:.3f}-{intervals['specificity'][1]:.3f})"
        )
        print(
            f"    Accuracy {accuracy_score(y, predictions):.3f} | "
            f"flagged {(tp + fp) / len(y) * 100:.1f}% | "
            f"PPV {tp / (tp + fp):.3f} | NPV {tn / (tn + fn):.3f}"
        )
        print(
            f"    Calibration slope {calibration.coef_[0][0]:.2f} | "
            f"intercept {calibration.intercept_[0]:.2f}"
        )

    rng = np.random.default_rng(7)
    differences = []
    for _ in range(1000):
        sample = rng.integers(0, len(y), len(y))
        if np.unique(y[sample]).size < 2:
            continue
        differences.append(
            roc_auc_score(y[sample], nested_results["XGBoost"][sample])
            - roc_auc_score(y[sample], nested_results["LogReg"][sample])
        )
    if not differences:
        raise ValueError("Paired bootstrap samples did not contain both outcome classes.")
    print(
        "\nXGBoost minus LogReg AUROC (paired bootstrap): "
        f"{np.mean(differences):+.3f} "
        f"({np.percentile(differences, 2.5):+.3f} to "
        f"{np.percentile(differences, 97.5):+.3f})"
    )

    print("\nNet benefit by threshold (LogReg | XGBoost | treat all)")
    prevalence = y.mean()
    for threshold in [0.02, 0.05, 0.10, 0.20, 0.30]:
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        print(
            f"  {threshold:5.2f} | "
            f"{net_benefit(y, nested_results['LogReg'], threshold):6.4f} | "
            f"{net_benefit(y, nested_results['XGBoost'], threshold):7.4f} | "
            f"{treat_all:8.4f}"
        )

    print("\nAggregate evaluation complete. No patient-level predictions were written.")


if __name__ == "__main__":
    main()
