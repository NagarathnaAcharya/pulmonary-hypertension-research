"""Build 24-hour landmark laboratory features from authorized MIMIC-IV files."""

import argparse
from pathlib import Path

import pandas as pd


LABS = {
    "creatinine": "Creatinine",
    "bun": "Urea Nitrogen",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "chloride": "Chloride",
    "bicarbonate": "Bicarbonate",
    "anion_gap": "Anion Gap",
    "glucose": "Glucose",
    "hemoglobin": "Hemoglobin",
    "hematocrit": "Hematocrit",
    "wbc": "White Blood Cells",
    "platelets": "Platelet Count",
    "rdw": "RDW",
    "inr": "INR(PT)",
    "lactate": "Lactate",
    "ph": "pH",
    "pco2": "pCO2",
    "po2": "pO2",
    "bilirubin": "Bilirubin, Total",
    "alt": "Alanine Aminotransferase (ALT)",
    "ast": "Asparate Aminotransferase (AST)",
    "calcium": "Calcium, Total",
    "magnesium": "Magnesium",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add last-observed blood laboratory values from the first 24 hours "
            "after each mPAP index time."
        )
    )
    parser.add_argument("--landmark-table", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--lab-items", type=Path, required=True)
    parser.add_argument("--lab-events", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private patient-level output CSV; must be outside this repository.",
    )
    return parser.parse_args()


def require_columns(frame, columns, file_label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    if output == repo_root or repo_root in output.parents:
        raise ValueError("The patient-level output must be outside the repository.")

    input_files = {
        "landmark table": args.landmark_table.expanduser(),
        "features": args.features.expanduser(),
        "lab item dictionary": args.lab_items.expanduser(),
        "lab events": args.lab_events.expanduser(),
    }
    for label, path in input_files.items():
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    landmark = pd.read_csv(input_files["landmark table"])
    require_columns(landmark, ["subject_id"], "Landmark table")
    if landmark["subject_id"].duplicated().any():
        raise ValueError("Landmark table must contain one row per subject_id.")

    features = pd.read_csv(input_files["features"], parse_dates=["index_time"])
    require_columns(features, ["subject_id", "index_time"], "Features table")
    if features["subject_id"].duplicated().any():
        raise ValueError("Features table must contain one row per subject_id.")
    if features["index_time"].isna().any():
        raise ValueError("Features table contains missing index_time values.")

    subject_ids = set(landmark["subject_id"])
    index = features[features["subject_id"].isin(subject_ids)]
    index_time = index.set_index("subject_id")["index_time"]
    if len(index_time) != len(landmark):
        raise ValueError(
            "Could not match every landmark subject to exactly one feature index time."
        )

    item_dictionary = pd.read_csv(input_files["lab item dictionary"])
    require_columns(item_dictionary, ["itemid", "label", "fluid"], "Lab item dictionary")
    item_to_lab = {}
    for feature_name, label in LABS.items():
        matches = item_dictionary[
            (item_dictionary["label"] == label)
            & (item_dictionary["fluid"] == "Blood")
        ]
        for item_id in matches["itemid"]:
            item_to_lab[item_id] = feature_name
    if not item_to_lab:
        raise ValueError("No laboratory item IDs matched; check the authorized inputs.")

    required_event_columns = ["subject_id", "itemid", "charttime", "valuenum"]
    event_columns = pd.read_csv(input_files["lab events"], nrows=0).columns
    require_columns(
        pd.DataFrame(columns=event_columns), required_event_columns, "Lab events"
    )
    kept_chunks = []
    kept_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_files["lab events"],
            usecols=required_event_columns,
            chunksize=5_000_000,
        ),
        start=1,
    ):
        selected = chunk[
            chunk["itemid"].isin(item_to_lab)
            & chunk["valuenum"].notna()
            & chunk["subject_id"].isin(index_time.index)
        ].copy()
        selected["charttime"] = pd.to_datetime(selected["charttime"])
        selected["index_time"] = selected["subject_id"].map(index_time)
        selected = selected[
            (selected["charttime"] >= selected["index_time"])
            & (
                selected["charttime"]
                < selected["index_time"] + pd.Timedelta(hours=24)
            )
        ]
        kept_chunks.append(selected)
        kept_rows += len(selected)
        print(f"Processed chunk {chunk_number}; matching records retained: {kept_rows:,}")

    if not kept_chunks or kept_rows == 0:
        raise ValueError("No valid laboratory records found in the 24-hour window.")

    labs = pd.concat(kept_chunks, ignore_index=True)
    labs["lab"] = labs["itemid"].map(item_to_lab)
    hours_after_index = (
        (labs["charttime"] - labs["index_time"]).dt.total_seconds() / 3600
    )
    if (hours_after_index < 0).any() or (hours_after_index >= 24).any():
        raise ValueError("Laboratory window validation failed.")

    labs = labs.sort_values("charttime")
    last_values = labs.groupby(["subject_id", "lab"])["valuenum"].last().unstack()
    last_values.columns = [f"lm_lab_{name}" for name in last_values.columns]
    output_table = landmark.merge(
        last_values, left_on="subject_id", right_index=True, how="left"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(output, index=False)
    print(f"Created private landmark feature table with {len(output_table):,} rows.")
    print("Laboratory coverage (% of landmark patients):")
    print((output_table.filter(regex=r"^lm_lab_").notna().mean() * 100).round(1))
    print(f"Private output written to: {output}")


if __name__ == "__main__":
    main()
