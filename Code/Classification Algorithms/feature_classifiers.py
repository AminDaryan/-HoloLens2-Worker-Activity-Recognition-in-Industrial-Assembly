"""Classify the four activities with the SVM, random-forest and extra-trees
classifiers of GEAR, from the 19 gaze features of each window.

Usage, from the repository root, after Code/Preprocessing/extract_features.py:

    python "Code/Classification Algorithms/feature_classifiers.py"
    python "Code/Classification Algorithms/feature_classifiers.py" --protocol participant

Writes the prediction for every evaluated window to
Code/output/predictions_<protocol>.csv and prints the scores.
Code/Postprocessing/evaluate_predictions.py turns the predictions into result
tables and confusion matrices.

Protocols:

- temporal (default): the time-ordered split of the paper. The models are
  trained on the training windows and predict the validation and test windows.
  Every participant appears in all three parts.
- participant: leave one participant out. The windows of each recorded
  participant (column recorded_by) are held out in turn, and the models are
  trained on the windows of the other nine. The paper does not report this
  protocol.

Adapted from the GEAR2 notebooks AnSVMClassifierForHL2GazeFeatures.ipynb,
RandomForestClassifier.ipynb and ExtraTreesClassifier.ipynb of GEAR
(https://github.com/Interactions-HSG/GEAR, Apache License 2.0), by the
Interaction- and Communication-based Systems group at the University of
St.Gallen. As that notebook states, its SVM set-up is a modified version of the
classifier in
https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02
The settings come from later versions of these notebooks written for this
dataset, not from upstream GEAR (see Code/README.md).

Changes made in 2026: the notebooks split the windows at random and fitted the
feature scaling on all windows. Here the windows are split by one of the
protocols above, and the scaling is fitted on the training windows only.

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

# The code folders are not Python packages, so the shared modules are imported by path.
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE_DIR / "Preprocessing"), str(CODE_DIR / "Postprocessing")]
from evaluate_predictions import print_scores, results_table  # noqa: E402
from extract_features import FEATURES_FILE, WINDOW_KEYS, read_features  # noqa: E402
from gaze_data import OUTPUT_DIR  # noqa: E402
from gear_features import FEATURE_NAMES  # noqa: E402


def make_models() -> dict:
    """The classifiers. As in the notebooks, the SVMs and the random forest get the
    features scaled by their maximum absolute value, and extra trees the unscaled
    features."""
    scaled = {
        "SVM linear": SVC(kernel="linear", C=5000, decision_function_shape="ovo"),
        "SVM polynomial": SVC(kernel="poly", degree=3, C=5000, decision_function_shape="ovo"),
        "SVM RBF": SVC(kernel="rbf", gamma="scale", C=2000, decision_function_shape="ovo"),
        "SVM sigmoid": SVC(kernel="sigmoid", C=1, decision_function_shape="ovo"),
        "Random forest": RandomForestClassifier(n_estimators=100, random_state=96),
    }
    models = {name: make_pipeline(MaxAbsScaler(), model) for name, model in scaled.items()}
    models["Extra trees"] = ExtraTreesClassifier(n_estimators=38, random_state=63)
    return models


def predictions_of(model_name: str, fitted, windows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """A table with the prediction of a fitted model for each of the windows."""
    table = windows[keys].copy()
    table.insert(0, "model", model_name)
    table["predicted"] = fitted.predict(windows[FEATURE_NAMES])
    return table


def predict_temporal(features: pd.DataFrame) -> pd.DataFrame:
    """Train on the training windows; predict the validation and test windows."""
    train = features[features["split"] == "train"]
    evaluated = features[features["split"].isin(["val", "test"])]
    tables = []
    for name, model in make_models().items():
        fitted = clone(model).fit(train[FEATURE_NAMES], train["activity"])
        tables.append(predictions_of(name, fitted, evaluated, WINDOW_KEYS + ["split"]))
    return pd.concat(tables, ignore_index=True)


def predict_participant(features: pd.DataFrame) -> pd.DataFrame:
    """Hold out the windows of each recorded participant in turn, train on the
    others, and predict the held-out windows."""
    tables = []
    for name, model in make_models().items():
        for person in sorted(features["recorded_by"].unique()):
            held_out = features["recorded_by"] == person
            train = features[~held_out]
            fitted = clone(model).fit(train[FEATURE_NAMES], train["activity"])
            tables.append(predictions_of(name, fitted, features[held_out], WINDOW_KEYS))
    return pd.concat(tables, ignore_index=True)


PROTOCOLS = {"temporal": predict_temporal, "participant": predict_participant}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features", type=Path, default=FEATURES_FILE,
                        help="feature table (default: Code/output/features.csv)")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="temporal",
                        help="temporal: the time-ordered split of the paper (default); "
                             "participant: leave one participant out")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help="folder for the predictions (default: Code/output)")
    args = parser.parse_args(argv)
    try:
        features = read_features(args.features)
    except ValueError as problem:
        parser.error(str(problem))

    predictions = PROTOCOLS[args.protocol](features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"predictions_{args.protocol}.csv"
    predictions.to_csv(out, index=False)
    print_scores(results_table(predictions))
    print(f"\nPredictions written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
