"""Score the predictions of the classifiers and draw their confusion matrices.

Usage, from the repository root, after a script in Code/Classification Algorithms:

    python Code/Postprocessing/evaluate_predictions.py

Reads every Code/output/predictions_<name>.csv and writes next to it:

- results_<name>.csv: accuracy, macro-averaged F1 and the recall of each
  activity, for every model and, where the file has them, every split and run;
- confusion_<name>.csv and confusion_<name>.png: the confusion matrix of every
  model on the test windows, summed over runs (for the participant protocol,
  over all windows, each predicted when its participant was held out).

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

# The code folders are not Python packages, so the shared modules are imported by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Preprocessing"))
from gaze_data import ACTIVITIES, OUTPUT_DIR  # noqa: E402

GROUP_COLUMNS = ("model", "split", "run")
PREDICTION_COLUMNS = ["model", "activity", "predicted"]
TITLES = {
    "temporal": "Temporal holdout, test windows",
    "participant": "Leave one participant out, all windows",
    "lstm": "LSTM, test windows, summed over runs",
}


def scores(true, predicted) -> dict:
    """Number of windows, accuracy, macro-averaged F1 and the recall of each activity."""
    labels = list(ACTIVITIES)
    recalls = recall_score(true, predicted, labels=labels, average=None, zero_division=0)
    return {
        "windows": len(true),
        "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, labels=labels, average="macro", zero_division=0),
        **{f"recall_{activity}": recall for activity, recall in zip(ACTIVITIES, recalls)},
    }


def results_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores for every model and every split and run in the predictions."""
    by = [column for column in GROUP_COLUMNS if column in predictions.columns]
    rows = []
    for key, group in predictions.groupby(by, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(by, key)), **scores(group["activity"], group["predicted"])})
    return pd.DataFrame(rows)


def confusion_matrices(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Confusion matrix of every model on the test windows, summed over runs.

    Rows are the true activities and columns the predictions, both in the order
    of ACTIVITIES. Predictions without a split column are all used.
    """
    if "split" in predictions.columns:
        predictions = predictions[predictions["split"] == "test"]
    return {model: confusion_matrix(group["activity"], group["predicted"], labels=list(ACTIVITIES))
            for model, group in predictions.groupby("model", sort=False)}


def confusion_table(matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    """The confusion matrices as one table with a row per model and true activity."""
    rows = []
    for model, matrix in matrices.items():
        for activity, counts in zip(ACTIVITIES, matrix):
            rows.append({"model": model, "true_activity": activity,
                         **{f"predicted_{a}": int(n) for a, n in zip(ACTIVITIES, counts)}})
    return pd.DataFrame(rows)


def plot_confusion_matrices(matrices: dict[str, np.ndarray], title: str, out: Path) -> Path:
    """Draw the confusion matrices side by side, with one shared color scale."""
    n_cols = min(len(matrices), 3)
    n_rows = math.ceil(len(matrices) / n_cols)
    fig = Figure(figsize=(3.3 * n_cols, 3.3 * n_rows + 0.4))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    top = max(int(matrix.max()) for matrix in matrices.values()) or 1
    ticks = range(len(ACTIVITIES))
    for ax, (model, matrix) in zip(axes.flat, matrices.items()):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=top)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > top / 2 else "#1f1f1f")
        ax.set_xticks(ticks, ACTIVITIES, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(ticks, ACTIVITIES, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(model, fontsize=10)
        # A 2-pixel white gap between the cells instead of a frame.
        ax.set_xticks(np.arange(len(ACTIVITIES) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ACTIVITIES) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(matrices):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def print_scores(results: pd.DataFrame) -> None:
    """Print the results, and their mean and standard deviation when there are several runs."""
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.round(3).to_string(index=False))
        if "run" in results.columns and results["run"].nunique() > 1:
            by = [column for column in ("model", "split") if column in results.columns]
            summary = results.groupby(by, sort=False)[["accuracy", "macro_f1"]].agg(["mean", "std"])
            print(f"\nMean and standard deviation over {results['run'].nunique()} runs:")
            print(summary.round(3).to_string())


def evaluate_file(path: Path) -> pd.DataFrame:
    """Write the result table and the confusion matrices of one predictions file."""
    name = path.stem.removeprefix("predictions_")
    predictions = pd.read_csv(path)
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    results = results_table(predictions)
    matrices = confusion_matrices(predictions)
    if not matrices:
        raise ValueError(f"{path} has no test windows to evaluate")
    results.to_csv(path.with_name(f"results_{name}.csv"), index=False)
    confusion_table(matrices).to_csv(path.with_name(f"confusion_{name}.csv"), index=False)
    plot_confusion_matrices(matrices, TITLES.get(name, name), path.with_name(f"confusion_{name}.png"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help="folder with the predictions_<name>.csv files, which also gets "
                             "the results (default: Code/output)")
    args = parser.parse_args(argv)
    files = sorted(args.out_dir.glob("predictions_*.csv"))
    if not files:
        parser.error(f"no predictions_*.csv in {args.out_dir}; "
                     "run a script in Code/Classification Algorithms first")
    for path in files:
        try:
            results = evaluate_file(path)
        except ValueError as problem:
            parser.error(str(problem))
        print(f"\n{path.name}:")
        print_scores(results)
    print(f"\nResult tables and confusion matrices written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
