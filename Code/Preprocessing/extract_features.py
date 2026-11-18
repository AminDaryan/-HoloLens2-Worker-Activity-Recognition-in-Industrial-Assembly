"""Compute the 19 GEAR gaze features of every window of the dataset.

Usage, from the repository root:

    python Code/Preprocessing/extract_features.py

Writes Code/output/features.csv with one row per window: its recording, labels,
split and rows in the file, the number of samples, valid samples and seconds it
holds, and the 19 features. The classifiers in Code/Classification Algorithms
read this table with read_features.

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gaze_data import (ACTIVITIES, DATA_DIR, OUTPUT_DIR, SAMPLES_PER_WINDOW, SPLITS,
                       load_index, load_recording, make_windows, window_rows)
from gear_features import FEATURE_NAMES, window_duration_s, window_features

FEATURES_FILE = OUTPUT_DIR / "features.csv"
SIZE_COLUMNS = ["n_samples", "n_valid", "duration_s"]
# Columns that identify a window and its activity; the classifiers copy them to their predictions.
WINDOW_KEYS = ["file", "window", "recorded_by", "activity"]


def extract_features(data_dir: Path = DATA_DIR, samples_per_window: int = SAMPLES_PER_WINDOW,
                     verbose: bool = True) -> pd.DataFrame:
    """Return one row per window with its size and its 19 features."""
    windows = make_windows(load_index(data_dir), samples_per_window)
    rows = []
    for file, file_windows in windows.groupby("file", sort=False):
        started = time.perf_counter()
        recording = load_recording(file, data_dir)
        for window in file_windows.itertuples(index=False):
            samples = window_rows(recording, window.start_row, window.end_row)
            rows.append({
                **window._asdict(),
                "n_samples": len(samples),
                "n_valid": int((samples["gazeHasValue"] & samples["isCalibrationValid"]).sum()),
                "duration_s": window_duration_s(samples["eyeDataTimestamp"].to_numpy()),
                **window_features(samples),
            })
        if verbose:
            print(f"{file}: {len(file_windows)} windows ({time.perf_counter() - started:.1f} s)")
    return pd.DataFrame(rows, columns=list(windows.columns) + SIZE_COLUMNS + FEATURE_NAMES)


def read_features(path: Path = FEATURES_FILE) -> pd.DataFrame:
    """Read a table written by this script and check that a classifier can use it.

    Raises ValueError, with the reason, if the table cannot be used.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} not found; run Code/Preprocessing/extract_features.py first")
    features = pd.read_csv(path)
    missing = [c for c in [*WINDOW_KEYS, "split", *FEATURE_NAMES] if c not in features.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    train = features["split"] == "train"
    checks = [
        (np.isfinite(features[FEATURE_NAMES].to_numpy(dtype=float)).all(),
         "contains missing or infinite feature values"),
        (features["activity"].isin(ACTIVITIES).all(), "contains unknown activities"),
        (features["split"].isin(SPLITS).all(), "contains unknown splits"),
        (features["recorded_by"].notna().all(), "contains windows without recorded_by"),
        (train.any() and (features["split"] == "test").any(), "needs training and test windows"),
        (features.loc[train, "activity"].nunique() >= 2,
         "needs at least two activities among the training windows"),
    ]
    for passed, problem in checks:
        if not passed:
            raise ValueError(f"{path} {problem}")
    return features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="folder with recordings.csv and P01 to P10 "
                             "(default: Participants Gaze Dataset)")
    parser.add_argument("--out", type=Path, default=FEATURES_FILE,
                        help="output file (default: Code/output/features.csv)")
    parser.add_argument("--samples-per-window", type=int, default=SAMPLES_PER_WINDOW,
                        help="rows per window (default: 240, 8 s at 30 Hz)")
    args = parser.parse_args(argv)
    if args.samples_per_window < 2:
        parser.error("--samples-per-window must be at least 2")
    if not (args.data_dir / "recordings.csv").is_file():
        parser.error(f"{args.data_dir} has no recordings.csv")

    features = extract_features(args.data_dir, args.samples_per_window)
    if features.empty:
        parser.error(f"no recording has a complete window of {args.samples_per_window} rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    counts = features["split"].value_counts()
    print(f"\n{len(features)} windows ({counts.get('train', 0)} training, "
          f"{counts.get('val', 0)} validation, {counts.get('test', 0)} test) "
          f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
