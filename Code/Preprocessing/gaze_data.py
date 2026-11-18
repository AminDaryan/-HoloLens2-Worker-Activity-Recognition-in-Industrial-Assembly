"""Read the recordings and cut them into the windows and splits of the paper.

Each recording is cut into consecutive, non-overlapping windows of 240 rows
(8 s at the headset's nominal 30 Hz), starting at its first row; an incomplete
last window is dropped. The windows of each recording are split in time order:
the first 70 % (rounded down) for training, the next 15 % (rounded down) for
validation and the rest for testing. On the 40 recordings this gives 338
windows: 217 for training, 39 for validation and 82 for testing.

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR.parent / "Participants Gaze Dataset"
OUTPUT_DIR = CODE_DIR / "output"

ACTIVITIES = ("Assembly", "Gluing", "Waiting", "Walking")
SPLITS = ("train", "val", "test")
SAMPLES_PER_WINDOW = 240
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

DIRECTION_COLUMNS = ["gazeDirection_x", "gazeDirection_y", "gazeDirection_z"]
# The columns of a recording that the features and the plots use.
GAZE_COLUMNS = ["eyeDataTimestamp", "isCalibrationValid", "gazeHasValue",
                "gazeOrigin_x", "gazeOrigin_y", "gazeOrigin_z", *DIRECTION_COLUMNS]
WINDOW_COLUMNS = ["file", "slot", "recorded_by", "activity", "window", "split",
                  "start_row", "end_row"]


def load_index(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read recordings.csv, which lists every recording with its labels."""
    index = pd.read_csv(Path(data_dir) / "recordings.csv")
    unknown = set(index["activity"]) - set(ACTIVITIES)
    if unknown:
        raise ValueError(f"unknown activities in recordings.csv: {sorted(unknown)}")
    return index


def load_recording(file: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the gaze columns of one recording, with every row as recorded."""
    recording = pd.read_csv(Path(data_dir) / file, usecols=GAZE_COLUMNS)
    for flag in ("isCalibrationValid", "gazeHasValue"):
        if recording[flag].dtype != bool:
            recording[flag] = recording[flag].astype(str).str.strip().str.lower().eq("true")
    return recording


def split_sizes(n_windows: int) -> tuple[int, int, int]:
    """Numbers of training, validation and test windows of one recording."""
    n_train = math.floor(TRAIN_FRACTION * n_windows)
    n_val = math.floor(VALIDATION_FRACTION * n_windows)
    return n_train, n_val, n_windows - n_train - n_val


def make_windows(index: pd.DataFrame,
                 samples_per_window: int = SAMPLES_PER_WINDOW) -> pd.DataFrame:
    """One row per window: its recording, labels, split, and rows in the file
    (start_row is included, end_row is not)."""
    rows = []
    for rec in index.itertuples(index=False):
        n_windows = int(rec.samples) // samples_per_window
        n_train, n_val, _ = split_sizes(n_windows)
        for k in range(n_windows):
            split = "train" if k < n_train else "val" if k < n_train + n_val else "test"
            rows.append((rec.file, rec.slot, rec.recorded_by, rec.activity, k, split,
                         k * samples_per_window, (k + 1) * samples_per_window))
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)



def window_rows(recording: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    """Return the rows of one positional window."""
    return recording.iloc[start_row:end_row].reset_index(drop=True)
