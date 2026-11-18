"""The 19 gaze features of GEAR, computed for one window of gaze samples.

Adapted from GEAR2/FeatureCalculation.ipynb of GEAR
(https://github.com/Interactions-HSG/GEAR, Apache License 2.0), by the
Interaction- and Communication-based Systems group at the University of
St.Gallen; supplementary material of K. Bektas et al., ETRA 2023
(doi:10.1145/3588015.3588402) and Computers & Graphics 2024
(doi:10.1016/j.cag.2024.103909). Its fixation detection is adapted from the
Pupil Labs tutorials (https://github.com/pupil-labs/pupil-tutorials).

Changes made in 2026 (Code/README.md explains each one):

- The functions take the samples of one window as arrays instead of row
  dictionaries. The fixation algorithm and its thresholds are unchanged.
- The dispersion of a fixation is that of its own samples; the original
  included the next sample, which had ended the fixation.
- Each run of consecutive samples without a gaze value is one blink, lasting
  from its first sample to the next sample. The original counted every such
  sample as a blink and timed it from a reference point that moved only after
  more than one second.
- The height of the bounding box in the fixation density is the vertical
  extent of the gaze directions; the original used the horizontal extent twice.
- Rates are divided by the window's measured duration, not the nominal one.
- Statistics that are undefined for a window are 0 instead of an error or NaN.
- Samples with a missing or zero-length gaze direction are not used for
  fixations, and the argument of the arc cosine is kept within [-1, 1]. Neither
  guard changes a value on this dataset.

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import scipy.spatial.distance

from gaze_data import DIRECTION_COLUMNS

#: Spatial threshold of the I-DT fixation detection: 1.6 degrees, in radians.
MAX_DISPERSION_RAD = float(np.deg2rad(1.6))
#: Temporal threshold of the I-DT fixation detection, in milliseconds.
MIN_DURATION_MS = 100.0

#: The 19 features, in the order of the original feature files.
FEATURE_NAMES = [
    "meanFix", "minFix", "maxFix", "varFix", "stdFix",
    "meanDis", "minDis", "maxDis", "varDis", "stdDisp",
    "freqDisPerSec",
    "number_of_blinks", "blinkMean", "blinkMin", "blinkMax", "blinkRate",
    "xDir", "yDir",
    "fixDensPerBB",
]


def vector_dispersion(vectors: np.ndarray) -> float:
    """Angular dispersion of gaze vectors (Pupil Labs): the arc cosine of one minus
    the mean of the largest fifth, but at least four, of all pairwise cosine
    distances, in radians."""
    distances = scipy.spatial.distance.pdist(vectors, metric="cosine")
    distances.sort()
    cut_off = int(np.max([distances.shape[0] // 5, 4]))
    # Clipping only guards against rounding just outside [-1, 1].
    return float(np.arccos(np.clip(1.0 - distances[-cut_off:].mean(), -1.0, 1.0)))


def gaze_dispersion(vectors: np.ndarray) -> float:
    """Dispersion of a group of gaze directions; infinite for fewer than two."""
    if len(vectors) < 2:
        return float("inf")
    return vector_dispersion(vectors)


def detect_fixations(timestamps_ms: np.ndarray, directions: np.ndarray,
                     max_dispersion: float = MAX_DISPERSION_RAD,
                     min_duration: float = MIN_DURATION_MS) -> list[dict]:
    """Dispersion-threshold (I-DT) fixation detection on valid gaze samples.

    ``timestamps_ms`` has shape (n,), ``directions`` has shape (n, 3). Returns
    one dictionary per fixation with start, end and duration (ms), the centroid
    of its gaze directions, and its dispersion (radians).
    """
    timestamps_ms = np.asarray(timestamps_ms, dtype=float)
    directions = np.asarray(directions, dtype=float)
    # The dispersion is computed in single precision, as in the original.
    directions32 = directions.astype(np.float32)
    fixations = []
    candidate: deque[int] = deque()
    future = deque(range(len(timestamps_ms)))
    while future:
        # Collect samples until the candidate spans the minimum duration.
        if (len(candidate) < 2
                or timestamps_ms[candidate[-1]] - timestamps_ms[candidate[0]] < min_duration):
            candidate.append(future.popleft())
            continue

        dispersion = gaze_dispersion(directions32[list(candidate)])
        if dispersion > max_dispersion:
            # Not a fixation: move the candidate forward by one sample.
            candidate.popleft()
            continue

        # A minimal fixation was found: extend it while the dispersion stays
        # below the threshold.
        while future:
            candidate.append(future[0])
            extended = gaze_dispersion(directions32[list(candidate)])
            if extended > max_dispersion:
                candidate.pop()
                break
            dispersion = extended
            future.popleft()

        members = list(candidate)
        start, end = timestamps_ms[members[0]], timestamps_ms[members[-1]]
        fixations.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "centroid": tuple(directions[members].mean(axis=0)),
            "dispersion": dispersion,
        })
        candidate.clear()
    return fixations


def fixation_features(fixations: pd.DataFrame, duration_s: float) -> dict:
    """The eleven fixation features: duration and dispersion statistics, and
    the number of fixations per second."""
    features = {}
    for column, prefix, std_name in (("duration", "Fix", "stdFix"),
                                     ("dispersion", "Dis", "stdDisp")):
        values = fixations[column] if len(fixations) else pd.Series(dtype=float)
        features[f"mean{prefix}"] = values.mean() if len(values) else 0.0
        features[f"min{prefix}"] = values.min() if len(values) else 0.0
        features[f"max{prefix}"] = values.max() if len(values) else 0.0
        # Sample variance and standard deviation, as in pandas (ddof=1).
        features[f"var{prefix}"] = values.var() if len(values) > 1 else 0.0
        features[std_name] = values.std() if len(values) > 1 else 0.0
    features["freqDisPerSec"] = len(fixations) / duration_s if duration_s > 0 else 0.0
    return features


def blink_features(timestamps_ms: np.ndarray, gaze_has_value: np.ndarray,
                   duration_s: float) -> dict:
    """The five blink features.

    Every run of consecutive samples without a gaze value counts as one blink.
    This includes short losses of tracking, which the headset reports in the
    same way. A run lasts from its first sample to the next sample; a run at the
    end of the window lasts until one median sample interval after its last
    sample.
    """
    timestamps_ms = np.asarray(timestamps_ms, dtype=float)
    invalid = ~np.asarray(gaze_has_value, dtype=bool)
    durations = []
    n = len(invalid)
    interval = float(np.median(np.diff(timestamps_ms))) if n > 1 else 0.0
    i = 0
    while i < n:
        if not invalid[i]:
            i += 1
            continue
        start = i
        while i < n and invalid[i]:
            i += 1
        end_time = timestamps_ms[i] if i < n else timestamps_ms[i - 1] + interval
        durations.append(end_time - timestamps_ms[start])
    count = len(durations)
    return {
        "number_of_blinks": count,
        "blinkMean": float(np.mean(durations)) if count else 0.0,
        "blinkMin": float(np.min(durations)) if count else 0.0,
        "blinkMax": float(np.max(durations)) if count else 0.0,
        "blinkRate": count / duration_s if duration_s > 0 else 0.0,
    }


def direction_features(fixations: pd.DataFrame) -> dict:
    """Share of successive fixation centroids that move in +x and in +y.

    -1 when there are fewer than two fixations, as in the original.
    """
    if len(fixations) < 2:
        return {"xDir": -1.0, "yDir": -1.0}
    centroids = np.array(fixations["centroid"].tolist(), dtype=float)
    steps = np.diff(centroids, axis=0)
    return {"xDir": float(np.mean(steps[:, 0] > 0)),
            "yDir": float(np.mean(steps[:, 1] > 0))}


def fixation_density(directions: np.ndarray, n_fixations: int) -> dict:
    """Fixations per area of the bounding box of the x and y gaze-direction
    components. -1 when the area is zero or undefined, as in the original."""
    if len(directions) < 2:
        return {"fixDensPerBB": -1.0}
    width = float(np.ptp(directions[:, 0]))
    height = float(np.ptp(directions[:, 1]))
    area = width * height
    if not np.isfinite(area) or area == 0:
        return {"fixDensPerBB": -1.0}
    return {"fixDensPerBB": n_fixations / area}


def window_duration_s(timestamps_ms: np.ndarray) -> float:
    """Measured length of a window in seconds: from its first sample to one median
    sample interval after its last sample (8.0 s for 240 samples at 30 Hz)."""
    timestamps_ms = np.asarray(timestamps_ms, dtype=float)
    if len(timestamps_ms) < 2:
        return 0.0
    interval = float(np.median(np.diff(timestamps_ms)))
    return (timestamps_ms[-1] - timestamps_ms[0] + interval) / 1000.0


def window_features(window: pd.DataFrame) -> dict:
    """All 19 features for one window of gaze samples.

    ``window`` needs the columns eyeDataTimestamp, isCalibrationValid,
    gazeHasValue, and gazeDirection_x/_y/_z, in time order. Fixations, their
    directions, and the density use only valid samples (gaze value present and
    calibration valid), as in the original, with a gaze direction of non-zero
    length.
    """
    timestamps = window["eyeDataTimestamp"].to_numpy(dtype=float)
    duration_s = window_duration_s(timestamps)

    valid = window[(window["gazeHasValue"]) & (window["isCalibrationValid"])]
    valid = valid.dropna(subset=DIRECTION_COLUMNS)
    lengths = np.linalg.norm(valid[DIRECTION_COLUMNS].to_numpy(dtype=float), axis=1)
    valid = valid[np.isfinite(lengths) & (lengths > 0)]
    valid_directions = valid[DIRECTION_COLUMNS].to_numpy(dtype=float)
    fixations = pd.DataFrame(detect_fixations(valid["eyeDataTimestamp"].to_numpy(dtype=float),
                                              valid_directions))

    features = {}
    features.update(fixation_features(fixations, duration_s))
    features.update(blink_features(timestamps, window["gazeHasValue"].to_numpy(), duration_s))
    features.update(direction_features(fixations))
    features.update(fixation_density(valid_directions, len(fixations)))
    return {name: float(features[name]) for name in FEATURE_NAMES}
