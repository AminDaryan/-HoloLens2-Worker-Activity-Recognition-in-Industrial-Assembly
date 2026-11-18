"""Checks of Code/Preprocessing: the dataset, the windows and the features."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

import extract_features as extraction
from extract_features import read_features
from gaze_data import (ACTIVITIES, DATA_DIR, load_index, load_recording, make_windows,
                       split_sizes, window_rows)
from gear_features import (FEATURE_NAMES, MAX_DISPERSION_RAD, blink_features, detect_fixations,
                           direction_features, fixation_density, window_duration_s,
                           window_features)

INTERVAL_MS = 1000 / 30


def unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def as_tuples(rows):
    return set(rows.astype(object).where(rows.notna(), None).itertuples(index=False, name=None))


# --- the dataset ---------------------------------------------------------------

def test_dataset_folder_holds_the_ten_participants():
    assert DATA_DIR.name == "Participants Gaze Dataset"
    assert sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) == [
        f"P{i:02d}" for i in range(1, 11)]


def test_index_matches_the_files():
    index = load_index()
    assert len(index) == 40
    assert set(index["activity"]) == set(ACTIVITIES)
    for rec in index.itertuples():
        assert len(load_recording(rec.file)) == rec.samples


# --- windows and splits ----------------------------------------------------------

def test_split_sizes():
    assert split_sizes(10) == (7, 1, 2)
    assert split_sizes(23) == (16, 3, 4)
    assert split_sizes(1) == (0, 0, 1)


def test_windows_match_the_paper():
    windows = make_windows(load_index())
    assert len(windows) == 338
    assert windows["split"].value_counts().to_dict() == {"train": 217, "test": 82, "val": 39}
    test = windows[windows["split"] == "test"]
    assert test["activity"].value_counts().to_dict() == {
        "Waiting": 21, "Assembly": 21, "Walking": 20, "Gluing": 20}


def test_a_recording_of_exactly_two_windows_gives_two_windows():
    index = pd.DataFrame({"file": ["x.csv"], "slot": ["P01"], "recorded_by": ["P01"],
                          "activity": ["Assembly"], "samples": [480]})
    assert make_windows(index)[["start_row", "end_row"]].values.tolist() == [[0, 240], [240, 480]]
    assert make_windows(index.assign(samples=479))[["start_row", "end_row"]].values.tolist() == [
        [0, 240]]


def test_windows_are_adjacent_and_splits_in_time_order():
    windows = make_windows(load_index())
    order = {"train": 0, "val": 1, "test": 2}
    for _, rec in windows.groupby("file"):
        rec = rec.sort_values("start_row")
        assert (rec["start_row"].to_numpy()[1:] == rec["end_row"].to_numpy()[:-1]).all()
        assert (np.diff(rec["split"].map(order).to_numpy()) >= 0).all()


# --- features on synthetic data ----------------------------------------------------

def test_two_fixations_are_found_and_stay_within_the_threshold():
    t = np.arange(60) * INTERVAL_MS
    directions = unit([[0.0, 0.0, 1.0]] * 30 + [[0.3, 0.0, 1.0]] * 30)
    fixations = detect_fixations(t, directions)
    assert len(fixations) == 2
    for fixation in fixations:
        assert fixation["dispersion"] <= MAX_DISPERSION_RAD
        assert fixation["duration"] == pytest.approx(29 * INTERVAL_MS)


def test_no_fixation_while_the_gaze_keeps_moving():
    t = np.arange(60) * INTERVAL_MS
    angles = np.deg2rad(np.arange(60) * 1.0)  # 30 degrees per second
    directions = np.column_stack([np.sin(angles), np.zeros(60), np.cos(angles)])
    assert detect_fixations(t, directions) == []


def test_each_run_of_invalid_samples_is_one_blink():
    t = np.arange(10) * 30.0
    has_value = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
    features = blink_features(t, has_value, duration_s=0.3)
    assert features["number_of_blinks"] == 3
    assert features["blinkMin"] == pytest.approx(30.0)    # one sample
    assert features["blinkMax"] == pytest.approx(60.0)    # two samples
    assert features["blinkMean"] == pytest.approx(50.0)
    assert features["blinkRate"] == pytest.approx(10.0)


def test_direction_features():
    fixations = pd.DataFrame({"centroid": [(0.0, 0.0, 1.0), (0.1, -0.1, 1.0), (0.2, 0.1, 1.0)]})
    assert direction_features(fixations) == {"xDir": 1.0, "yDir": 0.5}
    assert direction_features(fixations.iloc[:1]) == {"xDir": -1.0, "yDir": -1.0}


def test_density_uses_width_and_height():
    directions = np.array([[0.0, 0.0, 1.0], [0.2, 0.1, 1.0]])
    assert fixation_density(directions, 4)["fixDensPerBB"] == pytest.approx(4 / (0.2 * 0.1))
    assert fixation_density(directions[:1], 4)["fixDensPerBB"] == -1.0


def test_window_duration():
    assert window_duration_s(np.arange(240) * INTERVAL_MS) == pytest.approx(8.0)


def test_window_without_valid_samples():
    window = pd.DataFrame({
        "eyeDataTimestamp": np.arange(20) * INTERVAL_MS,
        "isCalibrationValid": True,
        "gazeHasValue": False,
        "gazeDirection_x": np.nan, "gazeDirection_y": np.nan, "gazeDirection_z": np.nan,
    })
    features = window_features(window)
    assert features["number_of_blinks"] == 1
    assert features["freqDisPerSec"] == 0.0
    assert all(np.isfinite(list(features.values())))


def test_rates_use_the_measured_window_duration():
    t = np.arange(60) * INTERVAL_MS                      # 60 samples = 2.0 s
    directions = unit([[0.0, 0.0, 1.0]] * 30 + [[0.3, 0.0, 1.0]] * 30)
    has_value = np.ones(60, dtype=bool)
    has_value[[10, 40]] = False                          # two one-sample blinks
    window = pd.DataFrame({
        "eyeDataTimestamp": t, "isCalibrationValid": True, "gazeHasValue": has_value,
        "gazeDirection_x": directions[:, 0], "gazeDirection_y": directions[:, 1],
        "gazeDirection_z": directions[:, 2],
    })
    window.loc[~has_value, ["gazeDirection_x", "gazeDirection_y", "gazeDirection_z"]] = np.nan
    features = window_features(window)
    assert features["freqDisPerSec"] == pytest.approx(2 / 2.0)
    assert features["number_of_blinks"] == 2
    assert features["blinkRate"] == pytest.approx(2 / 2.0)


# --- features of the dataset ---------------------------------------------------------

def test_features_of_all_windows_are_finite(features):
    assert len(features) == 338
    assert list(features.columns[-len(FEATURE_NAMES):]) == FEATURE_NAMES
    assert np.isfinite(features[FEATURE_NAMES].to_numpy()).all()
    assert features["maxDis"].max() <= MAX_DISPERSION_RAD + 1e-9


def test_features_come_from_the_window_rows(features):
    file = "P03/P03_Gluing.csv"
    recording = load_recording(file)
    for w in features[features["file"] == file].itertuples():
        expected = window_features(window_rows(recording, w.start_row, w.end_row))
        assert [getattr(w, n) for n in FEATURE_NAMES] == pytest.approx(
            [expected[n] for n in FEATURE_NAMES])
    assert (features["n_samples"] == 240).all()


@pytest.mark.parametrize("file, window, expected", [
    ("P03/P03_Gluing.csv", 0, {'meanFix': 200.0, 'minFix': 100.0, 'maxFix': 534.0, 'varFix': 15552.129032258064, 'stdFix': 124.708175482837, 'meanDis': 0.0232049344801146, 'minDis': 0.0032565097479205, 'maxDis': 0.0277423732129586, 'varDis': 2.5421438620210583e-05, 'stdDisp': 0.00504196773296, 'freqDisPerSec': 4.0, 'number_of_blinks': 9.0, 'blinkMean': 40.55555555555556, 'blinkMin': 33.0, 'blinkMax': 66.0, 'blinkRate': 1.125, 'xDir': 0.4516129032258064, 'yDir': 0.5161290322580645, 'fixDensPerBB': 1922.464830608622}),  # noqa: E501
    ("P09/P09_Waiting.csv", 0, {'meanFix': 133.5, 'minFix': 100.0, 'maxFix': 200.0, 'varFix': 2222.333333333333, 'stdFix': 47.14163057567412, 'meanDis': 0.0199604039631945, 'minDis': 0.0073541914737224, 'maxDis': 0.0247865511922631, 'varDis': 7.082688048111646e-05, 'stdDisp': 0.0084158707500244, 'freqDisPerSec': 0.9376465072667604, 'number_of_blinks': 8.0, 'blinkMean': 137.5, 'blinkMin': 33.0, 'blinkMax': 667.0, 'blinkRate': 1.875293014533521, 'xDir': 0.6666666666666666, 'yDir': 0.3333333333333333, 'fixDensPerBB': 4.327784118234566}),  # noqa: E501
])
def test_feature_values_stay_the_same(features, file, window, expected):
    row = features.set_index(["file", "window"]).loc[(file, window)]
    assert [row[n] for n in FEATURE_NAMES] == pytest.approx([expected[n] for n in FEATURE_NAMES],
                                                             rel=1e-9)


def test_extraction_script_uses_its_options_and_keeps_full_precision(tmp_path):
    subset = load_index().query("slot == 'P03'")
    data_dir = tmp_path / "data"
    (data_dir / "P03").mkdir(parents=True)
    for file in subset["file"]:
        shutil.copy(DATA_DIR / file, data_dir / file)
    subset.to_csv(data_dir / "recordings.csv", index=False)
    out = tmp_path / "features.csv"
    assert extraction.main(["--data-dir", str(data_dir), "--out", str(out),
                            "--samples-per-window", "120"]) == 0
    written = read_features(out)
    assert len(written) == (subset["samples"] // 120).sum()
    assert ((written["end_row"] - written["start_row"]) == 120).all()
    expected = extraction.extract_features(data_dir, 120, verbose=False)
    np.testing.assert_allclose(written[FEATURE_NAMES].to_numpy(float),
                               expected[FEATURE_NAMES].to_numpy(float), rtol=1e-12)


# --- reading the feature table back ------------------------------------------------

def test_feature_table_is_read_unchanged(features, features_file):
    loaded = read_features(features_file)
    np.testing.assert_allclose(loaded[FEATURE_NAMES].to_numpy(float),
                               features[FEATURE_NAMES].to_numpy(float), rtol=1e-12)


@pytest.mark.parametrize("column, value, problem", [
    ("meanFix", np.nan, "missing or infinite"),
    ("meanFix", np.inf, "missing or infinite"),
    ("activity", "Reading", "unknown activities"),
    ("split", "holdout", "unknown splits"),
    ("recorded_by", np.nan, "without recorded_by"),
])
def test_a_broken_feature_table_is_refused(features, tmp_path, column, value, problem):
    broken = features.copy()
    broken[column] = broken[column].astype(object)
    broken.loc[broken.index[0], column] = value
    broken.to_csv(tmp_path / "broken.csv", index=False)
    with pytest.raises(ValueError, match=problem):
        read_features(tmp_path / "broken.csv")


def test_a_missing_or_incomplete_feature_table_is_refused(features, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_features(tmp_path / "missing.csv")
    features.drop(columns=["window"]).to_csv(tmp_path / "no_window.csv", index=False)
    with pytest.raises(ValueError, match="lacks the columns"):
        read_features(tmp_path / "no_window.csv")
    features[features["split"] != "test"].to_csv(tmp_path / "no_test.csv", index=False)
    with pytest.raises(ValueError, match="training and test"):
        read_features(tmp_path / "no_test.csv")
    one = features[(features["split"] != "train") | (features["activity"] == "Assembly")]
    one.to_csv(tmp_path / "one_activity.csv", index=False)
    with pytest.raises(ValueError, match="two activities"):
        read_features(tmp_path / "one_activity.csv")
