"""Checks of Code/Classification Algorithms: which windows the models are trained
on, and the results they give."""

from __future__ import annotations

import copy
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MaxAbsScaler

import evaluate_predictions
import feature_classifiers
from evaluate_predictions import results_table
from feature_classifiers import make_models, predict_participant, predict_temporal
from gaze_data import ACTIVITIES
from gear_features import FEATURE_NAMES


class Spy:
    """Stands in for a classifier and adds the index of its training windows to ``seen``."""

    def __init__(self, seen):
        self.seen = seen

    def fit(self, x, y):
        self.seen.append(x.index)
        self.label = sorted(set(y))[0]
        return self

    def predict(self, x):
        return np.full(len(x), self.label, dtype=object)


@pytest.fixture
def trained_on(monkeypatch):
    """The index of the training windows of every model that is fitted, in order."""
    seen = []
    monkeypatch.setattr(feature_classifiers, "make_models", lambda: {"spy": Spy(seen)})
    monkeypatch.setattr(feature_classifiers, "clone", lambda model: Spy(seen))
    return seen


def test_temporal_models_train_on_the_training_windows_only(features, trained_on):
    predictions = predict_temporal(features)
    assert len(trained_on) == 1
    trained = features.loc[trained_on[0]]
    assert (trained["split"] == "train").all() and len(trained) == 217
    assert predictions["split"].value_counts().to_dict() == {"test": 82, "val": 39}


def test_participant_folds_leave_out_the_held_out_participant(features, trained_on):
    predictions = predict_participant(features)
    people = sorted(features["recorded_by"].unique())
    assert len(people) == len(trained_on) == 10
    for person, index in zip(people, trained_on):
        trained = features.loc[index]
        assert person not in set(trained["recorded_by"])
        assert len(trained) == (features["recorded_by"] != person).sum()
    # Every window is predicted once, when the participant who was recorded is held out.
    assert len(predictions) == 338
    assert predictions.groupby(["file", "window"]).size().max() == 1


# Correctly classified windows of each model, with the settings of the released code.

def correct_per_model(predictions):
    hits = predictions["activity"] == predictions["predicted"]
    return hits.groupby(predictions["model"], sort=False).sum().to_dict()


def test_participant_protocol_scores_all_windows(features_file, tmp_path):
    out_dir = tmp_path / "out"
    assert feature_classifiers.main(["--features", str(features_file), "--protocol", "participant",
                                     "--out-dir", str(out_dir)]) == 0
    predictions = pd.read_csv(out_dir / "predictions_participant.csv")
    assert "split" not in predictions.columns
    assert evaluate_predictions.main(["--out-dir", str(out_dir)]) == 0
    results = pd.read_csv(out_dir / "results_participant.csv").set_index("model")
    assert (results["windows"] == 338).all()
    assert results["accuracy"].between(0, 1).all()
    confusion = pd.read_csv(out_dir / "confusion_participant.csv")
    per_model = confusion.filter(like="predicted_").groupby(confusion["model"], sort=False).sum()
    assert per_model.sum(axis=1).to_dict() == {model: 338 for model in predictions["model"].unique()}


def test_classifier_script_writes_the_predictions(features_file, tmp_path):
    out_dir = tmp_path / "out"
    assert feature_classifiers.main(["--features", str(features_file),
                                     "--out-dir", str(out_dir)]) == 0
    predictions = pd.read_csv(out_dir / "predictions_temporal.csv")
    assert list(predictions.columns) == ["model", "file", "window", "recorded_by", "activity",
                                         "split", "predicted"]
    assert len(predictions) == 6 * (39 + 82)


# --- LSTM ---------------------------------------------------------------------------

def test_lstm_script_explains_itself_without_pytorch(monkeypatch, capsys):
    script = str(Path(feature_classifiers.__file__).with_name("lstm_classifier.py"))
    monkeypatch.setitem(sys.modules, "torch", None)  # makes "import torch" fail
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(script, run_name="__main__")
    assert stopped.value.code == 0 and "--runs" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [script])
    with pytest.raises(SystemExit, match="PyTorch is not installed"):
        runpy.run_path(script, run_name="__main__")


@pytest.fixture
def lstm(monkeypatch):
    pytest.importorskip("torch")
    import lstm_classifier
    monkeypatch.setattr(lstm_classifier, "MAX_EPOCHS", 5)
    return lstm_classifier


def test_lstm_restores_its_best_epoch_and_stops_after_the_patience(features, lstm, monkeypatch):
    data = lstm.to_tensors(features)
    x_val, y_val = data["val"]
    truth = y_val.numpy()
    accuracies = iter([0.2, 0.9, 0.3, 0.9, 0.3, 0.3, 0.3, 0.3])
    states = []

    def scripted_predict(model, x):
        # Records the weights after every epoch and returns predictions with a set accuracy.
        states.append(copy.deepcopy(model.state_dict()))
        right = np.arange(len(truth)) < round(next(accuracies) * len(truth))
        return np.where(right, truth, (truth + 1) % len(ACTIVITIES))

    monkeypatch.setattr(lstm, "predict", scripted_predict)
    monkeypatch.setattr(lstm, "MAX_EPOCHS", 8)
    monkeypatch.setattr(lstm, "PATIENCE", 3)
    model, best_accuracy, best_epoch = lstm.train_one_run(0, *data["train"], x_val, y_val)
    assert best_epoch == 2                      # epoch 4 is only as good, not better
    assert best_accuracy == pytest.approx(round(0.9 * len(truth)) / len(truth))
    assert len(states) == 5                     # three epochs without improvement, then stop
    restored = model.state_dict()
    assert all(lstm.torch.equal(restored[k], states[1][k]) for k in restored)
    assert not all(lstm.torch.equal(restored[k], states[-1][k]) for k in restored)


def test_lstm_trains_with_dropout_and_predicts_without(features, lstm, monkeypatch):
    monkeypatch.setattr(lstm, "MAX_EPOCHS", 2)
    calls = []
    real_forward = lstm.LSTMClassifier.forward

    def forward(self, x):
        calls.append((lstm.torch.is_grad_enabled(), self.training))
        return real_forward(self, x)

    monkeypatch.setattr(lstm.LSTMClassifier, "forward", forward)
    data = lstm.to_tensors(features)
    lstm.train_one_run(0, *data["train"], *data["val"])
    assert {training for grad, training in calls if grad} == {True}       # training steps
    assert {training for grad, training in calls if not grad} == {False}  # validation


def test_lstm_run_repeats_with_the_same_seed(features, lstm):
    data = lstm.to_tensors(features)
    first, _, _ = lstm.train_one_run(0, *data["train"], *data["val"])
    again, _, _ = lstm.train_one_run(0, *data["train"], *data["val"])
    assert all(lstm.torch.equal(t, again.state_dict()[k]) for k, t in first.state_dict().items())


def test_lstm_runs_use_the_temporal_split_and_their_own_seeds(features, features_file, lstm,
                                                              tmp_path, monkeypatch):
    seen = []
    real = lstm.train_one_run

    def spy(seed, x_train, y_train, x_val, y_val):
        seen.append((seed, x_train.numpy(), x_val.numpy()))
        return real(seed, x_train, y_train, x_val, y_val)

    monkeypatch.setattr(lstm, "train_one_run", spy)
    assert lstm.main(["--features", str(features_file), "--runs", "2",
                      "--out-dir", str(tmp_path)]) == 0
    train = features.loc[features["split"] == "train", FEATURE_NAMES]
    val = features.loc[features["split"] == "val", FEATURE_NAMES]
    scaler = MaxAbsScaler().fit(train)
    assert [seed for seed, _, _ in seen] == [0, 1]
    for _, x_train, x_val in seen:
        np.testing.assert_allclose(x_train, scaler.transform(train), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(x_val, scaler.transform(val), rtol=1e-6, atol=1e-7)
    predictions = pd.read_csv(tmp_path / "predictions_lstm.csv")
    assert predictions["run"].value_counts().to_dict() == {0: 82, 1: 82}
    assert (predictions["split"] == "test").all()
    assert set(predictions["predicted"]) <= set(ACTIVITIES)
    assert len(pd.read_csv(tmp_path / "lstm_runs.csv")) == 2


def test_lstm_beats_chance(features_file, lstm, tmp_path, monkeypatch):
    monkeypatch.setattr(lstm, "MAX_EPOCHS", 60)
    assert lstm.main(["--features", str(features_file), "--runs", "1",
                      "--out-dir", str(tmp_path)]) == 0
    results = results_table(pd.read_csv(tmp_path / "predictions_lstm.csv"))
    assert results["accuracy"].item() > 0.5
