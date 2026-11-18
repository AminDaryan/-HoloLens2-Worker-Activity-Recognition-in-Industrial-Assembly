"""Checks of Code/Postprocessing: scores, result tables and confusion matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import evaluate_predictions
from evaluate_predictions import confusion_matrices, confusion_table, results_table, scores
from gaze_data import ACTIVITIES


def make_predictions(rows, **columns):
    """A predictions table from (activity, predicted) pairs and constant columns."""
    table = pd.DataFrame(rows, columns=["activity", "predicted"])
    for name, value in columns.items():
        table.insert(0, name, value)
    return table


def test_scores():
    result = scores(["Assembly", "Assembly", "Gluing", "Waiting"],
                    ["Assembly", "Gluing", "Gluing", "Walking"])
    assert result["windows"] == 4
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["recall_Assembly"] == pytest.approx(0.5)
    assert result["recall_Gluing"] == pytest.approx(1.0)
    assert result["recall_Waiting"] == 0.0
    # F1 per activity: Assembly 2/3, Gluing 2/3, Waiting 0, Walking 0.
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 4)


def test_results_are_grouped_by_model_split_and_run():
    a = make_predictions([("Assembly", "Assembly"), ("Gluing", "Assembly")], split="test", run=0, model="A")
    b = make_predictions([("Assembly", "Assembly"), ("Gluing", "Gluing")], split="test", run=1, model="A")
    results = results_table(pd.concat([a, b], ignore_index=True))
    assert list(results.columns[:3]) == ["model", "split", "run"]
    assert results["accuracy"].tolist() == [0.5, 1.0]
    pooled = results_table(pd.concat([a, b], ignore_index=True).drop(columns=["split", "run"]))
    assert len(pooled) == 1 and pooled["accuracy"].item() == pytest.approx(0.75)


def test_confusion_matrices_use_the_test_windows_and_sum_over_runs():
    val = make_predictions([("Walking", "Gluing")], split="val", run=0, model="A")
    run0 = make_predictions([("Assembly", "Gluing"), ("Waiting", "Waiting")], split="test", run=0, model="A")
    run1 = make_predictions([("Assembly", "Gluing")], split="test", run=1, model="A")
    matrix = confusion_matrices(pd.concat([val, run0, run1], ignore_index=True))["A"]
    assert matrix.sum() == 3
    assert matrix[ACTIVITIES.index("Assembly"), ACTIVITIES.index("Gluing")] == 2
    assert matrix[ACTIVITIES.index("Waiting"), ACTIVITIES.index("Waiting")] == 1
    table = confusion_table({"A": matrix})
    assert list(table.columns) == ["model", "true_activity"] + [f"predicted_{a}" for a in ACTIVITIES]
    assert table.set_index("true_activity").loc["Assembly", "predicted_Gluing"] == 2


def test_confusion_figure_shows_the_matrix_and_the_protocol(tmp_path, monkeypatch):
    figures = []

    class RecordedFigure(Figure):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            figures.append(self)

    monkeypatch.setattr(evaluate_predictions, "Figure", RecordedFigure)
    make_predictions([("Assembly", "Gluing"), ("Assembly", "Gluing"), ("Gluing", "Assembly"),
                      ("Waiting", "Walking")], split="test", model="A").to_csv(
        tmp_path / "predictions_temporal.csv", index=False)
    evaluate_predictions.evaluate_file(tmp_path / "predictions_temporal.csv")
    expected = np.zeros((4, 4), dtype=int)  # rows true, columns predicted, in ACTIVITIES order
    expected[0, 1], expected[1, 0], expected[2, 3] = 2, 1, 1
    ax = figures[-1].axes[0]
    assert figures[-1].get_suptitle() == "Temporal holdout, test windows"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Predicted", "True")
    assert (np.asarray(ax.images[0].get_array()) == expected).all()
    assert sorted(int(t.get_text()) for t in ax.texts) == sorted(expected.ravel())
    for text in ax.texts:
        x, y = text.get_position()
        assert int(text.get_text()) == expected[round(y), round(x)]


def test_evaluation_script_writes_tables_and_figures(tmp_path):
    make_predictions([("Assembly", "Assembly"), ("Gluing", "Waiting")], split="test",
                     model="A").to_csv(tmp_path / "predictions_temporal.csv", index=False)
    make_predictions([("Walking", "Walking")], model="B").to_csv(
        tmp_path / "predictions_participant.csv", index=False)
    assert evaluate_predictions.main(["--out-dir", str(tmp_path)]) == 0
    for name in ("temporal", "participant"):
        assert (tmp_path / f"results_{name}.csv").is_file()
        assert (tmp_path / f"confusion_{name}.csv").is_file()
        assert (tmp_path / f"confusion_{name}.png").stat().st_size > 0
    assert pd.read_csv(tmp_path / "results_temporal.csv")["accuracy"].item() == 0.5


def test_evaluation_script_refuses_a_broken_file(tmp_path):
    pd.DataFrame({"model": ["A"], "activity": ["Assembly"]}).to_csv(
        tmp_path / "predictions_temporal.csv", index=False)
    with pytest.raises(SystemExit):
        evaluate_predictions.main(["--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        evaluate_predictions.main(["--out-dir", str(tmp_path / "empty")])
    val_only = tmp_path / "val_only"
    val_only.mkdir()
    make_predictions([("Assembly", "Gluing")], split="val", model="A").to_csv(
        val_only / "predictions_temporal.csv", index=False)
    with pytest.raises(SystemExit):
        evaluate_predictions.main(["--out-dir", str(val_only)])
    assert not (val_only / "results_temporal.csv").exists()
