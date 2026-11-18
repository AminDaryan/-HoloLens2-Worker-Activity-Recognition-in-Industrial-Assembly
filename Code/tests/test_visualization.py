"""Checks of the gaze plotting script in Code/Postprocessing: the 3D plots of the gaze directions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import plot_gaze_directions
from gaze_data import DIRECTION_COLUMNS, load_recording, window_rows
from plot_gaze_directions import panel_title, plot_participant, prepare_directions


def test_plot_is_written(tmp_path):
    out = plot_participant("P04", tmp_path / "p04.png")
    assert out.exists() and out.stat().st_size > 0


def test_panel_titles_name_the_recorded_participant(tmp_path, monkeypatch):
    assert panel_title("Waiting", "P10", "P04") == "Waiting (recorded by P10)"
    assert panel_title("Waiting", "P04", "P04") == "Waiting"
    figures = []
    monkeypatch.setattr(plot_gaze_directions.plt, "close", lambda fig=None: figures.append(fig))
    plot_participant("P04", tmp_path / "p04.png")
    assert "Waiting (recorded by P10)" in [ax.get_title() for ax in figures[-1].axes]
    plot_participant("P04", tmp_path / "p04_w08.png", window=8)  # waiting has windows 0 to 7
    assert "Waiting (recorded by P10) (no window 8)" in [ax.get_title() for ax in figures[-1].axes]


def test_a_window_uses_the_rows_of_that_window():
    recording = load_recording("P03/P03_Gluing.csv")
    rows = window_rows(recording, 240, 480)[DIRECTION_COLUMNS].dropna().to_numpy(float)
    rows = rows[(np.abs(stats.zscore(rows)) < 3).all(axis=1)]
    expected = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    np.testing.assert_allclose(prepare_directions(recording, window=1), expected)


def test_a_window_past_the_end_is_reported():
    recording = load_recording("P08/P08_Waiting.csv")
    assert prepare_directions(recording, window=len(recording) // 240) is None
    assert len(prepare_directions(recording, window=0)) > 0


def test_out_can_name_a_folder_or_a_file_without_suffix(tmp_path):
    args = ["--participant", "P02", "--window", "0", "--out"]
    assert plot_gaze_directions.main([*args, str(tmp_path)]) == 0
    assert (tmp_path / "gaze_directions_P02_w00.png").stat().st_size > 0
    assert plot_gaze_directions.main([*args, str(tmp_path / "new") + "/"]) == 0
    assert (tmp_path / "new" / "gaze_directions_P02_w00.png").is_file()
    assert plot_gaze_directions.main([*args, str(tmp_path / "plain")]) == 0
    assert (tmp_path / "plain.png").is_file()


def test_unknown_participant_and_image_type_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no recordings for P11"):
        plot_participant("P11", tmp_path / "p11.png")
    with pytest.raises(SystemExit):
        plot_gaze_directions.main(["--participant", "P11", "--out", str(tmp_path / "p11.png")])
    with pytest.raises(SystemExit):
        plot_gaze_directions.main(["--participant", "P03", "--out", str(tmp_path / "plot.xyz")])
