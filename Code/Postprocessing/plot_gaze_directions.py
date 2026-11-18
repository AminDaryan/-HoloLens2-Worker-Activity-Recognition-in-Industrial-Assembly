"""Plot the gaze directions of the four recordings in one participant folder in 3D.

Usage, from the repository root:

    python Code/Postprocessing/plot_gaze_directions.py --participant P08
    python Code/Postprocessing/plot_gaze_directions.py --participant P08 --window 3

Each panel shows the gaze-direction vectors (gazeDirection_x, _y, _z) of one
recording as points. With --window, only that 240-row window of each recording
is used, numbered from 0 as in features.csv. The axes of the panels have their
own ranges. A panel whose recording was made by another participant than the
folder's (P04 Waiting and P05 Walking; see recorded_by in recordings.csv) says
so in its title.

Adapted from an earlier notebook for this dataset (3DRepresentation.ipynb). As
in the notebook, samples without a gaze direction are dropped, samples more
than three standard deviations from the mean in any component are removed as
outliers, and each vector is scaled to length one.

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backend_bases import FigureCanvasBase  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy import stats  # noqa: E402

# The code folders are not Python packages, so the shared modules are imported by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Preprocessing"))
from gaze_data import (ACTIVITIES, DATA_DIR, DIRECTION_COLUMNS, OUTPUT_DIR,  # noqa: E402
                       SAMPLES_PER_WINDOW, load_index, load_recording, window_rows)


def prepare_directions(recording, window: int | None = None) -> np.ndarray | None:
    """Gaze directions of a recording, or of one of its windows, cleaned as in
    the notebook.

    Returns None if the recording has no complete window number ``window``, and
    an empty array if fewer than ten samples have a gaze direction.
    """
    if window is None:
        rows = recording
    else:
        start = window * SAMPLES_PER_WINDOW
        if window < 0 or start + SAMPLES_PER_WINDOW > len(recording):
            return None
        rows = window_rows(recording, start, start + SAMPLES_PER_WINDOW)
    directions = rows[DIRECTION_COLUMNS].dropna().to_numpy(dtype=float)
    if len(directions) < 10:
        return np.empty((0, 3))
    z_scores = np.abs(stats.zscore(directions))
    directions = directions[(z_scores < 3).all(axis=1)]
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def panel_title(activity: str, recorded_by: str, participant: str) -> str:
    """Title of one panel; names the recorded participant if it is not the folder's."""
    if recorded_by != participant:
        return f"{activity} (recorded by {recorded_by})"
    return activity


def plot_participant(participant: str, out: Path, window: int | None = None,
                     data_dir: Path = DATA_DIR) -> Path:
    """Draw the four panels of one participant folder and save them to ``out``.

    Raises ValueError if the folder has no recordings or nothing to plot.
    """
    index = load_index(data_dir)
    recordings = index[index["slot"] == participant].set_index("activity")
    if recordings.empty:
        raise ValueError(f"no recordings for {participant}; use P01 to P10")
    fig = plt.figure(figsize=(14, 3.6))
    plotted = 0
    for k, activity in enumerate(ACTIVITIES):
        ax = fig.add_subplot(1, len(ACTIVITIES), k + 1, projection="3d")
        if activity not in recordings.index:
            ax.set_title(f"{activity} (no recording)", fontsize=10)
            ax.set_axis_off()
            continue
        rec = recordings.loc[activity]
        title = panel_title(activity, rec["recorded_by"], participant)
        directions = prepare_directions(load_recording(rec["file"], data_dir), window)
        if directions is None or len(directions) == 0:
            reason = f"no window {window}" if directions is None else "too few samples"
            ax.set_title(f"{title} ({reason})", fontsize=10)
            ax.set_axis_off()
            continue
        plotted += 1
        ax.set_title(title, fontsize=10)
        ax.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
                   c="steelblue", s=2, alpha=0.5, depthshade=True)
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect(None, zoom=0.85)  # leaves room for the z label
        ax.set_xlabel("x", fontsize=9, labelpad=4)
        ax.set_ylabel("y", fontsize=9, labelpad=4)
        ax.set_zlabel("z", fontsize=9, labelpad=4)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=7)
    if plotted == 0:
        plt.close(fig)
        where = "" if window is None else f" in window {window}"
        raise ValueError(f"nothing to plot for {participant}{where}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--participant", default="P01",
                        help="participant folder, P01 to P10 (default: P01)")
    parser.add_argument("--window", type=int, default=None,
                        help="plot only this window (0 = the first 240 rows)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="folder with recordings.csv (default: Participants Gaze Dataset)")
    parser.add_argument("--out", default=None,
                        help="output image, or a folder for it (default: "
                             "Code/output/gaze_directions_<participant>[_wNN].png)")
    args = parser.parse_args(argv)
    if args.window is not None and args.window < 0:
        parser.error("--window must be 0 or larger")
    if not (args.data_dir / "recordings.csv").is_file():
        parser.error(f"{args.data_dir} has no recordings.csv")
    suffix = "" if args.window is None else f"_w{args.window:02d}"
    name = f"gaze_directions_{args.participant}{suffix}.png"
    out = Path(args.out) if args.out else OUTPUT_DIR / name
    if out.is_dir() or (args.out and args.out.endswith(("/", os.sep))):
        out = out / name
    elif out.suffix == "":
        out = out.with_suffix(".png")
    if out.suffix.lstrip(".").lower() not in FigureCanvasBase.get_supported_filetypes():
        parser.error(f"cannot write {out.suffix} images; use for example .png or .pdf")
    try:
        plot_participant(args.participant, out, args.window, args.data_dir)
    except ValueError as problem:
        parser.error(str(problem))
    print(f"Plot written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
