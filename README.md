# HoloLens 2 gaze recordings of four activities at an assembly workstation

This repository contains the dataset of the paper:

> **Gaze-Based Worker Activity Recognition in Industrial Assembly Using Temporal and Expert
> Feature Fusion**
> Pre-print on arXiv. The link and full citation will be added here once the pre-print is online.

It contains eye-tracking (gaze) recordings of ten participants, each performing four
activities at a workstation: assembling, gluing, waiting, and walking. The recordings are in
the folder `Participants Gaze Dataset/`. The folder `Code/` holds Python code that computes
gaze features from the recordings and classifies the four activities with feature-based
classifiers; it does not contain the models or the expert features that produced the paper's
results. The data are free to use under the CC BY 4.0 license, and the code under the Apache
License 2.0 (see [License and consent](#license-and-consent)).

---

## What the repository contains

| Item | Content |
|---|---|
| `Participants Gaze Dataset/P01/` to `P10/` | One folder per participant, each with four CSV files, one per activity: `Pxx_Assembly.csv`, `Pxx_Gluing.csv`, `Pxx_Waiting.csv`, `Pxx_Walking.csv`. |
| `Participants Gaze Dataset/recordings.csv` | An index with one row per recording file (see [The index file](#the-index-file-recordingscsv)). |
| `Code/` | Python code in four folders: `Preprocessing` (windows and gaze features), `Classification Algorithms` (the classifiers), `Postprocessing` (scores, confusion matrices, and 3D gaze plots). See [`Code/README.md`](Code/README.md). |
| `LICENSE` | The full text of the CC BY 4.0 license, which covers the recordings and their documentation. |

Paths in `recordings.csv`, and recording names such as `P04/P04_Waiting.csv` in the rest of
this README, are relative to the folder `Participants Gaze Dataset/` (see
[How to use the data](#how-to-use-the-data)).

In total there are 40 recordings with 85,827 gaze samples, about 48 minutes of data. Each file
is one continuous recording of one activity, between 57 and 107 seconds long.

---

## How the data were recorded

- **Participants.** Ten volunteers, identified only by the codes P01 to P10.
- **Device.** A Microsoft HoloLens 2 headset. Each participant put on the headset, and then
  its built-in eye calibration was run.
- **Software.** Gaze was logged with ARETT, the Augmented Reality Eye Tracking Toolkit [1],
  at the headset's nominal rate of 30 samples per second (30 Hz).
- **Place.** A workstation in the smart factory of the German Research Center for Artificial
  Intelligence (DFKI) in Kaiserslautern, Germany.
- **Activities.**
  - *Assembling*: joining 3D-printed parts (a hood and a cargo bed) onto a wheeled toy-truck
    chassis.
  - *Gluing*: joining a 3D-printed lid and cube with a hand-held glue gun.
  - *Waiting*: standing at the workstation without a specified object to look at.
  - *Walking*: walking around the workstation area without a destination or work goal.

---

## File format

Each recording is a comma-separated (CSV) file with one row per gaze sample and 61 columns.
The column names and their meaning follow Table 1 of the ARETT paper [1]. Missing values are
written as `NA`, and true/false values as `True` and `False`.

The columns most users need:

| Column | Meaning |
|---|---|
| `eyeDataTimestamp` | Time of the sample in milliseconds since the start of the recording |
| `eyeDataRelativeTimestamp` | Time of the sample from the headset's high-precision clock, in milliseconds since the start of the recording |
| `frameTimestamp` | Time of the display frame in which the sample was processed, in milliseconds on the same scale as `eyeDataTimestamp` |
| `isCalibrationValid` | `True` if the headset's eye calibration was valid |
| `gazeHasValue` | `True` if the sample contains a valid gaze origin and direction |
| `gazeOrigin_x`, `_y`, `_z` | Start point of the gaze ray, in the global (room-fixed) coordinate system of the scene. On HoloLens 2, one unit is one meter |
| `gazeDirection_x`, `_y`, `_z` | Direction of the gaze ray in the same coordinate system, as a vector of length 1 |

Because the coordinates are fixed to the room, not to the head, the gaze origin and direction
also change when the participant moves or turns their head.

The other columns describe where the gaze ray hit virtual objects, screens or areas of interest.
No such objects were used in these recordings, so these columns are almost always `NA` or
`False`.

---

## The index file `recordings.csv`

| Column | Meaning |
|---|---|
| `file` | Path of the recording file, relative to `Participants Gaze Dataset/` |
| `slot` | Participant folder that the file is in |
| `recorded_by` | Participant who was actually recorded |
| `activity` | Assembly, Gluing, Waiting, or Walking |
| `samples` | Number of rows (gaze samples) in the file |
| `duration_s` | Length of the recording in seconds |
| `valid_gaze_pct` | Percentage of samples with `gazeHasValue` = `True` |
| `max_gap_s` | Longest time between two consecutive samples, in seconds |
| `paper_file` | Name of the same recording in the paper's dataset |

---
## Before you analyze the data

1. **Small timestamp irregularities.** In seven files, one sample has an `eyeDataTimestamp`
   1 ms earlier than the sample before it. `eyeDataRelativeTimestamp` never
   decreases. The time between samples is about 33 ms but not exactly constant, so use the
   timestamps, not the row number, as the time axis.
2. **Invalid samples.** Samples with `gazeHasValue` = `False` (for example during blinks or when
   tracking was lost) have no gaze origin or direction. The share of valid samples is in the
   `valid_gaze_pct` column of `recordings.csv`.

---

## How to use the data

Any tool that reads CSV files works. A minimal example in Python with the pandas library:

```python
from pathlib import Path

import pandas as pd

data = Path("Participants Gaze Dataset")   # the dataset folder, seen from the repository root
df = pd.read_csv(data / "P01" / "P01_Walking.csv")
# pandas reads "NA" as missing and True/False as booleans
valid = df[df["gazeHasValue"]]             # keep samples with a valid gaze ray
time_s = valid["eyeDataTimestamp"] / 1000  # time axis in seconds
direction = valid[["gazeDirection_x", "gazeDirection_y", "gazeDirection_z"]]
```

To work with all 40 recordings, read `recordings.csv` and join each path in its `file` column
with the dataset folder. The `recorded_by` and `activity` columns give the participant and the
activity label of each file:

```python
index = pd.read_csv(data / "recordings.csv")
for rec in index.itertuples():
    recording = pd.read_csv(data / rec.file)   # labels: rec.recorded_by, rec.activity
```

The code in [`Code/`](Code/README.md) does this for you: it cuts the recordings into the
240-row windows and the training, validation, and test sets of the paper, computes 19
gaze features for each window, trains and tests classifiers on them, and scores their
predictions. These are not the models or the features that produced the paper's results; see
[`Code/README.md`](Code/README.md).

---

## License and consent

The recordings and their documentation are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The full text is in [`LICENSE`](LICENSE). You may copy, share, and adapt the data for any
purpose, including commercial use, as long as you give appropriate credit, link to the
license, and state whether you made changes. When you use the data, please cite the paper above.

The code in `Code/` is licensed under the Apache License 2.0; the full text is in
[`Code/LICENSE`](Code/LICENSE). It is adapted from GEAR
(<https://github.com/Interactions-HSG/GEAR>), which is also licensed under the Apache
License 2.0.

All ten participants gave informed consent to the use of their recordings in the paper and to
their public release under CC BY 4.0.

## Privacy

The dataset contains no information that identifies a participant:

- Participants are identified only by the codes P01 to P10. The key that links codes to
  people is kept privately by the authors and will not be published.
- The ARETT metadata files were not included, because they contain participant names and
  recording dates.
- All timestamps in these files were shifted so that each recording starts at 0, so the
  files contain no recording dates or times.
- The data contain only gaze measurements. There are no images, video, audio, or personal
  details.

Please do not try to identify the participants.

---

## Reference

[1] S. Kapp, M. Barz, S. Mukhametov, D. Sonntag, and J. Kuhn, "ARETT: Augmented Reality Eye
Tracking Toolkit for Head Mounted Displays," *Sensors*, vol. 21, no. 6, art. no. 2234, 2021,
doi: [10.3390/s21062234](https://doi.org/10.3390/s21062234).
